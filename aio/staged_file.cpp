// staged_file.cpp - temporary file, exclusive creation and atomic publication (see the header).
#include "aio/staged_file.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace hce::aio {
namespace {

namespace fs = std::filesystem;

constexpr int kNameAttempts = 64;

#if defined(_WIN32)
constexpr int kExists = ERROR_FILE_EXISTS;
constexpr int kNotSupported = ERROR_NOT_SUPPORTED;
constexpr int kIsDirectory = ERROR_DIRECTORY;
#else
constexpr int kExists = EEXIST;
constexpr int kNotSupported = ENOTSUP;
constexpr int kIsDirectory = EISDIR;
#endif

// 64 random bits per name: std::random_device once, then a mixed counter.
std::string random_suffix() {
  static const std::uint64_t seed = [] {
    std::random_device rd;
    return (std::uint64_t{rd()} << 32) ^ rd() ^
           static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  }();
  static std::atomic<std::uint64_t> counter{0};
  std::uint64_t z = seed + 0x9E3779B97F4A7C15ull * (counter.fetch_add(1, std::memory_order_relaxed) + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(z));
  return buf;
}

CommitResult fail(const char* stage, int sys_error, bool published = false) noexcept {
  return {Errc::io_error, sys_error, published, stage};
}

#if defined(_WIN32)
int last_error() noexcept { return static_cast<int>(::GetLastError()); }
#endif

#if !defined(_WIN32)
// The new directory entry, durable. File systems that cannot sync a
// directory (EINVAL, ENOTSUP) have nothing to sync. (Windows needs nothing:
// MoveFileExW's MOVEFILE_WRITE_THROUGH already waited for it.)
int sync_parent(const fs::path& p) noexcept {
  fs::path dir = p.parent_path();
  if (dir.empty()) dir = ".";
  const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return errno;
  int err = 0;
  if (::fsync(fd) != 0 && errno != EINVAL && errno != ENOTSUP && errno != EROFS) err = errno;
  ::close(fd);
  return err;
}

// One no-replace rename. ok, or the errno; ENOTSUP / ENOSYS / EINVAL mean
// "not on this file system", so the next method may try.
int rename_noreplace(const fs::path& from, const fs::path& to) noexcept {
#if defined(__linux__) && defined(SYS_renameat2)
  // The system call, not glibc's wrapper (2.28+): older C libraries run on newer kernels.
  constexpr unsigned kRenameNoReplace = 1;  // RENAME_NOREPLACE, <linux/fs.h>
  if (::syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), kRenameNoReplace) == 0) return 0;
  return errno;
#elif defined(__APPLE__)
  if (::renamex_np(from.c_str(), to.c_str(), RENAME_EXCL) == 0) return 0;
  return errno;
#else
  (void)from, (void)to;
  return ENOSYS;
#endif
}

bool unsupported(int e) noexcept { return e == ENOTSUP || e == EOPNOTSUPP || e == ENOSYS || e == EINVAL; }
#endif

}  // namespace

bool same_file(const fs::path& a, const fs::path& b) noexcept {
  std::error_code ec;
  const bool same = fs::equivalent(a, b, ec);
  return !ec && same;
}

namespace detail {

CommitResult publish_with(const fs::path& from, const fs::path& to, Publish how, unsigned methods) noexcept {
#if defined(_WIN32)
  const DWORD flags = MOVEFILE_WRITE_THROUGH | (how == Publish::replace ? MOVEFILE_REPLACE_EXISTING : 0);
  if (how == Publish::no_replace && (methods & kPublishMove) == 0) return fail("publish", kNotSupported);
  if (!::MoveFileExW(from.c_str(), to.c_str(), flags)) {
    const int e = last_error();
    return fail("publish", e == ERROR_ALREADY_EXISTS ? kExists : e);
  }
  return {Errc::ok, 0, true, ""};
#else
  if (how == Publish::replace) {
    if (::rename(from.c_str(), to.c_str()) != 0) return fail("publish", errno);
  } else {
    int e = kNotSupported;
    if (methods & kPublishRename) e = rename_noreplace(from, to);
    if (e != 0 && unsupported(e) && (methods & kPublishLink)) {
      // A hard link fails with EEXIST if the name exists: as atomic as the rename.
      if (::link(from.c_str(), to.c_str()) == 0) {
        ::unlink(from.c_str());  // the data is published; a leftover second name is harmless
        e = 0;
      } else {
        e = errno;
        // No hard links here (FAT, some FUSE and network file systems): nothing atomic is left.
        if (e == EPERM || e == EXDEV || e == EMLINK || e == ENOSYS || e == EOPNOTSUPP) e = kNotSupported;
      }
    }
    if (e != 0) return fail("publish", unsupported(e) ? kNotSupported : e);
  }
  if (const int e = sync_parent(to); e != 0) return fail("sync-dir", e, true);
  return {Errc::ok, 0, true, ""};
#endif
}

}  // namespace detail

CommitResult publish_file(const fs::path& from, const fs::path& to, Publish how) noexcept {
  return detail::publish_with(from, to, how, detail::kPublishAll);
}

std::unique_ptr<StagedFileOutput> StagedFileOutput::create(const fs::path& dest, const Options& opt,
                                                           OpenError* err) {
  OpenError e;
  std::error_code ec;
  if (fs::is_directory(dest, ec)) {
    if (err) *err = {Errc::io_error, kIsDirectory};
    return nullptr;
  }
  // Devices, pipes and sockets are not replaced by a rename (think /dev/null).
  const fs::file_type type = fs::symlink_status(dest, ec).type();
  if (!ec && type != fs::file_type::regular && type != fs::file_type::symlink && type != fs::file_type::not_found) {
    if (err) *err = {Errc::unsupported_feature, kNotSupported};
    return nullptr;
  }
  const fs::path name = dest.filename();
  if (name.empty() || name == "." || name == "..") {
    if (err) *err = {Errc::io_error, kIsDirectory};
    return nullptr;
  }
  for (int attempt = 0; attempt < kNameAttempts; ++attempt) {
    fs::path tmp = dest.parent_path() / ("." + name.string() + ".hce-" + random_suffix() + ".tmp");
    if (auto out = AsyncFileOutput::create(tmp, opt, &e, Create::exclusive)) {
      if (err) *err = {};
      return std::unique_ptr<StagedFileOutput>(new StagedFileOutput(std::move(out), std::move(tmp), dest));
    }
    if (e.status != Errc::io_error || e.sys_error != kExists) break;  // taken: another name; else a real failure
  }
  if (err) *err = e;
  return nullptr;
}

CommitResult StagedFileOutput::commit(Publish how) noexcept {
  if (done_) return fail("publish", kNotSupported);
  for (const char* stage : {"finish", "close"}) {
    const Errc e = stage[0] == 'f' ? out_->finish(true) : out_->close();
    if (e != Errc::ok) {
      const CommitResult r{e, out_->sys_error(), false, stage};
      discard();
      return r;
    }
  }
  // A replaced file keeps its permission bits (not its owner: that needs privileges).
  std::error_code ec;
  const fs::file_status old = fs::status(dest_, ec);
  if (!ec && fs::is_regular_file(old)) fs::permissions(tmp_, old.permissions(), ec);
  const CommitResult r = publish_file(tmp_, dest_, how);
  if (!r.published) {
    discard();
    return r;
  }
  done_ = true;  // the temporary name is gone: never remove the published file
  return r;
}

void StagedFileOutput::discard() noexcept {
  if (done_) return;
  done_ = true;
  out_.reset();  // waits for writes in flight and closes the file: Windows cannot remove an open one
  std::error_code ec;
  fs::remove(tmp_, ec);
}

}  // namespace hce::aio
