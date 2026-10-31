// tool_util.hpp - what the ultra and ultra_bench tools share: strict number
// parsing, checked file reads, transactional file writes, peak-RSS units and
// verified benchmark repetitions. Header-only, so tests/test_tools.cpp tests
// exactly what the tools run.
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ultracodec::tools {

// The whole string as an unsigned integer in [lo, hi]: decimal, or hex with
// 0x. No sign, spaces, suffix or overflow; nullopt otherwise.
inline std::optional<std::uint64_t> parse_uint(std::string_view s, std::uint64_t lo, std::uint64_t hi) {
  unsigned base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    s.remove_prefix(2);
    base = 16;
  }
  if (s.empty()) return std::nullopt;
  std::uint64_t v = 0;
  for (const char c : s) {
    unsigned d;
    if (c >= '0' && c <= '9') d = static_cast<unsigned>(c - '0');
    else if (base == 16 && c >= 'a' && c <= 'f') d = static_cast<unsigned>(c - 'a' + 10);
    else if (base == 16 && c >= 'A' && c <= 'F') d = static_cast<unsigned>(c - 'A' + 10);
    else return std::nullopt;
    if (v > (UINT64_MAX - d) / base) return std::nullopt;
    v = v * base + d;
  }
  if (v < lo || v > hi) return std::nullopt;
  return v;
}

// getrusage()'s ru_maxrss in MiB: Linux and the BSDs count KiB, macOS bytes.
constexpr double rss_to_mib(long ru_maxrss, bool in_bytes) noexcept {
  return static_cast<double>(ru_maxrss) / (in_bytes ? 1048576.0 : 1024.0);
}
constexpr double maxrss_mib(long ru_maxrss) noexcept {
#if defined(__APPLE__)
  return rss_to_mib(ru_maxrss, true);
#else
  return rss_to_mib(ru_maxrss, false);
#endif
}

struct IoError {
  std::string what;
};

// A whole file; read errors (not only open errors) fail.
inline std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::FILE* f = nullptr;
#if defined(_WIN32)
  f = ::_wfopen(path.c_str(), L"rb");
#else
  f = std::fopen(path.c_str(), "rb");
#endif
  if (!f) throw IoError{"cannot open " + path.string() + ": " + std::strerror(errno)};
  std::vector<std::uint8_t> data;
  std::uint8_t buf[1 << 16];
  for (;;) {
    const std::size_t n = std::fread(buf, 1, sizeof buf, f);
    data.insert(data.end(), buf, buf + n);
    if (n < sizeof buf) break;
  }
  const bool failed = std::ferror(f) != 0;
  const int err = errno;
  std::fclose(f);
  if (failed) throw IoError{"cannot read " + path.string() + ": " + std::strerror(err)};
  return data;
}

namespace detail {

inline int open_new(const std::filesystem::path& p) {
#if defined(_WIN32)
  return ::_wopen(p.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
  return ::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
#endif
}

inline bool write_all(int fd, std::span<const std::uint8_t> d) {
  while (!d.empty()) {
    const unsigned piece = static_cast<unsigned>(std::min<std::size_t>(d.size(), std::size_t{1} << 30));
#if defined(_WIN32)
    const int n = ::_write(fd, d.data(), piece);
#else
    const ssize_t n = ::write(fd, d.data(), piece);
    if (n < 0 && errno == EINTR) continue;
#endif
    if (n <= 0) {
      if (n == 0) errno = EIO;
      return false;
    }
    d = d.subspan(static_cast<std::size_t>(n));
  }
  return true;
}

inline bool sync_fd(int fd) {
#if defined(_WIN32)
  return ::_commit(fd) == 0;
#elif defined(__APPLE__)
  return ::fcntl(fd, F_FULLFSYNC) == 0 || ::fsync(fd) == 0;
#else
  return ::fsync(fd) == 0 || errno == EINVAL;  // EINVAL: a file that cannot be synced (a pipe, /dev/null)
#endif
}

inline bool close_fd(int fd) {
#if defined(_WIN32)
  return ::_close(fd) == 0;
#else
  return ::close(fd) == 0 || errno == EINTR;
#endif
}

}  // namespace detail

// Writes `d` to `path` so that `path` changes only once the data is complete
// and on stable storage: a new temporary file beside it, synced, closed (its
// errors checked) and renamed over it. On failure the old file is untouched
// and the temporary file removed. A destination that exists but is not a
// regular file (a device such as /dev/null, a pipe) is written in place.
// Throws IoError; `durable` is false after the rename when the directory
// could not be synced (the file is in place and kept).
inline void write_file(const std::filesystem::path& path, std::span<const std::uint8_t> d, bool* durable = nullptr) {
  namespace fs = std::filesystem;
  if (durable) *durable = true;
  std::error_code ec;
  const fs::file_status st = fs::status(path, ec);
  if (!ec && fs::exists(st) && !fs::is_regular_file(st)) {
    if (fs::is_directory(st)) throw IoError{path.string() + ": is a directory"};
#if defined(_WIN32)
    const int fd = ::_wopen(path.c_str(), _O_WRONLY | _O_BINARY);
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
#endif
    if (fd < 0) throw IoError{"cannot open " + path.string() + ": " + std::strerror(errno)};
    const bool ok = detail::write_all(fd, d);
    const int err = errno;
    if (!detail::close_fd(fd) || !ok) throw IoError{"cannot write " + path.string() + ": " + std::strerror(ok ? errno : err)};
    return;
  }
  std::random_device rd;
  fs::path tmp;
  int fd = -1;
  for (int attempt = 0; attempt < 64 && fd < 0; ++attempt) {
    char suffix[40];
    std::snprintf(suffix, sizeof suffix, ".ultra-%08x%08x.tmp", rd(), rd());
    tmp = path.parent_path() / ("." + path.filename().string() + suffix);
    fd = detail::open_new(tmp);
    if (fd < 0 && errno != EEXIST) break;
  }
  if (fd < 0) throw IoError{"cannot create a temporary file beside " + path.string() + ": " + std::strerror(errno)};
  const auto fail = [&](const char* what, int err) {
    fs::remove(tmp, ec);
    throw IoError{std::string(what) + " " + path.string() + ": " + std::strerror(err)};
  };
  if (!detail::write_all(fd, d)) {
    const int err = errno;
    detail::close_fd(fd);
    fail("cannot write", err);
  }
  if (!detail::sync_fd(fd)) {
    const int err = errno;
    detail::close_fd(fd);
    fail("cannot sync", err);
  }
  if (!detail::close_fd(fd)) fail("cannot close", errno);
  if (fs::is_regular_file(st)) fs::permissions(tmp, st.permissions(), ec);  // a replaced file keeps its mode
  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(tmp, ec);
    throw IoError{"cannot rename into " + path.string() + ": " + ec.message()};
  }
#if !defined(_WIN32)
  fs::path dir = path.parent_path();
  if (dir.empty()) dir = ".";
  const int dfd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
  const bool synced = dfd >= 0 && (::fsync(dfd) == 0 || errno == EINVAL);
  if (dfd >= 0) ::close(dfd);
  if (!synced && durable) *durable = false;
#endif
}

// One benchmark input, run `runs` times. Every repetition's output is
// compared with the input as soon as it is produced: a single mismatch fails
// the whole measurement, whatever the other repetitions did.
struct Repetitions {
  bool verified = true;
  int failed_run = -1;            // the first repetition that did not round-trip
  std::size_t packed_bytes = 0;   // of the first repetition (later ones must match it)
  std::vector<double> compress_s, decompress_s;
  bool deterministic = true;      // every repetition produced the same archive

  static double best(std::vector<double> v) { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); }
  static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }
};

template <class Engine>
Repetitions run_repetitions(Engine& engine, std::span<const std::uint8_t> data, int runs) {
  using clock = std::chrono::steady_clock;
  Repetitions r;
  std::vector<std::uint8_t> first;
  for (int i = 0; i < runs; ++i) {
    const auto t0 = clock::now();
    std::vector<std::uint8_t> packed = engine.compress(data);
    const auto t1 = clock::now();
    const std::vector<std::uint8_t> back = engine.decompress(packed);
    const auto t2 = clock::now();
    r.compress_s.push_back(std::chrono::duration<double>(t1 - t0).count());
    r.decompress_s.push_back(std::chrono::duration<double>(t2 - t1).count());
    if (back.size() != data.size() || !std::equal(back.begin(), back.end(), data.begin())) {
      r.verified = false;
      if (r.failed_run < 0) r.failed_run = i;
    }
    if (i == 0) {
      r.packed_bytes = packed.size();
      first = std::move(packed);
    } else if (packed != first) {
      r.deterministic = false;
    }
  }
  if (runs < 1) r.verified = false;
  return r;
}

}  // namespace ultracodec::tools
