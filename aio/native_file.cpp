// native_file.cpp - positional file I/O on POSIX (Linux, macOS) and Win32.
#include "aio/native_file.hpp"

#include <algorithm>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hce::aio {

NativeFile::NativeFile(NativeFile&& other) noexcept
    : handle_(std::exchange(other.handle_, kClosed)), size_(other.size_), direct_(other.direct_), id_(other.id_) {}

NativeFile& NativeFile::operator=(NativeFile&& other) noexcept {
  if (this != &other) {
    (void)close();
    handle_ = std::exchange(other.handle_, kClosed);
    size_ = other.size_;
    direct_ = other.direct_;
    id_ = other.id_;
  }
  return *this;
}

NativeFile::~NativeFile() { (void)close(); }

#if defined(_WIN32)

namespace {

// Overlapped handles let several workers transfer at once; each operation
// waits on its own event, so concurrent completions cannot be confused.
IoOutcome transfer(HANDLE h, bool write, std::byte* p, std::size_t len, u64 offset) noexcept {
  OVERLAPPED ov{};
  ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFu);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  ov.hEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (!ov.hEvent) return {0, static_cast<int>(::GetLastError())};
  const DWORD n = static_cast<DWORD>(std::min(len, kMaxIoBytes));
  const BOOL started = write ? ::WriteFile(h, p, n, nullptr, &ov) : ::ReadFile(h, p, n, nullptr, &ov);
  DWORD err = started ? ERROR_SUCCESS : ::GetLastError();
  DWORD done = 0;
  if (started || err == ERROR_IO_PENDING) err = ::GetOverlappedResult(h, &ov, &done, TRUE) ? ERROR_SUCCESS : ::GetLastError();
  ::CloseHandle(ov.hEvent);
  if (err == ERROR_HANDLE_EOF) return {0, 0};
  if (err != ERROR_SUCCESS) return {0, static_cast<int>(err)};
  return {done, 0};
}

}  // namespace

NativeFile NativeFile::open(const std::filesystem::path& path, Mode mode, bool direct, OpenError& err) noexcept {
  NativeFile f;
  const bool write = mode != Mode::read;
  const DWORD disposition = mode == Mode::read ? OPEN_EXISTING : mode == Mode::write ? CREATE_ALWAYS : CREATE_NEW;
  DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
  // NO_BUFFERING needs sector-aligned offsets, lengths and memory: the adapters use kIoAlign.
  if (!write) flags |= FILE_FLAG_SEQUENTIAL_SCAN | (direct ? FILE_FLAG_NO_BUFFERING : 0);
  const HANDLE h = ::CreateFileW(path.c_str(), write ? GENERIC_WRITE : GENERIC_READ,
                                 write ? FILE_SHARE_READ : FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, disposition, flags, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    err = {Errc::io_error, static_cast<int>(::GetLastError())};
    return f;
  }
  f.handle_ = h;
  LARGE_INTEGER size{};
  BY_HANDLE_FILE_INFORMATION info{};
  if (::GetFileType(h) != FILE_TYPE_DISK) {
    err = {Errc::unsupported_feature, ERROR_NOT_SUPPORTED};
    (void)f.close();
  } else if (!::GetFileSizeEx(h, &size) || !::GetFileInformationByHandle(h, &info)) {
    err = {Errc::io_error, static_cast<int>(::GetLastError())};
    (void)f.close();
  } else {
    f.size_ = static_cast<u64>(size.QuadPart);
    f.id_ = {info.dwVolumeSerialNumber, (u64{info.nFileIndexHigh} << 32) | info.nFileIndexLow};
    f.direct_ = !write && direct;
    err = {};
  }
  return f;
}

IoOutcome NativeFile::read_at(std::byte* dst, std::size_t len, u64 offset) noexcept {
  return transfer(handle_, false, dst, len, offset);
}

IoOutcome NativeFile::write_at(const std::byte* src, std::size_t len, u64 offset) noexcept {
  return transfer(handle_, true, const_cast<std::byte*>(src), len, offset);  // WriteFile only reads it
}

IoOutcome NativeFile::sync() noexcept {
  return ::FlushFileBuffers(handle_) ? IoOutcome{} : IoOutcome{0, static_cast<int>(::GetLastError())};
}

int NativeFile::close() noexcept {
  int err = 0;
  if (handle_ != kClosed && !::CloseHandle(handle_)) err = static_cast<int>(::GetLastError());
  handle_ = kClosed;
  return err;
}

#else  // POSIX

NativeFile NativeFile::open(const std::filesystem::path& path, Mode mode, bool direct, OpenError& err) noexcept {
  NativeFile f;
  const bool write = mode != Mode::read;
  // O_NONBLOCK keeps a FIFO from blocking the open; it is rejected just below,
  // and has no effect on regular files. O_EXCL fails on any existing name,
  // symbolic links included, so a new file is never someone else's.
  int flags = (mode == Mode::read    ? O_RDONLY
               : mode == Mode::write ? O_WRONLY | O_CREAT | O_TRUNC
                                     : O_WRONLY | O_CREAT | O_EXCL) |
              O_CLOEXEC | O_NONBLOCK;
  const mode_t perms = mode == Mode::create_private ? 0600 : 0666;
#if defined(__linux__)
  if (direct && !write) flags |= O_DIRECT;
#endif
  int fd = ::open(path.c_str(), flags, perms);
#if defined(__linux__)
  if (fd < 0 && (flags & O_DIRECT) != 0 && errno == EINVAL) {  // tmpfs and some FUSE file systems
    flags &= ~O_DIRECT;
    fd = ::open(path.c_str(), flags, perms);
  }
#endif
  if (fd < 0) {
    err = {Errc::io_error, errno};
    return f;
  }
  f.handle_ = fd;
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    err = {Errc::io_error, errno};
    (void)f.close();
    return f;
  }
  if (!S_ISREG(st.st_mode)) {
    err = {Errc::unsupported_feature, ESPIPE};
    (void)f.close();
    return f;
  }
  f.size_ = static_cast<u64>(st.st_size);
  f.id_ = {static_cast<u64>(st.st_dev), static_cast<u64>(st.st_ino)};
#if defined(__linux__)
  f.direct_ = (flags & O_DIRECT) != 0;
  if (!write) (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#elif defined(__APPLE__)
  if (!write) {
    f.direct_ = direct && ::fcntl(fd, F_NOCACHE, 1) == 0;
    (void)::fcntl(fd, F_RDAHEAD, 1);
  }
#endif
  err = {};
  return f;
}

IoOutcome NativeFile::read_at(std::byte* dst, std::size_t len, u64 offset) noexcept {
  for (;;) {
    const ssize_t r = ::pread(handle_, dst, std::min(len, kMaxIoBytes), static_cast<off_t>(offset));
    if (r >= 0) return {static_cast<std::size_t>(r), 0};
    if (errno != EINTR) return {0, errno};
  }
}

IoOutcome NativeFile::write_at(const std::byte* src, std::size_t len, u64 offset) noexcept {
  for (;;) {
    const ssize_t r = ::pwrite(handle_, src, std::min(len, kMaxIoBytes), static_cast<off_t>(offset));
    if (r >= 0) return {static_cast<std::size_t>(r), 0};
    if (errno != EINTR) return {0, errno};
  }
}

IoOutcome NativeFile::sync() noexcept {
#if defined(__APPLE__)
  if (::fcntl(handle_, F_FULLFSYNC) == 0) return {};  // fsync alone leaves data in the drive's cache
  if (::fsync(handle_) == 0) return {};
#elif defined(__linux__)
  if (::fdatasync(handle_) == 0) return {};
#else
  if (::fsync(handle_) == 0) return {};
#endif
  return {0, errno};
}

int NativeFile::close() noexcept {
  int err = 0;
  // POSIX leaves the descriptor unspecified after EINTR; Linux and macOS have closed it: never retry.
  if (handle_ != kClosed && ::close(handle_) != 0 && errno != EINTR) err = errno;
  handle_ = kClosed;
  return err;
}

#endif

}  // namespace hce::aio
