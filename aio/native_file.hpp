// native_file.hpp - an owned file handle with positional I/O (POSIX, Win32).
#pragma once

#include "aio/async_io.hpp"

#include <cstddef>
#include <filesystem>

namespace hce::aio {

inline constexpr std::size_t kMaxIoBytes = std::size_t{1} << 30;  // per operation; longer ones end short

struct IoOutcome {
  std::size_t bytes = 0;  // a read of 0 bytes without an error: end of file
  int error = 0;          // errno or Win32 error code
};

class NativeFile {
 public:
#if defined(_WIN32)
  using Handle = void*;  // HANDLE, opened for overlapped I/O
  static constexpr Handle kClosed = nullptr;
#else
  using Handle = int;
  static constexpr Handle kClosed = -1;
#endif
  enum class Mode { read, write, create_new, create_private };

  NativeFile() noexcept = default;
  NativeFile(NativeFile&& other) noexcept;
  NativeFile& operator=(NativeFile&& other) noexcept;
  NativeFile(const NativeFile&) = delete;
  NativeFile& operator=(const NativeFile&) = delete;
  ~NativeFile();

  // read: an existing regular file, with `direct` honored where the file system
  // allows it. write: created or truncated. create_new / create_private: a new
  // file, failing if the name exists (even as a dangling symbolic link), mode
  // 0666 less the umask / 0600. On failure the result is closed.
  static NativeFile open(const std::filesystem::path& path, Mode mode, bool direct, OpenError& err) noexcept;

  bool is_open() const noexcept { return handle_ != kClosed; }
  Handle handle() const noexcept { return handle_; }
  u64 size() const noexcept { return size_; }
  bool direct() const noexcept { return direct_; }
  FileId id() const noexcept { return id_; }

  // One positional transfer, possibly short. Safe to call from several threads.
  IoOutcome read_at(std::byte* dst, std::size_t len, u64 offset) noexcept;
  IoOutcome write_at(const std::byte* src, std::size_t len, u64 offset) noexcept;
  // Data and size to stable storage: fdatasync, F_FULLFSYNC on macOS, FlushFileBuffers.
  IoOutcome sync() noexcept;
  // Closes the handle; the error is what close()/CloseHandle() reported (0: none).
  int close() noexcept;

 private:
  Handle handle_ = kClosed;
  u64 size_ = 0;
  bool direct_ = false;
  FileId id_{};
};

}  // namespace hce::aio
