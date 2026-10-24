// posix_io.hpp - POSIX streams and atomic output files for the hce CLI.
#pragma once

#include "io/stream.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <stop_token>
#include <string>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hce::cli {

inline constexpr int kPollMs = 100;

// True when `fd` is ready for `events`, or in an error state that the next
// read()/write() reports; false on timeout or EINTR.
inline bool poll_ready(int fd, short events) noexcept {
  pollfd p{fd, events, 0};
  const int r = ::poll(&p, 1, kPollMs);
  return r > 0 || (r < 0 && errno != EINTR);
}

// Reads a file descriptor. Pipes, sockets and terminals are polled, so a stop
// request (Ctrl-C, or a pipeline that failed elsewhere) wakes a blocked read.
class FdInput final : public InputStream {
 public:
  FdInput(int fd, bool pollable, std::stop_token stop) noexcept
      : fd_(fd), pollable_(pollable), stop_(std::move(stop)) {}

  std::size_t read(hc::MutBytes buf) override {
    while (err_ == hc::Errc::ok && !eof_) {
      if (stop_.stop_requested() || interrupted_.load(std::memory_order_relaxed)) {
        err_ = hc::Errc::cancelled;
        break;
      }
      if (pollable_ && !poll_ready(fd_, POLLIN)) continue;
      const ssize_t n = ::read(fd_, buf.data(), buf.size());
      if (n > 0) return static_cast<std::size_t>(n);
      if (n == 0) {
        eof_ = true;
        break;
      }
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        pollable_ = true;
        continue;
      }
      errno_ = errno;
      err_ = hc::Errc::io_error;
    }
    return 0;
  }

  hc::Errc error() const noexcept override { return err_; }
  void interrupt() noexcept override { interrupted_.store(true, std::memory_order_relaxed); }
  int sys_errno() const noexcept { return errno_; }

 private:
  int fd_;
  bool pollable_;
  std::stop_token stop_;
  std::atomic<bool> interrupted_{false};
  bool eof_ = false;
  hc::Errc err_ = hc::Errc::ok;
  int errno_ = 0;
};

// Writes a file descriptor completely. Pipes are written in pieces no larger
// than their default capacity, polling in between, so a stuck consumer
// cannot hold a stop request hostage. EPIPE is reported, never signalled
// (the CLI ignores SIGPIPE).
class FdOutput final : public OutputStream {
 public:
  FdOutput(int fd, bool pollable, std::stop_token stop) noexcept
      : fd_(fd), pollable_(pollable), stop_(std::move(stop)) {}

  bool write(hc::Bytes buf) override {
    while (!buf.empty()) {
      if (pollable_) {
        if (stop_.stop_requested() || interrupted_.load(std::memory_order_relaxed)) {
          errno_ = ECANCELED;
          return false;
        }
        if (!poll_ready(fd_, POLLOUT)) continue;
      }
      const std::size_t piece = std::min(buf.size(), pollable_ ? std::size_t{64} << 10 : std::size_t{1} << 30);
      const ssize_t n = ::write(fd_, buf.data(), piece);
      if (n > 0) {
        buf = buf.subspan(static_cast<std::size_t>(n));
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        pollable_ = true;
        continue;
      }
      errno_ = n < 0 ? errno : EIO;
      return false;
    }
    return true;
  }

  void interrupt() noexcept override { interrupted_.store(true, std::memory_order_relaxed); }
  int sys_errno() const noexcept { return errno_; }
  bool broken_pipe() const noexcept { return errno_ == EPIPE; }

 private:
  int fd_;
  bool pollable_;
  std::stop_token stop_;
  std::atomic<bool> interrupted_{false};
  int errno_ = 0;
};

// A file that appears at its path only once complete. Data goes to a
// temporary file in the same directory; commit() fsyncs it and publishes it
// atomically. Without `force` it is published with link(), which fails if the
// path exists, so a file created meanwhile is never overwritten. Anything not
// committed is removed.
class OutputFile {
 public:
  OutputFile() = default;
  OutputFile(const OutputFile&) = delete;
  OutputFile& operator=(const OutputFile&) = delete;
  ~OutputFile() { discard(); }

  bool open(const std::string& path, bool force, mode_t mode, std::string& err) {
    path_ = path;
    force_ = force;
    const std::size_t slash = path.find_last_of('/');
    dir_ = slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash);
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    if (base.empty() || base == "." || base == "..") {
      err = "not a file name";
      return false;
    }
    std::string tmp = dir_ + "/." + base + ".hce-XXXXXX";
    fd_ = ::mkostemp(tmp.data(), O_CLOEXEC);
    if (fd_ < 0) {
      err = std::strerror(errno);
      return false;
    }
    tmp_ = std::move(tmp);
    if (::fchmod(fd_, mode) != 0) {
      err = std::strerror(errno);
      return false;
    }
    return true;
  }

  int fd() const noexcept { return fd_; }

  bool commit(std::string& err) {
    if (::fsync(fd_) != 0 && errno != EINVAL) return fail(err);
    const int fd = std::exchange(fd_, -1);
    if (::close(fd) != 0) return fail(err);  // NFS reports deferred write errors here
    if (force_) {
      if (::rename(tmp_.c_str(), path_.c_str()) != 0) return fail(err);
    } else if (::link(tmp_.c_str(), path_.c_str()) == 0) {
      ::unlink(tmp_.c_str());
    } else if (errno == EEXIST) {
      err = "already exists (use -f to overwrite)";
      return false;
    } else {  // no hard links on this file system: best effort without them
      struct stat st {};
      if (::lstat(path_.c_str(), &st) == 0) {
        err = "already exists (use -f to overwrite)";
        return false;
      }
      if (::rename(tmp_.c_str(), path_.c_str()) != 0) return fail(err);
    }
    tmp_.clear();
    const int dir = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir >= 0) {  // make the new directory entry durable too
      ::fsync(dir);
      ::close(dir);
    }
    return true;
  }

  void discard() noexcept {
    if (fd_ >= 0) ::close(std::exchange(fd_, -1));
    if (!tmp_.empty()) ::unlink(tmp_.c_str());
    tmp_.clear();
  }

 private:
  static bool fail(std::string& err) {
    err = std::strerror(errno);
    return false;
  }

  std::string path_, dir_, tmp_;
  int fd_ = -1;
  bool force_ = false;
};

}  // namespace hce::cli
