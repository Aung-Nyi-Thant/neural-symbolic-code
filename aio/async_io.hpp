// async_io.hpp - asynchronous positional file I/O for hce.
//
// A proactor interface: callers queue reads or writes at explicit file
// offsets, each with a tag, start them in batches with submit(), and reap
// completions in any order. It maps one to one onto io_uring's submission and
// completion queues on Linux; a small pool of worker threads provides the
// same contract on macOS, Windows, and Linux systems where io_uring is not
// available or not permitted.
//
// The layer knows nothing about the codec or the container. The adapters in
// async_streams.hpp present it to the pipelines as the existing
// hce::InputStream / hce::OutputStream, so the engine is unchanged.
//
// Threading: one thread at a time drives an object (queue, submit, reap);
// interrupt() may be called from any thread.
#pragma once

#include "hc_framed.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace hce::aio {

using hc::Bytes;
using hc::Errc;
using hc::MutBytes;
using hc::u64;

enum class Backend : std::uint8_t {
  automatic,  // io_uring when compiled in and permitted, else threads; HCE_AIO=uring|threads overrides
  io_uring,   // Linux only; an explicit request fails when it is unavailable
  threads,    // portable: positional I/O on worker threads
};

struct Options {
  Backend backend = Backend::automatic;
  unsigned queue_depth = 8;                         // operations in flight per file, 1..256
  std::size_t chunk_bytes = std::size_t{1} << 20;   // stream adapters: bytes per operation
  bool direct = false;  // reads bypass the page cache where the file system allows it
};

inline constexpr std::size_t kIoAlign = 4096;  // buffers, offsets and chunk sizes: O_DIRECT safe

// Clamps depth to 1..256 and the chunk to 4 KiB..64 MiB, a multiple of kIoAlign.
Options normalize(Options o) noexcept;

// One finished operation. A read that transfers fewer bytes than asked reached
// end of file or was cut short; a write may also be short. Callers continue.
struct Completion {
  u64 tag = 0;
  std::size_t bytes = 0;
  int sys_error = 0;  // errno or Win32 error code; 0 = success
};

// What readers and writers share: batching and completions.
class IAsyncFile {
 public:
  virtual ~IAsyncFile() = default;
  virtual const char* backend() const noexcept = 0;
  virtual unsigned depth() const noexcept = 0;
  // Optional fast path: the memory every later operation will use. io_uring
  // pins it once (fixed buffers) instead of on every operation.
  virtual void register_buffers(std::span<const MutBytes> buffers) noexcept = 0;
  // Starts every queued operation: one system call for the batch on io_uring.
  virtual Errc submit() noexcept = 0;
  // Waits until `min` completions are ready (fewer after interrupt() or a
  // backend failure), then moves up to out.size() of them into `out`.
  virtual std::size_t reap(std::span<Completion> out, std::size_t min) noexcept = 0;
  // Makes a waiting and every later reap() return promptly. Any thread.
  virtual void interrupt() noexcept = 0;
};

// Tags are the caller's; ~0 is reserved. Every queue_*() returns output_full
// when depth() operations are outstanding (queued and not yet reaped).
class IAsyncReader : public IAsyncFile {
 public:
  virtual u64 size() const noexcept = 0;      // at open
  virtual bool direct() const noexcept = 0;   // the page cache is bypassed
  virtual Errc queue_read(u64 offset, MutBytes dst, u64 tag) noexcept = 0;
};

class IAsyncWriter : public IAsyncFile {
 public:
  virtual Errc queue_write(u64 offset, Bytes src, u64 tag) noexcept = 0;
  // Written data and the file size reach stable storage. Queue it after the
  // writes it must cover have completed: operations are not ordered.
  virtual Errc queue_sync(u64 tag) noexcept = 0;
};

struct OpenError {
  Errc status = Errc::ok;  // io_error, unsupported_feature (not a regular file; backend), out_of_memory
  int sys_error = 0;
};

// Opens an existing regular file. Pipes, terminals and devices are
// unsupported_feature: they stay on the blocking streams.
std::unique_ptr<IAsyncReader> open_reader(const std::filesystem::path& path, const Options& opt = {},
                                          OpenError* err = nullptr) noexcept;
// Creates or truncates a regular file.
std::unique_ptr<IAsyncWriter> open_writer(const std::filesystem::path& path, const Options& opt = {},
                                          OpenError* err = nullptr) noexcept;

// io_uring: compiled in, and this process may create a ring with the needed
// opcodes (seccomp profiles, kernel.io_uring_disabled and old kernels say no).
bool backend_available(Backend b) noexcept;
// What `b` means here: `automatic` honors HCE_AIO, then prefers io_uring.
Backend resolve(Backend b) noexcept;
const char* to_string(Backend b) noexcept;

}  // namespace hce::aio
