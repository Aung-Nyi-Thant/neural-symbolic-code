// backends.hpp - internal: the engines behind IAsyncReader / IAsyncWriter.
//
// Each backend has one engine type with the same members (kName, depth(),
// file(), close_file(), register_buffers(), queue(), submit(), reap(),
// interrupt()); Reader
// and Writer adapt it to the two public interfaces.
#pragma once

#include "aio/async_io.hpp"
#include "aio/native_file.hpp"

#include <cstddef>
#include <memory>
#include <utility>

namespace hce::aio::detail {

enum class OpKind : std::uint8_t { read, write, sync };

struct Op {
  OpKind kind = OpKind::read;
  u64 offset = 0;
  std::byte* buf = nullptr;  // writes only read it
  std::size_t len = 0;
  u64 tag = 0;
};

template <class Engine>
class Reader final : public IAsyncReader {
 public:
  template <class... Args>
  explicit Reader(Args&&... args) : engine_(std::forward<Args>(args)...) {}

  const char* backend() const noexcept override { return Engine::kName; }
  unsigned depth() const noexcept override { return engine_.depth(); }
  void register_buffers(std::span<const MutBytes> b) noexcept override { engine_.register_buffers(b); }
  Errc submit() noexcept override { return engine_.submit(); }
  std::size_t reap(std::span<Completion> out, std::size_t min) noexcept override { return engine_.reap(out, min); }
  void interrupt() noexcept override { engine_.interrupt(); }
  FileId file_id() const noexcept override { return engine_.file().id(); }
  u64 size() const noexcept override { return engine_.file().size(); }
  bool direct() const noexcept override { return engine_.file().direct(); }
  Errc queue_read(u64 offset, MutBytes dst, u64 tag) noexcept override {
    return engine_.queue(Op{OpKind::read, offset, dst.data(), dst.size(), tag});
  }
  Engine& engine() noexcept { return engine_; }

 private:
  Engine engine_;
};

template <class Engine>
class Writer final : public IAsyncWriter {
 public:
  template <class... Args>
  explicit Writer(Args&&... args) : engine_(std::forward<Args>(args)...) {}

  const char* backend() const noexcept override { return Engine::kName; }
  unsigned depth() const noexcept override { return engine_.depth(); }
  void register_buffers(std::span<const MutBytes> b) noexcept override { engine_.register_buffers(b); }
  Errc submit() noexcept override { return engine_.submit(); }
  std::size_t reap(std::span<Completion> out, std::size_t min) noexcept override { return engine_.reap(out, min); }
  void interrupt() noexcept override { engine_.interrupt(); }
  Errc queue_write(u64 offset, Bytes src, u64 tag) noexcept override {
    return engine_.queue(Op{OpKind::write, offset, const_cast<std::byte*>(src.data()), src.size(), tag});
  }
  Errc queue_sync(u64 tag) noexcept override { return engine_.queue(Op{OpKind::sync, 0, nullptr, 0, tag}); }
  FileId file_id() const noexcept override { return engine_.file().id(); }
  int close() noexcept override { return engine_.close_file(); }
  Engine& engine() noexcept { return engine_; }

 private:
  Engine engine_;
};

// Throw std::bad_alloc or std::system_error (threads); the factories catch.
std::unique_ptr<IAsyncReader> make_thread_reader(NativeFile&& file, const Options& opt);
std::unique_ptr<IAsyncWriter> make_thread_writer(NativeFile&& file, const Options& opt);

#if defined(HCE_AIO_HAVE_URING)
bool uring_probe() noexcept;
// nullptr with `sys_error` set when no ring could be created; `file` is then left open.
std::unique_ptr<IAsyncReader> make_uring_reader(NativeFile& file, const Options& opt, int& sys_error);
std::unique_ptr<IAsyncWriter> make_uring_writer(NativeFile& file, const Options& opt, int& sys_error);
#endif

}  // namespace hce::aio::detail
