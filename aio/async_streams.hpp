// async_streams.hpp - the asynchronous layer behind the engine's stream interfaces.
//
// AsyncFileInput keeps `queue_depth` reads of `chunk_bytes` in flight ahead of
// its consumer (read-ahead); AsyncFileOutput keeps as many writes in flight
// behind its producer (write-behind). Both are plain hce::InputStream /
// hce::OutputStream objects, so encode_parallel() and decode_stream() take
// them unchanged. Memory: queue_depth * chunk_bytes per stream (8 MiB by
// default), allocated once.
//
//   auto in  = hce::aio::AsyncFileInput::open("app.log");
//   auto out = hce::aio::AsyncFileOutput::create("app.log.hce");
//   auto r   = hce::encode_parallel<hce::NeuralCMCodec>(*in, *out, opts);
//   if (r.status == hc::Errc::ok) r.status = out->finish();
#pragma once

#include "aio/async_io.hpp"
#include "aio/buffer_pool.hpp"
#include "io/stream.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <vector>

namespace hce::aio {

class AsyncFileInput final : public InputStream {
 public:
  // nullptr on failure; `err` says why (unsupported_feature: not a regular file).
  static std::unique_ptr<AsyncFileInput> open(const std::filesystem::path& path, const Options& opt = {},
                                              OpenError* err = nullptr);
  AsyncFileInput(const AsyncFileInput&) = delete;
  AsyncFileInput& operator=(const AsyncFileInput&) = delete;
  ~AsyncFileInput() = default;  // io_ waits for the reads still in flight

  // InputStream: in file order; 0 at the end, or on failure (then error()).
  std::size_t read(MutBytes buf) override;
  Errc error() const noexcept override { return error_; }
  void interrupt() noexcept override;

  const char* backend() const noexcept { return io_->backend(); }
  bool direct() const noexcept { return io_->direct(); }
  u64 size() const noexcept { return size_; }
  int sys_error() const noexcept { return sys_error_; }

 private:
  struct Slot {
    enum class State : std::uint8_t { idle, in_flight, ready } state = State::idle;
    u64 offset = 0;           // file offset of the chunk
    std::size_t expect = 0;   // bytes the file holds there, as of open
    std::size_t filled = 0;
    std::size_t consumed = 0;
  };

  AsyncFileInput(std::unique_ptr<IAsyncReader> io, const Options& opt);
  void schedule(std::size_t slot) noexcept;
  void queue(std::size_t slot, std::size_t from) noexcept;
  void wait_some() noexcept;
  void on_complete(const Completion& c) noexcept;
  void fail(Errc e, int sys_error) noexcept;

  BufferPool pool_;  // before io_: destroyed after it, once no read can touch it
  std::unique_ptr<IAsyncReader> io_;
  std::vector<Slot> slots_;
  u64 size_ = 0;
  u64 next_ = 0;           // offset of the next chunk to schedule
  std::size_t head_ = 0;   // slot holding the next bytes in file order
  bool ended_early_ = false;  // the file shrank after open
  Errc error_ = Errc::ok;
  int sys_error_ = 0;
  std::atomic<bool> interrupted_{false};
};

class AsyncFileOutput final : public OutputStream {
 public:
  // Creates or truncates `path`. nullptr on failure; `err` says why.
  static std::unique_ptr<AsyncFileOutput> create(const std::filesystem::path& path, const Options& opt = {},
                                                 OpenError* err = nullptr);
  AsyncFileOutput(const AsyncFileOutput&) = delete;
  AsyncFileOutput& operator=(const AsyncFileOutput&) = delete;
  // Does not flush: output nobody finish()ed is incomplete by definition.
  ~AsyncFileOutput() = default;

  // OutputStream: copies into the current chunk; full chunks are written behind.
  bool write(Bytes buf) override;
  void interrupt() noexcept override;

  // Writes the partial chunk, waits for every write and, with `durable`, makes
  // the data reach stable storage. Call once, after the last write().
  Errc finish(bool durable = true) noexcept;

  const char* backend() const noexcept { return io_->backend(); }
  Errc error() const noexcept { return error_; }
  int sys_error() const noexcept { return sys_error_; }
  u64 bytes_written() const noexcept { return pos_; }

 private:
  struct Slot {
    enum class State : std::uint8_t { idle, in_flight } state = State::idle;
    u64 offset = 0;
    std::size_t fill = 0;  // bytes in the buffer
    std::size_t done = 0;  // bytes the file has from it
  };

  AsyncFileOutput(std::unique_ptr<IAsyncWriter> io, const Options& opt);
  void launch() noexcept;
  void queue(std::size_t slot, std::size_t from) noexcept;
  void wait_some() noexcept;
  void on_complete(const Completion& c) noexcept;
  void fail(Errc e, int sys_error) noexcept;

  BufferPool pool_;  // before io_
  std::unique_ptr<IAsyncWriter> io_;
  std::vector<Slot> slots_;
  std::size_t cur_ = 0;    // slot being filled
  u64 pos_ = 0;            // file offset of the current slot
  unsigned in_flight_ = 0;
  bool synced_ = false;
  bool finished_ = false;
  Errc error_ = Errc::ok;
  int sys_error_ = 0;
  std::atomic<bool> interrupted_{false};
};

}  // namespace hce::aio
