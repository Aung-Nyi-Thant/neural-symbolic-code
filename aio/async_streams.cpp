// async_streams.cpp - read-ahead and write-behind over IAsyncReader / IAsyncWriter.
#include "aio/async_streams.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <new>

namespace hce::aio {
namespace {

constexpr std::size_t kReapBatch = 64;
constexpr u64 kSyncTag = u64{1} << 62;  // slot indices are the other tags

}  // namespace

// ------------------------------------------------------------------ input ---

std::unique_ptr<AsyncFileInput> AsyncFileInput::open(const std::filesystem::path& path, const Options& opt,
                                                     OpenError* err) {
  const Options o = normalize(opt);
  std::unique_ptr<IAsyncReader> io = open_reader(path, o, err);
  if (!io) return nullptr;
  try {
    return std::unique_ptr<AsyncFileInput>(new AsyncFileInput(std::move(io), o));
  } catch (const std::bad_alloc&) {
    if (err) *err = {Errc::out_of_memory, 0};
    return nullptr;
  }
}

AsyncFileInput::AsyncFileInput(std::unique_ptr<IAsyncReader> io, const Options& opt)
    : pool_(opt.queue_depth, opt.chunk_bytes), io_(std::move(io)), slots_(opt.queue_depth), size_(io_->size()) {
  io_->register_buffers(pool_.all());
  for (std::size_t i = 0; i < slots_.size(); ++i) schedule(i);
  if (io_->submit() != Errc::ok) fail(Errc::io_error, 0);  // one system call for the initial window
}

void AsyncFileInput::schedule(std::size_t slot) noexcept {
  Slot& s = slots_[slot];
  s = Slot{};
  if (next_ >= size_ || ended_early_) return;  // nothing there: idle marks the end
  s.state = Slot::State::in_flight;
  s.offset = next_;
  s.expect = static_cast<std::size_t>(std::min<u64>(pool_.size(), size_ - next_));
  next_ += pool_.size();
  queue(slot, 0);
}

// Always asks for the rest of the whole chunk from an aligned position: valid
// for O_DIRECT, and the file's end simply makes the read short.
void AsyncFileInput::queue(std::size_t slot, std::size_t from) noexcept {
  if (io_->queue_read(slots_[slot].offset + from, pool_.buffer(slot).subspan(from), slot) != Errc::ok)
    fail(Errc::io_error, 0);
}

std::size_t AsyncFileInput::read(MutBytes buf) {
  while (!buf.empty() && error_ == Errc::ok) {
    if (interrupted_.load(std::memory_order_acquire)) {
      fail(Errc::cancelled, 0);
      break;
    }
    Slot& s = slots_[head_];
    if (s.state == Slot::State::idle) return 0;  // end of file
    if (s.state == Slot::State::in_flight) {
      wait_some();
      continue;
    }
    const std::size_t n = std::min(buf.size(), s.filled - s.consumed);
    std::memcpy(buf.data(), pool_.data(head_) + s.consumed, n);
    s.consumed += n;
    if (s.consumed == s.filled) {  // drained: reuse the buffer for the chunk after the window
      const std::size_t done = head_;
      head_ = (head_ + 1) % slots_.size();
      schedule(done);
      if (io_->submit() != Errc::ok) fail(Errc::io_error, 0);
    }
    if (n != 0) return n;
  }
  return 0;
}

void AsyncFileInput::wait_some() noexcept {
  std::array<Completion, kReapBatch> done;
  const std::size_t n = io_->reap(std::span(done.data(), std::min(done.size(), slots_.size())), 1);
  if (n == 0) fail(interrupted_.load(std::memory_order_acquire) ? Errc::cancelled : Errc::io_error, 0);
  for (std::size_t i = 0; i < n; ++i) on_complete(done[i]);
}

void AsyncFileInput::on_complete(const Completion& c) noexcept {
  if (c.tag >= slots_.size()) return;
  Slot& s = slots_[c.tag];
  if (c.sys_error != 0) {
    fail(Errc::io_error, c.sys_error);
    return;
  }
  s.filled += c.bytes;
  if (s.filled >= s.expect) {  // the chunk as of open; growth after open is not followed
    s.filled = s.expect;
    s.state = Slot::State::ready;
  } else if (c.bytes == 0) {  // the file shrank: deliver what there is, then end
    s.state = Slot::State::ready;
    ended_early_ = true;
  } else {  // cut short: continue from the last aligned position
    s.filled &= ~(kIoAlign - 1);
    queue(c.tag, s.filled);
    if (io_->submit() != Errc::ok) fail(Errc::io_error, 0);
  }
}

void AsyncFileInput::interrupt() noexcept {
  interrupted_.store(true, std::memory_order_release);
  io_->interrupt();
}

void AsyncFileInput::fail(Errc e, int sys_error) noexcept {
  if (error_ != Errc::ok) return;
  error_ = e;
  sys_error_ = sys_error;
}

// ----------------------------------------------------------------- output ---

std::unique_ptr<AsyncFileOutput> AsyncFileOutput::create(const std::filesystem::path& path, const Options& opt,
                                                         OpenError* err) {
  const Options o = normalize(opt);
  std::unique_ptr<IAsyncWriter> io = open_writer(path, o, err);
  if (!io) return nullptr;
  try {
    return std::unique_ptr<AsyncFileOutput>(new AsyncFileOutput(std::move(io), o));
  } catch (const std::bad_alloc&) {
    if (err) *err = {Errc::out_of_memory, 0};
    return nullptr;
  }
}

AsyncFileOutput::AsyncFileOutput(std::unique_ptr<IAsyncWriter> io, const Options& opt)
    : pool_(opt.queue_depth, opt.chunk_bytes), io_(std::move(io)), slots_(opt.queue_depth) {
  io_->register_buffers(pool_.all());
}

bool AsyncFileOutput::write(Bytes buf) {
  while (!buf.empty()) {
    if (error_ != Errc::ok || finished_) return false;
    if (interrupted_.load(std::memory_order_acquire)) {
      fail(Errc::cancelled, 0);
      return false;
    }
    Slot& s = slots_[cur_];
    if (s.state == Slot::State::in_flight) {  // every buffer is being written: wait for one
      wait_some();
      continue;
    }
    const std::size_t n = std::min(buf.size(), pool_.size() - s.fill);
    std::memcpy(pool_.data(cur_) + s.fill, buf.data(), n);
    s.fill += n;
    buf = buf.subspan(n);
    if (s.fill == pool_.size()) launch();
  }
  return error_ == Errc::ok;
}

void AsyncFileOutput::launch() noexcept {
  Slot& s = slots_[cur_];
  s.state = Slot::State::in_flight;
  s.offset = pos_;
  s.done = 0;
  pos_ += s.fill;
  ++in_flight_;
  queue(cur_, 0);
  if (io_->submit() != Errc::ok) fail(Errc::io_error, 0);
  cur_ = (cur_ + 1) % slots_.size();
}

void AsyncFileOutput::queue(std::size_t slot, std::size_t from) noexcept {
  const Slot& s = slots_[slot];
  if (io_->queue_write(s.offset + from, Bytes(pool_.data(slot) + from, s.fill - from), slot) != Errc::ok)
    fail(Errc::io_error, 0);
}

void AsyncFileOutput::wait_some() noexcept {
  std::array<Completion, kReapBatch> done;
  const std::size_t n = io_->reap(std::span(done.data(), std::min(done.size(), slots_.size())), 1);
  if (n == 0) fail(interrupted_.load(std::memory_order_acquire) ? Errc::cancelled : Errc::io_error, 0);
  for (std::size_t i = 0; i < n; ++i) on_complete(done[i]);
}

void AsyncFileOutput::on_complete(const Completion& c) noexcept {
  if (c.tag == kSyncTag) {
    synced_ = true;
    if (c.sys_error != 0) fail(Errc::io_error, c.sys_error);
    return;
  }
  if (c.tag >= slots_.size()) return;
  Slot& s = slots_[c.tag];
  if (c.sys_error != 0 || c.bytes == 0) {  // a write that makes no progress is a failure (ENOSPC)
    fail(Errc::io_error, c.sys_error);
    s.state = Slot::State::idle;
    --in_flight_;
    return;
  }
  s.done += c.bytes;
  if (s.done < s.fill) {  // short write: the rest
    queue(c.tag, s.done);
    if (io_->submit() != Errc::ok) fail(Errc::io_error, 0);
    return;
  }
  s.state = Slot::State::idle;
  s.fill = 0;
  --in_flight_;
}

Errc AsyncFileOutput::finish(bool durable) noexcept {
  if (finished_) return error_;
  finished_ = true;
  // The current slot is in flight when the last write filled every buffer.
  const Slot& last = slots_[cur_];
  if (error_ == Errc::ok && last.state == Slot::State::idle && last.fill > 0) launch();
  while (in_flight_ > 0 && error_ == Errc::ok) wait_some();
  if (error_ == Errc::ok && durable) {  // every write completed: the sync covers them all
    if (io_->queue_sync(kSyncTag) != Errc::ok || io_->submit() != Errc::ok) fail(Errc::io_error, 0);
    while (!synced_ && error_ == Errc::ok) wait_some();
  }
  return error_;
}

void AsyncFileOutput::interrupt() noexcept {
  interrupted_.store(true, std::memory_order_release);
  io_->interrupt();
}

void AsyncFileOutput::fail(Errc e, int sys_error) noexcept {
  if (error_ != Errc::ok) return;
  error_ = e;
  sys_error_ = sys_error;
}

}  // namespace hce::aio
