// thread_backend.cpp - the portable engine: positional I/O on worker threads.
//
// macOS and Windows have no io_uring equivalent for regular files that is
// both portable and simpler than this (kqueue does not cover file reads;
// POSIX AIO on macOS is limited; IOCP would be a Windows-only backend behind
// the same interface). Linux uses it when io_uring is unavailable. Up to four
// workers keep `depth` operations in flight, which is what an SSD needs; the
// cost is a lock and a wake-up per batch, small next to a megabyte transfer.
#include "aio/backends.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace hce::aio::detail {
namespace {

constexpr unsigned kMaxWorkers = 4;

class ThreadEngine {
 public:
  static constexpr const char* kName = "threads";

  ThreadEngine(NativeFile&& file, const Options& opt)
      : file_(std::move(file)), depth_(opt.queue_depth), ring_(depth_) {
    pending_.reserve(depth_);
    done_.reserve(depth_);
    const unsigned workers = std::clamp(depth_, 1u, kMaxWorkers);
    workers_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) workers_.emplace_back([this](std::stop_token st) { run(st); });
  }

  ThreadEngine(const ThreadEngine&) = delete;
  ThreadEngine& operator=(const ThreadEngine&) = delete;
  // workers_ is declared last: its jthreads stop and join (finishing the
  // operation each is running) before anything they use is destroyed.
  ~ThreadEngine() = default;

  unsigned depth() const noexcept { return depth_; }
  const NativeFile& file() const noexcept { return file_; }
  void register_buffers(std::span<const MutBytes>) noexcept {}

  Errc queue(const Op& op) noexcept {
    if (outstanding_ >= depth_) return Errc::output_full;
    pending_.push_back(op);  // capacity depth_: no allocation
    ++outstanding_;
    return Errc::ok;
  }

  Errc submit() noexcept {
    if (pending_.empty()) return Errc::ok;
    {
      const std::lock_guard lock(m_);
      for (const Op& op : pending_) ring_[(ring_head_ + ring_count_++) % ring_.size()] = op;
    }
    submitted_ += static_cast<unsigned>(pending_.size());
    pending_.clear();
    work_cv_.notify_all();
    return Errc::ok;
  }

  std::size_t reap(std::span<Completion> out, std::size_t min) noexcept {
    min = std::min({min, out.size(), static_cast<std::size_t>(submitted_)});
    std::unique_lock lock(m_);
    done_cv_.wait(lock, [&] { return done_.size() >= min || interrupted_; });
    const std::size_t n = std::min(out.size(), done_.size());
    std::copy(done_.end() - static_cast<std::ptrdiff_t>(n), done_.end(), out.begin());
    done_.resize(done_.size() - n);
    lock.unlock();
    submitted_ -= static_cast<unsigned>(n);
    outstanding_ -= static_cast<unsigned>(n);
    return n;
  }

  void interrupt() noexcept {
    {
      const std::lock_guard lock(m_);
      interrupted_ = true;
    }
    done_cv_.notify_all();
  }

 private:
  void run(std::stop_token st) noexcept {
    std::unique_lock lock(m_);
    for (;;) {
      if (!work_cv_.wait(lock, st, [&] { return ring_count_ > 0; })) return;  // stop requested
      const Op op = ring_[ring_head_];
      ring_head_ = (ring_head_ + 1) % ring_.size();
      --ring_count_;
      lock.unlock();
      const Completion c = execute(op);
      lock.lock();
      done_.push_back(c);  // at most depth_ outstanding: no allocation
      done_cv_.notify_all();
    }
  }

  Completion execute(const Op& op) noexcept {
    IoOutcome r;
    switch (op.kind) {
      case OpKind::read: r = file_.read_at(op.buf, op.len, op.offset); break;
      case OpKind::write: r = file_.write_at(op.buf, op.len, op.offset); break;
      case OpKind::sync: r = file_.sync(); break;
    }
    return Completion{op.tag, r.bytes, r.error};
  }

  NativeFile file_;
  unsigned depth_;
  // Driving thread only.
  std::vector<Op> pending_;   // queued, not yet submitted
  unsigned outstanding_ = 0;  // queued and not yet reaped
  unsigned submitted_ = 0;    // given to the workers, not yet reaped
  // Shared, under m_.
  std::mutex m_;
  std::condition_variable_any work_cv_;
  std::condition_variable done_cv_;
  std::vector<Op> ring_;  // submitted, not yet started: FIFO of capacity depth_
  std::size_t ring_head_ = 0, ring_count_ = 0;
  std::vector<Completion> done_;
  bool interrupted_ = false;
  std::vector<std::jthread> workers_;  // last member
};

}  // namespace

std::unique_ptr<IAsyncReader> make_thread_reader(NativeFile&& file, const Options& opt) {
  return std::make_unique<Reader<ThreadEngine>>(std::move(file), opt);
}

std::unique_ptr<IAsyncWriter> make_thread_writer(NativeFile&& file, const Options& opt) {
  return std::make_unique<Writer<ThreadEngine>>(std::move(file), opt);
}

}  // namespace hce::aio::detail
