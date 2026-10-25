// ncm_pipe.cpp - the encoder's two-stage pipeline (see the header).
//
// Waiting spins briefly, then sleeps on the atomic: the stages run at similar
// speeds, so most waits are short, and a stage that sleeps is woken by the
// other's next publication.
#include "codec/ncm_pipe.hpp"

#include <new>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace hce::ncm {
namespace {

constexpr int kSpins = 2048;

inline void relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

}  // namespace

std::unique_ptr<Pipe> Pipe::create() noexcept {
  std::unique_ptr<Pipe> p(new (std::nothrow) Pipe);
  if (!p) return nullptr;
  p->ring_.reset(new (std::nothrow) Rec[kRecs]);
  if (!p->ring_) return nullptr;
  try {
    p->helper_ = std::thread([q = p.get()] { q->loop(); });
  } catch (...) {
    return nullptr;
  }
  return p;
}

Pipe::~Pipe() {
  if (!helper_.joinable()) return;
  {
    const std::lock_guard lk(m_);
    quit_ = true;
  }
  cv_.notify_one();
  helper_.join();
}

void Pipe::reset_counters() noexcept {
  head_ = room_ = tail_ = avail_ = 0;
  finished_seen_ = false;
  pub_head_.store(0, std::memory_order_relaxed);
  pub_tail_.store(0, std::memory_order_relaxed);
  aborted_.store(false, std::memory_order_relaxed);
}

void Pipe::start(void (*job)(void*, Pipe&) noexcept, void* ctx) noexcept {
  reset_counters();  // the helper is idle: nothing else touches them now
  {
    const std::lock_guard lk(m_);  // publishes the reset and the job's inputs
    job_ = job;
    ctx_ = ctx;
  }
  cv_.notify_one();
}

void Pipe::loop() noexcept {
  for (;;) {
    void (*job)(void*, Pipe&) noexcept = nullptr;
    void* ctx = nullptr;
    {
      std::unique_lock lk(m_);
      cv_.wait(lk, [this] { return quit_ || job_; });
      if (quit_) return;
      job = std::exchange(job_, nullptr);
      ctx = ctx_;
    }
    job(ctx, *this);
    pub_head_.store(head_ | kDone, std::memory_order_release);
    pub_head_.notify_one();
  }
}

void Pipe::publish_head() noexcept {
  pub_head_.store(head_, std::memory_order_release);
  pub_head_.notify_one();
}

void Pipe::publish_tail() noexcept {
  pub_tail_.store(tail_, std::memory_order_release);
  pub_tail_.notify_one();
}

// Consumer: waits for more records; false when the job is done and all were taken.
bool Pipe::refill() noexcept {
  for (int spin = 0;; ++spin) {
    const std::size_t v = pub_head_.load(std::memory_order_acquire);
    if ((v & ~kDone) != avail_) {
      avail_ = v & ~kDone;
      return true;
    }
    if (v & kDone) {
      finished_seen_ = true;
      return false;
    }
    if (spin < kSpins) {
      relax();
    } else {
      pub_head_.wait(v, std::memory_order_acquire);
      spin = 0;
    }
  }
}

// Producer: waits for room in the ring; false once aborted.
bool Pipe::wait_room() noexcept {
  for (int spin = 0;; ++spin) {
    if (aborted_.load(std::memory_order_acquire)) return false;
    const std::size_t t = pub_tail_.load(std::memory_order_acquire);
    if (t != room_) {
      room_ = t;
      if (head_ - room_ < kRecs) return true;
    }
    if (spin < kSpins) {
      relax();
    } else {
      pub_tail_.wait(t, std::memory_order_acquire);
      spin = 0;
    }
  }
}

void Pipe::join() noexcept {
  for (int spin = 0; !finished_seen_; ++spin) {
    const std::size_t v = pub_head_.load(std::memory_order_acquire);
    if (v & kDone) {
      finished_seen_ = true;
    } else if (spin < kSpins) {
      relax();
    } else {
      pub_head_.wait(v, std::memory_order_acquire);
      spin = 0;
    }
  }
}

void Pipe::abort() noexcept {
  aborted_.store(true, std::memory_order_release);
  // Wake a producer sleeping for room: a tail it has not seen.
  pub_tail_.store(pub_tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
  pub_tail_.notify_one();
  join();
}

}  // namespace hce::ncm
