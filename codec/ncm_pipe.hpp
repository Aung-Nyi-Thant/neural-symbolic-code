// ncm_pipe.hpp - the encoder's two-stage pipeline (encoder side only, no
// format change).
//
// When encoding, every bit is known in advance, so the counter half of the
// model (contexts, buckets, match model: CMModel::produce_byte) never waits
// for the mixer. A helper thread runs it ahead and hands each coded decision
// to the coding thread as a 32-byte record through a single-producer,
// single-consumer ring; the coding thread mixes, codes and trains
// (CMModel::mix_code). Both halves touch disjoint tables and run the same
// arithmetic as the sequential loop, so the bits are identical (R15).
#pragma once

#include "codec/ncm_simd.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

namespace hce::ncm {

class Pipe {
 public:
  using Rec = simd::Lanes;
  static constexpr std::size_t kRecs = std::size_t{1} << 13;  // 256 KiB of records
  static constexpr std::size_t kBatch = 64;                   // records per publication
  static constexpr std::size_t kBytes = kRecs * sizeof(Rec);

  // A thread and a ring; nullptr when either cannot be had.
  static std::unique_ptr<Pipe> create() noexcept;
  ~Pipe();
  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;

  // Coding thread: runs job(ctx, *this) on the helper, which fills the ring.
  void start(void (*job)(void*, Pipe&) noexcept, void* ctx) noexcept;
  // Coding thread: the next record, or nullptr once the job has finished and
  // every record was taken. pop() releases it.
  const Rec* next() noexcept {
    if (tail_ == avail_ && !refill()) return nullptr;
    return &ring_[tail_ & (kRecs - 1)];
  }
  void pop() noexcept {
    if ((++tail_ & (kBatch - 1)) == 0) publish_tail();
  }
  // Coding thread: stops the job early and waits for it; then idle again.
  void abort() noexcept;
  // Coding thread: waits until the job has finished (after next() == nullptr
  // it already has).
  void join() noexcept;

  // Helper, inside the job: the slot of the next record, nullptr once aborted.
  Rec* slot() noexcept {
    if (head_ - room_ == kRecs && !wait_room()) return nullptr;
    return &ring_[head_ & (kRecs - 1)];
  }
  void commit() noexcept {
    if ((++head_ & (kBatch - 1)) == 0) publish_head();
  }

 private:
  Pipe() = default;
  void loop() noexcept;
  bool refill() noexcept;
  bool wait_room() noexcept;
  void publish_head() noexcept;
  void publish_tail() noexcept;
  void reset_counters() noexcept;

  std::unique_ptr<Rec[]> ring_;

  // Producer side (helper thread).
  alignas(64) std::size_t head_ = 0;  // records written
  std::size_t room_ = 0;              // the consumer's tail as last seen
  // Consumer side (coding thread).
  alignas(64) std::size_t tail_ = 0;  // records taken
  std::size_t avail_ = 0;             // the producer's head as last seen
  bool finished_seen_ = false;

  // The producer's head, with kDone once the job has returned: a change of
  // value, so a consumer sleeping on it always wakes.
  static constexpr std::size_t kDone = std::size_t{1} << (8 * sizeof(std::size_t) - 1);
  alignas(64) std::atomic<std::size_t> pub_head_{0};
  alignas(64) std::atomic<std::size_t> pub_tail_{0};
  std::atomic<bool> aborted_{false};

  // Job hand-off.
  std::mutex m_;
  std::condition_variable cv_;
  void (*job_)(void*, Pipe&) noexcept = nullptr;  // guarded by m_
  void* ctx_ = nullptr;
  bool quit_ = false;
  std::thread helper_;
};

}  // namespace hce::ncm
