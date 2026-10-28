// The block container shared by all tiers: splits the input into
// power-of-two blocks, runs the tier's BlockCodec on worker threads, frames
// the result (core.hpp has the layout) and validates every byte of an
// archive before trusting it.
//
// Wavefront scheduling (Tiers 3/4): block k is primed with the first
// prime_bytes of block k-1. Encoding is fully parallel (all raw data is
// known). Decoding is staggered: the worker on block k waits only until
// block k-1 has decoded its first prime_bytes, so with 1 MiB blocks and a
// 64 KiB prefix the decoders start ~6% of a block apart instead of a whole
// block apart, and long-range context survives the block boundary.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "common.hpp"

namespace ultracodec::detail {

// Set once a block's first prime_bytes are final in the output buffer.
class Progress {
 public:
  void signal() noexcept {
    if (!ready_.exchange(true, std::memory_order_acq_rel)) ready_.notify_all();
  }
  void wait() const noexcept { ready_.wait(false, std::memory_order_acquire); }
  bool ready() const noexcept { return ready_.load(std::memory_order_acquire); }

 private:
  std::atomic<bool> ready_{false};
};

// One per worker thread; owns that thread's model tables and scratch.
class BlockCodec {
 public:
  virtual ~BlockCodec() = default;
  // Appends the coded form of `raw` to `out`. `prime` is the wavefront
  // prefix (empty for block 0 or when priming is off).
  virtual void encode(std::span<const u8> prime, std::span<const u8> raw, std::vector<u8>& out) = 0;
  // Decodes `payload` into exactly out.size() bytes. Calls
  // progress->signal() as soon as the first `signal_at` bytes are final
  // (when progress is set). Throws Error on malformed input.
  virtual void decode(std::span<const u8> prime, std::span<const u8> payload, std::span<u8> out,
                      Progress* progress, std::size_t signal_at) = 0;
};

struct DedupDefaults {
  std::size_t far_min = 0;   // 0 = no long-range dedup pass
  std::size_t near_min = 0;
};

struct TierDefaults {
  unsigned block_log2;
  std::size_t prime_bytes;  // 0 = this tier never primes
  DedupDefaults dedup = {};
};

class BlockEngine : public ICompressionEngine {
 public:
  std::vector<u8> compress(std::span<const u8> input, EngineStats* stats) final;
  std::vector<u8> decompress(std::span<const u8> archive, EngineStats* stats) final;
  TierLevel get_tier() const noexcept final { return tier_; }
  std::string_view get_name() const noexcept final { return name_; }

 protected:
  BlockEngine(TierLevel tier, std::string_view name, const EngineOptions& options, TierDefaults defaults);

  // The tier's model parameter for this encode (stored in the header). It
  // must not depend on the thread count: archives are identical for any
  // number of threads.
  virtual u32 choose_model_param(std::size_t input_size) const = 0;
  // A codec for blocks of at most block_size bytes primed with at most
  // prime_bytes (decoders take all three from the header).
  virtual std::unique_ptr<BlockCodec> make_codec(u32 model_param, std::size_t block_size,
                                                 std::size_t prime_bytes) const = 0;
  // Rejects a header model parameter this tier cannot use (corrupt input).
  virtual void validate_model_param(u32 model_param) const = 0;
  // Most workers the memory budget allows for this model parameter.
  virtual unsigned worker_cap(u32) const { return UINT32_MAX; }

  const EngineOptions& options() const noexcept { return options_; }
  unsigned workers_for(std::size_t blocks, u32 model_param) const noexcept;

 private:
  struct CodecKey {
    u32 param = 0;
    std::size_t block = 0, prime = 0;
    bool operator==(const CodecKey&) const = default;
  };
  void prepare_codecs(const CodecKey& key, unsigned workers);
  BlockCodec& codec(unsigned worker);

  TierLevel tier_;
  std::string_view name_;
  EngineOptions options_;
  unsigned block_log2_;
  std::size_t prime_bytes_;
  DedupDefaults dedup_;
  CodecKey codec_key_;
  std::vector<std::unique_ptr<BlockCodec>> codecs_;
};

// Runs fn(index, worker) for index in [0, n) on `workers` threads (the
// caller is worker 0). The first exception is rethrown after all workers
// stopped; on_error runs once, before the rethrow, so blocked waiters can be
// released.
void parallel_blocks(std::size_t n, unsigned workers, const std::function<void(std::size_t, unsigned)>& fn,
                     const std::function<void()>& on_error);

}  // namespace ultracodec::detail
