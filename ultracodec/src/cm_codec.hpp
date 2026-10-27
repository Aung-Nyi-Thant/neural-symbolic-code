// BlockCodec and engine glue for the context-mixing tiers (3 and 4): runs a
// CmModel over the wavefront prefix and the block, coding with binary rANS.
//
// Payload: u32 stream (rans_binary.hpp). The model is reset for every block,
// so blocks decode independently apart from the prefix.
#pragma once

#include <algorithm>

#include "block_engine.hpp"
#include "cm_model.hpp"
#include "rans_binary.hpp"

namespace ultracodec::detail {

template <class P>
class CmBlockCodec final : public BlockCodec {
 public:
  CmBlockCodec(std::size_t pairs, std::size_t max_history) : model_(pairs, max_history) {}

  void encode(std::span<const u8> prime, std::span<const u8> raw, std::vector<u8>& out) override {
    model_.reset();
    model_.prime(prime);
    enc_.clear();
    enc_.reserve(raw.size() * 8);
    for (u8 c : raw) {
      for (int b = 7; b >= 0; --b) {
        const unsigned bit = (c >> b) & 1;
        enc_.put(static_cast<unsigned>(model_.p()), bit);
        model_.update(static_cast<int>(bit));
      }
    }
    enc_.finish(out);
  }

  void decode(std::span<const u8> prime, std::span<const u8> payload, std::span<u8> dst, Progress* progress,
              std::size_t signal_at) override {
    Reader r(payload);
    BinaryRansDecoder dec;
    dec.init(r);
    if (r.remaining()) throw Error("ultracodec: trailing CM data");
    model_.reset();
    model_.prime(prime);
    for (std::size_t i = 0; i < dst.size(); ++i) {
      unsigned c = 0;
      for (int b = 0; b < 8; ++b) {
        const unsigned bit = dec.get(static_cast<unsigned>(model_.p()));
        model_.update(static_cast<int>(bit));
        c = (c << 1) | bit;
      }
      dst[i] = static_cast<u8>(c);
      if (progress && i + 1 == signal_at) progress->signal();
    }
    if (!dec.finished()) throw Error("ultracodec: CM stream did not end cleanly (corrupt block)");
  }

 private:
  CmModel<P> model_;
  BinaryRansEncoder enc_;
};

// Engine for a CM profile. Each worker owns one model; model_param is the
// hashed-table size per worker in units of 1024 bucket pairs (128 KiB).
template <class P>
class CmEngine final : public BlockEngine {
 public:
  CmEngine(TierLevel tier, std::string_view name, const EngineOptions& o)
      : BlockEngine(tier, name, o, {P::kBlockLog2, P::kPrimeBytes}),
        budget_(o.memory_budget ? o.memory_budget : P::kDefaultBudget),
        block_(o.block_size ? o.block_size : std::size_t{1} << P::kBlockLog2) {}

 protected:
  // Fixed per-worker memory besides the hashed table.
  std::size_t fixed_bytes() const {
    return 2 * block_                                         // history (prefix + block)
           + block_ * 16                                      // rANS record (2 bytes per bit)
           + (std::size_t{4} << P::kMatchLog2) * P::kMatch    // match tables
           + 65536 * 4 + 65536 * 66                           // order-1 table, APM 1
           + (P::kSecondApm ? (std::size_t{1} << 20) * 66 : 0)
           + (std::size_t{2} << 20);                          // mixers, scratch
  }
  // Per-worker table: the profile's size, shrunk only when the whole budget
  // cannot hold one worker, and never larger than the input can use. The
  // budget then caps the number of workers, so the archive does not depend
  // on the thread count and decoding stays within the budget too.
  u32 choose_model_param(std::size_t input_size) const override {
    const std::size_t fixed = fixed_bytes();
    std::size_t table = P::kTableBytes;
    if (budget_ < fixed + table) table = budget_ > fixed + (std::size_t{1} << 20) ? budget_ - fixed : (std::size_t{1} << 20);
    const std::size_t useful = std::max<std::size_t>(std::min(input_size, block_) * P::kMaxTablePerByte, std::size_t{1} << 20);
    table = std::min(table, useful);
    return static_cast<u32>(std::max<std::size_t>(1, table / (128 * 1024)));
  }
  unsigned worker_cap(u32 p) const override {
    const std::size_t per = fixed_bytes() + std::size_t{p} * 128 * 1024;
    return static_cast<unsigned>(std::max<std::size_t>(1, std::min<std::size_t>(budget_ / per, UINT32_MAX)));
  }
  void validate_model_param(u32 p) const override {
    if (p == 0 || static_cast<u64>(p) * 128 * 1024 > P::kMaxTableBytes)
      throw Error("ultracodec: CM table size out of range");
  }
  std::unique_ptr<BlockCodec> make_codec(u32 p, std::size_t block, std::size_t prime) const override {
    return std::make_unique<CmBlockCodec<P>>(std::size_t{p} * 1024, block + prime);
  }

 private:
  std::size_t budget_;
  std::size_t block_;
};

}  // namespace ultracodec::detail
