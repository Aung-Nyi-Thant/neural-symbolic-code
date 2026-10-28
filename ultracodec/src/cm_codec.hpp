// BlockCodec and engine glue for the context-mixing tier (3): runs a CmModel
// over the wavefront prefix and the block, coding with binary rANS.
//
// Payload: one rANS stream (rans_binary.hpp) per P::kChunkBytes of the block
// (one for the whole block when kChunkBytes is 0). The encoder records two
// bytes per bit until a chunk ends, so chunks bound that record. The model is
// reset for every block, so blocks decode independently apart from the prefix.
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
    const std::size_t chunk = chunk_bytes(raw.size());
    enc_.reserve(chunk * 8);
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const u8 c = raw[i];
      for (int b = 7; b >= 0; --b) {
        const unsigned bit = (c >> b) & 1;
        enc_.put(static_cast<unsigned>(model_.p()), bit);
        model_.update(static_cast<int>(bit));
      }
      if ((i + 1) % chunk == 0 && i + 1 < raw.size()) enc_.finish(out);
    }
    enc_.finish(out);
  }

  void decode(std::span<const u8> prime, std::span<const u8> payload, std::span<u8> dst, Progress* progress,
              std::size_t signal_at) override {
    Reader r(payload);
    BinaryRansDecoder dec;
    dec.init(r);
    model_.reset();
    model_.prime(prime);
    const std::size_t chunk = chunk_bytes(dst.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
      if (i && i % chunk == 0) {
        if (!dec.finished()) throw Error("ultracodec: CM stream did not end cleanly (corrupt block)");
        dec.init(r);
      }
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
    if (r.remaining()) throw Error("ultracodec: trailing CM data");
  }

 private:
  static std::size_t chunk_bytes(std::size_t block) noexcept {
    return P::kChunkBytes ? P::kChunkBytes : std::max<std::size_t>(block, 1);
  }

  CmModel<P> model_;
  BinaryRansEncoder enc_;
};

// Engine for the CM profile P, plus Legacy, the profile of archives written
// before P. Each worker owns one model. model_param is the hashed-table size
// per worker in units of 1024 bucket pairs; P's archives add kCompactParam,
// so the two formats read apart. EngineOptions::tier3_compact picks which one
// the encoder writes.
template <class P, class Legacy>
class CmEngine final : public BlockEngine {
  static_assert(P::kCompact && !Legacy::kCompact);
  static constexpr u32 kCompactParam = 0x80000000u;

 public:
  CmEngine(TierLevel tier, std::string_view name, const EngineOptions& o)
      : BlockEngine(tier, name, o, {P::kBlockLog2, P::kPrimeBytes, {P::kDedupFar, P::kDedupNear}}),
        budget_(o.memory_budget ? o.memory_budget : P::kDefaultBudget),
        block_(o.block_size ? o.block_size : std::size_t{1} << P::kBlockLog2),
        compact_(o.tier3_compact) {}

 protected:
  u32 choose_model_param(std::size_t input_size) const override {
    return compact_ ? kCompactParam | table_units<P>(input_size) : table_units<Legacy>(input_size);
  }
  unsigned worker_cap(u32 p) const override {
    const bool compact = p & kCompactParam;
    const std::size_t per = compact ? fixed_bytes<P>() + table_bytes<P>(p & ~kCompactParam)
                                    : fixed_bytes<Legacy>() + table_bytes<Legacy>(p);
    return static_cast<unsigned>(std::max<std::size_t>(1, std::min<std::size_t>(budget_ / per, UINT32_MAX)));
  }
  void validate_model_param(u32 p) const override {
    const bool compact = p & kCompactParam;
    const u32 units = p & ~kCompactParam;
    const u64 bytes = compact ? table_bytes<P>(units) : table_bytes<Legacy>(units);
    if (units == 0 || bytes > (compact ? P::kMaxTableBytes : Legacy::kMaxTableBytes))
      throw Error("ultracodec: CM table size out of range");
  }
  std::unique_ptr<BlockCodec> make_codec(u32 p, std::size_t block, std::size_t prime) const override {
    if (p & kCompactParam) return std::make_unique<CmBlockCodec<P>>(std::size_t{p & ~kCompactParam} * 1024, block + prime);
    return std::make_unique<CmBlockCodec<Legacy>>(std::size_t{p} * 1024, block + prime);
  }

 private:
  template <class Q>
  static constexpr std::size_t kUnitBytes = 1024 * CmModel<Q>::kPairBytes;
  template <class Q>
  static std::size_t table_bytes(u32 units) noexcept { return std::size_t{units} * kUnitBytes<Q>; }

  // Fixed per-worker memory besides the hashed table.
  template <class Q>
  std::size_t fixed_bytes() const {
    const std::size_t record = Q::kChunkBytes ? std::min(block_, Q::kChunkBytes) * 16 * 2  // rANS record + words
                                              : block_ * 16;
    return 2 * block_                                         // history (prefix + block)
           + record                                           // rANS record (2 bytes per bit)
           + (std::size_t{4} << Q::kMatchLog2) * Q::kMatch    // match tables
           + 65536 * 4 + 65536 * 66                           // order-1 table, APM 1
           + (Q::kSecondApm ? (std::size_t{1} << 20) * 66 : 0)
           + (std::size_t{2} << 20);                          // mixers, scratch
  }
  // Per-worker table: the profile's size, shrunk only when the whole budget
  // cannot hold one worker, and never larger than the input can use. The
  // budget then caps the number of workers, so the archive does not depend
  // on the thread count and decoding stays within the budget too.
  template <class Q>
  u32 table_units(std::size_t input_size) const {
    const std::size_t fixed = fixed_bytes<Q>();
    std::size_t table = Q::kTableBytes;
    if (budget_ < fixed + table) table = budget_ > fixed + (std::size_t{1} << 20) ? budget_ - fixed : (std::size_t{1} << 20);
    const std::size_t useful = std::max<std::size_t>(std::min(input_size, block_) * Q::kMaxTablePerByte, std::size_t{1} << 20);
    table = std::min(table, useful);
    return static_cast<u32>(std::max<std::size_t>(1, table / kUnitBytes<Q>));
  }

  std::size_t budget_;
  std::size_t block_;
  bool compact_;
};

}  // namespace ultracodec::detail
