// Binary rANS for the context-mixing tiers, two interleaved states.
//
// The model's 12-bit P(bit = 1) is the frequency of 1 (range [0, p)), the
// rest is 0's range. Bit k of a block uses state k & 1, so the two coder
// dependency chains overlap in the decoder; 16-bit renormalization reads at
// most one word per bit. rANS is LIFO, so the encoder records (p, bit) per
// bit while the model runs forward and codes the record backwards at the end
// of the block; the decoder runs forward with the model in lock step.
//
// Encoder divisions use exact 128-bit reciprocal multiplication (a 32-bit
// state divided by a 12-bit frequency: m = ceil(2^44 / f) is exact for every
// x < 2^32, Granlund-Montgomery with l = 12).
//
// Stream: u32 word_count, then word_count u16 words (the two final states
// first, low half first).
#pragma once

#include <array>
#include <vector>

#include "common.hpp"

namespace ultracodec::detail {

__extension__ typedef unsigned __int128 u128;

struct RansRecipTable {
  u64 v[4097];
};
constexpr RansRecipTable make_rans_recip() {
  RansRecipTable t{};
  for (u64 f = 1; f <= 4096; ++f) t.v[f] = ((u64{1} << 44) + f - 1) / f;
  return t;
}
inline constexpr RansRecipTable kRansRecip = make_rans_recip();

class BinaryRansEncoder {
 public:
  static constexpr u32 kL = 1u << 16;

  // Record one bit (p = 12-bit probability of 1, clamped to [1, 4095]).
  ULTRA_INLINE void put(unsigned p, unsigned bit) { rec_.push_back(static_cast<u16>(p | (bit << 15))); }
  void clear() { rec_.clear(); }
  void reserve(std::size_t bits) { rec_.reserve(bits); }

  // Codes every recorded bit and appends the stream to out.
  void finish(std::vector<u8>& out) {
    const std::size_t n = rec_.size();
    words_.resize(n + 8);
    std::size_t pos = words_.size();
    u32 x[2] = {kL, kL};
    for (std::size_t i = n; i-- > 0;) {
      const u32 r = rec_[i];
      const u32 p = r & 0xFFF;
      const bool bit = r >> 15;
      const u32 f = bit ? p : 4096 - p;
      const u32 start = bit ? 0 : p;
      u32& s = x[i & 1];
      if (s >= (f << 20)) {
        words_[--pos] = static_cast<u16>(s);
        s >>= 16;
      }
      const u32 q = static_cast<u32>((static_cast<u128>(s) * kRansRecip.v[f]) >> 44);
      s = (q << 12) + (s - q * f) + start;
    }
    for (int l = 1; l >= 0; --l) {
      words_[--pos] = static_cast<u16>(x[l] >> 16);
      words_[--pos] = static_cast<u16>(x[l]);
    }
    const std::size_t count = words_.size() - pos;
    put_u32(out, static_cast<u32>(count));
    const std::size_t at = out.size();
    out.resize(at + 2 * count);
    std::memcpy(out.data() + at, words_.data() + pos, 2 * count);
    rec_.clear();
  }

 private:
  std::vector<u16> rec_;
  std::vector<u16> words_;
};

class BinaryRansDecoder {
 public:
  static constexpr u32 kL = 1u << 16;

  // Reads the stream header; throws Error on a truncated stream.
  void init(Reader& r) {
    const std::size_t count = r.u32v();
    if (count < 4) throw Error("ultracodec: truncated CM stream");
    auto s = r.take(2 * count);
    p_ = s.data() + 8;
    end_ = s.data() + s.size();
    x_[0] = load32(s.data());
    x_[1] = load32(s.data() + 4);
    k_ = 0;
  }

  ULTRA_INLINE unsigned get(unsigned p) {
    u32& x = x_[k_ & 1];
    ++k_;
    const u32 s = x & 0xFFF;
    unsigned bit;
    if (s < p) {
      bit = 1;
      x = p * (x >> 12) + s;
    } else {
      bit = 0;
      x = (4096 - p) * (x >> 12) + s - p;
    }
    if (x < kL) {
      if (ULTRA_UNLIKELY(end_ - p_ < 2)) throw Error("ultracodec: truncated CM stream");
      x = (x << 16) | load16(p_);
      p_ += 2;
    }
    return bit;
  }

  // True when the stream was consumed exactly and both states returned to
  // their initial value (a cheap corruption check).
  bool finished() const noexcept { return p_ == end_ && x_[0] == kL && x_[1] == kL; }

 private:
  const u8* p_ = nullptr;
  const u8* end_ = nullptr;
  u32 x_[2] = {0, 0};
  u64 k_ = 0;
};

}  // namespace ultracodec::detail
