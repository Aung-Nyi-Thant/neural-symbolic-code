// Entropy-coded LZ77 sequences, shared by Tier 2 and Tier 1's entropy mode.
//
// Stream layout (after any tier-specific prefix):
//   varint sequence count
//   kLitContexts literal streams   literal bytes split by the byte before them
//                                  (16 contexts: its high nibble; 1: one stream)
//   litlen, matchlen, offset       rANS code streams (rans_static.hpp)
//   varint size + extra bits       LSB-first, per sequence: LL, ML, OF
// Codes:
//   literal-length codes        litlen       -> code + extra bits
//   match-length codes          matchlen - 4 -> code + extra bits
//   offset codes                1..3 = repeat offset 0..2, else distance + 3,
//                               code = floor(log2(v)), extra = low bits of v
// Length codes: v < 16 is its own code; above, two codes per octave. Repeat
// offsets start at {1, 4, 8} and update as in zstd (a hit on rep k moves it
// to the front). The last literals of a block follow the last sequence.
//
// Decoding entropy-decodes every stream up front (vector rANS kernels), then
// replays the sequences with bounded copies; any inconsistency throws.
#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "common.hpp"
#include "lz_util.hpp"
#include "rans_static.hpp"

namespace ultracodec::detail::seq {

constexpr std::size_t kMinMatch = 4;

ULTRA_INLINE void length_code(u32 v, u8& code, unsigned& nbits, u32& extra) noexcept {
  if (v < 16) {
    code = static_cast<u8>(v);
    nbits = 0;
    extra = 0;
    return;
  }
  unsigned hb = highbit32(v);
  code = static_cast<u8>(16 + (hb - 4) * 2 + ((v >> (hb - 1)) & 1));
  nbits = hb - 1;
  extra = v & ((1u << nbits) - 1);
}
// Inverse of length_code; returns false for a code that cannot occur.
ULTRA_INLINE bool length_base(u8 code, u32& base, unsigned& nbits) noexcept {
  if (code < 16) { base = code; nbits = 0; return true; }
  unsigned k = code - 16u;
  unsigned hb = 4 + k / 2;
  if (hb > 26) return false;
  nbits = hb - 1;
  base = (1u << hb) | ((k & 1u) << (hb - 1));
  return true;
}

ULTRA_INLINE unsigned offset_cost(u32 offcode) noexcept { return highbit32(offcode); }

// Applies a sequence's offset code to the repeat offsets; returns the distance.
ULTRA_INLINE u32 update_repeats(u32 offcode, u32 rep[3]) noexcept {
  if (offcode <= 3) {
    const unsigned k = offcode - 1;
    const u32 d = rep[k];
    if (k == 1) std::swap(rep[0], rep[1]);
    else if (k == 2) { rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = d; }
    return d;
  }
  rep[2] = rep[1];
  rep[1] = rep[0];
  rep[0] = offcode - 3;
  return rep[0];
}

// LSB-first bit packer; `n` <= 26 per call.
class BitWriter {
 public:
  ULTRA_INLINE void put(u32 v, unsigned n) {
    acc_ |= static_cast<u64>(v) << fill_;
    fill_ += n;
    if (fill_ >= 32) {
      if (ULTRA_UNLIKELY(len_ + 8 > out_.size())) out_.resize(out_.size() * 2 + 64);
      store64(out_.data() + len_, acc_);  // only the low 4 bytes count; the rest is rewritten later
      len_ += 4;
      acc_ >>= 32;
      fill_ -= 32;
    }
  }
  std::vector<u8>& finish() {
    out_.resize(len_ + 8);
    while (fill_ > 0) {
      out_[len_++] = static_cast<u8>(acc_);
      acc_ >>= 8;
      fill_ = fill_ > 8 ? fill_ - 8 : 0;
    }
    out_.resize(len_);
    acc_ = 0;
    return out_;
  }
  void clear() { len_ = 0; acc_ = 0; fill_ = 0; out_.resize(std::max<std::size_t>(out_.capacity(), 64)); }

 private:
  std::vector<u8> out_;
  std::size_t len_ = 0;
  u64 acc_ = 0;
  unsigned fill_ = 0;
};

class BitReader {
 public:
  explicit BitReader(std::span<const u8> s) : p_(s.data()), end_(s.data() + s.size()) {}
  u32 get(unsigned n) {
    if (!n) return 0;
    while (fill_ < n) {
      if (p_ == end_) throw Error("ultracodec: truncated sequence extra bits");
      acc_ |= static_cast<u64>(*p_++) << fill_;
      fill_ += 8;
    }
    u32 v = static_cast<u32>(acc_ & ((u64{1} << n) - 1));
    acc_ >>= n;
    fill_ -= n;
    return v;
  }

 private:
  const u8* p_;
  const u8* end_;
  u64 acc_ = 0;
  unsigned fill_ = 0;
};

template <unsigned kLitContexts>
ULTRA_INLINE unsigned literal_context(unsigned previous) noexcept {
  static_assert(kLitContexts == 1 || kLitContexts == 16);
  return kLitContexts == 1 ? 0u : previous >> 4;
}

// Collects one block's sequences, then writes the streams.
template <unsigned kLitContexts>
class Writer {
 public:
  void reset(const u8* src) {
    src_ = src;
    for (auto& l : lits_) l.clear();
    ll_.clear();
    ml_.clear();
    of_.clear();
    bits_.clear();
  }

  void literals(std::size_t from, std::size_t to) {
    if constexpr (kLitContexts == 1) {
      lits_[0].insert(lits_[0].end(), src_ + from, src_ + to);
    } else {
      for (std::size_t i = from; i < to; ++i) lits_[literal_context<kLitContexts>(i ? src_[i - 1] : 0)].push_back(src_[i]);
    }
  }

  // Literals [anchor, ip), then a match of `len` bytes with `offcode`; updates rep.
  void sequence(std::size_t anchor, std::size_t ip, std::size_t len, u32 offcode, u32 rep[3]) {
    u8 code;
    unsigned nb;
    u32 extra;
    literals(anchor, ip);
    length_code(static_cast<u32>(ip - anchor), code, nb, extra);
    ll_.push_back(code);
    bits_.put(extra, nb);
    length_code(static_cast<u32>(len - kMinMatch), code, nb, extra);
    ml_.push_back(code);
    bits_.put(extra, nb);
    const unsigned oc = highbit32(offcode);
    of_.push_back(static_cast<u8>(oc));
    bits_.put(offcode & ((1u << oc) - 1), oc);
    update_repeats(offcode, rep);
  }

  void finish(std::vector<u8>& out) {
    put_varint(out, static_cast<u32>(ll_.size()));
    for (auto& l : lits_) rans_encode(l, out);
    rans_encode(ll_, out);
    rans_encode(ml_, out);
    rans_encode(of_, out);
    auto& eb = bits_.finish();
    put_varint(out, static_cast<u32>(eb.size()));
    out.insert(out.end(), eb.begin(), eb.end());
  }

 private:
  const u8* src_ = nullptr;
  std::vector<u8> lits_[kLitContexts], ll_, ml_, of_;
  BitWriter bits_;
};

// Decodes the streams that `r` points at into exactly dst.size() bytes; `r` must end there.
template <unsigned kLitContexts>
class Reader {
 public:
  void decode(detail::Reader& r, std::span<u8> dst) {
    const std::size_t nseq = r.varint();
    if (nseq > dst.size() / kMinMatch + 1) throw Error("ultracodec: corrupt sequence count");
    std::size_t nlits = 0;
    for (auto& l : lits_) {
      rans_decode(r, l, dst.size() - nlits);
      nlits += l.size();
    }
    rans_decode(r, ll_, nseq);
    rans_decode(r, ml_, nseq);
    rans_decode(r, of_, nseq);
    if (ll_.size() != nseq || ml_.size() != nseq || of_.size() != nseq) throw Error("ultracodec: corrupt sequence streams");
    const std::size_t eb = r.varint();
    BitReader br(r.take(eb));
    if (r.remaining()) throw Error("ultracodec: trailing block data");

    u8* op = dst.data();
    u8* const ostart = op;
    u8* const oend = op + dst.size();
    const u8* lp[kLitContexts];
    const u8* lend[kLitContexts];
    for (unsigned c = 0; c < kLitContexts; ++c) {
      lp[c] = lits_[c].data();
      lend[c] = lp[c] + lits_[c].size();
    }
    auto fail = [] { throw Error("ultracodec: corrupt sequence block"); };
    unsigned prev = 0;  // byte before op (0 at the block start)
    auto copy_literals = [&](std::size_t litlen) {
      if (litlen > static_cast<std::size_t>(oend - op)) fail();
      if constexpr (kLitContexts == 1) {
        if (litlen > static_cast<std::size_t>(lend[0] - lp[0])) fail();
        std::memcpy(op, lp[0], litlen);
        lp[0] += litlen;
        op += litlen;
      } else {
        for (u8* const stop = op + litlen; op < stop; ++op) {
          const unsigned c = literal_context<kLitContexts>(prev);
          if (ULTRA_UNLIKELY(lp[c] == lend[c])) fail();
          prev = *lp[c]++;
          *op = static_cast<u8>(prev);
        }
      }
    };
    u32 rep[3] = {1, 4, 8};
    for (std::size_t s = 0; s < nseq; ++s) {
      u32 base;
      unsigned nb;
      if (!length_base(ll_[s], base, nb)) fail();
      const std::size_t litlen = base + br.get(nb);
      if (!length_base(ml_[s], base, nb)) fail();
      const std::size_t mlen = base + br.get(nb) + kMinMatch;
      const unsigned oc = of_[s];
      if (oc > 30) fail();
      const u32 offcode = (1u << oc) | br.get(oc);
      copy_literals(litlen);
      const u32 dist = update_repeats(offcode, rep);
      if (dist == 0 || dist > static_cast<std::size_t>(op - ostart) || mlen > static_cast<std::size_t>(oend - op)) fail();
      copy_match(op, dist, mlen, oend);
      op += mlen;
      prev = op[-1];
    }
    copy_literals(static_cast<std::size_t>(oend - op));
    for (unsigned c = 0; c < kLitContexts; ++c)
      if (lp[c] != lend[c]) fail();
  }

 private:
  std::vector<u8> lits_[kLitContexts], ll_, ml_, of_;
};

}  // namespace ultracodec::detail::seq
