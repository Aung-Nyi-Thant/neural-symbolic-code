// Tier 2: Consumer. Lazy hash-chain LZ77 with repeat offsets, entropy coded
// with the 8-way interleaved static rANS (rans_static.hpp).
//
// A block (default 4 MiB, the LZ window) becomes four rANS streams and one
// raw bit stream:
//   literals                    bytes
//   literal-length codes        litlen       -> code + extra bits
//   match-length codes          matchlen - 4 -> code + extra bits
//   offset codes                1..3 = repeat offset 0..2, else distance + 3,
//                               code = floor(log2(v)), extra = low bits of v
//   extra bits                  LSB-first, per sequence: LL, ML, OF
// Length codes: v < 16 is its own code; above, two codes per octave.
//
// Parser: zstd-"lazy2"-style. At each position it checks the three repeat
// offsets and walks a hash chain (depth 48, 4-byte hash), then tries the next
// two positions and keeps the later match only when its estimated gain
// (4 * length - log2(distance)) is better. Repeat offsets update as in zstd
// (a hit on rep k moves it to the front).
//
// Decoding entropy-decodes the streams (vector rANS kernels), then replays
// the sequences with the same bounded copies as Tier 1.
#include <algorithm>

#include "block_engine.hpp"
#include "lz_util.hpp"
#include "rans_static.hpp"
#include "tiers.hpp"

namespace ultracodec::detail {
namespace {

constexpr std::size_t kMinMatch = 4;
constexpr unsigned kHashLog = 18;
#ifndef T2D
#define T2D 16
#endif
constexpr unsigned kChainDepth = T2D;
constexpr std::size_t kNiceLength = 128;

ULTRA_INLINE u32 hash4(u32 v) noexcept { return (v * 2654435761u) >> (32 - kHashLog); }

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

class BitWriter {
 public:
  void put(u32 v, unsigned n) {
    if (!n) return;
    acc_ |= static_cast<u64>(v) << fill_;
    fill_ += n;
    while (fill_ >= 8) { out_.push_back(static_cast<u8>(acc_)); acc_ >>= 8; fill_ -= 8; }
  }
  std::vector<u8>& finish() {
    if (fill_) out_.push_back(static_cast<u8>(acc_));
    acc_ = 0;
    fill_ = 0;
    return out_;
  }
  void clear() { out_.clear(); acc_ = 0; fill_ = 0; }

 private:
  std::vector<u8> out_;
  u64 acc_ = 0;
  unsigned fill_ = 0;
};

class BitReader {
 public:
  explicit BitReader(std::span<const u8> s) : p_(s.data()), end_(s.data() + s.size()) {}
  u32 get(unsigned n) {
    if (!n) return 0;
    while (fill_ < n) {
      if (p_ == end_) throw Error("ultracodec: truncated tier-2 extra bits");
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

struct Match {
  std::size_t len = 0;
  u32 offcode = 0;  // 1..3 repeat, else distance + 3
};

class ConsumerCodec final : public BlockCodec {
 public:
  void encode(std::span<const u8>, std::span<const u8> raw, std::vector<u8>& out) override {
    const std::size_t n = raw.size();
    src_ = raw.data();
    n_ = n;
    head_.assign(std::size_t{1} << kHashLog, UINT32_MAX);
    chain_.resize(n);
    next_insert_ = 0;
    lits_.clear();
    ll_.clear();
    ml_.clear();
    of_.clear();
    bits_.clear();
    u32 rep[3] = {1, 4, 8};
    std::size_t anchor = 0, ip = 0;
    const std::size_t ilimit = n > 16 ? n - 16 : 0;  // matches start before this; keeps 8-byte loads in bounds

    while (ip < ilimit) {
      Match m = best_match(ip, rep);
      if (m.len < kMinMatch) {
        ip += 1 + ((ip - anchor) >> 7);  // skip faster through incompressible runs
        continue;
      }
      // Lazy evaluation of the next two positions.
      for (int depth = 1; depth <= 2 && ip + 1 < ilimit; ++depth) {
        Match m2 = best_match(ip + 1, rep);
        long gain2 = static_cast<long>(m2.len * 4) - static_cast<long>(offset_cost(m2.offcode));
        long gain1 = static_cast<long>(m.len * 4) - static_cast<long>(offset_cost(m.offcode)) + (depth == 1 ? 4 : 7);
        if (m2.len >= kMinMatch && gain2 > gain1) {
          ++ip;
          m = m2;
        } else {
          break;
        }
      }
      // Extend real-offset matches backwards over pending literals.
      if (m.offcode > 3) {
        std::size_t dist = m.offcode - 3;
        while (ip > anchor && ip > dist && src_[ip - 1] == src_[ip - 1 - dist]) { --ip; ++m.len; }
      }
      emit(anchor, ip, m, rep);
      ip += m.len;
      anchor = ip;
    }
    // Trailing literals (no sequence): the decoder copies what is left.
    lits_.insert(lits_.end(), src_ + anchor, src_ + n);

    put_varint(out, static_cast<u32>(ll_.size()));
    rans_encode(lits_, out);
    rans_encode(ll_, out);
    rans_encode(ml_, out);
    rans_encode(of_, out);
    auto& eb = bits_.finish();
    put_varint(out, static_cast<u32>(eb.size()));
    out.insert(out.end(), eb.begin(), eb.end());
  }

  void decode(std::span<const u8>, std::span<const u8> payload, std::span<u8> dst, Progress*,
              std::size_t) override {
    Reader r(payload);
    const std::size_t nseq = r.varint();
    if (nseq > dst.size() / kMinMatch + 1) throw Error("ultracodec: corrupt tier-2 sequence count");
    rans_decode(r, lits_, dst.size());
    rans_decode(r, ll_, nseq);
    rans_decode(r, ml_, nseq);
    rans_decode(r, of_, nseq);
    if (ll_.size() != nseq || ml_.size() != nseq || of_.size() != nseq)
      throw Error("ultracodec: corrupt tier-2 streams");
    const std::size_t eb = r.varint();
    BitReader br(r.take(eb));
    if (r.remaining()) throw Error("ultracodec: trailing tier-2 data");

    u8* op = dst.data();
    u8* const ostart = op;
    u8* const oend = op + dst.size();
    const u8* lp = lits_.data();
    const u8* const lend = lp + lits_.size();
    u32 rep[3] = {1, 4, 8};
    auto fail = [] { throw Error("ultracodec: corrupt tier-2 block"); };
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
      if (litlen > static_cast<std::size_t>(lend - lp) || litlen > static_cast<std::size_t>(oend - op)) fail();
      if (op + litlen + 16 <= oend && lp + litlen + 16 <= lend) {
        wild_copy(op, lp, litlen);
      } else {
        std::memcpy(op, lp, litlen);
      }
      op += litlen;
      lp += litlen;
      u32 dist;
      if (offcode <= 3) {
        unsigned k = offcode - 1;
        dist = rep[k];
        if (k == 1) std::swap(rep[0], rep[1]);
        else if (k == 2) { rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = dist; }
      } else {
        dist = offcode - 3;
        rep[2] = rep[1];
        rep[1] = rep[0];
        rep[0] = dist;
      }
      if (dist == 0 || dist > static_cast<std::size_t>(op - ostart) || mlen > static_cast<std::size_t>(oend - op)) fail();
      copy_match(op, dist, mlen, oend);
      op += mlen;
    }
    const std::size_t tail = static_cast<std::size_t>(lend - lp);
    if (tail != static_cast<std::size_t>(oend - op)) fail();
    std::memcpy(op, lp, tail);
  }

 private:
  void insert_until(std::size_t ip) {
    for (; next_insert_ < ip; ++next_insert_) {
      u32 h = hash4(load32(src_ + next_insert_));
      chain_[next_insert_] = head_[h];
      head_[h] = static_cast<u32>(next_insert_);
    }
  }

  Match best_match(std::size_t ip, const u32 rep[3]) {
    insert_until(ip);
    const u8* cur = src_ + ip;
    const u8* limit = src_ + n_ - 8;  // last 8 bytes stay literals
    Match best;
    long best_gain = 0;
    for (unsigned k = 0; k < 3; ++k) {
      u32 d = rep[k];
      if (d > ip) continue;
      if (load32(cur) != load32(cur - d)) continue;
      std::size_t len = 4 + match_length(cur + 4, cur + 4 - d, limit);
      long gain = static_cast<long>(len * 4) - static_cast<long>(k);  // repeats are nearly free
      if (gain > best_gain) { best_gain = gain; best.len = len; best.offcode = k + 1; }
    }
    if (best.len >= kNiceLength) return best;
    const u32 v = load32(cur);
    u32 cand = head_[hash4(v)];
    std::size_t best_real = 0;
    for (unsigned depth = 0; depth < kChainDepth && cand != UINT32_MAX; ++depth, cand = chain_[cand]) {
      const u8* c = src_ + cand;
      if (best_real >= 4 && c[best_real] != cur[best_real]) continue;
      if (load32(c) != v) continue;
      std::size_t len = 4 + match_length(cur + 4, c + 4, limit);
      if (len <= best_real) continue;
      best_real = len;
      u32 offcode = static_cast<u32>(ip - cand) + 3;
      long gain = static_cast<long>(len * 4) - static_cast<long>(offset_cost(offcode));
      if (gain > best_gain) { best_gain = gain; best.len = len; best.offcode = offcode; }
      if (len >= kNiceLength) break;
    }
    return best;
  }

  void emit(std::size_t anchor, std::size_t ip, const Match& m, u32 rep[3]) {
    u8 code;
    unsigned nb;
    u32 extra;
    const std::size_t litlen = ip - anchor;
    lits_.insert(lits_.end(), src_ + anchor, src_ + ip);
    length_code(static_cast<u32>(litlen), code, nb, extra);
    ll_.push_back(code);
    bits_.put(extra, nb);
    length_code(static_cast<u32>(m.len - kMinMatch), code, nb, extra);
    ml_.push_back(code);
    bits_.put(extra, nb);
    unsigned oc = highbit32(m.offcode);
    of_.push_back(static_cast<u8>(oc));
    bits_.put(m.offcode & ((1u << oc) - 1), oc);
    if (m.offcode <= 3) {
      unsigned k = m.offcode - 1;
      u32 d = rep[k];
      if (k == 1) std::swap(rep[0], rep[1]);
      else if (k == 2) { rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = d; }
    } else {
      rep[2] = rep[1];
      rep[1] = rep[0];
      rep[0] = m.offcode - 3;
    }
  }

  const u8* src_ = nullptr;
  std::size_t n_ = 0;
  std::size_t next_insert_ = 0;
  std::vector<u32> head_, chain_;
  std::vector<u8> lits_, ll_, ml_, of_;
  BitWriter bits_;
};

class ConsumerEngine final : public BlockEngine {
 public:
  explicit ConsumerEngine(const EngineOptions& o)
      : BlockEngine(TierLevel::Tier2_Consumer, "Tier2 Consumer (LZ77 + rANS)", o, {22, 0}) {}

 protected:
  u32 choose_model_param(std::size_t) const override { return 0; }
  void validate_model_param(u32 p) const override {
    if (p != 0) throw Error("ultracodec: bad tier-2 parameter");
  }
  std::unique_ptr<BlockCodec> make_codec(u32, std::size_t, std::size_t) const override { return std::make_unique<ConsumerCodec>(); }
};

}  // namespace

std::unique_ptr<ICompressionEngine> make_tier2(const EngineOptions& o) {
  return std::make_unique<ConsumerEngine>(o);
}

}  // namespace ultracodec::detail
