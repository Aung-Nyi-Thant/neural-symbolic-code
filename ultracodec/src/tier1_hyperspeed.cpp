// Tier 1: Hyper-Speed. Byte-level LZ77 in the LZ4 block layout:
//
//   sequence = token, [litlen ext], literals, u16 offset, [matchlen ext]
//   token    = (min(litlen,15) << 4) | min(matchlen-4, 15)
//   ext      = run of 255s plus a final byte < 255, added to the nibble
//
// The last sequence of a block has literals only. The encoder keeps the last
// 5 bytes literal and starts no match in the last 12, so the decoder's fast
// path (16-byte vector copies that may run up to 15 bytes past a literal run
// or match) only has to check its two slack conditions; near the end of the
// block it falls back to exact copies. Every length and offset is checked
// against both buffers, so a corrupt block throws instead of overrunning.
//
// Match finder: one 64 Ki-entry table (256 KiB, stays in L2) indexed by a
// multiplicative hash of 5 bytes, positions relative to the block; misses
// accelerate the scan step (1 + misses/64) so incompressible data costs
// little. 64 KiB window (u16 offsets).
//
// Entropy mode (model_param 1, the default): the same block sizes, but the
// block is a stream of entropy-coded sequences (seq_codec.hpp, one literal
// stream) with three repeat offsets and a window of the whole block. The
// parser is zstd-"dfast"-style and greedy: it tries repeat offset 0 at the
// next position, then an 8-byte hash table (2^17) and a 5-byte one (2^16),
// takes the longer match, extends it backwards and checks repeat offset 1
// right after every match. Misses accelerate the scan like the raw mode.
#include <algorithm>

#include "block_engine.hpp"
#include "lz_util.hpp"
#include "seq_codec.hpp"
#include "tiers.hpp"

namespace ultracodec::detail {
namespace {

constexpr unsigned kHashLog = 16;  // 14 (64 KiB) is 1.5% larger at the same speed
constexpr std::size_t kMinMatch = 4;
constexpr std::size_t kLastLiterals = 5;
constexpr std::size_t kMfLimit = 12;
constexpr std::size_t kMaxOffset = 65535;
constexpr unsigned kSkipTrigger = 6;

ULTRA_INLINE u32 hash5(u64 v) noexcept {
  return static_cast<u32>(((v << 24) * 889523592379ull) >> (64 - kHashLog));
}

ULTRA_INLINE u8* put_length(u8* op, std::size_t len) noexcept {
  while (len >= 255) { *op++ = 255; len -= 255; }
  *op++ = static_cast<u8>(len);
  return op;
}

// `lit_room` = readable bytes from `lit` (the wild copy reads 16-byte chunks).
ULTRA_INLINE u8* emit_sequence(u8* op, const u8* lit, std::size_t litlen, std::size_t lit_room, std::size_t offset,
                               std::size_t mlen) noexcept {
  u8* token = op++;
  std::size_t ml = mlen - kMinMatch;
  u8 t = static_cast<u8>((std::min<std::size_t>(litlen, 15) << 4) | std::min<std::size_t>(ml, 15));
  if (litlen >= 15) op = put_length(op, litlen - 15);
  if (litlen + 16 <= lit_room) {
    wild_copy(op, lit, litlen);  // the output buffer carries 32 bytes of slack
  } else {
    std::memcpy(op, lit, litlen);
  }
  op += litlen;
  store16(op, static_cast<u16>(offset));
  op += 2;
  if (ml >= 15) op = put_length(op, ml - 15);
  *token = t;
  return op;
}

class HyperSpeedCodec final : public BlockCodec {
 public:
  void encode(std::span<const u8>, std::span<const u8> raw, std::vector<u8>& out) override {
    const std::size_t n = raw.size();
    const std::size_t start = out.size();
    out.resize(start + n + n / 255 + 32);  // worst case + wild-copy slack
    u8* op = out.data() + start;
    const u8* src = raw.data();
    std::size_t anchor = 0;

    if (n >= kMfLimit + 1) {
      std::fill(table_, table_ + (1u << kHashLog), 0u);
      const std::size_t mflimit = n - kMfLimit;
      const u8* matchlimit = src + n - kLastLiterals;
      std::size_t ip = 1;
      table_[hash5(load64(src))] = 0;
      while (ip <= mflimit) {
        // Find a 4-byte match.
        std::size_t cand = 0;
        unsigned attempts = 1u << kSkipTrigger;
        bool found = false;
        while (ip <= mflimit) {
          u64 v = load64(src + ip);
          u32 h = hash5(v);
          cand = table_[h];
          table_[h] = static_cast<u32>(ip);
          if (cand < ip && ip - cand <= kMaxOffset && load32(src + cand) == static_cast<u32>(v)) {
            found = true;
            break;
          }
          ip += attempts++ >> kSkipTrigger;
        }
        if (!found) break;
        while (ip > anchor && cand > 0 && src[ip - 1] == src[cand - 1]) { --ip; --cand; }
        std::size_t mlen = kMinMatch + match_length(src + ip + kMinMatch, src + cand + kMinMatch, matchlimit);
        op = emit_sequence(op, src + anchor, ip - anchor, n - anchor, ip - cand, mlen);
        ip += mlen;
        anchor = ip;
        if (ip > mflimit) break;
        // Re-seed the table just behind the cursor (cheap ratio win).
        table_[hash5(load64(src + ip - 2))] = static_cast<u32>(ip - 2);
      }
    }
    // Last literals.
    std::size_t litlen = n - anchor;
    *op++ = static_cast<u8>(std::min<std::size_t>(litlen, 15) << 4);
    if (litlen >= 15) op = put_length(op, litlen - 15);
    std::memcpy(op, src + anchor, litlen);
    op += litlen;
    out.resize(static_cast<std::size_t>(op - out.data()));
  }

  void decode(std::span<const u8>, std::span<const u8> payload, std::span<u8> dst, Progress*,
              std::size_t) override {
    const u8* ip = payload.data();
    const u8* const iend = ip + payload.size();
    u8* op = dst.data();
    u8* const ostart = op;
    u8* const oend = op + dst.size();
    auto fail = [] { throw Error("ultracodec: corrupt tier-1 block"); };
    auto read_ext = [&](std::size_t& len) {
      u8 b;
      do {
        if (ip >= iend) fail();
        b = *ip++;
        len += b;
      } while (b == 255);
      if (len > (std::size_t{1} << 27)) fail();
    };
    for (;;) {
      if (ip >= iend) fail();
      const unsigned token = *ip++;
      std::size_t litlen = token >> 4;
      if (litlen == 15) read_ext(litlen);
      if (litlen > static_cast<std::size_t>(iend - ip) || litlen > static_cast<std::size_t>(oend - op)) fail();
      if (ULTRA_LIKELY(op + litlen + 16 <= oend && ip + litlen + 16 <= iend)) {
        wild_copy(op, ip, litlen);
      } else {
        std::memcpy(op, ip, litlen);
      }
      op += litlen;
      ip += litlen;
      if (ip == iend) break;  // the literal-only last sequence
      if (iend - ip < 2) fail();
      const std::size_t offset = load16(ip);
      ip += 2;
      if (offset == 0 || offset > static_cast<std::size_t>(op - ostart)) fail();
      std::size_t mlen = (token & 15) + kMinMatch;
      if ((token & 15) == 15) read_ext(mlen);
      if (mlen > static_cast<std::size_t>(oend - op)) fail();
      copy_match(op, offset, mlen, oend);
      op += mlen;
    }
    if (op != oend) fail();
  }

 private:
  alignas(64) u32 table_[1u << kHashLog];
};

class EntropyCodec final : public BlockCodec {
 public:
  void encode(std::span<const u8>, std::span<const u8> raw, std::vector<u8>& out) override {
    const std::size_t n = raw.size();
    const u8* const src = raw.data();
    writer_.reset(src);
    std::fill(std::begin(long_), std::end(long_), 0u);
    std::fill(std::begin(short_), std::end(short_), 0u);
    u32 rep[3] = {1, 4, 8};
    std::size_t anchor = 0, ip = 1;
    const std::size_t ilimit = n > 16 ? n - 16 : 0;  // matches start before this; keeps 8-byte loads in bounds
    const u8* const limit = src + n - 8;            // the last 8 bytes stay literals
    // Tables hold position + 1; 0 is empty.
    const auto put = [&](std::size_t pos) {
      long_[hash8(load64(src + pos))] = static_cast<u32>(pos + 1);
      short_[hash5(load64(src + pos))] = static_cast<u32>(pos + 1);
    };
    while (ip < ilimit) {
      const u64 v = load64(src + ip);
      const u32 hl = hash8(v), hs = hash5(v);
      const std::size_t cl = long_[hl], cs = short_[hs];
      long_[hl] = short_[hs] = static_cast<u32>(ip + 1);
      std::size_t start = ip, len = 0;
      u32 offcode = 0;
      if (ip + 1 > rep[0] && load32(src + ip + 1) == load32(src + ip + 1 - rep[0])) {
        start = ip + 1;
        len = 4 + match_length(src + start + 4, src + start + 4 - rep[0], limit);
        offcode = 1;
      } else {
        if (cl && load64(src + cl - 1) == v) len = 8 + match_length(src + ip + 8, src + cl - 1 + 8, limit), offcode = static_cast<u32>(ip - (cl - 1)) + 3;
        if (cs && load32(src + cs - 1) == static_cast<u32>(v)) {
          const std::size_t ls = 4 + match_length(src + ip + 4, src + cs - 1 + 4, limit);
          if (ls > len) len = ls, offcode = static_cast<u32>(ip - (cs - 1)) + 3;
        }
        if (len < kMinMatch) {
          ip += 1 + ((ip - anchor) >> kSkipTrigger);
          continue;
        }
        const std::size_t dist = offcode - 3;
        while (start > anchor && start > dist && src[start - 1] == src[start - 1 - dist]) { --start; ++len; }
      }
      writer_.sequence(anchor, start, len, offcode, rep);
      ip = anchor = start + len;
      if (ip < ilimit) {
        put(start + 2);
        put(ip - 2);
      }
      // Repeat offset 1 straight after a match (zstd's "immediate repcode").
      while (ip < ilimit && ip >= rep[1] && load32(src + ip) == load32(src + ip - rep[1])) {
        const std::size_t l = 4 + match_length(src + ip + 4, src + ip + 4 - rep[1], limit);
        put(ip);
        writer_.sequence(anchor, ip, l, 2, rep);
        ip = anchor = ip + l;
      }
    }
    writer_.literals(anchor, n);
    writer_.finish(out);
  }

  void decode(std::span<const u8>, std::span<const u8> payload, std::span<u8> out, Progress*, std::size_t) override {
    Reader r(payload);
    reader_.decode(r, out);
  }

 private:
  static ULTRA_INLINE u32 hash8(u64 v) noexcept { return static_cast<u32>((v * 0xCF1BBCDCB7A56463ull) >> (64 - kLongLog)); }
  static ULTRA_INLINE u32 hash5(u64 v) noexcept { return static_cast<u32>(((v << 24) * 889523592379ull) >> (64 - kShortLog)); }
  static constexpr unsigned kLongLog = 17, kShortLog = 16;

  alignas(64) u32 long_[1u << kLongLog];
  alignas(64) u32 short_[1u << kShortLog];
  seq::Writer<1> writer_;
  seq::Reader<1> reader_;
};

class HyperSpeedEngine final : public BlockEngine {
 public:
  explicit HyperSpeedEngine(const EngineOptions& o)
      : BlockEngine(TierLevel::Tier1_HyperSpeed, "Tier1 Hyper-Speed (LZ77)", o, {20, 0}), entropy_(o.tier1_entropy) {}

 protected:
  u32 choose_model_param(std::size_t) const override { return entropy_ ? 1 : 0; }
  void validate_model_param(u32 p) const override {
    if (p > 1) throw Error("ultracodec: bad tier-1 parameter");
  }
  std::unique_ptr<BlockCodec> make_codec(u32 p, std::size_t, std::size_t) const override {
    if (p == 1) return std::make_unique<EntropyCodec>();
    return std::make_unique<HyperSpeedCodec>();
  }

 private:
  bool entropy_;
};

}  // namespace

std::unique_ptr<ICompressionEngine> make_tier1(const EngineOptions& o) {
  return std::make_unique<HyperSpeedEngine>(o);
}

}  // namespace ultracodec::detail
