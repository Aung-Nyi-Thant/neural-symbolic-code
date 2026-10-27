// Tier 2: Consumer. Optional word-dictionary pre-filter, then lazy
// hash-chain LZ77 with repeat offsets, entropy coded with the 8-way
// interleaved static rANS (rans_static.hpp).
//
// Block payload (default block 2 MiB, the LZ window; 4 MiB is 0.9% smaller
// and 2x slower):
//   u8 dict                     0 none | 1 prose | 2 code (word_filter.hpp)
//   [varint filtered size]      when dict != 0; LZ77 runs on the filtered bytes
//   sequences                   seq_codec.hpp, 16 literal streams split by the
//                               high nibble of the byte before them
//
// The filter is used when four 16 KiB samples of the block shrink below 85%
// with one of the two built-in dictionaries (English prose, code
// identifiers): on the benchmark corpus it takes 14% off the novel and 6-7%
// off Python and Perl, and is skipped for XML and binaries. Splitting the
// literals by context takes another 1%; the decoder entropy-decodes all 16
// streams up front with the vector kernels and only picks the stream per
// literal while replaying sequences.
//
// Parser: zstd-"lazy2"-style. At each position it checks the three repeat
// offsets and walks a hash chain (depth 16, 4-byte hash), then tries the next
// two positions and keeps the later match only when its estimated gain
// (4 * length - log2(distance)) is better. Repeat offsets update as in zstd
// (a hit on rep k moves it to the front).
//
// Decoding entropy-decodes the streams (vector rANS kernels), then replays
// the sequences with the same bounded copies as Tier 1, then undoes the
// filter.
#include <algorithm>

#include "block_engine.hpp"
#include "lz_util.hpp"
#include "seq_codec.hpp"
#include "tiers.hpp"
#include "word_filter.hpp"

namespace ultracodec::detail {
namespace {

using seq::kMinMatch;
using seq::offset_cost;
constexpr unsigned kHashLog = 18;
constexpr unsigned kChainDepth = 16;  // 48 gains ~1.6% for 2.5x less speed
constexpr std::size_t kNiceLength = 128;
constexpr unsigned kLitContexts = 16;  // literal stream = previous byte >> 4

ULTRA_INLINE u32 hash4(u32 v) noexcept { return (v * 2654435761u) >> (32 - kHashLog); }

struct Match {
  std::size_t len = 0;
  u32 offcode = 0;  // 1..3 repeat, else distance + 3
};

class ConsumerCodec final : public BlockCodec {
 public:
  void encode(std::span<const u8>, std::span<const u8> raw, std::vector<u8>& out) override {
    const WordDict dict = word_filter_choose(raw);
    std::span<const u8> lz = raw;
    if (dict != WordDict::None) {
      word_filter_encode(dict, raw, filtered_);
      if (filtered_.size() < raw.size()) lz = filtered_;
    }
    out.push_back(static_cast<u8>(lz.data() == raw.data() ? WordDict::None : dict));
    if (lz.data() != raw.data()) put_varint(out, static_cast<u32>(lz.size()));
    const std::size_t n = lz.size();
    src_ = lz.data();
    n_ = n;
    head_.assign(std::size_t{1} << kHashLog, UINT32_MAX);
    chain_.resize(n);
    next_insert_ = 0;
    writer_.reset(src_);
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
        if (m2.len < kMinMatch) break;
        long gain2 = static_cast<long>(m2.len * 4) - static_cast<long>(offset_cost(m2.offcode));
        long gain1 = static_cast<long>(m.len * 4) - static_cast<long>(offset_cost(m.offcode)) + (depth == 1 ? 4 : 7);
        if (gain2 > gain1) {
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
      writer_.sequence(anchor, ip, m.len, m.offcode, rep);
      ip += m.len;
      anchor = ip;
    }
    // Trailing literals (no sequence): the decoder copies what is left.
    writer_.literals(anchor, n);
    writer_.finish(out);
  }

  void decode(std::span<const u8>, std::span<const u8> payload, std::span<u8> out, Progress*,
              std::size_t) override {
    Reader r(payload);
    const u8 dict = r.u8v();
    if (dict > static_cast<u8>(WordDict::Code)) throw Error("ultracodec: bad tier-2 dictionary");
    std::span<u8> dst = out;
    if (dict) {
      const std::size_t fsize = r.varint();
      if (fsize > 2 * out.size()) throw Error("ultracodec: corrupt tier-2 filtered size");
      filtered_.resize(fsize);
      dst = filtered_;
    }
    reader_.decode(r, dst);
    if (dict) word_filter_decode(static_cast<WordDict>(dict), dst, out);
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

  const u8* src_ = nullptr;
  std::size_t n_ = 0;
  std::size_t next_insert_ = 0;
  std::vector<u32> head_, chain_;
  std::vector<u8> filtered_;
  seq::Writer<kLitContexts> writer_;
  seq::Reader<kLitContexts> reader_;
};

class ConsumerEngine final : public BlockEngine {
 public:
  explicit ConsumerEngine(const EngineOptions& o)
      : BlockEngine(TierLevel::Tier2_Consumer, "Tier2 Consumer (LZ77 + rANS)", o, {21, 0}) {}

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
