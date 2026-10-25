// ncm_model.hpp - the model of the neural CM codec (see neural_cm_codec.hpp):
// format constants (R17), the context, match and mixer state, and the bit and
// byte coding steps, templated on the SIMD kernel policy (ncm_simd.hpp) so the
// kernels inline into them. Internal to the codec's translation units.
//
// For every bit, ten counter-based experts (orders 0-4 and 6, word, column,
// log-field and sparse contexts) and a match model emit logits. Three layer-1
// mixers, whose weight sets are selected by the partial byte, the match state
// and the previous byte, combine them; a layer-2 mixer selected by bit
// position combines those; two APMs refine the result; a 32-bit binary
// arithmetic coder codes the bit. Inside long matches a byte costs one coded
// decision.
//
// Encoding, decoding and the encoder's replay run one templated code path
// that differs only in where each bit comes from, so they cannot disagree.
#pragma once

#include "codec/ncm_coder.hpp"
#include "codec/ncm_kernels.hpp"
#include "codec/ncm_simd.hpp"
#include "codec/ncm_tables.hpp"
#include "codec/neural_cm_codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hce::ncm {

using hc::u16;
using hc::u8;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// ------------------------------------------------ format constants (R17) ---
// Changing anything in this block changes the coded bits: that is a new profile.
constexpr int kHashed = 8;           // hashed experts: o2, o3, o4, o6, word, column, sparse, field
constexpr int kNodes = kHashed + 2;  // plus the direct o1 and o0 tables
constexpr int kInputs = kNodes + 3;  // plus two match inputs and the bias
static_assert(kInputs <= kLanes);
constexpr int kSetsA = 256;  // layer-1 weight sets selected by the partial byte
constexpr int kSetsB = 32;   // ... by the match state
constexpr int kSetsC = 256;  // ... by the previous byte
constexpr int kSetsF = 8;    // layer-2 weight sets, by bit position
constexpr int kLr1 = 36;     // layer-1 learning rate, Q10 of the error
constexpr int kLr2 = 16;     // layer-2 learning rate
// |error| <= 65536, so the scaled errors stay within the kernels' int16 bound.
static_assert(65536 * kLr1 >> 10 <= 32767 && 65536 * kLr2 >> 10 <= 32767);
constexpr i16 kInitW1 = 4096;  // Q14: 0.25
constexpr i16 kInitW2 = 5461;  // Q14: 1/3
constexpr i16 kBias = 256;     // constant input: logit 1.0
constexpr u32 kMinMatch = 6, kMaxVerify = 32, kMaxMatch = 65535;
constexpr u64 kWordScan = 64, kLineScan = 4096;
constexpr int kPMin = 4, kPMax = 65532;  // coded probabilities, Q16
constexpr int kApmRate = 6;
constexpr std::size_t kApmRow = 33;
constexpr u32 kWordMul = 0x2F0F3D33u;

// Pipeline records (produce_byte): the inputs in lanes [0, kInputs), then the
// match state, (c1 << 8) | c0, and bitpos | y << 3 | kRecFast.
constexpr int kRecFast = 16;  // a fast-path decision: P(hit) in lane 0
static_assert(kInputs + 3 <= kLanes);

struct LevelSpec {
  u8 bucket_bits;  // log2 of the buckets per hashed expert
  u8 match_bits;   // log2 of the match-table entries
  u32 fast_len;    // match length from which a byte is coded as one decision
};
constexpr std::array<LevelSpec, kNeuralCMMaxLevel + 1> kLevels{{
    {16, 18, 32},     // ~22 MiB
    {17, 19, 64},     // ~39 MiB
    {17, 20, 128},    // ~41 MiB
    {18, 21, 256},    // ~77 MiB
    {19, 22, 1024},   // ~149 MiB
}};
constexpr const LevelSpec& spec_for(u8 level) noexcept { return kLevels[std::min<u8>(level, kNeuralCMMaxLevel)]; }

// --------------------------------------------------------------- helpers ---
constexpr u64 hash(u64 v, u64 salt) noexcept {
  v += (salt + 1) * 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}

constexpr bool is_word(int c) noexcept { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128; }
// Field separators of logs and records: the field context counts them per line.
constexpr bool is_sep(int c) noexcept {
  return c == ' ' || c == '\t' || c == ',' || c == ';' || c == '=' || c == '|' || c == ':';
}
constexpr u32 fold(int c) noexcept { return static_cast<u32>(c >= 'A' && c <= 'Z' ? c + 32 : c); }

// Bit-history counters: 12-bit probability and a 4-bit count, packed so that
// zeroed memory is the initial state (p = 0.5, n = 0); node_next() is
// the update rule, applied through the table kNodeNext.

// Decision counter: a 22-bit probability over a 10-bit count, so that long
// runs can reach P(miss) ~ 1e-4; zero is p = 0.75, n = 0.
constexpr u32 kCounterInit = 0x300000;  // 0.75, Q22
constexpr int counter_p(u32 c) noexcept { return static_cast<int>(((c >> 10) ^ kCounterInit) >> 6); }  // Q16
inline void counter_update(u32& c, int y) noexcept {
  i64 p = (c >> 10) ^ kCounterInit;
  const u32 n = c & 1023;
  p += (((y ? 0x3FFFFF : 0) - p) * kRate[n] + (1 << 15)) >> 16;
  c = (static_cast<u32>(p ^ kCounterInit) << 10) | (n + (n < 1023));
}

constexpr int len_bucket(u32 len) noexcept {
  return len < 16 ? static_cast<int>(len) : std::min(31, 12 + static_cast<int>(std::bit_width(len)));
}

// log2(x) in Q16 for x >= 1, by repeated squaring: exact integer arithmetic.
constexpr u64 log2_q16(u64 x) noexcept {
  const int ip = static_cast<int>(std::bit_width(x)) - 1;
  u64 y = ip >= 31 ? x >> (ip - 31) : x << (31 - ip);  // [2^31, 2^32): x / 2^ip in Q31
  u64 r = static_cast<u64>(ip) << 16;
  for (int i = 15; i >= 0; --i) {
    y = (y * y) >> 31;
    if (y >= (u64{1} << 32)) {
      y >>= 1;
      r |= u64{1} << i;
    }
  }
  return r;
}
static_assert(log2_q16(1) == 0 && log2_q16(2) == 65536 && log2_q16(1024) == 10 * 65536);

struct Bucket {
  u16 s[16];  // s[0]: tag; s[1..15]: the counters of one nibble's binary tree
};
static_assert(sizeof(Bucket) == 32);


// Units of a table (cache lines, APM rows) written since the last reset. A
// frame reset clears only those, so it costs time in proportion to the work
// done in the frame, never to the table size: a stream of tiny frames cannot
// multiply the decoder's work (R19). The tables end up identical either way.
struct Dirty {
  u64* bits = nullptr;
  std::size_t units = 0;

  static constexpr std::size_t words(std::size_t n) noexcept { return (n + 63) / 64; }
  // One OR per mark, on the hot path; the count is taken once, at the reset.
  void mark(std::size_t u) noexcept { bits[u >> 6] |= u64{1} << (u & 63); }
  void forget() noexcept { std::memset(bits, 0, words(units) * sizeof(u64)); }
  std::size_t marked() const noexcept {
    std::size_t n = 0;
    for (std::size_t i = 0; i < words(units); ++i) n += static_cast<std::size_t>(std::popcount(bits[i]));
    return n;
  }
  // clear_one(u) for every marked unit, or clear_all() when that is cheaper.
  template <class One, class All>
  void drain(One&& clear_one, All&& clear_all) noexcept {
    if (marked() > units / 8) {
      clear_all();
    } else {
      for (std::size_t i = 0; i < words(units); ++i)
        for (u64 w = bits[i]; w; w &= w - 1) clear_one(i * 64 + static_cast<std::size_t>(std::countr_zero(w)));
    }
    forget();
  }
};

// The hash of a hashed expert's bucket for a nibble: its context alone for
// the first nibble (bitpos 0), mixed with the partial byte c0 for the second.
constexpr u64 nibble_hash(u64 ctx, u32 c0, int bitpos) noexcept {
  u64 hv = ctx;
  if (bitpos) {
    hv = (hv + (u64{c0} + 1) * 0x9E3779B97F4A7C15ull);
    hv = (hv ^ (hv >> 29)) * 0xBF58476D1CE4E5B9ull;
    hv ^= hv >> 32;
  }
  return hv;
}

// ----------------------------------------------------------------- model ---
// Placement-constructed at the start of the arena the codec is bound to.
struct CMModel {
  // Tables, carved from the arena.
  Bucket* buckets = nullptr;  // kHashed regions of 2^bucket_bits buckets
  u32* match_tab = nullptr;   // frame-relative positions, truncated to 32 bits (R21)
  u16* o1 = nullptr;          // [previous byte][partial byte]
  u16* o0 = nullptr;          // [partial byte]
  u32* flags = nullptr;       // fast-path decisions: [length bucket][expected byte]
  u32* msm = nullptr;         // match model: [length bucket][expected bit]
  u16* apm1 = nullptr;        // [partial byte][33]
  u16* apm2 = nullptr;        // [previous byte][partial byte][33]
  i16* wa = nullptr;          // layer-1 weights, kLanes per set
  i16* wb = nullptr;
  i16* wc = nullptr;
  i16* wf = nullptr;  // layer-2 weights, 4 per set
  Dirty dirty_buckets;  // 64-byte lines: two buckets each
  Dirty dirty_match;    // 64-byte lines: sixteen entries each
  Dirty dirty_apm;      // APM2 rows
  u16 apm_row[kApmRow]{};  // the initial APM row: squash over the interpolation knots
  bool stale = false;      // decoder: a frame started, the reset is still owed
  const Kernels* k = nullptr;  // the mixer kernels: which instantiation of the bit loop runs
  LevelSpec spec{};
  u64 window = 0;
  u32 bucket_mask = 0;
  int match_shift = 0;

  // History state: re-derived from the history at every block start.
  u64 pos = 0;  // frame-relative position of the next byte
  u64 c8 = 0;   // the last eight bytes
  u32 word0 = 0, word1 = 0;
  u32 field = 0;  // separators seen in the current line
  u64 line_start = 0, prev_line_start = 0;  // equal: no previous line known
  u64 mptr = 0;  // position of the byte the match predicts
  u32 mlen = 0;
  int expected = -1;
  u64 ctx[kHashed]{};

  // Per-bit state.
  u32 c0 = 1;   // partial byte with a leading 1
  u32 nib = 1;  // partial nibble with a leading 1: the counter inside a bucket
  int bitpos = 0;
  Bucket* cur[kHashed]{};  // this nibble's bucket of every hashed expert

  // ---------------------------------------------------------- layout ---
  struct Layout {
    std::size_t buckets, match, o1, o0, flags, msm, apm1, apm2, wa, wb, wc, wf, dbuckets, dmatch, dapm, total;
  };
  static constexpr std::size_t bucket_lines(const LevelSpec& s) noexcept {
    return (std::size_t{kHashed} << s.bucket_bits) / 2;  // two 32-byte buckets per line
  }
  static constexpr std::size_t match_lines(const LevelSpec& s) noexcept {
    return (std::size_t{1} << s.match_bits) / 16;  // sixteen entries per line
  }
  static constexpr std::size_t align64(std::size_t n) noexcept { return (n + 63) & ~std::size_t{63}; }
  static constexpr Layout layout(const LevelSpec& s) noexcept {
    Layout l{};
    std::size_t at = align64(sizeof(CMModel));
    const auto take = [&at](std::size_t bytes) {
      const std::size_t here = at;
      at = align64(at + bytes);
      return here;
    };
    l.buckets = take((std::size_t{kHashed} << s.bucket_bits) * sizeof(Bucket));
    l.match = take((std::size_t{1} << s.match_bits) * sizeof(u32));
    l.o1 = take(std::size_t{65536} * sizeof(u16));
    l.o0 = take(std::size_t{256} * sizeof(u16));
    l.flags = take(std::size_t{16 * 256} * sizeof(u32));
    l.msm = take(std::size_t{64} * sizeof(u32));
    l.apm1 = take(std::size_t{256} * kApmRow * sizeof(u16));
    l.apm2 = take(std::size_t{65536} * kApmRow * sizeof(u16));
    l.wa = take(std::size_t{kSetsA} * kLanes * sizeof(i16));
    l.wb = take(std::size_t{kSetsB} * kLanes * sizeof(i16));
    l.wc = take(std::size_t{kSetsC} * kLanes * sizeof(i16));
    l.wf = take(std::size_t{kSetsF} * 4 * sizeof(i16));
    l.dbuckets = take(Dirty::words(bucket_lines(s)) * sizeof(u64));
    l.dmatch = take(Dirty::words(match_lines(s)) * sizeof(u64));
    l.dapm = take(Dirty::words(65536) * sizeof(u64));
    l.total = at;
    return l;
  }

  void carve(std::byte* base, const LevelSpec& s, u64 window_bytes, const Kernels& kern) noexcept {
    const Layout l = layout(s);
    buckets = reinterpret_cast<Bucket*>(base + l.buckets);
    match_tab = reinterpret_cast<u32*>(base + l.match);
    o1 = reinterpret_cast<u16*>(base + l.o1);
    o0 = reinterpret_cast<u16*>(base + l.o0);
    flags = reinterpret_cast<u32*>(base + l.flags);
    msm = reinterpret_cast<u32*>(base + l.msm);
    apm1 = reinterpret_cast<u16*>(base + l.apm1);
    apm2 = reinterpret_cast<u16*>(base + l.apm2);
    wa = reinterpret_cast<i16*>(base + l.wa);
    wb = reinterpret_cast<i16*>(base + l.wb);
    wc = reinterpret_cast<i16*>(base + l.wc);
    wf = reinterpret_cast<i16*>(base + l.wf);
    dirty_buckets = {reinterpret_cast<u64*>(base + l.dbuckets), bucket_lines(s)};
    dirty_match = {reinterpret_cast<u64*>(base + l.dmatch), match_lines(s)};
    dirty_apm = {reinterpret_cast<u64*>(base + l.dapm), 65536};
    for (std::size_t j = 0; j < kApmRow; ++j)
      apm_row[j] = static_cast<u16>(std::clamp(squash((static_cast<int>(j) - 16) * 128), 0, 65535));
    k = &kern;
    spec = s;
    window = window_bytes;
    bucket_mask = (u32{1} << s.bucket_bits) - 1;
    match_shift = 64 - s.match_bits;
  }

  // R20: every table back to its initial image. `full` after binding, when the
  // arena holds garbage; otherwise only what the last frame wrote is cleared.
  void reset(bool full) noexcept {
    const std::size_t bucket_bytes = (std::size_t{kHashed} << spec.bucket_bits) * sizeof(Bucket);
    const std::size_t match_bytes = (std::size_t{1} << spec.match_bits) * sizeof(u32);
    const auto all_buckets = [this, bucket_bytes] { std::memset(buckets, 0, bucket_bytes); };
    const auto all_match = [this, match_bytes] { std::memset(match_tab, 0, match_bytes); };
    const auto apm_one = [this](std::size_t r) { std::memcpy(apm2 + r * kApmRow, apm_row, sizeof apm_row); };
    const auto apm_all = [&apm_one] {
      for (std::size_t r = 0; r < 65536; ++r) apm_one(r);
    };
    if (full) {
      all_buckets();
      all_match();
      apm_all();
      dirty_buckets.forget();
      dirty_match.forget();
      dirty_apm.forget();
    } else {
      dirty_buckets.drain([this](std::size_t line) { std::memset(buckets + 2 * line, 0, 64); }, all_buckets);
      dirty_match.drain([this](std::size_t line) { std::memset(match_tab + 16 * line, 0, 64); }, all_match);
      dirty_apm.drain(apm_one, apm_all);
    }
    stale = false;
    std::memset(o1, 0, std::size_t{65536} * sizeof(u16));
    std::memset(o0, 0, std::size_t{256} * sizeof(u16));
    std::memset(flags, 0, std::size_t{16 * 256} * sizeof(u32));
    std::memset(msm, 0, std::size_t{64} * sizeof(u32));
    for (std::size_t r = 0; r < 256; ++r) std::memcpy(apm1 + r * kApmRow, apm_row, sizeof apm_row);
    const auto init = [](i16* w, int sets, int lanes, int used, i16 v) {
      for (int s = 0; s < sets; ++s)
        for (int i = 0; i < lanes; ++i) w[s * lanes + i] = i < used ? v : i16{0};
    };
    init(wa, kSetsA, kLanes, kInputs - 1, kInitW1);  // the bias lane starts at 0
    init(wb, kSetsB, kLanes, kInputs - 1, kInitW1);
    init(wc, kSetsC, kLanes, kInputs - 1, kInitW1);
    init(wf, kSetsF, 4, 3, kInitW2);
    pos = c8 = 0;
    word0 = word1 = field = 0;
    line_start = prev_line_start = 0;
    mptr = 0;
    mlen = 0;
    expected = -1;
    c0 = nib = 1;
    bitpos = 0;
  }

  // ------------------------------------------------------- contexts ---
  template <class H>
  static u32 word_hash(const H& h, u64 from, u64 to) noexcept {
    u32 w = 0;
    for (u64 q = from; q < to; ++q) w = (w + fold(h.at(q)) + 1) * kWordMul;
    return w;
  }

  // Start of the line holding position end - 1: the byte after the last '\n'
  // in [lo, end), or 0 when the scan reaches the frame start. False: unknown.
  template <class H>
  static bool find_line_start(const H& h, u64 end, u64 lo, u64& start) noexcept {
    u64 q = end;
    while (q > lo && h.at(q - 1) != '\n') --q;
    if (q > lo || lo == 0) {
      start = q;
      return true;
    }
    return false;
  }

  void compute_contexts_base() noexcept {
    const u64 c1 = c8 & 0xFF;
    ctx[0] = hash(c8 & 0xFFFF, 1);
    ctx[1] = hash(c8 & 0xFFFFFF, 2);
    ctx[2] = hash(c8 & 0xFFFFFFFF, 3);
    ctx[3] = hash(c8 & 0xFFFFFFFFFFFF, 4);
    ctx[4] = word0 ? hash(word0, 5) : hash((u64{word1} << 8) | c1, 6);
    ctx[6] = hash((c8 >> 8) & 0xFFFF, 8);
    ctx[7] = hash((u64{std::min<u32>(field, 255)} << 16) | (c8 & 0xFFFF), 10);
  }

  template <class H>
  void compute_contexts(const H& h, u64 reach) noexcept {
    compute_contexts_base();
    // Column: the byte above in the previous line, when that is in the window.
    int above = 0;
    if (prev_line_start < line_start) {
      const u64 ap = prev_line_start + (pos - line_start);
      if (ap < line_start && pos - ap <= reach) above = h.at(ap) + 1;
    }
    ctx[5] = hash((static_cast<u64>(above) << 8) | (c8 & 0xFF), 7);
    for (int i = 0; i < kHashed; ++i) {  // the first nibble's buckets
      const std::size_t region = static_cast<std::size_t>(i) << spec.bucket_bits;
      __builtin_prefetch(buckets + region + (static_cast<u32>(ctx[i]) & bucket_mask));
    }
  }

  // Re-derives the history state at a block start from at most a few KiB of
  // history: identical on both sides whether the previous block was CM or STORED.
  template <class H>
  void begin_block(const H& h, u64 at) noexcept {
    pos = at;
    const u64 reach = std::min(window, pos);
    c8 = 0;
    for (u64 i = 1; i <= std::min<u64>(8, reach); ++i) c8 |= static_cast<u64>(h.at(pos - i)) << (8 * (i - 1));

    const u64 wlo = pos - std::min(kWordScan, reach);
    u64 j = pos;
    while (j > wlo && is_word(h.at(j - 1))) --j;
    word0 = word_hash(h, j, pos);
    u64 e = j;
    while (e > wlo && !is_word(h.at(e - 1))) --e;
    u64 s = e;
    while (s > wlo && is_word(h.at(s - 1))) --s;
    word1 = word_hash(h, s, e);

    const u64 llo = pos - std::min(kLineScan, reach);
    if (!find_line_start(h, pos, llo, line_start)) {
      line_start = prev_line_start = llo;  // column unknown: counted from the scan limit
    } else if (line_start == 0 || !find_line_start(h, line_start - 1, llo, prev_line_start)) {
      prev_line_start = line_start;  // no previous line
    }

    field = 0;
    for (u64 q = line_start; q < pos; ++q) field += is_sep(h.at(q));
    mlen = 0;  // matches never cross a block boundary
    expected = -1;
    compute_contexts(h, reach);
    c0 = nib = 1;
    bitpos = 0;
  }

  // ------------------------------------------------------- matching ---
  template <class H>
  void find_match(const H& h, u32 cand, u64 reach) noexcept {
    const u32 d = static_cast<u32>(pos) - cand;  // R21: mod 2^32, then range-checked
    if (d == 0 || d > reach) return;
    u32 len = 0;
    while (len < kMaxVerify && u64{d} + len + 1 <= reach && h.at(pos - 1 - len) == h.at(pos - 1 - len - d))
      ++len;
    if (len >= kMinMatch) {
      mlen = len;
      mptr = pos - d;
    }
  }

  template <class H>
  void byte_update(const H& h, int c) noexcept {
    if (mlen) {
      if (c == expected) {
        mlen += mlen < kMaxMatch;
        ++mptr;
      } else {
        mlen = 0;
      }
    }
    ++pos;
    c8 = (c8 << 8) | static_cast<u64>(c);
    if (is_word(c)) {
      word0 = (word0 + fold(c) + 1) * kWordMul;
    } else if (word0) {
      word1 = word0;
      word0 = 0;
    }
    if (c == '\n') {
      prev_line_start = line_start;
      line_start = pos;
      field = 0;
    } else {
      field += is_sep(c);
    }
    const u64 reach = std::min(window, pos);
    if (pos >= kMinMatch) {
      const std::size_t mi = static_cast<std::size_t>(hash(c8 & 0xFFFFFFFFFFFF, 9) >> match_shift);
      if (!mlen) find_match(h, match_tab[mi], reach);
      match_tab[mi] = static_cast<u32>(pos);
      dirty_match.mark(mi >> 4);
    }
    expected = mlen ? h.at(mptr) : -1;
    compute_contexts(h, reach);
    c0 = nib = 1;
    bitpos = 0;
  }

  // ---------------------------------------------------- bit modelling ---
  // At bits 0 and 4: the bucket of every hashed expert for this nibble.
  // Two-way associative; a miss evicts the less used way (count of its first
  // counter), ties evict way 0: deterministic replacement (R21).
  void lookup_nibble() noexcept {
    for (int i = 0; i < kHashed; ++i) {
      const u64 hv = nibble_hash(ctx[i], c0, bitpos);
      Bucket* const base = buckets + (static_cast<std::size_t>(i) << spec.bucket_bits);
      const u32 idx = static_cast<u32>(hv) & bucket_mask;
      const u16 tag = static_cast<u16>(hv >> 48);
      Bucket* const b0 = base + idx;
      Bucket* const b1 = base + (idx ^ 1u);
      Bucket* b;
      if (b0->s[0] == tag) {
        b = b0;
      } else if (b1->s[0] == tag) {
        b = b1;
      } else {
        b = (b1->s[1] & 15) < (b0->s[1] & 15) ? b1 : b0;
        *b = Bucket{{tag}};
      }
      cur[i] = b;
      dirty_buckets.mark(static_cast<std::size_t>(b - buckets) >> 1);
    }
  }

  // One bit in three steps. gather() reads the counters into the mixer's
  // inputs; mix_code() mixes, codes and trains the mixers and APMs; learn()
  // updates the counters with the coded bit. The first and last touch only
  // the counter tables, the middle one only the mixer and APM tables, so the
  // encoder can run them on two threads (ncm_pipe.hpp) and code the same bits.
  // The per-bit state lives in locals, which stores through the table pointers
  // cannot alias, so it stays in registers. K: the mixer kernels (ncm_simd.hpp).
  struct Gathered {
    u16* node[kNodes];
    u16 nv[kNodes];
    int ebit = -1, msi = 0;
  };

  // x: zeroed by the caller. Returns the match state, which selects wb's set.
  // kRaw: lanes [0, kNodes) get the counters themselves, for the pipeline's
  // second stage to stretch (stretch_nodes).
  template <bool kRaw = false>
  int gather(Gathered& g, simd::Lanes& x) const noexcept {
    const u32 c1 = static_cast<u32>(c8 & 0xFF);
    for (int i = 0; i < kHashed; ++i) g.node[i] = &cur[i]->s[nib];
    g.node[kHashed] = &o1[(c1 << 8) | c0];
    g.node[kHashed + 1] = &o0[c0];
    for (int i = 0; i < kNodes; ++i) {
      g.nv[i] = *g.node[i];
      x[i] = static_cast<i16>(kRaw ? g.nv[i] : node_stretch(g.nv[i]));
    }
    int ms = 0;
    if (expected >= 0 && static_cast<u32>((expected | 256) >> (8 - bitpos)) == c0) {
      g.ebit = (expected >> (7 - bitpos)) & 1;
      ms = len_bucket(mlen);
      g.msi = ms * 2 + g.ebit;
      const int st = stretch16(std::min(counter_p(msm[g.msi]), 65535));
      const int l = static_cast<int>(std::min<u32>(mlen, 32)) * 32;
      x[kNodes] = static_cast<i16>(g.ebit ? st : -st);
      x[kNodes + 1] = static_cast<i16>(g.ebit ? l : -l);
    }
    x[kNodes + 2] = kBias;
    return ms;
  }

  void learn(const Gathered& g, int y) noexcept {
    const u16* const next = kNodeNext.data() + (static_cast<std::size_t>(y) << 16);
    for (int i = 0; i < kNodes; ++i) *g.node[i] = next[g.nv[i]];
    if (g.ebit >= 0) counter_update(msm[g.msi], y == g.ebit);
    c0 = c0 * 2 + static_cast<u32>(y);
    nib = nib * 2 + static_cast<u32>(y);
    if (++bitpos == 4) {
      nib = 1;
      lookup_nibble();
    }
  }

  static void stretch_nodes(simd::Lanes& x) noexcept {
    for (int i = 0; i < kNodes; ++i) x[i] = static_cast<i16>(node_stretch(static_cast<u16>(x[i])));
  }

  // The tables mix_code() reads and writes, and nothing else of the model.
  struct MixTables {
    i16 *wa, *wb, *wc, *wf;
    u16 *apm1, *apm2;
    Dirty dirty_apm;
  };
  MixTables mix_tables() const noexcept { return {wa, wb, wc, wf, apm1, apm2, dirty_apm}; }

  template <class K, class IO>
  static int mix_code(const MixTables& mt, IO& io, const simd::Lanes& x, u32 c0, u32 c1, int ms, int bitpos,
                      int actual) noexcept {
    i16* const w1[kSets] = {mt.wa + c0 * kLanes, mt.wb + static_cast<u32>(ms) * kLanes, mt.wc + c1 * kLanes};
    i32 dots[kSets];
    K::mix(x, w1, dots);
    alignas(8) i16 t[4]{};  // layer 2's inputs; the fourth weight of every set stays 0
    static_assert(kSets == 3);
    for (int j = 0; j < kSets; ++j)
      t[j] = static_cast<i16>(std::clamp(dots[j] >> 14, -kStretchMax, kStretchMax));
    i16* const f = mt.wf + bitpos * 4;
    const i32 d2 = K::dot4(t, f);
    const int pm = std::max(squash16(std::clamp(d2 >> 14, -kLogitMax, kLogitMax)), 1);

    const int s = stretch16(pm) + 2048;  // 1..4095
    const int alo = s >> 7, awt = s & 127;
    u16* const a1 = mt.apm1 + c0 * kApmRow;
    const std::size_t arow = (c1 << 8) | c0;
    u16* const a2 = mt.apm2 + arow * kApmRow;
    const int q1 = (a1[alo] * (128 - awt) + a1[alo + 1] * awt) >> 7;
    const int q2 = (a2[alo] * (128 - awt) + a2[alo + 1] * awt) >> 7;
    const int y = io.bit(std::clamp((pm + q1 + 2 * q2 + 2) >> 2, kPMin, kPMax), actual);

    i32 errs[kSets];
    for (int j = 0; j < kSets; ++j) errs[j] = (((y << 16) - squash16(t[j])) * kLr1) >> 10;
    alignas(32) i16 xa[kLanes];
    std::memcpy(xa, &x, sizeof xa);
    K::train(xa, w1, errs);
    K::train4(t, f, (((y << 16) - pm) * kLr2) >> 10);

    const int target = y ? 65535 : 0;
    Dirty da = mt.dirty_apm;
    da.mark(arow);
    for (u16* a : {a1, a2}) {
      const int v0 = a[alo], v1 = a[alo + 1];
      a[alo] = static_cast<u16>(v0 + (((target - v0) * (128 - awt)) >> (7 + kApmRate)));
      a[alo + 1] = static_cast<u16>(v1 + (((target - v1) * awt) >> (7 + kApmRate)));
    }
    return y;
  }

  template <class K, class IO>
  int code_bit(IO& io, int actual) noexcept {
    Gathered g;
    simd::Lanes x{};
    const int ms = gather(g, x);
    const int y = mix_code<K>(mix_tables(), io, x, c0, static_cast<u32>(c8 & 0xFF), ms, bitpos, actual);
    learn(g, y);
    return y;
  }

  // One byte. `actual` is the true byte when known (encode, replay).
  template <class K, class IO>
  int code_byte(IO& io, int actual) noexcept {
    if (expected >= 0 && mlen >= spec.fast_len) {
      const u32 fb = std::min<u32>(15, static_cast<u32>(std::bit_width(mlen - spec.fast_len)));
      u32& f = flags[(fb << 8) | static_cast<u32>(expected)];
      const int hit = io.bit(std::clamp(counter_p(f), kPMin, kPMax), actual == expected);
      counter_update(f, hit);
      if (hit) return expected;
      mlen = 0;  // the byte is coded in full, without the failed prediction
      expected = -1;
    }
    lookup_nibble();
    for (int i = 7; i >= 0; --i) code_bit<K>(io, (actual >> i) & 1);
    return static_cast<int>(c0 & 0xFF);
  }

  // The encoder's first pipeline stage: code_byte() without the mixing. For
  // every coded decision it hands sink.put() the record the second stage
  // needs (ncm_pipe.hpp); false when the sink was closed.
  template <class Sink>
  bool produce_byte(Sink& sink, int c) noexcept {
    if (expected >= 0 && mlen >= spec.fast_len) {
      const u32 fb = std::min<u32>(15, static_cast<u32>(std::bit_width(mlen - spec.fast_len)));
      u32& f = flags[(fb << 8) | static_cast<u32>(expected)];
      const int hit = c == expected;
      simd::Lanes* const r = sink.slot();
      if (!r) return false;
      *r = simd::Lanes{};
      (*r)[0] = static_cast<i16>(std::clamp(counter_p(f), kPMin, kPMax));
      (*r)[kLanes - 1] = static_cast<i16>(kRecFast | (hit << 3));
      sink.commit();
      counter_update(f, hit);
      if (hit) return true;
      mlen = 0;
      expected = -1;
    }
    for (int i = 0; i < kHashed; ++i) {  // the byte is known: its second nibble's buckets
      const Bucket* const region = buckets + (static_cast<std::size_t>(i) << spec.bucket_bits);
      __builtin_prefetch(region + (static_cast<u32>(nibble_hash(ctx[i], static_cast<u32>(c | 256) >> 4, 4)) & bucket_mask));
    }
    lookup_nibble();
    for (int i = 7; i >= 0; --i) {
      const int y = (c >> i) & 1;
      simd::Lanes* const r = sink.slot();
      if (!r) return false;
      *r = simd::Lanes{};
      Gathered g;
      const int ms = gather<true>(g, *r);
      (*r)[kLanes - 3] = static_cast<i16>(ms);
      (*r)[kLanes - 2] = static_cast<i16>(((c8 & 0xFF) << 8) | c0);
      (*r)[kLanes - 1] = static_cast<i16>(bitpos | (y << 3));
      sink.commit();
      learn(g, y);
    }
    return true;
  }
};

}  // namespace hce::ncm
