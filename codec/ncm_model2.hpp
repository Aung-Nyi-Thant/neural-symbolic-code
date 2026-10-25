// ncm_model2.hpp - the model of codec profile 2 (prior_id 0x4E434D02), the
// neural CM codec's higher-ratio profile. Profile 1 (ncm_model.hpp) is frozen
// by its golden streams (R17), so profile 2 is its own model; it shares the
// helpers, the counters' layout, the coder and the SIMD kernels.
//
// What profile 2 adds, all of it bounded by the level's arena (R21):
//  - seven more hashed experts: orders 5 and 8, word pairs, two indirect
//    models (the bytes that followed the last occurrences of the order-1 and
//    order-2 contexts) and two structure models for JSON, XML and source code
//    (the enclosing bracket or element, and the current key or attribute);
//  - a second, long-range match model that indexes 16-byte contexts and
//    verifies 16 bytes before it predicts;
//  - two inputs per expert, the logit and the probability itself, which
//    weighs confident contexts better; 48 mixer inputs in three groups of
//    sixteen, each group one call of the 16-lane kernels (R15 holds per
//    group, and the groups' sums are added in 64 bits);
//  - weight sets selected by both match models, a third APM over a hashed
//    order-2 context, and faster-adapting counters (kNodeNext2).
#pragma once

#include "codec/ncm_model.hpp"

#include <tuple>

namespace hce::ncm::p2 {

// ------------------------------------------------ format constants (R17) ---
// Changing anything in this block changes profile 2's coded bits.
enum Expert : int {
  kO2, kO3, kO4, kO6, kWord, kColumn, kSparse, kField,  // as in profile 1
  kO5, kO8, kWordPair, kIndirect2, kIndirect1, kNest, kKey,
  kHashed
};
constexpr int kNodes = kHashed + 2;  // plus the direct o1 and o0 tables
constexpr int kGroups = 3;           // mixer inputs: kGroups x kLanes
constexpr int kWidth = kGroups * kLanes;
// Lanes: [0, 16) the logits of nodes 0..15, [16, 32) their probabilities,
// then the last node's logit and probability, the two match models' inputs
// and the bias. The rest are zero.
constexpr int kLastNode = 2 * kLanes;
constexpr int kMatch1 = kLastNode + 2, kMatch2 = kMatch1 + 2, kBiasLane = kMatch2 + 2;
static_assert(kNodes == kLanes + 1 && kBiasLane < kWidth);
constexpr int kSetsA = 256;     // layer-1 weight sets selected by the partial byte
constexpr int kSetsB = 32 * 3;  // ... by match 1's length and whether match 2 agrees
constexpr int kSetsC = 256;     // ... by the previous byte
constexpr int kSetsF = 8 * 3;   // layer-2 weight sets: bit position x active match models
constexpr int kLr1 = 36, kLr2 = 16;
static_assert(65536 * kLr1 >> 10 <= 32767 && 65536 * kLr2 >> 10 <= 32767);
constexpr i16 kInitW1 = 2048, kInitW2 = 5461, kBias = 256;  // 48 inputs: start at 0.125
constexpr u32 kMinMatch2 = 16, kMaxVerify2 = 64;
constexpr int kNestDepth = 32;  // the structure models' stack, a ring: deeper levels overwrite
constexpr std::size_t kApm3Rows = 65536;

// Table size of each hashed expert relative to the level's, log2: the
// experts that gain most from room get twice the buckets, those that gain
// least half (measured on text, source code, logs, JSON and XML).
constexpr std::array<int, kHashed> kSizeShift{
    // o2 o3 o4 o6 word column sparse field o5 o8 pair ind2 ind1 nest key
    0, 0, 1, 0, 1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1};

struct LevelSpec2 {
  u8 bucket_bits;  // log2 of the buckets of an expert of size shift 0
  u8 match_bits;   // log2 of the entries of each match table
  u32 fast_len;    // match length from which a byte is coded as one decision
};
constexpr std::array<LevelSpec2, kNeuralCMMaxLevel + 1> kLevels2{{
    {15, 18, 32},
    {16, 19, 64},
    {16, 20, 128},
    {17, 21, 256},
    {18, 22, 1024},
}};
constexpr const LevelSpec2& spec2_for(u8 level) noexcept { return kLevels2[std::min<u8>(level, kNeuralCMMaxLevel)]; }

// The structure models' state: a bracket and element stack and the last key,
// tracked byte by byte over JSON, XML, HTML and C-like source alike. Model
// state, reset at every frame; like every model, it sees CM-coded bytes only.
struct Structure {
  u32 stack[kNestDepth];
  u32 depth;
  u32 word;   // hash of the word being read, 0 outside one
  u32 tok;    // hash of the last completed word or string
  u32 key;    // tok as of the last ':' or '=': the current key or attribute
  u32 str;    // hash of the string being read
  u32 tagh;   // hash of the element name being read
  u8 in_str;  // inside a "string"
  u8 esc;     // the previous byte of the string was a backslash
  u8 tag;     // element syntax: one of the Tag values
  u8 slash;   // the previous byte inside a tag was '/'

  // Element syntax. A '<' that is not followed by a name, or a name that is
  // not followed by whitespace, '>' or '/', is no tag (a comparison, say); so
  // is a "tag" interrupted by a bracket or ';', and its push is undone.
  enum Tag : u8 { kNone, kOpen, kName, kCloseName, kAttrs, kCloseAttrs };

  u32 top() const noexcept { return depth ? stack[(depth - 1) % kNestDepth] : 0; }
  void push(u32 h) noexcept { stack[depth++ % kNestDepth] = h; }
  void pop() noexcept { depth -= depth > 0; }
  void open_string() noexcept {
    in_str = 1;
    str = 0;
  }

  void update(int c) noexcept {
    if (in_str) {
      if (esc) {
        esc = 0;
      } else if (c == '\\') {
        esc = 1;
      } else if (c == '"' || c == '\n') {
        in_str = 0;
        tok = str;
        return;
      }
      str = (str + static_cast<u32>(c) + 1) * kWordMul;
      return;
    }
    const bool wc = is_word(c) || (c >= '0' && c <= '9') || c == '_';
    if (wc) {
      word = (word + fold(c) + 1) * kWordMul;
    } else if (word) {
      tok = word;
      word = 0;
    }
    const bool space = c == ' ' || c == '\t' || c == '\r' || c == '\n';
    switch (tag) {
      case kOpen:
        if (c == '/') {
          tag = kCloseName;
          tagh = 0;
          return;
        }
        if (is_word(c)) {
          tag = kName;
          tagh = (fold(c) + 1) * kWordMul;
          return;
        }
        tag = kNone;
        break;
      case kName:
      case kCloseName:
        if (wc || c == ':' || c == '-' || c == '.') {
          tagh = (tagh + fold(c) + 1) * kWordMul;
          return;
        }
        if (!space && c != '>' && c != '/') {
          tag = kNone;
          break;
        }
        if (tag == kName) {
          push(tagh ^ 0x5A5A5A5Au);
          tag = kAttrs;
        } else {
          pop();
          tag = kCloseAttrs;
        }
        slash = 0;
        break;
      default: break;
    }
    if (tag == kAttrs || tag == kCloseAttrs) {
      switch (c) {
        case '>':
          if (slash && tag == kAttrs) pop();
          tag = kNone;
          return;
        case ';': case '{': case '}': case '(': case ')':
          if (tag == kAttrs) pop();
          tag = kNone;
          break;  // and the byte counts as code
        default:
          slash = c == '/';
          if (c == '=') key = tok;
          if (c == '"') open_string();
          return;
      }
    }
    switch (c) {
      case '"': open_string(); break;
      case ':': case '=': key = tok; break;
      case ',': case ';': key = 0; break;
      case '{': case '[': case '(':
        push(static_cast<u32>(hash((u64{tok} << 8) | static_cast<u32>(c), 20)));
        key = 0;
        break;
      case '}': case ']': case ')': pop(); break;
      case '<': tag = kOpen; break;
      default: break;
    }
  }
};

// ----------------------------------------------------------------- model ---
// Placement-constructed at the start of the arena the codec is bound to.
struct CMModel2 {
  // Tables, carved from the arena.
  Bucket* buckets = nullptr;  // kHashed regions, expert i's of 2^(bucket_bits + kSizeShift[i]) buckets
  u32* match_tab = nullptr;   // match 1: 6-byte contexts -> frame-relative positions (R21)
  u32* match2_tab = nullptr;  // match 2: 16-byte contexts
  u16* ind2 = nullptr;        // [order-2 context]: the two bytes that followed it last
  u16* o1 = nullptr;
  u16* o0 = nullptr;
  u32* flags = nullptr;
  u32* msm = nullptr;   // match 1: [length bucket][expected bit]
  u32* msm2 = nullptr;  // match 2
  u16* apm1 = nullptr;  // [partial byte][33]
  u16* apm2 = nullptr;  // [previous byte][partial byte][33]
  u16* apm3 = nullptr;  // [hashed order 2][partial byte][33]
  i16* wa = nullptr;    // layer-1 weights, kWidth per set
  i16* wb = nullptr;
  i16* wc = nullptr;
  i16* wf = nullptr;  // layer-2 weights, 4 per set
  Dirty dirty_buckets;  // 64-byte lines: two buckets each
  Dirty dirty_match;    // 64-byte lines of both match tables: sixteen entries each
  Dirty dirty_ind;      // 64-byte lines of ind2
  Dirty dirty_apm;      // APM2 rows, then APM3 rows
  u16 apm_row[kApmRow]{};
  bool stale = false;
  const Kernels* k = nullptr;
  LevelSpec2 spec{};
  u64 window = 0;
  u32 region[kHashed]{};       // first bucket of each expert's region
  u32 bucket_mask[kHashed]{};  // its buckets - 1
  int match_shift = 0;

  // Model state beyond the tables, reset at every frame.
  u16 ind1[256]{};  // [order-1 context]: the two bytes that followed it last
  Structure st{};

  // History state: re-derived from the history at every block start.
  u64 pos = 0;
  u64 c8 = 0;   // the last eight bytes
  u64 c16 = 0;  // the eight before them
  u32 word0 = 0, word1 = 0;
  u32 field = 0;
  u64 line_start = 0, prev_line_start = 0;
  u64 mptr = 0, mptr2 = 0;
  u32 mlen = 0, mlen2 = 0;
  int expected = -1, expected2 = -1;
  u64 ctx[kHashed]{};

  // Per-bit state.
  u32 c0 = 1;
  u32 nib = 1;
  int bitpos = 0;
  Bucket* cur[kHashed]{};

  // ---------------------------------------------------------- layout ---
  struct Layout {
    std::size_t buckets, match, match2, ind2, o1, o0, flags, msm, msm2, apm1, apm2, apm3, wa, wb, wc, wf, dbuckets,
        dmatch, dind, dapm, total;
  };
  static constexpr std::size_t region_buckets(const LevelSpec2& s, int i) noexcept {
    return std::size_t{1} << (s.bucket_bits + kSizeShift[static_cast<std::size_t>(i)]);
  }
  static constexpr std::size_t total_buckets(const LevelSpec2& s) noexcept {
    std::size_t n = 0;
    for (int i = 0; i < kHashed; ++i) n += region_buckets(s, i);
    return n;
  }
  static constexpr std::size_t bucket_lines(const LevelSpec2& s) noexcept { return total_buckets(s) / 2; }
  static constexpr std::size_t match_entries(const LevelSpec2& s) noexcept { return std::size_t{1} << s.match_bits; }
  static constexpr std::size_t match_lines(const LevelSpec2& s) noexcept { return 2 * match_entries(s) / 16; }
  static constexpr std::size_t kIndLines = 65536 * sizeof(u16) / 64;
  static constexpr std::size_t kApmUnits = 65536 + kApm3Rows;
  static constexpr std::size_t align64(std::size_t n) noexcept { return (n + 63) & ~std::size_t{63}; }
  static constexpr Layout layout(const LevelSpec2& s) noexcept {
    Layout l{};
    std::size_t at = align64(sizeof(CMModel2));
    const auto take = [&at](std::size_t bytes) {
      const std::size_t here = at;
      at = align64(at + bytes);
      return here;
    };
    l.buckets = take(total_buckets(s) * sizeof(Bucket));
    l.match = take(2 * match_entries(s) * sizeof(u32));  // match 2's table follows match 1's
    l.match2 = l.match + match_entries(s) * sizeof(u32);
    l.ind2 = take(std::size_t{65536} * sizeof(u16));
    l.o1 = take(std::size_t{65536} * sizeof(u16));
    l.o0 = take(std::size_t{256} * sizeof(u16));
    l.flags = take(std::size_t{16 * 256} * sizeof(u32));
    l.msm = take(std::size_t{64} * sizeof(u32));
    l.msm2 = take(std::size_t{64} * sizeof(u32));
    l.apm1 = take(std::size_t{256} * kApmRow * sizeof(u16));
    l.apm2 = take(std::size_t{65536} * kApmRow * sizeof(u16));
    l.apm3 = take(kApm3Rows * kApmRow * sizeof(u16));
    l.wa = take(std::size_t{kSetsA} * kWidth * sizeof(i16));
    l.wb = take(std::size_t{kSetsB} * kWidth * sizeof(i16));
    l.wc = take(std::size_t{kSetsC} * kWidth * sizeof(i16));
    l.wf = take(std::size_t{kSetsF} * 4 * sizeof(i16));
    l.dbuckets = take(Dirty::words(bucket_lines(s)) * sizeof(u64));
    l.dmatch = take(Dirty::words(match_lines(s)) * sizeof(u64));
    l.dind = take(Dirty::words(kIndLines) * sizeof(u64));
    l.dapm = take(Dirty::words(kApmUnits) * sizeof(u64));
    l.total = at;
    return l;
  }

  void carve(std::byte* base, const LevelSpec2& s, u64 window_bytes, const Kernels& kern) noexcept {
    const Layout l = layout(s);
    buckets = reinterpret_cast<Bucket*>(base + l.buckets);
    match_tab = reinterpret_cast<u32*>(base + l.match);
    match2_tab = reinterpret_cast<u32*>(base + l.match2);
    ind2 = reinterpret_cast<u16*>(base + l.ind2);
    o1 = reinterpret_cast<u16*>(base + l.o1);
    o0 = reinterpret_cast<u16*>(base + l.o0);
    flags = reinterpret_cast<u32*>(base + l.flags);
    msm = reinterpret_cast<u32*>(base + l.msm);
    msm2 = reinterpret_cast<u32*>(base + l.msm2);
    apm1 = reinterpret_cast<u16*>(base + l.apm1);
    apm2 = reinterpret_cast<u16*>(base + l.apm2);
    apm3 = reinterpret_cast<u16*>(base + l.apm3);
    wa = reinterpret_cast<i16*>(base + l.wa);
    wb = reinterpret_cast<i16*>(base + l.wb);
    wc = reinterpret_cast<i16*>(base + l.wc);
    wf = reinterpret_cast<i16*>(base + l.wf);
    dirty_buckets = {reinterpret_cast<u64*>(base + l.dbuckets), bucket_lines(s)};
    dirty_match = {reinterpret_cast<u64*>(base + l.dmatch), match_lines(s)};
    dirty_ind = {reinterpret_cast<u64*>(base + l.dind), kIndLines};
    dirty_apm = {reinterpret_cast<u64*>(base + l.dapm), kApmUnits};
    for (std::size_t j = 0; j < kApmRow; ++j)
      apm_row[j] = static_cast<u16>(std::clamp(squash((static_cast<int>(j) - 16) * 128), 0, 65535));
    k = &kern;
    spec = s;
    window = window_bytes;
    u32 at = 0;
    for (int i = 0; i < kHashed; ++i) {
      region[i] = at;
      bucket_mask[i] = static_cast<u32>(region_buckets(s, i) - 1);
      at += static_cast<u32>(region_buckets(s, i));
    }
    match_shift = 64 - s.match_bits;
  }

  // R20: every table back to its initial image; after binding (`full`) all
  // of it, otherwise only what the last frame wrote (R19).
  void reset(bool full) noexcept {
    const std::size_t bucket_bytes = total_buckets(spec) * sizeof(Bucket);
    const std::size_t match_bytes = 2 * match_entries(spec) * sizeof(u32);
    const auto all_buckets = [this, bucket_bytes] { std::memset(buckets, 0, bucket_bytes); };
    const auto all_match = [this, match_bytes] { std::memset(match_tab, 0, match_bytes); };
    const auto all_ind = [this] { std::memset(ind2, 0, std::size_t{65536} * sizeof(u16)); };
    const auto apm_one = [this](std::size_t r) {
      u16* const row = r < 65536 ? apm2 + r * kApmRow : apm3 + (r - 65536) * kApmRow;
      std::memcpy(row, apm_row, sizeof apm_row);
    };
    const auto apm_all = [&apm_one] {
      for (std::size_t r = 0; r < kApmUnits; ++r) apm_one(r);
    };
    if (full) {
      all_buckets();
      all_match();
      all_ind();
      apm_all();
      dirty_buckets.forget();
      dirty_match.forget();
      dirty_ind.forget();
      dirty_apm.forget();
    } else {
      dirty_buckets.drain([this](std::size_t line) { std::memset(buckets + 2 * line, 0, 64); }, all_buckets);
      dirty_match.drain([this](std::size_t line) { std::memset(match_tab + 16 * line, 0, 64); }, all_match);
      dirty_ind.drain([this](std::size_t line) { std::memset(ind2 + 32 * line, 0, 64); }, all_ind);
      dirty_apm.drain(apm_one, apm_all);
    }
    stale = false;
    std::memset(o1, 0, std::size_t{65536} * sizeof(u16));
    std::memset(o0, 0, std::size_t{256} * sizeof(u16));
    std::memset(flags, 0, std::size_t{16 * 256} * sizeof(u32));
    std::memset(msm, 0, std::size_t{64} * sizeof(u32));
    std::memset(msm2, 0, std::size_t{64} * sizeof(u32));
    for (std::size_t r = 0; r < 256; ++r) std::memcpy(apm1 + r * kApmRow, apm_row, sizeof apm_row);
    // Logits and match inputs start at 0.25; probabilities and the bias at 0.
    std::array<i16, kWidth> w0{};
    for (int i = 0; i < kLanes; ++i) w0[static_cast<std::size_t>(i)] = kInitW1;
    for (const int i : {kLastNode, kMatch1, kMatch1 + 1, kMatch2}) w0[static_cast<std::size_t>(i)] = kInitW1;
    const auto init = [&w0](i16* w, int sets) {
      for (int s = 0; s < sets; ++s) std::memcpy(w + s * kWidth, w0.data(), sizeof w0);
    };
    init(wa, kSetsA);
    init(wb, kSetsB);
    init(wc, kSetsC);
    for (int s = 0; s < kSetsF; ++s)
      for (int i = 0; i < 4; ++i) wf[s * 4 + i] = i < 3 ? kInitW2 : i16{0};
    std::memset(ind1, 0, sizeof ind1);
    st = Structure{};
    pos = c8 = c16 = 0;
    word0 = word1 = field = 0;
    line_start = prev_line_start = 0;
    mptr = mptr2 = 0;
    mlen = mlen2 = 0;
    expected = expected2 = -1;
    c0 = nib = 1;
    bitpos = 0;
  }

  // ------------------------------------------------------- contexts ---
  template <class H>
  void compute_contexts(const H& h, u64 reach) noexcept {
    const u64 c1 = c8 & 0xFF, c2 = c8 & 0xFFFF;
    ctx[kO2] = hash(c2, 1);
    ctx[kO3] = hash(c8 & 0xFFFFFF, 2);
    ctx[kO4] = hash(c8 & 0xFFFFFFFF, 3);
    ctx[kO6] = hash(c8 & 0xFFFFFFFFFFFF, 4);
    ctx[kWord] = word0 ? hash(word0, 5) : hash((u64{word1} << 8) | c1, 6);
    ctx[kSparse] = hash((c8 >> 8) & 0xFFFF, 8);
    ctx[kField] = hash((u64{std::min<u32>(field, 255)} << 16) | c2, 10);
    ctx[kO5] = hash(c8 & 0xFFFFFFFFFF, 11);
    ctx[kO8] = hash(c8, 15);
    ctx[kWordPair] = hash((u64{word0} << 32) | word1, 12);
    ctx[kIndirect2] = hash(c2 | (u64{ind2[c2]} << 16) | (u64{ind1[c1]} << 32), 13);
    ctx[kIndirect1] = hash(c1 | (u64{ind1[c1]} << 8), 14);
    ctx[kNest] = hash((u64{st.top()} << 32) | (u64{std::min<u32>(st.depth, 63)} << 16) | (u64{st.in_str} << 8) | c1, 16);
    ctx[kKey] = hash((u64{st.key} << 32) | (u64{st.tag} << 24) | c2, 17);
    int above = 0;
    if (prev_line_start < line_start) {
      const u64 ap = prev_line_start + (pos - line_start);
      if (ap < line_start && pos - ap <= reach) above = h.at(ap) + 1;
    }
    ctx[kColumn] = hash((static_cast<u64>(above) << 8) | c1, 7);
    for (int i = 0; i < kHashed; ++i) {
      __builtin_prefetch(buckets + region[i] + (static_cast<u32>(ctx[i]) & bucket_mask[i]));
    }
  }

  template <class H>
  void begin_block(const H& h, u64 at) noexcept {
    pos = at;
    const u64 reach = std::min(window, pos);
    c8 = c16 = 0;
    for (u64 i = 1; i <= std::min<u64>(8, reach); ++i) c8 |= static_cast<u64>(h.at(pos - i)) << (8 * (i - 1));
    for (u64 i = 9; i <= std::min<u64>(16, reach); ++i) c16 |= static_cast<u64>(h.at(pos - i)) << (8 * (i - 9));

    const u64 wlo = pos - std::min(kWordScan, reach);
    u64 j = pos;
    while (j > wlo && is_word(h.at(j - 1))) --j;
    word0 = CMModel::word_hash(h, j, pos);
    u64 e = j;
    while (e > wlo && !is_word(h.at(e - 1))) --e;
    u64 s = e;
    while (s > wlo && is_word(h.at(s - 1))) --s;
    word1 = CMModel::word_hash(h, s, e);

    const u64 llo = pos - std::min(kLineScan, reach);
    if (!CMModel::find_line_start(h, pos, llo, line_start)) {
      line_start = prev_line_start = llo;
    } else if (line_start == 0 || !CMModel::find_line_start(h, line_start - 1, llo, prev_line_start)) {
      prev_line_start = line_start;
    }

    field = 0;
    for (u64 q = line_start; q < pos; ++q) field += is_sep(h.at(q));
    mlen = mlen2 = 0;  // matches never cross a block boundary
    expected = expected2 = -1;
    compute_contexts(h, reach);
    c0 = nib = 1;
    bitpos = 0;
  }

  // ------------------------------------------------------- matching ---
  // The candidate at `cand`, verified over up to `verify` bytes: its length
  // when that reaches `min_len`, else 0 (R21: distances range-checked).
  template <class H>
  u32 verify(const H& h, u32 cand, u64 reach, u32 min_len, u32 max_verify, u64& ptr) const noexcept {
    const u32 d = static_cast<u32>(pos) - cand;
    if (d == 0 || d > reach) return 0;
    u32 len = 0;
    while (len < max_verify && u64{d} + len + 1 <= reach && h.at(pos - 1 - len) == h.at(pos - 1 - len - d)) ++len;
    if (len < min_len) return 0;
    ptr = pos - d;
    return len;
  }

  template <class H>
  void byte_update(const H& h, int c) noexcept {
    for (auto [len, ptr, exp] : {std::tie(mlen, mptr, expected), std::tie(mlen2, mptr2, expected2)}) {
      if (!len) continue;
      if (c == exp) {
        len += len < kMaxMatch;
        ++ptr;
      } else {
        len = 0;
      }
    }
    u16& i1 = ind1[c8 & 0xFF];
    i1 = static_cast<u16>((u32{i1} << 8) | static_cast<u32>(c));
    const std::size_t i2 = c8 & 0xFFFF;
    ind2[i2] = static_cast<u16>((u32{ind2[i2]} << 8) | static_cast<u32>(c));
    dirty_ind.mark(i2 >> 5);
    st.update(c);
    ++pos;
    c16 = (c16 << 8) | (c8 >> 56);
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
      if (!mlen) mlen = verify(h, match_tab[mi], reach, kMinMatch, kMaxVerify, mptr);
      match_tab[mi] = static_cast<u32>(pos);
      dirty_match.mark(mi >> 4);
    }
    if (pos >= kMinMatch2) {
      const std::size_t mi = static_cast<std::size_t>(hash(c8 ^ hash(c16, 18), 19) >> match_shift);
      if (!mlen2) mlen2 = verify(h, match2_tab[mi], reach, kMinMatch2, kMaxVerify2, mptr2);
      match2_tab[mi] = static_cast<u32>(pos);
      dirty_match.mark((match_entries(spec) + mi) >> 4);
    }
    expected = mlen ? h.at(mptr) : -1;
    expected2 = mlen2 ? h.at(mptr2) : -1;
    compute_contexts(h, reach);
    c0 = nib = 1;
    bitpos = 0;
  }

  // ---------------------------------------------------- bit modelling ---
  // As in profile 1: two-way associative buckets, deterministic replacement.
  static constexpr u64 nibble_hash(u64 hv, u32 c) noexcept {
    hv = (hv + (u64{c} + 1) * 0x9E3779B97F4A7C15ull);
    hv = (hv ^ (hv >> 29)) * 0xBF58476D1CE4E5B9ull;
    return hv ^ (hv >> 32);
  }
  // After three bits: both buckets the second nibble may use, so its lookup
  // finds them in cache. Not part of the format: it only touches memory.
  void prefetch_second_nibble() const noexcept {
    for (int i = 0; i < kHashed; ++i)
      for (const u32 c : {c0 * 2, c0 * 2 + 1})
        __builtin_prefetch(buckets + region[i] + (static_cast<u32>(nibble_hash(ctx[i], c)) & bucket_mask[i]));
  }

  void lookup_nibble() noexcept {
    for (int i = 0; i < kHashed; ++i) {
      const u64 hv = bitpos ? nibble_hash(ctx[i], c0) : ctx[i];
      Bucket* const base = buckets + region[i];
      const u32 idx = static_cast<u32>(hv) & bucket_mask[i];
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

  // A match model's inputs for this bit: whether it predicts one, its index
  // into the decision counters, and the signed logit.
  struct MatchBit {
    int ebit = -1, si = 0;
    i16 st = 0, len = 0;
  };
  MatchBit match_bit(int exp, u32 len, const u32* sm) const noexcept {
    MatchBit m;
    if (exp < 0 || static_cast<u32>((exp | 256) >> (8 - bitpos)) != c0) return m;
    m.ebit = (exp >> (7 - bitpos)) & 1;
    m.si = len_bucket(len) * 2 + m.ebit;
    const int s = stretch16(std::min(counter_p(sm[m.si]), 65535));
    const int l = static_cast<int>(std::min<u32>(len, 32)) * 32;
    m.st = static_cast<i16>(m.ebit ? s : -s);
    m.len = static_cast<i16>(m.ebit ? l : -l);
    return m;
  }

  template <class K, class IO>
  int code_bit(IO& io, int actual) noexcept {
    const u32 c1 = static_cast<u32>(c8 & 0xFF);
    u16* node[kNodes];
    for (int i = 0; i < kHashed; ++i) node[i] = &cur[i]->s[nib];
    node[kHashed] = &o1[(c1 << 8) | c0];
    node[kHashed + 1] = &o0[c0];
    u16 nv[kNodes];
    simd::Lanes xl{}, xp{}, xr{};  // the three groups: logits, probabilities, the rest
    for (int i = 0; i < kNodes; ++i) nv[i] = *node[i];
    // A probability input: p - 1/2, Q12 over two, so |input| <= 1024; for
    // the first sixteen nodes in one vector operation.
    const auto lin = [](u16 v) { return static_cast<i16>((static_cast<int>((v >> 4) ^ 0x800) - 2048) >> 1); };
    typedef u16 U16x16 __attribute__((vector_size(2 * kLanes)));
    U16x16 nvv;
    std::memcpy(&nvv, nv, sizeof nvv);
    xp = (reinterpret_cast<simd::Lanes>((nvv >> 4) ^ 0x800) - 2048) >> 1;
    for (int i = 0; i < kLanes; ++i) xl[i] = static_cast<i16>(node_stretch(nv[i]));
    xr[kLastNode - 2 * kLanes] = static_cast<i16>(node_stretch(nv[kLanes]));
    xr[kLastNode - 2 * kLanes + 1] = lin(nv[kLanes]);

    const MatchBit m1 = match_bit(expected, mlen, msm);
    const MatchBit m2 = match_bit(expected2, mlen2, msm2);
    xr[kMatch1 - 2 * kLanes] = m1.st;
    xr[kMatch1 - 2 * kLanes + 1] = m1.len;
    xr[kMatch2 - 2 * kLanes] = m2.st;
    xr[kMatch2 - 2 * kLanes + 1] = m2.len;
    xr[kBiasLane - 2 * kLanes] = kBias;

    // Match 2: absent, agreeing with match 1 (or alone), or disagreeing.
    const int m2s = m2.ebit < 0 ? 0 : (m1.ebit < 0 || m2.ebit == m1.ebit) ? 1 : 2;
    const u32 ms = m1.ebit < 0 ? 0u : static_cast<u32>(len_bucket(mlen));
    i16* const w1[kSets] = {wa + c0 * kWidth, wb + (ms * 3 + static_cast<u32>(m2s)) * kWidth, wc + c1 * kWidth};
    const simd::Lanes* const xg[kGroups] = {&xl, &xp, &xr};
    i64 acc[kSets]{};
    for (int g = 0; g < kGroups; ++g) {
      i16* const wg[kSets] = {w1[0] + g * kLanes, w1[1] + g * kLanes, w1[2] + g * kLanes};
      i32 dots[kSets];
      K::mix(*xg[g], wg, dots);
      for (int j = 0; j < kSets; ++j) acc[j] += dots[j];
    }
    alignas(8) i16 t[4]{};
    static_assert(kSets == 3);
    for (int j = 0; j < kSets; ++j)
      t[j] = static_cast<i16>(std::clamp<i64>(acc[j] >> 14, -kStretchMax, kStretchMax));
    i16* const f = wf + (bitpos * 3 + (m1.ebit >= 0) + (m2.ebit >= 0)) * 4;
    const i32 d2 = K::dot4(t, f);
    const int pm = std::max(squash16(std::clamp(d2 >> 14, -kLogitMax, kLogitMax)), 1);

    const int s = stretch16(pm) + 2048;
    const int alo = s >> 7, awt = s & 127;
    u16* const a1 = apm1 + c0 * kApmRow;
    const std::size_t arow = (c1 << 8) | c0;
    u16* const a2 = apm2 + arow * kApmRow;
    const std::size_t arow3 = ((static_cast<std::size_t>(hash(c8 & 0xFFFF, 21)) & 0xFF) << 8) | c0;
    u16* const a3 = apm3 + arow3 * kApmRow;
    const auto interp = [alo, awt](const u16* a) { return (a[alo] * (128 - awt) + a[alo + 1] * awt) >> 7; };
    const int p = (2 * pm + 2 * interp(a1) + 3 * interp(a2) + interp(a3) + 4) >> 3;
    const int y = io.bit(std::clamp(p, kPMin, kPMax), actual);

    const u16* const next = kNodeNext2.data() + (static_cast<std::size_t>(y) << 16);
    for (int i = 0; i < kNodes; ++i) *node[i] = next[nv[i]];
    if (m1.ebit >= 0) counter_update(msm[m1.si], y == m1.ebit);
    if (m2.ebit >= 0) counter_update(msm2[m2.si], y == m2.ebit);

    i32 errs[kSets];
    for (int j = 0; j < kSets; ++j) errs[j] = (((y << 16) - squash16(t[j])) * kLr1) >> 10;
    for (int g = 0; g < kGroups; ++g) {
      i16* const wg[kSets] = {w1[0] + g * kLanes, w1[1] + g * kLanes, w1[2] + g * kLanes};
      alignas(32) i16 xa[kLanes];
      std::memcpy(xa, xg[g], sizeof xa);
      K::train(xa, wg, errs);
    }
    K::train4(t, f, (((y << 16) - pm) * kLr2) >> 10);

    const int target = y ? 65535 : 0;
    dirty_apm.mark(arow);
    dirty_apm.mark(65536 + arow3);
    for (u16* a : {a1, a2, a3}) {
      const int v0 = a[alo], v1 = a[alo + 1];
      a[alo] = static_cast<u16>(v0 + (((target - v0) * (128 - awt)) >> (7 + kApmRate)));
      a[alo + 1] = static_cast<u16>(v1 + (((target - v1) * awt) >> (7 + kApmRate)));
    }

    c0 = c0 * 2 + static_cast<u32>(y);
    nib = nib * 2 + static_cast<u32>(y);
    if (++bitpos == 4) {
      nib = 1;
      lookup_nibble();
    } else if (bitpos == 3) {
      prefetch_second_nibble();
    }
    return y;
  }

  template <class K, class IO>
  int code_byte(IO& io, int actual) noexcept {
    if (expected >= 0 && mlen >= spec.fast_len) {
      const u32 fb = std::min<u32>(15, static_cast<u32>(std::bit_width(mlen - spec.fast_len)));
      u32& f = flags[(fb << 8) | static_cast<u32>(expected)];
      const int hit = io.bit(std::clamp(counter_p(f), kPMin, kPMax), actual == expected);
      counter_update(f, hit);
      if (hit) return expected;
      mlen = 0;
      expected = -1;
    }
    lookup_nibble();
    for (int i = 7; i >= 0; --i) code_bit<K>(io, (actual >> i) & 1);
    return static_cast<int>(c0 & 0xFF);
  }
};

}  // namespace hce::ncm::p2
