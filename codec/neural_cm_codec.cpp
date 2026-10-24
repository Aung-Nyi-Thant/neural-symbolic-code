// neural_cm_codec.cpp - the neural-symbolic context-mixing codec (see the header).
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
//
// R12: STORED blocks reach the history (contexts are re-derived from it at
// every block start) but never the models. A block that would expand is
// stored, and the encoder rebuilds its models by replaying the frame's CM
// blocks; a pre-check stores obviously incompressible blocks without trying.
#include "codec/neural_cm_codec.hpp"

#include "codec/ncm_kernels.hpp"
#include "codec/ncm_tables.hpp"
#include "encoder/parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <new>

namespace hce {
namespace {

using hc::u16;
using hc::u32;
using hc::u64;
using hc::u8;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// ------------------------------------------------ format constants (R17) ---
// Changing anything in this block changes the coded bits: that is a new profile.
constexpr int kHashed = 8;           // hashed experts: o2, o3, o4, o6, word, column, sparse, field
constexpr int kNodes = kHashed + 2;  // plus the direct o1 and o0 tables
constexpr int kInputs = kNodes + 3;  // plus two match inputs and the bias
static_assert(kInputs <= ncm::kLanes);
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

// ------------------------------------------------- encoder-side policy ---
// Not part of the format: the decoder never sees these decisions.
constexpr unsigned kMaxReplays = 3;     // per frame; afterwards the rest of the frame is STORED
constexpr std::size_t kMinCmBlock = 16;  // smaller blocks are STORED
constexpr u64 kStoreEntropyQ16 = 511181;  // 7.8 bits/byte, Q16
constexpr u64 kEntropyBiasQ16 = u64{184} << 16;  // small-sample bias of the empirical entropy

std::atomic<bool> g_force_cm{false};
std::atomic<const ncm::Kernels*> g_kernels{nullptr};

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

// Bit-history counter: 12-bit probability and a 4-bit count, packed so that
// zeroed memory is the initial state (p = 0.5, n = 0). Model::update moves p
// toward the bit by kRate[n] and counts up to 15.
constexpr int node_p(u16 n) noexcept { return ((n >> 4) ^ 0x800); }

// Decision counter: a 22-bit probability over a 10-bit count, so that long
// runs can reach P(miss) ~ 1e-4; zero is p = 0.75, n = 0.
constexpr u32 kCounterInit = 0x300000;  // 0.75, Q22
constexpr int counter_p(u32 c) noexcept { return static_cast<int>(((c >> 10) ^ kCounterInit) >> 6); }  // Q16
inline void counter_update(u32& c, int y) noexcept {
  i64 p = (c >> 10) ^ kCounterInit;
  const u32 n = c & 1023;
  p += (((y ? 0x3FFFFF : 0) - p) * ncm::kRate[n] + (1 << 15)) >> 16;
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

// --------------------------------------------------- arithmetic coder ---
// 32-bit carryless binary coder. P(1) is Q16 in [1, 65535]; xmid is exact
// (64-bit product), so every platform splits the range identically.
class ArithEncoder {
 public:
  ArithEncoder(std::byte* out, std::size_t cap) noexcept : out_(out), cap_(cap) {}
  void encode(int bit, int p16) noexcept {
    const u32 xmid = x1_ + static_cast<u32>((u64{x2_ - x1_} * static_cast<u32>(p16)) >> 16);
    if (bit) x2_ = xmid;
    else x1_ = xmid + 1;
    while (((x1_ ^ x2_) & 0xFF000000u) == 0) {
      put(x2_ >> 24);
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
    }
  }
  // Four bytes of x1: the decoder then reads exactly the payload, no more.
  void flush() noexcept {
    for (int i = 0; i < 4; ++i, x1_ <<= 8) put(x1_ >> 24);
  }
  bool full() const noexcept { return full_; }
  std::size_t size() const noexcept { return n_; }

 private:
  void put(u32 b) noexcept {
    if (n_ < cap_) out_[n_++] = static_cast<std::byte>(b);
    else full_ = true;
  }
  std::byte* out_;
  std::size_t cap_, n_ = 0;
  u32 x1_ = 0, x2_ = 0xFFFFFFFFu;
  bool full_ = false;
};

class ArithDecoder {
 public:
  explicit ArithDecoder(hc::Bytes in) noexcept : in_(in) {
    for (int i = 0; i < 4; ++i) x_ = (x_ << 8) | next();
  }
  int decode(int p16) noexcept {
    const u32 xmid = x1_ + static_cast<u32>((u64{x2_ - x1_} * static_cast<u32>(p16)) >> 16);
    const int bit = x_ <= xmid;
    if (bit) x2_ = xmid;
    else x1_ = xmid + 1;
    while (((x1_ ^ x2_) & 0xFF000000u) == 0) {
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
      x_ = (x_ << 8) | next();
    }
    return bit;
  }
  bool overrun() const noexcept { return overrun_; }
  // The whole payload was consumed and nothing beyond it was needed.
  bool exact() const noexcept { return !overrun_ && pos_ == in_.size(); }

 private:
  u32 next() noexcept {
    if (pos_ < in_.size()) return static_cast<u32>(in_[pos_++]);
    overrun_ = true;
    return 0;
  }
  hc::Bytes in_;
  std::size_t pos_ = 0;
  u32 x1_ = 0, x2_ = 0xFFFFFFFFu, x_ = 0;
  bool overrun_ = false;
};

// Where each coded decision comes from. `y` is the true bit where it is known.
struct EncodeIO {
  ArithEncoder& c;
  int bit(int p, int y) noexcept {
    c.encode(y, p);
    return y;
  }
};
struct DecodeIO {
  ArithDecoder& c;
  int bit(int p, int) noexcept { return c.decode(p); }
};
struct ReplayIO {
  int bit(int, int y) noexcept { return y; }
};

// History as seen by the models: frame-relative positions. The models only
// read positions within min(window, position) of the current one (R21), which
// is exactly what the decoder's ring window holds.
struct FrameHistory {
  const std::byte* data;
  int at(u64 i) const noexcept { return static_cast<int>(data[i]); }
};
struct WindowHistory {
  const hc::Window* w;
  u64 base;  // absolute window position of the frame start
  int at(u64 i) const noexcept { return static_cast<int>(w->at(base + i)); }
};

// Units of a table (cache lines, APM rows) written since the last reset. A
// frame reset clears only those, so it costs time in proportion to the work
// done in the frame, never to the table size: a stream of tiny frames cannot
// multiply the decoder's work (R19). The tables end up identical either way.
struct Dirty {
  u64* bits = nullptr;
  std::size_t units = 0;
  std::size_t marked = 0;

  static constexpr std::size_t words(std::size_t n) noexcept { return (n + 63) / 64; }
  void mark(std::size_t u) noexcept {
    u64& w = bits[u >> 6];
    const u64 b = u64{1} << (u & 63);
    marked += (w & b) == 0;
    w |= b;
  }
  void forget() noexcept {
    std::memset(bits, 0, words(units) * sizeof(u64));
    marked = 0;
  }
  // clear_one(u) for every marked unit, or clear_all() when that is cheaper.
  template <class One, class All>
  void drain(One&& clear_one, All&& clear_all) noexcept {
    if (marked > units / 8) {
      clear_all();
    } else {
      for (std::size_t i = 0; i < words(units); ++i)
        for (u64 w = bits[i]; w; w &= w - 1) clear_one(i * 64 + static_cast<std::size_t>(std::countr_zero(w)));
    }
    forget();
  }
};

}  // namespace

// ----------------------------------------------------------------- model ---
struct NeuralCMCodec::Model {
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
  const ncm::Kernels* k = nullptr;
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
  int ebit = -1;  // expected bit while the match input is active
  int msi = 0;
  Bucket* cur[kHashed]{};
  u16* node[kNodes]{};
  int np[kNodes]{};  // the nodes' probabilities and counts, read once per bit
  int nn[kNodes]{};
  alignas(32) i16 x[ncm::kLanes]{};
  i16* w1[ncm::kSets]{};
  i32 dots[ncm::kSets]{};
  i32 errs[ncm::kSets]{};
  int t[ncm::kSets]{};
  int pm = 32768;
  u16* a1 = nullptr;
  u16* a2 = nullptr;
  std::size_t arow = 0;
  int alo = 0, awt = 0;

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
    std::size_t at = align64(sizeof(Model));
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
    l.wa = take(std::size_t{kSetsA} * ncm::kLanes * sizeof(i16));
    l.wb = take(std::size_t{kSetsB} * ncm::kLanes * sizeof(i16));
    l.wc = take(std::size_t{kSetsC} * ncm::kLanes * sizeof(i16));
    l.wf = take(std::size_t{kSetsF} * 4 * sizeof(i16));
    l.dbuckets = take(Dirty::words(bucket_lines(s)) * sizeof(u64));
    l.dmatch = take(Dirty::words(match_lines(s)) * sizeof(u64));
    l.dapm = take(Dirty::words(65536) * sizeof(u64));
    l.total = at;
    return l;
  }

  void carve(std::byte* base, const LevelSpec& s, u64 window_bytes, const ncm::Kernels& kern) noexcept {
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
    dirty_buckets = {reinterpret_cast<u64*>(base + l.dbuckets), bucket_lines(s), 0};
    dirty_match = {reinterpret_cast<u64*>(base + l.dmatch), match_lines(s), 0};
    dirty_apm = {reinterpret_cast<u64*>(base + l.dapm), 65536, 0};
    for (std::size_t j = 0; j < kApmRow; ++j)
      apm_row[j] = static_cast<u16>(std::clamp(ncm::squash((static_cast<int>(j) - 16) * 128), 0, 65535));
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
    init(wa, kSetsA, ncm::kLanes, kInputs - 1, kInitW1);  // the bias lane starts at 0
    init(wb, kSetsB, ncm::kLanes, kInputs - 1, kInitW1);
    init(wc, kSetsC, ncm::kLanes, kInputs - 1, kInitW1);
    init(wf, kSetsF, 4, 3, kInitW2);
    std::fill(std::begin(x), std::end(x), i16{0});
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
      u64 hv = ctx[i];
      if (bitpos) {
        hv = (hv + (u64{c0} + 1) * 0x9E3779B97F4A7C15ull);
        hv = (hv ^ (hv >> 29)) * 0xBF58476D1CE4E5B9ull;
        hv ^= hv >> 32;
      }
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

  int predict() noexcept {
    const u32 c1 = static_cast<u32>(c8 & 0xFF);
    for (int i = 0; i < kHashed; ++i) node[i] = &cur[i]->s[nib];
    node[kHashed] = &o1[(c1 << 8) | c0];
    node[kHashed + 1] = &o0[c0];
    for (int i = 0; i < kNodes; ++i) {
      const u16 v = *node[i];
      np[i] = node_p(v);
      nn[i] = v & 15;
      x[i] = static_cast<i16>(ncm::stretch12(np[i]));
    }

    int ms = 0;
    ebit = -1;
    if (expected >= 0 && static_cast<u32>((expected | 256) >> (8 - bitpos)) == c0) {
      ebit = (expected >> (7 - bitpos)) & 1;
      ms = len_bucket(mlen);
      msi = ms * 2 + ebit;
      const int st = ncm::stretch16(std::min(counter_p(msm[msi]), 65535));
      const int l = static_cast<int>(std::min<u32>(mlen, 32)) * 32;
      x[kNodes] = static_cast<i16>(ebit ? st : -st);
      x[kNodes + 1] = static_cast<i16>(ebit ? l : -l);
    } else {
      x[kNodes] = x[kNodes + 1] = 0;
    }
    x[kNodes + 2] = kBias;

    w1[0] = wa + c0 * ncm::kLanes;
    w1[1] = wb + static_cast<u32>(ms) * ncm::kLanes;
    w1[2] = wc + c1 * ncm::kLanes;
    k->mix(x, w1, dots);
    for (int j = 0; j < ncm::kSets; ++j) t[j] = std::clamp(dots[j] >> 14, -ncm::kStretchMax, ncm::kStretchMax);
    const i16* f = wf + bitpos * 4;
    static_assert(ncm::kSets == 3);
    const i32 d2 = t[0] * f[0] + t[1] * f[1] + t[2] * f[2];
    pm = std::max(ncm::squash16(std::clamp(d2 >> 14, -ncm::kLogitMax, ncm::kLogitMax)), 1);

    const int s = ncm::stretch16(pm) + 2048;  // 1..4095
    alo = s >> 7;
    awt = s & 127;
    a1 = apm1 + c0 * kApmRow;
    arow = (c1 << 8) | c0;
    a2 = apm2 + arow * kApmRow;
    const int q1 = (a1[alo] * (128 - awt) + a1[alo + 1] * awt) >> 7;
    const int q2 = (a2[alo] * (128 - awt) + a2[alo + 1] * awt) >> 7;
    return std::clamp((pm + q1 + 2 * q2 + 2) >> 2, kPMin, kPMax);
  }

  void update(int y) noexcept {
    const int ty = (y << 12) - y;
    for (int i = 0; i < kNodes; ++i) {
      const int pn = np[i] + (((ty - np[i]) * ncm::kRate[static_cast<std::size_t>(nn[i])] + (1 << 15)) >> 16);
      *node[i] = static_cast<u16>(((pn ^ 0x800) << 4) | (nn[i] + (nn[i] < 15)));
    }
    if (ebit >= 0) counter_update(msm[msi], y == ebit);

    for (int j = 0; j < ncm::kSets; ++j) errs[j] = (((y << 16) - ncm::squash16(t[j])) * kLr1) >> 10;
    k->train(x, w1, errs);
    i16* f = wf + bitpos * 4;
    const int e2 = (((y << 16) - pm) * kLr2) >> 10;
    for (int j = 0; j < ncm::kSets; ++j)
      f[j] = static_cast<i16>(std::clamp(f[j] + ((((t[j] * 16) * e2 >> 16) + 1) >> 1), -32768, 32767));

    const int target = y ? 65535 : 0;
    dirty_apm.mark(arow);
    for (u16* a : {a1, a2}) {
      const int v0 = a[alo], v1 = a[alo + 1];
      a[alo] = static_cast<u16>(v0 + (((target - v0) * (128 - awt)) >> (7 + kApmRate)));
      a[alo + 1] = static_cast<u16>(v1 + (((target - v1) * awt) >> (7 + kApmRate)));
    }

    c0 = c0 * 2 + static_cast<u32>(y);
    nib = nib * 2 + static_cast<u32>(y);
    if (++bitpos == 4) {
      nib = 1;
      lookup_nibble();
    }
  }

  // One byte. `actual` is the true byte when known (encode, replay).
  template <class IO>
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
    for (int i = 7; i >= 0; --i) update(io.bit(predict(), (actual >> i) & 1));
    return static_cast<int>(c0 & 0xFF);
  }
};

// ------------------------------------------------------------- the codec ---
NeuralCMCodec::NeuralCMCodec() noexcept = default;

NeuralCMCodec::NeuralCMCodec(const hc::StreamParams& p) {
  if (accepts(p) != hc::Errc::ok) return;  // no model: every block is STORED
  const u64 n = memory_required(p);
  owned_.reset(new std::byte[static_cast<std::size_t>(n)]);
  (void)bind(p, hc::MutBytes(owned_.get(), static_cast<std::size_t>(n)));
}

NeuralCMCodec::NeuralCMCodec(NeuralCMCodec&& other) noexcept
    : m_(std::exchange(other.m_, nullptr)),
      owned_(std::move(other.owned_)),
      frame_(other.frame_),
      cm_blocks_(std::move(other.cm_blocks_)),
      replays_(other.replays_),
      store_rest_(other.store_rest_),
      counters_(other.counters_) {}

NeuralCMCodec& NeuralCMCodec::operator=(NeuralCMCodec&& other) noexcept {
  if (this != &other) {
    m_ = std::exchange(other.m_, nullptr);
    owned_ = std::move(other.owned_);
    frame_ = other.frame_;
    cm_blocks_ = std::move(other.cm_blocks_);
    replays_ = other.replays_;
    store_rest_ = other.store_rest_;
    counters_ = other.counters_;
  }
  return *this;
}

NeuralCMCodec::~NeuralCMCodec() = default;  // Model is trivially destructible

hc::Errc NeuralCMCodec::accepts(const hc::StreamParams& p) noexcept {
  if ((p.prior_id & 0xFFFFFF00u) != kNeuralCMFamily || (p.prior_id & 0xFFu) != kNeuralCMProfile)
    return hc::Errc::unsupported_feature;
  if (p.level > kNeuralCMMaxLevel) return hc::Errc::unsupported_feature;
  return hc::Errc::ok;
}

// Depends on the level only: the decoder pipeline sizes workers before it has
// seen a stream header. Levels above the maximum are rejected by accepts().
hc::u64 NeuralCMCodec::memory_required(const hc::StreamParams& p) noexcept {
  return Model::layout(spec_for(p.level)).total + 63;  // + slack to align the arena to 64 bytes
}

hc::Errc NeuralCMCodec::bind(const hc::StreamParams& p, hc::MutBytes arena) noexcept {
  m_ = nullptr;
  if (const hc::Errc e = accepts(p); e != hc::Errc::ok) return e;
  const LevelSpec& s = spec_for(p.level);
  if (arena.size() < memory_required(p)) return hc::Errc::out_of_memory;
  const auto addr = reinterpret_cast<std::uintptr_t>(arena.data());
  std::byte* const base = arena.data() + ((64 - (addr & 63)) & 63);
  const ncm::Kernels* forced = g_kernels.load(std::memory_order_relaxed);
  Model* const m = new (base) Model{};
  m->carve(base, s, p.window_size(), forced ? *forced : ncm::active_kernels());
  m_ = m;
  m_->reset(true);
  return hc::Errc::ok;
}

// Lazy: the reset happens at the frame's first CM block, so frames without one
// (STORED data, damaged frames, headers alone) cost nothing.
void NeuralCMCodec::reset_frame() noexcept {
  if (m_) m_->stale = true;
}

hc::Errc NeuralCMCodec::decode_block(hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept {
  if (!m_) return hc::Errc::payload_corrupt;
  Model& m = *m_;
  if (m.stale) m.reset(false);
  ArithDecoder dec(payload);
  DecodeIO io{dec};
  const WindowHistory h{&w, w.pos() - w.frame_offset()};
  m.begin_block(h, w.frame_offset());
  for (u32 i = 0; i < usize; ++i) {
    const int c = m.code_byte(io, 0);
    if (dec.overrun()) return hc::Errc::payload_corrupt;  // ran past the payload: corrupt
    w.put(static_cast<std::byte>(c));
    m.byte_update(h, c);
  }
  return dec.exact() ? hc::Errc::ok : hc::Errc::payload_corrupt;
}

void NeuralCMCodec::reset_frame(hc::Bytes frame) noexcept {
  frame_ = frame;
  cm_blocks_.clear();
  replays_ = 0;
  store_rest_ = false;
  if (m_) m_->reset(false);  // now: the pre-check reads the match table
}

// Encoder-side guess, never seen by the decoder: is this block hopeless for
// CM? High order-0 entropy and almost no repeats the match model could use.
static bool looks_incompressible(hc::Bytes frame, std::size_t begin, std::size_t end, const u32* match_tab,
                                 int match_shift, u64 window) noexcept {
  const hc::Bytes b = frame.subspan(begin, end - begin);
  std::array<u32, 256> count{};
  for (const std::byte c : b) ++count[static_cast<u8>(c)];
  const u64 n = b.size();
  u64 sum = 0;
  for (const u32 c : count)
    if (c) sum += u64{c} * log2_q16(c);
  const u64 bits = n * log2_q16(n) - sum;
  if (bits + kEntropyBiasQ16 < n * kStoreEntropyQ16) return false;

  // Sampled every 64 bytes: does the 8-byte string ending here occur earlier in
  // this block (local table) or in earlier CM-coded data (the match table)?
  std::array<u32, 4096> local{};
  u64 samples = 0, repeats = 0;
  u64 v = 0;  // the last eight bytes, most recent lowest: the models' c8
  for (std::size_t i = begin; i < std::min(end, begin + 8); ++i) v = (v << 8) | static_cast<u8>(frame[i]);
  for (std::size_t i = begin + 8; i < end; v = (v << 8) | static_cast<u8>(frame[i]), ++i) {
    const u32 slot = static_cast<u32>((v * 0x9E3779B97F4A7C15ull) >> 52);
    if ((i & 63) == 0) {
      ++samples;
      bool hit = false;
      if (const u32 prev = local[slot]; prev && std::memcmp(frame.data() + prev - 8, frame.data() + i - 8, 8) == 0)
        hit = true;
      const u32 cand = match_tab[hash(v & 0xFFFFFFFFFFFF, 9) >> match_shift];  // as in byte_update
      const u64 d = i - cand;
      if (!hit && cand < i && d <= std::min<u64>(window, i) && cand >= 8 &&
          std::memcmp(frame.data() + cand - 8, frame.data() + i - 8, 8) == 0)
        hit = true;
      repeats += hit;
    }
    local[slot] = static_cast<u32>(i);
  }
  return repeats * 16 < samples;
}

std::size_t NeuralCMCodec::encode_block(std::size_t begin, std::size_t end, hc::MutBytes out) noexcept {
  const std::size_t n = end - begin;
  if (!m_ || store_rest_ || n < kMinCmBlock || out.size() < n) {
    ++counters_.stored_blocks;
    return 0;
  }
  Model& m = *m_;
  if (!g_force_cm.load(std::memory_order_relaxed) &&
      looks_incompressible(frame_, begin, end, m.match_tab, m.match_shift, m.window)) {
    ++counters_.stored_blocks;
    return 0;
  }
  if (cm_blocks_.size() == cm_blocks_.capacity()) {
    try {
      cm_blocks_.reserve(std::max<std::size_t>(64, cm_blocks_.capacity() * 2));
    } catch (...) {  // cannot record it for a replay: code nothing more with CM
      store_rest_ = true;
      ++counters_.stored_blocks;
      return 0;
    }
  }

  ArithEncoder enc(out.data(), n - 1);  // a CM payload must be strictly smaller
  EncodeIO io{enc};
  const FrameHistory h{frame_.data()};
  m.begin_block(h, begin);
  for (std::size_t i = begin; i < end && !enc.full(); ++i) {
    const int c = static_cast<int>(frame_[i]);
    m.code_byte(io, c);
    m.byte_update(h, c);
  }
  if (!enc.full()) enc.flush();
  if (enc.full()) {  // expands: store it, and rebuild the models without it (R12)
    replay();
    ++counters_.replays;
    if (++replays_ >= kMaxReplays) store_rest_ = true;
    ++counters_.stored_blocks;
    return 0;
  }
  cm_blocks_.emplace_back(begin, end);
  ++counters_.cm_blocks;
  return enc.size();
}

// Rebuilds the models exactly as the decoder will hold them: reset, then the
// frame's CM blocks again, updating without coding.
void NeuralCMCodec::replay() noexcept {
  Model& m = *m_;
  m.reset(false);
  ReplayIO io;
  const FrameHistory h{frame_.data()};
  for (const auto& [b, e] : cm_blocks_) {
    m.begin_block(h, b);
    for (std::size_t i = b; i < e; ++i) {
      const int c = static_cast<int>(frame_[i]);
      m.code_byte(io, c);
      m.byte_update(h, c);
    }
  }
}

const char* NeuralCMCodec::simd_path() noexcept {
  const ncm::Kernels* forced = g_kernels.load(std::memory_order_relaxed);
  return (forced ? *forced : ncm::active_kernels()).name;
}

void NeuralCMCodec::test_force_cm(bool on) noexcept { g_force_cm.store(on, std::memory_order_relaxed); }

bool NeuralCMCodec::test_select_simd(const char* name) noexcept {
  if (!name) {
    g_kernels.store(nullptr, std::memory_order_relaxed);
    return true;
  }
  for (const ncm::SimdPath p : {ncm::SimdPath::scalar, ncm::SimdPath::sse2, ncm::SimdPath::avx2, ncm::SimdPath::neon}) {
    if (std::strcmp(ncm::kernels_for(p).name, name) == 0 && ncm::simd_available(p)) {
      g_kernels.store(&ncm::kernels_for(p), std::memory_order_relaxed);
      return true;
    }
  }
  return false;
}

static_assert(hc::BlockCodec<NeuralCMCodec>);
static_assert(FrameEncoder<NeuralCMCodec>);

}  // namespace hce
