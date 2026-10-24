// structural_cm_codec.cpp - token-level context mixing with a frozen global model (see the header).
//
// Encoding, decoding and the encoder's replay run one templated code path
// that differs only in where each decision's bit comes from, so they cannot
// disagree. As in NeuralCMCodec (R12), STORED blocks reach the history
// (contexts are re-derived from it at every block start) but never the
// models, and a block that would expand is stored and the models rebuilt by
// replaying the frame's CM blocks.
#include "structural/structural_cm_codec.hpp"

#include "codec/ncm_kernels.hpp"
#include "codec/ncm_tables.hpp"
#include "encoder/parallel.hpp"
#include "structural/global_model.hpp"

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
using structural::GlobalModel;
using structural::kNoSymbol;
using structural::TokenType;

// ------------------------------------------------ format constants (R17) ---
// Changing anything in this block changes the coded bits: that is a new profile.
constexpr int kCtx = 8;           // counter experts per decision
constexpr int kInputs = kCtx + 1;  // plus the bias
static_assert(kInputs <= ncm::kLanes);
constexpr int kNodeLocal = 0;      // decision kinds; each has its own weight sets
constexpr int kNodeGlobal = 1;
constexpr int kNodeLength = 2;     // 2..32: the 31 inner nodes of the 5-bit length tree
constexpr int kNodeMantissa = 64;  // 64..127: by length and bit index
constexpr int kNodes = 128;
constexpr int kLengthBits = 5;
constexpr int kSetsA = kNodes;  // layer-1 weight sets selected by the decision kind
constexpr int kSetsB = 64;      // ... by the previous token's type and the scope depth
constexpr int kSetsC = 64;      // ... by the recent local and global flag outcomes
constexpr int kSetsF = kNodes;  // layer-2 weight sets, by decision kind
constexpr int kLr1 = 36;
constexpr int kLr2 = 16;
static_assert(65536 * kLr1 >> 10 <= 32767 && 65536 * kLr2 >> 10 <= 32767);
constexpr i16 kInitW1 = 4096;  // Q14: 0.25
constexpr i16 kInitW2 = 5461;  // Q14: 1/3
constexpr i16 kBias = 256;
constexpr int kPMin = 4, kPMax = 65532;
constexpr u64 kScanTokens = 1024;  // history replayed at every block start
constexpr u32 kMaxNesting = u32{1} << 20;

struct LevelSpec {
  u8 ctx_bits;   // log2 of the counters per expert
  u8 succ_bits;  // log2 of the local successor table's entries
};
constexpr std::array<LevelSpec, kStructuralCMMaxLevel + 1> kLevels{{
    {17, 16},  // ~2.3 MiB
    {18, 17},  // ~4.5 MiB
    {19, 18},  // ~9 MiB
    {20, 19},  // ~18 MiB
    {21, 20},  // ~36 MiB
}};
constexpr const LevelSpec& spec_for(u8 level) noexcept {
  return kLevels[std::min<u8>(level, kStructuralCMMaxLevel)];
}

// ------------------------------------------------- encoder-side policy ---
constexpr unsigned kMaxReplays = 3;
constexpr std::size_t kMinCmBlock = 16;  // bytes: four tokens

std::atomic<const ncm::Kernels*> g_kernels{nullptr};  // tests: forced kernels

// --------------------------------------------------------------- helpers ---
constexpr u64 hash(u64 v, u64 salt) noexcept {
  v += (salt + 1) * 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}

// Counter: a 12-bit probability and a 4-bit count; zeroed memory is p = 0.5, n = 0.
constexpr int node_p(u16 n) noexcept { return ((n >> 4) ^ 0x800); }

// Units (64-byte lines) of a table written since the last reset (R19).
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

// The 32-bit binary arithmetic coder of NeuralCMCodec (P(1) in Q16, exact
// 64-bit split), repeated here so that codec's file stays byte-for-byte as released.
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

constexpr u32 load_id(const std::byte* p) noexcept {
  return std::to_integer<u32>(p[0]) | std::to_integer<u32>(p[1]) << 8 | std::to_integer<u32>(p[2]) << 16 |
         std::to_integer<u32>(p[3]) << 24;
}

// Token history, frame-relative token indices. The models read at most
// kScanTokens back and never past min(window, position) (R21).
struct FrameHistory {
  const std::byte* data;
  u32 id(u64 t) const noexcept { return load_id(data + 4 * t); }
};
struct WindowHistory {
  const hc::Window* w;
  u64 base;  // absolute window position of the frame start
  u32 id(u64 t) const noexcept {
    const u64 a = base + 4 * t;
    const std::byte b[4] = {w->at(a), w->at(a + 1), w->at(a + 2), w->at(a + 3)};
    return load_id(b);
  }
};

}  // namespace

// ----------------------------------------------------------------- model ---
struct StructuralCMCodec::Model {
  u16* ctr = nullptr;  // kCtx regions of 2^ctx_bits counters
  u32* succ = nullptr; // local order-2 successor: ID + 1, 0 = empty
  i16* wa = nullptr;
  i16* wb = nullptr;
  i16* wc = nullptr;
  i16* wf = nullptr;
  Dirty dirty_ctr;   // 64-byte lines: 32 counters each
  Dirty dirty_succ;  // 64-byte lines: 16 entries each
  const ncm::Kernels* k = nullptr;
  const GlobalModel* g = nullptr;  // shared, read-only
  LevelSpec spec{};
  u64 window_tokens = 0;
  u32 ctx_mask = 0, succ_mask = 0;
  bool stale = false;

  // History state: re-derived from the history at every block start.
  u64 pos = 0;  // frame-relative index of the next token
  u32 h1 = kNoSymbol, h2 = kNoSymbol, h3 = kNoSymbol;
  u32 line_first = kNoSymbol, last_kw = kNoSymbol;
  u32 depth = 0, brackets = 0, indent = 0, line_pos = 0;
  bool at_line_start = true;
  u32 hist_l = 0, hist_g = 0;  // recent flag outcomes, most recent lowest
  u64 tctx[kCtx]{};
  int set_b = 0;
  int path = 0;  // how the last token was coded: 0 local flag, 1 global flag, 2 explicit

  // Per-decision state.
  std::size_t idx[kCtx]{};
  int np[kCtx]{};
  int nn[kCtx]{};
  alignas(32) i16 x[ncm::kLanes]{};
  i16* w1[ncm::kSets]{};
  i32 dots[ncm::kSets]{};
  i32 errs[ncm::kSets]{};
  int t[ncm::kSets]{};
  i16* f = nullptr;
  int pm = 32768;

  // ---------------------------------------------------------- layout ---
  struct Layout {
    std::size_t ctr, succ, wa, wb, wc, wf, dctr, dsucc, total;
  };
  static constexpr std::size_t ctr_lines(const LevelSpec& s) noexcept { return (std::size_t{kCtx} << s.ctx_bits) / 32; }
  static constexpr std::size_t succ_lines(const LevelSpec& s) noexcept { return (std::size_t{1} << s.succ_bits) / 16; }
  static constexpr std::size_t align64(std::size_t n) noexcept { return (n + 63) & ~std::size_t{63}; }
  static constexpr Layout layout(const LevelSpec& s) noexcept {
    Layout l{};
    std::size_t at = align64(sizeof(Model));
    const auto take = [&at](std::size_t bytes) {
      const std::size_t here = at;
      at = align64(at + bytes);
      return here;
    };
    l.ctr = take((std::size_t{kCtx} << s.ctx_bits) * sizeof(u16));
    l.succ = take((std::size_t{1} << s.succ_bits) * sizeof(u32));
    l.wa = take(std::size_t{kSetsA} * ncm::kLanes * sizeof(i16));
    l.wb = take(std::size_t{kSetsB} * ncm::kLanes * sizeof(i16));
    l.wc = take(std::size_t{kSetsC} * ncm::kLanes * sizeof(i16));
    l.wf = take(std::size_t{kSetsF} * 4 * sizeof(i16));
    l.dctr = take(Dirty::words(ctr_lines(s)) * sizeof(u64));
    l.dsucc = take(Dirty::words(succ_lines(s)) * sizeof(u64));
    l.total = at;
    return l;
  }

  void carve(std::byte* base, const LevelSpec& s, u64 window_bytes) noexcept {
    const Layout l = layout(s);
    ctr = reinterpret_cast<u16*>(base + l.ctr);
    succ = reinterpret_cast<u32*>(base + l.succ);
    wa = reinterpret_cast<i16*>(base + l.wa);
    wb = reinterpret_cast<i16*>(base + l.wb);
    wc = reinterpret_cast<i16*>(base + l.wc);
    wf = reinterpret_cast<i16*>(base + l.wf);
    dirty_ctr = {reinterpret_cast<u64*>(base + l.dctr), ctr_lines(s), 0};
    dirty_succ = {reinterpret_cast<u64*>(base + l.dsucc), succ_lines(s), 0};
    const ncm::Kernels* forced = g_kernels.load(std::memory_order_relaxed);
    k = forced ? forced : &ncm::active_kernels();
    spec = s;
    window_tokens = window_bytes / kTokenRecordBytes;
    ctx_mask = (u32{1} << s.ctx_bits) - 1;
    succ_mask = (u32{1} << s.succ_bits) - 1;
  }

  // R20: every table back to its initial image; `full` after binding.
  void reset(bool full) noexcept {
    const std::size_t ctr_bytes = (std::size_t{kCtx} << spec.ctx_bits) * sizeof(u16);
    const std::size_t succ_bytes = (std::size_t{1} << spec.succ_bits) * sizeof(u32);
    const auto all_ctr = [this, ctr_bytes] { std::memset(ctr, 0, ctr_bytes); };
    const auto all_succ = [this, succ_bytes] { std::memset(succ, 0, succ_bytes); };
    if (full) {
      all_ctr();
      all_succ();
      dirty_ctr.forget();
      dirty_succ.forget();
    } else {
      dirty_ctr.drain([this](std::size_t line) { std::memset(ctr + 32 * line, 0, 64); }, all_ctr);
      dirty_succ.drain([this](std::size_t line) { std::memset(succ + 16 * line, 0, 64); }, all_succ);
    }
    stale = false;
    const auto init = [](i16* w, int sets, int lanes, int used, i16 v) {
      for (int s = 0; s < sets; ++s)
        for (int i = 0; i < lanes; ++i) w[s * lanes + i] = i < used ? v : i16{0};
    };
    init(wa, kSetsA, ncm::kLanes, kInputs - 1, kInitW1);  // the bias lane starts at 0
    init(wb, kSetsB, ncm::kLanes, kInputs - 1, kInitW1);
    init(wc, kSetsC, ncm::kLanes, kInputs - 1, kInitW1);
    init(wf, kSetsF, 4, 3, kInitW2);
    std::fill(std::begin(x), std::end(x), i16{0});
  }

  // ------------------------------------------------------- contexts ---
  u32 type_of(u32 id) const noexcept { return id < g->size() ? static_cast<u32>(g->type(id)) : 6u; }

  void compute_contexts() noexcept {
    const u32 ty1 = type_of(h1);
    const u64 st = u64{ty1} | u64{std::min(depth, 15u)} << 3 | u64{std::min(brackets, 7u)} << 7 |
                   u64{std::min(indent >> 1, 31u)} << 10 | u64{at_line_start} << 15 | u64{std::min(line_pos, 7u)} << 16;
    tctx[0] = hash(0, 1);                                 // order 0
    tctx[1] = hash(h1, 2);                                // order 1
    tctx[2] = hash(u64{h1} << 32 | h2, 3);                // order 2
    tctx[3] = hash(hash(u64{h1} << 32 | h2, 4) ^ h3, 5);  // order 3
    tctx[4] = hash(st, 6);                                // structural state
    tctx[5] = hash(u64{line_first} << 32 | u64{std::min(line_pos, 15u)} << 1 | u64{at_line_start}, 7);
    tctx[6] = hash(u64{last_kw} << 32 | h1, 8);           // the governing keyword
    tctx[7] = hash(u64{h2} << 32 | h3, 9);                // skip-gram
    set_b = static_cast<int>(ty1 | std::min(depth, 7u) << 3);
  }

  // History update by one token (no model learning). Follows the rules of
  // structural::StructuralTracker through the symbols' precomputed traits.
  void advance(u32 id) noexcept {
    const bool known = id < g->size();
    const u8 tr = known ? g->traits(id) : 0;
    h3 = h2;
    h2 = h1;
    h1 = known ? id : kNoSymbol;
    ++pos;
    if (tr & structural::kTraitNewline) {
      at_line_start = true;
      indent = line_pos = 0;
      line_first = kNoSymbol;
      return;
    }
    if ((tr & structural::kTraitIndent) && at_line_start) {
      indent = g->indent(id);
      return;
    }
    if (tr & structural::kTraitOpenBrace) {
      if (depth < kMaxNesting) ++depth;
    } else if (tr & structural::kTraitCloseBrace) {
      if (depth > 0) --depth;
    } else if (tr & structural::kTraitOpenBracket) {
      if (brackets < kMaxNesting) ++brackets;
    } else if (tr & structural::kTraitCloseBracket) {
      if (brackets > 0) --brackets;
    }
    if (at_line_start) line_first = h1;
    at_line_start = false;
    if (line_pos < 255) ++line_pos;
    if (known && g->type(id) == TokenType::KEYWORD) last_kw = id;
  }

  // Re-derives the history state at a block start from at most kScanTokens
  // tokens: identical on both sides whether the previous block was CM or STORED.
  template <class H>
  void begin_block(const H& h, u64 at) noexcept {
    h1 = h2 = h3 = line_first = last_kw = kNoSymbol;
    depth = brackets = indent = line_pos = 0;
    at_line_start = true;
    hist_l = hist_g = 0;
    const u64 scan = std::min({kScanTokens, window_tokens, at});
    pos = at - scan;
    for (u64 q = at - scan; q < at; ++q) advance(h.id(q));
    compute_contexts();
  }

  std::size_t succ_slot() const noexcept { return static_cast<std::size_t>(hash(u64{h1} << 32 | h2, 11) & succ_mask); }

  // ---------------------------------------------------- bit modelling ---
  int predict(int node, u64 key) noexcept {
    const u64 salt = (key + 1) * 0x9E3779B97F4A7C15ull + static_cast<u32>(node);  // node >= 0
    for (int i = 0; i < kCtx; ++i) {
      u64 hv = tctx[i] ^ salt;
      hv = (hv ^ (hv >> 29)) * 0xBF58476D1CE4E5B9ull;
      hv ^= hv >> 32;
      idx[i] = (static_cast<std::size_t>(i) << spec.ctx_bits) | (static_cast<u32>(hv) & ctx_mask);
      const u16 v = ctr[idx[i]];
      np[i] = node_p(v);
      nn[i] = v & 15;
      x[i] = static_cast<i16>(ncm::stretch12(np[i]));
    }
    x[kCtx] = kBias;
    w1[0] = wa + node * ncm::kLanes;
    w1[1] = wb + set_b * ncm::kLanes;
    w1[2] = wc + static_cast<int>((hist_l & 7) | (hist_g & 7) << 3) * ncm::kLanes;
    k->mix(x, w1, dots);
    for (int j = 0; j < ncm::kSets; ++j) t[j] = std::clamp(dots[j] >> 14, -ncm::kStretchMax, ncm::kStretchMax);
    f = wf + node * 4;
    static_assert(ncm::kSets == 3);
    const i32 d2 = t[0] * f[0] + t[1] * f[1] + t[2] * f[2];
    pm = std::max(ncm::squash16(std::clamp(d2 >> 14, -ncm::kLogitMax, ncm::kLogitMax)), 1);
    return std::clamp(pm, kPMin, kPMax);
  }

  void update(int y) noexcept {
    const int ty = (y << 12) - y;
    for (int i = 0; i < kCtx; ++i) {
      const int pn = np[i] + (((ty - np[i]) * ncm::kRate[static_cast<std::size_t>(nn[i])] + (1 << 15)) >> 16);
      ctr[idx[i]] = static_cast<u16>(((pn ^ 0x800) << 4) | (nn[i] + (nn[i] < 15)));
      dirty_ctr.mark(idx[i] >> 5);
    }
    for (int j = 0; j < ncm::kSets; ++j) errs[j] = (((y << 16) - ncm::squash16(t[j])) * kLr1) >> 10;
    k->train(x, w1, errs);
    const int e2 = (((y << 16) - pm) * kLr2) >> 10;
    for (int j = 0; j < ncm::kSets; ++j)
      f[j] = static_cast<i16>(std::clamp(f[j] + ((((t[j] * 16) * e2 >> 16) + 1) >> 1), -32768, 32767));
  }

  template <class IO>
  int decide(IO& io, int node, u64 key, int y) noexcept {
    const int bit = io.bit(predict(node, key), y);
    update(bit);
    return bit;
  }

  // One token. `actual` is the true ID where it is known (encode, replay).
  // Returns the coded ID, or kNoSymbol when a decoded ID is outside the model.
  template <class IO>
  u32 code_token(IO& io, u32 actual) noexcept {
    const u32 n = g->size();
    const u32 sl = succ[succ_slot()];
    const u32 el = sl != 0 && sl - 1 < n ? sl - 1 : kNoSymbol;
    const u32 eg = h1 < n ? g->successor(h1) : kNoSymbol;
    if (el != kNoSymbol) {
      const int y = decide(io, kNodeLocal, u64{el} | u64{hist_l & 15} << 32, actual == el);
      hist_l = ((hist_l << 1) | static_cast<u32>(y)) & 0xFF;
      if (y) {
        path = 0;
        return el;
      }
    }
    if (eg != kNoSymbol && eg != el) {
      const int y = decide(io, kNodeGlobal, u64{eg} | u64{hist_g & 15} << 32, actual == eg);
      hist_g = ((hist_g << 1) | static_cast<u32>(y)) & 0xFF;
      if (y) {
        path = 1;
        return eg;
      }
    }
    path = 2;
    // Elias gamma of v = ID + 1: the length k = floor(log2 v) in 5 bits, then
    // the k bits below the leading one, most significant first.
    const u64 v = u64{actual} + 1;
    const int len = static_cast<int>(std::bit_width(v)) - 1;
    int kk = 0;
    for (int b = kLengthBits - 1, node = 1; b >= 0; --b) {
      const int bit = decide(io, kNodeLength + node - 1, 0, (len >> b) & 1);
      node = node * 2 + bit;
      kk = kk * 2 + bit;
    }
    u64 r = 1;
    for (int j = 0; j < kk; ++j) {
      const int node = kNodeMantissa + (std::min(kk, 15) << 2 | std::min(j, 3));
      const int bit = decide(io, node, r << 5 | static_cast<u64>(kk), static_cast<int>((v >> (kk - 1 - j)) & 1));
      r = r * 2 + static_cast<u64>(bit);
    }
    return r - 1 < n ? static_cast<u32>(r - 1) : kNoSymbol;
  }

  // After a token: the local successor table learns it, then the history moves on.
  void learn(u32 id) noexcept {
    const std::size_t s = succ_slot();
    succ[s] = id + 1;
    dirty_succ.mark(s >> 4);
    advance(id);
    compute_contexts();
  }
};

// ------------------------------------------------------------- the codec ---
StructuralCMCodec::StructuralCMCodec() noexcept = default;

StructuralCMCodec::StructuralCMCodec(const hc::StreamParams& p) {
  if (accepts(p) != hc::Errc::ok) return;  // no model: every block is STORED
  const u64 n = memory_required(p);
  owned_.reset(new std::byte[static_cast<std::size_t>(n)]);
  const auto addr = reinterpret_cast<std::uintptr_t>(owned_.get());
  std::byte* const base = owned_.get() + ((64 - (addr & 63)) & 63);
  Model* const m = new (base) Model{};
  m->carve(base, spec_for(p.level), p.window_size());
  m->reset(true);
  m_ = m;
}

StructuralCMCodec::StructuralCMCodec(StructuralCMCodec&& other) noexcept
    : m_(std::exchange(other.m_, nullptr)),
      owned_(std::move(other.owned_)),
      global_(other.global_),
      frame_(other.frame_),
      cm_blocks_(std::move(other.cm_blocks_)),
      replays_(other.replays_),
      store_rest_(other.store_rest_),
      counters_(other.counters_) {}

StructuralCMCodec& StructuralCMCodec::operator=(StructuralCMCodec&& other) noexcept {
  if (this != &other) {
    m_ = std::exchange(other.m_, nullptr);
    owned_ = std::move(other.owned_);
    global_ = other.global_;
    frame_ = other.frame_;
    cm_blocks_ = std::move(other.cm_blocks_);
    replays_ = other.replays_;
    store_rest_ = other.store_rest_;
    counters_ = other.counters_;
  }
  return *this;
}

StructuralCMCodec::~StructuralCMCodec() = default;  // Model is trivially destructible

void StructuralCMCodec::set_context(const void* global_model) noexcept {
  global_ = static_cast<const structural::GlobalModel*>(global_model);
  if (m_) m_->g = global_;
}

hc::Errc StructuralCMCodec::accepts(const hc::StreamParams& p) noexcept {
  if ((p.prior_id & 0xFFFFFF00u) != kStructuralCMFamily || (p.prior_id & 0xFFu) != kStructuralCMProfile)
    return hc::Errc::unsupported_feature;
  if (p.level > kStructuralCMMaxLevel) return hc::Errc::unsupported_feature;
  return hc::Errc::ok;
}

hc::u64 StructuralCMCodec::memory_required(const hc::StreamParams& p) noexcept {
  return Model::layout(spec_for(p.level)).total + 63;
}

hc::Errc StructuralCMCodec::bind(const hc::StreamParams& p, hc::MutBytes arena) noexcept {
  m_ = nullptr;
  if (const hc::Errc e = accepts(p); e != hc::Errc::ok) return e;
  if (!global_) return hc::Errc::unsupported_feature;  // the dictionary is not in the stream
  if (arena.size() < memory_required(p)) return hc::Errc::out_of_memory;
  const auto addr = reinterpret_cast<std::uintptr_t>(arena.data());
  std::byte* const base = arena.data() + ((64 - (addr & 63)) & 63);
  Model* const m = new (base) Model{};
  m->carve(base, spec_for(p.level), p.window_size());
  m->g = global_;
  m->reset(true);
  m_ = m;
  return hc::Errc::ok;
}

void StructuralCMCodec::reset_frame() noexcept {
  if (m_) m_->stale = true;
}

hc::Errc StructuralCMCodec::decode_block(hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept {
  if (!m_ || !m_->g) return hc::Errc::payload_corrupt;
  if (usize % kTokenRecordBytes != 0 || w.frame_offset() % kTokenRecordBytes != 0) return hc::Errc::payload_corrupt;
  Model& m = *m_;
  if (m.stale) m.reset(false);
  ArithDecoder dec(payload);
  DecodeIO io{dec};
  const WindowHistory h{&w, w.pos() - w.frame_offset()};
  m.begin_block(h, w.frame_offset() / kTokenRecordBytes);
  for (u32 i = 0; i < usize / kTokenRecordBytes; ++i) {
    const u32 id = m.code_token(io, 0);
    if (dec.overrun() || id == kNoSymbol) return hc::Errc::payload_corrupt;
    for (int b = 0; b < 4; ++b) w.put(static_cast<std::byte>(id >> (8 * b)));
    m.learn(id);
  }
  return dec.exact() ? hc::Errc::ok : hc::Errc::payload_corrupt;
}

void StructuralCMCodec::reset_frame(hc::Bytes frame) noexcept {
  frame_ = frame;
  cm_blocks_.clear();
  replays_ = 0;
  store_rest_ = false;
  if (m_) m_->reset(false);
}

std::size_t StructuralCMCodec::encode_block(std::size_t begin, std::size_t end, hc::MutBytes out) noexcept {
  const std::size_t n = end - begin;
  const auto store = [this] {
    ++counters_.stored_blocks;
    return std::size_t{0};
  };
  if (!m_ || !m_->g || store_rest_ || n < kMinCmBlock || out.size() < n || begin % kTokenRecordBytes != 0 ||
      n % kTokenRecordBytes != 0)
    return store();
  Model& m = *m_;
  for (std::size_t i = begin; i < end; i += kTokenRecordBytes)  // IDs the model cannot code: STORED
    if (load_id(frame_.data() + i) >= m.g->size()) return store();
  if (cm_blocks_.size() == cm_blocks_.capacity()) {
    try {
      cm_blocks_.reserve(std::max<std::size_t>(64, cm_blocks_.capacity() * 2));
    } catch (...) {
      store_rest_ = true;
      return store();
    }
  }

  ArithEncoder enc(out.data(), n - 1);  // a CM payload must be strictly smaller
  EncodeIO io{enc};
  const FrameHistory h{frame_.data()};
  m.begin_block(h, begin / kTokenRecordBytes);
  Counters c{};
  for (std::size_t i = begin; i < end && !enc.full(); i += kTokenRecordBytes) {
    const u32 id = load_id(frame_.data() + i);
    m.code_token(io, id);
    m.learn(id);
    ++c.tokens;
    c.local_hits += m.path == 0;
    c.global_hits += m.path == 1;
    c.explicit_ids += m.path == 2;
  }
  if (!enc.full()) enc.flush();
  if (enc.full()) {  // expands: store it, and rebuild the models without it (R12)
    replay();
    ++counters_.replays;
    if (++replays_ >= kMaxReplays) store_rest_ = true;
    return store();
  }
  cm_blocks_.emplace_back(begin, end);
  ++counters_.cm_blocks;
  counters_.tokens += c.tokens;
  counters_.local_hits += c.local_hits;
  counters_.global_hits += c.global_hits;
  counters_.explicit_ids += c.explicit_ids;
  return enc.size();
}

void StructuralCMCodec::replay() noexcept {
  Model& m = *m_;
  m.reset(false);
  ReplayIO io;
  const FrameHistory h{frame_.data()};
  for (const auto& [b, e] : cm_blocks_) {
    m.begin_block(h, b / kTokenRecordBytes);
    for (std::size_t i = b; i < e; i += kTokenRecordBytes) {
      const u32 id = load_id(frame_.data() + i);
      m.code_token(io, id);
      m.learn(id);
    }
  }
}

bool StructuralCMCodec::test_select_simd(const char* name) noexcept {
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

static_assert(hc::BlockCodec<StructuralCMCodec>);
static_assert(FrameEncoder<StructuralCMCodec>);

}  // namespace hce
