// Context-mixing bit predictor shared by Tiers 3 and 4.
//
// Bytes are coded MSB first as 8 binary decisions. Per bit:
//
//   inputs   stretch(p) of: an order-1 counter (direct table), kHashed hashed
//            context counters, kMatch match models; plus a bias input
//   mixer    integer gated linear mixer(s) in the logistic domain, weight row
//            chosen by a small context; Tier 4 runs three first-layer mixers
//            and a final mixer over their outputs (two layers)
//   APM      adaptive probability maps (SSE) refine the mixer output
//   output   12-bit P(bit = 1) for the binary rANS coder
//
// Research levers built in (each switchable in the Profile, measured in
// ultracodec/README.md):
//   * LUT distillation: stretch/squash are integer look-up tables (no exp/log
//     at run time; the squash table is lpaq's 33-point integer curve, so the
//     tables and thus the archives are identical on every platform);
//   * confident-prediction skipping: when the mixer output is already near
//     certain, the APM stage (and its update) is bypassed; optionally, inside
//     long, reliable matches, the whole mixer and all context counters are
//     bypassed and the match model alone predicts;
//   * memory-wall mitigation: counters live in 64-byte buckets (one cache
//     line holds a whole nibble's 15 tree nodes plus a tag), two-way set
//     associative pairs on 128 bytes, looked up twice per byte; the buckets
//     of the next nibble are prefetched (__builtin_prefetch) for both
//     possible values of the pending bit, one bit early; tables are
//     huge-page backed (common.hpp).
//
// Counters store (p22 << 10 | n) XOR 0x80000000, so zeroed memory reads as
// p = 0.5, n = 0 and tables reset with one madvise/memset. Adaptation rate is
// 1/(n + 1.5) up to a per-profile limit (state-count adaptive counters).
#pragma once

#include <algorithm>
#include <array>

#include "common.hpp"

namespace ultracodec::detail {

// ---------------------------------------------------------------- LUTs
struct LogisticTables {
  short stretch[4096];   // p (12-bit) -> ln(p/(1-p)) * 256, clamped to +-2047
  short squash[4096];    // d + 2048 -> 4096 / (1 + e^-d/256)
};
constexpr int squash_slow(int d) {
  constexpr int t[33] = {1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047,
                         2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
  if (d > 2047) return 4095;
  if (d < -2047) return 1;
  int w = d & 127;
  int i = (d >> 7) + 16;
  return (t[i] * (128 - w) + t[i + 1] * w + 64) >> 7;
}
constexpr LogisticTables make_logistic() {
  LogisticTables t{};
  for (int d = -2048; d < 2048; ++d) t.squash[d + 2048] = static_cast<short>(squash_slow(d));
  int pi = 0;
  for (int x = -2047; x <= 2047; ++x) {
    int v = squash_slow(x);
    for (int i = pi; i <= v; ++i) t.stretch[i] = static_cast<short>(x);
    pi = v + 1;
  }
  for (int i = pi; i < 4096; ++i) t.stretch[i] = 2047;
  return t;
}
inline constexpr LogisticTables kLogistic = make_logistic();
ULTRA_INLINE int stretch(int p) noexcept { return kLogistic.stretch[p]; }
ULTRA_INLINE int squash(int d) noexcept {
  d = std::clamp(d, -2047, 2047);
  return kLogistic.squash[d + 2048];
}

// 1/(n + 1.5) in 2.14 fixed point, the adaptive counter rate.
struct RateTable {
  int v[1024];
};
constexpr RateTable make_rates() {
  RateTable t{};
  for (int n = 0; n < 1024; ++n) t.v[n] = 32768 / (2 * n + 3);
  return t;
}
inline constexpr RateTable kRates = make_rates();

// Counter: (p16 << 16 | n) ^ 0x80000000; all arithmetic fits in 32 bits.
constexpr u32 kFlip = 0x80000000u;
ULTRA_INLINE int counter_p(u32 stored) noexcept { return static_cast<int>((stored ^ kFlip) >> 20); }  // 12-bit
template <int Limit>
ULTRA_INLINE void counter_update(u32& stored, int y) noexcept {
  const u32 t = stored ^ kFlip;
  const int n = static_cast<int>(t & 0xFFFF);
  int p = static_cast<int>(t >> 16);
  p += ((y ? 65535 : 0) - p) * kRates.v[n] >> 14;
  stored = ((static_cast<u32>(p) << 16) | static_cast<u32>(n + (n < Limit))) ^ kFlip;
}

ULTRA_INLINE u32 hash32(u64 x) noexcept {
  x *= 0x9E3779B97F4A7C15ull;
  x ^= x >> 29;
  x *= 0xBF58476D1CE4E5B9ull;
  return static_cast<u32>(x >> 32);
}

// ---------------------------------------------------------------- APM
class Apm {
 public:
  void init(std::size_t contexts) {
    u16 row[33];
    for (int j = 0; j < 33; ++j) row[j] = static_cast<u16>(squash_slow((j - 16) * 128) * 16);
    t_.resize(contexts * 33);
    for (std::size_t i = 0; i < t_.size(); i += 33) std::memcpy(&t_[i], row, sizeof row);
  }
  ULTRA_INLINE void prefetch(std::size_t cx) const noexcept {
    ULTRA_PREFETCH(&t_[cx * 33]);
    ULTRA_PREFETCH(&t_[cx * 33 + 32]);
  }
  ULTRA_INLINE int pp(int st, std::size_t cx) noexcept {
    int s = st + 2048;  // 1..4095
    int lo = s >> 7, w = s & 127;
    idx_ = cx * 33 + static_cast<std::size_t>(lo);
    w_ = w;
    return (t_[idx_] * (128 - w) + t_[idx_ + 1] * w) >> 11;
  }
  ULTRA_INLINE void update(int y) noexcept {
    int g = (y << 16) + (y << kRate) - y - y;
    u16& a = t_[idx_];
    u16& b = t_[idx_ + 1];
    if (w_ < 96) a = static_cast<u16>(a + ((g - a) >> kRate));
    if (w_ > 32) b = static_cast<u16>(b + ((g - b) >> kRate));
  }

 private:
  static constexpr int kRate = 7;
  std::vector<u16> t_;
  std::size_t idx_ = 0;
  int w_ = 0;
};

// ---------------------------------------------------------------- mixer
// Integer mixer in 16-bit lanes (the paq8 SIMD mixer formulation): inputs are
// stretched probabilities (+-2047), weights are int16 with 1.0 = 2^14, so
// the dot product is exact in 32 bits (SSE2 pmaddwd / NEON vmlal) and
// training is one multiply-high and two saturating adds per lane. The scalar
// fallback computes the same values bit for bit.
template <unsigned N>
struct MixerRow {
  alignas(16) short w[N];
};

// Returns sum(x * w) >> 14, i.e. the mixed logit.
template <unsigned N>
ULTRA_INLINE int mix_dot(const short* x, const short* w) noexcept {
  static_assert(N % 8 == 0);
#if defined(ULTRA_SSE2)
  __m128i acc = _mm_setzero_si128();
  for (unsigned i = 0; i < N; i += 8)
    acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(x + i)),
                                            _mm_load_si128(reinterpret_cast<const __m128i*>(w + i))));
  acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0x4E));
  acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0xB1));
  return _mm_cvtsi128_si32(acc) >> 14;
#elif defined(ULTRA_NEON)
  int32x4_t acc = vdupq_n_s32(0);
  for (unsigned i = 0; i < N; i += 8) {
    int16x8_t a = vld1q_s16(x + i), b = vld1q_s16(w + i);
    acc = vmlal_s16(acc, vget_low_s16(a), vget_low_s16(b));
    acc = vmlal_high_s16(acc, a, b);
  }
  return vaddvq_s32(acc) >> 14;
#else
  int s = 0;
  for (unsigned i = 0; i < N; ++i) s += x[i] * w[i];
  return s >> 14;
#endif
}

// Writes N values as int16 with whole-vector stores built in registers. (A
// store per element followed by a vector load of the same bytes defeats
// store-to-load forwarding and stalls every bit by ~15 cycles.)
template <unsigned N>
ULTRA_INLINE void pack_inputs(const int* v, short* x) noexcept {
  for (unsigned i = 0; i < N; i += 8) {
#if defined(ULTRA_SSE2)
    __m128i lo = _mm_setr_epi32(v[i], v[i + 1], v[i + 2], v[i + 3]);
    __m128i hi = _mm_setr_epi32(v[i + 4], v[i + 5], v[i + 6], v[i + 7]);
    _mm_store_si128(reinterpret_cast<__m128i*>(x + i), _mm_packs_epi32(lo, hi));
#elif defined(ULTRA_NEON)
    int32x4_t lo = {v[i], v[i + 1], v[i + 2], v[i + 3]};
    int32x4_t hi = {v[i + 4], v[i + 5], v[i + 6], v[i + 7]};
    vst1q_s16(x + i, vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
#else
    for (unsigned k = 0; k < 8; ++k) x[i + k] = static_cast<short>(v[i + k]);
#endif
  }
}

ULTRA_INLINE short sat16(int v) noexcept { return static_cast<short>(std::clamp(v, -32768, 32767)); }

// w += round(x * err / 2^16), saturating; err = (y - p) * learning rate, |err| < 2^15.
template <unsigned N>
ULTRA_INLINE void mix_train(const short* x, short* w, int err) noexcept {
#if defined(ULTRA_SSE2)
  const __m128i e = _mm_set1_epi16(static_cast<short>(err));
  const __m128i one = _mm_set1_epi16(1);
  for (unsigned i = 0; i < N; i += 8) {
    __m128i xi = _mm_load_si128(reinterpret_cast<const __m128i*>(x + i));
    __m128i t = _mm_mulhi_epi16(_mm_adds_epi16(xi, xi), e);
    t = _mm_srai_epi16(_mm_adds_epi16(t, one), 1);
    __m128i* wp = reinterpret_cast<__m128i*>(w + i);
    _mm_store_si128(wp, _mm_adds_epi16(_mm_load_si128(wp), t));
  }
#elif defined(ULTRA_NEON)
  const int16x8_t e = vdupq_n_s16(static_cast<short>(err));
  const int16x8_t one = vdupq_n_s16(1);
  for (unsigned i = 0; i < N; i += 8) {
    int16x8_t t = vqdmulhq_s16(vld1q_s16(x + i), e);  // (2 x err) >> 16
    t = vshrq_n_s16(vqaddq_s16(t, one), 1);
    vst1q_s16(w + i, vqaddq_s16(vld1q_s16(w + i), t));
  }
#else
  for (unsigned i = 0; i < N; ++i) {
    int t = ((2 * x[i]) * err) >> 16;
    t = sat16(t + 1) >> 1;
    w[i] = sat16(w[i] + t);
  }
#endif
}

// ---------------------------------------------------------------- model
// Byte-level state the context functions read.
struct ByteState {
  u32 c4 = 0;      // last 4 bytes, most recent in the low byte
  u32 c8 = 0;      // the 4 before those
  u64 w0 = 0;      // hash of the current word (ASCII letters, case-folded)
  u64 w1 = 0;      // hash of the previous word
  u32 col = 0;     // bytes since the last '\n'
  u32 above = 0;   // byte at the same column of the previous line (0 if none)
};

template <class P>
class CmModel {
 public:
  static constexpr unsigned kH = P::kHashed;
  static constexpr unsigned kM = P::kMatch;
  // Confidence maps: one adaptive counter per (context, log2 count, p bucket)
  // re-estimates each context's prediction from how often it was seen
  // (an indirect, PAQ-StateMap-like second opinion; one extra input each).
  static constexpr unsigned kC = P::kConfMaps ? 1 + kH : 0;
  static constexpr unsigned kInputs = (2 + kH + kM + kC + 7) & ~7u;
  static constexpr unsigned kMixers = P::kTwoLayer ? 3 : 1;

  // bucket_pairs: hashed-counter table size in 128-byte pairs.
  CmModel(std::size_t bucket_pairs, std::size_t max_history)
      : pairs_(bucket_pairs), table_(bucket_pairs * 32), o1_(65536) {
    for (unsigned m = 0; m < kM; ++m) mt_[m].reset(std::size_t{1} << P::kMatchLog2);
    hist_.resize(max_history);
    for (unsigned k = 0; k < kMixers; ++k) w_[k].resize(P::mixer_rows(k));
    if constexpr (P::kTwoLayer) fw_.resize(256);
    a1_.init(65536);
    if constexpr (P::kSecondApm) a2_.init(std::size_t{1} << 20);
    initialized_ = true;
  }

  // Fresh model for a block: every table back to its initial state.
  void reset() {
    if (!initialized_) {
      table_.zero();
      o1_.zero();
      for (unsigned m = 0; m < kM; ++m) mt_[m].zero();
      a1_.init(65536);
      if constexpr (P::kSecondApm) a2_.init(std::size_t{1} << 20);
    }
    initialized_ = false;
    for (unsigned k = 0; k < kMixers; ++k)
      for (auto& row : w_[k]) init_row(row.w, P::kInitWeight);
    if constexpr (P::kTwoLayer)
      for (auto& row : fw_) init_row8(row.w, 16384 / 3);
    std::fill(std::begin(cmap_), std::end(cmap_), 0u);
    for (unsigned m = 0; m < kM; ++m) {
      std::fill(std::begin(mctr_[m]), std::end(mctr_[m]), 0u);
      mlen_[m] = 0;
      mptr_[m] = 0;
    }
    bs_ = ByteState{};
    line_start_ = prev_line_ = 0;
    pos_ = 0;
    c0_ = 1;
    bit_ = 0;
    nib_ = 1;
    for (unsigned i = 0; i < kH; ++i) h_[i] = 0;
    pending_ = true;
    std::fill(std::begin(x_), std::end(x_), short{0});
  }

  // Model bytes without coding them (wavefront prefix).
  void prime(std::span<const u8> bytes) {
    for (u8 c : bytes)
      for (int b = 7; b >= 0; --b) {
        (void)p();
        update((c >> b) & 1);
      }
  }

  std::size_t pos() const noexcept { return pos_; }
  u8 last_byte() const noexcept { return static_cast<u8>(bs_.c4); }

  // 12-bit P(next bit = 1), in [1, 4095].
  ULTRA_INLINE int p() {
    const u32 c1 = bs_.c4 & 0xFF;
    // Match model inputs.
    int ms = 0;
    for (unsigned m = 0; m < kM; ++m) {
      mcur_[m] = nullptr;
      if (mlen_[m] == 0) continue;
      const u32 exp = hist_[mptr_[m]];
      const unsigned eb = (exp >> (7 - bit_)) & 1;
      const unsigned lb = len_bucket(mlen_[m]);
      mcur_[m] = &mctr_[m][lb * 2 + eb];
      if (m == 0) ms = lb < 16 ? 1 : lb < 24 ? 2 : 3;
    }
    if constexpr (P::kMatchSkip) {
      // Confident-prediction skipping: inside a long, reliable match the
      // match model alone predicts; mixer, APMs and counters are bypassed.
      if (mcur_[0] && mlen_[0] >= P::kSkipLen) {
        const int pm = counter_p(*mcur_[0]);
        if (pm >= 4096 - P::kSkipMargin || pm <= P::kSkipMargin) {
          skip_ = true;
          return pr_ = std::clamp(pm, 1, 4095);
        }
      }
      skip_ = false;
    }
    if (pending_) lookup_nibble();
    o1cur_ = &o1_[(c1 << 8) | c0_];
    int v[kInputs] = {};
    v[0] = stretch(counter_p(*o1cur_));
    for (unsigned i = 0; i < kH; ++i) v[1 + i] = stretch(counter_p(bkt_[i][nib_]));
    if constexpr (kC > 0) {
      const u32 nib = nib_;
      for (unsigned i = 0; i < kC; ++i) {
        const u32 t = (i == kH ? *o1cur_ : bkt_[i][nib]) ^ kFlip;
        const u32 n = t & 0xFFFF;
        const unsigned nq = n == 0 ? 0 : std::min(1 + highbit32(n), 7u);
        ccur_[i] = &cmap_[((i * 8 + nq) << 6) | (t >> 26)];
        v[2 + kH + kM + i] = stretch(counter_p(*ccur_[i]));
      }
    }
    for (unsigned m = 0; m < kM; ++m) v[1 + kH + m] = mcur_[m] ? stretch(counter_p(*mcur_[m])) : 0;
    v[1 + kH + kM] = 256;
    pack_inputs<kInputs>(v, x_);
    const short* x = x_;

    int st;
    if constexpr (P::kTwoLayer) {
      row_[0] = w_[0][c0_].w;
      row_[1] = w_[1][static_cast<unsigned>(ms) * 8 + bit_].w;
      row_[2] = w_[2][c1 * 8 + bit_].w;
      for (unsigned k = 0; k < 3; ++k) {
        st_[k] = std::clamp(mix_dot<kInputs>(x, row_[k]), -2047, 2047);
        pm_[k] = squash(st_[k]);
      }
      const int fv[8] = {st_[0], st_[1], st_[2], 256, 0, 0, 0, 0};
      pack_inputs<8>(fv, fx_);
      frow_ = fw_[c0_].w;
      st = std::clamp(mix_dot<8>(fx_, frow_), -2047, 2047);
    } else {
      row_[0] = w_[0][static_cast<unsigned>(ms) * 256 + c0_].w;
      st = std::clamp(mix_dot<kInputs>(x, row_[0]), -2047, 2047);
    }
    pmix_ = squash(st);
    int pr;
    if (P::kApmBypass && (st >= P::kBypassSt || st <= -P::kBypassSt)) {
      apm_used_ = false;
      pr = pmix_;
    } else {
      apm_used_ = true;
      const int a = a1_.pp(st, (c1 << 8) | c0_);
      if constexpr (P::kSecondApm) {
        const u32 h2 = ((bs_.c4 & 0xFFFF) * 0x9E3779B1u) >> 20;
        const int b = a2_.pp(st, (static_cast<std::size_t>(h2) << 8) | c0_);
        pr = (pmix_ + a + 2 * b + 2) >> 2;
      } else {
        pr = (pmix_ + 3 * a + 2) >> 2;
      }
    }
    return pr_ = std::clamp(pr, 1, 4095);
  }

  ULTRA_INLINE void update(int y) {
    // Locals: counter stores are u32 and would otherwise force reloads of
    // every u32 member (nib_, c0_, ...) after each store.
    const u32 nib = nib_, c0 = c0_;
    const unsigned bit = bit_;
    if (!(P::kMatchSkip && skip_)) {
      counter_update<P::kO1Limit>(*o1cur_, y);
      u32* nodes[kH];
      for (unsigned i = 0; i < kH; ++i) nodes[i] = bkt_[i] + nib;
      for (unsigned i = 0; i < kH; ++i) counter_update<P::kCounterLimit>(*nodes[i], y);
      if constexpr (kC > 0)
        for (unsigned i = 0; i < kC; ++i) counter_update<255>(*ccur_[i], y);
      if constexpr (P::kTwoLayer) {
        for (unsigned k = 0; k < 3; ++k) mix_train<kInputs>(x_, row_[k], ((y << 12) - pm_[k]) * P::kMixerLr);
        mix_train<8>(fx_, frow_, ((y << 12) - pmix_) * P::kFinalLr);
      } else {
        mix_train<kInputs>(x_, row_[0], ((y << 12) - pmix_) * P::kMixerLr);
      }
      if (apm_used_) {
        a1_.update(y);
        if constexpr (P::kSecondApm) a2_.update(y);
      }
    }
    for (unsigned m = 0; m < kM; ++m) {
      if (!mcur_[m]) continue;
      counter_update<1023>(*mcur_[m], y);
      if (static_cast<int>((hist_[mptr_[m]] >> (7 - bit)) & 1) != y) mlen_[m] = 0;
    }
    c0_ = (c0 << 1) | static_cast<u32>(y);
    nib_ = (nib << 1) | static_cast<u32>(y);
    bit_ = bit + 1;
    if (bit_ == 3) {
      prefetch_second_nibble();
    } else if (bit_ == 7) {
      prepare_next_byte();
    } else if (bit_ == 4) {
      nib_ = 1;
      pending_ = true;
    } else if (bit_ == 8) {
      byte_update();
    }
    // The next bit's APM row and order-1 counter are known now: start the
    // loads while the coder and the next bit's other inputs run.
    const u32 c1 = bs_.c4 & 0xFF;
    a1_.prefetch((c1 << 8) | c0_);
    ULTRA_PREFETCH(&o1_[(c1 << 8) | c0_]);
  }

 private:
  static void init_row(short* w, int v) {
    for (unsigned i = 0; i < kInputs; ++i) w[i] = 0;
    for (unsigned i = 0; i < 1 + kH + kM; ++i) w[i] = static_cast<short>(v);
    for (unsigned i = 0; i < kC; ++i) w[2 + kH + kM + i] = static_cast<short>(v);
  }
  static void init_row8(short* w, int v) {
    for (unsigned i = 0; i < 8; ++i) w[i] = static_cast<short>(i < 3 ? v : 0);
  }
  static ULTRA_INLINE unsigned len_bucket(u32 len) noexcept {
    return len < 16 ? len : 16 + std::min<u32>((len - 16) >> 3, 15);
  }
  // Hash of a context for the second nibble, keyed by the first nibble.
  static ULTRA_INLINE u32 nibble_hash(u32 h, u32 c0) noexcept { return (h + c0 * 0x6F4F2A2Bu) * 0x9E3779B1u; }

  ULTRA_INLINE u32* pair_of(u32 h) noexcept {
    const std::size_t pair = static_cast<std::size_t>((static_cast<u64>(h) * pairs_) >> 32);
    return table_.data() + pair * 32;
  }
  // Both 64-byte ways of a pair.
  ULTRA_INLINE void prefetch_pair(u32 h) noexcept {
    u32* b = pair_of(h);
    ULTRA_PREFETCH(b);
    ULTRA_PREFETCH(b + 16);
  }

  // Two-way set-associative bucket lookup; replaces the less used way.
  ULTRA_INLINE u32* find(u32 h) noexcept {
    u32* b = pair_of(h);
    u32 tag = h & 0xFFFF;
    if (!tag) tag = 1;
    if ((b[0] & 0xFFFF) == tag) return b;
    if ((b[16] & 0xFFFF) == tag) return b + 16;
    u32* v = ((b[1] ^ kFlip) & 0xFFFF) <= ((b[17] ^ kFlip) & 0xFFFF) ? b : b + 16;
    std::memset(v, 0, 64);
    v[0] = tag;
    return v;
  }

  ULTRA_INLINE void lookup_nibble() noexcept {
    pending_ = false;
    if (bit_ == 0) {
      for (unsigned i = 0; i < kH; ++i) bkt_[i] = find(h_[i]);
    } else {
      for (unsigned i = 0; i < kH; ++i) bkt_[i] = find(nibble_hash(h_[i], c0_));
    }
  }

  // One bit before the nibble boundary: prefetch both possible buckets.
  ULTRA_INLINE void prefetch_second_nibble() noexcept {
    const u32 a = c0_ << 1, b = a | 1;
    for (unsigned i = 0; i < kH; ++i) {
      prefetch_pair(nibble_hash(h_[i], a));
      prefetch_pair(nibble_hash(h_[i], b));
    }
  }

  // The byte-level state after appending byte c (pure: used both for the
  // real byte and, one bit early, for the two bytes it can still be).
  struct Next {
    ByteState bs;
    std::size_t line_start, prev_line;
  };
  ULTRA_INLINE Next advance(u32 c) const noexcept {
    Next n{bs_, line_start_, prev_line_};
    n.bs.c8 = (bs_.c8 << 8) | (bs_.c4 >> 24);
    n.bs.c4 = (bs_.c4 << 8) | c;
    const u32 lc = c | 32;
    if (lc >= 'a' && lc <= 'z') {
      n.bs.w0 = (bs_.w0 + lc) * 0x100000001B3ull;
    } else if (bs_.w0) {
      n.bs.w1 = bs_.w0;
      n.bs.w0 = 0;
    }
    const std::size_t p1 = pos_ + 1;
    if (c == '\n') {
      n.prev_line = line_start_;
      n.line_start = p1;
    }
    n.bs.col = static_cast<u32>(p1 - n.line_start);
    const std::size_t a = n.prev_line + n.bs.col;
    n.bs.above = a < n.line_start ? (a == pos_ ? c : hist_[a]) : 0;
    return n;
  }

  // Match-table slot for the last minlen bytes once the next byte is appended.
  ULTRA_INLINE std::size_t match_slot(unsigned m, const ByteState& b) const noexcept {
    const unsigned minlen = P::kMatchMin[m];
    u64 lo = (static_cast<u64>(b.c8) << 32) | b.c4;
    if (minlen < 8) lo &= (u64{1} << (8 * minlen)) - 1;
    u64 h = (lo + minlen) * 0x9E3779B97F4A7C15ull;
    if (minlen > 8 && pos_ + 1 >= minlen) {
      u64 older = load64(hist_.data() + pos_ + 1 - minlen);  // bytes before the newest 8
      if (minlen < 16) older &= (u64{1} << (8 * (minlen - 8))) - 1;
      h ^= older * 0xBF58476D1CE4E5B9ull;
    }
    return hash32(h) >> (32 - P::kMatchLog2);
  }

  // One bit before the byte boundary: compute the context hashes and match
  // slots for both possible bytes and prefetch their table lines.
  ULTRA_INLINE void prepare_next_byte() noexcept {
    for (unsigned y = 0; y < 2; ++y) {
      const u32 c = ((c0_ << 1) | y) & 0xFF;
      next_[y] = advance(c);
      u64 ctx[kH];
      P::contexts(next_[y].bs, ctx);
      for (unsigned i = 0; i < kH; ++i) {
        hc_[y][i] = hash32(ctx[i] + (static_cast<u64>(i + 1) << 58));
        prefetch_pair(hc_[y][i]);
      }
      for (unsigned m = 0; m < kM; ++m) {
        ms_[y][m] = match_slot(m, next_[y].bs);
        ULTRA_PREFETCH(&mt_[m][ms_[y][m]]);
      }
    }
  }

  void byte_update() {
    const u32 c = c0_ & 0xFF;
    const unsigned y = c & 1;
    hist_[pos_++] = static_cast<u8>(c);
    bs_ = next_[y].bs;
    line_start_ = next_[y].line_start;
    prev_line_ = next_[y].prev_line;
    for (unsigned m = 0; m < kM; ++m) update_match(m, ms_[y][m]);
    for (unsigned i = 0; i < kH; ++i) h_[i] = hc_[y][i];
    c0_ = 1;
    bit_ = 0;
    nib_ = 1;
    pending_ = true;
  }

  void update_match(unsigned m, std::size_t slot) {
    const unsigned minlen = P::kMatchMin[m];
    if (mlen_[m] > 0) {  // still matching (mismatching bits zeroed it already)
      if (mlen_[m] < 65535) ++mlen_[m];
      ++mptr_[m];
    }
    if (pos_ < minlen) return;
    if (mlen_[m] == 0) {
      const u32 cand = mt_[m][slot];
      if (cand > 0) {
        u32 len = 0;
        while (len < 32 && len < cand && hist_[cand - 1 - len] == hist_[pos_ - 1 - len]) ++len;
        if (len >= minlen) {
          mlen_[m] = len;
          mptr_[m] = cand;
        }
      }
    }
    mt_[m][slot] = static_cast<u32>(pos_);
  }

  std::size_t pairs_;
  BigArray<u32> table_;
  BigArray<u32> o1_;
  BigArray<u32> mt_[kM];
  std::vector<u8> hist_;
  std::size_t pos_ = 0, line_start_ = 0, prev_line_ = 0;
  bool initialized_ = false;
  bool pending_ = true;
  bool skip_ = false;
  bool apm_used_ = false;

  ByteState bs_;
  Next next_[2] = {};
  u32 hc_[2][kH] = {};
  std::size_t ms_[2][kM] = {};
  u32 h_[kH] = {};
  u32* bkt_[kH] = {};
  u32 c0_ = 1, nib_ = 1;
  unsigned bit_ = 0;
  u32* o1cur_ = nullptr;

  u32 cmap_[(kH + 1) * 8 * 64] = {};
  u32* ccur_[kH + 1] = {};
  u32 mlen_[kM] = {};
  std::size_t mptr_[kM] = {};
  u32 mctr_[kM][64] = {};
  u32* mcur_[kM] = {};

  std::vector<MixerRow<kInputs>> w_[kMixers];
  std::vector<MixerRow<8>> fw_;
  short* row_[kMixers] = {};
  short* frow_ = nullptr;
  alignas(16) short x_[kInputs] = {};
  alignas(16) short fx_[8] = {};
  int st_[kMixers] = {};
  int pm_[kMixers] = {};
  int pmix_ = 2048, pr_ = 2048;
  Apm a1_, a2_;
};

}  // namespace ultracodec::detail
