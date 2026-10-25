// ncm_simd.hpp - the mixer kernels as inline policies, one per instruction set.
//
// The codecs instantiate their bit loop once per policy, so the kernels inline
// into it instead of being called through a pointer. ncm_kernels.cpp wraps the
// same policies in the function table that tests and diagnostics use. Every
// policy computes exactly the integers of Scalar (R15), as documented in
// ncm_kernels.hpp; tests check them lane by lane.
//
// Layer 1: mix() and train() over kSets weight sets of kLanes inputs.
// Layer 2: dot4() and train4() over one set of four weights, with the inputs
// (layer-1 outputs, |t| <= 2047) in t[0..3]. train4() is train()'s lane rule:
//   f[i] = sat16(f[i] + (((((t[i] << 4) * e) >> 16) + 1) >> 1)), |e| <= 32767.
//
// Avx2 and Avx512 functions carry target attributes; a caller must itself be
// compiled for that target (see HCE_NCM_TARGET_*), and run only where
// simd_available() says the path exists.
#pragma once

#include "codec/ncm_kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__)
#include <immintrin.h>
#define HCE_NCM_X86 1
#define HCE_NCM_TARGET_AVX2 __attribute__((target("avx2")))
#define HCE_NCM_TARGET_AVX512 __attribute__((target("avx2,avx512f,avx512bw,avx512vl")))
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define HCE_NCM_NEON 1
#endif

namespace hce::ncm::simd {

using i16 = std::int16_t;
using i32 = std::int32_t;

// The sixteen layer-1 inputs as one value, so the bit loop can assemble them
// in registers: an array written lane by lane and read back as a vector would
// stall on store forwarding on the critical path of every bit.
typedef i16 Lanes __attribute__((vector_size(2 * kLanes)));

constexpr i16 sat16(i32 v) noexcept { return static_cast<i16>(std::clamp(v, -32768, 32767)); }
constexpr i32 lane_step(i32 x, i32 e) noexcept { return (((x * 16 * e) >> 16) + 1) >> 1; }

struct Scalar {
  static constexpr SimdPath kPath = SimdPath::scalar;
  static void mix(const Lanes& x, i16* const* w, i32* out) noexcept {
    for (int j = 0; j < kSets; ++j) {
      i32 s = 0;
      for (int i = 0; i < kLanes; ++i) s += i32{x[i]} * i32{w[j][i]};
      out[j] = s;
    }
  }
  static void mix(const i16* x, i16* const* w, i32* out) noexcept {
    for (int j = 0; j < kSets; ++j) {
      i32 s = 0;
      for (int i = 0; i < kLanes; ++i) s += i32{x[i]} * i32{w[j][i]};
      out[j] = s;
    }
  }
  static void train(const i16* x, i16* const* w, const i32* e) noexcept {
    for (int j = 0; j < kSets; ++j)
      for (int i = 0; i < kLanes; ++i) w[j][i] = sat16(i32{w[j][i]} + lane_step(x[i], e[j]));
  }
  static i32 dot4(const i16* t, const i16* f) noexcept {
    return i32{t[0]} * f[0] + i32{t[1]} * f[1] + i32{t[2]} * f[2] + i32{t[3]} * f[3];
  }
  static void train4(const i16* t, i16* f, i32 e) noexcept {
    for (int i = 0; i < 4; ++i) f[i] = sat16(i32{f[i]} + lane_step(t[i], e));
  }
};

#if HCE_NCM_X86
// Lanes 0, 1 and 2 of [sum a, sum b, sum c, sum c], the int32 lane sums of a, b, c.
inline __m128i hsum3(__m128i a, __m128i b, __m128i c) noexcept {
  const __m128i ab = _mm_add_epi32(_mm_unpacklo_epi32(a, b), _mm_unpackhi_epi32(a, b));  // a02 b02 a13 b13
  const __m128i cc = _mm_add_epi32(_mm_unpacklo_epi32(c, c), _mm_unpackhi_epi32(c, c));  // c02 c02 c13 c13
  return _mm_add_epi32(_mm_unpacklo_epi64(ab, cc), _mm_unpackhi_epi64(ab, cc));
}
inline void store3(i32* out, __m128i s) noexcept {
  out[0] = _mm_cvtsi128_si32(s);
  out[1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(s, 0x55));
  out[2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(s, 0xAA));
}
// The rounded high-half step of train() on eight lanes: ((mulhi(x << 4, e) + 1) >> 1).
inline __m128i step8(__m128i xs, __m128i ev) noexcept {
  return _mm_srai_epi16(_mm_add_epi16(_mm_mulhi_epi16(xs, ev), _mm_set1_epi16(1)), 1);
}
inline i32 dot4_sse2(const i16* t, const i16* f) noexcept {
  const __m128i p = _mm_madd_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(t)),
                                   _mm_loadl_epi64(reinterpret_cast<const __m128i*>(f)));
  return _mm_cvtsi128_si32(_mm_add_epi32(p, _mm_shuffle_epi32(p, 0x55)));
}
inline void train4_sse2(const i16* t, i16* f, i32 e) noexcept {
  const __m128i ts = _mm_slli_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(t)), 4);
  auto* fv = reinterpret_cast<__m128i*>(f);
  _mm_storel_epi64(fv, _mm_adds_epi16(_mm_loadl_epi64(fv), step8(ts, _mm_set1_epi16(static_cast<short>(e)))));
}

struct Sse2 {
  static constexpr SimdPath kPath = SimdPath::sse2;
  static void mix(const Lanes& x, i16* const* w, i32* out) noexcept {
    const __m128i x0 = reinterpret_cast<__m128i>(__builtin_shufflevector(x, x, 0, 1, 2, 3, 4, 5, 6, 7));
    const __m128i x1 = reinterpret_cast<__m128i>(__builtin_shufflevector(x, x, 8, 9, 10, 11, 12, 13, 14, 15));
    __m128i p[kSets];
    for (int j = 0; j < kSets; ++j) {
      const auto* wv = reinterpret_cast<const __m128i*>(w[j]);
      p[j] = _mm_add_epi32(_mm_madd_epi16(x0, _mm_loadu_si128(wv)), _mm_madd_epi16(x1, _mm_loadu_si128(wv + 1)));
    }
    store3(out, hsum3(p[0], p[1], p[2]));
  }
  static void mix(const i16* x, i16* const* w, i32* out) noexcept {
    const auto* xv = reinterpret_cast<const __m128i*>(x);
    const __m128i x0 = _mm_loadu_si128(xv), x1 = _mm_loadu_si128(xv + 1);
    __m128i p[kSets];
    for (int j = 0; j < kSets; ++j) {
      const auto* wv = reinterpret_cast<const __m128i*>(w[j]);
      p[j] = _mm_add_epi32(_mm_madd_epi16(x0, _mm_loadu_si128(wv)), _mm_madd_epi16(x1, _mm_loadu_si128(wv + 1)));
    }
    store3(out, hsum3(p[0], p[1], p[2]));
  }
  static void train(const i16* x, i16* const* w, const i32* e) noexcept {
    const auto* xv = reinterpret_cast<const __m128i*>(x);
    const __m128i xs0 = _mm_slli_epi16(_mm_loadu_si128(xv), 4), xs1 = _mm_slli_epi16(_mm_loadu_si128(xv + 1), 4);
    for (int j = 0; j < kSets; ++j) {
      const __m128i ev = _mm_set1_epi16(static_cast<short>(e[j]));
      auto* wv = reinterpret_cast<__m128i*>(w[j]);
      _mm_storeu_si128(wv, _mm_adds_epi16(_mm_loadu_si128(wv), step8(xs0, ev)));
      _mm_storeu_si128(wv + 1, _mm_adds_epi16(_mm_loadu_si128(wv + 1), step8(xs1, ev)));
    }
  }
  static i32 dot4(const i16* t, const i16* f) noexcept { return dot4_sse2(t, f); }
  static void train4(const i16* t, i16* f, i32 e) noexcept { train4_sse2(t, f, e); }
};

struct Avx2 {
  static constexpr SimdPath kPath = SimdPath::avx2;
  HCE_NCM_TARGET_AVX2 static void mix(const Lanes& x, i16* const* w, i32* out) noexcept {
    const __m256i xv = reinterpret_cast<__m256i>(x);
    __m128i p[kSets];
    for (int j = 0; j < kSets; ++j) {
      const __m256i m = _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[j])));
      p[j] = _mm_add_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
    }
    store3(out, hsum3(p[0], p[1], p[2]));
  }
  HCE_NCM_TARGET_AVX2 static void mix(const i16* x, i16* const* w, i32* out) noexcept {
    const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    __m128i p[kSets];
    for (int j = 0; j < kSets; ++j) {
      const __m256i m = _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[j])));
      p[j] = _mm_add_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
    }
    store3(out, hsum3(p[0], p[1], p[2]));
  }
  HCE_NCM_TARGET_AVX2 static void train(const i16* x, i16* const* w, const i32* e) noexcept {
    const __m256i xs = _mm256_slli_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)), 4);
    const __m256i one = _mm256_set1_epi16(1);
    for (int j = 0; j < kSets; ++j) {
      auto* wv = reinterpret_cast<__m256i*>(w[j]);
      const __m256i hi = _mm256_mulhi_epi16(xs, _mm256_set1_epi16(static_cast<short>(e[j])));
      const __m256i d = _mm256_srai_epi16(_mm256_add_epi16(hi, one), 1);
      _mm256_storeu_si256(wv, _mm256_adds_epi16(_mm256_loadu_si256(wv), d));
    }
  }
  static i32 dot4(const i16* t, const i16* f) noexcept { return dot4_sse2(t, f); }
  static void train4(const i16* t, i16* f, i32 e) noexcept { train4_sse2(t, f, e); }
};

// AVX-512BW/VL: the AVX2 kernels, compiled into loops built for AVX-512
// (ncm_run.cpp), so they get EVEX encodings, one-uop broadcasts and 32 vector
// registers, but stay 256 bits wide: no 512-bit frequency licence. Each bit
// mixes one 16-lane vector per set on a serial chain, so 512-bit registers
// have nothing to fill: pairing sets in them measured 5-10% slower.
struct Avx512 : Avx2 {
  static constexpr SimdPath kPath = SimdPath::avx512;
};
#endif

#if HCE_NCM_NEON
// High halves of the 32-bit products of eight lanes and e: an arithmetic shift, then an exact narrow.
inline int16x8_t step8(int16x8_t xs, int16x4_t ev) noexcept {
  const int16x8_t hi = vcombine_s16(vshrn_n_s32(vmull_s16(vget_low_s16(xs), ev), 16),
                                    vshrn_n_s32(vmull_s16(vget_high_s16(xs), ev), 16));
  return vshrq_n_s16(vaddq_s16(hi, vdupq_n_s16(1)), 1);
}

struct Neon {
  static constexpr SimdPath kPath = SimdPath::neon;
  static void mix(const Lanes& x, i16* const* w, i32* out) noexcept {
    mix2(vreinterpretq_s16_s64(reinterpret_cast<int64x2_t>(__builtin_shufflevector(x, x, 0, 1, 2, 3, 4, 5, 6, 7))),
         vreinterpretq_s16_s64(reinterpret_cast<int64x2_t>(__builtin_shufflevector(x, x, 8, 9, 10, 11, 12, 13, 14, 15))),
         w, out);
  }
  static void mix(const i16* x, i16* const* w, i32* out) noexcept { mix2(vld1q_s16(x), vld1q_s16(x + 8), w, out); }
  static void mix2(int16x8_t x0, int16x8_t x1, i16* const* w, i32* out) noexcept {
    for (int j = 0; j < kSets; ++j) {
      const int16x8_t w0 = vld1q_s16(w[j]), w1 = vld1q_s16(w[j] + 8);
      int32x4_t acc = vmull_s16(vget_low_s16(x0), vget_low_s16(w0));
      acc = vmlal_s16(acc, vget_high_s16(x0), vget_high_s16(w0));
      acc = vmlal_s16(acc, vget_low_s16(x1), vget_low_s16(w1));
      acc = vmlal_s16(acc, vget_high_s16(x1), vget_high_s16(w1));
      out[j] = vaddvq_s32(acc);
    }
  }
  static void train(const i16* x, i16* const* w, const i32* e) noexcept {
    const int16x8_t xs[2] = {vshlq_n_s16(vld1q_s16(x), 4), vshlq_n_s16(vld1q_s16(x + 8), 4)};
    for (int j = 0; j < kSets; ++j) {
      const int16x4_t ev = vdup_n_s16(static_cast<i16>(e[j]));
      for (int h = 0; h < 2; ++h) vst1q_s16(w[j] + 8 * h, vqaddq_s16(vld1q_s16(w[j] + 8 * h), step8(xs[h], ev)));
    }
  }
  static i32 dot4(const i16* t, const i16* f) noexcept { return vaddvq_s32(vmull_s16(vld1_s16(t), vld1_s16(f))); }
  static void train4(const i16* t, i16* f, i32 e) noexcept {
    const int16x4_t ts = vshl_n_s16(vld1_s16(t), 4);
    const int16x4_t hi = vshrn_n_s32(vmull_s16(ts, vdup_n_s16(static_cast<i16>(e))), 16);
    vst1_s16(f, vqadd_s16(vld1_s16(f), vshr_n_s16(vadd_s16(hi, vdup_n_s16(1)), 1)));
  }
};
#endif

}  // namespace hce::ncm::simd
