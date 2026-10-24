// ncm_kernels.cpp - scalar, SSE2, AVX2 and NEON implementations of the mixer
// kernels (see ncm_kernels.hpp). The scalar code is the reference; the others
// must produce identical results, which tests check lane by lane.
#include "codec/ncm_kernels.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string_view>

#if defined(__x86_64__)
#include <immintrin.h>
#define HCE_NCM_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define HCE_NCM_NEON 1
#endif

namespace hce::ncm {
namespace {

void mix_scalar(const std::int16_t* x, std::int16_t* const* w, std::int32_t* out) noexcept {
  for (int j = 0; j < kSets; ++j) {
    std::int32_t s = 0;
    for (int i = 0; i < kLanes; ++i) s += std::int32_t{x[i]} * std::int32_t{w[j][i]};
    out[j] = s;
  }
}

void train_scalar(const std::int16_t* x, std::int16_t* const* w, const std::int32_t* e) noexcept {
  for (int j = 0; j < kSets; ++j) {
    for (int i = 0; i < kLanes; ++i) {
      const std::int32_t d = (((std::int32_t{x[i]} * 16 * e[j]) >> 16) + 1) >> 1;
      w[j][i] = static_cast<std::int16_t>(std::clamp(std::int32_t{w[j][i]} + d, -32768, 32767));
    }
  }
}

#if HCE_NCM_X86
// Sums of the four int32 lanes of a, b and c: [sum a, sum b, sum c, sum c].
__m128i hsum3(__m128i a, __m128i b, __m128i c) noexcept {
  // Transpose-free reduction with SSE2 only: unpack pairs, then add.
  const __m128i ab_lo = _mm_unpacklo_epi32(a, b);  // a0 b0 a1 b1
  const __m128i ab_hi = _mm_unpackhi_epi32(a, b);  // a2 b2 a3 b3
  const __m128i ab = _mm_add_epi32(ab_lo, ab_hi);  // a02 b02 a13 b13
  const __m128i cc_lo = _mm_unpacklo_epi32(c, c);
  const __m128i cc_hi = _mm_unpackhi_epi32(c, c);
  const __m128i cc = _mm_add_epi32(cc_lo, cc_hi);  // c02 c02 c13 c13
  const __m128i lo = _mm_unpacklo_epi64(ab, cc);   // a02 b02 c02 c02
  const __m128i hi = _mm_unpackhi_epi64(ab, cc);   // a13 b13 c13 c13
  return _mm_add_epi32(lo, hi);
}

void store3(std::int32_t* out, __m128i s) noexcept {
  alignas(16) std::int32_t t[4];
  _mm_store_si128(reinterpret_cast<__m128i*>(t), s);
  out[0] = t[0];
  out[1] = t[1];
  out[2] = t[2];
}

void mix_sse2(const std::int16_t* x, std::int16_t* const* w, std::int32_t* out) noexcept {
  const auto* xv = reinterpret_cast<const __m128i*>(x);
  const __m128i x0 = _mm_loadu_si128(xv), x1 = _mm_loadu_si128(xv + 1);
  __m128i p[kSets];
  for (int j = 0; j < kSets; ++j) {
    const auto* wv = reinterpret_cast<const __m128i*>(w[j]);
    p[j] = _mm_add_epi32(_mm_madd_epi16(x0, _mm_loadu_si128(wv)), _mm_madd_epi16(x1, _mm_loadu_si128(wv + 1)));
  }
  store3(out, hsum3(p[0], p[1], p[2]));
}

void train_sse2(const std::int16_t* x, std::int16_t* const* w, const std::int32_t* e) noexcept {
  const auto* xv = reinterpret_cast<const __m128i*>(x);
  const __m128i xs0 = _mm_slli_epi16(_mm_loadu_si128(xv), 4), xs1 = _mm_slli_epi16(_mm_loadu_si128(xv + 1), 4);
  const __m128i one = _mm_set1_epi16(1);
  for (int j = 0; j < kSets; ++j) {
    const __m128i ev = _mm_set1_epi16(static_cast<short>(e[j]));
    auto* wv = reinterpret_cast<__m128i*>(w[j]);
    const __m128i d0 = _mm_srai_epi16(_mm_add_epi16(_mm_mulhi_epi16(xs0, ev), one), 1);
    const __m128i d1 = _mm_srai_epi16(_mm_add_epi16(_mm_mulhi_epi16(xs1, ev), one), 1);
    _mm_storeu_si128(wv, _mm_adds_epi16(_mm_loadu_si128(wv), d0));
    _mm_storeu_si128(wv + 1, _mm_adds_epi16(_mm_loadu_si128(wv + 1), d1));
  }
}

__attribute__((target("avx2"))) void mix_avx2(const std::int16_t* x, std::int16_t* const* w,
                                              std::int32_t* out) noexcept {
  const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
  __m128i p[kSets];
  for (int j = 0; j < kSets; ++j) {
    const __m256i m = _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[j])));
    p[j] = _mm_add_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
  }
  store3(out, hsum3(p[0], p[1], p[2]));
}

__attribute__((target("avx2"))) void train_avx2(const std::int16_t* x, std::int16_t* const* w,
                                                const std::int32_t* e) noexcept {
  const __m256i xs = _mm256_slli_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)), 4);
  const __m256i one = _mm256_set1_epi16(1);
  for (int j = 0; j < kSets; ++j) {
    auto* wv = reinterpret_cast<__m256i*>(w[j]);
    const __m256i hi = _mm256_mulhi_epi16(xs, _mm256_set1_epi16(static_cast<short>(e[j])));
    const __m256i d = _mm256_srai_epi16(_mm256_add_epi16(hi, one), 1);
    _mm256_storeu_si256(wv, _mm256_adds_epi16(_mm256_loadu_si256(wv), d));
  }
}
#endif

#if HCE_NCM_NEON
void mix_neon(const std::int16_t* x, std::int16_t* const* w, std::int32_t* out) noexcept {
  const int16x8_t x0 = vld1q_s16(x), x1 = vld1q_s16(x + 8);
  for (int j = 0; j < kSets; ++j) {
    const int16x8_t w0 = vld1q_s16(w[j]), w1 = vld1q_s16(w[j] + 8);
    int32x4_t acc = vmull_s16(vget_low_s16(x0), vget_low_s16(w0));
    acc = vmlal_s16(acc, vget_high_s16(x0), vget_high_s16(w0));
    acc = vmlal_s16(acc, vget_low_s16(x1), vget_low_s16(w1));
    acc = vmlal_s16(acc, vget_high_s16(x1), vget_high_s16(w1));
    out[j] = vaddvq_s32(acc);
  }
}

void train_neon(const std::int16_t* x, std::int16_t* const* w, const std::int32_t* e) noexcept {
  const int16x8_t xs[2] = {vshlq_n_s16(vld1q_s16(x), 4), vshlq_n_s16(vld1q_s16(x + 8), 4)};
  const int16x8_t one = vdupq_n_s16(1);
  for (int j = 0; j < kSets; ++j) {
    const int16x4_t ev = vdup_n_s16(static_cast<std::int16_t>(e[j]));
    for (int h = 0; h < 2; ++h) {
      // High halves of the 32-bit products: an arithmetic shift, then an exact narrow.
      const int16x8_t hi = vcombine_s16(vshrn_n_s32(vmull_s16(vget_low_s16(xs[h]), ev), 16),
                                        vshrn_n_s32(vmull_s16(vget_high_s16(xs[h]), ev), 16));
      const int16x8_t d = vshrq_n_s16(vaddq_s16(hi, one), 1);
      vst1q_s16(w[j] + 8 * h, vqaddq_s16(vld1q_s16(w[j] + 8 * h), d));
    }
  }
}
#endif

constexpr Kernels kScalar{mix_scalar, train_scalar, SimdPath::scalar, "scalar"};
#if HCE_NCM_X86
constexpr Kernels kSse2{mix_sse2, train_sse2, SimdPath::sse2, "sse2"};
constexpr Kernels kAvx2{mix_avx2, train_avx2, SimdPath::avx2, "avx2"};
#endif
#if HCE_NCM_NEON
constexpr Kernels kNeon{mix_neon, train_neon, SimdPath::neon, "neon"};
#endif

const Kernels& choose() noexcept {
  if (const char* env = std::getenv("HCE_SIMD")) {
    const std::string_view want(env);
    for (const SimdPath p : {SimdPath::scalar, SimdPath::sse2, SimdPath::avx2, SimdPath::neon})
      if (want == kernels_for(p).name && simd_available(p)) return kernels_for(p);
  }
  for (const SimdPath p : {SimdPath::avx2, SimdPath::neon, SimdPath::sse2})
    if (simd_available(p)) return kernels_for(p);
  return kScalar;
}

}  // namespace

bool simd_available(SimdPath path) noexcept {
  switch (path) {
    case SimdPath::scalar: return true;
#if HCE_NCM_X86
    case SimdPath::sse2: return true;  // part of x86-64
    case SimdPath::avx2:
      __builtin_cpu_init();
      return __builtin_cpu_supports("avx2") != 0;
#endif
#if HCE_NCM_NEON
    case SimdPath::neon: return true;  // part of AArch64
#endif
    default: return false;
  }
}

const Kernels& kernels_for(SimdPath path) noexcept {
  if (!simd_available(path)) return kScalar;
  switch (path) {
#if HCE_NCM_X86
    case SimdPath::sse2: return kSse2;
    case SimdPath::avx2: return kAvx2;
#endif
#if HCE_NCM_NEON
    case SimdPath::neon: return kNeon;
#endif
    default: return kScalar;
  }
}

const Kernels& active_kernels() noexcept {
  static const Kernels& chosen = choose();
  return chosen;
}

}  // namespace hce::ncm
