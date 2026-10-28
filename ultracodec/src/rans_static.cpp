#include "rans_static.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <atomic>

#if defined(__x86_64__)
#include <immintrin.h>
#define ULTRA_RANS_AVX2 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define ULTRA_RANS_NEON 1
#endif

namespace ultracodec::detail {
namespace {

constexpr unsigned kScaleBits = 12;
constexpr u32 kScale = 1u << kScaleBits;
constexpr u32 kL = 1u << 16;
constexpr unsigned kLanes = 8;

std::atomic<RansKernel> g_kernel{RansKernel::Auto};

// Scales counts to frequencies summing to kScale, every used symbol >= 1.
std::array<u32, 256> normalize(const std::array<u64, 256>& count, u64 total) {
  std::array<u32, 256> f{};
  u64 sum = 0;
  unsigned largest = 0;
  for (unsigned s = 0; s < 256; ++s) {
    if (!count[s]) continue;
    u64 v = (count[s] * kScale + total / 2) / total;
    f[s] = static_cast<u32>(std::max<u64>(v, 1));
    sum += f[s];
    if (count[s] > count[largest]) largest = s;
  }
  // Fix the sum: take from / give to the symbols where it costs least.
  while (sum > kScale) {
    unsigned best = 256;
    for (unsigned s = 0; s < 256; ++s)
      if (f[s] > 1 && (best == 256 || f[s] * count[best] > f[best] * count[s])) best = s;  // max f/count
    --f[best];
    --sum;
  }
  f[largest] += static_cast<u32>(kScale - sum);
  return f;
}

struct DecodeTable {
  alignas(64) u32 e[kScale];  // start | (freq << 12) | (sym << 24)
};

ULTRA_INLINE void scalar_step(const u32* T, u32& x, u8& sym, const u8*& wp, const u8* wend) {
  u32 e = T[x & (kScale - 1)];
  sym = static_cast<u8>(e >> 24);
  x = ((e >> 12) & 0xFFF) * (x >> kScaleBits) + (x & (kScale - 1)) - (e & 0xFFF);
  if (x < kL) {
    if (wend - wp < 2) throw Error("ultracodec: truncated rANS stream");
    x = (x << 16) | load16(wp);
    wp += 2;
  }
}

#if defined(ULTRA_RANS_AVX2)
struct PermTable {
  alignas(32) u32 idx[256][8];
};
constexpr PermTable make_perm() {
  PermTable t{};
  for (unsigned m = 0; m < 256; ++m) {
    unsigned rank = 0;
    for (unsigned j = 0; j < 8; ++j) t.idx[m][j] = (m >> j & 1) ? rank++ : 0;
  }
  return t;
}
constexpr PermTable kPerm = make_perm();

const bool kHaveAvx2 = __builtin_cpu_supports("avx2");

// Decodes whole 8-symbol groups while 16 stream bytes are readable.
__attribute__((target("avx2"))) std::size_t decode_avx2(const u32* T, u32* xs, const u8*& wp, const u8* wend,
                                                         u8* out, std::size_t groups) {
  __m256i X = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs));
  const __m256i mask12 = _mm256_set1_epi32(0xFFF);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i pick3 = _mm256_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                         3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  std::size_t g = 0;
  for (; g < groups && wend - wp >= 16; ++g) {
    __m256i slot = _mm256_and_si256(X, mask12);
    __m256i e = _mm256_i32gather_epi32(reinterpret_cast<const int*>(T), slot, 4);
    __m256i freq = _mm256_and_si256(_mm256_srli_epi32(e, 12), mask12);
    __m256i start = _mm256_and_si256(e, mask12);
    X = _mm256_add_epi32(_mm256_mullo_epi32(freq, _mm256_srli_epi32(X, 12)), _mm256_sub_epi32(slot, start));
    __m256i s = _mm256_shuffle_epi8(e, pick3);
    store32(out + g * 8, static_cast<u32>(_mm256_extract_epi32(s, 0)));
    store32(out + g * 8 + 4, static_cast<u32>(_mm256_extract_epi32(s, 4)));
    __m256i need = _mm256_cmpeq_epi32(_mm256_srli_epi32(X, 16), zero);
    unsigned m = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(need)));
    __m256i words = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp)));
    __m256i w = _mm256_permutevar8x32_epi32(words, _mm256_load_si256(reinterpret_cast<const __m256i*>(kPerm.idx[m])));
    X = _mm256_blendv_epi8(X, _mm256_or_si256(_mm256_slli_epi32(X, 16), w), need);
    wp += 2 * static_cast<unsigned>(__builtin_popcount(m));
  }
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(xs), X);
  return g;
}
#endif

#if defined(ULTRA_RANS_NEON)
struct NeonPerm {
  alignas(16) u8 idx[16][16];
};
constexpr NeonPerm make_neon_perm() {
  NeonPerm t{};
  for (unsigned m = 0; m < 16; ++m) {
    unsigned rank = 0;
    for (unsigned j = 0; j < 4; ++j) {
      bool on = (m >> j) & 1;
      for (unsigned b = 0; b < 4; ++b) t.idx[m][j * 4 + b] = on ? static_cast<u8>(rank * 4 + b) : 0xFF;
      if (on) ++rank;
    }
  }
  return t;
}
constexpr NeonPerm kNeonPerm = make_neon_perm();

ULTRA_INLINE uint32x4_t neon_half(const u32* T, uint32x4_t X, const u8*& wp, u8* out) {
  const uint32x4_t mask12 = vdupq_n_u32(0xFFF);
  uint32x4_t slot = vandq_u32(X, mask12);
  alignas(16) u32 s[4];
  vst1q_u32(s, slot);
  alignas(16) u32 ev[4] = {T[s[0]], T[s[1]], T[s[2]], T[s[3]]};
  uint32x4_t e = vld1q_u32(ev);
  uint32x4_t freq = vandq_u32(vshrq_n_u32(e, 12), mask12);
  uint32x4_t start = vandq_u32(e, mask12);
  X = vaddq_u32(vmulq_u32(freq, vshrq_n_u32(X, 12)), vsubq_u32(slot, start));
  out[0] = static_cast<u8>(ev[0] >> 24);
  out[1] = static_cast<u8>(ev[1] >> 24);
  out[2] = static_cast<u8>(ev[2] >> 24);
  out[3] = static_cast<u8>(ev[3] >> 24);
  uint32x4_t need = vceqq_u32(vshrq_n_u32(X, 16), vdupq_n_u32(0));
  const uint32x4_t bits = {1, 2, 4, 8};
  unsigned m = vaddvq_u32(vandq_u32(need, bits));
  uint32x4_t words = vmovl_u16(vld1_u16(reinterpret_cast<const uint16_t*>(wp)));
  uint8x16_t wb = vqtbl1q_u8(vreinterpretq_u8_u32(words), vld1q_u8(kNeonPerm.idx[m]));
  uint32x4_t w = vreinterpretq_u32_u8(wb);
  X = vbslq_u32(need, vorrq_u32(vshlq_n_u32(X, 16), w), X);
  wp += 2 * static_cast<unsigned>(__builtin_popcount(m));
  return X;
}

std::size_t decode_neon(const u32* T, u32* xs, const u8*& wp, const u8* wend, u8* out, std::size_t groups) {
  uint32x4_t A = vld1q_u32(xs), B = vld1q_u32(xs + 4);
  std::size_t g = 0;
  for (; g < groups && wend - wp >= 16; ++g) {
    A = neon_half(T, A, wp, out + g * 8);
    B = neon_half(T, B, wp, out + g * 8 + 4);
  }
  vst1q_u32(xs, A);
  vst1q_u32(xs + 4, B);
  return g;
}
#endif

}  // namespace

void rans_set_kernel(RansKernel k) noexcept { g_kernel.store(k, std::memory_order_relaxed); }

RansKernel rans_active_kernel() noexcept {
  RansKernel k = g_kernel.load(std::memory_order_relaxed);
#if defined(ULTRA_RANS_AVX2)
  if (k == RansKernel::Auto) return kHaveAvx2 ? RansKernel::Avx2 : RansKernel::Scalar;
  if (k == RansKernel::Avx2 && !kHaveAvx2) return RansKernel::Scalar;
  if (k == RansKernel::Neon) return RansKernel::Scalar;
#elif defined(ULTRA_RANS_NEON)
  if (k == RansKernel::Auto) return RansKernel::Neon;
  if (k == RansKernel::Avx2) return RansKernel::Scalar;
#else
  if (k != RansKernel::Scalar) return RansKernel::Scalar;
#endif
  return k;
}

void rans_encode(std::span<const u8> sym, std::vector<u8>& out) {
  const std::size_t n = sym.size();
  if (n > UINT32_MAX) throw Error("ultracodec: rANS stream too long");
  put_varint(out, static_cast<u32>(n));
  std::array<u64, 256> count{};
  {
    // Four sub-histograms break the store-to-load chain on runs of one symbol.
    std::array<std::array<u32, 256>, 4> part{};
    std::size_t i = 0;
    for (; i + 4 <= n && i < (std::size_t{1} << 31); i += 4) {
      ++part[0][sym[i]];
      ++part[1][sym[i + 1]];
      ++part[2][sym[i + 2]];
      ++part[3][sym[i + 3]];
    }
    for (unsigned s = 0; s < 256; ++s) count[s] = u64{part[0][s]} + part[1][s] + part[2][s] + part[3][s];
    for (; i < n; ++i) ++count[sym[i]];
  }
  unsigned distinct = 0, max_sym = 0;
  for (unsigned s = 0; s < 256; ++s)
    if (count[s]) { ++distinct; max_sym = s; }
  if (n < 32 || distinct == 0) {
    out.push_back(0);
    out.insert(out.end(), sym.begin(), sym.end());
    return;
  }
  if (distinct == 1) {
    out.push_back(1);
    out.push_back(static_cast<u8>(max_sym));
    return;
  }
  const auto f = normalize(count, n);
  std::array<u32, 256> start{};
  for (unsigned s = 1; s < 256; ++s) start[s] = start[s - 1] + f[s - 1];

  // Words are produced back to front.
  const std::size_t buf_size = n + 2 * kLanes + 1;
  const std::unique_ptr<u16[]> buf(new u16[buf_size]);  // written back to front, read only from pos
  std::size_t pos = buf_size;
  u32 x[kLanes];
  std::fill(x, x + kLanes, kL);
  for (std::size_t i = n; i-- > 0;) {
    u32& xs = x[i & (kLanes - 1)];
    const u8 c = sym[i];
    const u32 fr = f[c];
    // Branch-free renormalization: the word is always stored, pos moves only when it counts.
    const u32 out16 = xs >= (fr << 20) ? 1u : 0u;
    buf[pos - 1] = static_cast<u16>(xs);
    pos -= out16;
    xs >>= out16 * 16;
    xs = ((xs / fr) << kScaleBits) + (xs % fr) + start[c];
  }
  for (unsigned l = kLanes; l-- > 0;) {
    buf[--pos] = static_cast<u16>(x[l] >> 16);
    buf[--pos] = static_cast<u16>(x[l]);
  }
  const std::size_t words = buf_size - pos;
  std::vector<u8> table;
  table.push_back(static_cast<u8>(max_sym));
  for (unsigned s = 0; s <= max_sym; ++s) put_varint(table, f[s]);
  if (table.size() + 2 * words + 5 >= n) {
    out.push_back(0);
    out.insert(out.end(), sym.begin(), sym.end());
    return;
  }
  out.push_back(2);
  out.insert(out.end(), table.begin(), table.end());
  put_varint(out, static_cast<u32>(words));
  const std::size_t at = out.size();
  out.resize(at + 2 * words);
  std::memcpy(out.data() + at, buf.get() + pos, 2 * words);
}

void rans_decode(Reader& in, std::vector<u8>& out, std::size_t max_symbols) {
  const std::size_t n = in.varint();
  if (n > max_symbols) throw Error("ultracodec: rANS stream longer than allowed");
  const u8 mode = in.u8v();
  out.resize(n);
  if (mode == 0) {
    auto s = in.take(n);
    if (n) std::memcpy(out.data(), s.data(), n);
    return;
  }
  if (mode == 1) {
    std::fill(out.begin(), out.end(), in.u8v());
    return;
  }
  if (mode != 2) throw Error("ultracodec: bad rANS mode");
  const unsigned max_sym = in.u8v();
  thread_local DecodeTable table;
  u32 start = 0;
  for (unsigned s = 0; s <= max_sym; ++s) {
    u32 f = in.varint();
    if (f >= kScale || start + f > kScale) throw Error("ultracodec: bad rANS frequencies");
    for (u32 k = 0; k < f; ++k) table.e[start + k] = (start) | (f << 12) | (s << 24);
    start += f;
  }
  if (start != kScale) throw Error("ultracodec: rANS frequencies do not sum to 4096");
  const std::size_t words = in.varint();
  if (words < 2 * kLanes) throw Error("ultracodec: truncated rANS stream");
  auto data = in.take(2 * words);
  const u8* wp = data.data();
  const u8* const wend = wp + data.size();
  alignas(32) u32 x[kLanes];
  for (unsigned l = 0; l < kLanes; ++l) {
    x[l] = load32(wp);
    wp += 4;
  }
  const u32* T = table.e;
  std::size_t i = 0;
  const std::size_t groups = n / kLanes;
  const RansKernel k = rans_active_kernel();
#if defined(ULTRA_RANS_AVX2)
  if (k == RansKernel::Avx2) i = decode_avx2(T, x, wp, wend, out.data(), groups) * kLanes;
#endif
#if defined(ULTRA_RANS_NEON)
  if (k == RansKernel::Neon) i = decode_neon(T, x, wp, wend, out.data(), groups) * kLanes;
#endif
  (void)k;
  (void)groups;
  for (; i < n; ++i) scalar_step(T, x[i & (kLanes - 1)], out[i], wp, wend);
  for (unsigned l = 0; l < kLanes; ++l)
    if (x[l] != kL) throw Error("ultracodec: rANS final state mismatch (corrupt stream)");
  if (wp != wend) throw Error("ultracodec: trailing rANS data");
}

}  // namespace ultracodec::detail
