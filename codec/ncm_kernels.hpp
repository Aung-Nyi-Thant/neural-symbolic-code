// ncm_kernels.hpp - exact integer SIMD kernels of the neural CM mixer.
#pragma once

#include <cstdint>

namespace hce::ncm {

inline constexpr int kLanes = 16;  // inputs per mixer weight set (unused lanes are zero)
inline constexpr int kSets = 3;    // layer-1 weight sets evaluated per bit

// Layer 1 of the mixer: kSets weight sets over the same 16 inputs. Every
// implementation computes exactly the same integers (R15):
//   mix:   out[j] = sum of x[i] * w[j][i] in int32. Inputs satisfy |x| <= 2047
//          and weights are int16, so each product is below 2^26 and 16 of them
//          below 2^30: nothing overflows, and integer addition is associative,
//          so every reduction order (pmaddwd, phaddd, smlal, scalar) agrees.
//   train: w[j][i] = sat16(w[j][i] + (((((x[i] << 4) * e[j]) >> 16) + 1) >> 1))
//          lane by lane: a rounded high-half product and a saturating add, no
//          reduction. Requires |x| <= 2047 and |e[j]| <= 32767.
//   dot4, train4: layer 2, one set of four weights over t[0..3], |t| <= 2047:
//          the sum of t[i] * f[i], and train's lane rule with a single error.
// The codecs inline these kernels into their bit loops (ncm_simd.hpp); the
// table below wraps the same code for tests and diagnostics.
using MixFn = void (*)(const std::int16_t* x, std::int16_t* const* w, std::int32_t* out) noexcept;
using TrainFn = void (*)(const std::int16_t* x, std::int16_t* const* w, const std::int32_t* e) noexcept;
using Dot4Fn = std::int32_t (*)(const std::int16_t* t, const std::int16_t* f) noexcept;
using Train4Fn = void (*)(const std::int16_t* t, std::int16_t* f, std::int32_t e) noexcept;

enum class SimdPath : std::uint8_t { scalar, sse2, avx2, neon, avx512 };
// Every path, in the order tests and HCE_SIMD look them up.
inline constexpr SimdPath kAllPaths[] = {SimdPath::scalar, SimdPath::sse2, SimdPath::avx2, SimdPath::avx512,
                                         SimdPath::neon};

struct Kernels {
  MixFn mix;
  TrainFn train;
  Dot4Fn dot4;
  Train4Fn train4;
  SimdPath path;
  const char* name;
};

bool simd_available(SimdPath path) noexcept;
// Kernels for `path`; falls back to scalar when the path is not available here.
const Kernels& kernels_for(SimdPath path) noexcept;
// The kernels in use: the best available path, or the one named by the
// environment variable HCE_SIMD (scalar, sse2, avx2, avx512, neon) when it is available.
const Kernels& active_kernels() noexcept;

}  // namespace hce::ncm
