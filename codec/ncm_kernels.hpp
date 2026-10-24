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
using MixFn = void (*)(const std::int16_t* x, std::int16_t* const* w, std::int32_t* out) noexcept;
using TrainFn = void (*)(const std::int16_t* x, std::int16_t* const* w, const std::int32_t* e) noexcept;

enum class SimdPath : std::uint8_t { scalar, sse2, avx2, neon };

struct Kernels {
  MixFn mix;
  TrainFn train;
  SimdPath path;
  const char* name;
};

bool simd_available(SimdPath path) noexcept;
// Kernels for `path`; falls back to scalar when the path is not available here.
const Kernels& kernels_for(SimdPath path) noexcept;
// The kernels in use: the best available path, or the one named by the
// environment variable HCE_SIMD (scalar, sse2, avx2, neon) when it is available.
const Kernels& active_kernels() noexcept;

}  // namespace hce::ncm
