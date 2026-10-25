// ncm_kernels.cpp - the kernel table (see ncm_kernels.hpp) over the inline
// policies of ncm_simd.hpp, and run-time selection of the path in use.
#include "codec/ncm_kernels.hpp"

#include "codec/ncm_simd.hpp"

#include <cstdlib>
#include <string_view>

namespace hce::ncm {
namespace {

template <class K>
constexpr Kernels table(const char* name) noexcept {
  return {K::mix, K::train, K::dot4, K::train4, K::kPath, name};
}

constexpr Kernels kScalar = table<simd::Scalar>("scalar");
#if HCE_NCM_X86
constexpr Kernels kSse2 = table<simd::Sse2>("sse2");
constexpr Kernels kAvx2 = table<simd::Avx2>("avx2");
constexpr Kernels kAvx512 = table<simd::Avx512>("avx512");
#endif
#if HCE_NCM_NEON
constexpr Kernels kNeon = table<simd::Neon>("neon");
#endif

// The default: AVX2 over AVX-512, whose wider registers do not pay off on
// sixteen-lane sets (and can lower the clock); AVX-512 is used when asked for.
const Kernels& choose() noexcept {
  if (const char* env = std::getenv("HCE_SIMD")) {
    const std::string_view want(env);
    for (const SimdPath p : kAllPaths)
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
    case SimdPath::avx512:
      __builtin_cpu_init();
      return __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("avx512f") != 0 &&
             __builtin_cpu_supports("avx512bw") != 0 && __builtin_cpu_supports("avx512vl") != 0;
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
    case SimdPath::avx512: return kAvx512;
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
