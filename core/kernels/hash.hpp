// 64-bit hashing for context selection: fixed constants, unsigned (wrapping) arithmetic only.
#pragma once

#include <cstdint>

namespace hce {

// SplitMix64 finalizer (Stafford's "Mix13" constants).
constexpr uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

}  // namespace hce
