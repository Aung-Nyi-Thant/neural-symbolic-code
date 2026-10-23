// Integer helpers for the decode path.
//
// Signed arithmetic that could overflow goes through these helpers: they compute in the unsigned
// type and convert back, and C++20 defines both steps as modular. Overflow therefore wraps the same
// way on every platform instead of being undefined behaviour. Every call site is also bounded so
// that wrapping does not happen in practice; the bounds are noted at the call sites.
#pragma once

#include <cstdint>

namespace hce::arith {

constexpr int64_t add64(int64_t a, int64_t b) noexcept {
    return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

constexpr int64_t mul64(int64_t a, int64_t b) noexcept {
    return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

// Arithmetic shift right, i.e. floor(a / 2^s). Well-defined for negative a since C++20.
constexpr int64_t asr64(int64_t a, unsigned s) noexcept { return a >> s; }

// floor((a + 2^(s-1)) / 2^s): division by 2^s, rounding halves up. Requires s >= 1.
constexpr int64_t round_shift64(int64_t a, unsigned s) noexcept {
    return asr64(add64(a, int64_t{1} << (s - 1)), s);
}

constexpr int64_t clamp64(int64_t v, int64_t lo, int64_t hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace hce::arith
