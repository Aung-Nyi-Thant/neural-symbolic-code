// Adaptive map from a bit-history state to a probability.
#pragma once

#include <array>
#include <cstdint>

#include "core/kernels/arith.hpp"
#include "core/kernels/tables.hpp"

namespace hce {

// Entry = p22 << 10 | n: p22 is P(bit = 1) with 22 bits, n counts updates (saturating at `limit`,
// at most 1023). Each update moves p22 toward the observed bit by (target - p22) / (n + 1.5).
class StateMap {
public:
    StateMap() = default;
    StateMap(const Tables& t, uint32_t limit) : limit_(limit) {
        for (size_t s = 0; s < entries_.size(); ++s) entries_[s] = t.p22_init[s] << 10;
    }

    // 16-bit probability of a 1 for this state, in [0, 65535].
    uint32_t p16(uint8_t state) const noexcept { return entries_[state] >> 16; }

    void update(uint8_t state, uint32_t bit, const Tables& t) noexcept {
        uint32_t& e = entries_[state];
        const uint32_t n = e & 1023u;
        const int64_t p22 = e >> 10;
        const int64_t target = bit ? (int64_t{1} << 22) - 1 : 0;
        // |target - p22| < 2^22 and recip <= 43690, so the product stays below 2^38. Because the
        // rate is below 1, the new value lies between p22 and target: always within 22 bits.
        const int64_t delta = arith::round_shift64(arith::mul64(target - p22, t.recip[n]), 16);
        const auto p = static_cast<uint32_t>(p22 + delta);
        e = (p << 10) | (n < limit_ ? n + 1 : limit_);
    }

private:
    std::array<uint32_t, 256> entries_{};
    uint32_t limit_ = 0;
};

}  // namespace hce
