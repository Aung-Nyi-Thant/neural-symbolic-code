// Encoder-only COPY selection; the decoder never needs the cost estimate.
#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "core/kernels/anchor_history.hpp"

namespace hce {

struct CopyToken { uint32_t anchor_id; uint32_t delta; uint32_t length; };

// -log2(probability / 65536), Q8, calculated by integer repeated squaring.
uint32_t bit_cost_q8(uint32_t probability) noexcept;

class CopyPlanner {
public:
    std::optional<CopyToken> choose(const AnchorHistory& history, std::span<const uint8_t> raw) const noexcept;
    void observe_literal(uint32_t cost_q8) noexcept {
        average_q8_ = (average_q8_ * 31 + cost_q8) / 32;
    }
private:
    uint32_t average_q8_ = 8 * 256;
};

}  // namespace hce
