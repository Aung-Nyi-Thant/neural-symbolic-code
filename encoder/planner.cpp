#include "encoder/planner.hpp"

#include <algorithm>
#include <bit>
#include <cassert>

#include "core/kernels/no_float.hpp"

namespace hce {

uint32_t bit_cost_q8(uint32_t probability) noexcept {
    assert(probability >= 1 && probability <= 65535);
    const auto exponent = static_cast<uint32_t>(std::bit_width(probability) - 1);
    uint64_t x = uint64_t{probability} << (31 - exponent);
    uint32_t fraction = 0;
    for (unsigned i = 0; i < 8; ++i) {
        x = (x * x) >> 31;
        fraction <<= 1;
        if (x >= (uint64_t{1} << 32)) {
            x >>= 1;
            fraction |= 1;
        }
    }
    return (16 - exponent) * 256 - fraction;
}

std::optional<CopyToken> CopyPlanner::choose(const AnchorHistory& history,
                                            std::span<const uint8_t> raw) const noexcept {
    const auto candidate = history.candidate();
    if (!candidate) return std::nullopt;
    const uint32_t copy_bits = 8u + static_cast<uint32_t>(std::bit_width(history.anchor_count() - 1)) +
                               kCopyDeltaBits + 32u;
    std::optional<CopyToken> best;
    // shortcut: search only 16 bytes before the anchor and at most 64 KiB ahead; widen after corpus tuning.
    for (uint32_t delta = 0; delta <= 16; ++delta) {
        const auto source = history.copy_source(candidate->anchor_id, delta, 1);
        if (!source) continue;
        const uint64_t limit = std::min({uint64_t{raw.size()}, history.position() - *source, uint64_t{65536}});
        uint32_t length = 0;
        // This comparison is the required absolute byte verification, never just a hash check.
        while (length < limit && raw[length] == history.byte_at(*source + length)) ++length;
        // The fixed flag costs 8 bits for COPY and 2/256 bits (rounded up) for a literal.
        const uint64_t literal_cost = uint64_t{length} * (average_q8_ + 2u);
        if (length >= 8 && literal_cost > uint64_t{copy_bits + 8u} * 256u &&
            (!best || length > best->length)) {
            best = CopyToken{candidate->anchor_id, delta, length};
        }
    }
    return best;
}

}  // namespace hce
