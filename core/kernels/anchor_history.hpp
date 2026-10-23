// Content-defined Gear anchors over the whole stream, and the implicit match state built on them.
// Shared by encoder and decoder: every update is a pure function of the stream bytes.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/kernels/hash.hpp"

namespace hce {

inline constexpr uint32_t kCopyFlagP1 = 256;   // fixed 1/256 prior for the COPY token flag
inline constexpr unsigned kCopyDeltaBits = 5;  // COPY delta field width; the planner uses deltas 0..16

struct AnchorConfig {
    bool enabled = false;
    uint8_t index_bits = 16;   // 2^index_bits fingerprint buckets, each keeping its latest anchor
    uint8_t density_bits = 6;  // ~1 anchor per 2^density_bits bytes, at least 2^(density_bits-2) apart
    uint64_t gear_seed = 0x4843452D47454152ull;
};

constexpr std::array<uint64_t, 256> gear_table(uint64_t seed) noexcept {
    std::array<uint64_t, 256> table{};
    for (uint64_t& word : table) {
        seed += 0x9E3779B97F4A7C15ull;
        word = mix64(seed);
    }
    return table;
}

// Anchor ids are ordinals over the whole stream (id 0 is reserved and never resolves), so a COPY can
// reach any earlier anchor, however far back. History bytes are read from the attached stream.
class AnchorHistory {
public:
    struct Match { uint32_t anchor_id; uint64_t position; };
    explicit AnchorHistory(const AnchorConfig& config);

    // The stream that positions refer to. It must hold every byte before position(); the byte being
    // observed may still be missing, since observe() receives it directly.
    void attach(std::span<const uint8_t> stream) noexcept { stream_ = stream; }
    void observe(uint8_t byte) noexcept;
    uint64_t position() const noexcept { return position_; }
    uint8_t byte_at(uint64_t position) const noexcept;
    std::optional<Match> candidate() const noexcept { return candidate_; }
    uint32_t anchor_count() const noexcept { return static_cast<uint32_t>(anchors_->size()); }
    std::optional<uint64_t> resolve(uint32_t anchor_id) const noexcept;
    // COPY sources must be entirely in existing history; overlapping expansion is forbidden.
    std::optional<uint64_t> copy_source(uint32_t anchor_id, uint32_t delta,
                                        uint32_t length) const noexcept;
    std::optional<uint8_t> prediction() const noexcept;
    uint32_t match_length() const noexcept { return match_length_; }
    uint8_t divergence() const noexcept { return divergence_; }
    // Encoder rollback: forget anchors recorded after the first `count`.
    void truncate(uint32_t count) { anchors_->resize(count); }

private:
    struct Lookup { uint64_t fingerprint = 0; uint32_t id = 0; };  // id 0 = empty bucket
    AnchorConfig config_;
    std::array<uint64_t, 256> gear_;
    std::span<const uint8_t> stream_;
    std::vector<Lookup> lookup_;
    // Ordinal -> position. Copies share it: anchors depend only on the stream bytes, so rolling a
    // model back to a snapshot only needs truncate().
    std::shared_ptr<std::vector<uint64_t>> anchors_;
    uint64_t min_gap_ = 1;
    uint64_t position_ = 0;
    uint64_t hash_ = 0;
    std::optional<Match> candidate_;
    std::optional<uint64_t> predicted_position_;
    uint32_t match_length_ = 0;
    uint8_t divergence_ = 0;
};

}  // namespace hce
