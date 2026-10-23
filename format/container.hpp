// Container format v1: a fixed header followed by STORED or MODELED frames (see SPEC.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "format/byte_io.hpp"
#include "format/profiles.hpp"

namespace hce {

inline constexpr uint8_t kFormatVersion = 1;
inline constexpr size_t kHeaderSize = 28;
inline constexpr size_t kFrameHeaderSize = 13;

enum class FrameType : uint8_t { kStored = 0, kModeled = 1, kHybrid = 2 };

// Diagnostics for tests and analysis. `trace` is an FNV-1a hash over (p1, bit) of every modeled
// decision: equal encoder and decoder traces prove both sides computed identical probabilities.
struct CodecStats {
    uint64_t trace = 0xCBF29CE484222325ull;
    uint64_t decisions = 0;
    uint32_t frames_stored = 0;
    uint32_t frames_modeled = 0;
    uint64_t copies = 0;
    uint64_t copied_bytes = 0;
};

struct DecodeLimits {
    uint64_t max_output = uint64_t{1} << 34;  // refuse streams declaring more output than this
};

// Throws std::invalid_argument for an unknown profile id.
std::vector<uint8_t> compress(std::span<const uint8_t> input, uint8_t profile_id = kDefaultProfile,
                              CodecStats* stats = nullptr);

// Throws FormatError for any malformed, corrupted, truncated or unsupported stream. Never returns
// data that fails its checksums.
std::vector<uint8_t> decompress(std::span<const uint8_t> input, const DecodeLimits& limits = {},
                                CodecStats* stats = nullptr);

}  // namespace hce
