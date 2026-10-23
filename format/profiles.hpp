// Profiles pin everything that shapes the bitstream: model configuration, table size, frame size.
// A shipped profile is frozen; any change needs a new profile id (and new conformance vectors).
#pragma once

#include <cstdint>
#include <span>

#include "core/kernels/cm_model.hpp"

namespace hce {

struct Profile {
    uint8_t id;
    const char* name;
    bool modeled;  // false: every frame is STORED
    ModelConfig model;
    uint32_t frame_size;  // maximum raw bytes per frame
};

inline constexpr uint8_t kProfileStore = 0;
inline constexpr uint8_t kProfileCm6Small = 1;  // 1 MiB of tables, 64 KiB frames
inline constexpr uint8_t kProfileCm6 = 2;       // 64 MiB of tables, 1 MiB frames
inline constexpr uint8_t kDefaultProfile = kProfileCm6;
inline constexpr uint8_t kProfileHybridSmall = 3;  // cm6-1m + anchors over the whole stream (1 MiB index)
inline constexpr uint8_t kProfileHybrid = 4;       // cm6-64m + anchors over the whole stream (16 MiB index)

std::span<const Profile> profiles() noexcept;
const Profile* find_profile(uint8_t id) noexcept;

// The header's model id: model_fingerprint() for modeled profiles, a fixed tag CRC otherwise.
uint32_t model_id(const Profile& profile);

}  // namespace hce
