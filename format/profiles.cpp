#include "format/profiles.hpp"

#include <array>

#include "core/kernels/crc32c.hpp"
#include "core/kernels/no_float.hpp"

namespace hce {
namespace {

constexpr ModelConfig cm6(uint8_t table_bits) {
    ModelConfig c;
    c.orders = {0, 1, 2, 3, 4, 6};
    c.num_orders = 6;
    c.table_bits = table_bits;
    c.mixer_lr = 7;
    c.statemap_limit = 255;
    return c;
}

constexpr ModelConfig hybrid(uint8_t table_bits, uint8_t index_bits) {
    ModelConfig c = cm6(table_bits);
    c.anchors.enabled = true;
    c.anchors.index_bits = index_bits;
    return c;
}

constexpr std::array<Profile, 5> kProfiles{{
    {kProfileStore, "store", false, ModelConfig{}, uint32_t{1} << 20},
    {kProfileCm6Small, "cm6-1m", true, cm6(16), uint32_t{1} << 16},
    {kProfileCm6, "cm6-64m", true, cm6(22), uint32_t{1} << 20},
    {kProfileHybridSmall, "hybrid-1m", true, hybrid(16, 16), uint32_t{1} << 16},
    {kProfileHybrid, "hybrid-64m", true, hybrid(22, 20), uint32_t{1} << 20},
}};

}  // namespace

std::span<const Profile> profiles() noexcept { return kProfiles; }

const Profile* find_profile(uint8_t id) noexcept {
    for (const Profile& p : kProfiles) {
        if (p.id == id) return &p;
    }
    return nullptr;
}

uint32_t model_id(const Profile& profile) {
    if (profile.modeled) return model_fingerprint(profile.model);
    constexpr std::array<uint8_t, 11> kTag{'h', 'c', 'e', '-', 's', 't', 'o', 'r', 'e', '-', '1'};
    return crc32c(kTag);
}

}  // namespace hce
