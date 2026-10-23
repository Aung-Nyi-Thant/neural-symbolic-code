#include "core/kernels/crc32c.hpp"

#include <array>

#include "core/kernels/no_float.hpp"

namespace hce {
namespace {

constexpr std::array<uint32_t, 256> make_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
        table[i] = c;
    }
    return table;
}

constexpr std::array<uint32_t, 256> kTable = make_table();

}  // namespace

uint32_t crc32c(std::span<const uint8_t> data, uint32_t crc) noexcept {
    crc = ~crc;
    for (const uint8_t byte : data) crc = kTable[(crc ^ byte) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

}  // namespace hce
