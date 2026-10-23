// CRC-32C (Castagnoli): reflected polynomial 0x82F63B78, initial value and final XOR 0xFFFFFFFF.
// crc32c("123456789") == 0xE3069283. Chaining: crc32c(b, crc32c(a)) == crc32c(a followed by b).
#pragma once

#include <cstdint>
#include <span>

namespace hce {

uint32_t crc32c(std::span<const uint8_t> data, uint32_t crc = 0) noexcept;

}  // namespace hce
