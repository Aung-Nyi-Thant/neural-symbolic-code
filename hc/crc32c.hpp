// hc/crc32c.hpp - CRC-32C (Castagnoli), slicing-by-8.
// Part of the header-only container library; hc_framed.hpp includes every part.
#pragma once

#include "hc/base.hpp"

namespace hc {

// ---------------------------------------------------------------- CRC-32C ---

namespace detail {
consteval std::array<std::array<u32, 256>, 8> make_crc32c_tables() {
  std::array<std::array<u32, 256>, 8> t{};
  for (u32 i = 0; i < 256; ++i) {
    u32 c = i;
    for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u)));
    t[0][i] = c;
  }
  for (std::size_t s = 1; s < 8; ++s)
    for (std::size_t i = 0; i < 256; ++i) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFFu];
  return t;
}
inline constexpr std::array<std::array<u32, 256>, 8> kCrc32c = make_crc32c_tables();
}  // namespace detail

// Slicing-by-8 software CRC-32C. SSE4.2 _mm_crc32_u64 and ARMv8 __crc32cd
// produce bit-identical results and may replace it. Chains: crc32c(b, crc32c(a)).
constexpr u32 crc32c(Bytes data, u32 crc = 0) noexcept {
  const auto& t = detail::kCrc32c;
  crc = ~crc;
  std::size_t i = 0;
  for (; i + 8 <= data.size(); i += 8) {
    const u32 lo = crc ^ load_le<u32>(&data[i]);
    const u32 hi = load_le<u32>(&data[i + 4]);
    crc = t[7][lo & 0xFFu] ^ t[6][(lo >> 8) & 0xFFu] ^ t[5][(lo >> 16) & 0xFFu] ^ t[4][lo >> 24] ^
          t[3][hi & 0xFFu] ^ t[2][(hi >> 8) & 0xFFu] ^ t[1][(hi >> 16) & 0xFFu] ^ t[0][hi >> 24];
  }
  for (; i < data.size(); ++i) crc = t[0][(crc ^ std::to_integer<u32>(data[i])) & 0xFFu] ^ (crc >> 8);
  return ~crc;
}

static_assert(
    [] {
      constexpr char s[] = "123456789";
      std::array<std::byte, 9> b{};
      for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::byte>(s[i]);
      return crc32c(b) == 0xE3069283u;
    }(),
    "CRC-32C check value");

}  // namespace hc
