// fuzz_util.hpp - what the fuzz targets share: recomputing the CRCs of a
// mutated FRAMED v3 stream, so that mutations reach the parsers and codecs
// behind the CRC checks (an attacker can compute CRCs too).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hc/format.hpp"

namespace hce::fuzz {

// Walks the items of the FRAMED stream(s) starting at `pos` and rewrites
// every header CRC and every payload CRC it can compute without decoding:
// stream headers, frame headers, block headers and their payload CRCs, END
// trailers. Content CRCs and END list checks are left alone (they need the
// decoded bytes), so a mutated coded block still reaches decode_block() and
// fails afterwards. Stops at the first item that does not fit.
inline void fix_framed_crcs(std::vector<std::byte>& d, std::size_t pos = 0) {
  using namespace hc;
  const auto fix = [&](std::size_t at, std::size_t covered, std::size_t crc_at) {
    store_le<u32>(&d[at + crc_at], crc32c(Bytes(d).subspan(at, covered)));
  };
  while (pos < d.size()) {
    const std::size_t left = d.size() - pos;
    if (left >= kStreamHeaderSize && std::equal(kMagic.begin(), kMagic.end(), d.begin() + static_cast<std::ptrdiff_t>(pos))) {
      fix(pos, 28, 28);
      pos += kStreamHeaderSize;
    } else if (left >= kFrameHeaderSize &&
               std::equal(kSyncMarker.begin(), kSyncMarker.end(), d.begin() + static_cast<std::ptrdiff_t>(pos))) {
      fix(pos, 36, 36);
      pos += kFrameHeaderSize;
    } else if (left >= kEndSize && std::to_integer<u8>(d[pos]) == kTagEnd) {
      fix(pos, 28, 28);
      pos += kEndSize;
    } else if (left >= kBlockHeaderSize) {
      const std::size_t csize = load_le24(&d[pos + 4]);
      if (csize > left - kBlockHeaderSize) return;
      store_le<u32>(&d[pos + 8], crc32c(Bytes(d).subspan(pos + kBlockHeaderSize, csize)));
      fix(pos, 16, 16);
      pos += kBlockHeaderSize + csize;
    } else {
      return;
    }
  }
}

inline std::vector<std::byte> as_bytes(const std::uint8_t* data, std::size_t size) {
  std::vector<std::byte> v(size);
  for (std::size_t i = 0; i < size; ++i) v[i] = std::byte{data[i]};
  return v;
}

}  // namespace hce::fuzz
