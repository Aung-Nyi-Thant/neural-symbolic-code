// LZ77 helpers shared by Tiers 1 and 2: match length, vector copies.
#pragma once

#include "common.hpp"

namespace ultracodec::detail {

// Length of the common prefix of a and b, reading no further than limit on a's side.
ULTRA_INLINE std::size_t match_length(const u8* a, const u8* b, const u8* limit) noexcept {
  const u8* start = a;
  while (a + 8 <= limit) {
    u64 x = load64(a) ^ load64(b);
    if (x) return static_cast<std::size_t>(a - start) + (ctz64(x) >> 3);
    a += 8;
    b += 8;
  }
  while (a < limit && *a == *b) { ++a; ++b; }
  return static_cast<std::size_t>(a - start);
}

// Copies n bytes in 16-byte vector chunks; may write up to 15 bytes past dst + n
// and read up to 15 past src + n.
ULTRA_INLINE void wild_copy(u8* dst, const u8* src, std::size_t n) noexcept {
  u8* end = dst + n;
  do { copy16(dst, src); dst += 16; src += 16; } while (dst < end);
}

// Copies an LZ match of `len` bytes from `offset` back (0 < offset <= op - start).
// `oend` bounds the writes; the fast path needs 16 bytes of room past the match.
ULTRA_INLINE void copy_match(u8* op, std::size_t offset, std::size_t len, const u8* oend) noexcept {
  const u8* match = op - offset;
  u8* end = op + len;
  if (ULTRA_LIKELY(offset >= 16 && end + 16 <= oend)) {
    do { copy16(op, match); op += 16; match += 16; } while (op < end);
  } else if (offset >= 8) {
    while (op + 8 <= end) { store64(op, load64(match)); op += 8; match += 8; }
    while (op < end) *op++ = *match++;
  } else {
    while (op < end) *op++ = *match++;
  }
}

}  // namespace ultracodec::detail
