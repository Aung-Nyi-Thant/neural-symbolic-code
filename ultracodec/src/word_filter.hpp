// Tier 2 text pre-filter: a static word dictionary plus capitalization flags,
// applied to a block before LZ77 when a sample says it pays.
//
// Output byte stream (reversible, at most 2 bytes per input byte):
//   0x00, 0x03..0x7F   the byte itself
//   0xFF b             the byte b (escapes 0x01, 0x02 and 0x80..0xFF)
//   0x80 + id          dictionary word id < 64
//   0xC0 + h, l        dictionary word 64 + h * 256 + l (h < 63)
//   0x01 w             w with its first letter upper-cased
//   0x02 w             w with every letter upper-cased
// where w after a flag is a dictionary code or a run of lower-case word
// characters. A word is a maximal run of word characters: letters (Prose),
// or letters, digits and '_' starting with a letter or '_' (Code). Words in
// other mixed case, or longer than 255 bytes, are copied as they are.
#pragma once

#include <span>
#include <vector>

#include "common.hpp"

namespace ultracodec::detail {

enum class WordDict : u8 { None = 0, Prose = 1, Code = 2 };

// The dictionary whose filter shrinks samples of `raw` the most, or None
// when neither shrinks them below 85%.
WordDict word_filter_choose(std::span<const u8> raw);
// Replaces `out` with the filtered form of `raw`.
void word_filter_encode(WordDict dict, std::span<const u8> raw, std::vector<u8>& out);
// Writes exactly dst.size() bytes; throws Error on malformed input.
void word_filter_decode(WordDict dict, std::span<const u8> in, std::span<u8> dst);

}  // namespace ultracodec::detail
