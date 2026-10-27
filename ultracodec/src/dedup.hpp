// Long-range deduplication front end for the CM tiers (rzip-style).
//
// Block-parallel CM cannot see repeats whose source lies in another block,
// and every byte it models costs the same time whether or not it repeats. A
// serial pre-pass over the whole input (hundreds of MB/s) therefore replaces
// (a) repeats of >= far_min bytes whose source is in another block-sized
// segment and (b) repeats of >= near_min bytes anywhere by references. The
// CM blocks code only the residual; the decoder decodes the residual blocks
// in parallel and then replays the references in one sequential pass.
//
// Finder: 32-byte windows hashed at every 16th position into a 4 Mi-entry
// table, looked up at every position, verified, extended both ways; finds
// every repeat of >= 48 bytes and many shorter ones.
//
// Encoded form (dedup section of the archive):
//   varint count, u32 crc32c(original), then three rans_static streams of
//   LEB128 varints: literal gaps, lengths, distances.
#pragma once

#include <span>
#include <vector>

#include "common.hpp"

namespace ultracodec::detail {

struct DedupParams {
  std::size_t segment;   // block size: repeats across segments count as far
  std::size_t far_min;   // minimum length of a cross-segment repeat
  std::size_t near_min;  // minimum length of any repeat
};

// Returns the residual; appends the encoded reference list to `section`.
std::vector<u8> dedup_encode(std::span<const u8> input, const DedupParams& p, std::vector<u8>& section);

// Rebuilds the original from the residual. Throws Error on any
// inconsistency (sizes, distances, CRC).
void dedup_decode(Reader& section, std::span<const u8> residual, std::span<u8> out);

// Residual size the section implies (validated against the archive before
// any allocation).
std::size_t dedup_residual_size(Reader section_copy, std::size_t original_size);

}  // namespace ultracodec::detail
