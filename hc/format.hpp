// hc/format.hpp - the Container v3 FRAMED wire format: headers, blocks, END trailer.
// Part of the header-only container library; hc_framed.hpp includes every part.
#pragma once

#include "hc/crc32c.hpp"

namespace hc {

// ------------------------------------------------------------ wire format ---
//
// stream  := StreamHeader (Frame | AncillaryBlock)* End
// frame   := FrameHeader Block*            -- a sync point: total model reset
// Item kind is decided by its first byte: kSyncMarker[0] -> frame header,
// kTagEnd -> END trailer, anything else -> block header. All items are fixed-size.

inline constexpr std::array<std::byte, 4> kMagic{std::byte{0x89}, std::byte{'H'}, std::byte{'C'},
                                                 std::byte{'Z'}};  // placeholder for the real v3 magic
inline constexpr u8 kVersion3 = 3;
inline constexpr u8 kFeatFramed = 0x01;  // incompat bit: the R18-R21 layout
inline constexpr std::size_t kStreamHeaderSize = 32;
inline constexpr std::size_t kFrameHeaderSize = 40;
inline constexpr std::size_t kBlockHeaderSize = 20;
inline constexpr std::size_t kEndSize = 32;
inline constexpr std::size_t kMaxItemSize = 40;

// First byte has high bits set, so it can never be a valid block tag.
inline constexpr std::array<std::byte, 8> kSyncMarker{std::byte{0xC3}, std::byte{0x5A}, std::byte{0x9E},
                                                      std::byte{0x27}, std::byte{0xB1}, std::byte{0x6D},
                                                      std::byte{0x04}, std::byte{0xF8}};
inline constexpr u8 kMinWindowLog = 16, kMaxWindowLog = 30;
inline constexpr u8 kMinBlockLog = 10, kMaxBlockLog = 23;  // usize is a u24 field
inline constexpr u32 kMaxAncillaryBytes = u32{1} << 20;

enum class BlockType : u8 { cm = 0, stored = 1 };
inline constexpr u8 kTagEnd = 0x07;
inline constexpr u8 kTagTypeMask = 0x07, kTagAncillary = 0x08, kTagReserved = 0xF0;

struct StreamParams {
  u8 level = 0;
  u8 window_log = 23;
  u8 block_log = 18;
  u32 prior_id = 0;
  u32 stream_id = 0;
  std::optional<u64> content_size;  // nullopt: size unknown upfront (R18 streaming)

  constexpr u64 window_size() const noexcept { return u64{1} << window_log; }
  constexpr u32 block_capacity() const noexcept { return u32{1} << block_log; }
  constexpr bool same_coding(const StreamParams& o) const noexcept {
    return level == o.level && window_log == o.window_log && block_log == o.block_log &&
           prior_id == o.prior_id && stream_id == o.stream_id;
  }
};

struct DecodeLimits {
  u64 max_memory_bytes = u64{1} << 30;    // window + model tables + block buffer
  u32 max_block_bytes = u32{1} << 23;
  u8 max_window_log = 27;                 // 128 MiB
  u8 max_level = 4;
  u32 max_ratio = 0;                      // 0 = off; else output <= ratio * input + burst
  u64 ratio_burst_bytes = u64{64} << 20;  // token-bucket depth; keep >= the largest block accepted
  u64 max_output_bytes = ~u64{0};

  static constexpr DecodeLimits trusted() noexcept { return {}; }
  static constexpr DecodeLimits untrusted() noexcept {
    return {.max_memory_bytes = u64{256} << 20,
            .max_block_bytes = u32{1} << 20,
            .max_window_log = 25,
            .max_level = 3,
            .max_ratio = 100,
            .ratio_burst_bytes = u64{64} << 20,
            .max_output_bytes = u64{64} << 30};
  }
};

// Structural checks first (corruption), then policy (R19). Never allocates.
constexpr Errc validate_params(const StreamParams& p, const DecodeLimits& lim) noexcept {
  if (p.window_log < kMinWindowLog || p.window_log > kMaxWindowLog) return Errc::header_corrupt;
  if (p.block_log < kMinBlockLog || p.block_log > kMaxBlockLog || p.block_log > p.window_log)
    return Errc::header_corrupt;
  if (p.window_log > lim.max_window_log) return Errc::limit_window;
  if (p.block_capacity() > lim.max_block_bytes) return Errc::limit_block;
  if (p.level > lim.max_level) return Errc::limit_level;
  if (p.content_size && *p.content_size > lim.max_output_bytes) return Errc::limit_output;
  return Errc::ok;
}

// Stream header, 32 bytes:
//  0 magic[4]  4 version  5 incompat  6 compat  7 level  8 window_log  9 block_log
// 10 reserved[2]  12 prior_id u32  16 content_size+1 u64 (0 = unknown)
// 24 stream_id u32  28 CRC-32C(0..27)
using StreamHeaderBytes = std::array<std::byte, kStreamHeaderSize>;

constexpr StreamHeaderBytes encode_stream_header(const StreamParams& p) noexcept {
  StreamHeaderBytes h{};
  std::copy(kMagic.begin(), kMagic.end(), h.begin());
  h[4] = std::byte{kVersion3};
  h[5] = std::byte{kFeatFramed};
  h[7] = std::byte{p.level};
  h[8] = std::byte{p.window_log};
  h[9] = std::byte{p.block_log};
  store_le<u32>(&h[12], p.prior_id);
  // Size n is stored as n + 1 so that 0 can mean "unknown" while an empty input stays representable.
  store_le<u64>(&h[16], p.content_size ? *p.content_size + 1 : 0);
  store_le<u32>(&h[24], p.stream_id);
  store_le<u32>(&h[28], crc32c(Bytes(h).first(28)));
  return h;
}

constexpr Errc parse_stream_header(Bytes h, StreamParams& p) noexcept {
  if (!std::equal(kMagic.begin(), kMagic.end(), h.begin())) return Errc::bad_magic;
  if (std::to_integer<u8>(h[4]) != kVersion3) return Errc::unsupported_version;
  if (load_le<u32>(&h[28]) != crc32c(h.first(28))) return Errc::header_corrupt;
  if (std::to_integer<u8>(h[5]) != kFeatFramed) return Errc::unsupported_feature;  // unknown incompat bit
  if (h[10] != std::byte{0} || h[11] != std::byte{0}) return Errc::header_corrupt;
  p.level = std::to_integer<u8>(h[7]);
  p.window_log = std::to_integer<u8>(h[8]);
  p.block_log = std::to_integer<u8>(h[9]);
  p.prior_id = load_le<u32>(&h[12]);
  const u64 size_plus_one = load_le<u64>(&h[16]);
  p.content_size = size_plus_one ? std::optional<u64>{size_plus_one - 1} : std::nullopt;
  p.stream_id = load_le<u32>(&h[24]);
  return Errc::ok;
}

// Frame header, 40 bytes. Repeats the coding parameters so every frame is
// self-describing (late join on a live stream, damaged stream header):
//  0 sync marker[8]  8 stream_id  12 prior_id  16 level  17 window_log  18 block_log
// 19 flags (0)  20 frame index u32  24 out_offset u64  32 reserved u32  36 CRC-32C(0..35)
struct FrameHeader {
  StreamParams params;
  u32 index = 0;
  u64 out_offset = 0;  // uncompressed offset of the frame's first byte
};
using FrameHeaderBytes = std::array<std::byte, kFrameHeaderSize>;

constexpr FrameHeaderBytes encode_frame_header(const FrameHeader& f) noexcept {
  FrameHeaderBytes h{};
  std::copy(kSyncMarker.begin(), kSyncMarker.end(), h.begin());
  store_le<u32>(&h[8], f.params.stream_id);
  store_le<u32>(&h[12], f.params.prior_id);
  h[16] = std::byte{f.params.level};
  h[17] = std::byte{f.params.window_log};
  h[18] = std::byte{f.params.block_log};
  store_le<u32>(&h[20], f.index);
  store_le<u64>(&h[24], f.out_offset);
  store_le<u32>(&h[36], crc32c(Bytes(h).first(36)));
  return h;
}

constexpr Errc parse_frame_header(Bytes h, FrameHeader& f) noexcept {
  if (!std::equal(kSyncMarker.begin(), kSyncMarker.end(), h.begin())) return Errc::header_corrupt;
  if (load_le<u32>(&h[36]) != crc32c(h.first(36))) return Errc::header_corrupt;
  if (h[19] != std::byte{0} || load_le<u32>(&h[32]) != 0) return Errc::header_corrupt;
  f.params.stream_id = load_le<u32>(&h[8]);
  f.params.prior_id = load_le<u32>(&h[12]);
  f.params.level = std::to_integer<u8>(h[16]);
  f.params.window_log = std::to_integer<u8>(h[17]);
  f.params.block_log = std::to_integer<u8>(h[18]);
  f.params.content_size = std::nullopt;
  f.index = load_le<u32>(&h[20]);
  f.out_offset = load_le<u64>(&h[24]);
  return Errc::ok;
}

// Block header, 20 bytes:
//  0 tag (type:3 | ancillary:1 | reserved:4)  1 usize u24  4 csize u24  7 reserved
//  8 payload CRC-32C  12 content CRC-32C (0 for STORED/ancillary)  16 CRC-32C(0..15)
struct BlockHeader {
  u8 tag = 0;
  u32 usize = 0;
  u32 csize = 0;
  u32 payload_crc = 0;
  u32 content_crc = 0;

  constexpr bool ancillary() const noexcept { return (tag & kTagAncillary) != 0; }
  constexpr BlockType type() const noexcept { return static_cast<BlockType>(tag & kTagTypeMask); }
};
using BlockHeaderBytes = std::array<std::byte, kBlockHeaderSize>;

constexpr BlockHeaderBytes encode_block_header(const BlockHeader& b) noexcept {
  BlockHeaderBytes h{};
  h[0] = std::byte{b.tag};
  store_le24(&h[1], b.usize);
  store_le24(&h[4], b.csize);
  store_le<u32>(&h[8], b.payload_crc);
  store_le<u32>(&h[12], b.content_crc);
  store_le<u32>(&h[16], crc32c(Bytes(h).first(16)));
  return h;
}

constexpr Errc parse_block_header(Bytes h, BlockHeader& b) noexcept {
  if (load_le<u32>(&h[16]) != crc32c(h.first(16))) return Errc::header_corrupt;
  b.tag = std::to_integer<u8>(h[0]);
  if ((b.tag & kTagReserved) != 0 || h[7] != std::byte{0}) return Errc::header_corrupt;
  b.usize = load_le24(&h[1]);
  b.csize = load_le24(&h[4]);
  b.payload_crc = load_le<u32>(&h[8]);
  b.content_crc = load_le<u32>(&h[12]);
  return Errc::ok;
}

// R19 structural rules. A CRC-valid header can still be hostile (an attacker
// can compute CRCs), so none of these checks rely on the CRC.
constexpr Errc check_block_shape(const BlockHeader& b, const StreamParams& p) noexcept {
  if (b.ancillary())
    return (b.usize == 0 && b.csize <= kMaxAncillaryBytes && b.content_crc == 0) ? Errc::ok
                                                                                : Errc::header_corrupt;
  if (b.usize == 0 || b.usize > p.block_capacity()) return Errc::header_corrupt;
  switch (b.type()) {
    case BlockType::stored:
      return (b.csize == b.usize && b.content_crc == 0) ? Errc::ok : Errc::header_corrupt;
    case BlockType::cm:  // the STORED fallback guarantees a coded block is strictly smaller
      return (b.csize != 0 && b.csize < b.usize) ? Errc::ok : Errc::header_corrupt;
  }
  return Errc::unsupported_feature;
}

// END trailer, 32 bytes:
//  0 tag (0x07)  1 reserved[7]  8 total_out u64  16 frame_count u32
// 20 list_check u32 (CRC-32C over the per-block content CRCs, in order)
// 24 index_back u32 (distance back to a seek-index ancillary block, 0 = none)
// 28 CRC-32C(0..27)
struct EndTrailer {
  u64 total_out = 0;
  u32 frame_count = 0;
  u32 list_check = 0;
  u32 index_back = 0;
};
using EndBytes = std::array<std::byte, kEndSize>;

constexpr EndBytes encode_end(const EndTrailer& t) noexcept {
  EndBytes h{};
  h[0] = std::byte{kTagEnd};
  store_le<u64>(&h[8], t.total_out);
  store_le<u32>(&h[16], t.frame_count);
  store_le<u32>(&h[20], t.list_check);
  store_le<u32>(&h[24], t.index_back);
  store_le<u32>(&h[28], crc32c(Bytes(h).first(28)));
  return h;
}

constexpr Errc parse_end(Bytes h, EndTrailer& t) noexcept {
  if (load_le<u32>(&h[28]) != crc32c(h.first(28))) return Errc::header_corrupt;
  for (std::size_t i = 1; i < 8; ++i)
    if (h[i] != std::byte{0}) return Errc::header_corrupt;
  t.total_out = load_le<u64>(&h[8]);
  t.frame_count = load_le<u32>(&h[16]);
  t.list_check = load_le<u32>(&h[20]);
  t.index_back = load_le<u32>(&h[24]);
  return Errc::ok;
}

}  // namespace hc
