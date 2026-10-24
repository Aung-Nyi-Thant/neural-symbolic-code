// hc_framed.hpp - Container v3 FRAMED layout: reference patterns for R18-R21.
//
// Covers framing (R18), resource limits (R19), resynchronization and partial
// recovery (R20) and the bounded history window (R21). The tiered CM codec plugs
// in through the BlockCodec concept. Integer-only, no exceptions; the only heap
// allocation is in bind_stream(), at most once per stream, and none at all when
// the arena from a previous stream is already large enough.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stop_token>
#include <utility>

namespace hc {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Bytes = std::span<const std::byte>;
using MutBytes = std::span<std::byte>;

// ---------------------------------------------------------------- status ----

enum class Errc : u8 {
  ok = 0,
  need_input,     // flow control: feed more input, or finish
  output_full,    // flow control: drain the output buffer
  end_of_stream,  // flow control: END trailer processed
  // format and integrity
  bad_magic, unsupported_version, unsupported_feature, header_corrupt, payload_corrupt,
  content_mismatch, bad_reference, sequence_gap, truncated, trailing_garbage,
  // R19 policy: fatal in every recovery mode
  limit_memory, limit_block, limit_window, limit_level, limit_ratio, limit_output,
  // environment
  out_of_memory, io_error, cancelled,
};

constexpr bool is_error(Errc e) noexcept { return e > Errc::end_of_stream; }
constexpr bool is_limit(Errc e) noexcept {
  return e >= Errc::limit_memory && e <= Errc::limit_output;
}

constexpr const char* to_string(Errc e) noexcept {
  switch (e) {
    case Errc::ok: return "ok";
    case Errc::need_input: return "need_input";
    case Errc::output_full: return "output_full";
    case Errc::end_of_stream: return "end_of_stream";
    case Errc::bad_magic: return "bad_magic";
    case Errc::unsupported_version: return "unsupported_version";
    case Errc::unsupported_feature: return "unsupported_feature";
    case Errc::header_corrupt: return "header_corrupt";
    case Errc::payload_corrupt: return "payload_corrupt";
    case Errc::content_mismatch: return "content_mismatch";
    case Errc::bad_reference: return "bad_reference";
    case Errc::sequence_gap: return "sequence_gap";
    case Errc::truncated: return "truncated";
    case Errc::trailing_garbage: return "trailing_garbage";
    case Errc::limit_memory: return "limit_memory";
    case Errc::limit_block: return "limit_block";
    case Errc::limit_window: return "limit_window";
    case Errc::limit_level: return "limit_level";
    case Errc::limit_ratio: return "limit_ratio";
    case Errc::limit_output: return "limit_output";
    case Errc::out_of_memory: return "out_of_memory";
    case Errc::io_error: return "io_error";
    case Errc::cancelled: return "cancelled";
  }
  return "unknown";
}

// ------------------------------------------------- integer serialization ----

template <std::unsigned_integral T>
constexpr T load_le(const std::byte* p) noexcept {
  T v = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i)
    v = static_cast<T>(v | (static_cast<T>(std::to_integer<u8>(p[i])) << (8 * i)));
  return v;
}

template <std::unsigned_integral T>
constexpr void store_le(std::byte* p, T v) noexcept {
  for (std::size_t i = 0; i < sizeof(T); ++i) p[i] = static_cast<std::byte>(static_cast<u8>(v >> (8 * i)));
}

constexpr u32 load_le24(const std::byte* p) noexcept {
  return load_le<u16>(p) | (u32{std::to_integer<u8>(p[2])} << 16);
}

constexpr void store_le24(std::byte* p, u32 v) noexcept {
  store_le<u16>(p, static_cast<u16>(v & 0xFFFFu));
  p[2] = static_cast<std::byte>((v >> 16) & 0xFFu);
}

constexpr std::array<std::byte, 4> le32(u32 v) noexcept {
  std::array<std::byte, 4> b{};
  store_le<u32>(b.data(), v);
  return b;
}

// LEB128 for codec payloads: at most 10 bytes, canonical (no redundant zero group).
constexpr Errc read_varint(Bytes in, std::size_t& pos, u64& out) noexcept {
  u64 v = 0;
  for (unsigned i = 0; i < 10; ++i) {
    if (pos >= in.size()) return Errc::payload_corrupt;
    const u8 b = std::to_integer<u8>(in[pos++]);
    if (i == 9 && b > 1) return Errc::payload_corrupt;  // would exceed 64 bits
    v |= u64{b & 0x7Fu} << (7 * i);
    if ((b & 0x80u) == 0) {
      if (b == 0 && i != 0) return Errc::payload_corrupt;  // overlong encoding
      out = v;
      return Errc::ok;
    }
  }
  return Errc::payload_corrupt;
}

constexpr std::size_t write_varint(std::byte* p, u64 v) noexcept {
  std::size_t n = 0;
  do {
    const u8 low = static_cast<u8>(v & 0x7Fu);
    v >>= 7;
    p[n++] = static_cast<std::byte>(low | (v ? 0x80u : 0u));
  } while (v);
  return n;
}

constexpr u64 sat_add(u64 a, u64 b) noexcept { return b > ~u64{0} - a ? ~u64{0} : a + b; }
constexpr u64 sat_mul(u64 a, u64 b) noexcept {
  return (a != 0 && b > ~u64{0} / a) ? ~u64{0} : a * b;
}

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

// ----------------------------------------------------------------- budget ---

// R19 expansion control as a token bucket: each compressed byte earns
// `max_ratio` bytes of output credit (capped at the burst depth), and each
// output byte, decoded or recovery fill, spends one before it is produced.
// Long-run output <= ratio * input + burst, while single highly compressible
// blocks (64 KiB of zeros codes to ~25 bytes) still pass.
class Budget {
 public:
  explicit constexpr Budget(const DecodeLimits& l) noexcept
      : ratio_(l.max_ratio),
        burst_(l.ratio_burst_bytes),
        credit_(burst_),
        max_out_(l.max_output_bytes) {}

  constexpr void earn(u64 input_bytes) noexcept {
    if (ratio_) credit_ = std::min(burst_, sat_add(credit_, sat_mul(input_bytes, ratio_)));
  }

  [[nodiscard]] constexpr Errc spend(u64 output_bytes) noexcept {
    if (output_bytes > max_out_ - out_) return Errc::limit_output;
    if (ratio_) {
      if (output_bytes > credit_) return Errc::limit_ratio;
      credit_ -= output_bytes;
    }
    out_ += output_bytes;
    return Errc::ok;
  }

  // Returns credit for output that was reserved but never produced.
  constexpr void refund(u64 output_bytes) noexcept {
    out_ -= std::min(out_, output_bytes);
    if (ratio_) credit_ = std::min(burst_, sat_add(credit_, output_bytes));
  }

  constexpr u64 committed() const noexcept { return out_; }

 private:
  u64 ratio_, burst_, credit_, max_out_, out_ = 0;
};

// ----------------------------------------------------------------- window ---

// R21: fixed history ring of 2^window_log bytes, addressed by absolute 64-bit
// positions. Memory never grows with input size. A sync point is O(1): it
// only moves the reachability floor, nothing is cleared.
class Window {
 public:
  void bind(MutBytes mem) noexcept {  // mem.size() must be a power of two
    buf_ = mem.data();
    cap_ = mem.size();
    mask_ = cap_ - 1;
    pos_ = start_ = 0;
  }

  u64 pos() const noexcept { return pos_; }
  u64 capacity() const noexcept { return cap_; }
  // Bytes since the current sync point, STORED blocks included: the frame-relative
  // position a codec's models must use so encoder and decoder agree.
  u64 frame_offset() const noexcept { return pos_ - start_; }
  void reset_frame() noexcept { start_ = pos_; }
  u64 reachable() const noexcept { return std::min<u64>(cap_, pos_ - start_); }
  std::byte at(u64 abs) const noexcept { return buf_[abs & mask_]; }  // caller validates abs
  void put(std::byte b) noexcept { buf_[pos_++ & mask_] = b; }

  void append(Bytes src) noexcept {  // precondition: src.size() <= capacity()
    while (!src.empty()) {
      const std::size_t at = static_cast<std::size_t>(pos_ & mask_);
      const std::size_t n = std::min(src.size(), cap_ - at);
      std::memcpy(buf_ + at, src.data(), n);
      src = src.subspan(n);
      pos_ += n;
    }
  }

  // Validated back-reference with LZ77 semantics (overlap allowed: offset 1 is a run).
  [[nodiscard]] Errc copy(u64 offset, u64 len) noexcept {
    if (offset == 0 || offset > reachable()) return Errc::bad_reference;
    u64 src = pos_ - offset;
    if (len <= offset && len <= cap_ - offset) {
      // No logical overlap (len <= offset) and no physical overlap in the ring
      // (offset + len <= cap): at most three memcpy calls, split at wrap points.
      for (u64 dst = pos_, end = pos_ + len; dst != end;) {
        const std::size_t s = static_cast<std::size_t>(src & mask_);
        const std::size_t d = static_cast<std::size_t>(dst & mask_);
        const std::size_t n = static_cast<std::size_t>(std::min<u64>({end - dst, cap_ - s, cap_ - d}));
        std::memcpy(buf_ + d, buf_ + s, n);
        src += n;
        dst += n;
      }
    } else {
      // Read-before-write per byte is exact for every offset <= cap.
      for (u64 i = 0; i < len; ++i) buf_[(pos_ + i) & mask_] = buf_[(src + i) & mask_];
    }
    pos_ += len;
    return Errc::ok;
  }

  // First contiguous piece of [from, from + len). Precondition: pos() - from <= capacity().
  Bytes contiguous(u64 from, u64 len) const noexcept {
    const std::size_t at = static_cast<std::size_t>(from & mask_);
    return {buf_ + at, static_cast<std::size_t>(std::min<u64>(len, cap_ - at))};
  }

 private:
  std::byte* buf_ = nullptr;
  std::size_t cap_ = 0;
  std::size_t mask_ = 0;
  u64 pos_ = 0;
  u64 start_ = 0;
};

// R21: model tables store 32-bit truncated positions. A candidate is usable iff
// it lies inside the reachable window; both sides evaluate this identically.
// Aliasing after 2^32 bytes is deterministic within a frame, which is why
// tables must be reset at every sync point (see BlockCodec::reset_frame).
constexpr bool candidate_valid(u64 pos, u32 cand, u64 reachable) noexcept {
  const u32 d = static_cast<u32>(pos) - cand;  // wraps mod 2^32 by definition
  return d != 0 && d <= reachable;
}

// R21: bounded side-info dictionary (log templates, keyed-delta state). The
// encoder picks the slot to overwrite (explicit eviction); the decoder only
// validates, so its memory is Slots * SlotBytes regardless of input.
template <std::size_t Slots, std::size_t SlotBytes>
class SlotTable {
  static_assert(SlotBytes <= 0xFFFF);

 public:
  [[nodiscard]] Errc define(u64 slot, Bytes body) noexcept {
    if (slot >= Slots || body.empty() || body.size() > SlotBytes) return Errc::payload_corrupt;
    std::memcpy(body_[slot].data(), body.data(), body.size());
    len_[slot] = static_cast<u16>(body.size());
    return Errc::ok;
  }
  [[nodiscard]] Errc get(u64 slot, Bytes& out) const noexcept {
    if (slot >= Slots || len_[slot] == 0) return Errc::payload_corrupt;
    out = Bytes(body_[slot]).first(len_[slot]);
    return Errc::ok;
  }
  void reset() noexcept { len_.fill(0); }  // sync point

 private:
  std::array<u16, Slots> len_{};
  std::array<std::array<std::byte, SlotBytes>, Slots> body_{};
};

// ------------------------------------------------------------- interfaces ---

// The tiered CM decoder (or any block codec) as seen by the container.
template <class C>
concept BlockCodec = requires(C& c, const StreamParams& p, MutBytes mem, Bytes payload, u32 usize, Window& w) {
  { C::accepts(p) } noexcept -> std::same_as<Errc>;  // codec/prior supported? checked before allocating
  { C::memory_required(p) } noexcept -> std::same_as<u64>;
  { c.bind(p, mem) } noexcept -> std::same_as<Errc>;  // once per stream; tables carved from `mem`
  { c.reset_frame() } noexcept -> std::same_as<void>;  // R20: restore the prior image, every table
  // Appends exactly `usize` bytes to `w`, never more; every token yields >= 1 byte;
  // the whole payload is consumed. Anything else is payload_corrupt or bad_reference.
  { c.decode_block(payload, usize, w) } noexcept -> std::same_as<Errc>;
};

template <class S>
concept ByteSource = requires(S& s, MutBytes buf) {
  { s.read(buf) } -> std::same_as<std::size_t>;  // 0 = end of input; short reads are fine
};

template <class S>
concept ByteSink = requires(S& s, Bytes buf) {
  { s.write(buf) } -> std::same_as<bool>;  // false = downstream closed or I/O error
};

enum class Recovery : u8 { strict, skip, skip_and_fill };

struct GapInfo {
  u32 stream_index;   // which concatenated stream
  u64 out_offset;     // uncompressed offset where the loss starts
  u64 length;         // bytes lost; meaningful only when length_known
  bool length_known;  // false when the stream was truncated
  u64 in_offset;      // compressed offset where the damage was detected
  Errc cause;
};
using GapFn = void (*)(void* ctx, const GapInfo&) noexcept;

struct DecoderOptions {
  DecodeLimits limits{};
  Recovery recovery = Recovery::strict;
  std::byte fill{0};
  GapFn on_gap = nullptr;
  void* gap_ctx = nullptr;
  std::stop_token stop{};
};

struct Progress {
  std::size_t consumed = 0;
  std::size_t produced = 0;
  Errc status = Errc::ok;
};

struct DecodeStats {
  u64 frames = 0, blocks_ok = 0, blocks_salvaged = 0, blocks_lost = 0;
  u64 damage_events = 0, gaps = 0, bytes_lost = 0, bytes_filled = 0;
  bool content_verified = false;  // last END matched and nothing was lost in that stream
};

// Parallel decoding, splitter side (decoder/pipeline.cpp). A decoder in split
// mode validates the framing exactly as in sequential mode (headers, ordering,
// resynchronization, and the R19 budget on declared sizes) but forwards every
// accepted item to the sink instead of decoding it, and writes no output.
// Worker threads decode the forwarded frames in job mode.
class SplitSink {
 public:
  // A frame header was accepted; `header` is its 40 bytes, found at `in_offset`.
  virtual Errc frame(const FrameHeader& f, Bytes header, u64 in_offset, u32 stream_index) noexcept = 0;
  // A valid block header (data or ancillary) inside the current frame, then its payload.
  virtual Errc block(const BlockHeader& b, Bytes header) noexcept = 0;
  virtual Errc payload(Bytes bytes) noexcept = 0;
  // A damaged item ends the current frame; a resync scan follows.
  virtual Errc damage(Errc cause, u64 logical, u64 in_offset, u32 stream_index) noexcept = 0;
  // The END trailer after the splitter's own checks; `verified` covers what headers can show.
  virtual Errc end(const EndTrailer& t, bool verified, u64 in_offset, u32 stream_index) noexcept = 0;
  // Recovery modes: the input ended inside a stream whose parameters were known.
  virtual void truncated(u64 logical, u64 in_offset, u32 stream_index) noexcept = 0;

 protected:
  ~SplitSink() = default;
};

// ---------------------------------------------------------------- decoder ---

template <BlockCodec Codec>
class StreamDecoder {
 public:
  StreamDecoder(Codec& codec, const DecoderOptions& opt) noexcept
      : codec_(codec), opt_(opt), budget_(opt.limits) {}

  // Push API (R18). Never seeks and never looks ahead past what it consumes.
  // Call again with more input after need_input, or with fresh output space
  // after output_full. `input_finished` means the source hit EOF. After
  // end_of_stream, more input is decoded as a concatenated stream.
  [[nodiscard]] Progress decode(Bytes in, MutBytes out, bool input_finished) noexcept {
    Progress p;
    for (;;) {
      if (st_ == St::failed) {
        p.status = err_;
        return p;
      }
      if (opt_.stop.stop_requested()) {
        p.status = fail(Errc::cancelled);
        return p;
      }
      Errc r = step(in, out, p);
      if (r == Errc::ok) continue;
      if (r == Errc::need_input && input_finished) {
        const bool job_done = job_ && st_ == St::item && hdr_len_ == 0;
        r = (st_ == St::done || job_done) ? Errc::end_of_stream : on_truncated();
      }
      p.status = is_error(r) ? fail(r) : r;
      return p;
    }
  }

  const DecodeStats& stats() const noexcept { return stats_; }
  const StreamParams& params() const noexcept { return params_; }
  bool params_valid() const noexcept { return params_valid_; }
  u64 logical() const noexcept { return logical_; }

  // Split mode (see SplitSink). Set before the first decode().
  void set_split_sink(SplitSink* sink) noexcept { split_ = sink; }

  // ---- Job mode: parallel decoding, worker side ----

  struct JobStart {
    u64 out_offset = 0;  // from the frame header
    u64 frame_index = 0;
    u64 in_offset = 0;  // input offset of the frame header
    u32 stream_index = 0;
  };
  // A recovery fill, reported with its position in the job's output instead of
  // being written: a fill can be larger than any fixed output buffer.
  using FillFn = void (*)(void* ctx, u64 at, u64 len) noexcept;
  struct PendingGap {
    u64 out_offset;
    u64 in_offset;
    Errc cause;
  };

  // Prepares to decode one frame forwarded by a splitter. The bytes passed to
  // decode() afterwards are exactly that frame's items, starting with its
  // header, possibly over several calls that each end on an item boundary;
  // the final call (input_finished) reports end_of_stream. The splitter has
  // already enforced the R19 budget.
  [[nodiscard]] Errc begin_job(const StreamParams& p, const JobStart& j, FillFn fill, void* fill_ctx) noexcept {
    if (!params_valid_ || !p.same_coding(params_)) {
      params_valid_ = false;
      if (const Errc e = bind_stream(p); e != Errc::ok) return e;
    }
    st_ = St::item;
    err_ = Errc::ok;
    hdr_len_ = 0;
    in_frame_ = frame_broken_ = gap_open_ = false;
    emit_left_ = fill_left_ = 0;
    logical_ = j.out_offset;
    frames_ = j.frame_index;
    in_offset_ = j.in_offset;
    stream_index_ = j.stream_index;
    list_crc_ = 0;
    stream_damage_ = 0;
    stats_ = {};
    job_ = true;
    job_fill_ = fill;
    job_ctx_ = fill_ctx;
    job_produced_ = 0;
    return Errc::ok;
  }

  // After a job: the open gap that the next frame header will close, if any.
  std::optional<PendingGap> pending_gap() const noexcept {
    if (!gap_open_) return std::nullopt;
    return PendingGap{gap_start_, gap_in_, gap_cause_};
  }

 private:
  enum class St : u8 { stream_header, item, payload, skip, emit, fill, scan, done, failed };

  Errc step(Bytes& in, MutBytes& out, Progress& p) noexcept {
    switch (st_) {
      case St::stream_header: return on_stream_header(in, p);
      case St::item: return on_item(in, p);
      case St::payload: return on_payload(in, p);
      case St::skip: return on_skip(in, p);
      case St::emit: return on_emit(out, p);
      case St::fill: return on_fill(out, p);
      case St::scan: return on_scan(in, p);
      case St::done:
        if (in.empty()) return Errc::need_input;
        begin_stream();  // concatenated stream follows
        return Errc::ok;
      case St::failed: return err_;
    }
    return Errc::header_corrupt;
  }

  bool recovering() const noexcept { return opt_.recovery != Recovery::strict; }
  Bytes staged() const noexcept { return Bytes(hdr_).first(hdr_len_); }

  void consume(Bytes& in, std::size_t n, Progress& p) noexcept {
    in = in.subspan(n);
    p.consumed += n;
    in_offset_ += n;
  }

  // Restartable fixed-size header staging: works for any input chunking.
  bool gather(Bytes& in, std::size_t n, Progress& p) noexcept {
    const std::size_t take = std::min(n - hdr_len_, in.size());
    if (take) std::memcpy(hdr_.data() + hdr_len_, in.data(), take);
    hdr_len_ += take;
    consume(in, take, p);
    return hdr_len_ == n;
  }

  Errc then(St next) noexcept {
    after_fill_ = next;
    st_ = fill_left_ ? St::fill : next;
    return st_ == St::done ? Errc::end_of_stream : Errc::ok;
  }

  Errc fail(Errc e) noexcept {
    st_ = St::failed;
    err_ = e;
    return e;
  }

  void begin_stream() noexcept {
    ++stream_index_;
    st_ = St::stream_header;
    hdr_len_ = 0;
    params_valid_ = in_frame_ = frame_broken_ = gap_open_ = false;
    logical_ = 0;
    frames_ = 0;
    list_crc_ = 0;
    stream_damage_ = 0;
  }

  // The only allocation: once per stream, sized from validated parameters (R19).
  Errc bind_stream(const StreamParams& sp) noexcept {
    if (const Errc a = Codec::accepts(sp); a != Errc::ok) return a;
    const u64 window = sp.window_size();
    const u64 block = sp.block_capacity();
    const u64 model = Codec::memory_required(sp);
    const u64 total = sat_add(sat_add(window, block), model);
    if (total > opt_.limits.max_memory_bytes || !std::in_range<std::size_t>(total)) return Errc::limit_memory;
    if (split_) {  // split mode decodes nothing: same checks, no window, model or payload buffer
      params_ = sp;
      params_valid_ = true;
      return Errc::ok;
    }
    if (total > arena_cap_) {
      arena_.reset(new (std::nothrow) std::byte[static_cast<std::size_t>(total)]);
      arena_cap_ = arena_ ? total : 0;
      if (!arena_) return Errc::out_of_memory;
    }
    const MutBytes mem(arena_.get(), static_cast<std::size_t>(total));
    window_.bind(mem.first(static_cast<std::size_t>(window)));
    payload_ = mem.subspan(static_cast<std::size_t>(window), block);
    if (const Errc e = codec_.bind(sp, mem.subspan(static_cast<std::size_t>(window + block))); e != Errc::ok)
      return e;
    params_ = sp;
    params_valid_ = true;
    return Errc::ok;
  }

  Errc on_stream_header(Bytes& in, Progress& p) noexcept {
    if (!gather(in, kStreamHeaderSize, p)) return Errc::need_input;
    StreamParams sp;
    Errc e = parse_stream_header(staged(), sp);
    if (e == Errc::ok) e = validate_params(sp, opt_.limits);
    if (e == Errc::ok) {
      if (const Errc b = bind_stream(sp); b != Errc::ok) return b;
      budget_.earn(kStreamHeaderSize);
      hdr_len_ = 0;
      st_ = St::item;
      return Errc::ok;
    }
    if (is_limit(e) || e == Errc::unsupported_version || e == Errc::unsupported_feature) return e;
    if (!recovering()) return (e == Errc::bad_magic && stream_index_ > 0) ? Errc::trailing_garbage : e;
    // Damaged or missing stream header (late join): frames are self-describing,
    // so resync on the first one. Rescan from byte 0: a frame may start right here.
    if (split_) {
      if (const Errc s = split_->damage(e, logical_, in_offset_, stream_index_); s != Errc::ok) return s;
    }
    note_damage();
    open_gap(e);
    return enter_scan(0);
  }

  Errc on_item(Bytes& in, Progress& p) noexcept {
    if (hdr_len_ == 0 && in.empty()) return Errc::need_input;
    const std::byte first = hdr_len_ ? hdr_[0] : in[0];
    const std::size_t size = first == kSyncMarker[0]        ? kFrameHeaderSize
                             : first == std::byte{kTagEnd} ? kEndSize
                                                           : kBlockHeaderSize;
    if (!gather(in, size, p)) return Errc::need_input;
    if (size == kFrameHeaderSize) {
      FrameHeader f;
      if (const Errc e = parse_frame_header(staged(), f); e != Errc::ok) return damaged(e);
      return accept_frame(f);
    }
    return size == kEndSize ? on_end() : on_block_header();
  }

  // A CRC-valid frame header: accept it as the next sync point, or reject it.
  Errc accept_frame(const FrameHeader& f) noexcept {
    const bool scanning = st_ == St::scan;
    if (!params_valid_) {  // the stream header was damaged: adopt the frame's self-description
      if (const Errc v = validate_params(f.params, opt_.limits); v != Errc::ok)
        return is_limit(v) ? v : reject(v, scanning);
      if (const Errc b = bind_stream(f.params); b != Errc::ok) return b;
    }
    const bool consistent = f.params.same_coding(params_) &&
                            (!params_.content_size || f.out_offset <= *params_.content_size);
    const bool in_order = recovering() ? (f.out_offset >= logical_ && f.index >= frames_)
                                       : (f.out_offset == logical_ && f.index == frames_);
    if (!consistent || !in_order) return reject(Errc::header_corrupt, scanning);
    budget_.earn(kFrameHeaderSize);
    if (f.out_offset > logical_) {  // whole frames are missing
      note_damage();
      open_gap(Errc::sequence_gap);
    }
    if (const Errc g = close_gap(f.out_offset); g != Errc::ok) return g;
    logical_ = f.out_offset;
    frames_ = u64{f.index} + 1;
    ++stats_.frames;
    in_frame_ = true;
    frame_broken_ = false;
    if (split_) {  // hdr_ still holds the 40 accepted bytes
      const Errc s =
          split_->frame(f, Bytes(hdr_).first(kFrameHeaderSize), in_offset_ - kFrameHeaderSize, stream_index_);
      if (s != Errc::ok) return s;
    } else {
      window_.reset_frame();
      codec_.reset_frame();  // R20: nothing decoded from here on depends on earlier bytes
    }
    hdr_len_ = 0;
    return then(St::item);
  }

  Errc on_block_header() noexcept {
    BlockHeader b;
    Errc e = parse_block_header(staged(), b);
    if (e == Errc::ok && !in_frame_) e = Errc::header_corrupt;  // blocks live inside frames
    if (e == Errc::ok) e = check_block_shape(b, params_);
    if (e == Errc::ok && params_.content_size && b.usize > *params_.content_size - logical_)
      e = Errc::header_corrupt;
    if (e != Errc::ok) return damaged(e);
    hdr_len_ = 0;
    cur_ = b;
    budget_.earn(kBlockHeaderSize + u64{b.csize});
    skip_left_ = b.csize;
    if (b.ancillary()) {  // skippable metadata: seek index, provenance, ...
      if (split_) {
        if (const Errc s = split_->block(b, Bytes(hdr_).first(kBlockHeaderSize)); s != Errc::ok) return s;
      }
      skip_verify_ = split_ == nullptr;  // in split mode the worker verifies it
      skip_crc_ = 0;
      st_ = St::skip;
      return Errc::ok;
    }
    if (frame_broken_ && b.type() != BlockType::stored) {
      // R20: the models that would decode this block were trained on lost data.
      open_gap(broken_cause_);
      logical_ += b.usize;
      ++stats_.blocks_lost;
      skip_verify_ = false;
      st_ = St::skip;
      return Errc::ok;
    }
    if (const Errc s = budget_.spend(b.usize); s != Errc::ok) return s;  // R19: before any work
    if (split_) {
      if (const Errc s = split_->block(b, Bytes(hdr_).first(kBlockHeaderSize)); s != Errc::ok) return s;
    }
    payload_len_ = 0;
    st_ = St::payload;
    return Errc::ok;
  }

  Errc on_skip(Bytes& in, Progress& p) noexcept {
    const std::size_t n = static_cast<std::size_t>(std::min<u64>(skip_left_, in.size()));
    if (skip_verify_) skip_crc_ = crc32c(in.first(n), skip_crc_);
    if (split_ && n) {
      if (const Errc s = split_->payload(in.first(n)); s != Errc::ok) return s;
    }
    consume(in, n, p);
    skip_left_ -= n;
    if (skip_left_) return Errc::need_input;
    if (skip_verify_ && skip_crc_ != cur_.payload_crc) {
      if (!recovering()) return Errc::payload_corrupt;
      note_damage();  // metadata only: no content lost
    }
    return then(St::item);
  }

  Errc on_payload(Bytes& in, Progress& p) noexcept {
    const std::size_t n = std::min<std::size_t>(cur_.csize - payload_len_, in.size());
    if (split_) {  // forward; a worker decodes and verifies this block
      if (n) {
        if (const Errc s = split_->payload(in.first(n)); s != Errc::ok) return s;
      }
      payload_len_ += n;
      consume(in, n, p);
      if (payload_len_ < cur_.csize) return Errc::need_input;
      const u32 check = cur_.type() == BlockType::stored ? cur_.payload_crc : cur_.content_crc;
      list_crc_ = crc32c(le32(check), list_crc_);
      logical_ += cur_.usize;
      return then(St::item);
    }
    if (n) std::memcpy(payload_.data() + payload_len_, in.data(), n);
    payload_len_ += n;
    consume(in, n, p);
    if (payload_len_ < cur_.csize) return Errc::need_input;

    const Bytes payload = Bytes(payload_).first(cur_.csize);
    if (crc32c(payload) != cur_.payload_crc) return lose_block(Errc::payload_corrupt);
    const u64 start = window_.pos();
    u32 check = cur_.payload_crc;  // STORED: content == payload
    if (cur_.type() == BlockType::stored) {
      window_.append(payload);
    } else {
      if (const Errc e = codec_.decode_block(payload, cur_.usize, window_); e != Errc::ok) return lose_block(e);
      if (window_.pos() - start != cur_.usize) return lose_block(Errc::payload_corrupt);
      check = window_crc(start, cur_.usize);
      if (check != cur_.content_crc) return lose_block(Errc::content_mismatch);
    }
    // Verified: only now may these bytes reach the sink (R20 verify-then-emit).
    list_crc_ = crc32c(le32(check), list_crc_);
    if (frame_broken_) {  // salvaged STORED block inside a broken frame
      ++stats_.blocks_salvaged;
      if (const Errc g = close_gap(logical_); g != Errc::ok) return g;
    } else {
      ++stats_.blocks_ok;
    }
    logical_ += cur_.usize;
    emit_pos_ = start;
    emit_left_ = cur_.usize;
    return then(St::emit);
  }

  Errc on_end() noexcept {
    EndTrailer t;
    if (const Errc e = parse_end(staged(), t); e != Errc::ok) return damaged(e);
    hdr_len_ = 0;
    budget_.earn(kEndSize);
    if (!recovering()) {
      const bool ok = t.total_out == logical_ && t.frame_count == frames_ && t.list_check == list_crc_ &&
                      (!params_.content_size || *params_.content_size == logical_);
      if (!ok) return Errc::content_mismatch;
      stats_.content_verified = true;
    } else {
      if (t.total_out > logical_) {  // data missing at the tail
        note_damage();
        open_gap(Errc::sequence_gap);
      }
      if (const Errc g = close_gap(std::max(t.total_out, logical_)); g != Errc::ok) return g;
      stats_.content_verified =
          stream_damage_ == 0 && t.total_out == logical_ && t.list_check == list_crc_ && t.frame_count == frames_;
      logical_ = std::max(t.total_out, logical_);
    }
    if (split_) {
      if (const Errc s = split_->end(t, stats_.content_verified, in_offset_, stream_index_); s != Errc::ok) return s;
    }
    return then(St::done);
  }

  Errc on_emit(MutBytes& out, Progress& p) noexcept {
    while (emit_left_) {
      if (out.empty()) return Errc::output_full;
      const Bytes piece = window_.contiguous(emit_pos_, std::min<u64>(emit_left_, out.size()));
      std::memcpy(out.data(), piece.data(), piece.size());
      out = out.subspan(piece.size());
      p.produced += piece.size();
      emit_pos_ += piece.size();
      emit_left_ -= piece.size();
      job_produced_ += piece.size();
    }
    st_ = St::item;
    return Errc::ok;
  }

  Errc on_fill(MutBytes& out, Progress& p) noexcept {
    while (fill_left_) {
      if (out.empty()) return Errc::output_full;
      const std::size_t n = static_cast<std::size_t>(std::min<u64>(fill_left_, out.size()));
      std::memset(out.data(), std::to_integer<int>(opt_.fill), n);
      out = out.subspan(n);
      p.produced += n;
      fill_left_ -= n;
    }
    return then(after_fill_);
  }

  // ---- R20 recovery ----

  void note_damage() noexcept {
    ++stats_.damage_events;
    ++stream_damage_;
  }

  void open_gap(Errc cause) noexcept {
    if (gap_open_) return;
    gap_open_ = true;
    gap_start_ = logical_;
    gap_cause_ = cause;
    gap_in_ = in_offset_;
  }

  // Ends the open gap at uncompressed offset `at` (>= its start). In fill mode
  // the gap becomes output, so it is charged to the R19 budget like any other.
  [[nodiscard]] Errc close_gap(u64 at) noexcept {
    if (!gap_open_) return Errc::ok;
    gap_open_ = false;
    const u64 len = at - gap_start_;
    if (len == 0) return Errc::ok;
    ++stats_.gaps;
    stats_.bytes_lost += len;
    if (opt_.on_gap) opt_.on_gap(opt_.gap_ctx, GapInfo{stream_index_, gap_start_, len, true, gap_in_, gap_cause_});
    if (opt_.recovery != Recovery::skip_and_fill) return Errc::ok;
    if (const Errc s = budget_.spend(len); s != Errc::ok) return s;
    stats_.bytes_filled += len;
    if (split_) return Errc::ok;  // the pipeline writes fills, in stream order
    if (job_fill_) {
      job_fill_(job_ctx_, job_produced_, len);
      return Errc::ok;
    }
    fill_left_ = len;
    return Errc::ok;
  }

  // Payload or content check failed but the header was valid, so the block's
  // extent is known; the rest of the frame is lost except STORED blocks.
  Errc lose_block(Errc cause) noexcept {
    if (!recovering()) return cause;
    note_damage();
    ++stats_.blocks_lost;
    budget_.refund(cur_.usize);
    open_gap(cause);
    broken_cause_ = cause;
    frame_broken_ = true;
    logical_ += cur_.usize;
    return then(St::item);
  }

  // A header that cannot be trusted: sizes are unknown, so resync on the next frame.
  Errc damaged(Errc cause) noexcept {
    if (!recovering()) return cause;
    if (split_) {
      if (const Errc s = split_->damage(cause, logical_, in_offset_, stream_index_); s != Errc::ok) return s;
    }
    note_damage();
    open_gap(cause);
    return enter_scan();
  }

  // The staged item was not valid: resume the search at byte `from` of it.
  Errc enter_scan(std::size_t from = 1) noexcept {
    in_frame_ = false;
    st_ = St::scan;
    drop_candidate(from);
    return Errc::ok;
  }

  Errc reject(Errc e, bool scanning) noexcept {
    if (!scanning) return damaged(e);
    drop_candidate();
    return Errc::ok;
  }

  // Keeps only the staged suffix that starts at the first marker byte at or
  // after `from`; everything before it is discarded. Nothing is re-read from input.
  void drop_candidate(std::size_t from = 1) noexcept {
    std::size_t keep = 0;
    if (hdr_len_ > from) {
      const auto first = hdr_.begin() + static_cast<std::ptrdiff_t>(from);
      const auto last = hdr_.begin() + static_cast<std::ptrdiff_t>(hdr_len_);
      keep = static_cast<std::size_t>(last - std::find(first, last, kSyncMarker[0]));
      if (keep) std::memmove(hdr_.data(), hdr_.data() + (hdr_len_ - keep), keep);
    }
    budget_.earn(hdr_len_ - keep);
    hdr_len_ = keep;
  }

  // Slides over the input looking for a frame header that passes marker, CRC,
  // stream identity and ordering checks. Candidates are staged in hdr_, so it
  // works across any input chunking and never needs to seek backwards.
  Errc on_scan(Bytes& in, Progress& p) noexcept {
    for (;;) {
      if (hdr_len_ == 0) {
        const auto hit = std::find(in.begin(), in.end(), kSyncMarker[0]);
        const auto skipped = static_cast<std::size_t>(hit - in.begin());
        budget_.earn(skipped);
        consume(in, skipped, p);
        if (in.empty()) return Errc::need_input;
      }
      const bool complete = gather(in, kFrameHeaderSize, p);
      const std::size_t prefix = std::min(hdr_len_, kSyncMarker.size());
      if (!std::equal(hdr_.begin(), hdr_.begin() + static_cast<std::ptrdiff_t>(prefix), kSyncMarker.begin())) {
        drop_candidate();
        continue;
      }
      if (!complete) return Errc::need_input;
      FrameHeader f;
      if (parse_frame_header(staged(), f) != Errc::ok) {
        drop_candidate();
        continue;
      }
      if (const Errc e = accept_frame(f); e != Errc::ok || st_ != St::scan) return e;
    }
  }

  Errc on_truncated() noexcept {
    if (st_ == St::stream_header && stream_index_ > 0 && !recovering()) return Errc::trailing_garbage;
    if (recovering() && params_valid_) {
      if (split_) split_->truncated(logical_, in_offset_, stream_index_);
      open_gap(Errc::truncated);
      gap_open_ = false;
      ++stats_.gaps;
      if (opt_.on_gap) opt_.on_gap(opt_.gap_ctx, GapInfo{stream_index_, gap_start_, 0, false, gap_in_, gap_cause_});
    }
    return Errc::truncated;
  }

  u32 window_crc(u64 from, u64 len) const noexcept {
    u32 c = 0;
    while (len) {
      const Bytes piece = window_.contiguous(from, len);
      c = crc32c(piece, c);
      from += piece.size();
      len -= piece.size();
    }
    return c;
  }

  Codec& codec_;
  DecoderOptions opt_;
  Budget budget_;
  DecodeStats stats_;
  StreamParams params_;
  Window window_;
  std::unique_ptr<std::byte[]> arena_;
  u64 arena_cap_ = 0;
  MutBytes payload_;

  St st_ = St::stream_header;
  St after_fill_ = St::item;
  Errc err_ = Errc::ok;
  std::array<std::byte, kMaxItemSize> hdr_{};
  std::size_t hdr_len_ = 0;
  BlockHeader cur_;
  std::size_t payload_len_ = 0;
  u64 skip_left_ = 0;
  u32 skip_crc_ = 0;
  bool skip_verify_ = false;
  u64 emit_pos_ = 0, emit_left_ = 0, fill_left_ = 0;

  bool params_valid_ = false, in_frame_ = false, frame_broken_ = false, gap_open_ = false;
  u32 stream_index_ = 0;
  u64 logical_ = 0;    // uncompressed offset of the next byte, including lost bytes
  u64 frames_ = 0;     // next expected frame index
  u32 list_crc_ = 0;   // running END check
  u64 in_offset_ = 0;  // total input consumed, for diagnostics
  u64 stream_damage_ = 0;
  u64 gap_start_ = 0, gap_in_ = 0;
  Errc gap_cause_ = Errc::ok, broken_cause_ = Errc::ok;

  SplitSink* split_ = nullptr;  // split mode
  bool job_ = false;            // job mode
  FillFn job_fill_ = nullptr;
  void* job_ctx_ = nullptr;
  u64 job_produced_ = 0;  // bytes emitted in the current job
};

// Drives any push decoder from a non-seekable source (stdin, socket) to a sink
// with two fixed buffers: constant memory regardless of stream length (R18).
template <ByteSource Src, ByteSink Snk, class Dec>
Errc pump(Src& src, Snk& snk, Dec& dec, MutBytes in_buf, MutBytes out_buf) {
  std::size_t have = 0, used = 0;
  bool eof = false;
  for (;;) {
    if (used == have && !eof) {
      have = src.read(in_buf);
      used = 0;
      eof = have == 0;
    }
    const Progress p = dec.decode(Bytes(in_buf).subspan(used, have - used), out_buf, eof);
    used += p.consumed;
    if (p.produced && !snk.write(Bytes(out_buf).first(p.produced))) return Errc::io_error;
    switch (p.status) {
      case Errc::need_input:
      case Errc::output_full: continue;
      case Errc::end_of_stream:
        if (eof && used == have) return Errc::ok;
        continue;
      default: return p.status;
    }
  }
}

// R19 for legacy (v1/v2/classic v3) decoders whose formats never declared
// block sizes: police input credit and output volume at the I/O boundary.
template <ByteSource Src>
struct GuardedSource {
  Src& inner;
  Budget& budget;
  std::size_t read(MutBytes b) {
    const std::size_t n = inner.read(b);
    budget.earn(n);
    return n;
  }
};

template <ByteSink Snk>
struct GuardedSink {
  Snk& inner;
  Budget& budget;
  Errc status = Errc::ok;
  bool write(Bytes b) {
    if (status == Errc::ok) status = budget.spend(b.size());
    return status == Errc::ok && inner.write(b);
  }
};

// ----------------------------------------------------------------- writer ---

// Encoder-side framing (R18). Append-only: it never seeks, so stdout and pipes
// work and the total size is only needed at END. The caller codes the blocks.
template <ByteSink Snk>
class FramedWriter {
 public:
  FramedWriter(Snk& sink, const StreamParams& p, u64 sync_interval) noexcept
      : sink_(sink), p_(p), sync_(sync_interval) {}

  [[nodiscard]] bool begin() { return put(encode_stream_header(p_)); }

  // True when the next block opens a frame: the caller must reset its models
  // to the prior image before coding that block (R20 sync point).
  [[nodiscard]] bool frame_due() const noexcept { return !in_frame_ || logical_ - frame_start_ >= sync_; }

  // `coded` is the CM payload of `content`. Empty, or not smaller than
  // `content`, means STORED; the codec must then roll back its model updates (R12).
  [[nodiscard]] bool block(Bytes content, Bytes coded) {
    if (content.empty() || content.size() > p_.block_capacity()) return false;
    if (frame_due() && !start_frame()) return false;
    const bool stored = coded.empty() || coded.size() >= content.size();
    const Bytes payload = stored ? content : coded;
    BlockHeader b;
    b.tag = static_cast<u8>(stored ? BlockType::stored : BlockType::cm);
    b.usize = static_cast<u32>(content.size());
    b.csize = static_cast<u32>(payload.size());
    b.payload_crc = crc32c(payload);
    b.content_crc = stored ? 0 : crc32c(content);
    list_ = crc32c(le32(stored ? b.payload_crc : b.content_crc), list_);
    logical_ += content.size();
    return put(encode_block_header(b)) && put(payload);
  }

  [[nodiscard]] bool ancillary(u8 kind, Bytes payload) {
    if (payload.size() > kMaxAncillaryBytes || kind > kTagTypeMask) return false;
    BlockHeader b;
    b.tag = static_cast<u8>(kTagAncillary | kind);
    b.csize = static_cast<u32>(payload.size());
    b.payload_crc = crc32c(payload);
    return put(encode_block_header(b)) && put(payload);
  }

  [[nodiscard]] bool end() { return put(encode_end(EndTrailer{logical_, frames_, list_, 0})); }

  u64 logical() const noexcept { return logical_; }

 private:
  bool start_frame() {
    FrameHeader f{p_, frames_, logical_};
    ++frames_;
    in_frame_ = true;
    frame_start_ = logical_;
    return put(encode_frame_header(f));
  }

  template <class T>
  bool put(const T& bytes) {
    return sink_.write(Bytes(bytes));
  }

  Snk& sink_;
  StreamParams p_;
  u64 sync_;
  u64 logical_ = 0, frame_start_ = 0;
  u32 frames_ = 0, list_ = 0;
  bool in_frame_ = false;
};

}  // namespace hc
