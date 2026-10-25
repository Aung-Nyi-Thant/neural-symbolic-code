// hc/base.hpp - types, status codes, integer serialization and saturating arithmetic.
// Part of the header-only container library; hc_framed.hpp includes every part.
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

}  // namespace hc
