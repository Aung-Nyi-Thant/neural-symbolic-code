// ultracodec core: tier identifiers, the archive header and the engine
// interface every tier implements (Strategy pattern; engine.hpp holds the
// factory).
//
// Archive layout (all integers little-endian):
//
//   ArchiveHeader          32 bytes, CRC-protected (see below)
//   [dedup section]        only with kFlagDedup: u32 size + section (dedup.hpp)
//   block table            block_count x {u32 payload_size, u32 raw_crc32c}
//   u32                    CRC-32C of the block table
//   payloads               block_count payloads, concatenated
//
// Each payload starts with a one-byte method (0 = stored, 1 = coded by the
// tier). Blocks are independent apart from the optional wavefront prefix
// (Tiers 3 and 4): block k may be primed with the first prime_bytes of block
// k-1, which the decoder has as soon as it has decoded that much of block k-1.
// With kFlagDedup (Tiers 3 and 4) the blocks hold the residual of a
// long-range deduplication pass and the section lists the removed repeats.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ultracodec {

enum class TierLevel : std::uint8_t {
  Tier1_HyperSpeed = 1,  // byte-level LZ77, LZ4-class speed
  Tier2_Consumer = 2,    // lazy hash-chain LZ77 + 8-way interleaved static rANS
  Tier3_Balanced = 3,    // 1 MiB primed blocks, lean 6-context CM + binary rANS
  Tier4_Enterprise = 4,  // large blocks, 12-context CM with two mixer layers
};

// Every failure the library reports: malformed or truncated archives, CRC
// mismatches, limits exceeded. Decoders never read or write out of bounds on
// hostile input; they throw this instead.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline constexpr std::uint8_t kMagic[4] = {'U', 'L', 'T', 'R'};
inline constexpr std::uint8_t kFormatVersion = 1;

// The fixed 32-byte archive header.
struct ArchiveHeader {
  static constexpr std::size_t kSize = 32;
  static constexpr std::uint8_t kFlagWavefront = 1;  // blocks primed by their predecessor's head
  static constexpr std::uint8_t kFlagDedup = 2;      // blocks hold a long-range dedup residual

  std::uint8_t version = kFormatVersion;
  TierLevel tier = TierLevel::Tier1_HyperSpeed;
  std::uint8_t flags = 0;
  std::uint8_t block_log2 = 20;           // block size = 1 << block_log2
  std::uint64_t uncompressed_size = 0;
  std::uint32_t block_count = 0;          // blocks of the (residual) data
  std::uint32_t prime_bytes = 0;          // wavefront prefix length (0 = off)
  std::uint32_t model_param = 0;          // tier-specific (CM table size, ...)
  // bytes 28..31: CRC-32C of bytes 0..27 (magic included)

  std::size_t block_size() const noexcept { return std::size_t{1} << block_log2; }

  void serialize(std::uint8_t out[kSize]) const noexcept;
  // Throws Error when the bytes are not a valid version-1 header.
  static ArchiveHeader parse(std::span<const std::uint8_t> bytes);
};

// Knobs shared by all tiers. Zero means "the tier's default".
struct EngineOptions {
  unsigned threads = 0;                 // 0 = std::thread::hardware_concurrency()
  std::size_t block_size = 0;           // power of two, 64 KiB .. 64 MiB
  std::size_t memory_budget = 0;        // bytes for all model tables (Tiers 3/4)
  std::size_t prime_bytes = SIZE_MAX;   // wavefront prefix; SIZE_MAX = tier default, 0 = off
  std::uint64_t max_output_bytes = std::uint64_t{1} << 36;  // decompress refuses larger archives
};

// What one compress/decompress call measured (filled when requested).
struct EngineStats {
  std::size_t blocks = 0;
  std::size_t stored_blocks = 0;  // blocks the tier could not shrink
  double seconds = 0.0;
};

class ICompressionEngine {
 public:
  virtual ~ICompressionEngine() = default;

  // Thread-safe for concurrent calls on distinct engines; one engine runs one
  // call at a time (it owns per-thread scratch).
  virtual std::vector<std::uint8_t> compress(std::span<const std::uint8_t> input,
                                             EngineStats* stats = nullptr) = 0;
  // Accepts only archives of this engine's tier; engine.hpp's decompress()
  // picks the tier from the header.
  virtual std::vector<std::uint8_t> decompress(std::span<const std::uint8_t> archive,
                                               EngineStats* stats = nullptr) = 0;

  virtual TierLevel get_tier() const noexcept = 0;
  virtual std::string_view get_name() const noexcept = 0;
};

std::string_view tier_name(TierLevel tier) noexcept;

}  // namespace ultracodec
