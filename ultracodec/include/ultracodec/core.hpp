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
// (Tier 3): block k may be primed with the first prime_bytes of block k-1,
// which the decoder has as soon as it has decoded that much of block k-1.
// With kFlagDedup (Tier 3) the blocks hold the residual of a long-range
// deduplication pass and the section lists the removed repeats. With
// kFlagSnapshot (Tier 4) every block after block 0 is coded by a model that
// has already read block 0 (snapshot warm-start, tier4_enterprise.cpp).
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
  Tier4_Enterprise = 4,  // 2 MiB blocks warm-started from block 0, 22-context CM
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
  static constexpr std::uint8_t kFlagSnapshot = 4;   // Tier 4: blocks 1.. start from block 0's model

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
  // Tier 1 entropy mode (recorded in the archive): entropy-coded sequences, about 1.4x smaller than
  // the raw LZ4-layout mode and slower to decode. false = the raw mode.
  bool tier1_entropy = true;
  // Tier 3 compact format (recorded in the archive): 16-bit hashed counters and a rANS stream per
  // 64 KiB, about 47 MiB less per worker. false writes the format before it.
  bool tier3_compact = true;
  bool tier4_primer = true;             // Tier 4: model the built-in primer first (archive records it)
  // Tier 4 context-table layout, recorded in the archive: 0 full (~194 MiB per model), 1 balanced
  // (~158 MiB, +0.2% size), 2 compact (~138 MiB, +0.5%), 3 small (~74 MiB, +2.2%), 4 rebalanced
  // (~159 MiB, -0.2% vs balanced), 5 lean (~123 MiB, +0.07% vs balanced). -1 picks rebalanced,
  // else lean, else small: the first whose snapshot plus four workers fit memory_budget.
  int tier4_tables = -1;
  // Tier 4 long-match bypass (recorded in the archive): bytes inside a match of 32+ bytes cost one
  // flag instead of the full model when predicted right. ~1.1x speed for ~+0.7% size.
  bool tier4_bypass = false;
  // Tier 4 mixer weights as int16 instead of int32 (recorded in the archive): half the weight
  // memory and SIMD work, same density. false writes the original wide mixer.
  bool tier4_narrow_mixer = true;
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
