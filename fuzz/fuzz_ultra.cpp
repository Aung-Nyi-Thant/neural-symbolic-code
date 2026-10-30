// fuzz_ultra: ultracodec::decompress() of hostile archives, as `ultra d` runs
// it but with a 64 MiB output cap and one thread. Each input is decoded as
// given and with its header and block-table CRCs recomputed (the per-block
// content CRCs need the decoded bytes and are left alone).
//
// fuzz_ultra takes Tiers 1-3 and fuzz_ultra_t4 (ULTRA_FUZZ_TIER4) Tier 4,
// whose decoder trains its model on a built-in primer first: seconds per
// input under the sanitizers, so it gets a target of its own rather than
// slowing the others down. Each skips archives of the other's tiers.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common.hpp"  // crc32c, load32 (internal header)
#include "ultracodec/engine.hpp"

namespace {

void decode(const std::vector<std::uint8_t>& d) {
  ultracodec::EngineOptions o;
  o.threads = 1;
  o.max_output_bytes = std::uint64_t{64} << 20;
  try {
    (void)ultracodec::decompress(d, o);
  } catch (const ultracodec::Error&) {
  }
}

void fix_crcs(std::vector<std::uint8_t>& d) {
  using namespace ultracodec::detail;
  if (d.size() < ultracodec::ArchiveHeader::kSize) return;
  store32(d.data() + 28, crc32c(d.data(), 28));
  std::size_t table = ultracodec::ArchiveHeader::kSize;
  if (d[6] & ultracodec::ArchiveHeader::kFlagDedup) {
    if (d.size() < table + 4) return;
    table += 4 + std::size_t{load32(d.data() + table)};
  }
  const std::size_t blocks = load32(d.data() + 16);
  if (table > d.size() || (d.size() - table) / 8 < blocks || d.size() - table - blocks * 8 < 4) return;
  store32(d.data() + table + blocks * 8, crc32c(d.data() + table, blocks * 8));
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
#if defined(ULTRA_FUZZ_TIER4)
  constexpr bool kTier4 = true;
#else
  constexpr bool kTier4 = false;
#endif
  if (size > 5 && (data[5] == 4) != kTier4) return 0;  // byte 5: the tier
  std::vector<std::uint8_t> d(data, data + size);
  decode(d);
  fix_crcs(d);
  decode(d);
  return 0;
}
