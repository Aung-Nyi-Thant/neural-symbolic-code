#include "ultracodec/engine.hpp"

#include "tiers.hpp"

namespace ultracodec {

std::unique_ptr<ICompressionEngine> create_engine(TierLevel tier, const EngineOptions& options) {
  switch (tier) {
    case TierLevel::Tier1_HyperSpeed: return detail::make_tier1(options);
    case TierLevel::Tier2_Consumer: return detail::make_tier2(options);
    case TierLevel::Tier3_Balanced: return detail::make_tier3(options);
    case TierLevel::Tier4_Enterprise: return detail::make_tier4(options);
  }
  throw Error("ultracodec: unknown tier");
}

ArchiveHeader peek_header(std::span<const std::uint8_t> archive) { return ArchiveHeader::parse(archive); }

std::vector<std::uint8_t> compress(TierLevel tier, std::span<const std::uint8_t> input, const EngineOptions& options) {
  return create_engine(tier, options)->compress(input);
}

std::vector<std::uint8_t> decompress(std::span<const std::uint8_t> archive, const EngineOptions& options) {
  return create_engine(peek_header(archive).tier, options)->decompress(archive);
}

}  // namespace ultracodec
