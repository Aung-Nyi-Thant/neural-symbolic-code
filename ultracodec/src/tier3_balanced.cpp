// Tier 3: Balanced / Fast Ultra. A long-range dedup pass (cross-block
// repeats >= 40 bytes, any repeat >= 256), then 1 MiB blocks, each primed
// with the first 64 KiB of its predecessor (wavefront), through a lean
// 6-context CM: order-1 direct counters, hashed orders 2, 3, 4 and a word
// context, one match model; one mixer (weight row by match state and partial
// byte) and one APM with confident-prediction bypass; two-state binary rANS.
// Blocks run in parallel.
//
// Measured trade-off (ultracodec/README.md): a 5th hashed context (order 6)
// gains 0.8% density for ~25% less speed; dropping order 3 gains ~30% speed
// for 0.6% density.
#include "cm_codec.hpp"
#include "tiers.hpp"

namespace ultracodec::detail {
// Profiles have external linkage: some knobs are read only by disabled
// (if constexpr) features, which Clang would flag in an anonymous namespace.
namespace profiles {

struct LeanProfile {
  static constexpr unsigned kHashed = 4;
  static constexpr unsigned kMatch = 1;
  static constexpr unsigned kMatchMin[1] = {6};
  static constexpr unsigned kMatchLog2 = 20;
  static constexpr bool kConfMaps = false;
  static constexpr bool kTwoLayer = false;
  static constexpr bool kSecondApm = false;
  static constexpr int kCounterLimit = 16;
  static constexpr int kO1Limit = 255;
  static constexpr int kMixerLr = 6;
  static constexpr int kFinalLr = 2;
  static constexpr int kInitWeight = 5000;
  static constexpr bool kApmBypass = true;
  static constexpr int kBypassSt = 1900;
  static constexpr bool kMatchSkip = false;
  static constexpr u32 kSkipLen = 32;
  static constexpr int kSkipMargin = 8;
  static constexpr std::size_t kDedupFar = 40;   // long-range dedup: cross-block repeats
  static constexpr std::size_t kDedupNear = 256;  // and repeats this long anywhere
  static constexpr unsigned kBlockLog2 = 20;
  static constexpr std::size_t kPrimeBytes = 64 * 1024;
  static constexpr std::size_t kDefaultBudget = std::size_t{512} << 20;
  static constexpr std::size_t kTableBytes = std::size_t{32} << 20;
  static constexpr std::size_t kMaxTablePerByte = 128;
  static constexpr u64 kMaxTableBytes = u64{8} << 30;

  static constexpr std::size_t mixer_rows(unsigned) { return 4 * 256; }
  static ULTRA_INLINE void contexts(const ByteState& s, u64* out) noexcept {
    out[0] = s.c4 & 0xFFFF;
    out[1] = s.c4 & 0xFFFFFF;
    out[2] = s.c4;
    out[3] = s.w0 ^ (static_cast<u64>(s.c4 & 0xFF) << 56);
  }
};

}  // namespace profiles

std::unique_ptr<ICompressionEngine> make_tier3(const EngineOptions& o) {
  return std::make_unique<CmEngine<profiles::LeanProfile>>(TierLevel::Tier3_Balanced, "Tier3 Balanced (lean CM + rANS)", o);
}

}  // namespace ultracodec::detail
