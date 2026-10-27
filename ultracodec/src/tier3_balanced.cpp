// Tier 3: Balanced / Fast Ultra. 1 MiB blocks, each primed with the first
// 64 KiB of its predecessor (wavefront), through a lean CM: order-1 direct
// counters, 5 hashed contexts (orders 2, 3, 4, 6 and a word context), one
// match model, one mixer (weight row by match state and partial byte) and
// one APM, coded with two-state binary rANS. Blocks run in parallel.
#include "cm_codec.hpp"
#include "tiers.hpp"

namespace ultracodec::detail {
namespace {

struct LeanProfile {
  static constexpr unsigned kHashed = 5;
  static constexpr unsigned kMatch = 1;
  static constexpr unsigned kMatchMin[1] = {6};
  static constexpr unsigned kMatchLog2 = 20;
  static constexpr bool kTwoLayer = false;
  static constexpr bool kSecondApm = false;
  static constexpr int kCounterLimit = 127;
  static constexpr int kO1Limit = 255;
  static constexpr int kMixerLr = 6;
  static constexpr int kFinalLr = 2;
  static constexpr int kInitWeight = 5000;
  static constexpr bool kApmBypass = true;
  static constexpr int kBypassSt = 1900;
  static constexpr bool kMatchSkip = false;
  static constexpr u32 kSkipLen = 32;
  static constexpr int kSkipMargin = 8;
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
    out[3] = s.c4 | (static_cast<u64>(s.c8 & 0xFFFF) << 32);
    out[4] = s.w0 ^ (static_cast<u64>(s.c4 & 0xFF) << 56);
  }
};

}  // namespace

std::unique_ptr<ICompressionEngine> make_tier3(const EngineOptions& o) {
  return std::make_unique<CmEngine<LeanProfile>>(TierLevel::Tier3_Balanced, "Tier3 Balanced (lean CM + rANS)", o);
}

}  // namespace ultracodec::detail
