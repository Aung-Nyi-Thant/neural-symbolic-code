// Tier 4: Enterprise TR3. 4 MiB blocks primed with the first 256 KiB of
// their predecessor, through the full CM: order-1 direct counters, 11 hashed
// contexts (orders 2-6 and 8, word, word bigram, column, two sparse), two
// match models (5- and 16-byte minimum), three first-layer mixers (by
// partial byte, by match state x bit, by previous byte x bit) feeding a
// final mixer, and two APMs (order 1, hashed order 2). Tables are sized from
// the memory budget (default 3.5 GiB over all workers) and huge-page backed.
#include "cm_codec.hpp"
#include "tiers.hpp"

namespace ultracodec::detail {
namespace {

struct FullProfile {
  static constexpr unsigned kHashed = 11;
  static constexpr unsigned kMatch = 2;
  static constexpr unsigned kMatchMin[2] = {5, 16};
  static constexpr unsigned kMatchLog2 = 22;
  static constexpr bool kTwoLayer = true;
  static constexpr bool kSecondApm = true;
  static constexpr int kCounterLimit = 255;
  static constexpr int kO1Limit = 255;
  static constexpr int kMixerLr = 5;
  static constexpr int kFinalLr = 3;
  static constexpr int kInitWeight = 3000;
  static constexpr bool kApmBypass = true;
  static constexpr int kBypassSt = 1950;
  static constexpr bool kMatchSkip = false;
  static constexpr u32 kSkipLen = 48;
  static constexpr int kSkipMargin = 4;
  static constexpr unsigned kBlockLog2 = 22;
  static constexpr std::size_t kPrimeBytes = 256 * 1024;
  static constexpr std::size_t kDefaultBudget = std::size_t{3584} << 20;
  static constexpr std::size_t kTableBytes = std::size_t{640} << 20;
  static constexpr std::size_t kMaxTablePerByte = 256;
  static constexpr u64 kMaxTableBytes = u64{32} << 30;

  static constexpr std::size_t mixer_rows(unsigned k) { return k == 0 ? 256 : k == 1 ? 32 : 2048; }
  static ULTRA_INLINE void contexts(const ByteState& s, u64* out) noexcept {
    const u64 c4 = s.c4, c8 = s.c8;
    out[0] = c4 & 0xFFFF;
    out[1] = c4 & 0xFFFFFF;
    out[2] = c4;
    out[3] = c4 | ((c8 & 0xFF) << 32);
    out[4] = c4 | ((c8 & 0xFFFF) << 32);
    out[5] = c4 | (c8 << 32);
    out[6] = s.w0 ^ ((c4 & 0xFF) << 56);
    out[7] = s.w0 + s.w1 * 0x2545F4914F6CDD1Dull;
    out[8] = s.above | ((c4 & 0xFF) << 8) | (static_cast<u64>(std::min<u32>(s.col, 255)) << 16);
    out[9] = c4 & 0xFFFF0000;
    out[10] = c4 & 0xFF00FF00;
  }
};

}  // namespace

std::unique_ptr<ICompressionEngine> make_tier4(const EngineOptions& o) {
  return std::make_unique<CmEngine<FullProfile>>(TierLevel::Tier4_Enterprise, "Tier4 Enterprise TR3 (full CM + rANS)", o);
}

}  // namespace ultracodec::detail
