// ncm_tables.hpp - fixed-point tables of the neural CM codec (R14 spec constants).
//
// Probabilities are Q16: P(bit = 1) * 65536. Logits ("stretch domain") are Q8:
// ln(p / (1 - p)) * 256. Every table is integer data or is derived from integer
// data at compile time, so all platforms compute bit-identical predictions.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace hce::ncm {

inline constexpr int kStretchMax = 2047;  // model inputs: logits clamped to +-8
inline constexpr int kLogitMax = 3071;    // mixer outputs: logits clamped to +-12

// squash(x) = 65536 / (1 + e^(-x/256)), sampled every 64 units over [-3072, 3072]
// (logits -12..12) and interpolated linearly. These 97 integers are part of the
// format: the encoder and the decoder must use exactly these values.
inline constexpr std::array<std::int32_t, 97> kSquashKnots{
    0,     1,     1,     1,     1,     1,     2,     2,     3,     4,     5,     6,
    8,     10,    13,    17,    22,    28,    36,    47,    60,    77,    98,    126,
    162,   208,   267,   342,   439,   562,   720,   922,   1179,  1506,  1921,  2446,
    3108,  3938,  4971,  6249,  7812,  9702,  11955, 14595, 17625, 21025, 24743, 28693,
    32768, 36843, 40793, 44511, 47911, 50941, 53581, 55834, 57724, 59287, 60565, 61598,
    62428, 63090, 63615, 64030, 64357, 64614, 64816, 64974, 65097, 65194, 65269, 65328,
    65374, 65410, 65438, 65459, 65476, 65489, 65500, 65508, 65514, 65519, 65523, 65526,
    65528, 65530, 65531, 65532, 65533, 65534, 65534, 65535, 65535, 65535, 65535, 65535,
    65536,
};

// Logit (Q8) -> probability (Q16).
constexpr int squash(int x) noexcept {
  x = std::clamp(x, -3072, 3071);
  const int i = (x + 3072) >> 6;
  const int f = (x + 3072) & 63;
  return (kSquashKnots[static_cast<std::size_t>(i)] * (64 - f) + kSquashKnots[static_cast<std::size_t>(i) + 1] * f +
          32) >> 6;
}

// squash() as a table over its whole domain, saturated to 65535 so it fits 16
// bits: squash16(x) == min(squash(x), 65535), one load instead of an
// interpolation. Every caller clamps to 65535 anyway or never reaches the top.
consteval std::array<std::uint16_t, 6144> make_squash16() {
  std::array<std::uint16_t, 6144> t{};
  for (int x = -3072; x < 3072; ++x)
    t[static_cast<std::size_t>(x + 3072)] = static_cast<std::uint16_t>(std::min(squash(x), 65535));
  return t;
}
inline constexpr std::array<std::uint16_t, 6144> kSquash16 = make_squash16();
// Precondition: -3072 <= x <= 3071 (callers clamp to kStretchMax or kLogitMax).
constexpr int squash16(int x) noexcept { return kSquash16[static_cast<std::size_t>(x + 3072)]; }

// stretch = squash^-1 for 12-bit probabilities, computed at compile time with
// integers only: for each bucket centre, the smallest x with squash(x) >= it
// (3071 if none). squash is monotonic, so one forward sweep finds them all.
consteval std::array<std::int16_t, 4096> make_stretch() {
  std::array<std::int16_t, 4096> t{};
  int x = -3072;
  for (int i = 0; i < 4096; ++i) {
    const int target = i * 16 + 8;  // bucket centre, Q16
    while (x < 3071 && squash(x) < target) ++x;
    t[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(std::clamp(x, -kStretchMax, kStretchMax));
  }
  return t;
}
inline constexpr std::array<std::int16_t, 4096> kStretch = make_stretch();

// Probability (Q12) -> logit (Q8), clamped to +-kStretchMax.
constexpr int stretch12(int p12) noexcept { return kStretch[static_cast<std::size_t>(p12)]; }
// Probability (Q16) -> logit (Q8).
constexpr int stretch16(int p16) noexcept { return kStretch[static_cast<std::size_t>(p16) >> 4]; }

// Adaptation rates (Q16) of the counters: 1/(1.25 + 0.75 n) after n
// observations, so a new context learns in a few hits and an old one settles.
// Context counters stop at n = 15 (rate 1/12.5: they must keep tracking
// nonstationary data); decision counters continue up to n = 1023.
consteval std::array<std::int32_t, 1024> make_rates() {
  std::array<std::int32_t, 1024> r{};
  for (int n = 0; n < 1024; ++n) {
    const int d = 20 + 12 * n;  // (1.25 + 0.75 n) in Q4
    r[static_cast<std::size_t>(n)] = (65536 * 16 + d / 2) / d;
  }
  return r;
}
inline constexpr std::array<std::int32_t, 1024> kRate = make_rates();

static_assert(squash(0) == 32768 && squash(-3072) == 0 && squash(3071) == 65536);
static_assert(squash16(0) == 32768 && squash16(-3072) == 0 && squash16(3071) == 65535);
static_assert(stretch16(32768) >= 0 && stretch16(32767) <= 0);
static_assert([] {
  for (std::size_t i = 1; i < kSquashKnots.size(); ++i)
    if (kSquashKnots[i] < kSquashKnots[i - 1]) return false;
  for (std::size_t i = 1; i < kStretch.size(); ++i)
    if (kStretch[i] < kStretch[i - 1]) return false;
  return true;
}(), "squash and stretch must be monotonic");
static_assert(kRate[0] == 52429 && kRate[1] == 32768 && kRate[15] == 5243 && kRate[255] == 340 && kRate[1023] == 85);

}  // namespace hce::ncm
