// Static tables of the model arithmetic. They are generated once at startup by integer-only code,
// so every platform and compiler gets bit-identical values (pinned by model_fingerprint()).
#pragma once

#include <array>
#include <cstdint>

namespace hce {

// Logits are Q8 fixed point (256 = 1.0 nat). The mixer output is clamped to [kLogitMin, kLogitMax].
inline constexpr int32_t kLogitMin = -3071;
inline constexpr int32_t kLogitMax = 3071;
inline constexpr int32_t kSquashOffset = 3072;  // squash index = logit + kSquashOffset
inline constexpr int kSquashSize = 6144;
inline constexpr int kStretchSize = 4096;  // stretch index = p16 >> 4
inline constexpr int kRecipSize = 1024;
// Bit-history counts saturate at kStateCap; with the discount rule this yields kNumStates states.
inline constexpr int kStateCap = 35;
inline constexpr uint32_t kNumStates = 254;

struct Tables {
    // squash[x + 3072] = round(65536 / (1 + e^(-x/256))), clamped to [1, 65535].
    std::array<uint16_t, kSquashSize> squash;
    // stretch[i] = smallest logit x with squash(x) >= 16*i + 8: the inverse of squash.
    std::array<int16_t, kStretchSize> stretch;
    // recip[n] = floor(2^17 / (2n + 3)), i.e. 2^16 / (n + 1.5): the StateMap adaptation rate.
    std::array<uint32_t, kRecipSize> recip;
    // Bit-history state machine. State 0 is the empty history; each state stands for counts (n0, n1).
    std::array<std::array<uint8_t, 2>, 256> next;  // next[state][bit]
    std::array<uint8_t, 256> n0;
    std::array<uint8_t, 256> n1;
    std::array<uint32_t, 256> p22_init;  // (2*n1 + 1) / (2*(n0 + n1) + 2), 22-bit fixed point
};

const Tables& tables();

}  // namespace hce
