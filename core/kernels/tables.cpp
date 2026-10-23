#include "core/kernels/tables.hpp"

#include <cstddef>
#include <cstdlib>

#include "core/kernels/no_float.hpp"

namespace hce {
namespace {

// e^(-1/256) in Q62, from its Taylor series. With u = 2^-8, u^k is exactly 2^(62 - 8k) in Q62.
constexpr uint64_t exp_neg_1_256_q62() {
    constexpr uint64_t one = uint64_t{1} << 62;
    const uint64_t even = one + (one >> 16) / 2 + (one >> 32) / 24 + (one >> 48) / 720;
    const uint64_t odd = (one >> 8) + (one >> 24) / 6 + (one >> 40) / 120;
    return even - odd;
}

void build_squash_stretch(Tables& t) {
    constexpr uint64_t kOneQ31 = uint64_t{1} << 31;
    constexpr uint64_t kHalfQ31 = uint64_t{1} << 30;
    constexpr uint64_t r = (exp_neg_1_256_q62() + kHalfQ31) >> 31;  // e^(-1/256) in Q31

    // e[x] = e^(-x/256) in Q31 for x = 0..3072. Both factors are below 2^31, so products fit.
    std::array<uint64_t, kSquashOffset + 1> e{};
    e[0] = kOneQ31;
    for (size_t x = 1; x < e.size(); ++x) e[x] = (e[x - 1] * r + kHalfQ31) >> 31;

    for (size_t i = 0; i < t.squash.size(); ++i) {
        const int64_t x = static_cast<int64_t>(i) - kSquashOffset;
        const uint64_t den = kOneQ31 + e[static_cast<size_t>(x < 0 ? -x : x)];
        uint64_t p = ((uint64_t{1} << 47) + den / 2) / den;  // 65536 / (1 + e^-|x|), rounded
        if (x < 0) p = 65536 - p;                             // exact symmetry
        t.squash[i] = static_cast<uint16_t>(p < 1 ? 1 : (p > 65535 ? 65535 : p));
    }

    for (size_t i = 0; i < t.stretch.size(); ++i) {
        const uint32_t target = static_cast<uint32_t>(i) * 16 + 8;
        size_t lo = 0;
        size_t hi = t.squash.size() - 1;
        while (lo < hi) {  // squash is non-decreasing
            const size_t mid = (lo + hi) / 2;
            if (t.squash[mid] >= target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        t.stretch[i] = static_cast<int16_t>(static_cast<int64_t>(lo) - kSquashOffset);
    }
}

void build_recip(Tables& t) {
    for (uint32_t n = 0; n < t.recip.size(); ++n) t.recip[n] = (uint32_t{1} << 17) / (2 * n + 3);
}

struct Counts {
    uint8_t n0;
    uint8_t n1;
};

// On a bit, its count grows (saturating at kStateCap) and an opposite count above 2 is discounted
// to n/2 + 1, so old evidence fades and the state stays responsive to change.
Counts step(Counts c, uint32_t bit) {
    const auto grow = [](uint8_t n) { return static_cast<uint8_t>(n < kStateCap ? n + 1 : kStateCap); };
    const auto fade = [](uint8_t n) { return static_cast<uint8_t>(n > 2 ? n / 2 + 1 : n); };
    return bit ? Counts{fade(c.n0), grow(c.n1)} : Counts{grow(c.n0), fade(c.n1)};
}

void build_states(Tables& t) {
    constexpr size_t kSide = kStateCap + 1;
    std::array<std::array<bool, kSide>, kSide> reached{};
    std::array<Counts, kSide * kSide> queue{};
    size_t head = 0;
    size_t tail = 0;
    reached[0][0] = true;
    queue[tail++] = Counts{0, 0};
    while (head < tail) {
        const Counts c = queue[head++];
        for (uint32_t bit = 0; bit < 2; ++bit) {
            const Counts n = step(c, bit);
            if (!reached[n.n0][n.n1]) {
                reached[n.n0][n.n1] = true;
                queue[tail++] = n;
            }
        }
    }

    // Canonical numbering: by total count, then by n1. State 0 is the empty history.
    std::array<std::array<uint8_t, kSide>, kSide> id{};
    uint32_t count = 0;
    for (size_t total = 0; total <= 2 * kStateCap; ++total) {
        for (size_t n1 = 0; n1 <= total && n1 < kSide; ++n1) {
            const size_t n0 = total - n1;
            if (n0 >= kSide || !reached[n0][n1]) continue;
            if (count >= 256) std::abort();  // cannot happen for kStateCap = 35
            id[n0][n1] = static_cast<uint8_t>(count);
            t.n0[count] = static_cast<uint8_t>(n0);
            t.n1[count] = static_cast<uint8_t>(n1);
            ++count;
        }
    }
    if (count != kNumStates) std::abort();  // the state machine definition changed

    for (uint32_t s = 0; s < 256; ++s) {
        if (s < count) {
            for (uint32_t bit = 0; bit < 2; ++bit) {
                const Counts n = step(Counts{t.n0[s], t.n1[s]}, bit);
                t.next[s][bit] = id[n.n0][n.n1];
            }
        } else {  // unused codes: never produced by next[]
            t.next[s] = {0, 0};
            t.n0[s] = 0;
            t.n1[s] = 0;
        }
        const uint64_t num = (uint64_t{t.n1[s]} * 2 + 1) << 22;
        const uint64_t den = (uint64_t{t.n0[s]} + t.n1[s]) * 2 + 2;
        t.p22_init[s] = static_cast<uint32_t>(num / den);
    }
}

Tables build() {
    Tables t{};
    build_squash_stretch(t);
    build_recip(t);
    build_states(t);
    return t;
}

}  // namespace

const Tables& tables() {
    static const Tables t = build();
    return t;
}

}  // namespace hce
