// Kernel invariants: CRC-32C, generated tables, state machine, StateMap, mixer, model lockstep.
#include <cstring>
#include <span>
#include <vector>

#include "check.hpp"
#include "core/kernels/cm_model.hpp"
#include "core/kernels/crc32c.hpp"
#include "core/kernels/mixer.hpp"
#include "core/kernels/state_map.hpp"
#include "core/kernels/tables.hpp"
#include "corpus.hpp"
#include "format/profiles.hpp"

namespace {

std::span<const uint8_t> bytes_of(const char* s) { return {reinterpret_cast<const uint8_t*>(s), std::strlen(s)}; }

void test_crc32c() {
    CHECK(hce::crc32c(bytes_of("123456789")) == 0xE3069283u);  // standard CRC-32C check value
    CHECK(hce::crc32c(bytes_of("")) == 0u);
    CHECK(hce::crc32c(bytes_of("456789"), hce::crc32c(bytes_of("123"))) == 0xE3069283u);
}

void test_squash_stretch() {
    const hce::Tables& t = hce::tables();
    CHECK(t.squash[hce::kSquashOffset] == 32768);
    for (size_t i = 0; i < t.squash.size(); ++i) {
        CHECK(t.squash[i] >= 1 && t.squash[i] <= 65535);
        if (i > 0) CHECK(t.squash[i] >= t.squash[i - 1]);
    }
    for (int32_t x = hce::kLogitMin; x <= hce::kLogitMax; ++x) {  // exact symmetry
        const auto pos = static_cast<size_t>(x + hce::kSquashOffset);
        const auto neg = static_cast<size_t>(-x + hce::kSquashOffset);
        CHECK(uint32_t{t.squash[pos]} + t.squash[neg] == 65536u);
    }
    for (size_t i = 0; i < t.stretch.size(); ++i) {
        if (i > 0) CHECK(t.stretch[i] >= t.stretch[i - 1]);
        const uint32_t target = static_cast<uint32_t>(i) * 16 + 8;
        const auto at = static_cast<size_t>(t.stretch[i] + hce::kSquashOffset);
        CHECK(t.squash[at] >= target);
        if (at > 0) CHECK(t.squash[at - 1] < target);  // smallest such logit
    }
    CHECK(t.stretch[0] < -2000 && t.stretch[4095] > 2000);
}

void test_state_machine() {
    const hce::Tables& t = hce::tables();
    CHECK(t.n0[0] == 0 && t.n1[0] == 0);
    for (uint32_t s = 0; s < hce::kNumStates; ++s) {
        CHECK(t.n0[s] <= hce::kStateCap && t.n1[s] <= hce::kStateCap);
        CHECK(t.p22_init[s] > 0 && t.p22_init[s] < (1u << 22));
        for (uint32_t bit = 0; bit < 2; ++bit) {
            const uint8_t n = t.next[s][bit];
            CHECK(n < hce::kNumStates);
            const uint32_t same = bit ? t.n1[s] : t.n0[s];
            const uint32_t other = bit ? t.n0[s] : t.n1[s];
            const uint32_t same_after = bit ? t.n1[n] : t.n0[n];
            const uint32_t other_after = bit ? t.n0[n] : t.n1[n];
            CHECK(same_after == (same < hce::kStateCap ? same + 1 : same));
            CHECK(other_after == (other > 2 ? other / 2 + 1 : other));
        }
    }
}

void test_state_map() {
    const hce::Tables& t = hce::tables();
    hce::StateMap map(t, 255);
    uint32_t prev = map.p16(7);
    for (int i = 0; i < 2000; ++i) {
        map.update(7, 1, t);
        const uint32_t p = map.p16(7);
        CHECK(p >= prev && p <= 65535);
        prev = p;
    }
    CHECK(prev > 65000);
    for (int i = 0; i < 2000; ++i) map.update(7, 0, t);
    CHECK(map.p16(7) < 600);
}

void test_mixer_learns_and_stays_bounded() {
    const hce::Tables& t = hce::tables();
    hce::Mixer mixer(3, 1, 1 << 14, 64);
    uint32_t first = 0;
    uint32_t last = 0;
    for (int i = 0; i < 100000; ++i) {
        mixer.set_input(0, 3000);
        mixer.set_input(1, -3072);
        mixer.set_input(2, 256);
        last = mixer.mix(0, t);
        if (i == 0) first = last;
        CHECK(last >= 1 && last <= 65535);
        mixer.update(1);
    }
    CHECK(last > first);
    CHECK(last >= 65000);  // converges toward the clamp without overflowing; rounded updates stall just below it
}

void test_model_lockstep() {
    // Two independent models fed the same bits produce the same probabilities, bit for bit.
    const auto data = hce::corpus::text(20000, 42);
    const hce::ModelConfig& cfg = hce::find_profile(hce::kProfileCm6Small)->model;
    hce::CmModel a(cfg);
    hce::CmModel b(cfg);
    bool same = true;
    for (const uint8_t byte : data) {
        for (int i = 7; i >= 0; --i) {
            const uint32_t bit = (uint32_t{byte} >> i) & 1u;
            const uint32_t pa = a.predict();
            const uint32_t pb = b.predict();
            same = same && pa == pb && pa >= 1 && pa <= 65535;
            a.update(bit);
            b.update(bit);
        }
    }
    CHECK(same);
    CHECK(a.at_byte_boundary());
}

void test_fingerprint() {
    const hce::ModelConfig cfg = hce::find_profile(hce::kProfileCm6)->model;
    CHECK(hce::model_fingerprint(cfg) == hce::model_fingerprint(cfg));
    hce::ModelConfig other = cfg;
    other.mixer_lr = static_cast<uint8_t>(cfg.mixer_lr + 1);
    CHECK(hce::model_fingerprint(other) != hce::model_fingerprint(cfg));
    hce::ModelConfig bad = cfg;
    bad.num_orders = 0;
    CHECK_THROWS(hce::CmModel(bad));
}

}  // namespace

int main() {
    test_crc32c();
    test_squash_stretch();
    test_state_machine();
    test_state_map();
    test_mixer_learns_and_stays_bounded();
    test_model_lockstep();
    test_fingerprint();
    return hce_test::finish("kernels");
}
