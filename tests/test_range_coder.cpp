// Range coder: exact round trips under extreme and adversarial probabilities, exact payload length,
// overflow handling and a pinned known-answer vector.
#include <vector>

#include "check.hpp"
#include "corpus.hpp"
#include "format/range_coder.hpp"

namespace {

struct Symbol {
    uint32_t bit;
    uint32_t p1;
};

// Mix of uniform probabilities, both extremes, and bits that contradict a confident prediction.
std::vector<Symbol> make_symbols(size_t n, uint64_t seed) {
    hce::corpus::SplitMix64 rng(seed);
    std::vector<Symbol> s(n);
    for (auto& sym : s) {
        switch (rng.below(5)) {
            case 0: sym.p1 = 1; break;
            case 1: sym.p1 = hce::kProbMax; break;
            case 2: sym.p1 = 1 + rng.below(hce::kProbMax); break;
            case 3: sym.p1 = 32768; break;
            default: sym.p1 = rng.below(2) ? 65000 : 500; break;
        }
        // Mostly follow the prediction, sometimes contradict it.
        const bool likely_one = sym.p1 >= 32768;
        sym.bit = rng.below(8) == 0 ? !likely_one : likely_one;
    }
    return s;
}

std::vector<uint8_t> encode(const std::vector<Symbol>& symbols, size_t capacity, bool* overflowed = nullptr) {
    std::vector<uint8_t> buf(capacity);
    hce::RangeEncoder enc(buf);
    for (const auto& s : symbols) enc.encode(s.bit, s.p1);
    buf.resize(enc.finish());
    if (overflowed != nullptr) *overflowed = enc.overflowed();
    return buf;
}

std::vector<uint32_t> decode(const std::vector<uint8_t>& payload, const std::vector<Symbol>& symbols,
                             bool* exact) {
    hce::RangeDecoder dec(payload);
    std::vector<uint32_t> bits;
    for (const auto& s : symbols) bits.push_back(dec.decode(s.p1));
    *exact = dec.consumed_exactly();
    return bits;
}

void test_round_trip() {
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        const auto symbols = make_symbols(20000, seed);
        bool overflowed = true;
        const auto payload = encode(symbols, 1 << 20, &overflowed);
        CHECK(!overflowed);
        CHECK(payload.size() >= 4);
        bool exact = false;
        const auto bits = decode(payload, symbols, &exact);
        CHECK(exact);
        bool same = true;
        for (size_t i = 0; i < symbols.size(); ++i) same = same && bits[i] == symbols[i].bit;
        CHECK(same);
    }
}

void test_payload_length_is_exact() {
    const auto symbols = make_symbols(5000, 99);
    const auto payload = encode(symbols, 1 << 20);
    bool exact = true;
    auto truncated = payload;
    truncated.pop_back();
    decode(truncated, symbols, &exact);
    CHECK(!exact);
    auto padded = payload;
    padded.push_back(0);
    decode(padded, symbols, &exact);
    CHECK(!exact);
}

void test_out_of_range_probabilities_are_clamped_identically() {
    std::vector<Symbol> raw{{1, 0}, {0, 0}, {1, 70000}, {0, 70000}, {1, 65536}};
    std::vector<Symbol> clamped{{1, 1}, {0, 1}, {1, 65535}, {0, 65535}, {1, 65535}};
    CHECK(encode(raw, 64) == encode(clamped, 64));
    bool exact = false;
    const auto bits = decode(encode(raw, 64), raw, &exact);
    CHECK(exact);
    for (size_t i = 0; i < raw.size(); ++i) CHECK(bits[i] == raw[i].bit);
}

void test_overflow_is_reported() {
    const auto symbols = make_symbols(4000, 7);
    bool overflowed = false;
    encode(symbols, 16, &overflowed);
    CHECK(overflowed);
}

void test_known_answer() {
    // Pinned output: any change here is a format change.
    std::vector<Symbol> symbols;
    const uint64_t pattern = 0xDEADBEEFCAFEF00Dull;
    for (int i = 63; i >= 0; --i) {
        symbols.push_back({static_cast<uint32_t>((pattern >> i) & 1u), 1000u + 1000u * static_cast<uint32_t>(i)});
    }
    const std::vector<uint8_t> expected{0xEB, 0xF4, 0xEF, 0x07, 0x33, 0x25, 0xA7,
                                        0x77, 0xAE, 0x41, 0x55, 0x30, 0x00};
    CHECK(encode(symbols, 64) == expected);
    bool exact = false;
    const auto bits = decode(expected, symbols, &exact);
    CHECK(exact);
    for (size_t i = 0; i < symbols.size(); ++i) CHECK(bits[i] == symbols[i].bit);
}

}  // namespace

int main() {
    test_round_trip();
    test_payload_length_is_exact();
    test_out_of_range_probabilities_are_clamped_identically();
    test_overflow_is_reported();
    test_known_answer();
    return hce_test::finish("range_coder");
}
