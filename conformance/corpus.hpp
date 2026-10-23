// Deterministic inputs for the conformance vectors and unit tests. Integer-only generation, and no
// <random>: its distributions differ between standard libraries, which would break golden vectors.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hce::corpus {

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() noexcept {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform enough for test data (modulo bias is irrelevant here).
    uint32_t below(uint32_t n) noexcept { return static_cast<uint32_t>(next() % n); }

private:
    uint64_t state_;
};

struct Vector {
    std::string name;
    std::vector<uint8_t> data;
};

std::vector<uint8_t> random_bytes(size_t n, uint64_t seed);
std::vector<uint8_t> text(size_t n, uint64_t seed);  // pseudo-English prose
std::vector<uint8_t> logs(size_t n, uint64_t seed);  // templated service log lines

// The named vectors pinned by conformance/golden_v1.txt.
std::vector<Vector> conformance_vectors();

}  // namespace hce::corpus
