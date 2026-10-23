// Container: round trips across profiles and sizes, STORED/MODELED frame choice, bounded expansion,
// and rejection of corrupted, truncated, padded and unsupported streams.
#include <span>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "core/kernels/crc32c.hpp"
#include "corpus.hpp"
#include "format/container.hpp"

namespace {

using Bytes = std::vector<uint8_t>;

size_t frame_count(size_t size, uint8_t profile) {
    const size_t frame = hce::find_profile(profile)->frame_size;
    return (size + frame - 1) / frame;
}

bool round_trips(const Bytes& data, uint8_t profile, hce::CodecStats* enc_out = nullptr) {
    hce::CodecStats enc;
    hce::CodecStats dec;
    const Bytes packed = hce::compress(data, profile, &enc);
    const Bytes unpacked = hce::decompress(packed, {}, &dec);
    if (enc_out != nullptr) *enc_out = enc;
    const size_t bound = hce::kHeaderSize + frame_count(data.size(), profile) * hce::kFrameHeaderSize + data.size();
    return unpacked == data && packed.size() <= bound && enc.trace == dec.trace && enc.decisions == dec.decisions &&
           enc.frames_stored == dec.frames_stored && enc.frames_modeled == dec.frames_modeled &&
           enc.copies == dec.copies && enc.copied_bytes == dec.copied_bytes;
}

void test_corpus_round_trips() {
    const auto vectors = hce::corpus::conformance_vectors();
    for (const hce::Profile& p : hce::profiles()) {
        for (const auto& v : vectors) CHECK(round_trips(v.data, p.id));
    }
}

void test_frame_boundaries() {
    const Bytes text = hce::corpus::text(140000, 11);
    for (const size_t n : {0u, 1u, 3u, 4u, 5u, 100u, 65535u, 65536u, 65537u, 131072u, 131073u}) {
        const Bytes data(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(n));
        CHECK(round_trips(data, hce::kProfileStore));
        CHECK(round_trips(data, hce::kProfileCm6Small));
    }
}

void test_frame_choice() {
    hce::CodecStats s;
    const Bytes noise = hce::corpus::random_bytes(200000, 21);
    CHECK(round_trips(noise, hce::kProfileCm6Small, &s));
    CHECK(s.frames_modeled == 0 && s.frames_stored == 4 && s.decisions == 0);

    const Bytes text = hce::corpus::text(200000, 22);
    CHECK(round_trips(text, hce::kProfileCm6Small, &s));
    CHECK(s.frames_stored == 0 && s.frames_modeled == 4 && s.decisions == 8u * text.size());
    CHECK(hce::compress(text, hce::kProfileCm6Small).size() < text.size() * 4 / 10);

    const Bytes logs = hce::corpus::logs(200000, 23);
    CHECK(hce::compress(logs, hce::kProfileCm6Small).size() < logs.size() / 2);

    for (const auto& v : hce::corpus::conformance_vectors()) {
        if (v.name != "mixed-250k") continue;
        CHECK(round_trips(v.data, hce::kProfileCm6Small, &s));
        CHECK(s.frames_stored >= 1 && s.frames_modeled >= 2);  // the model resumes after STORED frames
    }
}

void reseal_header(Bytes& stream) {
    const uint32_t crc = hce::crc32c(std::span<const uint8_t>(stream.data(), hce::kHeaderSize - 4));
    for (unsigned i = 0; i < 4; ++i) stream[hce::kHeaderSize - 4 + i] = static_cast<uint8_t>(crc >> (8 * i));
}

void test_rejects_bit_flips() {
    // Store profile: every single-bit flip anywhere in the stream is rejected.
    const Bytes small = hce::corpus::text(600, 31);
    const Bytes stored = hce::compress(small, hce::kProfileStore);
    for (size_t i = 0; i < stored.size(); ++i) {
        for (unsigned bit = 0; bit < 8; ++bit) {
            Bytes bad = stored;
            bad[i] = static_cast<uint8_t>(bad[i] ^ (1u << bit));
            CHECK_THROWS(hce::decompress(bad));
        }
    }
    // Modeled profile: a deterministic sample of flips across header, frame headers and payload.
    const Bytes text = hce::corpus::text(6000, 32);
    const Bytes packed = hce::compress(text, hce::kProfileCm6Small);
    hce::corpus::SplitMix64 rng(33);
    for (int k = 0; k < 400; ++k) {
        Bytes bad = packed;
        const size_t i = rng.below(static_cast<uint32_t>(bad.size()));
        bad[i] = static_cast<uint8_t>(bad[i] ^ (1u << rng.below(8)));
        CHECK_THROWS(hce::decompress(bad));
    }
}

void test_rejects_truncation_and_padding() {
    const Bytes small = hce::corpus::logs(500, 41);
    const Bytes stored = hce::compress(small, hce::kProfileStore);
    for (size_t n = 0; n < stored.size(); ++n) {
        CHECK_THROWS(hce::decompress(std::span<const uint8_t>(stored.data(), n)));
    }
    const Bytes packed = hce::compress(hce::corpus::logs(5000, 42), hce::kProfileCm6Small);
    for (size_t n = 0; n < packed.size(); n += 7) {
        CHECK_THROWS(hce::decompress(std::span<const uint8_t>(packed.data(), n)));
    }
    Bytes padded = packed;
    padded.push_back(0);
    CHECK_THROWS(hce::decompress(padded));
}

void test_header_validation() {
    const Bytes data = hce::corpus::text(3000, 51);
    const Bytes good = hce::compress(data, hce::kProfileCm6Small);
    CHECK(hce::decompress(good) == data);

    const auto mutated = [&good](size_t offset, uint8_t value) {
        Bytes s = good;
        s[offset] = value;
        reseal_header(s);
        return s;
    };
    CHECK_THROWS(hce::decompress(mutated(4, 2)));     // format version
    CHECK_THROWS(hce::decompress(mutated(5, 200)));   // unknown profile
    CHECK_THROWS(hce::decompress(mutated(5, hce::kProfileCm6)));  // other profile: model id mismatch
    CHECK_THROWS(hce::decompress(mutated(6, 1)));     // flags
    CHECK_THROWS(hce::decompress(mutated(8, static_cast<uint8_t>(good[8] ^ 1))));  // model id
    CHECK_THROWS(hce::decompress(mutated(0, 'X')));   // magic

    hce::DecodeLimits limits;
    limits.max_output = data.size() - 1;
    CHECK_THROWS(hce::decompress(good, limits));
    limits.max_output = data.size();
    CHECK(hce::decompress(good, limits) == data);

    CHECK_THROWS(hce::compress(data, 77));  // unknown profile id
}

}  // namespace

int main() {
    test_corpus_round_trips();
    test_frame_boundaries();
    test_frame_choice();
    test_rejects_bit_flips();
    test_rejects_truncation_and_padding();
    test_header_validation();
    return hce_test::finish("container");
}
