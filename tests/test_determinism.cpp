// Bit-exact reproducibility within a process: repeated and interleaved runs give identical bytes,
// encoder and decoder compute identical probabilities, and model ids are stable. Reproducibility
// across processes, builds and platforms is pinned by the conformance vectors.
#include <set>
#include <vector>

#include "check.hpp"
#include "corpus.hpp"
#include "format/container.hpp"
#include "format/profiles.hpp"

namespace {

using Bytes = std::vector<uint8_t>;

void test_repeatable_output() {
    const Bytes text = hce::corpus::text(100000, 61);
    const Bytes logs = hce::corpus::logs(100000, 62);
    for (const uint8_t profile : {hce::kProfileCm6Small, hce::kProfileCm6}) {
        const Bytes a = hce::compress(text, profile);
        const Bytes other = hce::compress(logs, profile);  // interleave unrelated work
        const Bytes b = hce::compress(text, profile);
        CHECK(a == b);
        CHECK(hce::decompress(a) == hce::decompress(b));
        CHECK(hce::decompress(other) == logs);
    }
}

void test_encoder_decoder_lockstep() {
    const Bytes logs = hce::corpus::logs(150000, 63);
    hce::CodecStats enc;
    hce::CodecStats dec;
    const Bytes packed = hce::compress(logs, hce::kProfileCm6, &enc);
    CHECK(hce::decompress(packed, {}, &dec) == logs);
    CHECK(enc.trace == dec.trace);
    CHECK(enc.decisions == dec.decisions && enc.decisions == 8u * logs.size());
}

void test_model_ids() {
    std::set<uint32_t> ids;
    for (const hce::Profile& p : hce::profiles()) {
        CHECK(hce::model_id(p) == hce::model_id(p));
        ids.insert(hce::model_id(p));
    }
    CHECK(ids.size() == hce::profiles().size());
}

}  // namespace

int main() {
    test_repeatable_output();
    test_encoder_decoder_lockstep();
    test_model_ids();
    return hce_test::finish("determinism");
}
