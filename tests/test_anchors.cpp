#include <algorithm>
#include <array>
#include <bit>
#include <span>
#include <vector>

#include "check.hpp"
#include "corpus.hpp"
#include "core/kernels/anchor_history.hpp"
#include "core/kernels/cm_model.hpp"
#include "core/kernels/crc32c.hpp"
#include "encoder/planner.hpp"
#include "format/container.hpp"
#include "format/range_coder.hpp"

namespace {
using Bytes = std::vector<uint8_t>;

void test_history_and_planner() {
    hce::AnchorConfig cfg;
    cfg.enabled = true;
    cfg.index_bits = 10;
    cfg.density_bits = 0;  // every boundary after the first 64 bytes is an anchor
    hce::AnchorHistory history(cfg);
    const Bytes block = hce::corpus::random_bytes(128, 99);
    // The history reads bytes from the attached stream, so lay out everything this test observes.
    Bytes stream = block;
    stream.insert(stream.end(), block.begin(), block.begin() + 64);
    stream.push_back(block[64]);
    stream.push_back(static_cast<uint8_t>(block[65] ^ 1));
    for (uint32_t i = 0; i < 3000; ++i) stream.push_back(static_cast<uint8_t>(i * 7));
    history.attach(stream);
    for (size_t i = 0; i < 192; ++i) history.observe(stream[i]);
    CHECK(history.candidate().has_value());
    CHECK(history.candidate()->position == 64);
    CHECK(history.prediction() == block[64]);

    hce::CopyPlanner planner;
    const auto raw = std::span<const uint8_t>(block).subspan(64);
    const auto copy = planner.choose(history, raw);
    CHECK(copy.has_value());
    CHECK(copy->delta == 0 && copy->length == raw.size());
    const auto source = history.copy_source(copy->anchor_id, copy->delta, copy->length);
    CHECK(source == 64);
    for (size_t i = 0; i < raw.size(); ++i) CHECK(raw[i] == history.byte_at(*source + i));
    const auto shifted = planner.choose(history, std::span<const uint8_t>(block).subspan(61));
    CHECK(shifted && shifted->delta == 3 && shifted->length == 67);
    const Bytes mismatch(64, static_cast<uint8_t>(block[64] ^ 0xFF));
    CHECK(!planner.choose(history, mismatch));
    for (unsigned i = 0; i < 400; ++i) planner.observe_literal(0);
    CHECK(!planner.choose(history, raw.first(8)));
    CHECK(hce::bit_cost_q8(32768) == 256);
    CHECK(hce::bit_cost_q8(1) == 4096);
    CHECK(hce::bit_cost_q8(65535) == 1);
    for (uint32_t p = 2; p <= 65535; ++p) CHECK(hce::bit_cost_q8(p) <= hce::bit_cost_q8(p - 1));

    CHECK(!history.copy_source(copy->anchor_id, 65, 8));   // subtraction would underflow
    CHECK(!history.copy_source(copy->anchor_id, 0, 129));  // reaches bytes not yet in history
    CHECK(!history.copy_source(copy->anchor_id, 0, 0));
    CHECK(!history.resolve(0));  // reserved
    CHECK(!history.resolve(history.anchor_count()));
    history.observe(stream[192]);
    CHECK(history.match_length() == 1);
    history.observe(stream[193]);
    CHECK(history.divergence() == 1);
    CHECK(history.match_length() == 0);

    // Ids never expire: thousands of anchors later (with index evictions) the first one still
    // resolves, and every earlier byte stays addressable.
    for (size_t i = 194; i < stream.size(); ++i) history.observe(stream[i]);
    CHECK(history.anchor_count() > 3000);
    CHECK(history.resolve(copy->anchor_id) == 64);
    for (uint64_t i = 0; i < history.position(); ++i) CHECK(history.byte_at(i) == stream[i]);

    auto model_cfg = hce::find_profile(hce::kProfileHybridSmall)->model;
    const uint32_t fingerprint = hce::model_fingerprint(model_cfg);
    model_cfg.anchors.gear_seed ^= 1;
    CHECK(hce::model_fingerprint(model_cfg) != fingerprint);
    model_cfg.anchors.index_bits -= 1;
    CHECK(hce::model_fingerprint(model_cfg) != fingerprint);
    cfg.index_bits = 63;
    CHECK_THROWS(hce::AnchorHistory{cfg});
}

Bytes copy_payload(uint32_t id, uint32_t delta, uint32_t length, unsigned id_bits) {
    Bytes payload(64);
    hce::RangeEncoder enc(payload);
    const auto put = [&](uint32_t value, unsigned bits) {
        for (unsigned i = bits; i > 0; --i) enc.encode((value >> (i - 1)) & 1u, 32768);
    };
    enc.encode(1, hce::kCopyFlagP1);
    put(id, id_bits);
    put(delta, hce::kCopyDeltaBits);
    put(length, 32);
    payload.resize(enc.finish());
    CHECK(!enc.overflowed());
    return payload;
}

Bytes stream_with_copy(const Bytes& prefix, const Bytes& suffix, const Bytes& payload) {
    const hce::Profile& p = *hce::find_profile(hce::kProfileHybridSmall);
    Bytes all = prefix;
    all.insert(all.end(), suffix.begin(), suffix.end());
    Bytes stream{'H', 'C', 'E', 0x1A};
    hce::put_le(stream, hce::kFormatVersion, 1);
    hce::put_le(stream, p.id, 1);
    hce::put_le(stream, 0, 2);
    hce::put_le(stream, hce::model_id(p), 4);
    hce::put_le(stream, all.size(), 8);
    hce::put_le(stream, hce::crc32c(all), 4);
    hce::put_le(stream, hce::crc32c(stream), 4);
    const auto frame = [&](uint8_t type, std::span<const uint8_t> raw, std::span<const uint8_t> packed) {
        hce::put_le(stream, type, 1);
        hce::put_le(stream, raw.size(), 4);
        hce::put_le(stream, packed.size(), 4);
        hce::put_le(stream, hce::crc32c(raw), 4);
        stream.insert(stream.end(), packed.begin(), packed.end());
    };
    for (size_t offset = 0; offset < prefix.size(); offset += p.frame_size) {
        const auto raw = std::span<const uint8_t>(prefix).subspan(offset, std::min<size_t>(p.frame_size, prefix.size() - offset));
        frame(0, raw, raw);
    }
    frame(2, suffix, payload);
    return stream;
}

// The first boundary after `from` where an anchor matches an earlier one: the decoder codes a COPY
// flag only there, so crafted COPY tokens must start at such a point.
struct MatchPoint {
    size_t position;
    hce::AnchorHistory::Match match;
    uint32_t anchor_count;
    unsigned id_bits;
};

MatchPoint first_match_after(const Bytes& stream, size_t from, const hce::AnchorConfig& cfg) {
    hce::AnchorHistory history(cfg);
    history.attach(stream);
    for (size_t i = 0; i < stream.size(); ++i) {
        history.observe(stream[i]);
        if (i + 1 > from && history.candidate()) {
            const uint32_t count = history.anchor_count();
            return {i + 1, *history.candidate(), count, static_cast<unsigned>(std::bit_width(count - 1))};
        }
    }
    CHECK(false);
    return {};
}

void test_decoder_bounds() {
    const auto cfg = hce::find_profile(hce::kProfileHybridSmall)->model.anchors;
    // A random block followed by itself: anchors in the repeat match anchors in the original.
    const Bytes block = hce::corpus::random_bytes(4096, 123);
    Bytes both = block;
    both.insert(both.end(), block.begin(), block.end());
    const MatchPoint at = first_match_after(both, block.size(), cfg);
    const Bytes prefix(both.begin(), both.begin() + static_cast<std::ptrdiff_t>(at.position));
    const auto source = static_cast<std::ptrdiff_t>(at.match.position);
    const Bytes suffix(both.begin() + source, both.begin() + source + 64);
    const uint32_t id = at.match.anchor_id;
    const auto payload = [&at](uint32_t anchor, uint32_t delta, uint32_t length) {
        return copy_payload(anchor, delta, length, at.id_bits);
    };
    const Bytes good = stream_with_copy(prefix, suffix, payload(id, 0, 64));
    Bytes expected = prefix;
    expected.insert(expected.end(), suffix.begin(), suffix.end());
    hce::CodecStats stats;
    CHECK(hce::decompress(good, {}, &stats) == expected);
    CHECK(stats.copies == 1 && stats.copied_bytes == 64);
    CHECK_THROWS(hce::decompress(stream_with_copy(prefix, suffix, payload(id, 0, 65))));  // past the frame
    CHECK_THROWS(hce::decompress(stream_with_copy(prefix, suffix, payload(id, 0, UINT32_MAX))));
    CHECK_THROWS(hce::decompress(stream_with_copy(prefix, suffix, payload(id, 0, 0))));
    CHECK_THROWS(hce::decompress(stream_with_copy(prefix, suffix, payload(id, 0, 7))));  // below the minimum
    CHECK_THROWS(hce::decompress(stream_with_copy(prefix, suffix, payload(0, 0, 64))));  // reserved id
    // The newest anchor sits at the current position: its source would overlap the destination.
    CHECK_THROWS(hce::decompress(stream_with_copy(prefix, suffix, payload(at.anchor_count - 1, 0, 64))));
    if (at.anchor_count < (1u << at.id_bits)) {  // an id that was never issued, if it fits the field
        CHECK_THROWS(hce::decompress(stream_with_copy(prefix, suffix, payload(at.anchor_count, 0, 64))));
    }
    // Without the prefix there is no matched anchor, so the payload decodes as literals and fails.
    CHECK_THROWS(hce::decompress(stream_with_copy({}, suffix, payload(id, 0, 64))));
    Bytes truncated = good;
    truncated.pop_back();
    CHECK_THROWS(hce::decompress(truncated));
    for (size_t i = hce::kHeaderSize + prefix.size() + 2 * hce::kFrameHeaderSize; i < good.size(); ++i) {
        Bytes bad = good;
        bad[i] ^= 0x80;
        // Different terminal range intervals may encode the same bytes; changed output must fail CRC.
        try {
            CHECK(hce::decompress(bad) == expected);
        } catch (const hce::FormatError&) {
        }
    }

    // No window: copy from an anchor more than 2 MiB back (twice the reach of a 1 MiB ring).
    Bytes far = block;
    const Bytes filler = hce::corpus::random_bytes(size_t{2} << 20, 971);
    far.insert(far.end(), filler.begin(), filler.end());
    far.insert(far.end(), block.begin(), block.end());
    const MatchPoint far_at = first_match_after(far, block.size() + filler.size(), cfg);
    CHECK(far_at.position - far_at.match.position > (size_t{2} << 20));
    const Bytes far_prefix(far.begin(), far.begin() + static_cast<std::ptrdiff_t>(far_at.position));
    const auto far_source = static_cast<std::ptrdiff_t>(far_at.match.position);
    const Bytes far_suffix(far.begin() + far_source, far.begin() + far_source + 2048);
    Bytes far_expected = far_prefix;
    far_expected.insert(far_expected.end(), far_suffix.begin(), far_suffix.end());
    const Bytes far_payload = copy_payload(far_at.match.anchor_id, 0, 2048, far_at.id_bits);
    CHECK(hce::decompress(stream_with_copy(far_prefix, far_suffix, far_payload)) == far_expected);
}

void test_roundtrip_and_rollback() {
    Bytes data = hce::corpus::random_bytes(65536, 991);
    const Bytes block = data;
    for (unsigned i = 0; i < 4; ++i) data.insert(data.end(), block.begin(), block.end());
    for (const uint8_t profile : {hce::kProfileHybridSmall, hce::kProfileHybrid}) {
        hce::CodecStats enc, dec;
        const Bytes packed = hce::compress(data, profile, &enc);
        CHECK(hce::decompress(packed, {}, &dec) == data);
        CHECK(enc.copies > 0 && enc.copied_bytes > 200000);
        CHECK(enc.copies == dec.copies && enc.copied_bytes == dec.copied_bytes);
        CHECK(enc.trace == dec.trace && enc.decisions == dec.decisions);
        CHECK(packed == hce::compress(data, profile));
        CHECK(packed.size() < data.size() / 2);
        if (profile == hce::kProfileHybridSmall) CHECK(enc.frames_stored >= 1 && enc.frames_modeled >= 1);
    }

    // End to end: text repeated after 3 MiB of noise is still found and copied.
    Bytes far = hce::corpus::text(65536, 992);
    const Bytes text = far;
    const Bytes noise = hce::corpus::random_bytes(size_t{3} << 20, 993);
    far.insert(far.end(), noise.begin(), noise.end());
    far.insert(far.end(), text.begin(), text.end());
    hce::CodecStats enc, dec;
    const Bytes packed = hce::compress(far, hce::kProfileHybridSmall, &enc);
    CHECK(hce::decompress(packed, {}, &dec) == far);
    CHECK(enc.trace == dec.trace && enc.copied_bytes > 60000);
}
}  // namespace

int main() {
    test_history_and_planner();
    test_decoder_bounds();
    test_roundtrip_and_rollback();
    return hce_test::finish("anchors");
}
