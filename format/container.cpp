#include "format/container.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/kernels/cm_model.hpp"
#include "core/kernels/crc32c.hpp"
#include "encoder/planner.hpp"
#include "format/range_coder.hpp"
#include "core/kernels/no_float.hpp"

namespace hce {
namespace {

constexpr std::array<uint8_t, 4> kMagic{'H', 'C', 'E', 0x1A};

void trace_step(CodecStats& stats, uint32_t p1, uint32_t bit) noexcept {
    stats.trace = (stats.trace ^ ((uint64_t{p1} << 1) | bit)) * 0x100000001B3ull;
    ++stats.decisions;
}

// COPY anchor ids take just enough bits for every id issued so far; both sides see the same count.
unsigned anchor_id_bits(const AnchorHistory& history) noexcept {
    return static_cast<unsigned>(std::bit_width(history.anchor_count() - 1));
}

// Codes `raw` with the model into `buf`. Returns the payload size, or nothing if the payload would
// not be smaller than the raw bytes (the model has then been advanced and must be rolled back).
std::optional<size_t> encode_frame(CmModel& model, std::span<const uint8_t> raw, std::span<uint8_t> buf,
                                   const AnchorConfig& anchors, CodecStats& stats) {
    RangeEncoder enc(buf.first(raw.size() - 1));
    CodecStats local = stats;
    CopyPlanner planner;
    const auto put = [&](uint32_t value, unsigned bits) {
        for (unsigned i = bits; i > 0; --i) {
            const uint32_t bit = (value >> (i - 1)) & 1u;
            enc.encode(bit, 32768);
            trace_step(local, 32768, bit);
        }
    };
    for (size_t offset = 0; offset < raw.size();) {
        if (anchors.enabled && model.anchors().candidate()) {  // a COPY can only start at a matched anchor
            const auto copy = planner.choose(model.anchors(), raw.subspan(offset));
            const uint32_t flag = copy ? 1u : 0u;
            enc.encode(flag, kCopyFlagP1);
            trace_step(local, kCopyFlagP1, flag);
            if (copy) {
                put(copy->anchor_id, anchor_id_bits(model.anchors()));
                put(copy->delta, kCopyDeltaBits);
                put(copy->length, 32);
                for (uint32_t j = 0; j < copy->length; ++j) model.observe_byte(raw[offset + j]);
                offset += copy->length;
                ++local.copies;
                local.copied_bytes += copy->length;
                if (enc.overflowed()) return std::nullopt;
                continue;
            }
        }
        const uint8_t byte = raw[offset++];
        uint32_t cost_q8 = 0;
        for (int i = 7; i >= 0; --i) {
            const uint32_t bit = (uint32_t{byte} >> i) & 1u;
            const uint32_t p1 = model.predict();
            enc.encode(bit, p1);
            model.update(bit);
            trace_step(local, p1, bit);
            if (anchors.enabled) cost_q8 += bit_cost_q8(bit != 0 ? p1 : 65536u - p1);
        }
        if (anchors.enabled) planner.observe_literal(cost_q8);
        if (enc.overflowed()) return std::nullopt;
    }
    const size_t size = enc.finish();
    if (enc.overflowed()) return std::nullopt;
    stats = local;
    return size;
}

void decode_frame(CmModel& model, std::span<const uint8_t> payload, std::span<uint8_t> out,
                  const AnchorConfig& anchors, CodecStats& stats) {
    RangeDecoder dec(payload);
    const auto get = [&](unsigned bits) {
        uint32_t value = 0;
        for (unsigned i = 0; i < bits; ++i) {
            const uint32_t bit = dec.decode(32768);
            trace_step(stats, 32768, bit);
            value = (value << 1) | bit;
        }
        return value;
    };
    for (size_t offset = 0; offset < out.size();) {
        uint32_t flag = 0;
        if (anchors.enabled && model.anchors().candidate()) {
            flag = dec.decode(kCopyFlagP1);
            trace_step(stats, kCopyFlagP1, flag);
        }
        if (flag != 0) {
            const uint32_t id = get(anchor_id_bits(model.anchors()));
            const uint32_t delta = get(kCopyDeltaBits);
            const uint32_t length = get(32);
            if (length < 8 || length > out.size() - offset) throw FormatError("COPY: invalid length");
            const auto source = model.anchors().copy_source(id, delta, length);
            if (!source) throw FormatError("COPY: reference outside history");
            for (uint32_t j = 0; j < length; ++j) {
                const uint8_t byte = model.anchors().byte_at(*source + j);
                out[offset++] = byte;
                model.observe_byte(byte);
            }
            ++stats.copies;
            stats.copied_bytes += length;
            continue;
        }
        uint32_t c = 0;
        for (int i = 0; i < 8; ++i) {
            const uint32_t p1 = model.predict();
            const uint32_t bit = dec.decode(p1);
            model.update(bit);
            trace_step(stats, p1, bit);
            c = (c << 1) | bit;
        }
        out[offset++] = static_cast<uint8_t>(c);
    }
    if (!dec.consumed_exactly()) throw FormatError("modeled frame: payload length mismatch");
}

}  // namespace

std::vector<uint8_t> compress(std::span<const uint8_t> input, uint8_t profile_id, CodecStats* stats_out) {
    const Profile* profile = find_profile(profile_id);
    if (profile == nullptr) throw std::invalid_argument("unknown profile id " + std::to_string(profile_id));

    const size_t frame_size = profile->frame_size;
    const size_t num_frames = (input.size() + frame_size - 1) / frame_size;
    std::vector<uint8_t> out;
    out.reserve(kHeaderSize + num_frames * kFrameHeaderSize + input.size());  // worst case: all STORED

    out.insert(out.end(), kMagic.begin(), kMagic.end());
    put_le(out, kFormatVersion, 1);
    put_le(out, profile->id, 1);
    put_le(out, 0, 2);  // flags: none defined in v1
    put_le(out, model_id(*profile), 4);
    put_le(out, input.size(), 8);
    put_le(out, crc32c(input), 4);
    put_le(out, crc32c(std::span<const uint8_t>(out)), 4);  // header CRC over bytes [0, 24)

    CodecStats stats;
    std::optional<CmModel> model;
    std::optional<CmModel> snapshot;
    std::vector<uint8_t> buf;
    if (profile->modeled) {
        model.emplace(profile->model);
        model->attach(input);
        snapshot.emplace(*model);
        buf.resize(frame_size);
    }

    for (size_t offset = 0; offset < input.size(); offset += frame_size) {
        const auto raw = input.subspan(offset, std::min(frame_size, input.size() - offset));
        std::optional<size_t> coded;
        if (model) {
            const uint32_t anchor_count = model->anchors().anchor_count();
            *snapshot = *model;  // same-size copy: reuses the snapshot's storage
            coded = encode_frame(*model, raw, buf, profile->model.anchors, stats);
            if (!coded) {
                // The decoder never sees the rejected decisions: restore and only extend history.
                std::swap(*model, *snapshot);
                model->truncate_anchors(anchor_count);  // the anchor list is shared with the snapshot
                for (const uint8_t b : raw) model->observe_byte(b);
            }
        }
        const FrameType type = coded ? (profile->model.anchors.enabled ? FrameType::kHybrid : FrameType::kModeled)
                                     : FrameType::kStored;
        put_le(out, static_cast<uint8_t>(type), 1);
        put_le(out, raw.size(), 4);
        put_le(out, coded ? *coded : raw.size(), 4);
        put_le(out, crc32c(raw), 4);
        if (coded) {
            out.insert(out.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(*coded));
            ++stats.frames_modeled;
        } else {
            out.insert(out.end(), raw.begin(), raw.end());
            ++stats.frames_stored;
        }
    }

    if (stats_out != nullptr) *stats_out = stats;
    return out;
}

std::vector<uint8_t> decompress(std::span<const uint8_t> input, const DecodeLimits& limits,
                                CodecStats* stats_out) {
    Reader r(input);
    const auto header = r.bytes(kHeaderSize, "header");
    Reader h(header);
    const auto magic = h.bytes(kMagic.size(), "magic");
    if (!std::equal(magic.begin(), magic.end(), kMagic.begin())) throw FormatError("not an HCE stream");
    const uint64_t version = h.le(1, "version");
    const uint64_t profile_id = h.le(1, "profile id");
    const uint64_t flags = h.le(2, "flags");
    const uint64_t stored_model_id = h.le(4, "model id");
    const uint64_t size = h.le(8, "size");
    const uint64_t content_crc = h.le(4, "content crc");
    const uint64_t header_crc = h.le(4, "header crc");
    if (header_crc != crc32c(header.first(kHeaderSize - 4))) throw FormatError("header checksum mismatch");
    if (version != kFormatVersion) throw FormatError("unsupported format version " + std::to_string(version));
    if (flags != 0) throw FormatError("unsupported header flags");
    const Profile* profile = find_profile(static_cast<uint8_t>(profile_id));
    if (profile == nullptr) throw FormatError("unknown profile id " + std::to_string(profile_id));
    if (stored_model_id != model_id(*profile)) {
        throw FormatError("model id mismatch: the stream was written by a different model build");
    }
    if (size > limits.max_output) throw FormatError("declared size exceeds the decode limit");

    CodecStats stats;
    std::optional<CmModel> model;
    if (profile->modeled) model.emplace(profile->model);
    std::vector<uint8_t> out;

    while (out.size() < size) {
        const uint64_t type = r.le(1, "frame type");
        const uint64_t raw_size = r.le(4, "frame size");
        const uint64_t payload_size = r.le(4, "frame payload size");
        const uint64_t frame_crc = r.le(4, "frame crc");
        if (raw_size == 0 || raw_size > profile->frame_size || raw_size > size - out.size()) {
            throw FormatError("invalid frame size");
        }
        const auto payload = r.bytes(payload_size, "frame payload");
        const size_t start = out.size();
        out.resize(start + static_cast<size_t>(raw_size));
        if (model) model->attach(out);  // resize may have moved the buffer
        const std::span<uint8_t> dst(out.data() + start, static_cast<size_t>(raw_size));

        if (type == static_cast<uint8_t>(FrameType::kStored)) {
            if (payload_size != raw_size) throw FormatError("stored frame: size mismatch");
            std::copy(payload.begin(), payload.end(), dst.begin());
            if (model) {
                for (const uint8_t b : dst) model->observe_byte(b);
            }
            ++stats.frames_stored;
        } else if (model && type == static_cast<uint8_t>(profile->model.anchors.enabled ? FrameType::kHybrid
                                                                                    : FrameType::kModeled)) {
            // The encoder only emits a modeled frame when it is smaller than the raw bytes.
            if (payload_size < 4 || payload_size >= raw_size) throw FormatError("modeled frame: invalid payload size");
            decode_frame(*model, payload, dst, profile->model.anchors, stats);
            ++stats.frames_modeled;
        } else {
            throw FormatError("invalid frame type");
        }
        if (crc32c(dst) != frame_crc) throw FormatError("frame checksum mismatch");
    }

    if (r.remaining() != 0) throw FormatError("trailing bytes after the last frame");
    if (crc32c(out) != content_crc) throw FormatError("content checksum mismatch");
    if (stats_out != nullptr) *stats_out = stats;
    return out;
}

}  // namespace hce
