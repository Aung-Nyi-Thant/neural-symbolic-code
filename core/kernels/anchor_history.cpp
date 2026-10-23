#include "core/kernels/anchor_history.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>

#include "core/kernels/no_float.hpp"

namespace hce {

AnchorHistory::AnchorHistory(const AnchorConfig& config)
    : config_(config), gear_(gear_table(config.gear_seed)), anchors_(std::make_shared<std::vector<uint64_t>>(1, 0)) {
    if (!config.enabled) return;
    if (config.index_bits < 1 || config.index_bits > 24 || config.density_bits > 20) {
        throw std::invalid_argument("AnchorConfig: invalid index or density bits");
    }
    lookup_.resize(size_t{1} << config.index_bits);
    min_gap_ = config.density_bits >= 2 ? uint64_t{1} << (config.density_bits - 2) : 1;
}

uint8_t AnchorHistory::byte_at(uint64_t position) const noexcept {
    assert(position < position_ && position < stream_.size());
    return stream_[static_cast<size_t>(position)];
}

std::optional<uint64_t> AnchorHistory::resolve(uint32_t id) const noexcept {
    if (id == 0 || id >= anchors_->size()) return std::nullopt;
    return (*anchors_)[id];
}

std::optional<uint64_t> AnchorHistory::copy_source(uint32_t id, uint32_t delta,
                                                  uint32_t length) const noexcept {
    const auto anchor = resolve(id);  // anchor positions never exceed position_
    if (!anchor || length == 0 || delta > *anchor) return std::nullopt;
    const uint64_t source = *anchor - delta;
    if (length > position_ - source) return std::nullopt;
    return source;
}

std::optional<uint8_t> AnchorHistory::prediction() const noexcept {
    if (!predicted_position_ || *predicted_position_ >= position_) return std::nullopt;
    return byte_at(*predicted_position_);
}

void AnchorHistory::observe(uint8_t byte) noexcept {
    if (!config_.enabled) return;
    if (const auto predicted = prediction()) {
        divergence_ = static_cast<uint8_t>(*predicted ^ byte);
        if (*predicted == byte) {
            if (match_length_ != std::numeric_limits<uint32_t>::max()) ++match_length_;
            ++*predicted_position_;
        } else {
            predicted_position_.reset();
            match_length_ = 0;
        }
    } else {
        predicted_position_.reset();
        match_length_ = 0;
    }
    ++position_;
    hash_ = (hash_ << 1) + gear_[byte];
    candidate_.reset();
    std::vector<uint64_t>& anchors = *anchors_;
    // After 64 bytes the Gear fingerprint is independent of any earlier prefix. The minimum gap bounds
    // the anchor list on adversarial input.
    const uint64_t mask = (uint64_t{1} << config_.density_bits) - 1;
    if (position_ < 64 || (hash_ & mask) != 0 || position_ - anchors.back() < min_gap_ ||
        anchors.size() == std::numeric_limits<uint32_t>::max()) {
        return;
    }
    Lookup& slot = lookup_[static_cast<size_t>(mix64(hash_) & (lookup_.size() - 1))];
    if (slot.id != 0 && slot.fingerprint == hash_) {
        const uint64_t earlier = anchors[slot.id];
        candidate_ = Match{slot.id, earlier};
        if (predicted_position_ != earlier) match_length_ = 0;
        predicted_position_ = earlier;
    }
    // Direct-mapped buckets keep the latest anchor per bucket; exact fingerprints prevent false matches.
    // shortcut: the list keeps 8 bytes per anchor for the whole stream (at most N/2 bytes at 6 density bits);
    // spill or cap it if streams outgrow memory.
    anchors.push_back(position_);
    slot = Lookup{hash_, static_cast<uint32_t>(anchors.size() - 1)};
}

}  // namespace hce
