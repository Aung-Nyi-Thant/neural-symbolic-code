#include "core/kernels/cm_model.hpp"

#include <cassert>
#include <algorithm>
#include <bit>
#include <stdexcept>
#include <vector>

#include "core/kernels/crc32c.hpp"
#include "core/kernels/hash.hpp"
#include "core/kernels/no_float.hpp"

namespace hce {
namespace {

constexpr int32_t kBiasInput = 256;         // constant logit of 1.0
constexpr int32_t kInitialWeight = 1 << 14;  // 0.25 per context input
constexpr size_t kGates = 256;              // one mixer weight set per partial byte c0
constexpr uint64_t kNibbleSalt = 0x9E3779B97F4A7C15ull;

const ModelConfig& validated(const ModelConfig& c) {
    if (c.num_orders == 0 || c.num_orders > kMaxOrders) {
        throw std::invalid_argument("ModelConfig: num_orders must be 1..8");
    }
    for (size_t i = 0; i < c.num_orders; ++i) {
        if (c.orders[i] > 7) throw std::invalid_argument("ModelConfig: orders must be 0..7");
    }
    if (c.table_bits < 3 || c.table_bits > 30) {
        throw std::invalid_argument("ModelConfig: table_bits must be 3..30");
    }
    if (c.mixer_lr == 0 || c.mixer_lr > 64) throw std::invalid_argument("ModelConfig: mixer_lr must be 1..64");
    if (c.statemap_limit > 1023) throw std::invalid_argument("ModelConfig: statemap_limit must be <= 1023");
    return c;
}

}  // namespace

CmModel::CmModel(const ModelConfig& config)
    : t_(&tables()),
      cfg_(validated(config)),
      table_(cfg_.table_bits),
      mixer_(size_t{cfg_.num_orders} + 1 + (cfg_.anchors.enabled ? 2u : 0u),
             kGates, kInitialWeight, cfg_.mixer_lr),
      anchors_(cfg_.anchors) {
    for (size_t i = 0; i < cfg_.num_orders; ++i) maps_[i] = StateMap(*t_, cfg_.statemap_limit);
    mixer_.set_weight(cfg_.num_orders, 0);  // the bias input starts neutral
    rehash();
}

void CmModel::rehash() noexcept {
    for (size_t i = 0; i < cfg_.num_orders; ++i) {
        const uint64_t k = cfg_.orders[i];
        const uint64_t mask = k == 0 ? 0 : ~uint64_t{0} >> (64 - 8 * k);
        ctx_[i] = mix64((history_ & mask) ^ ((k + 1) << 56));
    }
    slots_valid_ = false;
}

void CmModel::lookup() noexcept {
    // Nibble 0 is keyed by c0 = 1; nibble 1 by c0 = 1hhhh, which includes the high nibble's bits.
    for (size_t i = 0; i < cfg_.num_orders; ++i) {
        slot_[i] = table_.find(mix64(ctx_[i] + c0_ * kNibbleSalt), *t_);
    }
    slots_valid_ = true;
}

uint32_t CmModel::predict() noexcept {
    if (!slots_valid_) lookup();
    for (size_t i = 0; i < cfg_.num_orders; ++i) {
        state_[i] = table_.slot(slot_[i])[nib_];
        mixer_.set_input(i, t_->stretch[maps_[i].p16(state_[i]) >> 4]);
    }
    mixer_.set_input(cfg_.num_orders, kBiasInput);
    if (cfg_.anchors.enabled) {
        int32_t match = 0;
        int32_t divergence = 0;
        if (const auto predicted = anchors_.prediction();
            predicted && (c0_ ^ (1u << bpos_)) == (uint32_t{*predicted} >> (8 - bpos_))) {
            const uint32_t bit = (uint32_t{*predicted} >> (7 - bpos_)) & 1u;
            const int32_t strength = static_cast<int32_t>(
                std::min(8u, static_cast<uint32_t>(std::bit_width(anchors_.match_length() | 1u)))) * 256;
            match = bit != 0 ? strength : -strength;
            const uint32_t diverged = (uint32_t{anchors_.divergence()} >> (7 - bpos_)) & 1u;
            divergence = (bit ^ diverged) != 0 ? 256 : -256;
        }
        mixer_.set_input(size_t{cfg_.num_orders} + 1, match);
        mixer_.set_input(size_t{cfg_.num_orders} + 2, divergence);
    }
    return mixer_.mix(c0_, *t_);
}

void CmModel::update(uint32_t bit) noexcept {
    assert(bit <= 1);
    mixer_.update(bit);
    for (size_t i = 0; i < cfg_.num_orders; ++i) {
        maps_[i].update(state_[i], bit, *t_);
        table_.slot(slot_[i])[nib_] = t_->next[state_[i]][bit];
    }
    c0_ = (c0_ << 1) | bit;
    nib_ = (nib_ << 1) | bit;
    if (++bpos_ == 8) {
        anchors_.observe(static_cast<uint8_t>(c0_));
        history_ = (history_ << 8) | (c0_ & 0xFFu);
        c0_ = 1;
        nib_ = 1;
        bpos_ = 0;
        rehash();
    } else if (bpos_ == 4) {
        nib_ = 1;
        slots_valid_ = false;
    }
}

void CmModel::observe_byte(uint8_t byte) noexcept {
    assert(bpos_ == 0);
    anchors_.observe(byte);
    history_ = (history_ << 8) | byte;
    rehash();
}

uint32_t model_fingerprint(const ModelConfig& config) {
    const Tables& t = tables();
    std::vector<uint8_t> bytes;
    const auto put = [&bytes](uint64_t v, unsigned n) {
        for (unsigned i = 0; i < n; ++i) bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
    };
    put(kModelArithmeticVersion, 4);
    put(config.num_orders, 1);
    for (size_t i = 0; i < config.num_orders; ++i) put(config.orders[i], 1);
    put(config.table_bits, 1);
    put(config.mixer_lr, 1);
    put(config.statemap_limit, 2);
    put(static_cast<uint32_t>(kBiasInput), 4);
    put(static_cast<uint32_t>(kInitialWeight), 4);
    put(kGates, 4);
    put(kNibbleSalt, 8);
    put(mix64(1), 8);
    put(mix64(0x0123456789ABCDEFull), 8);
    if (config.anchors.enabled) {
        put(3, 4);  // hybrid arithmetic and token grammar version; legacy fingerprints stay frozen
        put(config.anchors.index_bits, 1);
        put(config.anchors.density_bits, 1);
        put(kCopyFlagP1, 4);
        put(kCopyDeltaBits, 1);
        put(config.anchors.gear_seed, 8);
        for (const uint64_t word : gear_table(config.anchors.gear_seed)) put(word, 8);
    }
    for (const uint16_t v : t.squash) put(v, 2);
    for (const int16_t v : t.stretch) put(static_cast<uint16_t>(v), 2);
    for (const uint32_t v : t.recip) put(v, 4);
    for (size_t s = 0; s < 256; ++s) {
        put(t.next[s][0], 1);
        put(t.next[s][1], 1);
        put(t.n0[s], 1);
        put(t.n1[s], 1);
        put(t.p22_init[s], 4);
    }
    return crc32c(bytes);
}

}  // namespace hce
