// Order-N context-mixing bit predictor: the baseline model of the modeled profiles.
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "core/kernels/anchor_history.hpp"
#include "core/kernels/hash_table.hpp"
#include "core/kernels/mixer.hpp"
#include "core/kernels/state_map.hpp"
#include "core/kernels/tables.hpp"

namespace hce {

inline constexpr size_t kMaxOrders = 8;

struct ModelConfig {
    std::array<uint8_t, kMaxOrders> orders{};  // context lengths in bytes, each 0..7
    uint8_t num_orders = 0;                    // 1..8
    uint8_t table_bits = 22;                   // 2^table_bits slots of 16 bytes, 3..30
    uint8_t mixer_lr = 7;                      // 1..64
    uint16_t statemap_limit = 255;             // 0..1023
    AnchorConfig anchors{};
};

// Predicts one bit at a time, most significant bit of each byte first. The sequence of predict(),
// update() and observe_byte() calls fully determines every probability on every platform: the
// encoder and decoder stay in lockstep by making the same calls.
//
// For each order k, the context is a hash of the previous k bytes. A slot per nibble holds the bit
// histories of that nibble's tree, and a per-order StateMap turns the current node's history into a
// probability. The mixer combines the stretched probabilities plus a bias input; its weight set is
// selected by the bits of the current byte seen so far.
class CmModel {
public:
    explicit CmModel(const ModelConfig& config);  // throws std::invalid_argument on a bad config

    uint32_t predict() noexcept;               // P(next bit = 1), 16-bit, in [1, 65535]
    void update(uint32_t bit) noexcept;        // bit is 0 or 1; call once after each predict()
    void observe_byte(uint8_t byte) noexcept;  // extend the history without learning (byte boundary only)
    bool at_byte_boundary() const noexcept { return bpos_ == 0; }
    const AnchorHistory& anchors() const noexcept { return anchors_; }
    void attach(std::span<const uint8_t> stream) noexcept { anchors_.attach(stream); }  // see AnchorHistory
    void truncate_anchors(uint32_t count) { anchors_.truncate(count); }

private:
    void rehash() noexcept;
    void lookup() noexcept;

    const Tables* t_;
    ModelConfig cfg_;
    HashTable table_;
    std::array<StateMap, kMaxOrders> maps_{};
    Mixer mixer_;
    AnchorHistory anchors_;
    uint64_t history_ = 0;  // previous 8 bytes, most recent in the low byte
    uint32_t c0_ = 1;       // bits of the current byte so far, behind a leading 1
    uint32_t nib_ = 1;      // bits of the current nibble so far, behind a leading 1 (= tree node)
    uint32_t bpos_ = 0;
    bool slots_valid_ = false;
    std::array<uint64_t, kMaxOrders> ctx_{};
    std::array<uint32_t, kMaxOrders> slot_{};
    std::array<uint8_t, kMaxOrders> state_{};
};

// Version of the arithmetic defined by code rather than tables (hashing, update rules, constants).
// Bump it whenever that code changes behaviour; the fingerprint then changes with it.
inline constexpr uint32_t kModelArithmeticVersion = 1;

// CRC-32C over the configuration, kModelArithmeticVersion, hash probes and every generated table.
// Stored in the container header as the model id, so a decoder refuses streams from another model.
uint32_t model_fingerprint(const ModelConfig& config);

}  // namespace hce
