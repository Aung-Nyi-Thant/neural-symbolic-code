// Gated logistic mixer in fixed point.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/kernels/arith.hpp"
#include "core/kernels/tables.hpp"

namespace hce {

// Combines stretched predictions s_i (Q8 logits, |s_i| <= 3072) with one weight vector per gate:
//   p = squash(clamp((sum_i s_i * w_i) >> 16, kLogitMin, kLogitMax))   exact 64-bit sum, Q16 weights
//   w_i += round(s_i * (bit * 2^16 - p) * lr / 2^20), clamped to +-kWeightMax
// The update is online gradient descent on the coding cost -log2 P(bit).
class Mixer {
public:
    static constexpr size_t kMaxInputs = 16;
    static constexpr int64_t kWeightMax = (int64_t{1} << 20) - 1;  // just under 16.0

    Mixer() = default;
    Mixer(size_t num_inputs, size_t num_gates, int32_t initial_weight, uint32_t lr)
        : weights_(num_inputs * num_gates, initial_weight), n_(num_inputs), lr_(lr) {}

    // Sets the weight of input i in every gate.
    void set_weight(size_t i, int32_t w) {
        for (size_t row = 0; row < weights_.size(); row += n_) weights_[row + i] = w;
    }

    void set_input(size_t i, int32_t s) noexcept { inputs_[i] = s; }

    // Returns the mixed 16-bit probability, in [1, 65535].
    uint32_t mix(size_t gate, const Tables& t) noexcept {
        row_ = gate * n_;
        int64_t dot = 0;  // at most 16 * 3072 * 2^20 < 2^36 in magnitude
        for (size_t i = 0; i < n_; ++i) {
            dot = arith::add64(dot, arith::mul64(inputs_[i], weights_[row_ + i]));
        }
        const int64_t logit = arith::clamp64(arith::asr64(dot, 16), kLogitMin, kLogitMax);
        pr_ = t.squash[static_cast<size_t>(logit + kSquashOffset)];
        return pr_;
    }

    void update(uint32_t bit) noexcept {
        const int64_t err = static_cast<int64_t>(bit << 16) - pr_;  // [-65535, 65535]
        const int64_t scaled = arith::mul64(err, lr_);               // below 2^22 for lr <= 64
        for (size_t i = 0; i < n_; ++i) {
            // |s_i * scaled| < 2^34 before the shift.
            const int64_t delta = arith::round_shift64(arith::mul64(inputs_[i], scaled), 20);
            int32_t& w = weights_[row_ + i];
            w = static_cast<int32_t>(arith::clamp64(arith::add64(w, delta), -kWeightMax, kWeightMax));
        }
    }

private:
    std::vector<int32_t> weights_;
    std::array<int32_t, kMaxInputs> inputs_{};
    size_t n_ = 0;
    size_t row_ = 0;
    uint32_t pr_ = 32768;
    uint32_t lr_ = 0;
};

}  // namespace hce
