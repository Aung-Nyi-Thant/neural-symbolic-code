// Binary range coder: LZMA-style with carry propagation, 16-bit probabilities.
//
//   p1    = P(bit = 1) in [1, 65535]; out-of-range values are clamped identically on both sides
//   bound = (range * p1) >> 16          64-bit product, never (range >> 16) * p1
//   bit 1 -> [low, low + bound)         bit 0 -> [low + bound, low + range)
//   renormalize while range < 2^24      => both sub-intervals are always at least 256 wide
//
// LZMA's first output byte is always 0, so it is not stored, and finish() performs 5 shift steps.
// The encoder therefore writes exactly (renormalizations + 4) bytes and the decoder reads exactly
// that many, which makes a truncated or padded payload detectable.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace hce {

inline constexpr uint32_t kProbBits = 16;
inline constexpr uint32_t kProbMax = (uint32_t{1} << kProbBits) - 1;
inline constexpr uint32_t kRangeTop = uint32_t{1} << 24;

constexpr uint32_t clamp_p1(uint32_t p1) noexcept { return p1 < 1 ? 1 : (p1 > kProbMax ? kProbMax : p1); }

// Writes into a caller-provided buffer and never allocates. If the buffer fills up, overflowed()
// turns true and further output is dropped; the caller then stores the frame raw instead.
class RangeEncoder {
public:
    explicit RangeEncoder(std::span<uint8_t> buf) noexcept : buf_(buf) {}

    void encode(uint32_t bit, uint32_t p1) noexcept {
        const auto bound = static_cast<uint32_t>((uint64_t{range_} * clamp_p1(p1)) >> kProbBits);
        if (bit) {
            range_ = bound;
        } else {
            low_ += bound;
            range_ -= bound;
        }
        while (range_ < kRangeTop) {
            range_ <<= 8;
            shift_low();
        }
    }

    // Flushes the final interval. Returns the payload size (meaningless if overflowed()).
    size_t finish() noexcept {
        for (int i = 0; i < 5; ++i) shift_low();
        return pos_;
    }

    bool overflowed() const noexcept { return overflow_; }

private:
    void shift_low() noexcept {
        if (static_cast<uint32_t>(low_) < 0xFF000000u || (low_ >> 32) != 0) {
            const auto carry = static_cast<uint8_t>(low_ >> 32);
            uint8_t byte = cache_;
            do {
                put(static_cast<uint8_t>(byte + carry));
                byte = 0xFF;
            } while (--pending_ != 0);
            cache_ = static_cast<uint8_t>(low_ >> 24);
        }
        ++pending_;
        low_ = (low_ & 0x00FFFFFFu) << 8;
    }

    void put(uint8_t b) noexcept {
        if (first_) {  // always 0 by construction; anything else would be an encoder bug
            first_ = false;
            if (b != 0) overflow_ = true;
            return;
        }
        if (pos_ < buf_.size()) {
            buf_[pos_++] = b;
        } else {
            overflow_ = true;
        }
    }

    std::span<uint8_t> buf_;
    size_t pos_ = 0;
    uint64_t low_ = 0;
    uint32_t range_ = 0xFFFFFFFFu;
    uint64_t pending_ = 1;  // cache byte plus any 0xFF bytes waiting for a possible carry
    uint8_t cache_ = 0;
    bool first_ = true;
    bool overflow_ = false;
};

class RangeDecoder {
public:
    explicit RangeDecoder(std::span<const uint8_t> in) noexcept : in_(in) {
        for (int i = 0; i < 4; ++i) code_ = (code_ << 8) | next();
    }

    uint32_t decode(uint32_t p1) noexcept {
        const auto bound = static_cast<uint32_t>((uint64_t{range_} * clamp_p1(p1)) >> kProbBits);
        uint32_t bit;
        if (code_ < bound) {
            range_ = bound;
            bit = 1;
        } else {
            code_ -= bound;
            range_ -= bound;
            bit = 0;
        }
        while (range_ < kRangeTop) {
            range_ <<= 8;
            code_ = (code_ << 8) | next();
        }
        return bit;
    }

    // True iff every payload byte was consumed and none was missing.
    bool consumed_exactly() const noexcept { return !overrun_ && pos_ == in_.size(); }

private:
    uint32_t next() noexcept {
        if (pos_ < in_.size()) return in_[pos_++];
        overrun_ = true;
        return 0;
    }

    std::span<const uint8_t> in_;
    size_t pos_ = 0;
    uint32_t code_ = 0;
    uint32_t range_ = 0xFFFFFFFFu;
    bool overrun_ = false;
};

}  // namespace hce
