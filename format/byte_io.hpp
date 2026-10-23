// Little-endian serialization and bounds-checked parsing of untrusted input.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace hce {

// Any malformed, truncated, corrupted or unsupported stream.
class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Writes the low `bytes` bytes of v, least significant first, independent of host byte order.
inline void put_le(std::vector<uint8_t>& out, uint64_t v, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

class Reader {
public:
    explicit Reader(std::span<const uint8_t> in) : in_(in) {}

    uint64_t le(unsigned bytes, const char* what) {
        need(bytes, what);
        uint64_t v = 0;
        for (unsigned i = 0; i < bytes; ++i) v |= uint64_t{in_[pos_ + i]} << (8 * i);
        pos_ += bytes;
        return v;
    }

    std::span<const uint8_t> bytes(uint64_t n, const char* what) {
        need(n, what);
        const auto s = in_.subspan(pos_, static_cast<size_t>(n));
        pos_ += static_cast<size_t>(n);
        return s;
    }

    size_t remaining() const noexcept { return in_.size() - pos_; }

private:
    void need(uint64_t n, const char* what) const {
        if (n > remaining()) throw FormatError(std::string("truncated stream: ") + what);
    }

    std::span<const uint8_t> in_;
    size_t pos_ = 0;
};

}  // namespace hce
