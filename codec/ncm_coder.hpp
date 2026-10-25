// ncm_coder.hpp - the binary arithmetic coder of the neural CM codecs, the
// three ways a coded decision is sourced (encode, decode, replay) and the two
// views of the history the models read.
#pragma once

#include "hc_framed.hpp"

#include <cstddef>
#include <cstdint>

namespace hce::ncm {

using hc::u32;
using hc::u64;

// --------------------------------------------------- arithmetic coder ---
// 32-bit carryless binary coder. P(1) is Q16 in [1, 65535]; xmid is exact
// (64-bit product), so every platform splits the range identically.
class ArithEncoder {
 public:
  ArithEncoder(std::byte* out, std::size_t cap) noexcept : out_(out), cap_(cap) {}
  void encode(int bit, int p16) noexcept {
    const u32 xmid = x1_ + static_cast<u32>((u64{x2_ - x1_} * static_cast<u32>(p16)) >> 16);
    if (bit) x2_ = xmid;
    else x1_ = xmid + 1;
    while (((x1_ ^ x2_) & 0xFF000000u) == 0) {
      put(x2_ >> 24);
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
    }
  }
  // Four bytes of x1: the decoder then reads exactly the payload, no more.
  void flush() noexcept {
    for (int i = 0; i < 4; ++i, x1_ <<= 8) put(x1_ >> 24);
  }
  bool full() const noexcept { return full_; }
  std::size_t size() const noexcept { return n_; }

 private:
  void put(u32 b) noexcept {
    if (n_ < cap_) out_[n_++] = static_cast<std::byte>(b);
    else full_ = true;
  }
  std::byte* out_;
  std::size_t cap_, n_ = 0;
  u32 x1_ = 0, x2_ = 0xFFFFFFFFu;
  bool full_ = false;
};

class ArithDecoder {
 public:
  explicit ArithDecoder(hc::Bytes in) noexcept : in_(in) {
    for (int i = 0; i < 4; ++i) x_ = (x_ << 8) | next();
  }
  int decode(int p16) noexcept {
    const u32 xmid = x1_ + static_cast<u32>((u64{x2_ - x1_} * static_cast<u32>(p16)) >> 16);
    const int bit = x_ <= xmid;
    if (bit) x2_ = xmid;
    else x1_ = xmid + 1;
    while (((x1_ ^ x2_) & 0xFF000000u) == 0) {
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
      x_ = (x_ << 8) | next();
    }
    return bit;
  }
  bool overrun() const noexcept { return overrun_; }
  // The whole payload was consumed and nothing beyond it was needed.
  bool exact() const noexcept { return !overrun_ && pos_ == in_.size(); }

 private:
  u32 next() noexcept {
    if (pos_ < in_.size()) return static_cast<u32>(in_[pos_++]);
    overrun_ = true;
    return 0;
  }
  hc::Bytes in_;
  std::size_t pos_ = 0;
  u32 x1_ = 0, x2_ = 0xFFFFFFFFu, x_ = 0;
  bool overrun_ = false;
};

// Where each coded decision comes from. `y` is the true bit where it is known.
struct EncodeIO {
  ArithEncoder& c;
  int bit(int p, int y) noexcept {
    c.encode(y, p);
    return y;
  }
};
struct DecodeIO {
  ArithDecoder& c;
  int bit(int p, int) noexcept { return c.decode(p); }
};
struct ReplayIO {
  int bit(int, int y) noexcept { return y; }
};

// History as seen by the models: frame-relative positions. The models only
// read positions within min(window, position) of the current one (R21), which
// is exactly what the decoder's ring window holds.
struct FrameHistory {
  const std::byte* data;
  int at(u64 i) const noexcept { return static_cast<int>(data[i]); }
};
struct WindowHistory {
  const hc::Window* w;
  u64 base;  // absolute window position of the frame start
  int at(u64 i) const noexcept { return static_cast<int>(w->at(base + i)); }
};

}  // namespace hce::ncm
