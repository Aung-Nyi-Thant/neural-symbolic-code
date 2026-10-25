// hc/window.hpp - the R19 expansion budget, the R21 history window and side-info slots.
// Part of the header-only container library; hc_framed.hpp includes every part.
#pragma once

#include "hc/format.hpp"

namespace hc {

// ----------------------------------------------------------------- budget ---

// R19 expansion control as a token bucket: each compressed byte earns
// `max_ratio` bytes of output credit (capped at the burst depth), and each
// output byte, decoded or recovery fill, spends one before it is produced.
// Long-run output <= ratio * input + burst, while single highly compressible
// blocks (64 KiB of zeros codes to ~25 bytes) still pass.
class Budget {
 public:
  explicit constexpr Budget(const DecodeLimits& l) noexcept
      : ratio_(l.max_ratio),
        burst_(l.ratio_burst_bytes),
        credit_(burst_),
        max_out_(l.max_output_bytes) {}

  constexpr void earn(u64 input_bytes) noexcept {
    if (ratio_) credit_ = std::min(burst_, sat_add(credit_, sat_mul(input_bytes, ratio_)));
  }

  [[nodiscard]] constexpr Errc spend(u64 output_bytes) noexcept {
    if (output_bytes > max_out_ - out_) return Errc::limit_output;
    if (ratio_) {
      if (output_bytes > credit_) return Errc::limit_ratio;
      credit_ -= output_bytes;
    }
    out_ += output_bytes;
    return Errc::ok;
  }

  // Returns credit for output that was reserved but never produced.
  constexpr void refund(u64 output_bytes) noexcept {
    out_ -= std::min(out_, output_bytes);
    if (ratio_) credit_ = std::min(burst_, sat_add(credit_, output_bytes));
  }

  constexpr u64 committed() const noexcept { return out_; }

 private:
  u64 ratio_, burst_, credit_, max_out_, out_ = 0;
};

// ----------------------------------------------------------------- window ---

// R21: fixed history ring of 2^window_log bytes, addressed by absolute 64-bit
// positions. Memory never grows with input size. A sync point is O(1): it
// only moves the reachability floor, nothing is cleared.
class Window {
 public:
  void bind(MutBytes mem) noexcept {  // mem.size() must be a power of two
    buf_ = mem.data();
    cap_ = mem.size();
    mask_ = cap_ - 1;
    pos_ = start_ = 0;
  }

  u64 pos() const noexcept { return pos_; }
  u64 capacity() const noexcept { return cap_; }
  // Bytes since the current sync point, STORED blocks included: the frame-relative
  // position a codec's models must use so encoder and decoder agree.
  u64 frame_offset() const noexcept { return pos_ - start_; }
  void reset_frame() noexcept { start_ = pos_; }
  u64 reachable() const noexcept { return std::min<u64>(cap_, pos_ - start_); }
  std::byte at(u64 abs) const noexcept { return buf_[abs & mask_]; }  // caller validates abs
  void put(std::byte b) noexcept { buf_[pos_++ & mask_] = b; }

  void append(Bytes src) noexcept {  // precondition: src.size() <= capacity()
    while (!src.empty()) {
      const std::size_t at = static_cast<std::size_t>(pos_ & mask_);
      const std::size_t n = std::min(src.size(), cap_ - at);
      std::memcpy(buf_ + at, src.data(), n);
      src = src.subspan(n);
      pos_ += n;
    }
  }

  // Validated back-reference with LZ77 semantics (overlap allowed: offset 1 is a run).
  [[nodiscard]] Errc copy(u64 offset, u64 len) noexcept {
    if (offset == 0 || offset > reachable()) return Errc::bad_reference;
    u64 src = pos_ - offset;
    if (len <= offset && len <= cap_ - offset) {
      // No logical overlap (len <= offset) and no physical overlap in the ring
      // (offset + len <= cap): at most three memcpy calls, split at wrap points.
      for (u64 dst = pos_, end = pos_ + len; dst != end;) {
        const std::size_t s = static_cast<std::size_t>(src & mask_);
        const std::size_t d = static_cast<std::size_t>(dst & mask_);
        const std::size_t n = static_cast<std::size_t>(std::min<u64>({end - dst, cap_ - s, cap_ - d}));
        std::memcpy(buf_ + d, buf_ + s, n);
        src += n;
        dst += n;
      }
    } else {
      // Read-before-write per byte is exact for every offset <= cap.
      for (u64 i = 0; i < len; ++i) buf_[(pos_ + i) & mask_] = buf_[(src + i) & mask_];
    }
    pos_ += len;
    return Errc::ok;
  }

  // First contiguous piece of [from, from + len). Precondition: pos() - from <= capacity().
  Bytes contiguous(u64 from, u64 len) const noexcept {
    const std::size_t at = static_cast<std::size_t>(from & mask_);
    return {buf_ + at, static_cast<std::size_t>(std::min<u64>(len, cap_ - at))};
  }

 private:
  std::byte* buf_ = nullptr;
  std::size_t cap_ = 0;
  std::size_t mask_ = 0;
  u64 pos_ = 0;
  u64 start_ = 0;
};

// R21: model tables store 32-bit truncated positions. A candidate is usable iff
// it lies inside the reachable window; both sides evaluate this identically.
// Aliasing after 2^32 bytes is deterministic within a frame, which is why
// tables must be reset at every sync point (see BlockCodec::reset_frame).
constexpr bool candidate_valid(u64 pos, u32 cand, u64 reachable) noexcept {
  const u32 d = static_cast<u32>(pos) - cand;  // wraps mod 2^32 by definition
  return d != 0 && d <= reachable;
}

// R21: bounded side-info dictionary (log templates, keyed-delta state). The
// encoder picks the slot to overwrite (explicit eviction); the decoder only
// validates, so its memory is Slots * SlotBytes regardless of input.
template <std::size_t Slots, std::size_t SlotBytes>
class SlotTable {
  static_assert(SlotBytes <= 0xFFFF);

 public:
  [[nodiscard]] Errc define(u64 slot, Bytes body) noexcept {
    if (slot >= Slots || body.empty() || body.size() > SlotBytes) return Errc::payload_corrupt;
    std::memcpy(body_[slot].data(), body.data(), body.size());
    len_[slot] = static_cast<u16>(body.size());
    return Errc::ok;
  }
  [[nodiscard]] Errc get(u64 slot, Bytes& out) const noexcept {
    if (slot >= Slots || len_[slot] == 0) return Errc::payload_corrupt;
    out = Bytes(body_[slot]).first(len_[slot]);
    return Errc::ok;
  }
  void reset() noexcept { len_.fill(0); }  // sync point

 private:
  std::array<u16, Slots> len_{};
  std::array<std::array<std::byte, SlotBytes>, Slots> body_{};
};

}  // namespace hc
