// hc/interfaces.hpp - the codec, source and sink concepts, decoder options and the split sink.
// Part of the header-only container library; hc_framed.hpp includes every part.
#pragma once

#include "hc/window.hpp"

namespace hc {

// ------------------------------------------------------------- interfaces ---

// The tiered CM decoder (or any block codec) as seen by the container.
template <class C>
concept BlockCodec = requires(C& c, const StreamParams& p, MutBytes mem, Bytes payload, u32 usize, Window& w) {
  { C::accepts(p) } noexcept -> std::same_as<Errc>;  // codec/prior supported? checked before allocating
  { C::memory_required(p) } noexcept -> std::same_as<u64>;
  { c.bind(p, mem) } noexcept -> std::same_as<Errc>;  // once per stream; tables carved from `mem`
  { c.reset_frame() } noexcept -> std::same_as<void>;  // R20: restore the prior image, every table
  // Appends exactly `usize` bytes to `w`, never more; every token yields >= 1 byte;
  // the whole payload is consumed. Anything else is payload_corrupt or bad_reference.
  { c.decode_block(payload, usize, w) } noexcept -> std::same_as<Errc>;
};

template <class S>
concept ByteSource = requires(S& s, MutBytes buf) {
  { s.read(buf) } -> std::same_as<std::size_t>;  // 0 = end of input; short reads are fine
};

template <class S>
concept ByteSink = requires(S& s, Bytes buf) {
  { s.write(buf) } -> std::same_as<bool>;  // false = downstream closed or I/O error
};

enum class Recovery : u8 { strict, skip, skip_and_fill };

struct GapInfo {
  u32 stream_index;   // which concatenated stream
  u64 out_offset;     // uncompressed offset where the loss starts
  u64 length;         // bytes lost; meaningful only when length_known
  bool length_known;  // false when the stream was truncated
  u64 in_offset;      // compressed offset where the damage was detected
  Errc cause;
};
using GapFn = void (*)(void* ctx, const GapInfo&) noexcept;

struct DecoderOptions {
  DecodeLimits limits{};
  Recovery recovery = Recovery::strict;
  std::byte fill{0};
  GapFn on_gap = nullptr;
  void* gap_ctx = nullptr;
  std::stop_token stop{};
};

struct Progress {
  std::size_t consumed = 0;
  std::size_t produced = 0;
  Errc status = Errc::ok;
};

struct DecodeStats {
  u64 frames = 0, blocks_ok = 0, blocks_salvaged = 0, blocks_lost = 0;
  u64 damage_events = 0, gaps = 0, bytes_lost = 0, bytes_filled = 0;
  bool content_verified = false;  // last END matched and nothing was lost in that stream
};

// Parallel decoding, splitter side (decoder/pipeline.cpp). A decoder in split
// mode validates the framing exactly as in sequential mode (headers, ordering,
// resynchronization, and the R19 budget on declared sizes) but forwards every
// accepted item to the sink instead of decoding it, and writes no output.
// Worker threads decode the forwarded frames in job mode.
class SplitSink {
 public:
  // A frame header was accepted; `header` is its 40 bytes, found at `in_offset`.
  virtual Errc frame(const FrameHeader& f, Bytes header, u64 in_offset, u32 stream_index) noexcept = 0;
  // A valid block header (data or ancillary) inside the current frame, then its payload.
  virtual Errc block(const BlockHeader& b, Bytes header) noexcept = 0;
  virtual Errc payload(Bytes bytes) noexcept = 0;
  // A damaged item ends the current frame; a resync scan follows.
  virtual Errc damage(Errc cause, u64 logical, u64 in_offset, u32 stream_index) noexcept = 0;
  // The END trailer after the splitter's own checks; `verified` covers what headers can show.
  virtual Errc end(const EndTrailer& t, bool verified, u64 in_offset, u32 stream_index) noexcept = 0;
  // Recovery modes: the input ended inside a stream whose parameters were known.
  virtual void truncated(u64 logical, u64 in_offset, u32 stream_index) noexcept = 0;

 protected:
  ~SplitSink() = default;
};

}  // namespace hc
