// hc/framed_writer.hpp - encoder-side framing (R18).
// Part of the header-only container library; hc_framed.hpp includes every part.
#pragma once

#include "hc/format.hpp"

namespace hc {

// ----------------------------------------------------------------- writer ---

// Encoder-side framing (R18). Append-only: it never seeks, so stdout and pipes
// work and the total size is only needed at END. The caller codes the blocks.
template <ByteSink Snk>
class FramedWriter {
 public:
  FramedWriter(Snk& sink, const StreamParams& p, u64 sync_interval) noexcept
      : sink_(sink), p_(p), sync_(sync_interval) {}

  [[nodiscard]] bool begin() { return put(encode_stream_header(p_)); }

  // True when the next block opens a frame: the caller must reset its models
  // to the prior image before coding that block (R20 sync point).
  [[nodiscard]] bool frame_due() const noexcept { return !in_frame_ || logical_ - frame_start_ >= sync_; }

  // `coded` is the CM payload of `content`. Empty, or not smaller than
  // `content`, means STORED; the codec must then roll back its model updates (R12).
  [[nodiscard]] bool block(Bytes content, Bytes coded) {
    if (content.empty() || content.size() > p_.block_capacity()) return false;
    if (frame_due() && !start_frame()) return false;
    const bool stored = coded.empty() || coded.size() >= content.size();
    const Bytes payload = stored ? content : coded;
    BlockHeader b;
    b.tag = static_cast<u8>(stored ? BlockType::stored : BlockType::cm);
    b.usize = static_cast<u32>(content.size());
    b.csize = static_cast<u32>(payload.size());
    b.payload_crc = crc32c(payload);
    b.content_crc = stored ? 0 : crc32c(content);
    list_ = crc32c(le32(stored ? b.payload_crc : b.content_crc), list_);
    logical_ += content.size();
    return put(encode_block_header(b)) && put(payload);
  }

  [[nodiscard]] bool ancillary(u8 kind, Bytes payload) {
    if (payload.size() > kMaxAncillaryBytes || kind > kTagTypeMask) return false;
    BlockHeader b;
    b.tag = static_cast<u8>(kTagAncillary | kind);
    b.csize = static_cast<u32>(payload.size());
    b.payload_crc = crc32c(payload);
    return put(encode_block_header(b)) && put(payload);
  }

  [[nodiscard]] bool end() { return put(encode_end(EndTrailer{logical_, frames_, list_, 0})); }

  u64 logical() const noexcept { return logical_; }

 private:
  bool start_frame() {
    FrameHeader f{p_, frames_, logical_};
    ++frames_;
    in_frame_ = true;
    frame_start_ = logical_;
    return put(encode_frame_header(f));
  }

  template <class T>
  bool put(const T& bytes) {
    return sink_.write(Bytes(bytes));
  }

  Snk& sink_;
  StreamParams p_;
  u64 sync_;
  u64 logical_ = 0, frame_start_ = 0;
  u32 frames_ = 0, list_ = 0;
  bool in_frame_ = false;
};

}  // namespace hc
