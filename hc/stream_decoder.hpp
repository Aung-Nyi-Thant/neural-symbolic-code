// hc/stream_decoder.hpp - the push decoder (R18-R21) and the drivers around it.
// Part of the header-only container library; hc_framed.hpp includes every part.
#pragma once

#include "hc/interfaces.hpp"
#include "hc/arena.hpp"

namespace hc {

// ---------------------------------------------------------------- decoder ---

template <BlockCodec Codec>
class StreamDecoder {
 public:
  StreamDecoder(Codec& codec, const DecoderOptions& opt) noexcept
      : codec_(codec), opt_(opt), budget_(opt.limits) {}

  // Push API (R18). Never seeks and never looks ahead past what it consumes.
  // Call again with more input after need_input, or with fresh output space
  // after output_full. `input_finished` means the source hit EOF. After
  // end_of_stream, more input is decoded as a concatenated stream.
  [[nodiscard]] Progress decode(Bytes in, MutBytes out, bool input_finished) noexcept {
    Progress p;
    for (;;) {
      if (st_ == St::failed) {
        p.status = err_;
        return p;
      }
      if (opt_.stop.stop_requested()) {
        p.status = fail(Errc::cancelled);
        return p;
      }
      Errc r = step(in, out, p);
      if (r == Errc::ok) continue;
      if (r == Errc::need_input && input_finished) {
        const bool job_done = job_ && st_ == St::item && hdr_len_ == 0;
        r = (st_ == St::done || job_done) ? Errc::end_of_stream : on_truncated();
      }
      p.status = is_error(r) ? fail(r) : r;
      return p;
    }
  }

  const DecodeStats& stats() const noexcept { return stats_; }
  const StreamParams& params() const noexcept { return params_; }
  bool params_valid() const noexcept { return params_valid_; }
  u64 logical() const noexcept { return logical_; }

  // Split mode (see SplitSink). Set before the first decode().
  void set_split_sink(SplitSink* sink) noexcept { split_ = sink; }

  // ---- Job mode: parallel decoding, worker side ----

  struct JobStart {
    u64 out_offset = 0;  // from the frame header
    u64 frame_index = 0;
    u64 in_offset = 0;  // input offset of the frame header
    u32 stream_index = 0;
  };
  // A recovery fill, reported with its position in the job's output instead of
  // being written: a fill can be larger than any fixed output buffer.
  using FillFn = void (*)(void* ctx, u64 at, u64 len) noexcept;
  struct PendingGap {
    u64 out_offset;
    u64 in_offset;
    Errc cause;
  };

  // Prepares to decode one frame forwarded by a splitter. The bytes passed to
  // decode() afterwards are exactly that frame's items, starting with its
  // header, possibly over several calls that each end on an item boundary;
  // the final call (input_finished) reports end_of_stream. The splitter has
  // already enforced the R19 budget.
  [[nodiscard]] Errc begin_job(const StreamParams& p, const JobStart& j, FillFn fill, void* fill_ctx) noexcept {
    if (!params_valid_ || !p.same_coding(params_)) {
      params_valid_ = false;
      if (const Errc e = bind_stream(p); e != Errc::ok) return e;
    }
    st_ = St::item;
    err_ = Errc::ok;
    hdr_len_ = 0;
    in_frame_ = frame_broken_ = gap_open_ = false;
    emit_left_ = fill_left_ = 0;
    logical_ = j.out_offset;
    frames_ = j.frame_index;
    in_offset_ = j.in_offset;
    stream_index_ = j.stream_index;
    list_crc_ = 0;
    stream_damage_ = 0;
    stats_ = {};
    job_ = true;
    job_fill_ = fill;
    job_ctx_ = fill_ctx;
    job_produced_ = 0;
    return Errc::ok;
  }

  // After a job: the open gap that the next frame header will close, if any.
  std::optional<PendingGap> pending_gap() const noexcept {
    if (!gap_open_) return std::nullopt;
    return PendingGap{gap_start_, gap_in_, gap_cause_};
  }

 private:
  enum class St : u8 { stream_header, item, payload, skip, emit, fill, scan, done, failed };

  Errc step(Bytes& in, MutBytes& out, Progress& p) noexcept {
    switch (st_) {
      case St::stream_header: return on_stream_header(in, p);
      case St::item: return on_item(in, p);
      case St::payload: return on_payload(in, p);
      case St::skip: return on_skip(in, p);
      case St::emit: return on_emit(out, p);
      case St::fill: return on_fill(out, p);
      case St::scan: return on_scan(in, p);
      case St::done:
        if (in.empty()) return Errc::need_input;
        begin_stream();  // concatenated stream follows
        return Errc::ok;
      case St::failed: return err_;
    }
    return Errc::header_corrupt;
  }

  bool recovering() const noexcept { return opt_.recovery != Recovery::strict; }
  Bytes staged() const noexcept { return Bytes(hdr_).first(hdr_len_); }

  void consume(Bytes& in, std::size_t n, Progress& p) noexcept {
    in = in.subspan(n);
    p.consumed += n;
    in_offset_ += n;
  }

  // Restartable fixed-size header staging: works for any input chunking.
  bool gather(Bytes& in, std::size_t n, Progress& p) noexcept {
    const std::size_t take = std::min(n - hdr_len_, in.size());
    if (take) std::memcpy(hdr_.data() + hdr_len_, in.data(), take);
    hdr_len_ += take;
    consume(in, take, p);
    return hdr_len_ == n;
  }

  Errc then(St next) noexcept {
    after_fill_ = next;
    st_ = fill_left_ ? St::fill : next;
    return st_ == St::done ? Errc::end_of_stream : Errc::ok;
  }

  Errc fail(Errc e) noexcept {
    st_ = St::failed;
    err_ = e;
    return e;
  }

  void begin_stream() noexcept {
    ++stream_index_;
    st_ = St::stream_header;
    hdr_len_ = 0;
    params_valid_ = in_frame_ = frame_broken_ = gap_open_ = false;
    logical_ = 0;
    frames_ = 0;
    list_crc_ = 0;
    stream_damage_ = 0;
  }

  // The only allocation: once per stream, sized from validated parameters (R19).
  Errc bind_stream(const StreamParams& sp) noexcept {
    if (const Errc a = Codec::accepts(sp); a != Errc::ok) return a;
    const u64 window = sp.window_size();
    const u64 block = sp.block_capacity();
    const u64 model = Codec::memory_required(sp);
    const u64 total = sat_add(sat_add(window, block), model);
    if (total > opt_.limits.max_memory_bytes || !std::in_range<std::size_t>(total)) return Errc::limit_memory;
    if (split_) {  // split mode decodes nothing: same checks, no window, model or payload buffer
      params_ = sp;
      params_valid_ = true;
      return Errc::ok;
    }
    if (total > arena_cap_) {
      arena_.reset();  // the old arena goes first: peak memory is one arena
      arena_ = allocate_arena(static_cast<std::size_t>(total));
      arena_cap_ = arena_ ? total : 0;
      if (!arena_) return Errc::out_of_memory;
    }
    const MutBytes mem(arena_.get(), static_cast<std::size_t>(total));
    window_.bind(mem.first(static_cast<std::size_t>(window)));
    payload_ = mem.subspan(static_cast<std::size_t>(window), block);
    if (const Errc e = codec_.bind(sp, mem.subspan(static_cast<std::size_t>(window + block))); e != Errc::ok)
      return e;
    params_ = sp;
    params_valid_ = true;
    return Errc::ok;
  }

  Errc on_stream_header(Bytes& in, Progress& p) noexcept {
    if (!gather(in, kStreamHeaderSize, p)) return Errc::need_input;
    StreamParams sp;
    Errc e = parse_stream_header(staged(), sp);
    if (e == Errc::ok) e = validate_params(sp, opt_.limits);
    if (e == Errc::ok) {
      if (const Errc b = bind_stream(sp); b != Errc::ok) return b;
      budget_.earn(kStreamHeaderSize);
      hdr_len_ = 0;
      st_ = St::item;
      return Errc::ok;
    }
    if (is_limit(e) || e == Errc::unsupported_version || e == Errc::unsupported_feature) return e;
    if (!recovering()) return (e == Errc::bad_magic && stream_index_ > 0) ? Errc::trailing_garbage : e;
    // Damaged or missing stream header (late join): frames are self-describing,
    // so resync on the first one. Rescan from byte 0: a frame may start right here.
    if (split_) {
      if (const Errc s = split_->damage(e, logical_, in_offset_, stream_index_); s != Errc::ok) return s;
    }
    note_damage();
    open_gap(e);
    return enter_scan(0);
  }

  Errc on_item(Bytes& in, Progress& p) noexcept {
    if (hdr_len_ == 0 && in.empty()) return Errc::need_input;
    const std::byte first = hdr_len_ ? hdr_[0] : in[0];
    const std::size_t size = first == kSyncMarker[0]        ? kFrameHeaderSize
                             : first == std::byte{kTagEnd} ? kEndSize
                                                           : kBlockHeaderSize;
    if (!gather(in, size, p)) return Errc::need_input;
    if (size == kFrameHeaderSize) {
      FrameHeader f;
      if (const Errc e = parse_frame_header(staged(), f); e != Errc::ok) return damaged(e);
      return accept_frame(f);
    }
    return size == kEndSize ? on_end() : on_block_header();
  }

  // A CRC-valid frame header: accept it as the next sync point, or reject it.
  Errc accept_frame(const FrameHeader& f) noexcept {
    const bool scanning = st_ == St::scan;
    if (!params_valid_) {  // the stream header was damaged: adopt the frame's self-description
      if (const Errc v = validate_params(f.params, opt_.limits); v != Errc::ok)
        return is_limit(v) ? v : reject(v, scanning);
      if (const Errc b = bind_stream(f.params); b != Errc::ok) return b;
    }
    const bool consistent = f.params.same_coding(params_) &&
                            (!params_.content_size || f.out_offset <= *params_.content_size);
    const bool in_order = recovering() ? (f.out_offset >= logical_ && f.index >= frames_)
                                       : (f.out_offset == logical_ && f.index == frames_);
    if (!consistent || !in_order) return reject(Errc::header_corrupt, scanning);
    budget_.earn(kFrameHeaderSize);
    if (f.out_offset > logical_) {  // whole frames are missing
      note_damage();
      open_gap(Errc::sequence_gap);
    }
    if (const Errc g = close_gap(f.out_offset); g != Errc::ok) return g;
    logical_ = f.out_offset;
    frames_ = u64{f.index} + 1;
    ++stats_.frames;
    in_frame_ = true;
    frame_broken_ = false;
    if (split_) {  // hdr_ still holds the 40 accepted bytes
      const Errc s =
          split_->frame(f, Bytes(hdr_).first(kFrameHeaderSize), in_offset_ - kFrameHeaderSize, stream_index_);
      if (s != Errc::ok) return s;
    } else {
      window_.reset_frame();
      codec_.reset_frame();  // R20: nothing decoded from here on depends on earlier bytes
    }
    hdr_len_ = 0;
    return then(St::item);
  }

  Errc on_block_header() noexcept {
    BlockHeader b;
    Errc e = parse_block_header(staged(), b);
    if (e == Errc::ok && !in_frame_) e = Errc::header_corrupt;  // blocks live inside frames
    if (e == Errc::ok) e = check_block_shape(b, params_);
    if (e == Errc::ok && params_.content_size && b.usize > *params_.content_size - logical_)
      e = Errc::header_corrupt;
    if (e != Errc::ok) return damaged(e);
    hdr_len_ = 0;
    cur_ = b;
    budget_.earn(kBlockHeaderSize + u64{b.csize});
    skip_left_ = b.csize;
    if (b.ancillary()) {  // skippable metadata: seek index, provenance, ...
      if (split_) {
        if (const Errc s = split_->block(b, Bytes(hdr_).first(kBlockHeaderSize)); s != Errc::ok) return s;
      }
      skip_verify_ = split_ == nullptr;  // in split mode the worker verifies it
      skip_crc_ = 0;
      st_ = St::skip;
      return Errc::ok;
    }
    if (frame_broken_ && b.type() != BlockType::stored) {
      // R20: the models that would decode this block were trained on lost data.
      open_gap(broken_cause_);
      logical_ += b.usize;
      ++stats_.blocks_lost;
      skip_verify_ = false;
      st_ = St::skip;
      return Errc::ok;
    }
    if (const Errc s = budget_.spend(b.usize); s != Errc::ok) return s;  // R19: before any work
    if (split_) {
      if (const Errc s = split_->block(b, Bytes(hdr_).first(kBlockHeaderSize)); s != Errc::ok) return s;
    }
    payload_len_ = 0;
    st_ = St::payload;
    return Errc::ok;
  }

  Errc on_skip(Bytes& in, Progress& p) noexcept {
    const std::size_t n = static_cast<std::size_t>(std::min<u64>(skip_left_, in.size()));
    if (skip_verify_) skip_crc_ = crc32c(in.first(n), skip_crc_);
    if (split_ && n) {
      if (const Errc s = split_->payload(in.first(n)); s != Errc::ok) return s;
    }
    consume(in, n, p);
    skip_left_ -= n;
    if (skip_left_) return Errc::need_input;
    if (skip_verify_ && skip_crc_ != cur_.payload_crc) {
      if (!recovering()) return Errc::payload_corrupt;
      note_damage();  // metadata only: no content lost
    }
    return then(St::item);
  }

  Errc on_payload(Bytes& in, Progress& p) noexcept {
    const std::size_t n = std::min<std::size_t>(cur_.csize - payload_len_, in.size());
    if (split_) {  // forward; a worker decodes and verifies this block
      if (n) {
        if (const Errc s = split_->payload(in.first(n)); s != Errc::ok) return s;
      }
      payload_len_ += n;
      consume(in, n, p);
      if (payload_len_ < cur_.csize) return Errc::need_input;
      const u32 check = cur_.type() == BlockType::stored ? cur_.payload_crc : cur_.content_crc;
      list_crc_ = crc32c(le32(check), list_crc_);
      logical_ += cur_.usize;
      return then(St::item);
    }
    if (n) std::memcpy(payload_.data() + payload_len_, in.data(), n);
    payload_len_ += n;
    consume(in, n, p);
    if (payload_len_ < cur_.csize) return Errc::need_input;

    const Bytes payload = Bytes(payload_).first(cur_.csize);
    if (crc32c(payload) != cur_.payload_crc) return lose_block(Errc::payload_corrupt);
    const u64 start = window_.pos();
    u32 check = cur_.payload_crc;  // STORED: content == payload
    if (cur_.type() == BlockType::stored) {
      window_.append(payload);
    } else {
      if (const Errc e = codec_.decode_block(payload, cur_.usize, window_); e != Errc::ok) return lose_block(e);
      if (window_.pos() - start != cur_.usize) return lose_block(Errc::payload_corrupt);
      check = window_crc(start, cur_.usize);
      if (check != cur_.content_crc) return lose_block(Errc::content_mismatch);
    }
    // Verified: only now may these bytes reach the sink (R20 verify-then-emit).
    list_crc_ = crc32c(le32(check), list_crc_);
    if (frame_broken_) {  // salvaged STORED block inside a broken frame
      ++stats_.blocks_salvaged;
      if (const Errc g = close_gap(logical_); g != Errc::ok) return g;
    } else {
      ++stats_.blocks_ok;
    }
    logical_ += cur_.usize;
    emit_pos_ = start;
    emit_left_ = cur_.usize;
    return then(St::emit);
  }

  Errc on_end() noexcept {
    EndTrailer t;
    if (const Errc e = parse_end(staged(), t); e != Errc::ok) return damaged(e);
    hdr_len_ = 0;
    budget_.earn(kEndSize);
    if (!recovering()) {
      const bool ok = t.total_out == logical_ && t.frame_count == frames_ && t.list_check == list_crc_ &&
                      (!params_.content_size || *params_.content_size == logical_);
      if (!ok) return Errc::content_mismatch;
      stats_.content_verified = true;
    } else {
      if (t.total_out > logical_) {  // data missing at the tail
        note_damage();
        open_gap(Errc::sequence_gap);
      }
      if (const Errc g = close_gap(std::max(t.total_out, logical_)); g != Errc::ok) return g;
      stats_.content_verified =
          stream_damage_ == 0 && t.total_out == logical_ && t.list_check == list_crc_ && t.frame_count == frames_;
      logical_ = std::max(t.total_out, logical_);
    }
    if (split_) {
      if (const Errc s = split_->end(t, stats_.content_verified, in_offset_, stream_index_); s != Errc::ok) return s;
    }
    return then(St::done);
  }

  Errc on_emit(MutBytes& out, Progress& p) noexcept {
    while (emit_left_) {
      if (out.empty()) return Errc::output_full;
      const Bytes piece = window_.contiguous(emit_pos_, std::min<u64>(emit_left_, out.size()));
      std::memcpy(out.data(), piece.data(), piece.size());
      out = out.subspan(piece.size());
      p.produced += piece.size();
      emit_pos_ += piece.size();
      emit_left_ -= piece.size();
      job_produced_ += piece.size();
    }
    st_ = St::item;
    return Errc::ok;
  }

  Errc on_fill(MutBytes& out, Progress& p) noexcept {
    while (fill_left_) {
      if (out.empty()) return Errc::output_full;
      const std::size_t n = static_cast<std::size_t>(std::min<u64>(fill_left_, out.size()));
      std::memset(out.data(), std::to_integer<int>(opt_.fill), n);
      out = out.subspan(n);
      p.produced += n;
      fill_left_ -= n;
    }
    return then(after_fill_);
  }

  // ---- R20 recovery ----

  void note_damage() noexcept {
    ++stats_.damage_events;
    ++stream_damage_;
  }

  void open_gap(Errc cause) noexcept {
    if (gap_open_) return;
    gap_open_ = true;
    gap_start_ = logical_;
    gap_cause_ = cause;
    gap_in_ = in_offset_;
  }

  // Ends the open gap at uncompressed offset `at` (>= its start). In fill mode
  // the gap becomes output, so it is charged to the R19 budget like any other.
  [[nodiscard]] Errc close_gap(u64 at) noexcept {
    if (!gap_open_) return Errc::ok;
    gap_open_ = false;
    const u64 len = at - gap_start_;
    if (len == 0) return Errc::ok;
    ++stats_.gaps;
    stats_.bytes_lost += len;
    if (opt_.on_gap) opt_.on_gap(opt_.gap_ctx, GapInfo{stream_index_, gap_start_, len, true, gap_in_, gap_cause_});
    if (opt_.recovery != Recovery::skip_and_fill) return Errc::ok;
    if (const Errc s = budget_.spend(len); s != Errc::ok) return s;
    stats_.bytes_filled += len;
    if (split_) return Errc::ok;  // the pipeline writes fills, in stream order
    if (job_fill_) {
      job_fill_(job_ctx_, job_produced_, len);
      return Errc::ok;
    }
    fill_left_ = len;
    return Errc::ok;
  }

  // Payload or content check failed but the header was valid, so the block's
  // extent is known; the rest of the frame is lost except STORED blocks.
  Errc lose_block(Errc cause) noexcept {
    if (!recovering()) return cause;
    note_damage();
    ++stats_.blocks_lost;
    budget_.refund(cur_.usize);
    open_gap(cause);
    broken_cause_ = cause;
    frame_broken_ = true;
    logical_ += cur_.usize;
    return then(St::item);
  }

  // A header that cannot be trusted: sizes are unknown, so resync on the next frame.
  Errc damaged(Errc cause) noexcept {
    if (!recovering()) return cause;
    if (split_) {
      if (const Errc s = split_->damage(cause, logical_, in_offset_, stream_index_); s != Errc::ok) return s;
    }
    note_damage();
    open_gap(cause);
    return enter_scan();
  }

  // The staged item was not valid: resume the search at byte `from` of it.
  Errc enter_scan(std::size_t from = 1) noexcept {
    in_frame_ = false;
    st_ = St::scan;
    drop_candidate(from);
    return Errc::ok;
  }

  Errc reject(Errc e, bool scanning) noexcept {
    if (!scanning) return damaged(e);
    drop_candidate();
    return Errc::ok;
  }

  // Keeps only the staged suffix that starts at the first marker byte at or
  // after `from`; everything before it is discarded. Nothing is re-read from input.
  void drop_candidate(std::size_t from = 1) noexcept {
    std::size_t keep = 0;
    if (hdr_len_ > from) {
      const auto first = hdr_.begin() + static_cast<std::ptrdiff_t>(from);
      const auto last = hdr_.begin() + static_cast<std::ptrdiff_t>(hdr_len_);
      keep = static_cast<std::size_t>(last - std::find(first, last, kSyncMarker[0]));
      if (keep) std::memmove(hdr_.data(), hdr_.data() + (hdr_len_ - keep), keep);
    }
    budget_.earn(hdr_len_ - keep);
    hdr_len_ = keep;
  }

  // Slides over the input looking for a frame header that passes marker, CRC,
  // stream identity and ordering checks. Candidates are staged in hdr_, so it
  // works across any input chunking and never needs to seek backwards.
  Errc on_scan(Bytes& in, Progress& p) noexcept {
    for (;;) {
      if (hdr_len_ == 0) {
        const auto hit = std::find(in.begin(), in.end(), kSyncMarker[0]);
        const auto skipped = static_cast<std::size_t>(hit - in.begin());
        budget_.earn(skipped);
        consume(in, skipped, p);
        if (in.empty()) return Errc::need_input;
      }
      const bool complete = gather(in, kFrameHeaderSize, p);
      const std::size_t prefix = std::min(hdr_len_, kSyncMarker.size());
      if (!std::equal(hdr_.begin(), hdr_.begin() + static_cast<std::ptrdiff_t>(prefix), kSyncMarker.begin())) {
        drop_candidate();
        continue;
      }
      if (!complete) return Errc::need_input;
      FrameHeader f;
      if (parse_frame_header(staged(), f) != Errc::ok) {
        drop_candidate();
        continue;
      }
      if (const Errc e = accept_frame(f); e != Errc::ok || st_ != St::scan) return e;
    }
  }

  Errc on_truncated() noexcept {
    if (st_ == St::stream_header && stream_index_ > 0 && !recovering()) return Errc::trailing_garbage;
    if (recovering() && params_valid_) {
      if (split_) split_->truncated(logical_, in_offset_, stream_index_);
      open_gap(Errc::truncated);
      gap_open_ = false;
      ++stats_.gaps;
      if (opt_.on_gap) opt_.on_gap(opt_.gap_ctx, GapInfo{stream_index_, gap_start_, 0, false, gap_in_, gap_cause_});
    }
    return Errc::truncated;
  }

  u32 window_crc(u64 from, u64 len) const noexcept {
    u32 c = 0;
    while (len) {
      const Bytes piece = window_.contiguous(from, len);
      c = crc32c(piece, c);
      from += piece.size();
      len -= piece.size();
    }
    return c;
  }

  Codec& codec_;
  DecoderOptions opt_;
  Budget budget_;
  DecodeStats stats_;
  StreamParams params_;
  Window window_;
  ArenaPtr arena_;
  u64 arena_cap_ = 0;
  MutBytes payload_;

  St st_ = St::stream_header;
  St after_fill_ = St::item;
  Errc err_ = Errc::ok;
  std::array<std::byte, kMaxItemSize> hdr_{};
  std::size_t hdr_len_ = 0;
  BlockHeader cur_;
  std::size_t payload_len_ = 0;
  u64 skip_left_ = 0;
  u32 skip_crc_ = 0;
  bool skip_verify_ = false;
  u64 emit_pos_ = 0, emit_left_ = 0, fill_left_ = 0;

  bool params_valid_ = false, in_frame_ = false, frame_broken_ = false, gap_open_ = false;
  u32 stream_index_ = 0;
  u64 logical_ = 0;    // uncompressed offset of the next byte, including lost bytes
  u64 frames_ = 0;     // next expected frame index
  u32 list_crc_ = 0;   // running END check
  u64 in_offset_ = 0;  // total input consumed, for diagnostics
  u64 stream_damage_ = 0;
  u64 gap_start_ = 0, gap_in_ = 0;
  Errc gap_cause_ = Errc::ok, broken_cause_ = Errc::ok;

  SplitSink* split_ = nullptr;  // split mode
  bool job_ = false;            // job mode
  FillFn job_fill_ = nullptr;
  void* job_ctx_ = nullptr;
  u64 job_produced_ = 0;  // bytes emitted in the current job
};

// Drives any push decoder from a non-seekable source (stdin, socket) to a sink
// with two fixed buffers: constant memory regardless of stream length (R18).
template <ByteSource Src, ByteSink Snk, class Dec>
Errc pump(Src& src, Snk& snk, Dec& dec, MutBytes in_buf, MutBytes out_buf) {
  std::size_t have = 0, used = 0;
  bool eof = false;
  for (;;) {
    if (used == have && !eof) {
      have = src.read(in_buf);
      used = 0;
      eof = have == 0;
    }
    const Progress p = dec.decode(Bytes(in_buf).subspan(used, have - used), out_buf, eof);
    used += p.consumed;
    if (p.produced && !snk.write(Bytes(out_buf).first(p.produced))) return Errc::io_error;
    switch (p.status) {
      case Errc::need_input:
      case Errc::output_full: continue;
      case Errc::end_of_stream:
        if (eof && used == have) return Errc::ok;
        continue;
      default: return p.status;
    }
  }
}

// R19 for legacy (v1/v2/classic v3) decoders whose formats never declared
// block sizes: police input credit and output volume at the I/O boundary.
template <ByteSource Src>
struct GuardedSource {
  Src& inner;
  Budget& budget;
  std::size_t read(MutBytes b) {
    const std::size_t n = inner.read(b);
    budget.earn(n);
    return n;
  }
};

template <ByteSink Snk>
struct GuardedSink {
  Snk& inner;
  Budget& budget;
  Errc status = Errc::ok;
  bool write(Bytes b) {
    if (status == Errc::ok) status = budget.spend(b.size());
    return status == Errc::ok && inner.write(b);
  }
};

}  // namespace hc
