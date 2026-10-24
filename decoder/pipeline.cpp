// pipeline.cpp - multi-threaded FRAMED v3 decoder (see pipeline.hpp).
//
//   splitter (caller) --chunks--> T workers --results--+
//        |                                             v
//        +------------- ordered entries ---------> writer --> output
//
// The splitter is a StreamDecoder in split mode: it validates every header,
// resynchronizes after damage and enforces the R19 budget in stream order,
// exactly as the sequential decoder does, but decodes nothing. It cuts each
// frame into chunks of bounded input and output size held in T+2
// preallocated slots (counting-semaphore back-pressure). Workers claim frames
// in order; a worker decodes all chunks of its frame with a job-mode
// StreamDecoder and private codec state (R11). The writer emits chunks in
// stream order and replays the only recovery logic that spans frames: a gap
// left open at the end of one frame is closed by the next frame header.
//
// Chunk k always occupies slot k mod (T+2) and frame f record f mod (T+2):
// slots are taken in order and returned in order, and every frame in flight
// holds at least one slot. Coordination is one mutex and three condition
// variables, touched a few times per chunk (megabytes of work).
#include "decoder/pipeline.hpp"

#include "codec/neural_cm_codec.hpp"
#include "structural/structural_cm_codec.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <semaphore>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace hce {
namespace {

using hc::Bytes;
using hc::Errc;
using hc::MutBytes;
using hc::u32;
using hc::u64;

constexpr std::size_t kReadBytes = std::size_t{1} << 20;
constexpr std::size_t kFillBytes = std::size_t{64} << 10;
constexpr unsigned kMaxWorkers = 256;

// DecodeOptions::codec_context, for codecs that take shared read-only state.
template <class C>
void attach_context(C& codec, const void* ctx) noexcept {
  if constexpr (requires { codec.set_context(ctx); }) codec.set_context(ctx);
}

hc::DecoderOptions decoder_options(const DecodeOptions& o) {
  hc::DecoderOptions d;
  d.limits = o.limits;
  d.recovery = o.recovery;
  d.fill = o.fill;
  d.on_gap = o.on_gap;
  d.gap_ctx = o.gap_ctx;
  d.stop = o.stop;
  return d;
}

// ------------------------------------------------------------ sequential ---

class CountingInput final : public InputStream {
 public:
  explicit CountingInput(InputStream& in) : in_(in) {}
  std::size_t read(MutBytes buf) override {
    const std::size_t n = in_.read(buf);
    count += n;
    return n;
  }
  Errc error() const noexcept override { return in_.error(); }
  u64 count = 0;

 private:
  InputStream& in_;
};

class CountingOutput final : public OutputStream {
 public:
  explicit CountingOutput(OutputStream& out) : out_(out) {}
  bool write(Bytes buf) override {
    if (!out_.write(buf)) return false;
    count += buf.size();
    return true;
  }
  u64 count = 0;

 private:
  OutputStream& out_;
};

template <hc::BlockCodec Codec>
DecodeResult decode_sequential(InputStream& in, OutputStream& out, const DecodeOptions& opt) {
  DecodeResult r;
  Codec codec;
  attach_context(codec, opt.codec_context);
  hc::StreamDecoder<Codec> dec(codec, decoder_options(opt));
  const std::unique_ptr<std::byte[]> ib(new std::byte[kReadBytes]);
  const std::unique_ptr<std::byte[]> ob(new std::byte[kReadBytes]);
  CountingInput counted_in(in);
  CountingOutput counted_out(out);
  r.status = hc::pump(counted_in, counted_out, dec, MutBytes(ib.get(), kReadBytes), MutBytes(ob.get(), kReadBytes));
  if (in.error() != Errc::ok) r.status = in.error();  // a failed read is not a truncated stream
  r.stats = dec.stats();
  r.bytes_in = counted_in.count;
  r.bytes_out = counted_out.count;
  r.memory_bound = 2 * kReadBytes;
  if (dec.params_valid()) {
    const hc::StreamParams& p = dec.params();
    r.memory_bound += p.window_size() + p.block_capacity() + Codec::memory_required(p);
  }
  r.params_known = dec.params_valid() || r.stats.frames > 0 || r.bytes_out > 0;
  return r;
}

// ------------------------------------------------------------- pipeline ---

struct Plan {
  std::size_t in_cap = 0, out_cap = 0;  // per chunk
  unsigned threads = 0;
  u64 worker_memory = 0;  // window + block + model of the largest stream the limits accept
  u64 memory = 0;
};

struct Fill {
  u64 at = 0, len = 0;  // position within the chunk's output
};

struct Chunk {
  std::unique_ptr<std::byte[]> in, out;
  std::size_t in_len = 0, out_len = 0;
  u64 declared_out = 0;  // decoded bytes its blocks declare
  u64 frame = 0;         // frame sequence number
  bool first = false, last = false;
  // Results; the worker publishes them by setting `done` under the mutex.
  bool done = false;
  Errc status = Errc::ok;
  std::vector<Fill> fills;  // recovery fills, out of band
  std::vector<hc::GapInfo> gaps;
  hc::DecodeStats stats;  // the frame's job totals, on its last chunk
  u64 logical_end = 0;
  bool pending = false;  // a gap is still open at the end of the frame
  u64 pending_start = 0, pending_in = 0;
  Errc pending_cause = Errc::ok;
};

template <class JobStart>
struct FrameRec {
  hc::StreamParams params;
  JobStart start;
  std::vector<u64> ring;  // chunk sequence numbers, guarded by the mutex
  u64 pushed = 0;
};

struct Entry {
  enum class Kind : hc::u8 { chunk, damage, end, truncated, error, finish };
  Kind kind = Kind::finish;
  u64 chunk = 0;
  Errc cause = Errc::ok;
  u64 in_offset = 0;
  u32 stream_index = 0;
  hc::EndTrailer trailer{};
  bool verified = false;
};

template <hc::BlockCodec Codec>
struct Worker {
  explicit Worker(const hc::DecoderOptions& base) : dec(codec, with_callbacks(base, this)) {}

  static hc::DecoderOptions with_callbacks(hc::DecoderOptions o, Worker* self) {
    o.on_gap = &on_gap;
    o.gap_ctx = self;
    return o;
  }
  static void on_gap(void* ctx, const hc::GapInfo& g) noexcept {
    auto* w = static_cast<Worker*>(ctx);
    try {
      w->chunk->gaps.push_back(g);
    } catch (...) {
      w->oom = true;
    }
  }
  static void on_fill(void* ctx, u64 at, u64 len) noexcept {
    auto* w = static_cast<Worker*>(ctx);
    try {
      w->chunk->fills.push_back(Fill{at - w->chunk_base, len});
    } catch (...) {
      w->oom = true;
    }
  }

  Codec codec;
  hc::StreamDecoder<Codec> dec;
  Chunk* chunk = nullptr;  // chunk being decoded, for the callbacks
  u64 chunk_base = 0;      // job output produced before it
  u64 produced = 0;        // job output so far
  bool oom = false;
};

template <hc::BlockCodec Codec>
std::optional<Plan> make_plan(const DecodeOptions& o) {
  const hc::DecodeLimits& lim = o.limits;
  // Every worker must be able to hold the largest stream the limits accept.
  hc::StreamParams p;
  p.window_log = std::min(lim.max_window_log, hc::kMaxWindowLog);
  if (p.window_log < hc::kMinWindowLog || lim.max_block_bytes < (u32{1} << hc::kMinBlockLog)) return std::nullopt;
  const int block_log = static_cast<int>(std::bit_width(lim.max_block_bytes)) - 1;
  p.block_log = static_cast<hc::u8>(std::min<int>({block_log, hc::kMaxBlockLog, p.window_log}));
  p.level = lim.max_level;
  const u64 worker = p.window_size() + p.block_capacity() + Codec::memory_required(p);
  const u64 out_cap = std::max<u64>(o.chunk_bytes, p.block_capacity());
  const u64 in_cap = std::max<u64>(out_cap, hc::kMaxAncillaryBytes) + hc::kFrameHeaderSize + hc::kBlockHeaderSize;
  const u64 slot = in_cap + out_cap + sizeof(Chunk) + sizeof(FrameRec<int>) + 5 * sizeof(Entry);
  for (unsigned t = std::min(o.threads, kMaxWorkers); t >= 2; --t) {
    const u64 total = (u64{t} + 2) * slot + u64{t} * (worker + sizeof(Worker<Codec>)) + kReadBytes + kFillBytes;
    if (total <= lim.max_memory_bytes)
      return Plan{static_cast<std::size_t>(in_cap), static_cast<std::size_t>(out_cap), t, worker, total};
  }
  return std::nullopt;  // not even two workers fit: decode sequentially
}

template <hc::BlockCodec Codec>
class Pipeline final : public hc::SplitSink {
  using Decoder = hc::StreamDecoder<Codec>;
  using JobStart = typename Decoder::JobStart;

 public:
  Pipeline(InputStream& in, OutputStream& out, const DecodeOptions& opt, const Plan& plan)
      : in_(in),
        out_(out),
        opt_(opt),
        plan_(plan),
        chunks_(plan.threads + 2),
        frames_(plan.threads + 2),
        entries_(4 * (plan.threads + 2) + 16) {}

  DecodeResult run() {
    DecodeResult r;
    r.threads = plan_.threads;
    r.memory_bound = plan_.memory;
    if (!allocate()) {
      r.status = Errc::out_of_memory;
      return r;
    }
    hc::DecoderOptions wopt = decoder_options(opt_);
    wopt.limits.max_ratio = 0;  // the splitter owns the R19 budget
    wopt.limits.max_output_bytes = ~u64{0};
    wopt.limits.max_memory_bytes = plan_.worker_memory;
    std::vector<std::unique_ptr<Worker<Codec>>> workers;
    for (unsigned i = 0; i < plan_.threads; ++i) workers.push_back(std::make_unique<Worker<Codec>>(wopt));
    for (auto& wp : workers) attach_context(wp->codec, opt_.codec_context);  // before any worker starts

    const std::stop_callback on_stop(opt_.stop, [this]() noexcept { fail(Errc::cancelled); });
    {
      std::vector<std::jthread> threads;
      try {
        threads.reserve(plan_.threads + 1);
        threads.emplace_back([this] { write_loop(); });
        for (auto& wp : workers) threads.emplace_back([this, &w = *wp] { work_loop(w); });
      } catch (const std::system_error&) {
        fail(Errc::out_of_memory);  // could not start threads: unwind cleanly
      }
      split_loop();
    }  // joins every thread

    r.status = complete_ ? Errc::ok : error_;
    r.stats = stats_;
    r.stats.damage_events += splitter_damage_;
    r.bytes_in = bytes_in_;
    r.bytes_out = bytes_out_;
    r.params_known = splitter_params_ || r.stats.frames > 0 || r.bytes_out > 0;
    return r;
  }

  // ---- hc::SplitSink: called on the splitter thread ----

  Errc frame(const hc::FrameHeader& f, Bytes header, u64 in_offset, u32 stream_index) noexcept override {
    if (!close_frame() || !acquire_slot()) return Errc::cancelled;
    // Safe without the lock: holding a slot proves frame (seq - slots) is fully written.
    FrameRec<JobStart>& fr = frames_[frame_seq_ % frames_.size()];
    fr.params = f.params;
    fr.start = JobStart{f.out_offset, f.index, in_offset, stream_index};
    fr.pushed = 0;
    cur_frame_ = &fr;
    cur_frame_seq_ = frame_seq_++;
    frame_open_ = true;
    open_chunk(true);
    return append(header);
  }

  Errc block(const hc::BlockHeader& b, Bytes header) noexcept override {
    if (!cur_chunk_) return Errc::cancelled;
    const Chunk& c = *cur_chunk_;
    if (c.in_len + hc::kBlockHeaderSize + b.csize > plan_.in_cap || c.declared_out + b.usize > plan_.out_cap) {
      if (!close_chunk(false) || !acquire_slot()) return Errc::cancelled;
      open_chunk(false);  // the frame continues in a new chunk, decoded by the same worker
    }
    cur_chunk_->declared_out += b.usize;
    return append(header);
  }

  Errc payload(Bytes bytes) noexcept override { return append(bytes); }

  Errc damage(Errc cause, u64, u64 in_offset, u32 stream_index) noexcept override {
    Entry e;
    e.kind = Entry::Kind::damage;
    e.cause = cause;
    e.in_offset = in_offset;
    e.stream_index = stream_index;
    return close_frame() && push(e) ? Errc::ok : Errc::cancelled;
  }

  Errc end(const hc::EndTrailer& t, bool verified, u64 in_offset, u32 stream_index) noexcept override {
    Entry e;
    e.kind = Entry::Kind::end;
    e.trailer = t;
    e.verified = verified;
    e.in_offset = in_offset;
    e.stream_index = stream_index;
    return close_frame() && push(e) ? Errc::ok : Errc::cancelled;
  }

  void truncated(u64, u64 in_offset, u32 stream_index) noexcept override {
    Entry e;
    e.kind = Entry::Kind::truncated;
    e.in_offset = in_offset;
    e.stream_index = stream_index;
    if (close_frame()) push(e);
  }

 private:
  struct Gap {
    bool open = false;
    u64 start = 0, in = 0;
    Errc cause = Errc::ok;
  };

  bool allocate() noexcept {
    for (Chunk& c : chunks_) {
      // Uninitialized: untouched pages cost no RSS, so small inputs stay small.
      c.in.reset(new (std::nothrow) std::byte[plan_.in_cap]);
      c.out.reset(new (std::nothrow) std::byte[plan_.out_cap]);
      if (!c.in || !c.out) return false;
    }
    try {
      for (FrameRec<JobStart>& f : frames_) f.ring.assign(chunks_.size(), 0);
    } catch (...) {
      return false;
    }
    read_buf_.reset(new (std::nothrow) std::byte[kReadBytes]);
    fill_buf_.reset(new (std::nothrow) std::byte[kFillBytes]);
    if (!read_buf_ || !fill_buf_) return false;
    std::fill_n(fill_buf_.get(), kFillBytes, opt_.fill);
    return true;
  }

  void fail(Errc e) noexcept {
    {
      const std::lock_guard lk(m_);
      if (stop_) return;
      stop_ = true;
      error_ = e;
    }
    stopping_.store(true, std::memory_order_relaxed);
    work_cv_.notify_all();
    writer_cv_.notify_all();
    space_cv_.notify_all();
    in_.interrupt();  // wake a read or write blocked in the I/O layer
    out_.interrupt();
  }

  bool acquire_slot() {
    while (!free_.try_acquire_for(std::chrono::milliseconds(50)))
      if (stopping_.load(std::memory_order_relaxed)) return false;
    return !stopping_.load(std::memory_order_relaxed);
  }

  bool push(const Entry& e) {
    {
      std::unique_lock lk(m_);
      space_cv_.wait(lk, [this] { return stop_ || entries_tail_ - entries_head_ < entries_.size(); });
      if (stop_) return false;
      entries_[entries_tail_++ % entries_.size()] = e;
    }
    writer_cv_.notify_one();
    return true;
  }

  // ---- splitter side ----

  void split_loop() {
    Codec unused;  // split mode never binds or calls the codec
    hc::DecoderOptions so = decoder_options(opt_);
    so.on_gap = nullptr;
    so.gap_ctx = nullptr;
    Decoder split(unused, so);
    split.set_split_sink(this);
    const MutBytes buf(read_buf_.get(), kReadBytes);
    std::size_t have = 0, used = 0;
    bool eof = false;
    Errc status = Errc::ok;
    for (;;) {
      if (used == have && !eof) {
        have = in_.read(buf);
        used = 0;
        bytes_in_ += have;
        if (have == 0) {
          eof = true;
          if (in_.error() != Errc::ok) {
            status = in_.error();  // a failed read is not a truncated stream
            break;
          }
        }
      }
      const hc::Progress p = split.decode(Bytes(buf).subspan(used, have - used), MutBytes{}, eof);
      used += p.consumed;
      if (p.status == Errc::need_input) continue;
      if (p.status == Errc::end_of_stream) {
        if (eof && used == have) break;
        continue;  // a concatenated stream follows
      }
      status = p.status;
      break;
    }
    splitter_damage_ = split.stats().damage_events;
    splitter_params_ = split.params_valid() || split.stats().frames > 0;
    Entry last;
    last.kind = status == Errc::ok ? Entry::Kind::finish : Entry::Kind::error;
    last.cause = status;
    if (close_frame()) push(last);
    {
      const std::lock_guard lk(m_);
      splitter_done_ = true;
    }
    work_cv_.notify_all();
  }

  void open_chunk(bool first) {
    Chunk& c = chunks_[chunk_seq_ % chunks_.size()];
    c.in_len = c.out_len = 0;
    c.declared_out = 0;
    c.frame = cur_frame_seq_;
    c.first = first;
    c.last = false;
    c.status = Errc::ok;
    c.fills.clear();
    c.gaps.clear();
    c.stats = {};
    c.logical_end = 0;
    c.pending = false;
    cur_chunk_ = &c;
    cur_chunk_seq_ = chunk_seq_++;
  }

  // Hands the open chunk to its frame's worker and the writer. From here on the
  // splitter must not touch it, so cur_chunk_ is cleared before it is published.
  bool close_chunk(bool last) {
    Chunk& c = *std::exchange(cur_chunk_, nullptr);
    c.last = last;
    {
      const std::lock_guard lk(m_);
      FrameRec<JobStart>& f = *cur_frame_;
      f.ring[f.pushed % f.ring.size()] = cur_chunk_seq_;
      ++f.pushed;
      if (c.first) ++frames_published_;
    }
    work_cv_.notify_all();
    Entry e;
    e.kind = Entry::Kind::chunk;
    e.chunk = cur_chunk_seq_;
    return push(e);
  }

  bool close_frame() {
    if (!frame_open_) return true;
    frame_open_ = false;
    // No open chunk: a cut dispatched one and could not get a new slot, which
    // only happens while the pipeline is stopping.
    return cur_chunk_ != nullptr && close_chunk(true);
  }

  Errc append(Bytes bytes) noexcept {
    if (!cur_chunk_) return Errc::cancelled;
    Chunk& c = *cur_chunk_;
    if (bytes.size() > plan_.in_cap - c.in_len) return Errc::out_of_memory;  // unreachable: block() made room
    std::memcpy(c.in.get() + c.in_len, bytes.data(), bytes.size());
    c.in_len += bytes.size();
    return Errc::ok;
  }

  // ---- worker side ----

  void work_loop(Worker<Codec>& w) {
    for (;;) {
      u64 fseq = 0;
      {
        std::unique_lock lk(m_);
        work_cv_.wait(lk, [this] { return stop_ || frames_claimed_ < frames_published_ || splitter_done_; });
        if (stop_ || frames_claimed_ >= frames_published_) return;
        fseq = frames_claimed_++;
      }
      FrameRec<JobStart>& f = frames_[fseq % frames_.size()];
      Errc job = Errc::ok;
      for (u64 k = 0;; ++k) {
        u64 cseq = 0;
        {
          std::unique_lock lk(m_);
          work_cv_.wait(lk, [&] { return stop_ || k < f.pushed; });
          if (stop_) return;
          cseq = f.ring[k % f.ring.size()];
        }
        Chunk& c = chunks_[cseq % chunks_.size()];
        decode_chunk(w, f, c, job);
        const bool last = c.last;  // read before publishing: the slot may be reused right after
        {
          const std::lock_guard lk(m_);
          c.done = true;
        }
        writer_cv_.notify_one();
        if (last) break;
      }
    }
  }

  void decode_chunk(Worker<Codec>& w, const FrameRec<JobStart>& f, Chunk& c, Errc& job) {
    w.chunk = &c;
    if (c.first) {
      w.produced = 0;
      w.oom = false;
      job = w.dec.begin_job(f.params, f.start, &Worker<Codec>::on_fill, &w);
    }
    if (job == Errc::ok) {
      w.chunk_base = w.produced;
      const hc::Progress p =
          w.dec.decode(Bytes(c.in.get(), c.in_len), MutBytes(c.out.get(), plan_.out_cap), c.last);
      c.out_len = p.produced;
      w.produced += p.produced;
      const Errc expected = c.last ? Errc::end_of_stream : Errc::need_input;
      if (p.status != expected) job = hc::is_error(p.status) ? p.status : Errc::out_of_memory;
      if (w.oom) job = Errc::out_of_memory;
    }
    c.status = job;
    if (c.last || job != Errc::ok) {
      c.stats = w.dec.stats();
      c.logical_end = w.dec.logical();
      if (const auto g = w.dec.pending_gap()) {
        c.pending = true;
        c.pending_start = g->out_offset;
        c.pending_in = g->in_offset;
        c.pending_cause = g->cause;
      }
    }
  }

  // ---- writer side ----

  void write_loop() {
    for (;;) {
      Entry e;
      {
        std::unique_lock lk(m_);
        writer_cv_.wait(lk, [this] { return stop_ || entries_head_ < entries_tail_; });
        if (stop_) return;
        e = entries_[entries_head_++ % entries_.size()];
      }
      space_cv_.notify_one();
      if (!handle(e)) return;
    }
  }

  bool handle(const Entry& e) {
    switch (e.kind) {
      case Entry::Kind::chunk: return write_chunk(e.chunk);
      case Entry::Kind::damage:
        enter_stream(e.stream_index);
        open_gap(e.cause, e.in_offset);
        return true;
      case Entry::Kind::end: enter_stream(e.stream_index); return end_stream(e);
      case Entry::Kind::truncated:
        enter_stream(e.stream_index);
        open_gap(Errc::truncated, e.in_offset);
        gap_.open = false;
        ++stats_.gaps;
        report(hc::GapInfo{stream_, gap_.start, 0, false, gap_.in, gap_.cause});
        return true;
      case Entry::Kind::error: fail(e.cause); return false;
      case Entry::Kind::finish: {
        const std::lock_guard lk(m_);
        complete_ = true;
        return false;
      }
    }
    return false;
  }

  bool write_chunk(u64 seq) {
    Chunk& c = chunks_[seq % chunks_.size()];
    if (c.first) {  // the frame header was accepted: close any gap carried from earlier frames
      const FrameRec<JobStart>& f = frames_[c.frame % frames_.size()];
      enter_stream(f.start.stream_index);
      if (f.start.out_offset > logical_) open_gap(Errc::sequence_gap, f.start.in_offset + hc::kFrameHeaderSize);
      if (!close_gap(f.start.out_offset)) return false;
      logical_ = f.start.out_offset;
    }
    {
      std::unique_lock lk(m_);
      writer_cv_.wait(lk, [&] { return stop_ || c.done; });
      if (stop_) return false;
    }
    std::size_t at = 0;  // decoded bytes interleaved with out-of-band fills
    for (const Fill& f : c.fills) {
      const std::size_t to = static_cast<std::size_t>(f.at);
      if (!emit(Bytes(c.out.get() + at, to - at)) || !emit_fill(f.len)) return false;
      at = to;
    }
    if (!emit(Bytes(c.out.get() + at, c.out_len - at))) return false;
    for (const hc::GapInfo& g : c.gaps) report(g);
    if (c.last || c.status != Errc::ok) {
      add_job(c.stats);
      logical_ = c.logical_end;
      if (c.pending) gap_ = Gap{true, c.pending_start, c.pending_in, c.pending_cause};
    }
    if (c.status != Errc::ok) {
      fail(c.status);
      return false;
    }
    {
      const std::lock_guard lk(m_);
      c.done = false;
    }
    free_.release();
    return true;
  }

  bool end_stream(const Entry& e) {
    if (opt_.recovery == hc::Recovery::strict) {  // the splitter already verified the trailer
      stats_.content_verified = true;
      return true;
    }
    if (e.trailer.total_out > logical_) open_gap(Errc::sequence_gap, e.in_offset);
    if (!close_gap(std::max(e.trailer.total_out, logical_))) return false;
    stats_.content_verified = e.verified && stream_damage_ == 0;
    logical_ = std::max(e.trailer.total_out, logical_);
    return true;
  }

  void enter_stream(u32 index) {
    if (index == stream_) return;
    stream_ = index;
    logical_ = 0;
    gap_ = Gap{};
    stream_damage_ = 0;
  }

  void add_job(const hc::DecodeStats& s) {
    stats_.frames += s.frames;
    stats_.blocks_ok += s.blocks_ok;
    stats_.blocks_salvaged += s.blocks_salvaged;
    stats_.blocks_lost += s.blocks_lost;
    stats_.damage_events += s.damage_events;
    stats_.gaps += s.gaps;
    stats_.bytes_lost += s.bytes_lost;
    stats_.bytes_filled += s.bytes_filled;
    stream_damage_ += s.damage_events;
  }

  void open_gap(Errc cause, u64 in_offset) {
    if (gap_.open) return;
    gap_ = Gap{true, logical_, in_offset, cause};
  }

  bool close_gap(u64 at) {
    if (!gap_.open) return true;
    gap_.open = false;
    const u64 len = at - gap_.start;
    if (len == 0) return true;
    ++stats_.gaps;
    stats_.bytes_lost += len;
    report(hc::GapInfo{stream_, gap_.start, len, true, gap_.in, gap_.cause});
    if (opt_.recovery != hc::Recovery::skip_and_fill) return true;
    stats_.bytes_filled += len;  // the splitter charged it to the budget
    return emit_fill(len);
  }

  void report(const hc::GapInfo& g) {
    if (opt_.on_gap) opt_.on_gap(opt_.gap_ctx, g);
  }

  bool emit(Bytes b) {
    if (b.empty()) return true;
    if (!out_.write(b)) {
      fail(Errc::io_error);
      return false;
    }
    bytes_out_ += b.size();
    return true;
  }

  bool emit_fill(u64 len) {
    while (len) {
      const std::size_t n = static_cast<std::size_t>(std::min<u64>(len, kFillBytes));
      if (!emit(Bytes(fill_buf_.get(), n))) return false;
      len -= n;
    }
    return true;
  }

  InputStream& in_;
  OutputStream& out_;
  const DecodeOptions& opt_;
  Plan plan_;
  std::vector<Chunk> chunks_;
  std::vector<FrameRec<JobStart>> frames_;
  std::vector<Entry> entries_;
  std::unique_ptr<std::byte[]> read_buf_, fill_buf_;
  std::counting_semaphore<kMaxWorkers + 2> free_{static_cast<std::ptrdiff_t>(plan_.threads + 2)};
  std::atomic<bool> stopping_{false};

  std::mutex m_;
  std::condition_variable work_cv_, writer_cv_, space_cv_;
  // Guarded by m_.
  u64 frames_published_ = 0, frames_claimed_ = 0;
  u64 entries_head_ = 0, entries_tail_ = 0;
  bool splitter_done_ = false, stop_ = false, complete_ = false;
  Errc error_ = Errc::ok;

  // Splitter thread.
  u64 frame_seq_ = 0, chunk_seq_ = 0, cur_frame_seq_ = 0, cur_chunk_seq_ = 0;
  FrameRec<JobStart>* cur_frame_ = nullptr;
  Chunk* cur_chunk_ = nullptr;
  bool frame_open_ = false;
  u64 bytes_in_ = 0, splitter_damage_ = 0;
  bool splitter_params_ = false;

  // Writer thread.
  hc::DecodeStats stats_;
  u64 bytes_out_ = 0, logical_ = 0, stream_damage_ = 0;
  u32 stream_ = 0;
  Gap gap_;
};

}  // namespace

template <hc::BlockCodec Codec>
DecodeResult decode_stream(InputStream& in, OutputStream& out, const DecodeOptions& opt) {
  try {
    if (opt.threads > 1) {
      if (const std::optional<Plan> plan = make_plan<Codec>(opt)) {
        Pipeline<Codec> pipeline(in, out, opt, *plan);
        return pipeline.run();
      }
    }
    return decode_sequential<Codec>(in, out, opt);
  } catch (const std::bad_alloc&) {
    DecodeResult r;
    r.status = Errc::out_of_memory;
    return r;
  }
}

template DecodeResult decode_stream<NeuralCMCodec>(InputStream&, OutputStream&, const DecodeOptions&);
template DecodeResult decode_stream<StructuralCMCodec>(InputStream&, OutputStream&, const DecodeOptions&);

}  // namespace hce
