// parallel.cpp - multi-threaded FRAMED v3 encoder (see parallel.hpp).
//
// The calling thread reads frames, T workers encode them, one writer emits
// them in order. Frame k always occupies slot k mod (T+2): the reader takes
// slots through a counting semaphore and the writer gives them back strictly
// in order, so a slot is never refilled before its previous frame was
// written. Coordination is one mutex and two condition variables, touched a
// few times per frame (megabytes of work), so it never shows in a profile.
#include "encoder/parallel.hpp"

#include "codec/neural_cm_codec.hpp"
#include "structural/structural_cm_codec.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <semaphore>
#include <system_error>
#include <thread>
#include <vector>

namespace hce {
namespace {

using hc::Bytes;
using hc::Errc;
using hc::MutBytes;
using hc::u32;
using hc::u64;

constexpr std::size_t kStreamIdSample = 64 * 1024;

// EncodeOptions::codec_context, for codecs that take shared read-only state.
template <class C>
void attach_context(C& codec, const void* ctx) noexcept {
  if constexpr (requires { codec.set_context(ctx); }) codec.set_context(ctx);
}

struct Plan {
  std::size_t frame_cap = 0;   // input bytes per frame
  std::size_t out_cap = 0;     // encoded bytes per frame in the worst case (all STORED)
  std::size_t max_blocks = 0;  // blocks per frame
  unsigned threads = 0;
  bool pipelined = false;  // each worker's codec also runs a helper thread
  u64 memory = 0;
};

struct Slot {
  std::unique_ptr<std::byte[]> in, out;
  std::unique_ptr<u32[]> checks;  // END list entry of every block
  std::size_t in_len = 0, out_len = 0, blocks = 0, stored = 0;
  u64 out_offset = 0;
  u32 index = 0;
  bool done = false;  // guarded by Pipeline::m_
};

template <FrameEncoder Enc>
Plan make_plan(const EncodeOptions& o, const hc::StreamParams& p) {
  Plan pl;
  const u64 frame = o.content_size ? std::clamp<u64>(*o.content_size, 1, o.frame_bytes) : o.frame_bytes;
  const u64 block = p.block_capacity();
  const u64 blocks = (frame + block - 1) / block;
  pl.frame_cap = static_cast<std::size_t>(frame);
  pl.max_blocks = static_cast<std::size_t>(blocks);
  pl.out_cap = static_cast<std::size_t>(hc::kFrameHeaderSize + blocks * hc::kBlockHeaderSize + frame);
  const u64 slot = frame + pl.out_cap + blocks * sizeof(u32) + sizeof(Slot);
  const u64 worker = Enc::memory_required(p) + sizeof(Enc);
  // A known input size caps the workers at its frame count: an idle worker
  // would still allocate and clear a whole model. With fewer frames than half
  // the threads, the spare threads go to the codecs' pipelines, one helper per
  // worker. The output is the same bytes either way.
  const u64 frames = o.content_size ? std::max<u64>(1, (*o.content_size + o.frame_bytes - 1) / o.frame_bytes) : 0;
  u64 helper = 0;
  if constexpr (requires(Enc& e) { e.set_pipeline(true); }) helper = Enc::kPipelineBytes;
  unsigned t = std::max(1u, o.threads);
  if (frames && frames < t) t = static_cast<unsigned>(frames);
  for (;; --t) {
    pl.threads = t;
    pl.pipelined = helper && frames && 2 * u64{t} <= o.threads;
    pl.memory = (u64{t} + 2) * slot + u64{t} * (worker + (pl.pipelined ? helper : 0));
    if (pl.memory <= o.max_memory || t == 1) break;
  }
  return pl;
}

template <FrameEncoder Enc>
void encode_frame(Enc& enc, Slot& s, const hc::StreamParams& p, std::size_t block) {
  const auto fh = hc::encode_frame_header(hc::FrameHeader{p, s.index, s.out_offset});
  std::memcpy(s.out.get(), fh.data(), fh.size());
  std::size_t o = fh.size();
  const Bytes frame(s.in.get(), s.in_len);
  enc.reset_frame(frame);
  s.blocks = s.stored = 0;
  for (std::size_t b = 0; b < s.in_len; b += block) {
    const std::size_t n = std::min(s.in_len - b, block);
    std::byte* const hdr = s.out.get() + o;
    std::byte* const payload = hdr + hc::kBlockHeaderSize;
    std::size_t csize = enc.encode_block(b, b + n, MutBytes(payload, n));
    hc::BlockHeader h;
    if (csize == 0) {  // STORED fallback: the coded form would not be smaller
      std::memcpy(payload, frame.data() + b, n);
      csize = n;
      h.tag = static_cast<hc::u8>(hc::BlockType::stored);
      h.payload_crc = hc::crc32c(Bytes(payload, n));
      s.checks[s.blocks] = h.payload_crc;
      ++s.stored;
    } else {
      h.tag = static_cast<hc::u8>(hc::BlockType::cm);
      h.payload_crc = hc::crc32c(Bytes(payload, csize));
      h.content_crc = hc::crc32c(frame.subspan(b, n));
      s.checks[s.blocks] = h.content_crc;
    }
    h.usize = static_cast<u32>(n);
    h.csize = static_cast<u32>(csize);
    const auto hb = hc::encode_block_header(h);
    std::memcpy(hdr, hb.data(), hb.size());
    o += hb.size() + csize;
    ++s.blocks;
  }
  s.out_len = o;
}

template <FrameEncoder Enc>
class Pipeline {
 public:
  Pipeline(InputStream& in, OutputStream& out, const EncodeOptions& opt, const hc::StreamParams& params,
           const Plan& plan)
      : in_(in), out_(out), opt_(opt), params_(params), plan_(plan), slots_(plan.threads + 2) {}

  EncodeResult run() {
    EncodeResult r;
    r.threads = plan_.threads;
    r.memory_bound = plan_.memory;
    if (!allocate()) {
      r.status = Errc::out_of_memory;
      return r;
    }
    std::vector<Enc> encoders;
    encoders.reserve(plan_.threads);
    for (unsigned i = 0; i < plan_.threads; ++i) encoders.emplace_back(params_);
    for (Enc& e : encoders) attach_context(e, opt_.codec_context);  // before any worker starts
    if constexpr (requires(Enc& e) { e.set_pipeline(true); })
      for (Enc& e : encoders) e.set_pipeline(plan_.pipelined);

    const std::stop_callback on_stop(opt_.stop, [this]() noexcept { fail(Errc::cancelled); });
    {
      std::vector<std::jthread> threads;
      try {
        threads.reserve(plan_.threads + 1);
        threads.emplace_back([this] { write_loop(); });
        for (unsigned i = 0; i < plan_.threads; ++i)
          threads.emplace_back([this, &enc = encoders[i]] { work_loop(enc); });
      } catch (const std::system_error&) {
        fail(Errc::out_of_memory);  // could not start threads: unwind cleanly
      }
      read_loop();
    }  // joins every thread

    r.status = complete_ ? Errc::ok : error_;
    r.size_mismatch = !complete_ && size_mismatch_;
    r.bytes_in = bytes_in_;
    r.bytes_out = bytes_out_;
    r.frames = frames_;
    r.blocks = blocks_;
    r.stored_blocks = stored_;
    r.stream_id = params_.stream_id;
    return r;
  }

 private:
  bool allocate() noexcept {
    for (Slot& s : slots_) {
      // Uninitialized: untouched pages cost no RSS, so small inputs stay small.
      s.in.reset(new (std::nothrow) std::byte[plan_.frame_cap]);
      s.out.reset(new (std::nothrow) std::byte[plan_.out_cap]);
      s.checks.reset(new (std::nothrow) u32[plan_.max_blocks]);
      if (!s.in || !s.out || !s.checks) return false;
    }
    return true;
  }

  void fail(Errc e, bool size_mismatch = false) noexcept {
    {
      const std::lock_guard lk(m_);
      if (stop_) return;
      stop_ = true;
      error_ = e;
      size_mismatch_ = size_mismatch;
    }
    stopping_.store(true, std::memory_order_relaxed);
    work_cv_.notify_all();
    done_cv_.notify_all();
    in_.interrupt();  // wake a read or write blocked in the I/O layer
    out_.interrupt();
  }

  bool acquire_slot() {
    while (!free_.try_acquire_for(std::chrono::milliseconds(50)))
      if (stopping_.load(std::memory_order_relaxed)) return false;
    return !stopping_.load(std::memory_order_relaxed);
  }

  // Reads until `cap` bytes or end of input.
  std::size_t fill(std::byte* dst, std::size_t cap, bool& eof) {
    std::size_t n = 0;
    while (n < cap) {
      const std::size_t got = in_.read(MutBytes(dst + n, cap - n));
      if (got == 0) {
        eof = true;
        break;
      }
      n += got;
    }
    return n;
  }

  // Stream id: a function of the parameters and the first input bytes, so
  // identical input and options give byte-identical output.
  void publish_header(Bytes sample) {
    hc::StreamParams p = params_;
    p.stream_id = 0;
    const auto h = hc::encode_stream_header(p);
    p.stream_id = hc::crc32c(sample, hc::crc32c(Bytes(h).first(24)));
    {
      const std::lock_guard lk(m_);
      params_ = p;
      header_ready_ = true;
    }
    done_cv_.notify_all();
  }

  void read_loop() {
    u64 total = 0;
    for (u64 seq = 0;; ++seq) {
      if (!acquire_slot()) return;
      Slot& s = slots_[seq % slots_.size()];
      bool eof = false;
      const std::size_t n = fill(s.in.get(), plan_.frame_cap, eof);
      if (eof && in_.error() != Errc::ok) return fail(in_.error());
      total += n;
      if (opt_.content_size && total > *opt_.content_size) return fail(Errc::content_mismatch, true);
      if (eof && opt_.content_size && total != *opt_.content_size) return fail(Errc::content_mismatch, true);
      if (n != 0 && seq > 0xFFFFFFFFu) return fail(Errc::limit_output);  // frame index is a u32
      if (seq == 0) publish_header(Bytes(s.in.get(), std::min(n, kStreamIdSample)));
      s.in_len = n;
      s.out_offset = total - n;
      s.index = static_cast<u32>(seq);
      {
        const std::lock_guard lk(m_);
        if (n != 0) published_ = seq + 1;
        eof_ = eof;
        bytes_in_ = total;
      }
      if (n == 0) free_.release();  // nothing to encode in this slot
      if (eof) {
        work_cv_.notify_all();
        done_cv_.notify_all();
        return;
      }
      work_cv_.notify_one();
    }
  }

  void work_loop(Enc& enc) {
    for (;;) {
      u64 seq = 0;
      hc::StreamParams p;
      {
        std::unique_lock lk(m_);
        work_cv_.wait(lk, [this] { return stop_ || claimed_ < published_ || eof_; });
        if (stop_ || claimed_ >= published_) return;
        seq = claimed_++;
        p = params_;
      }
      Slot& s = slots_[seq % slots_.size()];
      encode_frame(enc, s, p, p.block_capacity());
      {
        const std::lock_guard lk(m_);
        s.done = true;
      }
      done_cv_.notify_all();
    }
  }

  bool emit(Bytes b) {
    if (!out_.write(b)) {
      fail(Errc::io_error);
      return false;
    }
    bytes_out_ += b.size();
    return true;
  }

  void write_loop() {
    hc::StreamParams p;
    {
      std::unique_lock lk(m_);
      done_cv_.wait(lk, [this] { return stop_ || header_ready_; });
      if (stop_) return;
      p = params_;
    }
    if (!emit(hc::encode_stream_header(p))) return;
    u32 list = 0;
    u64 total_out = 0;
    for (u64 seq = 0;; ++seq) {
      Slot& s = slots_[seq % slots_.size()];
      {
        std::unique_lock lk(m_);
        done_cv_.wait(lk, [&] { return stop_ || s.done || (eof_ && seq >= published_); });
        if (stop_) return;
        if (!s.done) break;  // end of input and every frame written
      }
      if (!emit(Bytes(s.out.get(), s.out_len))) return;
      for (std::size_t k = 0; k < s.blocks; ++k) list = hc::crc32c(hc::le32(s.checks[k]), list);
      total_out += s.in_len;
      ++frames_;
      blocks_ += s.blocks;
      stored_ += s.stored;
      {
        const std::lock_guard lk(m_);
        s.done = false;
      }
      free_.release();
    }
    if (!emit(hc::encode_end(hc::EndTrailer{total_out, static_cast<u32>(frames_), list, 0}))) return;
    const std::lock_guard lk(m_);
    complete_ = true;
  }

  InputStream& in_;
  OutputStream& out_;
  const EncodeOptions& opt_;
  hc::StreamParams params_;  // stream_id is filled in by publish_header()
  Plan plan_;
  std::vector<Slot> slots_;
  std::counting_semaphore<kMaxThreads + 2> free_{static_cast<std::ptrdiff_t>(plan_.threads + 2)};
  std::atomic<bool> stopping_{false};

  std::mutex m_;
  std::condition_variable work_cv_, done_cv_;
  // Guarded by m_.
  u64 published_ = 0, claimed_ = 0;
  bool eof_ = false, header_ready_ = false, stop_ = false, complete_ = false, size_mismatch_ = false;
  Errc error_ = Errc::ok;
  u64 bytes_in_ = 0;
  // Owned by the writer thread; read after it is joined.
  u64 bytes_out_ = 0, frames_ = 0, blocks_ = 0, stored_ = 0;
};

}  // namespace

const char* validate(const EncodeOptions& o) noexcept {
  if (o.level > kNeuralCMMaxLevel) return "level must be 0..4";
  if (o.window_log < hc::kMinWindowLog || o.window_log > hc::kMaxWindowLog)
    return "window must be a power of two from 64 KiB to 1 GiB";
  if (o.block_log < hc::kMinBlockLog || o.block_log > hc::kMaxBlockLog)
    return "block size must be a power of two from 1 KiB to 8 MiB";
  if (o.block_log > o.window_log) return "block size must not exceed the window";
  if (o.frame_bytes < (u64{1} << o.block_log) || o.frame_bytes > kMaxFrameBytes)
    return "frame size must be between the block size and 1 GiB";
  if (o.threads < 1 || o.threads > kMaxThreads) return "threads must be 1..256";
  if (o.content_size && *o.content_size == ~u64{0}) return "content size too large";
  return nullptr;
}

template <FrameEncoder Enc>
EncodeResult encode_parallel(InputStream& in, OutputStream& out, const EncodeOptions& opt) {
  EncodeResult r;
  if (validate(opt)) {
    r.status = Errc::unsupported_feature;
    return r;
  }
  hc::StreamParams p;
  p.level = opt.level;
  p.window_log = opt.window_log;
  p.block_log = opt.block_log;
  p.prior_id = opt.prior_id ? opt.prior_id : Enc::codec_id();
  p.content_size = opt.content_size;
  if constexpr (requires { Enc::accepts(p); }) {
    if (Enc::accepts(p) != Errc::ok) {
      r.status = Errc::unsupported_feature;
      return r;
    }
  }
  const Plan plan = make_plan<Enc>(opt, p);
  r.memory_bound = plan.memory;
  r.threads = plan.threads;
  if (plan.memory > opt.max_memory) {
    r.status = Errc::limit_memory;
    return r;
  }
  try {
    Pipeline<Enc> pipeline(in, out, opt, p, plan);
    return pipeline.run();
  } catch (const std::bad_alloc&) {
    r.status = Errc::out_of_memory;
  }
  return r;
}

template EncodeResult encode_parallel<NeuralCMCodec>(InputStream&, OutputStream&, const EncodeOptions&);
template EncodeResult encode_parallel<StructuralCMCodec>(InputStream&, OutputStream&, const EncodeOptions&);

}  // namespace hce
