// archive.cpp - the two-pass structural compressor over the aio layer (see the header).
#include "structural/archive.hpp"

#include "aio/async_streams.hpp"
#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"
#include "io/stream.hpp"
#include "structural/structural_cm_codec.hpp"
#include "structural/token_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace hce::structural {
namespace {

namespace fs = std::filesystem;
using hc::Bytes;
using hc::Errc;
using hc::MutBytes;
using hc::u32;
using hc::u64;
using hc::u8;

constexpr u64 kMaxSpoolRecord = u64{1} << 30;
constexpr u8 kStreamWindowLog = 16;  // the token codec reads at most 4 KiB back
constexpr u8 kStreamBlockLog = 16;   // 16384 tokens per block
constexpr std::size_t kOutStage = std::size_t{1} << 20;

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Bytes as_bytes(std::string_view s) noexcept { return {reinterpret_cast<const std::byte*>(s.data()), s.size()}; }

void store_u32(std::byte* p, u32 v) noexcept {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<std::byte>(v >> (8 * i));
}
void store_u64(std::byte* p, u64 v) noexcept {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<std::byte>(v >> (8 * i));
}
u32 load_u32(const std::byte* p) noexcept {
  u32 v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | std::to_integer<u32>(p[i]);
  return v;
}
u64 load_u64(const std::byte* p) noexcept {
  u64 v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | std::to_integer<u64>(p[i]);
  return v;
}

// Reads until `buf` is full or the input ends; returns the bytes read.
std::size_t read_exact(InputStream& in, MutBytes buf) {
  std::size_t got = 0;
  while (got < buf.size()) {
    const std::size_t n = in.read(buf.subspan(got));
    if (n == 0) break;
    got += n;
  }
  return got;
}

ArchiveStats failed(ArchiveStats st, const char* stage, Errc e, int sys_error = 0) {
  st.status = e;
  st.stage = stage;
  st.sys_error = sys_error;
  return st;
}

// One spool record: [u32 LE length][HST1 image]. Returns ok with an empty
// vector at the end of the spool.
Errc read_record(InputStream& spool, std::vector<std::byte>& image) {
  image.clear();
  std::byte len[4];
  const std::size_t n = read_exact(spool, MutBytes(len, 4));
  if (n == 0) return spool.error() != Errc::ok ? spool.error() : Errc::ok;
  if (n < 4) return spool.error() != Errc::ok ? spool.error() : Errc::truncated;
  const u32 size = load_u32(len);
  if (size == 0 || size > kMaxSpoolRecord) return Errc::header_corrupt;
  image.resize(size);
  if (read_exact(spool, MutBytes(image.data(), size)) != size)
    return spool.error() != Errc::ok ? spool.error() : Errc::truncated;
  return Errc::ok;
}

// Pass 2 input: the spool's tokens as u32 LE IDs of the frozen model.
class TokenIdInput final : public InputStream {
 public:
  TokenIdInput(InputStream& spool, const GlobalModel& model, const TokenStreamLimits& lim)
      : spool_(spool), model_(model), lim_(lim) {}

  std::size_t read(MutBytes buf) override {
    std::size_t done = 0;
    while (done < buf.size() && error_ == Errc::ok) {
      if (pos_ == ids_.size() && !refill()) break;
      const std::size_t n = std::min(buf.size() - done, ids_.size() - pos_);
      std::memcpy(buf.data() + done, ids_.data() + pos_, n);
      pos_ += n;
      done += n;
    }
    return done;
  }
  Errc error() const noexcept override { return error_; }
  void interrupt() noexcept override { spool_.interrupt(); }

 private:
  bool refill() {
    ids_.clear();
    pos_ = 0;
    while (ids_.empty()) {
      if (const Errc e = read_record(spool_, image_); e != Errc::ok) {
        error_ = e;
        return false;
      }
      if (image_.empty()) return false;  // end of the spool
      if (const Errc e = deserialize_tokens(image_, tokens_, lim_); e != Errc::ok) {
        error_ = e;
        return false;
      }
      ids_.resize(tokens_.size() * kTokenRecordBytes);
      for (std::size_t i = 0; i < tokens_.size(); ++i) {
        const u32 id = model_.find(tokens_[i].type, tokens_[i].value);
        if (id == kNoSymbol) {  // the spool changed since pass 1
          error_ = Errc::content_mismatch;
          return false;
        }
        store_u32(ids_.data() + kTokenRecordBytes * i, id);
      }
    }
    return true;
  }

  InputStream& spool_;
  const GlobalModel& model_;
  TokenStreamLimits lim_;
  std::vector<std::byte> image_, ids_;
  std::vector<StructuralToken> tokens_;
  std::size_t pos_ = 0;
  Errc error_ = Errc::ok;
};

// At most `left` bytes of another stream: the model section of an archive.
class LimitedInput final : public InputStream {
 public:
  LimitedInput(InputStream& in, u64 left) noexcept : in_(in), left_(left) {}
  std::size_t read(MutBytes buf) override {
    const std::size_t n = in_.read(buf.first(static_cast<std::size_t>(std::min<u64>(buf.size(), left_))));
    left_ -= n;
    return n;
  }
  Errc error() const noexcept override { return in_.error(); }
  void interrupt() noexcept override { in_.interrupt(); }
  u64 left() const noexcept { return left_; }

 private:
  InputStream& in_;
  u64 left_;
};

// Decoder output: ID records in, source bytes out (through a staging buffer).
class SourceOutput final : public OutputStream {
 public:
  SourceOutput(const GlobalModel& model, OutputStream& out, u64 limit) : model_(model), out_(out), limit_(limit) {
    stage_.reserve(kOutStage);
  }

  bool write(Bytes buf) override {
    for (const std::byte b : buf) {
      rec_[fill_++] = b;
      if (fill_ < kTokenRecordBytes) continue;
      fill_ = 0;
      const u32 id = load_u32(rec_);
      if (id >= model_.size()) {  // a STORED block can hold anything: check every record
        bad_ = true;
        return false;
      }
      const std::string_view v = model_.value(id);
      if (v.size() > limit_ - bytes_) {  // more source than the header promises
        bad_ = true;
        return false;
      }
      crc_ = hc::crc32c(as_bytes(v), crc_);
      stage_.insert(stage_.end(), reinterpret_cast<const std::byte*>(v.data()),
                    reinterpret_cast<const std::byte*>(v.data()) + v.size());
      ++tokens_;
      bytes_ += v.size();
      if (stage_.size() >= kOutStage && !flush()) return false;
    }
    return true;
  }
  void interrupt() noexcept override { out_.interrupt(); }
  bool flush() {
    if (stage_.empty()) return true;
    const bool ok = out_.write(stage_);
    stage_.clear();
    return ok;
  }

  bool bad() const noexcept { return bad_; }
  bool partial() const noexcept { return fill_ != 0; }
  u64 tokens() const noexcept { return tokens_; }
  u64 bytes() const noexcept { return bytes_; }
  u32 crc() const noexcept { return crc_; }

 private:
  const GlobalModel& model_;
  OutputStream& out_;
  u64 limit_;
  std::vector<std::byte> stage_;
  std::byte rec_[kTokenRecordBytes]{};
  std::size_t fill_ = 0;
  u64 tokens_ = 0, bytes_ = 0;
  u32 crc_ = 0;
  bool bad_ = false;
};

struct Header {
  u8 level = 0;
  u64 source_bytes = 0, tokens = 0, model_bytes = 0;
  u32 source_crc = 0, model_crc = 0;
};

std::array<std::byte, kArchiveHeaderSize> encode_header(const Header& h) noexcept {
  std::array<std::byte, kArchiveHeaderSize> b{};
  std::memcpy(b.data(), "HCS1", 4);
  b[4] = std::byte{kArchiveVersion};
  b[5] = std::byte{h.level};
  store_u64(&b[8], h.source_bytes);
  store_u64(&b[16], h.tokens);
  store_u64(&b[24], h.model_bytes);
  store_u32(&b[32], h.source_crc);
  store_u32(&b[36], h.model_crc);
  store_u32(&b[40], hc::crc32c(Bytes(b).first(40)));
  return b;
}

Errc decode_header(Bytes b, Header& h) noexcept {
  if (b.size() < kArchiveHeaderSize || std::memcmp(b.data(), "HCS1", 4) != 0) return Errc::bad_magic;
  if (load_u32(&b[40]) != hc::crc32c(b.first(40))) return Errc::header_corrupt;
  if (b[4] != std::byte{kArchiveVersion}) return Errc::unsupported_version;
  if (b[6] != std::byte{0} || b[7] != std::byte{0}) return Errc::header_corrupt;
  h.level = std::to_integer<u8>(b[5]);
  if (h.level > kStructuralCMMaxLevel) return Errc::unsupported_feature;
  h.source_bytes = load_u64(&b[8]);
  h.tokens = load_u64(&b[16]);
  h.model_bytes = load_u64(&b[24]);
  h.source_crc = load_u32(&b[32]);
  h.model_crc = load_u32(&b[36]);
  if (h.tokens > ~u64{0} / kTokenRecordBytes) return Errc::header_corrupt;
  return Errc::ok;
}

const char* check_options(const ArchiveOptions& o) noexcept {
  if (o.level > kStructuralCMMaxLevel) return "level must be 0..4";
  if (o.threads < 1 || o.threads > kMaxThreads) return "threads must be 1..256";
  if (o.frame_tokens < (u64{1} << (kStreamBlockLog - 2)) || o.frame_tokens > kMaxFrameBytes / kTokenRecordBytes)
    return "frame_tokens must be 16384..2^28";
  if (o.chunk_bytes < 4096 || o.chunk_bytes > (u64{1} << 28)) return "chunk_bytes must be 4 KiB..256 MiB";
  return nullptr;
}

}  // namespace

// ------------------------------------------------------------- stage 0 ---

ArchiveStats write_token_spool(const fs::path& source, const fs::path& spool, const ArchiveOptions& opt,
                               u32& source_crc) {
  ArchiveStats st;
  source_crc = 0;
  if (check_options(opt)) return failed(st, "open", Errc::unsupported_feature);
  aio::OpenError oe;
  auto in = aio::AsyncFileInput::open(source, opt.io, &oe);
  if (!in) return failed(st, "open", oe.status, oe.sys_error);
  auto out = aio::AsyncFileOutput::create(spool, opt.io, &oe);
  if (!out) return failed(st, "open", oe.status, oe.sys_error);
  st.backend = in->backend();
  try {
    StructuralPreprocessor pre;
    std::string pending;
    const auto chunk = static_cast<std::size_t>(opt.chunk_bytes);
    bool eof = false;
    while (!eof || !pending.empty()) {
      while (!eof && pending.size() < chunk) {  // fill up to one chunk
        const std::size_t old = pending.size();
        pending.resize(chunk);
        const std::size_t n = in->read(MutBytes(reinterpret_cast<std::byte*>(pending.data() + old), chunk - old));
        pending.resize(old + n);
        source_crc = hc::crc32c(as_bytes(std::string_view(pending).substr(old)), source_crc);
        st.source_bytes += n;
        if (n == 0) eof = true;
      }
      if (in->error() != Errc::ok) return failed(st, "spool", in->error(), in->sys_error());
      if (pending.empty()) break;
      // Cut after the last newline so that lines stay whole; the tokens are
      // lossless wherever the cut falls, it only shapes them.
      std::size_t cut = pending.size();
      if (!eof || pending.size() > chunk) {
        const std::size_t nl = pending.rfind('\n', std::min(pending.size(), chunk) - 1);
        cut = nl == std::string::npos ? std::min(pending.size(), chunk) : nl + 1;
      }
      const std::string piece = pending.substr(0, cut);
      pending.erase(0, cut);
      const std::vector<StructuralToken> tokens = pre.tokenize(piece);
      const std::vector<std::uint8_t> image = pre.serialize_tokens(tokens);
      std::byte len[4];
      store_u32(len, static_cast<u32>(image.size()));
      if (!out->write(Bytes(len, 4)) || !out->write(Bytes(reinterpret_cast<const std::byte*>(image.data()), image.size())))
        return failed(st, "spool", out->error(), out->sys_error());
      st.tokens += tokens.size();
      st.spool_bytes += 4 + image.size();
    }
  } catch (const std::bad_alloc&) {
    return failed(st, "spool", Errc::out_of_memory);
  } catch (const std::length_error&) {
    return failed(st, "spool", Errc::out_of_memory);
  }
  if (const Errc e = out->finish(false); e != Errc::ok) return failed(st, "spool", e, out->sys_error());
  return st;
}

// ------------------------------------------------------------- pass 1 ---

Errc build_global_model(const fs::path& spool, const ArchiveOptions& opt, GlobalModel& model, u64* tokens) {
  aio::OpenError oe;
  auto in = aio::AsyncFileInput::open(spool, opt.io, &oe);
  if (!in) return oe.status;
  GlobalModelBuilder builder(opt.model_limits);
  TokenStreamLimits lim;
  lim.max_token_bytes = opt.model_limits.max_token_bytes;
  std::vector<std::byte> image;
  std::vector<StructuralToken> toks;
  for (;;) {
    if (const Errc e = read_record(*in, image); e != Errc::ok) return e;
    if (image.empty()) break;
    if (const Errc e = deserialize_tokens(image, toks, lim); e != Errc::ok) return e;
    if (const Errc e = builder.add(toks); e != Errc::ok) return e;
  }
  if (tokens) *tokens = builder.tokens();
  return builder.freeze(model);
}

// ---------------------------------------------------------- compression ---

ArchiveStats compress_file(const fs::path& source, const fs::path& archive, const ArchiveOptions& opt) {
  ArchiveStats st;
  st.threads = opt.threads;
  if (check_options(opt)) return failed(st, "open", Errc::unsupported_feature);
  const fs::path spool = opt.spool.empty() ? fs::path(archive) += ".spool" : opt.spool;
  struct Remove {
    const fs::path& p;
    ~Remove() {
      std::error_code ec;
      fs::remove(p, ec);
    }
  } remove_spool{spool};

  // Stage 0: tokenize the source into the spool.
  auto t0 = std::chrono::steady_clock::now();
  u32 source_crc = 0;
  st = write_token_spool(source, spool, opt, source_crc);
  st.threads = opt.threads;
  st.seconds[0] = seconds_since(t0);
  if (st.status != Errc::ok) return st;

  // Pass 1: the frozen global model.
  t0 = std::chrono::steady_clock::now();
  GlobalModel model;
  u64 tokens = 0;
  if (const Errc e = build_global_model(spool, opt, model, &tokens); e != Errc::ok) return failed(st, "pass1", e);
  if (tokens != st.tokens) return failed(st, "pass1", Errc::content_mismatch);
  st.symbols = model.size();
  st.dictionary_bytes = model.dictionary_bytes();
  const std::vector<std::uint8_t> image = model.serialize();
  const Bytes image_bytes(reinterpret_cast<const std::byte*>(image.data()), image.size());

  // The model section: the HGM1 image through the byte-level codec.
  MemoryInput model_in(image_bytes);
  MemoryOutput model_out;
  EncodeOptions me;
  me.level = opt.level;
  me.threads = opt.threads;
  me.content_size = image.size();
  if (const EncodeResult r = encode_parallel<NeuralCMCodec>(model_in, model_out, me); r.status != Errc::ok)
    return failed(st, "model", r.status);
  st.model_bytes = model_out.data.size();
  st.seconds[1] = seconds_since(t0);

  // Pass 2: header and model section, then the token stream, frames in parallel.
  t0 = std::chrono::steady_clock::now();
  aio::OpenError oe;
  auto spool_in = aio::AsyncFileInput::open(spool, opt.io, &oe);
  if (!spool_in) return failed(st, "pass2", oe.status, oe.sys_error);
  auto out = aio::AsyncFileOutput::create(archive, opt.io, &oe);
  if (!out) return failed(st, "open", oe.status, oe.sys_error);
  Header h;
  h.level = opt.level;
  h.source_bytes = st.source_bytes;
  h.tokens = st.tokens;
  h.model_bytes = st.model_bytes;
  h.source_crc = source_crc;
  h.model_crc = hc::crc32c(image_bytes);
  const auto hb = encode_header(h);
  if (!out->write(hb) || !out->write(model_out.data)) return failed(st, "pass2", out->error(), out->sys_error());

  TokenStreamLimits lim;
  lim.max_token_bytes = opt.model_limits.max_token_bytes;
  TokenIdInput ids(*spool_in, model, lim);
  EncodeOptions so;
  so.level = opt.level;
  so.window_log = kStreamWindowLog;
  so.block_log = kStreamBlockLog;
  so.frame_bytes = opt.frame_tokens * kTokenRecordBytes;
  so.threads = opt.threads;
  so.content_size = st.tokens * kTokenRecordBytes;
  so.codec_context = &model;  // shared read-only by every worker
  const EncodeResult r = encode_parallel<StructuralCMCodec>(ids, *out, so);
  if (ids.error() != Errc::ok) return failed(st, "pass2", ids.error());
  if (r.status != Errc::ok) return failed(st, "pass2", r.status);
  st.stream_bytes = r.bytes_out;
  st.threads = r.threads;
  if (const Errc e = out->finish(true); e != Errc::ok) return failed(st, "finish", e, out->sys_error());
  st.archive_bytes = out->bytes_written();
  st.seconds[2] = seconds_since(t0);
  return st;
}

// -------------------------------------------------------- decompression ---

ArchiveStats decompress_file(const fs::path& archive, const fs::path& output, const ArchiveOptions& opt) {
  ArchiveStats st;
  st.threads = opt.threads;
  if (check_options(opt)) return failed(st, "open", Errc::unsupported_feature);
  aio::OpenError oe;
  auto in = aio::AsyncFileInput::open(archive, opt.io, &oe);
  if (!in) return failed(st, "open", oe.status, oe.sys_error);
  st.backend = in->backend();
  st.archive_bytes = in->size();

  auto t0 = std::chrono::steady_clock::now();
  std::array<std::byte, kArchiveHeaderSize> hb{};
  Header h;
  if (read_exact(*in, hb) != hb.size()) return failed(st, "header", in->error() != Errc::ok ? in->error() : Errc::truncated);
  if (const Errc e = decode_header(hb, h); e != Errc::ok) return failed(st, "header", e);
  if (h.model_bytes > in->size() - kArchiveHeaderSize) return failed(st, "header", Errc::truncated);

  // The model section, then the frozen model every worker shares.
  LimitedInput model_in(*in, h.model_bytes);
  MemoryOutput model_out;
  DecodeOptions md;
  md.threads = 1;  // a dictionary: small next to the token stream
  md.limits.max_memory_bytes = opt.max_memory;
  md.limits.max_output_bytes = opt.model_limits.max_dictionary_bytes + u64{opt.model_limits.max_symbols} * 21 + 16;
  if (const DecodeResult r = decode_stream<NeuralCMCodec>(model_in, model_out, md); r.status != Errc::ok)
    return failed(st, "model", r.status);
  if (model_in.left() != 0 || hc::crc32c(model_out.data) != h.model_crc) return failed(st, "model", Errc::payload_corrupt);
  GlobalModel model;
  if (const Errc e = GlobalModel::parse(model_out.data, opt.model_limits, model); e != Errc::ok)
    return failed(st, "model", e);
  st.symbols = model.size();
  st.dictionary_bytes = model.dictionary_bytes();
  st.model_bytes = h.model_bytes;
  st.seconds[0] = seconds_since(t0);

  t0 = std::chrono::steady_clock::now();
  std::unique_ptr<aio::AsyncFileOutput> out = aio::AsyncFileOutput::create(output, opt.io, &oe);
  if (!out) return failed(st, "open", oe.status, oe.sys_error);
  struct RemoveUnlessKept {  // closes the file first: Windows cannot remove an open one
    std::unique_ptr<aio::AsyncFileOutput>& out;
    const fs::path& p;
    bool keep = false;
    ~RemoveUnlessKept() {
      if (keep) return;
      out.reset();
      std::error_code ec;
      fs::remove(p, ec);
    }
  } guard{out, output};

  SourceOutput src(model, *out, h.source_bytes);
  DecodeOptions d;
  d.threads = opt.threads;
  d.limits.max_memory_bytes = opt.max_memory;
  d.limits.max_window_log = kStreamWindowLog;  // what the encoder writes: workers are sized for it
  d.limits.max_block_bytes = u32{1} << kStreamBlockLog;
  d.limits.max_output_bytes = h.tokens * kTokenRecordBytes;
  d.codec_context = &model;  // shared read-only by every worker
  const DecodeResult r = decode_stream<StructuralCMCodec>(*in, src, d);
  st.threads = r.threads;
  if (src.bad()) return failed(st, "decode", Errc::payload_corrupt);
  if (r.status != Errc::ok) return failed(st, "decode", r.status);
  if (!src.flush()) return failed(st, "decode", out->error(), out->sys_error());
  st.tokens = src.tokens();
  st.source_bytes = src.bytes();
  st.stream_bytes = r.bytes_in;
  if (src.partial() || src.tokens() != h.tokens || src.bytes() != h.source_bytes || src.crc() != h.source_crc)
    return failed(st, "verify", Errc::content_mismatch);
  if (const Errc e = out->finish(true); e != Errc::ok) return failed(st, "finish", e, out->sys_error());
  guard.keep = true;
  st.seconds[1] = seconds_since(t0);
  return st;
}

}  // namespace hce::structural
