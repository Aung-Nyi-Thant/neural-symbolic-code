// Tests for the parallel encoder and decoder pipeline. The central oracle:
// for any input, damaged or not, the pipeline must match the sequential
// decoder exactly (status, output bytes, gap reports, statistics).
#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"
#include "tests/legacy_streams.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace hc;
using Vec = std::vector<std::byte>;

static int g_failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

// ---------------------------------------------------------------- helpers ---

static Vec log_data(std::size_t lines, std::mt19937& rng, std::size_t noise_every = 3000) {
  Vec v;
  char buf[160];
  for (std::size_t i = 0; i < lines; ++i) {
    const int n = std::snprintf(buf, sizeof buf, "ts=%zu host-%u GET /api/v1/items/%u status=%u\n", i,
                                static_cast<unsigned>(rng() % 8), static_cast<unsigned>(rng() % 100000),
                                i % 13 ? 200u : 503u);
    for (int k = 0; k < n; ++k) v.push_back(static_cast<std::byte>(buf[k]));
    if (noise_every && i % noise_every == 0)
      for (int k = 0; k < 9000; ++k) v.push_back(static_cast<std::byte>(rng()));
  }
  return v;
}

using Codec = hce::NeuralCMCodec;

// Level 0 (21 MiB of model per worker, fast path from 32-byte matches) keeps
// the CM work of these tests small; test_codec covers every level.
static hce::EncodeOptions small_frames(unsigned threads) {
  hce::EncodeOptions o;
  o.threads = threads;
  o.level = 0;
  o.window_log = 16;
  o.block_log = 13;
  o.frame_bytes = 128 * 1024;
  return o;
}

static Vec encode(const Vec& data, hce::EncodeOptions o, hce::EncodeResult* out = nullptr) {
  hce::MemoryInput in(data, 77777);
  hce::MemoryOutput sink;
  const hce::EncodeResult r = hce::encode_parallel<Codec>(in, sink, o);
  CHECK(r.status == Errc::ok);
  if (out) *out = r;
  return sink.data;
}

struct Gaps {
  std::vector<GapInfo> v;
  static void cb(void* ctx, const GapInfo& g) noexcept { static_cast<Gaps*>(ctx)->v.push_back(g); }
};

struct Decoded {
  hce::DecodeResult r;
  Vec out;
  Gaps gaps;
};

static Decoded decode(const Vec& stream, hce::DecodeOptions o, unsigned threads, std::size_t max_read = 65536) {
  Decoded d;
  o.threads = threads;
  o.on_gap = Gaps::cb;
  o.gap_ctx = &d.gaps;
  hce::MemoryInput in(stream, max_read);
  hce::MemoryOutput sink;
  d.r = hce::decode_stream<Codec>(in, sink, o);
  d.out = std::move(sink.data);
  return d;
}

static bool same(const GapInfo& a, const GapInfo& b) {
  return a.stream_index == b.stream_index && a.out_offset == b.out_offset && a.length == b.length &&
         a.length_known == b.length_known && a.in_offset == b.in_offset && a.cause == b.cause;
}

static bool same(const DecodeStats& a, const DecodeStats& b) {
  return a.frames == b.frames && a.blocks_ok == b.blocks_ok && a.blocks_salvaged == b.blocks_salvaged &&
         a.blocks_lost == b.blocks_lost && a.damage_events == b.damage_events && a.gaps == b.gaps &&
         a.bytes_lost == b.bytes_lost && a.bytes_filled == b.bytes_filled && a.content_verified == b.content_verified;
}

// Pipeline vs sequential: everything observable must be identical.
static bool equivalent(const Decoded& seq, const Decoded& par) {
  if (seq.r.status != par.r.status || seq.out != par.out || !same(seq.r.stats, par.r.stats)) return false;
  if (seq.gaps.v.size() != par.gaps.v.size()) return false;
  for (std::size_t i = 0; i < seq.gaps.v.size(); ++i)
    if (!same(seq.gaps.v[i], par.gaps.v[i])) return false;
  return true;
}

static unsigned env_or(const char* name, unsigned fallback) {
  const char* v = std::getenv(name);
  return v ? static_cast<unsigned>(std::strtoul(v, nullptr, 10)) : fallback;
}

// ------------------------------------------------------------------ tests ---

static void test_crc_slicing() {
  std::mt19937 rng(1);
  Vec data(4096);
  for (auto& b : data) b = static_cast<std::byte>(rng());
  for (int i = 0; i < 2000; ++i) {
    const std::size_t off = rng() % 64, len = rng() % (data.size() - off);
    const Bytes s = Bytes(data).subspan(off, len);
    u32 ref = ~u32{0};
    for (const std::byte b : s) {  // bitwise reference
      ref ^= std::to_integer<u32>(b);
      for (int k = 0; k < 8; ++k) ref = (ref >> 1) ^ (0x82F63B78u & (0u - (ref & 1u)));
    }
    CHECK(crc32c(s) == ~ref);
    const std::size_t cut = len ? rng() % len : 0;  // chaining
    CHECK(crc32c(s.subspan(cut), crc32c(s.first(cut))) == ~ref);
  }
}

static void test_encoder_determinism_and_roundtrip() {
  std::mt19937 rng(2);
  const Vec data = log_data(9000, rng);
  hce::EncodeResult r1{};
  const Vec ref = encode(data, small_frames(1), &r1);
  CHECK(r1.frames > 3 && r1.blocks > r1.frames && r1.bytes_in == data.size() && r1.bytes_out == ref.size());
  CHECK(ref.size() * 4 < data.size());
  for (const unsigned t : {2u, 3u, 5u, 8u}) CHECK(encode(data, small_frames(t)) == ref);

  // Workers are sized for the largest stream the limits accept (level 4:
  // 149 MiB of model); a 1 MiB window cap leaves room for four of them.
  hce::DecodeOptions o;
  o.limits.max_window_log = 20;
  const Decoded seq = decode(ref, o, 1);
  CHECK(seq.r.status == Errc::ok && seq.out == data && seq.r.stats.content_verified && seq.r.threads == 0);
  for (const unsigned t : {2u, 3u, 4u}) {
    const Decoded par = decode(ref, o, t);
    CHECK(par.r.threads == t);
    CHECK(equivalent(seq, par));
  }
  // Frames split over many chunks: one worker must decode them in order.
  o.limits.max_block_bytes = 8192;
  o.chunk_bytes = 8192;
  const Decoded chunked = decode(ref, o, 3);
  CHECK(chunked.r.threads == 3 && equivalent(seq, chunked));
}

static void test_encoder_edges() {
  std::mt19937 rng(3);
  // Empty input: a header and an END trailer.
  const Vec empty = encode({}, small_frames(4));
  CHECK(empty.size() == kStreamHeaderSize + kEndSize);
  const Decoded d = decode(empty, {}, 3);
  CHECK(d.r.status == Errc::ok && d.out.empty() && d.r.stats.content_verified);

  // Exactly one and exactly two frames.
  for (const std::size_t n : {std::size_t{128 * 1024}, std::size_t{256 * 1024}}) {
    Vec data(n);
    for (auto& b : data) b = static_cast<std::byte>('a' + rng() % 3);
    const Decoded r = decode(encode(data, small_frames(2)), {}, 2);
    CHECK(r.r.status == Errc::ok && r.out == data);
  }

  // A declared size that the input does not match: failure, and no END.
  const Vec data = log_data(3000, rng, 0);
  hce::EncodeOptions o = small_frames(3);
  for (const u64 declared : {u64{data.size() - 1}, u64{data.size() + 1}}) {
    o.content_size = declared;
    hce::MemoryInput in(data);
    hce::MemoryOutput sink;
    const hce::EncodeResult r = hce::encode_parallel<Codec>(in, sink, o);
    CHECK(r.status == Errc::content_mismatch && r.size_mismatch);
    CHECK(decode(sink.data, {}, 1).r.status == Errc::truncated);
  }
  o.content_size = data.size();
  CHECK(decode(encode(data, o), {}, 2).out == data);

  // Memory cap: workers are shed to fit (each holds 21 MiB of level-0 model),
  // then the cap is reported.
  o = small_frames(8);
  o.max_memory = 50 * 1024 * 1024;
  hce::EncodeResult shed{};
  CHECK(encode(data, o, &shed) == encode(data, small_frames(1)));
  CHECK(shed.threads < 8 && shed.memory_bound <= o.max_memory);
  o.max_memory = 1024;
  hce::MemoryInput in(data);
  hce::MemoryOutput sink;
  CHECK(hce::encode_parallel<Codec>(in, sink, o).status == Errc::limit_memory);
}

static void test_concatenated_and_unsupported() {
  std::mt19937 rng(4);
  const Vec a = log_data(2500, rng), b = log_data(1500, rng);
  hce::EncodeOptions ob = small_frames(2);
  ob.window_log = 17;
  ob.block_log = 14;
  Vec both = encode(a, small_frames(2));
  const Vec sb = encode(b, ob);
  both.insert(both.end(), sb.begin(), sb.end());
  const Decoded seq = decode(both, {}, 1);
  Vec want = a;
  want.insert(want.end(), b.begin(), b.end());
  CHECK(seq.r.status == Errc::ok && seq.out == want);
  CHECK(equivalent(seq, decode(both, {}, 3)));

  // Another codec's stream, an unknown profile of this one, or a level this
  // build does not know, even with no frames: unsupported on both paths.
  hce::DecodeOptions any_level;
  any_level.limits.max_level = 255;
  for (const bool with_frames : {false, true}) {
    for (const u32 prior : {0x12345678u, hce_test::kLegacyLzCodecId, hce::kNeuralCMFamily | 2u}) {
      Vec s = encode(with_frames ? a : Vec{}, small_frames(1));
      store_le<u32>(&s[12], prior);  // prior_id
      store_le<u32>(&s[28], crc32c(Bytes(s).first(28)));
      CHECK(decode(s, {}, 1).r.status == Errc::unsupported_feature);
      CHECK(decode(s, {}, 3).r.status == Errc::unsupported_feature);
    }
    Vec s = encode(with_frames ? a : Vec{}, small_frames(1));
    s[7] = std::byte{hce::kNeuralCMMaxLevel + 1};  // level
    store_le<u32>(&s[28], crc32c(Bytes(s).first(28)));
    CHECK(decode(s, {}, 1).r.status == Errc::limit_level);  // above the default limit: R19 first
    CHECK(decode(s, any_level, 1).r.status == Errc::unsupported_feature);
    CHECK(decode(s, any_level, 3).r.status == Errc::unsupported_feature);
  }

  // A real stream of the retired stand-in LZ codec (hce 1.0.0): rejected as
  // unsupported in every mode, never misdecoded, nothing emitted.
  const Vec legacy(reinterpret_cast<const std::byte*>(hce_test::kLegacyLzStream),
                   reinterpret_cast<const std::byte*>(hce_test::kLegacyLzStream) + sizeof hce_test::kLegacyLzStream);
  CHECK(load_le<u32>(&legacy[12]) == hce_test::kLegacyLzCodecId);
  for (const hc::Recovery mode : {Recovery::strict, Recovery::skip, Recovery::skip_and_fill}) {
    hce::DecodeOptions o;
    o.recovery = mode;
    for (const unsigned t : {1u, 3u}) {
      const Decoded d = decode(legacy, o, t);
      CHECK(d.r.status == Errc::unsupported_feature && d.out.empty());
    }
  }
}

static void test_limits_match_sequential() {
  const Vec zeros(u64{6} << 20, std::byte{'A'});
  hce::EncodeOptions eo;
  eo.threads = 2;
  eo.level = 0;
  eo.frame_bytes = u64{1} << 20;
  const Vec bomb = encode(zeros, eo);
  CHECK(bomb.size() < 2000);
  hce::DecodeOptions o;
  o.limits.max_ratio = 100;
  o.limits.ratio_burst_bytes = u64{2} << 20;
  for (const hc::Recovery mode : {Recovery::strict, Recovery::skip_and_fill}) {
    o.recovery = mode;
    const Decoded seq = decode(bomb, o, 1);
    CHECK(seq.r.status == Errc::limit_ratio && seq.out.size() <= (u64{2} << 20));
    CHECK(equivalent(seq, decode(bomb, o, 3)));
  }
  o = {};
  o.limits.max_output_bytes = u64{3} << 20;
  const Decoded seq = decode(bomb, o, 1);
  CHECK(seq.r.status == Errc::limit_output);
  CHECK(equivalent(seq, decode(bomb, o, 2)));

  // Too little memory for two workers (each sized for level 4): the
  // sequential decoder takes over, binding only what this stream needs
  // (8 MiB window + 21 MiB level-0 model).
  o = {};
  o.limits.max_memory_bytes = 48 * 1024 * 1024;
  o.limits.max_window_log = 23;
  const Decoded small = decode(bomb, o, 4);
  CHECK(small.r.threads == 0 && small.r.status == Errc::ok && small.out == zeros);
}

static void test_differential_fuzz() {
  std::mt19937 rng(5);
  const Vec data = log_data(3000, rng);
  const Vec stream = encode(data, small_frames(3));
  const unsigned iters = env_or("HCE_FUZZ_ITERS", 60);
  unsigned compared = 0, recovered = 0;
  for (unsigned iter = 0; iter < iters; ++iter) {
    Vec bad = stream;
    const int edits = 1 + static_cast<int>(rng() % 3);
    for (int e = 0; e < edits; ++e) {
      const std::size_t at = rng() % bad.size();
      switch (rng() % 4) {
        case 0: bad[at] ^= static_cast<std::byte>(1u << (rng() % 8)); break;
        case 1: bad.insert(bad.begin() + static_cast<std::ptrdiff_t>(at), static_cast<std::byte>(rng())); break;
        case 2: bad.erase(bad.begin() + static_cast<std::ptrdiff_t>(at)); break;
        default: bad.resize(std::max<std::size_t>(at, bad.size() / 2)); break;
      }
    }
    const hc::Recovery mode = static_cast<hc::Recovery>(iter % 3);
    hce::DecodeOptions o;
    o.recovery = mode;
    o.fill = std::byte{0xEE};
    o.limits.max_block_bytes = 8192;
    o.chunk_bytes = iter % 2 ? 8192 : u64{1} << 20;
    const Decoded seq = decode(bad, o, 1, 1 + rng() % 70000);
    const Decoded par = decode(bad, o, 2 + iter % 3, 1 + rng() % 70000);
    ++compared;
    const bool ok = equivalent(seq, par);
    CHECK(ok);
    if (!ok) {
      std::printf("  iteration %u diverged: %s/%s, %zu/%zu bytes, %zu/%zu gaps\n", iter, to_string(seq.r.status),
                  to_string(par.r.status), seq.out.size(), par.out.size(), seq.gaps.v.size(), par.gaps.v.size());
      return;
    }
    recovered += mode != Recovery::strict && seq.r.stats.gaps > 0;
  }
  std::printf("differential fuzz: %u damaged streams, pipeline == sequential in all (%u recovered with gaps)\n",
              compared, recovered);
}

// Regression: the pipeline fails (strict mode, damage early in a long frame)
// while the splitter is still cutting that frame into chunks. The splitter
// must never touch a chunk it already handed to a worker.
static void test_failure_while_cutting() {
  std::mt19937 rng(7);
  const Vec data = log_data(9000, rng, 0);
  hce::EncodeOptions eo = small_frames(2);
  eo.frame_bytes = u64{1} << 20;  // one long frame ...
  Vec bad = encode(data, eo);
  bad[kStreamHeaderSize + kFrameHeaderSize + kBlockHeaderSize + 10] ^= std::byte{0x01};  // first block of frame 0
  hce::DecodeOptions o;
  o.limits.max_block_bytes = 8192;
  o.chunk_bytes = 8192;  // ... cut into one-block chunks
  const Decoded seq = decode(bad, o, 1);
  CHECK(seq.r.status == Errc::payload_corrupt && seq.out.empty());
  for (unsigned i = 0; i < 60; ++i) CHECK(equivalent(seq, decode(bad, o, 2 + i % 3, 4096)));
}

// An input that blocks until cancelled, like an idle pipe.
class StallingInput final : public hce::InputStream {
 public:
  StallingInput(Bytes prefix, std::stop_token stop) : prefix_(prefix), stop_(std::move(stop)) {}
  std::size_t read(MutBytes buf) override {
    if (pos_ < prefix_.size()) {
      const std::size_t n = std::min(buf.size(), prefix_.size() - pos_);
      std::memcpy(buf.data(), prefix_.data() + pos_, n);
      pos_ += n;
      return n;
    }
    while (!stop_.stop_requested() && !interrupted_) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return 0;
  }
  Errc error() const noexcept override { return stop_.stop_requested() || interrupted_ ? Errc::cancelled : Errc::ok; }
  void interrupt() noexcept override { interrupted_ = true; }

 private:
  Bytes prefix_;
  std::size_t pos_ = 0;
  std::stop_token stop_;
  std::atomic<bool> interrupted_{false};
};

// An output that fails after `budget` bytes, like a closed pipe.
class FailingOutput final : public hce::OutputStream {
 public:
  explicit FailingOutput(std::size_t budget) : budget_(budget) {}
  bool write(Bytes b) override {
    if (b.size() > budget_) return false;
    budget_ -= b.size();
    return true;
  }

 private:
  std::size_t budget_;
};

static void test_cancellation_and_io_failures() {
  std::mt19937 rng(6);
  const Vec data = log_data(8000, rng);
  const Vec stream = encode(data, small_frames(2));
  using clock = std::chrono::steady_clock;

  for (const unsigned threads : {1u, 3u}) {
    // Decoder: input stalls mid-stream, then a stop request arrives.
    hce::DecodeOptions o;
    o.threads = threads;
    std::stop_source stop;
    o.stop = stop.get_token();
    StallingInput in(Bytes(stream).first(stream.size() / 2), stop.get_token());
    hce::MemoryOutput sink;
    std::jthread canceller([stop]() mutable {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      stop.request_stop();
    });
    const auto t0 = clock::now();
    const hce::DecodeResult r = hce::decode_stream<Codec>(in, sink, o);
    CHECK(r.status == Errc::cancelled);
    CHECK(clock::now() - t0 < std::chrono::seconds(5));

    // Encoder: same.
    hce::EncodeOptions eo = small_frames(threads);
    std::stop_source estop;
    eo.stop = estop.get_token();
    StallingInput ein(Bytes(data).first(data.size() / 3), estop.get_token());
    hce::MemoryOutput esink;
    std::jthread ecanceller([estop]() mutable {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      estop.request_stop();
    });
    CHECK(hce::encode_parallel<Codec>(ein, esink, eo).status == Errc::cancelled);

    // Output fails part-way: io_error, no deadlock, on both pipelines.
    FailingOutput fail_out(data.size() / 4);
    hce::MemoryInput sin(stream);
    hce::DecodeOptions fo;
    fo.threads = threads;
    CHECK(hce::decode_stream<Codec>(sin, fail_out, fo).status == Errc::io_error);
    FailingOutput fail_enc(stream.size() / 4);
    hce::MemoryInput din(data);
    CHECK(hce::encode_parallel<Codec>(din, fail_enc, small_frames(threads)).status == Errc::io_error);
  }
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  test_crc_slicing();
  test_encoder_determinism_and_roundtrip();
  test_encoder_edges();
  test_concatenated_and_unsupported();
  test_limits_match_sequential();
  test_differential_fuzz();
  test_failure_while_cutting();
  test_cancellation_and_io_failures();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all pipeline tests passed\n");
  return 0;
}
