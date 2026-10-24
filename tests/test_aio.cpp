// Tests for the asynchronous file I/O layer (aio/): every backend available
// here, the engines' contract, the stream adapters, and the unchanged
// pipelines running on top of them.
#include "aio/async_streams.hpp"
#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace aio = hce::aio;
using hc::Bytes;
using hc::Errc;
using hc::MutBytes;
using hc::u64;
using Vec = std::vector<std::byte>;

static int g_failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

static fs::path g_dir;  // scratch directory, under the working directory (a disk, not tmpfs)

static Vec random_bytes(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Vec v(n);
  for (auto& b : v) b = static_cast<std::byte>(rng());
  return v;
}

static Vec log_text(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Vec v;
  char line[128];
  for (unsigned i = 0; v.size() < n; ++i) {
    const int k = std::snprintf(line, sizeof line, "2026-10-24T12:%02u:%02u host-%u GET /api/v1/items/%u status=%u\n",
                                i / 60 % 60, i % 60, static_cast<unsigned>(rng() % 8),
                                static_cast<unsigned>(rng() % 100000), i % 13 ? 200u : 503u);
    for (int j = 0; j < k; ++j) v.push_back(static_cast<std::byte>(line[j]));
  }
  v.resize(n);
  return v;
}

static void write_plain(const fs::path& p, const Vec& v) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
}

static Vec read_plain(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  Vec v(s.size());
  if (!s.empty()) std::memcpy(v.data(), s.data(), s.size());
  return v;
}

// Random request sizes, so reads and writes straddle chunk boundaries.
static Vec read_all(hce::InputStream& in, std::mt19937& rng) {
  Vec out, buf(150000);
  for (;;) {
    const std::size_t n = in.read(MutBytes(buf).first(1 + rng() % buf.size()));
    if (n == 0) return out;
    out.insert(out.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
  }
}

static bool write_all(hce::OutputStream& out, const Vec& v, std::mt19937& rng) {
  for (std::size_t at = 0; at < v.size();) {
    const std::size_t n = std::min<std::size_t>(v.size() - at, 1 + rng() % 150000);
    if (!out.write(Bytes(v).subspan(at, n))) return false;
    at += n;
  }
  return true;
}

static std::vector<aio::Backend> backends() {
  std::vector<aio::Backend> b{aio::Backend::threads};
  if (aio::backend_available(aio::Backend::io_uring)) b.push_back(aio::Backend::io_uring);
  return b;
}

// --------------------------------------------------------------- engines ---

// The proactor contract itself: depth bound, tags, short reads at the end,
// batched submission, completions in any order.
static void test_engine_contract(aio::Backend backend) {
  const fs::path p = g_dir / "contract.bin";
  const Vec data = random_bytes(10000, 1);
  write_plain(p, data);
  aio::Options o;
  o.backend = backend;
  o.queue_depth = 4;
  aio::OpenError err;
  const auto r = aio::open_reader(p, o, &err);
  CHECK(r && err.status == Errc::ok);
  if (!r) return;
  CHECK(std::string(r->backend()) == aio::to_string(backend));
  CHECK(r->depth() == 4 && r->size() == data.size());

  aio::BufferPool pool(5, 4096);
  r->register_buffers(pool.all());
  for (unsigned i = 0; i < 4; ++i) CHECK(r->queue_read(u64{i} * 4096, pool.buffer(i), 100 + i) == Errc::ok);
  CHECK(r->queue_read(0, pool.buffer(4), 999) == Errc::output_full);  // depth reached
  CHECK(r->submit() == Errc::ok);                                      // one batch

  const std::size_t want[4] = {4096, 4096, 10000 - 8192, 0};  // the last two end at, and past, EOF
  bool seen[4] = {};
  aio::Completion done[8];
  for (std::size_t got = 0; got < 4;) {
    const std::size_t n = r->reap(done, 1);
    CHECK(n >= 1);
    if (n == 0) return;
    for (std::size_t k = 0; k < n; ++k) {
      const u64 i = done[k].tag - 100;
      CHECK(i < 4 && !seen[i] && done[k].sys_error == 0 && done[k].bytes == want[i]);
      if (i < 4) seen[i] = true;
    }
    got += n;
  }
  for (unsigned i = 0; i < 3; ++i)
    CHECK(std::equal(data.begin() + i * 4096, data.begin() + i * 4096 + static_cast<std::ptrdiff_t>(want[i]),
                     pool.data(i)));
  CHECK(r->reap(done, 1) == 0);  // nothing outstanding: returns at once instead of waiting forever

  // Writer: several writes in one batch, then a sync once they are done.
  const fs::path q = g_dir / "contract.out";
  const auto w = aio::open_writer(q, o, &err);
  CHECK(w && err.status == Errc::ok);
  if (!w) return;
  for (unsigned i = 0; i < 3; ++i)
    CHECK(w->queue_write(u64{i} * 4096, Bytes(data).subspan(i * 4096, want[i]), i) == Errc::ok);
  CHECK(w->submit() == Errc::ok);
  for (std::size_t got = 0; got < 3;) got += w->reap(done, 3 - got);
  CHECK(w->queue_sync(7) == Errc::ok && w->submit() == Errc::ok);
  CHECK(w->reap(done, 1) == 1 && done[0].tag == 7 && done[0].sys_error == 0);
  CHECK(read_plain(q) == data);
}

// ------------------------------------------------------------- adapters ---

static void test_stream_roundtrips(aio::Backend backend) {
  std::mt19937 rng(2);
  unsigned direct_honored = 0;
  for (const std::size_t size : {std::size_t{0}, std::size_t{1}, std::size_t{4095}, std::size_t{65536},
                                 std::size_t{65537}, std::size_t{3'400'000}}) {
    for (const unsigned depth : {1u, 4u, 32u}) {
      aio::Options o;
      o.backend = backend;
      o.queue_depth = depth;
      o.chunk_bytes = 64 * 1024;
      const Vec data = random_bytes(size, static_cast<std::uint32_t>(size + depth));
      const fs::path p = g_dir / "stream.bin";
      aio::OpenError err;
      auto out = aio::AsyncFileOutput::create(p, o, &err);
      CHECK(out && err.status == Errc::ok);
      if (!out) continue;
      CHECK(std::string(out->backend()) == aio::to_string(backend));
      CHECK(write_all(*out, data, rng));
      CHECK(out->finish(true) == Errc::ok && out->bytes_written() == size);
      CHECK(!out->write(Bytes(data).first(std::min<std::size_t>(size, 1))) || size == 0);  // closed after finish
      out.reset();
      CHECK(read_plain(p) == data);

      for (const bool direct : {false, true}) {
        o.direct = direct;
        auto in = aio::AsyncFileInput::open(p, o, &err);
        CHECK(in && in->size() == size);
        if (!in) continue;
        direct_honored += in->direct();
        CHECK(read_all(*in, rng) == data);
        CHECK(in->error() == Errc::ok && in->read(MutBytes(Vec(16).data(), 16)) == 0);
      }
    }
  }
  std::printf("  %s: stream roundtrips ok (O_DIRECT honored in %u of 18 direct opens)\n",
              aio::to_string(backend), direct_honored);
}

static void test_interrupt_and_errors(aio::Backend backend) {
  const fs::path p = g_dir / "interrupt.bin";
  write_plain(p, random_bytes(1 << 20, 3));
  aio::Options o;
  o.backend = backend;
  auto in = aio::AsyncFileInput::open(p, o);
  CHECK(in != nullptr);
  if (in) {
    in->interrupt();
    Vec buf(1000);
    CHECK(in->read(buf) == 0 && in->error() == Errc::cancelled);
  }
  auto out = aio::AsyncFileOutput::create(g_dir / "interrupt.out", o);
  CHECK(out != nullptr);
  if (out) {
    out->interrupt();
    CHECK(!out->write(Bytes(random_bytes(10, 4))) && out->finish() == Errc::cancelled);
  }

  aio::OpenError err;
  CHECK(!aio::AsyncFileInput::open(g_dir / "missing.bin", o, &err) && err.status == Errc::io_error && err.sys_error != 0);
  CHECK(!aio::AsyncFileInput::open(g_dir, o, &err) && err.status != Errc::ok);  // a directory
  CHECK(!aio::AsyncFileOutput::create(g_dir / "no" / "such" / "dir.out", o, &err) && err.status == Errc::io_error);
#if !defined(_WIN32)
  // Devices and pipes stay on the blocking streams.
  CHECK(!aio::open_reader("/dev/zero", o, &err) && err.status == Errc::unsupported_feature);
#endif
}

// --------------------------------------------------------------- engine ---

// The codec, container and pipelines are untouched: through the async layer
// they write the same stream as from memory, and decode it back.
static void test_pipelines_on_async_files(aio::Backend backend) {
  const Vec data = log_text(1'500'000, 5);
  const fs::path src = g_dir / "app.log", packed = g_dir / "app.log.hce", back = g_dir / "app.out";
  write_plain(src, data);

  hce::EncodeOptions eo;
  eo.level = 0;
  eo.threads = 2;
  eo.frame_bytes = 256 * 1024;
  hce::MemoryInput mem_in(data);
  hce::MemoryOutput mem_out;
  CHECK(hce::encode_parallel<hce::NeuralCMCodec>(mem_in, mem_out, eo).status == Errc::ok);

  aio::Options o;
  o.backend = backend;
  o.chunk_bytes = 128 * 1024;
  {
    auto in = aio::AsyncFileInput::open(src, o);
    auto out = aio::AsyncFileOutput::create(packed, o);
    CHECK(in && out);
    if (!in || !out) return;
    CHECK(hce::encode_parallel<hce::NeuralCMCodec>(*in, *out, eo).status == Errc::ok);
    CHECK(out->finish() == Errc::ok);
  }
  CHECK(read_plain(packed) == mem_out.data);  // byte-identical stream

  hce::DecodeOptions d;
  d.threads = 3;
  d.limits.max_window_log = 23;
  {
    auto in = aio::AsyncFileInput::open(packed, o);
    auto out = aio::AsyncFileOutput::create(back, o);
    CHECK(in && out);
    if (!in || !out) return;
    const hce::DecodeResult r = hce::decode_stream<hce::NeuralCMCodec>(*in, *out, d);
    CHECK(r.status == Errc::ok && r.stats.content_verified);
    CHECK(out->finish() == Errc::ok);
  }
  CHECK(read_plain(back) == data);
}

static void test_selection() {
  CHECK(aio::backend_available(aio::Backend::threads));
  const aio::Backend automatic = aio::resolve(aio::Backend::automatic);
  CHECK(automatic == (aio::backend_available(aio::Backend::io_uring) ? aio::Backend::io_uring : aio::Backend::threads));
#if defined(_WIN32)
  _putenv_s("HCE_AIO", "threads");
#else
  setenv("HCE_AIO", "threads", 1);
#endif
  CHECK(aio::resolve(aio::Backend::automatic) == aio::Backend::threads);
#if defined(_WIN32)
  _putenv_s("HCE_AIO", "");
#else
  unsetenv("HCE_AIO");
#endif
  if (!aio::backend_available(aio::Backend::io_uring)) {  // an explicit request does not fall back
    const fs::path p = g_dir / "select.bin";
    write_plain(p, random_bytes(10, 6));
    aio::Options o;
    o.backend = aio::Backend::io_uring;
    aio::OpenError err;
    CHECK(!aio::open_reader(p, o, &err) && err.status == Errc::unsupported_feature);
  }
  std::printf("  automatic backend here: %s\n", aio::to_string(automatic));
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::random_device rd;
  g_dir = fs::current_path() / ("aio-test-" + std::to_string(rd()));
  fs::create_directories(g_dir);

  test_selection();
  for (const aio::Backend b : backends()) {
    test_engine_contract(b);
    test_stream_roundtrips(b);
    test_interrupt_and_errors(b);
    test_pipelines_on_async_files(b);
    std::printf("  %s: engine contract, adapters, errors and pipelines ok\n", aio::to_string(b));
  }

  if (g_failures) {
    std::printf("%d check(s) failed (files kept in %s)\n", g_failures, g_dir.string().c_str());
    return 1;
  }
  fs::remove_all(g_dir);
  std::printf("all aio tests passed\n");
  return 0;
}
