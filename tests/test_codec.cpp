// Tests for the neural CM codec: exact kernels (R15), golden streams (R14/R17),
// roundtrips across levels and geometries, hostile payloads, the encoder's
// replay path (R12), frame independence (R20), window bounds (R21) and the
// cost of frame resets (R19).
#include "codec/ncm_kernels.hpp"
#include "codec/ncm_tables.hpp"
#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"
#include "tests/legacy_streams.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace hc;
using Vec = std::vector<std::byte>;
using Codec = hce::NeuralCMCodec;

static int g_failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

// ------------------------------------------------------------------ data ---
// Generators use raw mt19937 output only (its sequence is fixed by the
// standard; distributions are not), so inputs are identical on every platform.

static void put(Vec& v, const char* s, int n) {
  for (int k = 0; k < n; ++k) v.push_back(static_cast<std::byte>(s[k]));
}

static Vec log_lines(std::size_t bytes, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Vec v;
  char buf[200];
  for (std::size_t i = 0; v.size() < bytes; ++i) {
    const auto a = static_cast<unsigned>(rng() % 8), b = static_cast<unsigned>(rng() % 100000),
               c = static_cast<unsigned>(rng() % 1000);
    const int n = std::snprintf(buf, sizeof buf, "2026-10-24T12:%02u:%02u.%03u host-%u GET /api/v1/items/%u status=%u\n",
                                static_cast<unsigned>(i / 60 % 60), static_cast<unsigned>(i % 60), c, a, b,
                                i % 13 ? 200u : 503u);
    put(v, buf, n);
  }
  v.resize(bytes);
  return v;
}

static Vec prose(std::size_t bytes, std::uint32_t seed) {
  static const char* const kWords[] = {
      "the",     "model",  "of",       "a",       "context", "mixing",  "codec",  "predicts", "every",  "bit",
      "and",     "learns", "weights",  "from",    "its",     "errors",  "while",  "frames",   "reset",  "all",
      "tables",  "at",     "sync",     "points",  "so",      "damage",  "stays",  "local",    "in",     "time",
      "History", "Window", "Recovery", "Encoder", "Decoder", "integer", "SIMD",   "lanes",    "exact",  "arena"};
  std::mt19937 rng(seed);
  Vec v;
  while (v.size() < bytes) {
    const char* w = kWords[rng() % (sizeof kWords / sizeof kWords[0])];
    put(v, w, static_cast<int>(std::strlen(w)));
    const auto r = static_cast<unsigned>(rng() % 20);
    put(v, r == 0 ? ".\n" : r == 1 ? ", " : " ", r == 0 ? 2 : r == 1 ? 2 : 1);
  }
  v.resize(bytes);
  return v;
}

// Structured binary: little-endian records, random runs and zeros.
static Vec binary(std::size_t bytes, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Vec v;
  for (std::uint32_t i = 0; v.size() < bytes; ++i) {
    switch (i / 64 % 3) {
      case 0:
        for (int k = 0; k < 4; ++k) v.push_back(static_cast<std::byte>((i * 2654435761u) >> (8 * k)));
        v.push_back(std::byte{0});
        v.push_back(static_cast<std::byte>(rng() % 4));
        break;
      case 1:
        for (int k = 0; k < 8; ++k) v.push_back(static_cast<std::byte>(rng()));
        break;
      default:
        v.insert(v.end(), 8, std::byte{0});
    }
  }
  v.resize(bytes);
  return v;
}

static Vec random_bytes(std::size_t bytes, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Vec v(bytes);
  for (auto& b : v) b = static_cast<std::byte>(rng());
  return v;
}

static Vec concat(std::initializer_list<Vec> parts) {
  Vec v;
  for (const Vec& p : parts) v.insert(v.end(), p.begin(), p.end());
  return v;
}

// ------------------------------------------------------------ pipelines ---

static hce::EncodeOptions opts(u8 level, u8 window_log = 16, u8 block_log = 13, u64 frame = 64 * 1024) {
  hce::EncodeOptions o;
  o.level = level;
  o.window_log = window_log;
  o.block_log = block_log;
  o.frame_bytes = frame;
  o.threads = 1;
  return o;
}

static Vec encode(const Vec& data, const hce::EncodeOptions& o, hce::EncodeResult* res = nullptr) {
  hce::MemoryInput in(data);
  hce::MemoryOutput out;
  const hce::EncodeResult r = hce::encode_parallel<Codec>(in, out, o);
  CHECK(r.status == Errc::ok);
  if (res) *res = r;
  return out.data;
}

struct Decoded {
  hce::DecodeResult r;
  Vec out;
};

static Decoded decode(const Vec& stream, hce::DecodeOptions o = {}, unsigned threads = 1) {
  o.threads = threads;
  hce::MemoryInput in(stream);
  hce::MemoryOutput out;
  Decoded d;
  d.r = hce::decode_stream<Codec>(in, out, o);
  d.out = std::move(out.data);
  return d;
}

static bool roundtrips(const Vec& data, const hce::EncodeOptions& o, Vec* stream_out = nullptr) {
  const Vec s = encode(data, o);
  const Decoded d = decode(s);
  if (stream_out) *stream_out = s;
  return d.r.status == Errc::ok && d.out == data && d.r.stats.content_verified;
}

// Items of a stream: where each block's header and payload are.
struct BlockAt {
  std::size_t header, payload;
  BlockHeader h;
};

static std::vector<BlockAt> blocks_of(const Vec& s) {
  std::vector<BlockAt> out;
  std::size_t at = kStreamHeaderSize;
  while (at < s.size()) {
    if (s[at] == kSyncMarker[0]) {
      at += kFrameHeaderSize;
    } else if (std::to_integer<u8>(s[at]) == kTagEnd) {
      break;
    } else {
      BlockHeader h;
      CHECK(parse_block_header(Bytes(s).subspan(at, kBlockHeaderSize), h) == Errc::ok);
      out.push_back({at, at + kBlockHeaderSize, h});
      at += kBlockHeaderSize + h.csize;
    }
  }
  return out;
}

// Rewrites a block header in place (CRC included) after its payload changed.
static void rewrite_header(Vec& s, const BlockAt& b, u32 csize) {
  BlockHeader h = b.h;
  h.csize = csize;
  h.payload_crc = crc32c(Bytes(s).subspan(b.payload, csize));
  const auto hb = encode_block_header(h);
  std::memcpy(&s[b.header], hb.data(), hb.size());
}

// ------------------------------------------------------------------ tests ---

// R15: every SIMD path computes exactly the scalar integers, lane by lane,
// including saturation and the extreme inputs the format allows.
static void test_kernel_parity() {
  using namespace hce::ncm;
  const Kernels& ref = kernels_for(SimdPath::scalar);
  std::mt19937 rng(11);
  unsigned paths = 0;
  for (const SimdPath p : kAllPaths) {
    if (p == SimdPath::scalar || !simd_available(p)) continue;
    ++paths;
    const Kernels& k = kernels_for(p);
    CHECK(k.path == p);
    for (int iter = 0; iter < 200000; ++iter) {
      alignas(32) std::int16_t x[kLanes];
      alignas(32) std::int16_t wa[kSets][kLanes], wb[kSets][kLanes];
      std::int32_t e[kSets];
      const bool edge = iter % 4 == 0;  // extremes: |x| = 2047, w at the int16 limits, |e| = 32767
      for (int i = 0; i < kLanes; ++i)
        x[i] = static_cast<std::int16_t>(edge ? (rng() & 1 ? 2047 : -2047) : static_cast<int>(rng() % 4095) - 2047);
      for (int j = 0; j < kSets; ++j) {
        for (int i = 0; i < kLanes; ++i) {
          const int w = edge ? (rng() & 1 ? 32767 : -32768) : static_cast<int>(rng() % 65536) - 32768;
          wa[j][i] = wb[j][i] = static_cast<std::int16_t>(w);
        }
        e[j] = edge ? (rng() & 1 ? 32767 : -32767) : static_cast<int>(rng() % 65535) - 32767;
      }
      std::int16_t* pa[kSets] = {wa[0], wa[1], wa[2]};
      std::int16_t* pb[kSets] = {wb[0], wb[1], wb[2]};
      std::int32_t oa[kSets], ob[kSets];
      ref.mix(x, pa, oa);
      k.mix(x, pb, ob);
      ref.train(x, pa, e);
      k.train(x, pb, e);
      // Layer 2: four weights over layer-1 outputs (|t| <= 2047), one error.
      alignas(8) std::int16_t t[4], fa[4], fb[4];
      for (int i = 0; i < 4; ++i) {
        t[i] = x[i];
        fa[i] = fb[i] = wa[0][i];
      }
      const std::int32_t e2 = e[1];
      const bool dot_same = ref.dot4(t, fa) == k.dot4(t, fb);
      ref.train4(t, fa, e2);
      k.train4(t, fb, e2);
      const bool same = std::equal(oa, oa + kSets, ob) && std::memcmp(wa, wb, sizeof wa) == 0 && dot_same &&
                        std::memcmp(fa, fb, sizeof fa) == 0;
      CHECK(same);
      if (!same) {
        std::printf("  %s diverges from scalar at iteration %d\n", k.name, iter);
        return;
      }
    }
  }
  std::printf("kernel parity: %u SIMD path(s) bit-exact with scalar (active: %s)\n", paths,
              active_kernels().name);
}

// The counter transition table is the update rule, for every state and bit.
static void test_node_table() {
  using namespace hce::ncm;
  unsigned bad = 0;
  for (int y = 0; y < 2; ++y)
    for (int v = 0; v < 65536; ++v)
      bad += kNodeNext[static_cast<std::size_t>(y << 16 | v)] != node_next(static_cast<std::uint16_t>(v), y);
  CHECK(bad == 0);
  // Spot values of the rule itself: from zero (p = 0.5, n = 0) one step toward each bit.
  CHECK(node_next(0, 1) == (((2048 + (((4095 - 2048) * 52429 + 32768) >> 16)) ^ 0x800) << 4 | 1));
  CHECK(node_next(0, 0) == (((2048 + (((0 - 2048) * 52429 + 32768) >> 16)) ^ 0x800) << 4 | 1));
  std::printf("node table: %u of 131072 transitions differ from the rule\n", bad);
}

// Accepts, memory_required and bind: the BlockCodec contract.
static void test_contract() {
  StreamParams p;
  p.prior_id = hce::kNeuralCMCodecId;
  for (u8 l = 0; l <= hce::kNeuralCMMaxLevel; ++l) {
    p.level = l;
    CHECK(Codec::accepts(p) == Errc::ok);
  }
  p.level = hce::kNeuralCMMaxLevel + 1;
  CHECK(Codec::accepts(p) == Errc::unsupported_feature);
  p.level = 2;
  for (const u32 id : {0u, hce_test::kLegacyLzCodecId, hce::kNeuralCMFamily, hce::kNeuralCMFamily | 2u,
                       hce::kNeuralCMCodecId ^ 0x01000000u}) {
    p.prior_id = id;
    CHECK(Codec::accepts(p) == Errc::unsupported_feature);
  }
  p.prior_id = hce::kNeuralCMCodecId;

  // Memory depends on the level only, grows with it, and is exact: binding
  // into one byte less fails before anything is written.
  u64 prev = 0;
  for (u8 l = 0; l <= hce::kNeuralCMMaxLevel; ++l) {
    p.level = l;
    p.window_log = 16;
    const u64 m = Codec::memory_required(p);
    p.window_log = 27;
    CHECK(Codec::memory_required(p) == m);
    CHECK(m > prev);
    prev = m;
  }
  p.level = 0;
  p.window_log = 16;
  const std::size_t need = static_cast<std::size_t>(Codec::memory_required(p));
  std::vector<std::byte> arena(need);
  Codec c;
  CHECK(c.bind(p, MutBytes(arena).first(need - 1)) == Errc::out_of_memory);
  StreamParams bad = p;
  bad.prior_id = hce_test::kLegacyLzCodecId;
  CHECK(c.bind(bad, MutBytes{}) == Errc::unsupported_feature);  // rejected before the arena is touched
  CHECK(c.bind(p, MutBytes(arena)) == Errc::ok);
  Window w;
  std::vector<std::byte> wmem(std::size_t{1} << 16);
  w.bind(wmem);
  // A payload that is not a CM payload of this codec: an error, exactly usize bytes or fewer.
  const std::byte junk[8] = {};
  c.reset_frame();
  const Errc e = c.decode_block(Bytes(junk), 100, w);
  CHECK(e == Errc::payload_corrupt || e == Errc::ok);
  CHECK(w.pos() <= 100);
}

// Roundtrips over every level, several data types, geometries and edge sizes.
static void test_roundtrip_matrix() {
  const Vec mixed = concat({log_lines(24000, 1), random_bytes(9000, 2), prose(20000, 3), binary(12000, 4)});
  for (u8 level = 0; level <= hce::kNeuralCMMaxLevel; ++level) CHECK(roundtrips(mixed, opts(level)));

  for (const Vec& d : {log_lines(60000, 5), prose(60000, 6), binary(60000, 7), random_bytes(30000, 8),
                       Vec(70000, std::byte{0}), Vec(70000, std::byte{'A'})}) {
    CHECK(roundtrips(d, opts(0)));
    CHECK(roundtrips(d, opts(2, 17, 15, 128 * 1024)));
  }

  // Sizes around the block and the CM minimum: 0, 1, 15, 16, 17 bytes, block +- 1.
  const Vec text = prose(20000, 9);
  for (const std::size_t n : {0u, 1u, 2u, 15u, 16u, 17u, 1023u, 1024u, 1025u, 8191u, 8192u, 8193u})
    CHECK(roundtrips(Vec(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(n)), opts(1, 16, 10)));

  // Largest blocks and windows, one frame.
  CHECK(roundtrips(log_lines(300000, 10), opts(3, 23, 18, u64{8} << 20)));

  // Compression is real: logs and prose shrink well below a quarter.
  for (const Vec& d : {log_lines(200000, 11), prose(200000, 12)}) {
    Vec s;
    CHECK(roundtrips(d, opts(2, 20, 16, u64{1} << 20), &s));
    CHECK(s.size() * 4 < d.size());
  }
}

// R15 end to end: every kernel path writes the same stream, and a stream
// written with one path decodes with any other.
static void test_simd_paths_agree() {
  const Vec data = concat({log_lines(40000, 21), prose(30000, 22)});
  Vec ref;
  unsigned paths = 0;
  for (const char* name : {"scalar", "sse2", "avx2", "avx512", "neon"}) {
    if (!Codec::test_select_simd(name)) continue;
    ++paths;
    const Vec s = encode(data, opts(1));
    if (ref.empty()) ref = s;
    CHECK(s == ref);
    for (const char* other : {"scalar", "sse2", "avx2", "avx512", "neon"}) {
      if (!Codec::test_select_simd(other)) continue;
      const Decoded d = decode(s);
      CHECK(d.r.status == Errc::ok && d.out == data);
    }
  }
  Codec::test_select_simd(nullptr);
  CHECK(paths >= 2);
  std::printf("simd paths: %u kernel set(s) write identical streams and cross-decode\n", paths);
}

// R14/R17: fixed inputs must give these exact streams on every platform and
// kernel path (checked on x86-64 scalar/SSE2/AVX2 and AArch64 NEON). A change
// here is a format change: it needs a new profile id, not new constants.
struct Golden {
  const char* name;
  u8 level;
  u64 size;
  u32 crc;
};

static Vec golden_input(const char* name) {
  if (std::strcmp(name, "log") == 0) return log_lines(48000, 31);
  if (std::strcmp(name, "prose") == 0) return prose(48000, 32);
  return binary(48000, 33);
}

static void test_golden_streams() {
  static const Golden kGolden[] = {
      {"log", 0, 3994, 0x7083CBB7},     {"log", 2, 3990, 0x76CC6613},     {"log", 4, 3991, 0xB9D30C08},
      {"prose", 0, 6881, 0x2C7D6B41},   {"prose", 2, 6880, 0x9EE996FF},   {"prose", 4, 6881, 0x61B9A566},
      {"binary", 0, 29072, 0xAD242307}, {"binary", 2, 29087, 0x894C7EE8}, {"binary", 4, 29075, 0xBAED65DA},
  };
  unsigned mismatches = 0;
  for (const Golden& g : kGolden) {
    const Vec data = golden_input(g.name);
    const Vec s = encode(data, opts(g.level, 16, 13, 32 * 1024));
    const u32 crc = crc32c(Bytes(s));
    if (s.size() != g.size || crc != g.crc) {
      ++mismatches;
      std::printf("  golden %s/L%u: got {\"%s\", %u, %zu, 0x%08X}\n", g.name, static_cast<unsigned>(g.level), g.name,
                  static_cast<unsigned>(g.level), s.size(), crc);
    }
    const Decoded d = decode(s);
    CHECK(d.r.status == Errc::ok && d.out == data);
  }
  CHECK(mismatches == 0);
  if (!mismatches)
    std::printf("golden streams: %zu/%zu identical (%s kernels)\n", std::size(kGolden), std::size(kGolden),
                Codec::simd_path());
}

// Hostile CM payloads with valid CRCs reach the model itself: every outcome
// is an error or the verified original bytes, never a crash or unverified output.
static void test_hostile_payloads() {
  const Vec data = concat({log_lines(12000, 41), prose(8000, 42)});
  const Vec stream = encode(data, opts(0, 16, 11));
  const std::vector<BlockAt> blocks = blocks_of(stream);
  std::size_t cm = 0;
  for (const BlockAt& b : blocks) cm += b.h.type() == BlockType::cm;
  CHECK(cm >= 4);

  std::mt19937 rng(43);
  unsigned rejected = 0, harmless = 0;
  for (int iter = 0; iter < 120; ++iter) {
    const BlockAt& b = blocks[rng() % blocks.size()];
    if (b.h.type() != BlockType::cm) continue;
    Vec s = stream;
    u32 csize = b.h.csize;
    switch (iter % 4) {
      case 0:  // flip bits inside the payload
        for (int k = 0; k < 1 + iter % 5; ++k) s[b.payload + rng() % csize] ^= static_cast<std::byte>(1u << (rng() % 8));
        break;
      case 1:  // random payload of the same size
        for (u32 k = 0; k < csize; ++k) s[b.payload + k] = static_cast<std::byte>(rng());
        break;
      case 2: {  // truncated payload
        const u32 cut = 1 + static_cast<u32>(rng() % std::max<u32>(1, csize - 1));
        s.erase(s.begin() + static_cast<std::ptrdiff_t>(b.payload + csize - cut),
                s.begin() + static_cast<std::ptrdiff_t>(b.payload + csize));
        csize -= cut;
        if (csize == 0) continue;
        break;
      }
      default: {  // extended payload (still smaller than the content)
        const u32 add = 1 + rng() % 16;
        if (csize + add >= b.h.usize) continue;
        s.insert(s.begin() + static_cast<std::ptrdiff_t>(b.payload + csize), add, static_cast<std::byte>(rng()));
        csize += add;
      }
    }
    rewrite_header(s, b, csize);
    for (const hc::Recovery mode : {Recovery::strict, Recovery::skip_and_fill}) {
      hce::DecodeOptions o;
      o.recovery = mode;
      o.fill = std::byte{0xEE};
      const Decoded d = decode(s, o);
      if (d.out == data && d.r.status == Errc::ok) {
        ++harmless;  // the change did not alter the decoded bytes (content CRC verified)
        continue;
      }
      if (mode == Recovery::strict) {  // stops there; what was emitted is a verified prefix
        CHECK(d.r.status == Errc::payload_corrupt || d.r.status == Errc::content_mismatch);
        CHECK(d.out.size() <= data.size() && std::equal(d.out.begin(), d.out.end(), data.begin()));
      } else {
        // Exactly the damaged block's frame is affected; the rest is verified data.
        CHECK(d.r.status == Errc::ok && d.out.size() == data.size() && d.r.stats.blocks_lost >= 1);
        std::size_t diff = 0;
        for (std::size_t i = 0; i < data.size(); ++i) diff += d.out[i] != data[i] && d.out[i] != std::byte{0xEE};
        CHECK(diff == 0);
      }
      ++rejected;
    }
  }
  std::printf("hostile payloads: %u rejected or contained, %u harmless edits\n", rejected, harmless);

  // A legacy LZ payload presented as a CM stream (ids rewritten, CRCs valid):
  // decoded as CM it is garbage, and the decoder says so.
  Vec legacy(reinterpret_cast<const std::byte*>(hce_test::kLegacyLzStream),
             reinterpret_cast<const std::byte*>(hce_test::kLegacyLzStream) + sizeof hce_test::kLegacyLzStream);
  store_le<u32>(&legacy[12], hce::kNeuralCMCodecId);
  store_le<u32>(&legacy[28], crc32c(Bytes(legacy).first(28)));
  const std::size_t fh = kStreamHeaderSize;
  store_le<u32>(&legacy[fh + 12], hce::kNeuralCMCodecId);
  store_le<u32>(&legacy[fh + 36], crc32c(Bytes(legacy).subspan(fh, 36)));
  for (const unsigned t : {1u, 3u}) {
    const Decoded d = decode(legacy, {}, t);
    CHECK(d.r.status == Errc::payload_corrupt || d.r.status == Errc::content_mismatch);
    CHECK(d.out.empty());
  }
}

// R12: a block that expands is stored and the encoder rebuilds its models by
// replay; the decoder, which never sees the failed attempt, must agree.
static void test_replay() {
  // Text and noise alternate within frames; forcing CM makes every noise block
  // fail and replay, three times per frame, then the rest of the frame is stored.
  Vec data;
  for (int i = 0; i < 12; ++i) data = concat({data, prose(6000, 50u + static_cast<unsigned>(i)), random_bytes(9000, 70u + static_cast<unsigned>(i))});
  Codec::test_force_cm(true);
  hce::EncodeResult forced{};
  const Vec s1 = encode(data, opts(0, 16, 12, 60 * 1024), &forced);
  Codec::test_force_cm(false);
  hce::EncodeResult normal{};
  const Vec s0 = encode(data, opts(0, 16, 12, 60 * 1024), &normal);
  CHECK(forced.stored_blocks > 0 && normal.stored_blocks > 0);
  for (const Vec* s : {&s0, &s1}) {
    for (const unsigned t : {1u, 3u}) {
      const Decoded d = decode(*s, {}, t);
      CHECK(d.r.status == Errc::ok && d.out == data);
    }
  }

  // Directly: the counters show the replays and the per-frame cap.
  StreamParams p;
  p.prior_id = hce::kNeuralCMCodecId;
  p.level = 0;
  p.window_log = 16;
  p.block_log = 12;
  Codec enc(p);
  const Vec frame = concat({prose(8192, 80), random_bytes(4096 * 5, 81), prose(8192, 82)});
  Codec::test_force_cm(true);
  enc.reset_frame(frame);
  Vec out(4096);
  std::size_t cm_blocks = 0;
  for (std::size_t b = 0; b < frame.size(); b += 4096) cm_blocks += enc.encode_block(b, b + 4096, MutBytes(out)) != 0;
  Codec::test_force_cm(false);
  CHECK(enc.counters().replays == 3);  // the cap: after three, the rest of the frame is stored
  CHECK(cm_blocks == 2);               // the two prose blocks before the noise
}

// The encoder's two-stage pipeline (ncm_pipe.hpp) codes the same bits as the
// single-threaded loop on every kernel path, through expanding blocks (the
// helper is stopped early, then the frame replays) and at every level.
static void test_pipeline() {
  StreamParams p;
  p.prior_id = hce::kNeuralCMCodecId;
  p.window_log = 16;
  p.block_log = 12;
  const Vec frame = concat({log_lines(9000, 100), random_bytes(4096 * 2, 101), prose(12000, 102),
                            binary(8192, 103), random_bytes(4096, 104), prose(6000, 105)});
  unsigned paths = 0;
  for (const char* name : {"scalar", "sse2", "avx2", "avx512", "neon"}) {
    if (!Codec::test_select_simd(name)) continue;
    ++paths;
    for (const u8 level : {u8{0}, u8{2}}) {
      p.level = level;
      Codec one(p), two(p);
      two.set_pipeline(true);
      Codec::test_force_cm(true);  // the noise blocks expand: stops and replays
      one.reset_frame(frame);
      two.reset_frame(frame);
      Vec o1(4096), o2(4096);
      for (std::size_t b = 0; b < frame.size(); b += 4096) {
        const std::size_t e = std::min(frame.size(), b + 4096);
        const std::size_t n1 = one.encode_block(b, e, MutBytes(o1).first(e - b));
        const std::size_t n2 = two.encode_block(b, e, MutBytes(o2).first(e - b));
        CHECK(n1 == n2 && std::equal(o1.begin(), o1.begin() + static_cast<std::ptrdiff_t>(n1), o2.begin()));
      }
      Codec::test_force_cm(false);
      CHECK(one.counters().replays > 0 && one.counters().replays == two.counters().replays);
      CHECK(one.counters().cm_blocks == two.counters().cm_blocks);
    }
  }
  Codec::test_select_simd(nullptr);
  CHECK(paths >= 2);

  // Through the parallel encoder: spare threads go to pipelines, the workers
  // are capped at the frame count, and the stream does not change.
  const Vec data = concat({log_lines(60000, 106), prose(50000, 107)});
  hce::EncodeOptions o = opts(2, 16, 13, 64 * 1024);
  o.content_size = data.size();  // the plan needs it to count frames
  const Vec ref = encode(data, o);
  for (const unsigned t : {2u, 4u, 5u}) {
    o.threads = t;
    hce::EncodeResult r{};
    CHECK(encode(data, o, &r) == ref);
    CHECK(r.threads == std::min(t, 2u));  // two frames
  }
  std::printf("pipeline: identical blocks on %u kernel set(s) and identical streams\n", paths);
}

// R20: frames are independent. Decoding from any frame header on (a late
// join) yields exactly the original bytes of the frames that follow.
static void test_late_join() {
  const Vec data = concat({log_lines(50000, 90), prose(40000, 91)});
  const Vec s = encode(data, opts(1, 16, 13, 16 * 1024));
  std::vector<std::size_t> frame_at;
  for (std::size_t i = kStreamHeaderSize; i + kFrameHeaderSize <= s.size(); ++i)
    if (std::equal(kSyncMarker.begin(), kSyncMarker.end(), s.begin() + static_cast<std::ptrdiff_t>(i))) {
      FrameHeader f;
      if (parse_frame_header(Bytes(s).subspan(i, kFrameHeaderSize), f) == Errc::ok) frame_at.push_back(i);
    }
  CHECK(frame_at.size() >= 5);
  for (const std::size_t k : {std::size_t{1}, frame_at.size() / 2, frame_at.size() - 1}) {
    FrameHeader f;
    CHECK(parse_frame_header(Bytes(s).subspan(frame_at[k], kFrameHeaderSize), f) == Errc::ok);
    const Vec tail(s.begin() + static_cast<std::ptrdiff_t>(frame_at[k]), s.end());
    hce::DecodeOptions o;
    o.recovery = Recovery::skip;
    for (const unsigned t : {1u, 2u}) {
      const Decoded d = decode(tail, o, t);
      CHECK(d.r.status == Errc::ok);
      CHECK(d.out.size() == data.size() - f.out_offset);
      CHECK(std::equal(d.out.begin(), d.out.end(), data.begin() + static_cast<std::ptrdiff_t>(f.out_offset)));
    }
  }
}

// R21: the match model only reaches history inside the window. A repeat
// farther back than the window cannot be matched (and decoding would break if
// the encoder used it); inside the window it costs almost nothing.
static void test_window_bound() {
  const Vec r = log_lines(100 * 1024, 95);
  const Vec far = concat({r, r});  // repeat at distance 100 KiB
  Vec s_one, s_far64, s_far1m;
  CHECK(roundtrips(r, opts(2, 16, 14, 1024 * 1024), &s_one));
  CHECK(roundtrips(far, opts(2, 16, 14, 1024 * 1024), &s_far64));
  CHECK(roundtrips(far, opts(2, 20, 14, 1024 * 1024), &s_far1m));
  std::printf("window bound: one copy %zu B; two copies %zu B with a 64 KiB window, %zu B with 1 MiB\n",
              s_one.size(), s_far64.size(), s_far1m.size());
  CHECK(s_far1m.size() < s_one.size() * 11 / 10);  // in reach: the second copy is nearly free
  CHECK(s_far64.size() > s_one.size() * 13 / 10);  // out of reach: it has to be coded again
  // Long text across a small window, multi-frame and multi-threaded both ways.
  const Vec text = concat({prose(150000, 97), log_lines(150000, 98)});
  hce::EncodeOptions o = opts(2, 16, 15, 256 * 1024);
  o.threads = 3;
  const Vec s = encode(text, o);
  o.threads = 1;
  CHECK(encode(text, o) == s);
  const Decoded d = decode(s, {}, 3);
  CHECK(d.r.status == Errc::ok && d.out == text);
}

// R19: resetting the models at a frame costs time in proportion to the work
// done in the frame, not to the tables: a stream of tiny frames must decode
// about as fast as one frame holding the same bytes.
static void test_reset_cost() {
  const Vec data = concat({log_lines(300000, 99), prose(100000, 100)});
  const Vec tiny = encode(data, opts(4, 16, 10, 1024));  // 1 KiB frames, 149 MiB of tables
  const Vec one = encode(data, opts(4, 16, 10, 1024 * 1024));
  using clock = std::chrono::steady_clock;
  const auto time = [](const Vec& s, const Vec& want) {
    double best = 1e9;
    for (int rep = 0; rep < 2; ++rep) {
      const auto t0 = clock::now();
      const Decoded d = decode(s);
      best = std::min(best, std::chrono::duration<double>(clock::now() - t0).count());
      CHECK(d.r.status == Errc::ok && d.out == want);
    }
    return best;
  };
  const double t_tiny = time(tiny, data), t_one = time(one, data);
  std::printf("reset cost: %zu one-KiB frames decode in %.2fs, one frame in %.2fs\n", data.size() / 1024, t_tiny,
              t_one);
  CHECK(t_tiny < 4 * t_one + 0.5);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  static constexpr struct {
    const char* name;
    void (*fn)();
  } kTests[] = {
      {"kernel_parity", test_kernel_parity}, {"node_table", test_node_table}, {"contract", test_contract},
      {"roundtrip_matrix", test_roundtrip_matrix}, {"simd_paths_agree", test_simd_paths_agree},
      {"golden_streams", test_golden_streams}, {"hostile_payloads", test_hostile_payloads},
      {"replay", test_replay}, {"pipeline", test_pipeline}, {"late_join", test_late_join},
      {"window_bound", test_window_bound}, {"reset_cost", test_reset_cost},
  };
  for (const auto& t : kTests) {
    const auto t0 = std::chrono::steady_clock::now();
    t.fn();
    std::printf("  [%s: %.1fs]\n", t.name,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all codec tests passed\n");
  return 0;
}
