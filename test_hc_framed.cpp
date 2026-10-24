// Tests for hc_framed.hpp. A tiny LZ token codec stands in for the CM codec
// (same BlockCodec contract: literals + window back-references).
#include "hc_framed.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace hc;

static int g_failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

using Vec = std::vector<std::byte>;

// ------------------------------------------------------------ test codec ----

struct TokenCodec {
  MutBytes table;  // stands in for model tables carved from the arena
  int binds = 0, resets = 0;

  static Errc accepts(const StreamParams&) noexcept { return Errc::ok; }
  static u64 memory_required(const StreamParams&) noexcept { return 4096; }
  Errc bind(const StreamParams&, MutBytes mem) noexcept {
    table = mem;
    ++binds;
    return Errc::ok;
  }
  void reset_frame() noexcept {
    std::memset(table.data(), 0, table.size());  // "restore the prior image"
    ++resets;
  }
  // Tokens: 0x00 varint(n) bytes[n] | 0x01 varint(offset) varint(n)
  Errc decode_block(Bytes payload, u32 usize, Window& w) noexcept {
    std::size_t i = 0;
    u64 produced = 0;
    while (produced < usize) {
      if (i >= payload.size()) return Errc::payload_corrupt;
      const u8 op = std::to_integer<u8>(payload[i++]);
      u64 a = 0, n = 0;
      if (op == 0) {
        if (read_varint(payload, i, n) != Errc::ok || n == 0 || n > usize - produced || n > payload.size() - i)
          return Errc::payload_corrupt;
        w.append(payload.subspan(i, static_cast<std::size_t>(n)));
        i += static_cast<std::size_t>(n);
      } else if (op == 1) {
        if (read_varint(payload, i, a) != Errc::ok || read_varint(payload, i, n) != Errc::ok || n == 0 ||
            n > usize - produced)
          return Errc::payload_corrupt;
        if (const Errc e = w.copy(a, n); e != Errc::ok) return e;
      } else {
        return Errc::payload_corrupt;
      }
      produced += n;
    }
    return i == payload.size() ? Errc::ok : Errc::payload_corrupt;
  }
};
static_assert(BlockCodec<TokenCodec>);

// Greedy LZ encoder producing TokenCodec payloads; history is frame-local.
struct TokenEncoder {
  Vec hist;
  std::vector<std::int64_t> head = std::vector<std::int64_t>(1 << 16, -1);
  u64 window;
  explicit TokenEncoder(u64 w) : window(w) {}
  void reset() {
    hist.clear();
    std::fill(head.begin(), head.end(), -1);
  }
  static void put_varint(Vec& out, u64 v) {
    std::byte tmp[10];
    const std::size_t n = write_varint(tmp, v);
    out.insert(out.end(), tmp, tmp + n);
  }
  std::size_t hash4(std::size_t i) const {
    const u32 v = load_le<u32>(&hist[i]);
    return (v * 2654435761u) >> 16;
  }
  Vec encode(Bytes content) {
    const std::size_t base = hist.size();
    hist.insert(hist.end(), content.begin(), content.end());
    Vec out;
    std::size_t lit = base;
    auto flush = [&](std::size_t to) {
      if (to == lit) return;
      out.push_back(std::byte{0});
      put_varint(out, to - lit);
      out.insert(out.end(), hist.begin() + static_cast<std::ptrdiff_t>(lit),
                 hist.begin() + static_cast<std::ptrdiff_t>(to));
    };
    for (std::size_t i = base; i < hist.size();) {
      std::size_t best = 0, off = 0;
      if (i + 4 <= hist.size()) {
        const std::size_t h = hash4(i);
        const std::int64_t cand = head[h];
        head[h] = static_cast<std::int64_t>(i);
        if (cand >= 0 && i - static_cast<std::size_t>(cand) <= window) {
          const std::size_t c = static_cast<std::size_t>(cand);
          std::size_t len = 0;
          while (i + len < hist.size() && hist[c + len] == hist[i + len]) ++len;
          if (len >= 4) best = len, off = i - c;
        }
      }
      if (best) {
        flush(i);
        out.push_back(std::byte{1});
        put_varint(out, off);
        put_varint(out, best);
        i += best;
        lit = i;
      } else {
        ++i;
      }
    }
    flush(hist.size());
    return out;
  }
};

// ----------------------------------------------------------------- helpers ---

struct VecSink {
  Vec data;
  bool write(Bytes b) {
    data.insert(data.end(), b.begin(), b.end());
    return true;
  }
};
static_assert(ByteSink<VecSink>);

struct ChunkSource {
  Bytes data;
  std::size_t pos = 0;
  std::mt19937* rng;
  std::size_t read(MutBytes b) {
    std::size_t n = std::min<std::size_t>({b.size(), data.size() - pos, 1 + (*rng)() % 4096});
    if (n) std::memcpy(b.data(), data.data() + pos, n);
    pos += n;
    return n;
  }
};
static_assert(ByteSource<ChunkSource>);

struct BlockInfo {
  std::size_t header_at;  // compressed offset of the block header
  u64 out_offset;
  u32 usize;
  bool stored;
  u32 frame;
};

struct Built {
  Vec stream;
  Vec content;
  std::vector<BlockInfo> blocks;
  std::vector<u64> frame_offsets;
};

static Vec make_content(std::size_t lines, std::mt19937& rng) {
  Vec v;
  char buf[160];
  for (std::size_t i = 0; i < lines; ++i) {
    if (i % 97 == 50) {  // incompressible span -> STORED blocks
      for (int k = 0; k < 16384; ++k) v.push_back(static_cast<std::byte>(rng()));
      continue;
    }
    const int n = std::snprintf(buf, sizeof buf, "2026-10-24T12:%02zu:%02zu host-%u GET /api/v1/items/%u status=%u latency=%ums\n",
                                (i / 60) % 60, i % 60, static_cast<unsigned>(rng() % 8),
                                static_cast<unsigned>(rng() % 100000), i % 13 ? 200u : 503u,
                                static_cast<unsigned>(rng() % 900));
    for (int k = 0; k < n; ++k) v.push_back(static_cast<std::byte>(buf[k]));
  }
  return v;
}

static StreamParams params(u8 window_log, u8 block_log, std::optional<u64> size, u32 id = 0x5EED1234) {
  StreamParams p;
  p.level = 2;
  p.window_log = window_log;
  p.block_log = block_log;
  p.prior_id = 7;
  p.stream_id = id;
  p.content_size = size;
  return p;
}

static Built build(const Vec& content, const StreamParams& p, u64 sync, std::size_t block, bool with_ancillary = true) {
  Built b;
  b.content = content;
  VecSink sink;
  FramedWriter<VecSink> w(sink, p, sync);
  TokenEncoder enc(p.window_size());
  CHECK(w.begin());
  u32 frame = 0;
  bool first = true;
  for (std::size_t at = 0; at < content.size(); at += block) {
    const Bytes chunk = Bytes(content).subspan(at, std::min(block, content.size() - at));
    if (w.frame_due()) {
      enc.reset();
      if (!first) ++frame;
      first = false;
      b.frame_offsets.push_back(at);
    }
    const Vec coded = enc.encode(chunk);
    const bool stored = coded.size() >= chunk.size();
    const std::size_t payload = stored ? chunk.size() : coded.size();
    CHECK(w.block(chunk, coded));
    b.blocks.push_back({sink.data.size() - payload - kBlockHeaderSize, at, static_cast<u32>(chunk.size()), stored, frame});
    if (with_ancillary && b.blocks.size() == 3) {
      const char meta[] = "provenance: test";
      CHECK(w.ancillary(5, Bytes(reinterpret_cast<const std::byte*>(meta), sizeof meta)));
    }
  }
  CHECK(w.end());
  b.stream = std::move(sink.data);
  return b;
}

struct Result {
  Errc status = Errc::ok;
  Vec out;
  DecodeStats stats;
  std::vector<GapInfo> gaps;
  int binds = 0;
};

static void collect_gap(void* ctx, const GapInfo& g) noexcept { static_cast<std::vector<GapInfo>*>(ctx)->push_back(g); }

static Result decode_all(Bytes stream, DecoderOptions opt, std::mt19937& rng, std::size_t max_in = 1 << 16,
                         std::size_t max_out = 1 << 16) {
  Result r;
  opt.on_gap = collect_gap;
  opt.gap_ctx = &r.gaps;
  TokenCodec codec;
  StreamDecoder<TokenCodec> dec(codec, opt);
  Vec outbuf(max_out);
  std::size_t pos = 0;
  for (int guard = 0; guard < 50'000'000; ++guard) {
    const std::size_t n = std::min<std::size_t>(stream.size() - pos, 1 + rng() % max_in);
    const bool fin = pos + n == stream.size();
    const std::size_t out_n = 1 + rng() % max_out;
    const Progress p = dec.decode(stream.subspan(pos, n), MutBytes(outbuf).first(out_n), fin);
    pos += p.consumed;
    r.out.insert(r.out.end(), outbuf.begin(), outbuf.begin() + static_cast<std::ptrdiff_t>(p.produced));
    if (p.status == Errc::need_input || p.status == Errc::output_full) continue;
    if (p.status == Errc::end_of_stream && !(fin && pos == stream.size())) continue;
    r.status = p.status;
    break;
  }
  r.stats = dec.stats();
  r.binds = codec.binds;
  return r;
}

static bool is_prefix(const Vec& a, const Vec& of) {
  return a.size() <= of.size() && std::equal(a.begin(), a.end(), of.begin());
}

// Expected skip/fill output when the blocks marked lost are removed/filled.
static Vec expect(const Built& b, const std::vector<bool>& lost, bool fill) {
  Vec v;
  for (std::size_t i = 0; i < b.blocks.size(); ++i) {
    const auto& bl = b.blocks[i];
    if (!lost[i])
      v.insert(v.end(), b.content.begin() + static_cast<std::ptrdiff_t>(bl.out_offset),
               b.content.begin() + static_cast<std::ptrdiff_t>(bl.out_offset + bl.usize));
    else if (fill)
      v.insert(v.end(), bl.usize, std::byte{0xEE});
  }
  return v;
}

static void restamp_stream_header(Vec& s, std::size_t at = 0) {
  store_le<u32>(&s[at + 28], crc32c(Bytes(s).subspan(at, 28)));
}

// ------------------------------------------------------------------- tests ---

static void test_varint() {
  for (u64 v : {u64{0}, u64{1}, u64{127}, u64{128}, u64{1} << 35, ~u64{0}}) {
    std::byte buf[10];
    const std::size_t n = write_varint(buf, v);
    std::size_t pos = 0;
    u64 got = 0;
    CHECK(read_varint(Bytes(buf, n), pos, got) == Errc::ok && got == v && pos == n);
  }
  const std::byte overlong[] = {std::byte{0x80}, std::byte{0x00}};
  std::size_t pos = 0;
  u64 v = 0;
  CHECK(read_varint(overlong, pos, v) == Errc::payload_corrupt);
  std::byte too_big[10];
  std::fill(std::begin(too_big), std::end(too_big), std::byte{0xFF});
  too_big[9] = std::byte{0x02};
  pos = 0;
  CHECK(read_varint(too_big, pos, v) == Errc::payload_corrupt);
}

static void test_window_property() {
  std::mt19937 rng(1);
  for (std::size_t cap : {std::size_t{16}, std::size_t{64}, std::size_t{256}}) {
    Vec mem(cap);
    Window w;
    w.bind(mem);
    Vec all;  // unbounded reference history
    std::size_t frame_start = 0;
    for (int op = 0; op < 20000; ++op) {
      const auto kind = rng() % 10;
      if (kind == 0) {
        w.reset_frame();
        frame_start = all.size();
      } else if (kind < 4) {
        Vec lit(1 + rng() % cap);
        for (auto& x : lit) x = static_cast<std::byte>(rng());
        w.append(lit);
        all.insert(all.end(), lit.begin(), lit.end());
      } else {
        const u64 reach = std::min<u64>(cap, all.size() - frame_start);
        const u64 off = 1 + rng() % (cap + 2);
        const u64 len = 1 + rng() % (3 * cap);
        const Errc e = w.copy(off, len);
        if (off > reach) {
          CHECK(e == Errc::bad_reference);
          continue;
        }
        CHECK(e == Errc::ok);
        for (u64 i = 0; i < len; ++i) all.push_back(all[all.size() - off]);
      }
      CHECK(w.pos() == all.size());
      const u64 live = std::min<u64>(cap, all.size());
      for (u64 a = all.size() - live; a < all.size(); ++a)
        if (w.at(a) != all[a]) {
          CHECK(false && "window content diverged");
          return;
        }
    }
  }
}

static void test_roundtrip() {
  std::mt19937 rng(2);
  const Vec content = make_content(4000, rng);
  for (const bool known : {false, true}) {
    const auto p = params(16, 13, known ? std::optional<u64>{content.size()} : std::nullopt);
    const Built b = build(content, p, u64{64} << 10, 8192);
    std::size_t stored = 0;
    for (const auto& bl : b.blocks) stored += bl.stored;
    CHECK(stored > 0 && stored < b.blocks.size());
    CHECK(b.frame_offsets.size() > 3);
    for (const Recovery mode : {Recovery::strict, Recovery::skip, Recovery::skip_and_fill}) {
      DecoderOptions opt;
      opt.recovery = mode;
      // tiny chunks exercise every restart point of the state machine
      const Result tiny = decode_all(b.stream, opt, rng, 97, 61);
      CHECK(tiny.status == Errc::end_of_stream);
      CHECK(tiny.out == content);
      CHECK(tiny.stats.content_verified);
      CHECK(tiny.stats.blocks_ok == b.blocks.size());
      CHECK(tiny.gaps.empty());
      const Result big = decode_all(b.stream, opt, rng);
      CHECK(big.status == Errc::end_of_stream && big.out == content);
    }
  }
}

static void test_empty_stream() {
  std::mt19937 rng(3);
  for (const bool known : {false, true}) {
    const Built b = build({}, params(16, 12, known ? std::optional<u64>{0} : std::nullopt), 1 << 16, 4096);
    const Result r = decode_all(b.stream, {}, rng);
    CHECK(r.status == Errc::end_of_stream && r.out.empty() && r.stats.content_verified);
  }
}

static void test_truncation() {
  std::mt19937 rng(4);
  const Vec content = make_content(600, rng);
  const Built b = build(content, params(16, 12, std::nullopt), 16 << 10, 4096);
  for (std::size_t cut = 0; cut < b.stream.size(); cut += 1 + cut / 7) {
    for (const Recovery mode : {Recovery::strict, Recovery::skip}) {
      DecoderOptions opt;
      opt.recovery = mode;
      const Result r = decode_all(Bytes(b.stream).first(cut), opt, rng, 211, 173);
      CHECK(r.status == Errc::truncated);
      CHECK(is_prefix(r.out, content));
    }
  }
}

static void test_payload_corruption() {
  std::mt19937 rng(5);
  const Vec content = make_content(4000, rng);
  const Built b = build(content, params(16, 13, std::nullopt), u64{64} << 10, 8192);
  // pick a CM block in the middle of a frame that is followed by a STORED block in the same frame
  std::size_t victim = b.blocks.size();
  for (std::size_t i = 1; i + 1 < b.blocks.size() && victim == b.blocks.size(); ++i) {
    const auto& bl = b.blocks[i];
    if (bl.stored || b.blocks[i - 1].frame != bl.frame) continue;
    for (std::size_t j = i + 1; j < b.blocks.size() && b.blocks[j].frame == bl.frame; ++j)
      if (b.blocks[j].stored) victim = i;
  }
  CHECK(victim < b.blocks.size());
  if (victim == b.blocks.size()) return;
  Vec bad = b.stream;
  bad[b.blocks[victim].header_at + kBlockHeaderSize + 3] ^= std::byte{0x40};

  std::vector<bool> lost(b.blocks.size(), false);
  for (std::size_t i = victim; i < b.blocks.size() && b.blocks[i].frame == b.blocks[victim].frame; ++i)
    lost[i] = (i == victim) || !b.blocks[i].stored;  // STORED blocks after the damage are salvaged

  DecoderOptions strict;
  const Result s = decode_all(bad, strict, rng, 333, 777);
  CHECK(s.status == Errc::payload_corrupt);
  CHECK(is_prefix(s.out, content));
  CHECK(s.out.size() <= b.blocks[victim].out_offset);  // nothing unverified was emitted

  DecoderOptions skip;
  skip.recovery = Recovery::skip;
  const Result k = decode_all(bad, skip, rng, 333, 777);
  CHECK(k.status == Errc::end_of_stream);
  CHECK(k.out == expect(b, lost, false));
  CHECK(!k.stats.content_verified && k.stats.blocks_salvaged > 0);
  CHECK(!k.gaps.empty() && k.gaps[0].out_offset == b.blocks[victim].out_offset && k.gaps[0].length_known);

  DecoderOptions fill;
  fill.recovery = Recovery::skip_and_fill;
  fill.fill = std::byte{0xEE};
  const Result f = decode_all(bad, fill, rng, 333, 777);
  CHECK(f.status == Errc::end_of_stream);
  CHECK(f.out.size() == content.size());
  CHECK(f.out == expect(b, lost, true));
}

static void test_header_corruption_resync() {
  std::mt19937 rng(6);
  const Vec content = make_content(4000, rng);
  const Built b = build(content, params(16, 13, std::nullopt), u64{64} << 10, 8192);
  std::size_t victim = 0;
  for (std::size_t i = 1; i < b.blocks.size(); ++i)
    if (b.blocks[i].frame == 2 && b.blocks[i - 1].frame == 2) {
      victim = i;
      break;
    }
  CHECK(victim > 0);
  if (victim == 0) return;
  Vec bad = b.stream;
  bad[b.blocks[victim].header_at + 2] ^= std::byte{0x10};  // usize field: sizes no longer trustworthy

  std::vector<bool> lost(b.blocks.size(), false);
  for (std::size_t i = victim; i < b.blocks.size() && b.blocks[i].frame == 2; ++i) lost[i] = true;

  DecoderOptions strict;
  CHECK(decode_all(bad, strict, rng).status == Errc::header_corrupt);

  DecoderOptions fill;
  fill.recovery = Recovery::skip_and_fill;
  fill.fill = std::byte{0xEE};
  const Result f = decode_all(bad, fill, rng, 501, 999);
  CHECK(f.status == Errc::end_of_stream);
  CHECK(f.out == expect(b, lost, true));
  CHECK(f.gaps.size() == 1 && f.gaps[0].out_offset == b.blocks[victim].out_offset);
}

static void test_fake_marker_rejected() {
  std::mt19937 rng(7);
  const auto p = params(16, 13, std::nullopt, 0xAAAA0001);
  Vec text = make_content(200, rng);
  text.resize(8000);
  Vec noise(8000);
  for (auto& x : noise) x = static_cast<std::byte>(rng());
  // a valid-looking frame header of a different stream (e.g. a .hcz file inside a tarball)
  const auto fake = encode_frame_header(FrameHeader{params(16, 13, std::nullopt, 0xBBBB0002), 1, 8000});
  std::copy(fake.begin(), fake.end(), noise.begin() + 100);

  VecSink sink;
  FramedWriter<VecSink> w(sink, p, 16000);
  TokenEncoder enc(p.window_size());
  CHECK(w.begin());
  CHECK(w.frame_due());
  enc.reset();
  const std::size_t a_header = sink.data.size() + kFrameHeaderSize;
  CHECK(w.block(text, enc.encode(text)));
  CHECK(w.block(noise, {}));  // STORED, contains the fake header
  CHECK(w.frame_due());
  enc.reset();
  CHECK(w.block(text, enc.encode(text)));
  CHECK(w.end());
  Vec bad = sink.data;
  bad[a_header + 5] ^= std::byte{0x01};  // damage block A's header: forces a scan over the fake

  DecoderOptions fill;
  fill.recovery = Recovery::skip_and_fill;
  fill.fill = std::byte{0xEE};
  const Result f = decode_all(bad, fill, rng, 64, 64);
  CHECK(f.status == Errc::end_of_stream);
  Vec want(16000, std::byte{0xEE});
  want.insert(want.end(), text.begin(), text.end());
  CHECK(f.out == want);
  CHECK(f.gaps.size() == 1 && f.gaps[0].out_offset == 0 && f.gaps[0].length == 16000);
}

static void test_bomb_and_limits() {
  std::mt19937 rng(8);
  const Vec zeros(u64{8} << 20, std::byte{'A'});
  const Built b = build(zeros, params(20, 20, std::nullopt), u64{1} << 30, u64{1} << 20, false);
  CHECK(b.stream.size() < 400);  // ~8 MiB from a few hundred bytes

  DecoderOptions trusted;
  const Result ok = decode_all(b.stream, trusted, rng);
  CHECK(ok.status == Errc::end_of_stream && ok.out == zeros);

  DecoderOptions strict_ratio;
  strict_ratio.limits.max_ratio = 100;
  strict_ratio.limits.ratio_burst_bytes = u64{4} << 20;
  const Result r = decode_all(b.stream, strict_ratio, rng);
  CHECK(r.status == Errc::limit_ratio);
  CHECK(r.out.size() <= (u64{4} << 20));

  DecoderOptions skip_ratio = strict_ratio;  // recovery never bypasses R19
  skip_ratio.recovery = Recovery::skip_and_fill;
  CHECK(decode_all(b.stream, skip_ratio, rng).status == Errc::limit_ratio);

  DecoderOptions out_cap;
  out_cap.limits.max_output_bytes = u64{3} << 20;
  const Result o = decode_all(b.stream, out_cap, rng);
  CHECK(o.status == Errc::limit_output && o.out.size() <= (u64{3} << 20));

  // Header-declared limits reject before any allocation or codec binding.
  const auto check_header = [&](StreamParams p, DecodeLimits lim, Errc want) {
    VecSink s;
    FramedWriter<VecSink> w(s, p, 1 << 20);
    CHECK(w.begin() && w.end());
    DecoderOptions opt;
    opt.limits = lim;
    const Result res = decode_all(s.data, opt, rng);
    CHECK(res.status == want);
    CHECK(res.binds == 0);
  };
  const DecodeLimits u = DecodeLimits::untrusted();
  check_header(params(27, 20, std::nullopt), u, Errc::limit_window);
  check_header(params(25, 21, std::nullopt), u, Errc::limit_block);
  auto lvl = params(20, 18, std::nullopt);
  lvl.level = 4;
  check_header(lvl, u, Errc::limit_level);
  check_header(params(20, 18, u64{1} << 40), u, Errc::limit_output);
  DecodeLimits small = u;
  small.max_memory_bytes = u64{16} << 20;
  check_header(params(25, 18, std::nullopt), small, Errc::limit_memory);
  check_header(params(31, 18, std::nullopt), DecodeLimits::trusted(), Errc::header_corrupt);
}

static void test_bad_reference() {
  std::mt19937 rng(9);
  const auto p = params(16, 12, std::nullopt);
  VecSink sink;
  FramedWriter<VecSink> w(sink, p, 4096);
  CHECK(w.begin());
  const Vec first(4096, std::byte{'x'});
  CHECK(w.block(first, {}));
  CHECK(w.frame_due());  // frame 1 starts here: history from frame 0 is unreachable
  Vec payload;
  payload.push_back(std::byte{0});
  TokenEncoder::put_varint(payload, 1);
  payload.push_back(std::byte{'y'});
  payload.push_back(std::byte{1});
  TokenEncoder::put_varint(payload, 100);  // reaches back into frame 0
  TokenEncoder::put_varint(payload, 50);
  const Vec claimed(51, std::byte{'x'});
  CHECK(w.block(claimed, payload));
  CHECK(w.end());

  CHECK(decode_all(sink.data, {}, rng).status == Errc::bad_reference);
  DecoderOptions skip;
  skip.recovery = Recovery::skip;
  const Result r = decode_all(sink.data, skip, rng);
  CHECK(r.status == Errc::end_of_stream && r.out == first);
  CHECK(r.gaps.size() == 1 && r.gaps[0].cause == Errc::bad_reference && r.gaps[0].length == 51);
}

static void test_concatenation_and_trailing() {
  std::mt19937 rng(10);
  const Vec c1 = make_content(300, rng), c2 = make_content(200, rng);
  const Built a = build(c1, params(16, 12, std::nullopt, 1), 8192, 4096);
  const Built b = build(c2, params(17, 13, c2.size(), 2), 1 << 20, 8192);
  Vec both = a.stream;
  both.insert(both.end(), b.stream.begin(), b.stream.end());
  const Result r = decode_all(both, {}, rng, 777, 555);
  Vec want = c1;
  want.insert(want.end(), c2.begin(), c2.end());
  CHECK(r.status == Errc::end_of_stream && r.out == want);

  for (const std::size_t junk : {std::size_t{10}, std::size_t{64}}) {
    Vec tail = a.stream;
    tail.insert(tail.end(), junk, std::byte{0x42});
    const Result t = decode_all(tail, {}, rng);
    CHECK(t.status == Errc::trailing_garbage && t.out == c1);
  }
}

static void test_damaged_stream_header_and_late_join() {
  std::mt19937 rng(11);
  const Vec content = make_content(3000, rng);
  const Built b = build(content, params(16, 13, std::nullopt), u64{64} << 10, 8192);
  Vec bad = b.stream;
  bad[13] ^= std::byte{0x01};  // prior_id byte: header CRC now fails
  CHECK(decode_all(bad, {}, rng).status == Errc::header_corrupt);
  DecoderOptions skip;
  skip.recovery = Recovery::skip;
  const Result r = decode_all(bad, skip, rng, 300, 300);
  CHECK(r.status == Errc::end_of_stream && r.out == content);
  CHECK(r.gaps.empty() && r.stats.damage_events >= 1 && !r.stats.content_verified);

  // Late join: a consumer attaches to a live stream at a frame boundary.
  const std::size_t join_block = [&] {
    for (std::size_t i = 0; i < b.blocks.size(); ++i)
      if (b.blocks[i].frame == 3) return i;
    return std::size_t{0};
  }();
  CHECK(join_block > 0);
  if (join_block == 0) return;
  const std::size_t join_at = b.blocks[join_block].header_at - kFrameHeaderSize;
  const Result late = decode_all(Bytes(b.stream).subspan(join_at), skip, rng, 300, 300);
  const u64 off = b.blocks[join_block].out_offset;
  CHECK(late.status == Errc::end_of_stream);
  CHECK(late.out == Vec(content.begin() + static_cast<std::ptrdiff_t>(off), content.end()));
  CHECK(late.gaps.size() == 1 && late.gaps[0].out_offset == 0 && late.gaps[0].length == off);
}

static void test_feature_bits_and_cancel() {
  std::mt19937 rng(12);
  const Built b = build(make_content(100, rng), params(16, 12, std::nullopt), 1 << 20, 4096);
  Vec future = b.stream;
  future[5] = std::byte{kFeatFramed | 0x02};  // an incompat feature this decoder does not know
  restamp_stream_header(future);
  DecoderOptions skip;
  skip.recovery = Recovery::skip;
  CHECK(decode_all(future, {}, rng).status == Errc::unsupported_feature);
  CHECK(decode_all(future, skip, rng).status == Errc::unsupported_feature);

  Vec compat = b.stream;
  compat[6] = std::byte{0x80};  // unknown compat bit: ignored
  restamp_stream_header(compat);
  CHECK(decode_all(compat, {}, rng).status == Errc::end_of_stream);

  std::stop_source stop;
  stop.request_stop();
  DecoderOptions opt;
  opt.stop = stop.get_token();
  CHECK(decode_all(b.stream, opt, rng).status == Errc::cancelled);
}

static void test_pump_and_guards() {
  std::mt19937 rng(13);
  const Vec content = make_content(2000, rng);
  const Built b = build(content, params(16, 13, std::nullopt), u64{64} << 10, 8192);
  ChunkSource src{b.stream, 0, &rng};
  VecSink sink;
  TokenCodec codec;
  StreamDecoder<TokenCodec> dec(codec, {});
  Vec in_buf(4096), out_buf(3000);
  CHECK(pump(src, sink, dec, in_buf, out_buf) == Errc::ok);
  CHECK(sink.data == content);

  DecodeLimits lim;
  lim.max_output_bytes = 10;
  Budget budget(lim);
  VecSink inner;
  GuardedSink<VecSink> guarded{inner, budget};
  const Vec twenty(20, std::byte{1});
  CHECK(!guarded.write(twenty) && guarded.status == Errc::limit_output && inner.data.empty());
  ChunkSource raw{b.stream, 0, &rng};
  GuardedSource<ChunkSource> gsrc{raw, budget};
  CHECK(gsrc.read(in_buf) > 0);
}

static void test_tables() {
  CHECK(candidate_valid((u64{1} << 32) + 5, 3, 1 << 20));
  CHECK(!candidate_valid((u64{1} << 32) + 5, 6, 1 << 20));
  CHECK(!candidate_valid(100, 100, 1 << 20));
  CHECK(!candidate_valid(100, 10, 50));
  SlotTable<4, 8> t;
  const Vec body(5, std::byte{9});
  Bytes got;
  CHECK(t.get(1, got) == Errc::payload_corrupt);
  CHECK(t.define(1, body) == Errc::ok && t.get(1, got) == Errc::ok && got.size() == 5);
  CHECK(t.define(4, body) == Errc::payload_corrupt);
  CHECK(t.define(0, Vec(9)) == Errc::payload_corrupt);
  t.reset();
  CHECK(t.get(1, got) == Errc::payload_corrupt);
}

// Mutation fuzz: flips, insertions, deletions, truncation. Invariants:
//  strict          -> output is a prefix of the original (verify-then-emit)
//  skip_and_fill   -> every byte outside a reported gap equals the original
static void test_mutation_fuzz() {
  std::mt19937 rng(14);
  const Vec content = make_content(1500, rng);
  const Built b = build(content, params(16, 12, std::nullopt), 32 << 10, 4096);
  std::size_t recovered_runs = 0;
  for (int iter = 0; iter < 3000; ++iter) {
    Vec bad = b.stream;
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
    DecoderOptions strict;
    strict.limits = DecodeLimits::untrusted();
    const Result s = decode_all(bad, strict, rng, 1 + rng() % 5000, 1 + rng() % 5000);
    CHECK(is_prefix(s.out, content));

    DecoderOptions fill;
    fill.limits = DecodeLimits::untrusted();
    fill.recovery = Recovery::skip_and_fill;
    fill.fill = std::byte{0xEE};
    const Result f = decode_all(bad, fill, rng, 1 + rng() % 5000, 1 + rng() % 5000);
    CHECK(f.out.size() <= content.size());
    std::vector<bool> in_gap(f.out.size(), false);
    for (const GapInfo& g : f.gaps) {
      if (!g.length_known) continue;
      for (u64 i = g.out_offset; i < g.out_offset + g.length && i < in_gap.size(); ++i) in_gap[i] = true;
    }
    bool ok = true;
    for (std::size_t i = 0; i < f.out.size() && ok; ++i) ok = in_gap[i] || f.out[i] == content[i];
    CHECK(ok);
    if (!ok) {
      std::printf("  iteration %d diverged (status %d)\n", iter, static_cast<int>(f.status));
      return;
    }
    recovered_runs += f.out.size() > s.out.size();
  }
  CHECK(recovered_runs > 100);  // recovery must actually recover something in practice
  std::printf("fuzz: recovery beat strict in %zu of 3000 runs\n", recovered_runs);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  test_varint();
  test_window_property();
  test_roundtrip();
  test_empty_stream();
  test_truncation();
  test_payload_corruption();
  test_header_corruption_resync();
  test_fake_marker_rejected();
  test_bomb_and_limits();
  test_bad_reference();
  test_concatenation_and_trailing();
  test_damaged_stream_header_and_late_join();
  test_feature_bits_and_cancel();
  test_pump_and_guards();
  test_tables();
  test_mutation_fuzz();
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
