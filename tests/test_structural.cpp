// Tests for structural compression (structural/): the Phase 1 tokenizer and
// token image, the frozen global model, the token-level codec through the
// parallel pipelines, and two-pass archives over every aio backend.
#include "aio/async_io.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"
#include "io/stream.hpp"
#include "structural/archive.hpp"
#include "structural/global_model.hpp"
#include "structural/structural_cm_codec.hpp"
#include "structural/structural_parser.hpp"
#include "structural/token_stream.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace aio = hce::aio;
namespace st = hce::structural;
using hc::Bytes;
using hc::Errc;
using hc::u32;
using hc::u64;
using st::TokenType;
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

static Bytes bytes_of(const std::vector<std::uint8_t>& v) { return {reinterpret_cast<const std::byte*>(v.data()), v.size()}; }

static std::string read_text(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}
static void write_text(const fs::path& p, const std::string& s) {
  std::ofstream f(p, std::ios::binary);
  f.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// Real source code: this repository's C++ files, in a fixed order.
static std::string corpus() {
  std::vector<fs::path> files;
  for (const char* dir : {"codec", "encoder", "decoder", "aio", "structural", "cli", "tests", "io"})
    for (const auto& e : fs::directory_iterator(fs::path(HCE_SOURCE_DIR) / dir))
      if (e.path().extension() == ".cpp" || e.path().extension() == ".hpp") files.push_back(e.path());
  files.push_back(fs::path(HCE_SOURCE_DIR) / "hc_framed.hpp");
  std::sort(files.begin(), files.end());
  std::string s;
  for (const fs::path& f : files) s += read_text(f);
  return s;
}

static std::string random_text(std::size_t n, std::uint32_t seed, bool code_like) {
  static constexpr char kCode[] = "  \n\tabcxyz_019{}()[];,.=+-*/<>!&|\"'`#\\\r";
  std::mt19937 rng(seed);
  std::string s(n, '\0');
  for (char& c : s)
    c = code_like ? kCode[rng() % (sizeof kCode - 1)] : static_cast<char>(static_cast<unsigned char>(rng()));
  return s;
}

static Vec ids_of(const st::GlobalModel& m, const std::vector<st::StructuralToken>& tokens) {
  Vec v(tokens.size() * 4);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const u32 id = m.find(tokens[i].type, tokens[i].value);
    for (int b = 0; b < 4; ++b) v[4 * i + static_cast<std::size_t>(b)] = static_cast<std::byte>(id >> (8 * b));
  }
  return v;
}

// --------------------------------------------------------------- Phase 1 ---

static void test_tokenizer() {
  st::StructuralPreprocessor pre;
  const std::string src =
      "int main() {\n"
      "    x <<= 2; // shift\n"
      "    return f(\"hi\\\"\", 'c', 1.5e-3);\r\n"
      "}\n"
      "/* multi\nline */ def g(): pass\n";
  const auto t = pre.tokenize(src);
  CHECK(st::detokenize(t) == src);
  const auto has = [&t](TokenType type, const char* v) {
    return std::any_of(t.begin(), t.end(), [&](const st::StructuralToken& k) { return k.type == type && k.value == v; });
  };
  CHECK(has(TokenType::KEYWORD, "int") && has(TokenType::KEYWORD, "return") && has(TokenType::KEYWORD, "def"));
  CHECK(has(TokenType::IDENTIFIER, "main") && has(TokenType::IDENTIFIER, "x"));
  CHECK(has(TokenType::STRUCTURE, "{") && has(TokenType::STRUCTURE, "\n") && has(TokenType::STRUCTURE, "\r\n"));
  CHECK(has(TokenType::INDENTATION, "    ") && has(TokenType::OPERATOR, "<<="));
  CHECK(has(TokenType::LITERAL, "\"hi\\\"\"") && has(TokenType::LITERAL, "'c'") && has(TokenType::LITERAL, "1.5e-3"));
  CHECK(has(TokenType::LITERAL, "// shift") && has(TokenType::LITERAL, "/* multi\nline */"));
  // Line numbers and interned IDs.
  const auto ret = std::find_if(t.begin(), t.end(), [](const auto& k) { return k.value == "return"; });
  CHECK(ret != t.end() && ret->line_number == 3);
  const auto def = std::find_if(t.begin(), t.end(), [](const auto& k) { return k.value == "def"; });
  CHECK(def != t.end() && def->line_number == 6);
  for (const auto& a : t)
    for (const auto& b : t)
      CHECK((a.type == b.type && a.value == b.value) == (a.checked_integer_id == b.checked_integer_id));

  // Total and lossless on anything, with bounded tokens.
  for (std::uint32_t seed = 0; seed < 300; ++seed) {
    const std::string s = random_text(seed * 37 % 6000, seed, seed % 3 != 0);
    const auto k = pre.tokenize(s);
    bool ok = st::detokenize(k) == s;
    for (const auto& x : k)
      ok = ok && !x.value.empty() && x.value.size() <= st::kMaxTokenBytes && static_cast<int>(x.type) < 6;
    CHECK(ok);
  }
  const std::string long_lit = "s = \"" + std::string(10000, 'a') + "\";\n/*" + std::string(9000, '*');
  const auto k = pre.tokenize(long_lit);
  CHECK(st::detokenize(k) == long_lit);
  CHECK(std::all_of(k.begin(), k.end(), [](const auto& x) { return x.value.size() <= st::kMaxTokenBytes; }));
  CHECK(pre.tokenize("").empty());
}

static void test_token_image() {
  st::StructuralPreprocessor pre;
  for (std::uint32_t seed = 0; seed < 50; ++seed) {
    const auto t = pre.tokenize(random_text(seed * 101 % 3000, seed + 7, true));
    const auto image = pre.serialize_tokens(t);
    std::vector<st::StructuralToken> back;
    CHECK(st::deserialize_tokens(bytes_of(image), back) == Errc::ok);
    bool same = back.size() == t.size();
    for (std::size_t i = 0; same && i < t.size(); ++i)
      same = back[i].type == t[i].type && back[i].value == t[i].value && back[i].line_number == t[i].line_number &&
             back[i].checked_integer_id == t[i].checked_integer_id;
    CHECK(same);
  }
  // Hostile images: every truncation fails cleanly; flips either parse or fail, never crash.
  const auto image = pre.serialize_tokens(pre.tokenize("if (a) { b = \"x\"; }\n  return 0;\n"));
  std::vector<st::StructuralToken> out;
  for (std::size_t n = 0; n < image.size(); ++n)
    CHECK(st::deserialize_tokens(bytes_of(image).first(n), out) != Errc::ok && out.empty());
  std::mt19937 rng(5);
  for (int i = 0; i < 2000; ++i) {
    auto bad = image;
    bad[rng() % bad.size()] ^= static_cast<std::uint8_t>(1u << (rng() % 8));
    const Errc e = st::deserialize_tokens(bytes_of(bad), out);
    CHECK(e == Errc::ok || out.empty());
  }
  st::TokenStreamLimits small;
  small.max_tokens = 3;
  CHECK(st::deserialize_tokens(bytes_of(image), out, small) == Errc::limit_output);
  small = {};
  small.max_token_bytes = 2;
  CHECK(st::deserialize_tokens(bytes_of(image), out, small) == Errc::limit_output);
  auto magic = image;
  magic[0] = 'X';
  CHECK(st::deserialize_tokens(bytes_of(magic), out) == Errc::bad_magic);
}

static void test_tracker() {
  st::StructuralPreprocessor pre;
  st::StructuralTracker tr;
  std::vector<st::StructuralContextState> after;
  for (const auto& t : pre.tokenize("{\n    f(a[1]);\n}\n")) {
    tr.advance(t.type, t.value);
    after.push_back(tr.state);
  }
  // 0 {  1 \n  2 "    "  3 f  4 (  5 a  6 [  7 1  8 ]  9 )  10 ;  11 \n  12 }  13 \n
  CHECK(after.size() == 14);
  if (after.size() == 14) {
    CHECK(after[0].scope_depth == 1 && after[2].indentation_level == 4 && tr.at_line_start);
    CHECK(after[4].block_brackets_count == 1 && after[6].block_brackets_count == 2);
    CHECK(after[9].block_brackets_count == 0 && after[11].indentation_level == 0 && after[12].scope_depth == 0);
  }
}

// ------------------------------------------------------- global model ---

static void test_global_model() {
  st::StructuralPreprocessor pre;
  st::GlobalModelBuilder b;
  CHECK(b.add(pre.tokenize("a b a b a b a c ")) == Errc::ok);
  st::GlobalModel m;
  CHECK(b.freeze(m) == Errc::ok);
  // IDs in order of first appearance: a, " ", b, c.
  CHECK(m.size() == 4 && m.value(0) == "a" && m.value(1) == " " && m.value(2) == "b" && m.value(3) == "c");
  CHECK(m.find(TokenType::IDENTIFIER, "a") == 0 && m.find(TokenType::IDENTIFIER, "zz") == st::kNoSymbol);
  CHECK(m.successor(0) == 1 && m.successor(2) == 1);  // a -> " " and b -> " " every time
  CHECK(m.successor(1) == 0 || m.successor(1) == 2);  // " " -> a or b (3 each), never c (once)
  CHECK(m.successor(3) == st::kNoSymbol);             // c -> " " once: not worth storing

  // Real code: deterministic, and the image round-trips exactly.
  const std::string code = corpus();
  const auto tokens = pre.tokenize(code);
  st::GlobalModel m1, m2, parsed;
  st::GlobalModelBuilder b1, b2;
  CHECK(b1.add(tokens) == Errc::ok && b1.freeze(m1) == Errc::ok);
  CHECK(b2.add(tokens) == Errc::ok && b2.freeze(m2) == Errc::ok);
  const auto image = m1.serialize();
  CHECK(image == m2.serialize());
  CHECK(st::GlobalModel::parse(bytes_of(image), {}, parsed) == Errc::ok);
  CHECK(parsed.serialize() == image && parsed.size() == m1.size());
  bool same = true;
  for (u32 id = 0; id < m1.size() && same; ++id)
    same = parsed.value(id) == m1.value(id) && parsed.type(id) == m1.type(id) &&
           parsed.successor(id) == m1.successor(id) && parsed.traits(id) == m1.traits(id);
  CHECK(same);
  std::printf("  global model: %zu tokens -> %u symbols, %llu dictionary bytes\n", tokens.size(), m1.size(),
              static_cast<unsigned long long>(m1.dictionary_bytes()));

  // Hostile images.
  const auto small = [&] {
    st::GlobalModel s;
    st::GlobalModelBuilder sb;
    (void)sb.add(pre.tokenize("x = y + 1;\n"));
    (void)sb.freeze(s);
    return s.serialize();
  }();
  for (std::size_t n = 0; n < small.size(); ++n) CHECK(st::GlobalModel::parse(bytes_of(small).first(n), {}, parsed) != Errc::ok);
  std::mt19937 rng(9);
  for (int i = 0; i < 2000; ++i) {
    auto bad = small;
    bad[rng() % bad.size()] ^= static_cast<std::uint8_t>(1u << (rng() % 8));
    st::GlobalModel g;
    if (st::GlobalModel::parse(bytes_of(bad), {}, g) == Errc::ok)
      for (u32 id = 0; id < g.size(); ++id) CHECK(g.successor(id) == st::kNoSymbol || g.successor(id) < g.size());
  }
  st::GlobalModelLimits lim;
  lim.max_symbols = 3;
  CHECK(st::GlobalModel::parse(bytes_of(small), lim, parsed) == Errc::limit_output);
  st::GlobalModelBuilder limited(lim);
  CHECK(limited.add(pre.tokenize("a b c d")) == Errc::limit_output);
}

// ------------------------------------------------------- codec, in memory ---

struct Encoded {
  Errc status;
  Vec data;
};
static Encoded encode_ids(const Vec& ids, const st::GlobalModel* m, unsigned threads, u64 frame_tokens) {
  hce::MemoryInput in(ids);
  hce::MemoryOutput out;
  hce::EncodeOptions o;
  o.level = 1;
  o.window_log = 16;
  o.block_log = 16;
  o.frame_bytes = frame_tokens * 4;
  o.threads = threads;
  o.content_size = ids.size();
  o.codec_context = m;
  const auto r = hce::encode_parallel<hce::StructuralCMCodec>(in, out, o);
  return {r.status, std::move(out.data)};
}
static Encoded decode_ids(const Vec& stream, const st::GlobalModel* m, unsigned threads) {
  hce::MemoryInput in(stream);
  hce::MemoryOutput out;
  hce::DecodeOptions o;
  o.threads = threads;
  o.limits.max_window_log = 16;
  o.limits.max_block_bytes = 1u << 16;
  o.codec_context = m;
  const auto r = hce::decode_stream<hce::StructuralCMCodec>(in, out, o);
  return {r.status, std::move(out.data)};
}

static void test_codec() {
  st::StructuralPreprocessor pre;
  const auto tokens = pre.tokenize(corpus());
  st::GlobalModelBuilder b;
  st::GlobalModel m;
  CHECK(b.add(tokens) == Errc::ok && b.freeze(m) == Errc::ok);
  const Vec ids = ids_of(m, tokens);

  // Several frames, so workers really run in parallel and share the model.
  const Encoded e1 = encode_ids(ids, &m, 1, 16384);
  const Encoded e4 = encode_ids(ids, &m, 4, 16384);
  CHECK(e1.status == Errc::ok && e4.status == Errc::ok);
  CHECK(e1.data == e4.data);  // byte-identical for every thread count
  CHECK(e1.data.size() < ids.size() / 3);
  for (const unsigned t : {1u, 3u}) {
    const Encoded d = decode_ids(e1.data, &m, t);
    CHECK(d.status == Errc::ok && d.data == ids);
  }
  std::printf("  codec: %zu tokens -> %zu bytes (%.3f bytes/token)\n", tokens.size(), e1.data.size(),
              static_cast<double>(e1.data.size()) / static_cast<double>(tokens.size()));

  // R15: every SIMD path codes the same bits, so archives move between machines.
  for (const char* enc_path : {"scalar", "sse2", "avx2", "neon"}) {
    if (!hce::StructuralCMCodec::test_select_simd(enc_path)) continue;
    const Encoded e = encode_ids(ids, &m, 2, 16384);
    CHECK(e.status == Errc::ok && e.data == e1.data);
    for (const char* dec_path : {"scalar", "sse2", "avx2", "neon"}) {
      if (!hce::StructuralCMCodec::test_select_simd(dec_path)) continue;
      CHECK(decode_ids(e.data, &m, 2).data == ids);
    }
  }
  hce::StructuralCMCodec::test_select_simd(nullptr);

  // No model: the decoder refuses; the encoder stores everything and still round-trips with one.
  CHECK(decode_ids(e1.data, nullptr, 1).status == Errc::unsupported_feature);
  const Encoded stored = encode_ids(ids, nullptr, 2, 16384);
  CHECK(stored.status == Errc::ok && stored.data.size() > ids.size());
  CHECK(decode_ids(stored.data, &m, 2).data == ids);

  // Damage: decoding fails cleanly, never crashes.
  std::mt19937 rng(11);
  for (int i = 0; i < 60; ++i) {
    Vec bad = e1.data;
    const std::size_t at = 64 + rng() % (bad.size() - 64);
    bad[at] ^= static_cast<std::byte>(1u << (rng() % 8));
    const Encoded d = decode_ids(bad, &m, 1 + static_cast<unsigned>(i % 3));
    CHECK(d.status != Errc::ok || d.data == ids);
  }
  // IDs outside the model are stored by the encoder, not coded.
  Vec junk(4096);
  for (std::size_t i = 0; i < junk.size(); ++i) junk[i] = static_cast<std::byte>(rng());
  const Encoded ej = encode_ids(junk, &m, 1, 16384);
  CHECK(ej.status == Errc::ok && decode_ids(ej.data, &m, 1).data == junk);
}

// ------------------------------------------------------------- archives ---

static std::vector<aio::Backend> backends() {
  std::vector<aio::Backend> b{aio::Backend::threads};
  if (aio::backend_available(aio::Backend::io_uring)) b.push_back(aio::Backend::io_uring);
  return b;
}

static void test_archives() {
  const fs::path src = g_dir / "corpus.txt";
  const std::string code = corpus();
  write_text(src, code);
  Vec reference;
  for (const aio::Backend backend : backends()) {
    for (const unsigned threads : {1u, 4u}) {
      st::ArchiveOptions o;
      o.level = 1;
      o.threads = threads;
      o.frame_tokens = 16384;
      o.chunk_bytes = 64 * 1024;  // many tokenize() calls and spool records
      o.io.backend = backend;
      o.io.direct = threads == 4;
      const fs::path arc = g_dir / "corpus.hcs", back = g_dir / "corpus.out";
      const st::ArchiveStats c = st::compress_file(src, arc, o);
      CHECK(c.status == Errc::ok);
      CHECK(c.source_bytes == code.size() && c.archive_bytes == fs::file_size(arc));
      CHECK(!fs::exists(fs::path(arc) += ".spool"));
      const std::string a = read_text(arc);
      const Vec av(reinterpret_cast<const std::byte*>(a.data()), reinterpret_cast<const std::byte*>(a.data()) + a.size());
      if (reference.empty()) reference = av;
      CHECK(av == reference);  // identical for every backend and thread count
      const st::ArchiveStats d = st::decompress_file(arc, back, o);
      CHECK(d.status == Errc::ok && read_text(back) == code);
      if (threads == 4 && backend == backends().back())
        std::printf("  archive (%s): %zu bytes -> %llu (model %llu, tokens %llu) in %.2f + %.2f + %.2f s\n",
                    aio::to_string(backend), code.size(), static_cast<unsigned long long>(c.archive_bytes),
                    static_cast<unsigned long long>(c.model_bytes), static_cast<unsigned long long>(c.stream_bytes),
                    c.seconds[0], c.seconds[1], c.seconds[2]);
    }
  }

  // Edge inputs: empty, binary, no newline at all.
  st::ArchiveOptions o;
  o.frame_tokens = 16384;
  o.chunk_bytes = 4096;
  for (const std::string& s : {std::string(), random_text(70000, 3, false), std::string(20000, 'x')}) {
    write_text(src, s);
    const fs::path arc = g_dir / "edge.hcs", back = g_dir / "edge.out";
    CHECK(st::compress_file(src, arc, o).status == Errc::ok);
    CHECK(st::decompress_file(arc, back, o).status == Errc::ok && read_text(back) == s);
  }

  // Damaged archives fail and leave no output behind.
  write_text(src, code);
  const fs::path arc = g_dir / "dmg.hcs", back = g_dir / "dmg.out";
  CHECK(st::compress_file(src, arc, o).status == Errc::ok);
  const std::string good = read_text(arc);
  std::mt19937 rng(13);
  for (int i = 0; i < 24; ++i) {
    std::string bad = good;
    if (i % 4 == 3) {
      bad.resize(rng() % bad.size());  // truncated
    } else {
      const std::size_t at = i % 4 == 0 ? rng() % st::kArchiveHeaderSize : rng() % bad.size();
      bad[at] = static_cast<char>(bad[at] ^ static_cast<char>(1 << (rng() % 8)));
    }
    write_text(arc, bad);
    fs::remove(back);
    const st::ArchiveStats d = st::decompress_file(arc, back, o);
    CHECK(d.status != Errc::ok || read_text(back) == code);
    if (d.status != Errc::ok) CHECK(!fs::exists(back));
  }
#if !defined(_WIN32)
  CHECK(st::compress_file("/dev/zero", g_dir / "z.hcs", o).status == Errc::unsupported_feature);
#endif
  CHECK(st::compress_file(g_dir / "missing.txt", g_dir / "m.hcs", o).status == Errc::io_error);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::random_device rd;
  g_dir = fs::current_path() / ("structural-test-" + std::to_string(rd()));
  fs::create_directories(g_dir);

  test_tokenizer();
  test_token_image();
  test_tracker();
  std::printf("  tokenizer, token image and tracker ok\n");
  test_global_model();
  test_codec();
  test_archives();

  if (g_failures) {
    std::printf("%d check(s) failed (files kept in %s)\n", g_failures, g_dir.string().c_str());
    return 1;
  }
  fs::remove_all(g_dir);
  std::printf("all structural tests passed\n");
  return 0;
}
