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
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace aio = hce::aio;

// ThreadSanitizer: its shadow memory inflates RSS, and it cannot see liburing's rings.
#if defined(__SANITIZE_THREAD__)
#define HCE_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define HCE_TEST_TSAN 1
#endif
#endif

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

// Like a file descriptor: a read that returns 0 (even of an empty buffer) is the end, for good.
class LatchingInput final : public hce::InputStream {
 public:
  explicit LatchingInput(Bytes data) noexcept : data_(data) {}
  std::size_t read(hc::MutBytes buf) override {
    if (eof_) return 0;
    const std::size_t n = std::min({buf.size(), data_.size() - pos_, std::size_t{4093}});
    std::memcpy(buf.data(), data_.data() + pos_, n);
    pos_ += n;
    eof_ = n == 0;
    return n;
  }

 private:
  Bytes data_;
  std::size_t pos_ = 0;
  bool eof_ = false;
};

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

static void test_languages() {
  CHECK(st::parse_language("py") == st::Language::python && st::parse_language("c++") == st::Language::cpp);
  CHECK(st::parse_language("ts") == st::Language::typescript && !st::parse_language("cobol"));
  CHECK(st::language_for_path("a/b.PY") == st::Language::python && st::language_for_path("x.hpp") == st::Language::cpp);
  CHECK(st::language_for_path("x.rs") == st::Language::rust && st::language_for_path("vector") == st::Language::generic);
  for (std::uint8_t l = 0; l < st::kLanguages; ++l) {
    const auto lang = static_cast<st::Language>(l);
    CHECK(st::parse_language(st::to_string(lang)) == lang);
  }
  // The table decides KEYWORD versus IDENTIFIER; every table is lossless.
  const std::string src = "def f(self): return function(x) if x else nil\n";
  const auto kind = [&](st::Language lang, const char* word) {
    for (const auto& t : st::tokenize(src, lang))
      if (t.value == word) return t.type;
    return TokenType::OPERATOR;
  };
  CHECK(kind(st::Language::python, "def") == TokenType::KEYWORD && kind(st::Language::cpp, "def") == TokenType::IDENTIFIER);
  CHECK(kind(st::Language::javascript, "function") == TokenType::KEYWORD && kind(st::Language::python, "function") == TokenType::IDENTIFIER);
  CHECK(kind(st::Language::go, "nil") == TokenType::KEYWORD && kind(st::Language::rust, "nil") == TokenType::IDENTIFIER);
  for (std::uint8_t l = 0; l < st::kLanguages; ++l)
    for (std::uint32_t seed = 0; seed < 20; ++seed) {
      const std::string text = random_text(2000, seed, true);
      CHECK(st::detokenize(st::tokenize(text, static_cast<st::Language>(l))) == text);
    }
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
  for (const char* enc_path : {"scalar", "sse2", "avx2", "avx512", "neon"}) {
    if (!hce::StructuralCMCodec::test_select_simd(enc_path)) continue;
    const Encoded e = encode_ids(ids, &m, 2, 16384);
    CHECK(e.status == Errc::ok && e.data == e1.data);
    for (const char* dec_path : {"scalar", "sse2", "avx2", "avx512", "neon"}) {
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

  // Streams: memory in, memory out; the language is recorded; cancellation stops every stage.
  {
    const std::string py = "def f(x):\n    return x + 1\n\nprint(f(2))\n";
    hce::MemoryInput in(Bytes(reinterpret_cast<const std::byte*>(py.data()), py.size()));
    hce::MemoryOutput arc;
    st::ArchiveOptions so = o;
    so.language = st::Language::python;
    const st::ArchiveStats c = st::compress_stream(in, arc, so);
    CHECK(c.status == Errc::ok && c.archive_bytes == arc.data.size() && st::is_archive(arc.data));
    LatchingInput ain(arc.data);  // the model section ends exactly where the token stream starts
    hce::MemoryOutput back;
    const st::ArchiveStats d = st::decompress_stream(ain, back, o);
    CHECK(d.status == Errc::ok && d.language == st::Language::python);
    CHECK(std::string(reinterpret_cast<const char*>(back.data.data()), back.data.size()) == py);
    std::stop_source stop;
    stop.request_stop();
    st::ArchiveOptions co = o;
    co.stop = stop.get_token();
    hce::MemoryInput in2(Bytes(reinterpret_cast<const std::byte*>(py.data()), py.size()));
    hce::MemoryOutput arc2;
    CHECK(st::compress_stream(in2, arc2, co).status == Errc::cancelled);
    hce::MemoryInput ain2(arc.data);
    hce::MemoryOutput back2;
    CHECK(st::decompress_stream(ain2, back2, co).status == Errc::cancelled);
    CHECK(!st::is_archive(Bytes(arc.data).first(3)));
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

// Files in `dir` whose names start with `prefix`: temporary and spool files left behind.
static int leftovers(const fs::path& dir, const std::string& prefix) {
  int n = 0;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().filename().string().rfind(prefix, 0) == 0) ++n;
  return n;
}

// Transactional file output: a destination changes only when the new content
// is complete and verified; failures, cancellation and aliases leave it alone.
static void test_file_safety() {
  const fs::path dir = g_dir / "safety";
  fs::create_directories(dir);
  const fs::path src = dir / "src.py", arc = dir / "src.hcs", dst = dir / "dst.txt";
  const std::string code = corpus().substr(0, 200000);
  write_text(src, code);
  st::ArchiveOptions o;
  o.frame_tokens = 16384;
  CHECK(st::compress_file(src, arc, o).status == Errc::ok);
  const std::string good = read_text(arc);

  // A corrupt archive over an existing destination: the destination survives.
  std::string bad = good;
  bad[bad.size() / 2] = static_cast<char>(bad[bad.size() / 2] ^ 0x10);
  write_text(dir / "bad.hcs", bad);
  write_text(dst, "PRECIOUS");
  st::ArchiveStats d = st::decompress_file(dir / "bad.hcs", dst, o);
  CHECK(d.status != Errc::ok && read_text(dst) == "PRECIOUS");
  // Not an archive at all, and a truncated one: the same.
  CHECK(st::decompress_file(src, dst, o).status == Errc::bad_magic && read_text(dst) == "PRECIOUS");
  write_text(dir / "short.hcs", good.substr(0, good.size() - 7));
  CHECK(st::decompress_file(dir / "short.hcs", dst, o).status != Errc::ok && read_text(dst) == "PRECIOUS");

  // Cancellation, compressing and decompressing, over existing files.
  std::stop_source stop;
  stop.request_stop();
  st::ArchiveOptions co = o;
  co.stop = stop.get_token();
  write_text(dir / "old.hcs", "OLD ARCHIVE");
  CHECK(st::compress_file(src, dir / "old.hcs", co).status == Errc::cancelled && read_text(dir / "old.hcs") == "OLD ARCHIVE");
  CHECK(st::decompress_file(arc, dst, co).status == Errc::cancelled && read_text(dst) == "PRECIOUS");

  // Success replaces the destination; nothing temporary is left anywhere.
  CHECK(st::decompress_file(arc, dst, o).status == Errc::ok && read_text(dst) == code);
  CHECK(leftovers(dir, ".") == 0);

  // Aliases: the same path, a hard link, a symbolic link. Refused before anything is written.
  CHECK(st::compress_file(src, src, o).stage == std::string("alias") && read_text(src) == code);
  CHECK(st::decompress_file(arc, arc, o).stage == std::string("alias") && read_text(arc) == good);
  std::error_code ec;
  fs::create_hard_link(arc, dir / "arc-link.hcs", ec);
  if (!ec) {
    CHECK(st::decompress_file(arc, dir / "arc-link.hcs", o).stage == std::string("alias") && read_text(arc) == good);
    fs::remove(dir / "arc-link.hcs");
  }
#if !defined(_WIN32)
  fs::create_symlink(src.filename(), dir / "src-link.py");
  CHECK(st::compress_file(src, dir / "src-link.py", o).stage == std::string("alias") && read_text(src) == code);
  // A symbolic link to another file is replaced, never written through.
  write_text(dir / "target.txt", "TARGET");
  fs::remove(dir / "src-link.py");
  fs::create_symlink("target.txt", dir / "out-link.txt");
  CHECK(st::decompress_file(arc, dir / "out-link.txt", o).status == Errc::ok);
  CHECK(!fs::is_symlink(dir / "out-link.txt") && read_text(dir / "out-link.txt") == code);
  CHECK(read_text(dir / "target.txt") == "TARGET");
  // A replaced file keeps its permission bits.
  ::chmod(dst.c_str(), 0640);
  CHECK(st::decompress_file(arc, dst, o).status == Errc::ok);
  struct stat sb {};
  CHECK(::stat(dst.c_str(), &sb) == 0 && (sb.st_mode & 0777) == 0640);

  // A write that fails (the file size limit) leaves the old destination.
  write_text(dst, "PRECIOUS");
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::signal(SIGXFSZ, SIG_IGN);
    const rlimit lim{4096, 4096};
    ::setrlimit(RLIMIT_FSIZE, &lim);
    const st::ArchiveStats r = st::decompress_file(arc, dst, o);
    ::_exit(r.status == Errc::io_error ? 0 : 1);
  }
  int status = -1;
  ::waitpid(pid, &status, 0);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  CHECK(read_text(dst) == "PRECIOUS");
#endif
  CHECK(leftovers(dir, ".") == 0);
  // A directory as the destination: refused, untouched.
  fs::create_directories(dir / "sub");
  CHECK(st::decompress_file(arc, dir / "sub", o).status == Errc::io_error && fs::is_directory(dir / "sub"));
  fs::remove_all(dir);
}

// The token spool: always a new owner-only file, never shared, never through
// a link, removed only when this run created it.
static void test_spool_safety() {
  const fs::path dir = g_dir / "spool";
  fs::create_directories(dir);
  const fs::path src = dir / "src.c", arc = dir / "src.hcs";
  const std::string code = corpus().substr(0, 120000);
  write_text(src, code);
  st::ArchiveOptions o;
  o.frame_tokens = 16384;

  // An explicit spool path that exists is refused and kept as it was.
  write_text(dir / "taken.spool", "MINE");
  o.spool = dir / "taken.spool";
  st::ArchiveStats c = st::compress_file(src, arc, o);
  CHECK(c.status == Errc::io_error && std::string(c.stage) == "spool");
  CHECK(read_text(dir / "taken.spool") == "MINE" && !fs::exists(arc));
#if !defined(_WIN32)
  // The same through a symbolic link, dangling or not: the target is untouched.
  fs::create_symlink("victim.txt", dir / "link.spool");
  write_text(dir / "victim.txt", "VICTIM");
  o.spool = dir / "link.spool";
  CHECK(st::compress_file(src, arc, o).status == Errc::io_error && read_text(dir / "victim.txt") == "VICTIM");
  CHECK(fs::is_symlink(dir / "link.spool"));
  fs::remove(dir / "victim.txt");
  CHECK(st::compress_file(src, arc, o).status == Errc::io_error && !fs::exists(dir / "victim.txt"));
  fs::remove(dir / "link.spool");
  // The old default name, <archive>.spool, is no longer used: a file there stays.
  o.spool.clear();
  fs::create_symlink("victim.txt", fs::path(arc) += ".spool");
  write_text(dir / "victim.txt", "VICTIM");
  CHECK(st::compress_file(src, arc, o).status == Errc::ok && read_text(dir / "victim.txt") == "VICTIM");
  CHECK(fs::is_symlink(fs::path(arc) += ".spool"));
  fs::remove(fs::path(arc) += ".spool");
  // write_token_spool() makes it owner-only.
  hc::u32 crc = 0;
  CHECK(st::write_token_spool(src, dir / "own.spool", o, crc).status == Errc::ok);
  struct stat sb {};
  CHECK(::stat((dir / "own.spool").c_str(), &sb) == 0 && (sb.st_mode & 0777) == 0600);
  // Pass 1 refuses a spool that is not the file it was told about.
  st::GlobalModel m;
  const aio::FileId other{1, 2};
  CHECK(st::build_global_model(dir / "own.spool", o, m, nullptr, &other) == Errc::content_mismatch);
  fs::remove(dir / "own.spool");
#endif
  o.spool.clear();
  const std::string reference = (st::compress_file(src, arc, o).status == Errc::ok) ? read_text(arc) : "";
  CHECK(!reference.empty());

  // Concurrent runs sharing a spool directory each get their own spool.
  o.spool_dir = dir;
#ifdef HCE_TEST_TSAN
  // liburing maps its rings with raw syscalls TSan does not intercept, so a ring
  // that reuses another thread's freed ring address reads as a race. This test is
  // about spool names; the threads backend gives TSan full sight of it.
  const aio::Backend saved_backend = o.io.backend;
  o.io.backend = aio::Backend::threads;
#endif
  std::vector<std::thread> runs;
  std::vector<st::ArchiveStats> results(4);
  for (int i = 0; i < 4; ++i)
    runs.emplace_back([&, i] { results[static_cast<std::size_t>(i)] = st::compress_file(src, dir / ("c" + std::to_string(i) + ".hcs"), o); });
  for (auto& t : runs) t.join();
  for (int i = 0; i < 4; ++i) {
    CHECK(results[static_cast<std::size_t>(i)].status == Errc::ok);
    CHECK(read_text(dir / ("c" + std::to_string(i) + ".hcs")) == reference);
  }
#ifdef HCE_TEST_TSAN
  o.io.backend = saved_backend;
#endif
  // Failure and cancellation remove the spool they made, and nothing else.
  std::stop_source stop;
  stop.request_stop();
  st::ArchiveOptions co = o;
  co.stop = stop.get_token();
  CHECK(st::compress_file(src, dir / "cancel.hcs", co).status == Errc::cancelled);
  CHECK(st::compress_file(dir / "missing.c", dir / "missing.hcs", o).status == Errc::io_error);
  CHECK(leftovers(dir, ".hce-spool-") == 0 && leftovers(dir, ".") == 0 && read_text(dir / "taken.spool") == "MINE");
  fs::remove_all(dir);
}

// Decoder limits: the source size against max_output before any decoding;
// the parsed dictionary and the output stage against max_memory.
static void test_decode_limits() {
  for (const std::string& s : {corpus().substr(0, 100000), std::string()}) {
    hce::MemoryInput in(Bytes(reinterpret_cast<const std::byte*>(s.data()), s.size()));
    hce::MemoryOutput arc;
    st::ArchiveOptions o;
    o.frame_tokens = 16384;
    CHECK(st::compress_stream(in, arc, o).status == Errc::ok);
    for (const u64 limit : {u64{s.size()}, u64{s.size()} + 1, u64{s.size()} - 1}) {
      if (s.empty() && limit != 0) continue;
      st::ArchiveOptions d = o;
      d.max_output = limit;
      hce::MemoryInput ain(arc.data);
      hce::MemoryOutput back;
      const st::ArchiveStats r = st::decompress_stream(ain, back, d);
      if (limit >= s.size()) {
        CHECK(r.status == Errc::ok && back.data.size() == s.size());
      } else {
        CHECK(r.status == Errc::limit_output && std::string(r.stage) == "header" && back.data.empty());
      }
    }
  }

  // A large dictionary: 300,000 distinct identifiers, about 3.6 MB of values.
  std::string big;
  for (u32 i = 0; i < 300000; ++i) big += "identifier_" + std::to_string(i * 2654435761u) + " = " + std::to_string(i) + ";\n";
  hce::MemoryInput in(Bytes(reinterpret_cast<const std::byte*>(big.data()), big.size()));
  hce::MemoryOutput arc;
  st::ArchiveOptions o;
  o.level = 1;
  o.threads = 4;
  const st::ArchiveStats c = st::compress_stream(in, arc, o);
  CHECK(c.status == Errc::ok && c.symbols > 300000);
  // A budget the retained model does not fit in fails before writing a byte;
  // the old decoder gave each phase the whole budget and accepted it.
  st::ArchiveOptions d = o;
  d.max_memory = c.dictionary_bytes + (u64{1} << 20);
  hce::MemoryInput ain(arc.data);
  hce::MemoryOutput back;
  st::ArchiveStats r = st::decompress_stream(ain, back, d);
  CHECK(r.status == Errc::limit_memory && back.data.empty());
  d.max_memory = u64{1} << 30;
  hce::MemoryInput ain2(arc.data);
  r = st::decompress_stream(ain2, back, d);
  CHECK(r.status == Errc::ok && back.data.size() == big.size());
#if defined(__linux__) && !defined(HCE_TEST_TSAN)  // TSan doubles RSS; release and ASan builds check this
  // Peak memory of a 4-worker decode stays within its budget (plus the
  // archive and output buffers this test holds, and allocator slack).
  const u64 budget = u64{96} << 20;
  const pid_t pid = ::fork();
  if (pid == 0) {
    rusage before {}, after {};
    ::getrusage(RUSAGE_SELF, &before);
    st::ArchiveOptions b = o;
    b.max_memory = budget;
    hce::MemoryInput bin(arc.data);
    hce::NullOutput sink;
    const st::ArchiveStats br = st::decompress_stream(bin, sink, b);
    ::getrusage(RUSAGE_SELF, &after);
    const u64 grew = static_cast<u64>(after.ru_maxrss - before.ru_maxrss) << 10;
    std::printf("  4-worker decode, %llu symbols, budget %llu MiB: %s, peak grew %.1f MiB\n",
                static_cast<unsigned long long>(c.symbols), static_cast<unsigned long long>(budget >> 20),
                hc::to_string(br.status), static_cast<double>(grew) / (1 << 20));
    ::_exit(br.status == Errc::ok && grew <= budget + (u64{16} << 20) ? 0 : 1);
  }
  int status = -1;
  ::waitpid(pid, &status, 0);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#endif
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::random_device rd;
  g_dir = fs::current_path() / ("structural-test-" + std::to_string(rd()));
  fs::create_directories(g_dir);

  test_tokenizer();
  test_languages();
  test_token_image();
  test_tracker();
  std::printf("  tokenizer, token image and tracker ok\n");
  test_global_model();
  test_codec();
  test_archives();
  test_file_safety();
  test_spool_safety();
  test_decode_limits();
  std::printf("  file safety, spool safety and decoder limits ok\n");

  if (g_failures) {
    std::printf("%d check(s) failed (files kept in %s)\n", g_failures, g_dir.string().c_str());
    return 1;
  }
  fs::remove_all(g_dir);
  std::printf("all structural tests passed\n");
  return 0;
}
