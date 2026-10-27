// ultracodec test suite: round trips for every tier over edge-case inputs,
// thread-count independence, the rANS vector kernels against the scalar
// decoder, header/framing validation, corruption (truncation and random bit
// flips must throw ultracodec::Error or decode to the original, never crash;
// the Tier 2 word filter gets the same treatment on its own;
// run under ASan/UBSan in CI) and peak-RSS limits per tier (Linux, not under
// sanitizers).
//
// HCE_FUZZ_ITERS (or ULTRA_FUZZ_ITERS) scales the corruption iterations.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"
#include "rans_static.hpp"
#include "sws_model.hpp"
#include "word_filter.hpp"
#include "ultracodec/engine.hpp"

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#define ULTRA_TEST_RSS 1
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define ULTRA_SANITIZED 1
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#define ULTRA_SANITIZED 1
#endif
#endif

using namespace ultracodec;
using Bytes = std::vector<std::uint8_t>;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

int fuzz_iters(int dflt) {
  for (const char* name : {"ULTRA_FUZZ_ITERS", "HCE_FUZZ_ITERS"})
    if (const char* v = std::getenv(name)) return std::max(1, std::atoi(v));
  return dflt;
}

// Sanitizer / emulator runs set the iteration knob low; they also get
// smaller inputs.
bool slow_env() { return fuzz_iters(150) < 50; }

Bytes random_bytes(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Bytes b(n);
  for (auto& c : b) c = static_cast<std::uint8_t>(rng());
  return b;
}

// Pseudo-text: words from a small vocabulary, punctuation and newlines,
// with some long repeats so the match paths are exercised.
Bytes text_bytes(std::size_t n, std::uint32_t seed) {
  static const char* words[] = {"the", "engine", "context", "mixing", "block", "of", "and", "a", "stream",
                                "def", "return", "self", "value", "import", "class", "for", "in", "range"};
  std::mt19937 rng(seed);
  std::string s;
  while (s.size() < n) {
    if (rng() % 50 == 0 && s.size() > 400) {
      std::size_t from = rng() % (s.size() - 300);
      s.append(s, from, 100 + rng() % 200);
      continue;
    }
    s += words[rng() % (sizeof words / sizeof *words)];
    unsigned r = static_cast<unsigned>(rng() % 12);
    s += r == 0 ? ".\n" : r == 1 ? ", " : r == 2 ? "(x)\n    " : " ";
  }
  s.resize(n);
  return Bytes(s.begin(), s.end());
}

// text_bytes with capitalized, upper-case and mixed-case words, digits,
// identifiers, UTF-8 and the filter's own escape bytes mixed in.
Bytes cased_text_bytes(std::size_t n, std::uint32_t seed) {
  static const char* extra[] = {"The", "THE", "McDonald", "iPhone", "A", "I", "X1", "_init_", "__INIT__",
                                "foo_bar9", "caf\xc3\xa9", "\x01\x02\xff", "2024", "HTTPServer", "Return"};
  Bytes base = text_bytes(n, seed);
  std::mt19937 rng(seed);
  Bytes out;
  for (std::size_t i = 0; i < base.size(); ++i) {
    out.push_back(base[i]);
    if (base[i] == ' ' && rng() % 6 == 0) {
      const char* w = extra[rng() % (sizeof extra / sizeof *extra)];
      out.insert(out.end(), w, w + std::strlen(w));
      out.push_back(' ');
    }
  }
  out.insert(out.end(), 300, 'q');  // a word longer than 255 bytes
  out.resize(n);
  return out;
}

Bytes structured_bytes(std::size_t n) {
  Bytes b(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t v = static_cast<std::uint32_t>(i / 16);
    b[i] = (i % 16 < 4) ? static_cast<std::uint8_t>(v >> (8 * (i % 4))) : static_cast<std::uint8_t>(i % 16 == 5 ? 0xFF : 0);
  }
  return b;
}

EngineOptions small_options(TierLevel tier, unsigned threads) {
  EngineOptions o;
  o.threads = threads;
  o.block_size = 64 * 1024;
  if (tier == TierLevel::Tier3_Balanced) {
    o.memory_budget = std::size_t{96} << 20;
    o.prime_bytes = 8 * 1024;
  }
  // Modelling Tier 4's 256 KiB primer takes about a second per call (much
  // longer under sanitizers); test_tier4_primer covers the primed path.
  if (tier == TierLevel::Tier4_Enterprise) o.tier4_primer = false;
  return o;
}

const TierLevel kTiers[] = {TierLevel::Tier1_HyperSpeed, TierLevel::Tier2_Consumer, TierLevel::Tier3_Balanced,
                            TierLevel::Tier4_Enterprise};

bool is_cm(TierLevel t) { return t == TierLevel::Tier3_Balanced || t == TierLevel::Tier4_Enterprise; }

void test_header() {
  ArchiveHeader h;
  h.tier = TierLevel::Tier3_Balanced;
  h.flags = ArchiveHeader::kFlagWavefront;
  h.block_log2 = 20;
  h.uncompressed_size = (3u << 20) + 5;
  h.block_count = 4;
  h.prime_bytes = 65536;
  h.model_param = 77;
  std::uint8_t buf[ArchiveHeader::kSize];
  h.serialize(buf);
  ArchiveHeader p = ArchiveHeader::parse(buf);
  CHECK(p.tier == h.tier && p.block_count == 4 && p.uncompressed_size == h.uncompressed_size &&
        p.prime_bytes == 65536 && p.model_param == 77 && p.block_size() == (1u << 20));
  auto rejects = [&](std::function<void(std::uint8_t*)> mutate) {
    std::uint8_t b[ArchiveHeader::kSize];
    std::memcpy(b, buf, sizeof b);
    mutate(b);
    try {
      ArchiveHeader::parse(b);
      return false;
    } catch (const Error&) {
      return true;
    }
  };
  CHECK(rejects([](std::uint8_t* b) { b[0] = 'X'; }));
  CHECK(rejects([](std::uint8_t* b) { b[9] ^= 1; }));  // CRC
  CHECK(rejects([](std::uint8_t* b) {
    b[4] = 9;  // version, with a fixed-up CRC
    detail::store32(b + 28, detail::crc32c(b, 28));
  }));
  CHECK(rejects([](std::uint8_t* b) {
    detail::store32(b + 16, 5);  // block count inconsistent with size
    detail::store32(b + 28, detail::crc32c(b, 28));
  }));
  CHECK(rejects([](std::uint8_t* b) {
    b[6] |= ArchiveHeader::kFlagSnapshot;  // snapshot warm-start belongs to Tier 4 only
    detail::store32(b + 28, detail::crc32c(b, 28));
  }));
  CHECK(rejects([](std::uint8_t* b) {
    b[5] = 4;  // Tier 4 without kFlagSnapshot
    detail::store32(b + 28, detail::crc32c(b, 28));
  }));
  CHECK(rejects([](std::uint8_t*) {}) == false);
  try {
    ArchiveHeader::parse(std::span<const std::uint8_t>(buf, 10));
    CHECK(false);
  } catch (const Error&) {
  }
  // CRC-32C known answer ("123456789" -> 0xE3069283).
  const char* kat = "123456789";
  CHECK(detail::crc32c(reinterpret_cast<const std::uint8_t*>(kat), 9) == 0xE3069283u);
}

void test_rans_kernels() {
  using detail::RansKernel;
  std::mt19937 rng(7);
  for (int round = 0; round < 40; ++round) {
    std::size_t n = (round % 5 == 0) ? rng() % 64 : rng() % 200000;
    Bytes sym(n);
    unsigned alphabet = static_cast<unsigned>(1 + rng() % 256);
    double skew = 0.2 + static_cast<double>(rng() % 100) / 25.0;
    std::exponential_distribution<double> ex(skew);
    for (auto& s : sym) s = static_cast<std::uint8_t>(std::min<double>(alphabet - 1, ex(rng) * alphabet / 4));
    std::vector<std::uint8_t> enc;
    detail::rans_encode(sym, enc);
    for (RansKernel k : {RansKernel::Scalar, RansKernel::Auto, RansKernel::Avx2, RansKernel::Neon}) {
      detail::rans_set_kernel(k);
      detail::Reader r(enc);
      std::vector<std::uint8_t> out;
      detail::rans_decode(r, out, n);
      CHECK(out == sym);
      CHECK(r.remaining() == 0);
    }
  }
  detail::rans_set_kernel(RansKernel::Auto);
}

void test_roundtrips() {
  std::vector<std::pair<std::string, Bytes>> inputs;
  inputs.emplace_back("empty", Bytes{});
  inputs.emplace_back("one", Bytes{42});
  inputs.emplace_back("tiny-random", random_bytes(100, 1));
  inputs.emplace_back("zeros", Bytes(150000, 0));
  inputs.emplace_back("text", text_bytes(300000, 2));
  inputs.emplace_back("cased-text", cased_text_bytes(200000, 7));
  inputs.emplace_back("random", random_bytes(140000, 3));
  inputs.emplace_back("structured", structured_bytes(200000));
  inputs.emplace_back("block-1", text_bytes(65535, 4));
  inputs.emplace_back("block", text_bytes(65536, 5));
  inputs.emplace_back("block+1", text_bytes(65537, 6));
  for (TierLevel tier : kTiers) {
    for (auto& [name, data] : inputs) {
      if (is_cm(tier) && data.size() > (slow_env() ? 70000u : 200000u)) continue;  // keep sanitizer runs short
      for (unsigned threads : {1u, 3u}) {
        auto engine = create_engine(tier, small_options(tier, threads));
        EngineStats st;
        Bytes packed = engine->compress(data, &st);
        Bytes back = engine->decompress(packed);
        if (back != data) std::fprintf(stderr, "round trip failed: %s on %s\n", std::string(engine->get_name()).c_str(), name.c_str());
        CHECK(back == data);
        CHECK(peek_header(packed).tier == tier);
        CHECK(decompress(packed) == data);  // auto-detecting entry point
        if (name == "random") CHECK(packed.size() <= data.size() + data.size() / 1000 + 128);
      }
    }
  }
}

void test_thread_independence() {
  Bytes data = text_bytes(slow_env() ? 160000 : 400000, 9);
  for (TierLevel tier : kTiers) {
    Bytes ref = create_engine(tier, small_options(tier, 1))->compress(data);
    for (unsigned t : {2u, 4u, 7u}) {
      CHECK(create_engine(tier, small_options(tier, t))->compress(data) == ref);
      CHECK(create_engine(tier, small_options(tier, t))->decompress(ref) == data);
    }
  }
}

void test_engine_rules() {
  Bytes data = text_bytes(5000, 11);
  Bytes t1 = compress(TierLevel::Tier1_HyperSpeed, data);
  auto t2 = create_engine(TierLevel::Tier2_Consumer);
  try {
    t2->decompress(t1);
    CHECK(false);
  } catch (const Error&) {
  }
  EngineOptions tight;
  tight.max_output_bytes = 1000;
  try {
    decompress(t1, tight);
    CHECK(false);
  } catch (const Error&) {
  }
  EngineOptions bad;
  bad.block_size = 3000;
  try {
    create_engine(TierLevel::Tier1_HyperSpeed, bad);
    CHECK(false);
  } catch (const Error&) {
  }
  CHECK(create_engine(TierLevel::Tier4_Enterprise)->get_tier() == TierLevel::Tier4_Enterprise);
  CHECK(!create_engine(TierLevel::Tier3_Balanced)->get_name().empty());
}

// Tier 1's two modes: both round-trip edge-case inputs, the archive records the mode, and the
// entropy mode is the smaller one on text.
void test_tier1_modes() {
  std::vector<Bytes> inputs = {{}, Bytes(1, 7), Bytes(100000, 0), random_bytes(70000, 41), text_bytes(300000, 42)};
  Bytes mixed = text_bytes(150000, 43);
  Bytes rnd = random_bytes(40000, 44);
  mixed.insert(mixed.begin() + 50000, rnd.begin(), rnd.end());
  inputs.push_back(mixed);
  for (const bool entropy : {false, true}) {
    EngineOptions o = small_options(TierLevel::Tier1_HyperSpeed, 2);
    o.tier1_entropy = entropy;
    auto e = create_engine(TierLevel::Tier1_HyperSpeed, o);
    for (const Bytes& in : inputs) {
      const Bytes a = e->compress(in);
      CHECK(e->decompress(a) == in);
      if (!in.empty()) CHECK(peek_header(a).model_param == (entropy ? 1u : 0u));
    }
  }
  EngineOptions raw;
  raw.tier1_entropy = false;
  const Bytes& text = inputs[4];
  CHECK(compress(TierLevel::Tier1_HyperSpeed, text).size() * 5 < create_engine(TierLevel::Tier1_HyperSpeed, raw)->compress(text).size() * 4);
}

// Tier 4 with its built-in primer: round trips over several blocks and
// thread counts, archives that do not depend on the thread count, and a
// primer ID the decoder does not know.
void test_tier4_primer() {
  Bytes data = text_bytes(slow_env() ? 100000 : 300000, 31);
  Bytes rnd = random_bytes(70000, 32);  // a stored block between coded ones
  data.insert(data.begin() + 65536, rnd.begin(), rnd.end());
  EngineOptions o;
  o.block_size = 64 * 1024;
  o.threads = 1;
  auto one = create_engine(TierLevel::Tier4_Enterprise, o);
  EngineStats st;
  Bytes ref = one->compress(data, &st);
  CHECK(st.stored_blocks >= 1 && st.stored_blocks < st.blocks);
  CHECK(peek_header(ref).model_param != 0);
  CHECK(one->decompress(ref) == data);
  o.threads = 3;
  auto three = create_engine(TierLevel::Tier4_Enterprise, o);
  CHECK(three->compress(data) == ref);
  CHECK(three->decompress(ref) == data);
  // Unprimed archives record model_param 0.
  o.tier4_primer = false;
  Bytes plain = create_engine(TierLevel::Tier4_Enterprise, o)->compress(data);
  CHECK(peek_header(plain).model_param == 0 && plain != ref);
  CHECK(decompress(plain) == data);
  Bytes bad = ref;
  detail::store32(bad.data() + 24, 12345);  // another primer
  detail::store32(bad.data() + 28, detail::crc32c(bad.data(), 28));
  try {
    decompress(bad);
    CHECK(false);
  } catch (const Error&) {
  }
}

// Every table layout round-trips, is recorded in the archive and changes the coded bits; the
// auto choice follows the memory budget, not the thread count.
void test_tier4_layouts() {
  const Bytes data = text_bytes(slow_env() ? 150000 : 400000, 33);
  EngineOptions o;
  o.block_size = 64 * 1024;
  o.threads = 2;
  std::vector<Bytes> archives;
  for (int l = 0; l < 4; ++l) {
    o.tier4_tables = l;
    archives.push_back(create_engine(TierLevel::Tier4_Enterprise, o)->compress(data));
    CHECK(decompress(archives.back()) == data);
  }
  CHECK(peek_header(archives[0]).model_param == detail::sws::primer_crc());
  CHECK(archives[0] != archives[3] && peek_header(archives[0]).model_param != peek_header(archives[3]).model_param);
  o.tier4_tables = -1;
  o.memory_budget = std::size_t{4} << 30;
  CHECK(create_engine(TierLevel::Tier4_Enterprise, o)->compress(data) == archives[0]);
  o.memory_budget = std::size_t{1} << 20;  // nothing fits: Small, one worker
  o.threads = 4;
  CHECK(create_engine(TierLevel::Tier4_Enterprise, o)->compress(data) == archives[3]);
  // Long-match bypass: deterministic across thread counts, recorded, and every primer x layout x
  // bypass combination gets its own model_param.
  o.tier4_tables = -1;
  o.memory_budget = 0;
  o.tier4_bypass = true;
  o.threads = 1;
  const Bytes by1 = create_engine(TierLevel::Tier4_Enterprise, o)->compress(data);
  o.threads = 3;
  CHECK(create_engine(TierLevel::Tier4_Enterprise, o)->compress(data) == by1);
  CHECK(by1 != archives[0] && decompress(by1) == data);
  const Bytes small = text_bytes(20000, 34);
  std::vector<std::uint32_t> params;
  for (int l = 0; l < 4; ++l)
    for (const bool primer : {false, true})
      for (const bool bypass : {false, true}) {
        o.tier4_tables = l;
        o.tier4_primer = primer;
        o.tier4_bypass = bypass;
        const Bytes a = create_engine(TierLevel::Tier4_Enterprise, o)->compress(small);
        CHECK(decompress(a) == small);
        params.push_back(peek_header(a).model_param);
      }
  std::sort(params.begin(), params.end());
  CHECK(std::adjacent_find(params.begin(), params.end()) == params.end());
  o.tier4_tables = 4;
  try {
    create_engine(TierLevel::Tier4_Enterprise, o);
    CHECK(false);
  } catch (const Error&) {
  }
}

void test_corruption() {
  const int iters = fuzz_iters(150);
  std::mt19937 rng(1234);
  for (TierLevel tier : kTiers) {
    Bytes data = text_bytes(is_cm(tier) ? 12000 : 150000, 21);
    // A few stored-friendly random bytes too, so both block methods are hit.
    Bytes rnd = random_bytes(3000, 22);
    data.insert(data.end(), rnd.begin(), rnd.end());
    EngineOptions o = small_options(tier, 2);
    o.block_size = 64 * 1024;
    auto engine = create_engine(tier, o);
    Bytes packed = engine->compress(data);
    int threw = 0, same = 0;
    auto attempt = [&](const Bytes& bad) {
      try {
        Bytes out = engine->decompress(bad);
        CHECK(out == data);
        ++same;
      } catch (const Error&) {
        ++threw;
      }
    };
    const int n = is_cm(tier) ? std::max(4, iters / 3) : iters;  // each CM attempt decodes a block
    for (int i = 0; i < n; ++i) {
      Bytes bad = packed;
      bad.resize(rng() % packed.size());  // truncation
      attempt(bad);
      bad = packed;
      int flips = 1 + static_cast<int>(rng() % 4);
      for (int f = 0; f < flips; ++f) bad[rng() % bad.size()] ^= static_cast<std::uint8_t>(1u << (rng() % 8));
      attempt(bad);
      // Corrupt only the payload area and fix nothing: exercises the block decoders.
      bad = packed;
      std::size_t payload_start = ArchiveHeader::kSize + 8 * peek_header(packed).block_count + 4;
      if (bad.size() > payload_start + 1) {
        std::size_t at = payload_start + rng() % (bad.size() - payload_start);
        bad[at] = static_cast<std::uint8_t>(rng());
        attempt(bad);
      }
    }
    CHECK(threw > 0);
    std::printf("  corruption %-40s %d rejected, %d harmless\n", std::string(engine->get_name()).c_str(), threw, same);
  }
}

void test_word_filter() {
  using detail::WordDict;
  std::vector<Bytes> inputs = {Bytes{}, Bytes{'A'}, Bytes{1, 2, 0xFF, 0, 0x80}, cased_text_bytes(50000, 31),
                               text_bytes(50000, 32), random_bytes(20000, 33), Bytes(400, 'Z')};
  for (WordDict d : {WordDict::Prose, WordDict::Code}) {
    for (const Bytes& in : inputs) {
      Bytes f;
      detail::word_filter_encode(d, in, f);
      CHECK(f.size() <= 2 * in.size());
      Bytes back(in.size());
      detail::word_filter_decode(d, f, back);
      CHECK(back == in);
    }
  }
  CHECK(detail::word_filter_choose(cased_text_bytes(200000, 34)) != WordDict::None);
  CHECK(detail::word_filter_choose(random_bytes(200000, 35)) == WordDict::None);
  CHECK(detail::word_filter_choose(structured_bytes(200000)) == WordDict::None);
  // Corrupt filtered data: throw or decode to some bytes, never overrun.
  std::mt19937 rng(36);
  Bytes in = cased_text_bytes(20000, 37), f;
  detail::word_filter_encode(WordDict::Code, in, f);
  int threw = 0;
  for (int i = 0; i < fuzz_iters(150) * 4; ++i) {
    Bytes bad = f;
    if (i % 2) bad.resize(rng() % bad.size());
    else bad[rng() % bad.size()] = static_cast<std::uint8_t>(rng());
    Bytes out(in.size() + (rng() % 3) - 1);
    try {
      detail::word_filter_decode(WordDict::Code, bad, out);
    } catch (const Error&) {
      ++threw;
    }
  }
  CHECK(threw > 0);
}

#if defined(ULTRA_TEST_RSS) && !defined(ULTRA_SANITIZED)
// Runs compress + decompress with default options in a child process and
// returns its peak RSS in MiB.
double child_peak_rss_mib(TierLevel tier, std::size_t input_size) {
  pid_t pid = fork();
  if (pid == 0) {
    Bytes data = text_bytes(input_size, 99);
    auto e = create_engine(tier);
    Bytes packed = e->compress(data);
    bool ok = e->decompress(packed) == data;
    _exit(ok ? 0 : 3);
  }
  int status = 0;
  struct rusage ru {};
  if (pid < 0 || wait4(pid, &status, 0, &ru) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return -1;
  return static_cast<double>(ru.ru_maxrss) / 1024.0;
}

void test_rss_limits() {
  if (std::getenv("ULTRA_SKIP_RSS")) return;
  struct Limit {
    TierLevel tier;
    std::size_t input;
    double mib;
  };
  // Targets: Tier 2 <= 256 MiB, Tier 3 ~500 MiB, Tier 4 ~200 MiB per model
  // (here the snapshot and one worker, as 4 MiB is two blocks); inputs are
  // counted in the RSS too.
  const Limit limits[] = {{TierLevel::Tier1_HyperSpeed, 32u << 20, 160},
                          {TierLevel::Tier2_Consumer, 32u << 20, 256},
                          {TierLevel::Tier3_Balanced, 4u << 20, 600},
                          {TierLevel::Tier4_Enterprise, 4u << 20, 640}};
  for (const auto& l : limits) {
    double rss = child_peak_rss_mib(l.tier, l.input);
    std::printf("  peak RSS %-40s %.0f MiB (limit %.0f)\n", std::string(tier_name(l.tier)).c_str(), rss, l.mib);
    CHECK(rss > 0 && rss <= l.mib);
  }
}
#endif

}  // namespace

int main() {
  std::printf("header\n");
  test_header();
  std::printf("rANS kernels (active: %d)\n", static_cast<int>(detail::rans_active_kernel()));
  test_rans_kernels();
  std::printf("round trips\n");
  test_roundtrips();
  std::printf("thread independence\n");
  test_thread_independence();
  std::printf("engine rules\n");
  test_engine_rules();
  std::printf("tier 1 modes\n");
  test_tier1_modes();
  std::printf("tier 4 primer\n");
  test_tier4_primer();
  std::printf("tier 4 table layouts\n");
  test_tier4_layouts();
  std::printf("word filter\n");
  test_word_filter();
  std::printf("corruption\n");
  test_corruption();
#if defined(ULTRA_TEST_RSS) && !defined(ULTRA_SANITIZED)
  std::printf("RSS limits\n");
  test_rss_limits();
#endif
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all ultracodec tests passed\n");
  return 0;
}
