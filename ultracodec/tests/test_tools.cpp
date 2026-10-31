// Tests for the ultra and ultra_bench tools: strict number parsing, peak-RSS
// units, per-repetition verification, checked reads and transactional writes,
// and (POSIX) the ultra binary itself, given as the first argument.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "../tools/tool_util.hpp"
#include "ultracodec/engine.hpp"

#if !defined(_WIN32)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace tools = ultracodec::tools;

static int g_failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                               \
    }                                                             \
  } while (0)

static fs::path g_dir;

static std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}
[[maybe_unused]] static void spit(const fs::path& p, const std::string& s) {  // POSIX tests only
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f << s;
}
static int temp_files(const fs::path& dir) {
  int n = 0;
  for (const auto& e : fs::directory_iterator(dir)) n += e.path().filename().string().find(".ultra-") != std::string::npos;
  return n;
}

static void test_parse_uint() {
  CHECK(tools::parse_uint("0", 0, 4) == 0u && tools::parse_uint("4", 0, 4) == 4u);
  CHECK(tools::parse_uint("0x10", 0, 100) == 16u && tools::parse_uint("0XfF", 0, 1000) == 255u);
  CHECK(tools::parse_uint("18446744073709551615", 0, UINT64_MAX) == UINT64_MAX);
  for (const char* bad : {"", "-1", "+1", " 1", "1 ", "2x", "1.5", "0x", "0xg", "1e3", "18446744073709551616",
                          "99999999999999999999999", "--3"})
    CHECK(!tools::parse_uint(bad, 0, UINT64_MAX));
  CHECK(!tools::parse_uint("5", 1, 4) && !tools::parse_uint("0", 1, 4) && tools::parse_uint("1", 1, 4) == 1u);
}

static void test_rss_units() {
  CHECK(tools::rss_to_mib(1048576, true) == 1.0 && tools::rss_to_mib(1024, false) == 1.0);
  CHECK(tools::rss_to_mib(3 * 1048576L, true) == 3.0 && tools::rss_to_mib(512, false) == 0.5);
#if !defined(_WIN32)
  // The platform's unit, measured: a child that touches 64 MiB reports at least 64 MiB of peak
  // RSS and well under 64 GiB (a wrong unit is off by 1024x either way). Sanitizer shadow
  // memory adds to the baseline: about 210 MiB under TSan.
  const pid_t pid = ::fork();
  if (pid == 0) {
    std::vector<char> v(std::size_t{64} << 20);
    volatile char* touch = v.data();  // volatile: clang -O2 otherwise drops the unread buffer
    for (std::size_t i = 0; i < v.size(); i += 4096) touch[i] = static_cast<char>(i);
    rusage ru{};
    ::getrusage(RUSAGE_SELF, &ru);
    const double mib = tools::maxrss_mib(ru.ru_maxrss);
    std::printf("  peak RSS after touching 64 MiB: %.1f MiB\n", mib);
    ::_exit(mib >= 64.0 && mib < 1024.0 ? 0 : 1);
  }
  int st = -1;
  ::waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
#endif
}

// An engine whose first round trip is wrong and every later one right.
struct FlakyEngine {
  int decodes = 0;
  std::vector<std::uint8_t> compress(std::span<const std::uint8_t> in) { return {in.begin(), in.end()}; }
  std::vector<std::uint8_t> decompress(std::span<const std::uint8_t> in) {
    std::vector<std::uint8_t> out(in.begin(), in.end());
    if (decodes++ == 0 && !out.empty()) out[out.size() / 2] ^= 1;
    return out;
  }
};

static void test_repetitions() {
  const std::vector<std::uint8_t> data(1000, 7);
  FlakyEngine flaky;
  const tools::Repetitions r = tools::run_repetitions(flaky, data, 2);
  CHECK(!r.verified && r.failed_run == 0 && r.compress_s.size() == 2);  // the good second run does not hide the first
  FlakyEngine late;
  late.decodes = 1;  // all good
  CHECK(tools::run_repetitions(late, data, 3).verified);
  CHECK(tools::Repetitions::median({3, 1, 2}) == 2 && tools::Repetitions::median({4, 1, 2, 3}) == 2.5);
  CHECK(tools::Repetitions::best({3, 1, 2}) == 1);
  auto engine = ultracodec::create_engine(ultracodec::TierLevel::Tier1_HyperSpeed, {});
  const tools::Repetitions real = tools::run_repetitions(*engine, data, 3);
  CHECK(real.verified && real.deterministic && real.packed_bytes > 0);
}

static void test_files() {
  const fs::path p = g_dir / "out.bin";
  const std::vector<std::uint8_t> a(100000, 1), b(5, 2);
  bool durable = false;
  tools::write_file(p, a, &durable);
  CHECK(durable && slurp(p) == std::string(a.begin(), a.end()) && temp_files(g_dir) == 0);
  tools::write_file(p, b);
  CHECK(tools::read_file(p) == b);
  bool threw = false;
  try {
    tools::read_file(g_dir / "missing");
  } catch (const tools::IoError&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    tools::write_file(g_dir / "no" / "such" / "dir", a);
  } catch (const tools::IoError&) {
    threw = true;
  }
  CHECK(threw && temp_files(g_dir) == 0);
#if !defined(_WIN32)
  // A device is written in place, not renamed over; its write errors are reported.
  tools::write_file("/dev/null", a);
  if (fs::exists("/dev/full")) {
    threw = false;
    try {
      tools::write_file("/dev/full", a);
    } catch (const tools::IoError&) {
      threw = true;
    }
    CHECK(threw);
  }
  // A failing write (the file size limit) keeps the old file.
  spit(p, "PRECIOUS");
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::signal(SIGXFSZ, SIG_IGN);
    const rlimit lim{4096, 4096};
    ::setrlimit(RLIMIT_FSIZE, &lim);
    try {
      tools::write_file(p, a);
    } catch (const tools::IoError&) {
      ::_exit(0);
    }
    ::_exit(1);
  }
  int st = -1;
  ::waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0 && slurp(p) == "PRECIOUS" && temp_files(g_dir) == 0);
#endif
}

#if !defined(_WIN32)
static int run(const std::string& cmd) {
  const int st = std::system(("cd '" + g_dir.string() + "' && " + cmd + " 2>/dev/null").c_str());
  return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static void test_ultra_binary(const std::string& ultra) {
  const std::string u = "'" + ultra + "'";
  std::mt19937 rng(5);
  std::string text;
  while (text.size() < 200000) text += "line " + std::to_string(rng() % 1000) + " of some repetitive text\n";
  spit(g_dir / "in.txt", text);
  // Malformed numbers and arguments: usage errors, no output.
  for (const char* args : {"c -t 0 in.txt o", "c -t 5 in.txt o", "c -t 2x in.txt o", "c -t -1 in.txt o",
                           "c -T -1 in.txt o", "c -T 1025 in.txt o", "c -T 99999999999999999999 in.txt o",
                           "c -T in.txt o", "c in.txt o -t", "c -x in.txt o", "d -t 1 in.txt o", "z in.txt o"})
    CHECK(run(u + " " + args) == 2 && !fs::exists(g_dir / "o"));
  // Boundaries: -T 0 (automatic) and 1024, tiers 1 and 4.
  CHECK(run(u + " c -t 1 -T 0 in.txt t1.u") == 0 && run(u + " d -T 1024 t1.u t1.txt") == 0);
  CHECK(slurp(g_dir / "t1.txt") == text);
  CHECK(run(u + " c -t 4 -T 1 in.txt t4.u") == 0 && run(u + " d t4.u t4.txt") == 0 && slurp(g_dir / "t4.txt") == text);
  // A corrupt archive, a missing input, the same file: failures, and the destination keeps its content.
  spit(g_dir / "keep.txt", "PRECIOUS");
  std::string bad = slurp(g_dir / "t1.u");
  bad[bad.size() / 2] = static_cast<char>(bad[bad.size() / 2] ^ 0x40);
  spit(g_dir / "bad.u", bad);
  CHECK(run(u + " d bad.u keep.txt") == 1 && slurp(g_dir / "keep.txt") == "PRECIOUS");
  CHECK(run(u + " d missing.u keep.txt") == 1 && slurp(g_dir / "keep.txt") == "PRECIOUS");
  CHECK(run(u + " c t1.u t1.u") == 1 && slurp(g_dir / "t1.u") != "");
  CHECK(run(u + " d t1.u t1.txt") == 0 && slurp(g_dir / "t1.txt") == text);
  // A failing write (ulimit -f 1: 512 or 1024 bytes) neither succeeds nor destroys the destination.
  CHECK(run("(ulimit -f 1; trap '' XFSZ; exec " + u + " d t1.u keep.txt)") == 1);
  CHECK(slurp(g_dir / "keep.txt") == "PRECIOUS");
  if (fs::exists("/dev/full")) CHECK(run(u + " c -t 1 in.txt /dev/full") == 1);
  CHECK(temp_files(g_dir) == 0);
}
#endif

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::random_device rd;
  g_dir = fs::current_path() / ("ultra-tools-test-" + std::to_string(rd()));
  fs::create_directories(g_dir);
  test_parse_uint();
  test_rss_units();
  test_repetitions();
  test_files();
#if !defined(_WIN32)
  if (argc > 1) test_ultra_binary(fs::absolute(argv[1]).string());
#else
  (void)argc, (void)argv;
#endif
  if (g_failures) {
    std::printf("%d check(s) failed (files kept in %s)\n", g_failures, g_dir.string().c_str());
    return 1;
  }
  fs::remove_all(g_dir);
  std::printf("all ultracodec tool tests passed\n");
  return 0;
}
