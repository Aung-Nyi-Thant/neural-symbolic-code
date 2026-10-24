// Black-box tests for the hce CLI: exit codes, files, pipes, signals.
// Usage: test_cli /path/to/hce
// HCE_TEST_WRAPPER, when set, prefixes every hce invocation (for example
// "qemu-aarch64 -cpu max" to test a cross-compiled binary).
#include "hc_framed.hpp"
#include "tests/legacy_streams.hpp"

#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

using Vec = std::vector<std::byte>;

static int g_failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

static std::string g_hce;   // the binary under test
static std::string g_dir;   // scratch directory
static std::string g_wrap;  // HCE_TEST_WRAPPER: an emulator to run it under, or empty

// The binary under test as a shell command word.
static std::string hce_cmd() { return (g_wrap.empty() ? "" : g_wrap + " ") + "'" + g_hce + "'"; }

static std::string path(const std::string& name) { return g_dir + "/" + name; }

static Vec read_file(const std::string& name) {
  std::ifstream f(path(name), std::ios::binary);
  const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  Vec v(s.size());
  if (!s.empty()) std::memcpy(v.data(), s.data(), s.size());
  return v;
}

static void write_file(const std::string& name, const Vec& v) {
  std::ofstream f(path(name), std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
}

static bool exists(const std::string& name) {
  struct stat st {};
  return ::stat(path(name).c_str(), &st) == 0;
}

static std::string text(const Vec& v) { return std::string(reinterpret_cast<const char*>(v.data()), v.size()); }

// A sanitizer report in the child's stderr fails the test even if the exit code looks right.
static void check_clean(const std::string& err) {
  const bool dirty = err.find("Sanitizer") != std::string::npos || err.find("runtime error:") != std::string::npos;
  CHECK(!dirty);
  if (dirty) std::printf("%s\n", err.c_str());
}

struct Run {
  int code;
  std::string err;
};

// Runs `args` (a shell fragment, hce first) inside the scratch directory.
static Run hce(const std::string& args, const std::string& redirections = "") {
  const std::string cmd = "cd '" + g_dir + "' && " + hce_cmd() + " " + args + " " + redirections + " 2>stderr.txt";
  const int st = std::system(cmd.c_str());
  Run r{WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0), text(read_file("stderr.txt"))};
  check_clean(r.err);
  return r;
}

static int shell(const std::string& cmd) {
  const int st = std::system(("cd '" + g_dir + "' && " + cmd).c_str());
  return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static bool temp_files_left() {
  DIR* d = ::opendir(g_dir.c_str());
  bool found = false;
  while (const dirent* e = d ? ::readdir(d) : nullptr) found |= std::strstr(e->d_name, ".hce-") != nullptr;
  if (d) ::closedir(d);
  return found;
}

// Spawns hce with the given stdin/stdout descriptors (or -1 to inherit).
static pid_t spawn(const std::vector<std::string>& args, int in_fd, int out_fd) {
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  if (in_fd >= 0) posix_spawn_file_actions_adddup2(&fa, in_fd, STDIN_FILENO);
  if (out_fd >= 0) posix_spawn_file_actions_adddup2(&fa, out_fd, STDOUT_FILENO);
  const std::string err = path("stderr.txt");
  posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, err.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  std::vector<std::string> all;
  for (std::size_t at = 0; at < g_wrap.size();) {  // wrapper words, split on spaces
    const std::size_t end = std::min(g_wrap.find(' ', at), g_wrap.size());
    if (end > at) all.push_back(g_wrap.substr(at, end - at));
    at = end + 1;
  }
  all.push_back(g_hce);
  all.insert(all.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (std::string& a : all) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = -1;
  const int rc = ::posix_spawnp(&pid, all[0].c_str(), &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  return rc == 0 ? pid : -1;
}

// Waits up to `limit` for the child; returns its exit code, or -1 on timeout (child killed).
static int wait_exit(pid_t pid, std::chrono::milliseconds limit) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  int st = 0;
  while (::waitpid(pid, &st, WNOHANG) == 0) {
    if (std::chrono::steady_clock::now() > deadline) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &st, 0);
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  check_clean(text(read_file("stderr.txt")));
  return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

static Vec log_data(std::size_t lines, unsigned seed) {
  std::mt19937 rng(seed);
  Vec v;
  char buf[160];
  for (std::size_t i = 0; i < lines; ++i) {
    const int n = std::snprintf(buf, sizeof buf, "2026-10-24T12:%02zu:%02zu host-%u GET /api/v1/items/%u status=%u\n",
                                (i / 60) % 60, i % 60, static_cast<unsigned>(rng() % 8),
                                static_cast<unsigned>(rng() % 100000), i % 13 ? 200u : 503u);
    for (int k = 0; k < n; ++k) v.push_back(static_cast<std::byte>(buf[k]));
  }
  return v;
}

// Parses the gap report into [offset, offset + length) ranges.
static std::vector<std::pair<std::uint64_t, std::uint64_t>> gap_ranges(const std::string& jsonl) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
  std::size_t pos = 0;
  while ((pos = jsonl.find("\"out_offset\":", pos)) != std::string::npos) {
    const std::uint64_t off = std::strtoull(jsonl.c_str() + pos + 13, nullptr, 10);
    const std::size_t lp = jsonl.find("\"length\":", pos);
    const std::uint64_t len = std::strtoull(jsonl.c_str() + lp + 9, nullptr, 10);
    out.emplace_back(off, off + len);
    pos = lp;
  }
  return out;
}

// ------------------------------------------------------------------ tests ---

static void test_roundtrips() {
  const Vec data = log_data(8000, 1);
  write_file("data.log", data);
  CHECK(hce("-i data.log -o data.hce").code == 0);
  CHECK(hce("-d -i data.hce -o out.log").code == 0);
  CHECK(read_file("out.log") == data);

  // stdin/stdout, explicit '-' and pipes in both directions.
  CHECK(shell("cat data.log | " + hce_cmd() + " -T 3 | " + hce_cmd() + " -d -T 2 > pipe.log") == 0);
  CHECK(read_file("pipe.log") == data);
  CHECK(hce("-i - -o -", "< data.log > dash.hce").code == 0);
  CHECK(hce("-d -", "< dash.hce > dash.log").code == 0);
  CHECK(read_file("dash.log") == data);

  // The compressed bytes do not depend on the thread count.
  CHECK(hce("-T 1 -i data.log -o t1.hce").code == 0);
  CHECK(hce("-T 4 -i data.log -o t4.hce").code == 0);
  CHECK(read_file("t1.hce") == read_file("t4.hce"));
  // Sequential and parallel decoding agree.
  CHECK(hce("-d -T 1 -i t4.hce -o s.log").code == 0 && hce("-d -T 4 -i t4.hce -o p.log").code == 0);
  CHECK(read_file("s.log") == data && read_file("p.log") == data);

  // Test mode verifies without writing anything.
  const Run t = hce("-t -i data.hce --stats=json");
  CHECK(t.code == 0 && t.err.find("\"verified\":1") != std::string::npos && t.err.find("\"exit\":0") != std::string::npos);
}

static void test_files_and_sizes() {
  const Vec before = read_file("data.hce");
  CHECK(hce("-i data.log -o data.hce").code == 1);  // exists
  CHECK(read_file("data.hce") == before);
  CHECK(hce("-f -i data.log -o data.hce").code == 0);
  CHECK(hce("-f -i data.log -o data.log").code == 1);  // same file, even with -f
  CHECK(read_file("data.log") == read_file("out.log"));  // untouched by the refused run
  CHECK(hce("-i data.log -o .").code == 1);  // a directory

  const std::string n = std::to_string(read_file("data.log").size());
  CHECK(hce("--expect-size " + n + " -i data.log -o exp.hce").code == 0);
  CHECK(hce("--expect-size 5 -i data.log -o bad_exp.hce").code == 1);
  CHECK(!exists("bad_exp.hce"));
  CHECK(shell("cat data.log | " + hce_cmd() + " --expect-size 100 > piped_exp.hce 2>/dev/null") == 1);
  CHECK(hce("-d --expect-size " + n + " -i exp.hce -o exp.log").code == 0);
  CHECK(hce("-d --expect-size 7 -i exp.hce -o exp_bad.log").code == 1);
  CHECK(!exists("exp_bad.log"));
  CHECK(!temp_files_left());
}

static void test_usage_errors() {
  for (const char* args : {"--bogus", "-l 9", "--window 1000", "--block 16M", "--frame 10", "-T 0",
                           "--report r.jsonl -i data.log", "-t -i data.hce -o x", "--recovery maybe -d",
                           "--fill 300 -d", "--stats=xml", "-i a -i b", "--keep-partial -d -i data.hce"})
    CHECK(hce(args).code == 1);
  CHECK(hce("--help", "> help.txt").code == 0);
  CHECK(text(read_file("help.txt")).find("Exit status") != std::string::npos);
  CHECK(hce("--version", "> version.txt").code == 0);
  CHECK(text(read_file("version.txt")).find("neural CM codec 0x4E434D01") != std::string::npos);
}

static void test_corruption_and_recovery() {
  const Vec data = read_file("data.log");
  Vec bad = read_file("data.hce");
  bad[bad.size() / 2] ^= std::byte{0x20};  // inside some payload
  write_file("bad.hce", bad);

  // Strict: exit 2, no output file unless --keep-partial (then a verified prefix).
  CHECK(hce("-d -i bad.hce -o strict.log").code == 2);
  CHECK(!exists("strict.log"));
  CHECK(hce("-d -i bad.hce -o partial.log --keep-partial").code == 2);
  const Vec partial = read_file("partial.log");
  CHECK(partial.size() < data.size() && std::equal(partial.begin(), partial.end(), data.begin()));
  CHECK(hce("-t -i bad.hce").code == 2);

  // skip: exit 3 and a report; skip_and_fill: same size, differences only inside reported gaps.
  for (const char* threads : {"1", "4"}) {
    const std::string t(threads);
    CHECK(hce("-d -f -T " + t + " --recovery skip -i bad.hce -o skip.log --report skip.jsonl").code == 3);
    CHECK(read_file("skip.log").size() < data.size());
    CHECK(text(read_file("skip.jsonl")).find("\"cause\":") != std::string::npos);

    const Run r = hce("-d -f -T " + t + " --recovery skip_and_fill --fill 0xEE -i bad.hce -o fill.log --report fill.jsonl");
    CHECK(r.code == 3 && r.err.find("data loss") != std::string::npos);
    const Vec filled = read_file("fill.log");
    CHECK(filled.size() == data.size());
    const auto gaps = gap_ranges(text(read_file("fill.jsonl")));
    CHECK(!gaps.empty());
    bool outside_ok = true, inside_fill = true;
    for (std::size_t i = 0; i < std::min(filled.size(), data.size()); ++i) {
      bool in_gap = false;
      for (const auto& [lo, hi] : gaps) in_gap |= i >= lo && i < hi;
      if (in_gap) inside_fill &= filled[i] == std::byte{0xEE};
      else outside_ok &= filled[i] == data[i];
    }
    CHECK(outside_ok && inside_fill);
  }

  // Truncation: corrupt in strict mode, data loss in recovery modes.
  const Vec full = read_file("data.hce");
  write_file("half.hce", Vec(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(full.size() / 2)));
  CHECK(hce("-d -i half.hce -o half.log").code == 2);
  CHECK(hce("-d --recovery skip -i half.hce -o half.log").code == 3);

  // Not an hce stream at all.
  write_file("garbage.bin", log_data(100, 9));
  CHECK(hce("-d -i garbage.bin -o g.log").code == 2);
  CHECK(hce("-d --recovery skip -i garbage.bin -o g2.log").code == 2);
  CHECK(!temp_files_left());
}

static void test_limits_and_unsupported() {
  write_file("zeros.bin", Vec(std::size_t{8} << 20, std::byte{0}));
  CHECK(hce("-i zeros.bin -o zeros.hce").code == 0);
  CHECK(read_file("zeros.hce").size() < 20000);
  for (const char* threads : {"1", "4"}) {
    const Run r = hce(std::string("-d -T ") + threads + " --limits untrusted --max-ratio 10 --ratio-burst 1M -i zeros.hce -o z.out");
    CHECK(r.code == 4 && r.err.find("limit_ratio") != std::string::npos);
    CHECK(!exists("z.out"));
    CHECK(hce(std::string("-t -T ") + threads + " --max-output 1M -i zeros.hce").code == 4);
  }
  CHECK(hce("-t -i zeros.hce --limits untrusted").code == 0);  // 8 MiB fits the 64 MiB burst

  Vec v4 = read_file("data.hce");
  v4[4] = std::byte{4};  // a future format version
  write_file("v4.hce", v4);
  CHECK(hce("-d -i v4.hce -o v4.log").code == 5);
  Vec other = read_file("data.hce");
  hc::store_le<hc::u32>(&other[12], 0x434D0001u);  // another codec's id, with a valid CRC
  hc::store_le<hc::u32>(&other[28], hc::crc32c(hc::Bytes(other).first(28)));
  write_file("other.hce", other);
  for (const char* threads : {"1", "4"}) CHECK(hce(std::string("-t -T ") + threads + " -i other.hce").code == 5);

  // A stream of the retired stand-in LZ codec (hce 1.0.0): unsupported in
  // every mode, and no output file appears.
  Vec legacy(sizeof hce_test::kLegacyLzStream);
  std::memcpy(legacy.data(), hce_test::kLegacyLzStream, legacy.size());
  write_file("legacy.hce", legacy);
  for (const char* mode : {"", " --recovery skip", " --recovery skip_and_fill"}) {
    for (const char* threads : {"1", "4"}) {
      const Run r = hce(std::string("-d -T ") + threads + mode + " -i legacy.hce -o legacy.log");
      CHECK(r.code == 5 && r.err.find("unsupported") != std::string::npos);
      CHECK(!exists("legacy.log"));
    }
  }
  CHECK(hce("-t -i legacy.hce").code == 5);
}

static void test_signals_and_pipes() {
  // SIGINT while blocked on an idle pipe: prompt exit 1, no output, no temp file.
  int in_pipe[2];
  CHECK(::pipe2(in_pipe, O_CLOEXEC) == 0);
  const pid_t pid = spawn({"-o", path("interrupted.hce")}, in_pipe[0], -1);
  CHECK(pid > 0);
  ::close(in_pipe[0]);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  ::kill(pid, SIGINT);
  CHECK(wait_exit(pid, std::chrono::seconds(5)) == 1);
  ::close(in_pipe[1]);
  CHECK(!exists("interrupted.hce") && !temp_files_left());
  CHECK(text(read_file("stderr.txt")).find("interrupted") != std::string::npos);

  // The reader of our stdout goes away: EPIPE ends the run with exit 1, without hanging.
  int out_pipe[2];
  CHECK(::pipe2(out_pipe, O_CLOEXEC) == 0);
  const pid_t dec = spawn({"-d", "-T", "3", "-i", path("data.hce")}, -1, out_pipe[1]);
  CHECK(dec > 0);
  ::close(out_pipe[1]);
  char buf[4096];
  CHECK(::read(out_pipe[0], buf, sizeof buf) > 0);
  ::close(out_pipe[0]);
  CHECK(wait_exit(dec, std::chrono::seconds(5)) == 1);
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 2) {
    std::printf("usage: test_cli /path/to/hce\n");
    return 2;
  }
  char real[PATH_MAX];
  if (!::realpath(argv[1], real)) {
    std::printf("cannot resolve %s\n", argv[1]);
    return 2;
  }
  g_hce = real;
  if (const char* w = std::getenv("HCE_TEST_WRAPPER")) g_wrap = w;
  char tmpl[] = "/tmp/hce-cli-test-XXXXXX";
  if (!::mkdtemp(tmpl)) return 2;
  g_dir = tmpl;

  test_roundtrips();
  test_files_and_sizes();
  test_usage_errors();
  test_corruption_and_recovery();
  test_limits_and_unsupported();
  test_signals_and_pipes();

  if (g_failures == 0) shell("rm -rf '" + g_dir + "'");
  if (g_failures) {
    std::printf("%d check(s) failed (scratch files kept in %s)\n", g_failures, g_dir.c_str());
    return 1;
  }
  std::printf("all CLI tests passed\n");
  return 0;
}
