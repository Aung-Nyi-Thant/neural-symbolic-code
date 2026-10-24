// hce.cpp - command-line front end for the hybrid compression engine.
//
// Exit status (stable, for scripts):
//   0  success; for -d/-t every output byte was verified
//   1  usage or I/O error (also: interrupted, --expect-size mismatch)
//   2  corrupt input (strict mode stops at the first damage)
//   3  finished with data loss: a recovery mode reported gaps
//   4  an R19 resource limit was exceeded (expansion budget, memory cap, ...)
//   5  unsupported format version or feature
#include "cli/posix_io.hpp"
#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"

#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using hc::Errc;
using hc::u64;
using hce::cli::FdInput;
using hce::cli::FdOutput;
using hce::cli::OutputFile;

constexpr std::string_view kVersion = "2.1.0";

enum Exit : int { kSuccess = 0, kUsageOrIo = 1, kCorrupt = 2, kDataLoss = 3, kLimit = 4, kUnsupported = 5 };

enum class Mode { compress, decompress, test };
enum class StatsFormat { none, text, json };

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

constexpr std::string_view kHelp = R"(Usage: hce [MODE] [OPTIONS] [-i INPUT] [-o OUTPUT]

Modes:
  -z, --compress          compress (default)
  -d, --decompress        decompress
  -t, --test              decompress and verify, discarding the output

Input and output:
  -i, --input PATH        input file, or - for stdin (default: stdin)
  -o, --output PATH       output file, or - for stdout (default: stdout);
                          files appear under their name only once complete
  -f, --force             overwrite existing files; allow terminal I/O
      --keep-partial      on failure, keep the partial output file
      --expect-size SIZE  fail unless the uncompressed size is exactly SIZE

Compression:
  -l, --level N           effort 0..4 (default 2); model memory per thread
                          21M, 38M, 40M, 76M, 149M; decoding needs the same
      --window SIZE       history window, power of two 64K..1G (default 8M)
      --block SIZE        block size, power of two 1K..8M (default 256K)
      --frame SIZE        frame length / sync interval, block..1G (default 8M)
      --no-content-size   do not record the input size in the header
  -T, --threads N         worker threads (default: one per CPU)

Decompression:
      --recovery MODE     strict | skip | skip_and_fill (default strict)
      --fill BYTE         fill byte for skip_and_fill, 0..255 or 0xNN (default 0)
      --report PATH       write every lost range as a JSON line
      --limits PROFILE    trusted | untrusted (default trusted)
      --max-memory SIZE   memory cap; compression sheds threads to meet it
      --max-window SIZE   largest accepted history window
      --max-block SIZE    largest accepted block
      --max-level N       highest accepted level
      --max-ratio N       expansion budget: output <= N * input + burst (0 = off)
      --ratio-burst SIZE  burst depth of the expansion budget (default 64M)
      --max-output SIZE   largest total output

Reporting:
      --stats[=FORMAT]    integer telemetry on stderr: text (default) or json
  -v, --verbose           same as --stats=text
  -q, --quiet             suppress warnings and error messages
  -h, --help              show this help
  -V, --version           show version information

SIZE takes an optional K, M, G or T suffix (powers of 1024).

Exit status: 0 success (with -d/-t: every byte verified), 1 usage or I/O error,
2 corrupt input, 3 finished with data loss, 4 resource limit exceeded,
5 unsupported version or feature.
)";

// ---------------------------------------------------------------- options ---

struct Options {
  Mode mode = Mode::compress;
  bool help = false, version = false;
  std::optional<std::string> input, output, report;
  bool force = false, keep_partial = false, quiet = false, record_size = true;
  StatsFormat stats = StatsFormat::none;
  // Compression.
  hc::u8 level = 2, window_log = 23, block_log = 18;
  u64 frame_bytes = u64{8} << 20;
  unsigned threads = 0;  // 0: one per hardware thread
  std::optional<u64> expect_size;
  // Decompression.
  hc::Recovery recovery = hc::Recovery::strict;
  std::byte fill{0};
  hc::DecodeLimits limits = hc::DecodeLimits::trusted();
};

std::optional<u64> parse_uint(std::string_view s) {
  int base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    s.remove_prefix(2);
    base = 16;
  }
  u64 v = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v, base);
  if (ec != std::errc{} || end != s.data() + s.size() || s.empty()) return std::nullopt;
  return v;
}

std::optional<u64> parse_size(std::string_view s) {
  u64 v = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc{} || end == s.data()) return std::nullopt;
  const std::string_view suffix(end, static_cast<std::size_t>(s.data() + s.size() - end));
  unsigned shift = 0;
  if (suffix.empty() || suffix == "B") shift = 0;
  else if (suffix == "K" || suffix == "k" || suffix == "KiB") shift = 10;
  else if (suffix == "M" || suffix == "MiB") shift = 20;
  else if (suffix == "G" || suffix == "GiB") shift = 30;
  else if (suffix == "T" || suffix == "TiB") shift = 40;
  else return std::nullopt;
  if (v > (~u64{0} >> shift)) return std::nullopt;
  return v << shift;
}

[[noreturn]] void bad_value(std::string_view opt, std::string_view value, std::string_view what) {
  throw UsageError(std::string(opt) + ": invalid value '" + std::string(value) + "' (" + std::string(what) + ")");
}

hc::u8 parse_log2(std::string_view opt, std::string_view v, unsigned lo, unsigned hi) {
  const std::optional<u64> n = parse_size(v);
  if (!n || !std::has_single_bit(*n)) bad_value(opt, v, "expected a power of two");
  const auto log = static_cast<unsigned>(std::countr_zero(*n));
  if (log < lo || log > hi) bad_value(opt, v, "out of range");
  return static_cast<hc::u8>(log);
}

Options parse(int argc, char** argv) {
  Options o;
  struct Overrides {
    std::optional<u64> memory, window, block, level, ratio, burst, output;
  } ov;
  std::optional<std::string_view> profile;
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  for (std::size_t i = 0; i < args.size(); ++i) {
    std::string_view a = args[i];
    std::optional<std::string_view> attached;
    if (a.starts_with("--")) {
      if (const std::size_t eq = a.find('='); eq != std::string_view::npos) {
        attached = a.substr(eq + 1);
        a = a.substr(0, eq);
      }
    } else if (a.size() > 2 && a[0] == '-' && std::string_view("ioTl").find(a[1]) != std::string_view::npos) {
      attached = a.substr(2);  // -T4, -oout.hce
      a = a.substr(0, 2);
    }
    const auto value = [&]() -> std::string_view {
      if (attached) return *attached;
      if (i + 1 >= args.size()) throw UsageError(std::string(a) + " needs a value");
      return args[++i];
    };
    const auto flag = [&] {
      if (attached) throw UsageError(std::string(a) + " takes no value");
    };

    // Flags.
    const auto is = [&](std::string_view s, std::string_view l) { return a == s || a == l; };
    if (is("-z", "--compress") || is("-d", "--decompress") || is("-t", "--test") || is("-h", "--help") ||
        is("-V", "--version") || is("-f", "--force") || is("-q", "--quiet") || is("-v", "--verbose") ||
        a == "--keep-partial" || a == "--no-content-size") {
      flag();
      if (is("-z", "--compress")) o.mode = Mode::compress;
      else if (is("-d", "--decompress")) o.mode = Mode::decompress;
      else if (is("-t", "--test")) o.mode = Mode::test;
      else if (is("-h", "--help")) o.help = true;
      else if (is("-V", "--version")) o.version = true;
      else if (is("-f", "--force")) o.force = true;
      else if (is("-q", "--quiet")) o.quiet = true;
      else if (is("-v", "--verbose")) o.stats = StatsFormat::text;
      else if (a == "--keep-partial") o.keep_partial = true;
      else o.record_size = false;
    } else if (a == "--stats") {
      const std::string_view f = attached.value_or("text");
      if (f == "text") o.stats = StatsFormat::text;
      else if (f == "json") o.stats = StatsFormat::json;
      else bad_value(a, f, "text or json");
    } else if (a == "-i" || a == "--input") {
      if (o.input) throw UsageError("more than one input");
      o.input = std::string(value());
    } else if (a == "-o" || a == "--output") {
      o.output = std::string(value());
    } else if (a == "--report") {
      o.report = std::string(value());
    } else if (a == "-l" || a == "--level") {
      const std::string_view v = value();
      const auto n = parse_uint(v);
      if (!n || *n > 4) bad_value(a, v, "0..4");
      o.level = static_cast<hc::u8>(*n);
    } else if (a == "--window") {
      o.window_log = parse_log2(a, value(), hc::kMinWindowLog, hc::kMaxWindowLog);
    } else if (a == "--block") {
      o.block_log = parse_log2(a, value(), hc::kMinBlockLog, hc::kMaxBlockLog);
    } else if (a == "--frame") {
      const std::string_view v = value();
      const auto n = parse_size(v);
      if (!n) bad_value(a, v, "expected a size");
      o.frame_bytes = *n;
    } else if (a == "-T" || a == "--threads") {
      const std::string_view v = value();
      const auto n = parse_uint(v);
      if (!n || *n < 1 || *n > hce::kMaxThreads) bad_value(a, v, "1..256");
      o.threads = static_cast<unsigned>(*n);
    } else if (a == "--expect-size") {
      const std::string_view v = value();
      o.expect_size = parse_size(v);
      if (!o.expect_size) bad_value(a, v, "expected a size");
    } else if (a == "--recovery") {
      const std::string_view v = value();
      if (v == "strict") o.recovery = hc::Recovery::strict;
      else if (v == "skip") o.recovery = hc::Recovery::skip;
      else if (v == "skip_and_fill" || v == "fill") o.recovery = hc::Recovery::skip_and_fill;
      else bad_value(a, v, "strict, skip or skip_and_fill");
    } else if (a == "--fill") {
      const std::string_view v = value();
      const auto n = parse_uint(v);
      if (!n || *n > 255) bad_value(a, v, "0..255");
      o.fill = static_cast<std::byte>(*n);
    } else if (a == "--limits") {
      profile = value();
      if (*profile != "trusted" && *profile != "untrusted") bad_value(a, *profile, "trusted or untrusted");
    } else if (a == "--max-memory" || a == "--max-window" || a == "--max-block" || a == "--max-output" ||
               a == "--ratio-burst") {
      const std::string_view v = value();
      const auto n = parse_size(v);
      if (!n) bad_value(a, v, "expected a size");
      (a == "--max-memory"   ? ov.memory
       : a == "--max-window" ? ov.window
       : a == "--max-block"  ? ov.block
       : a == "--max-output" ? ov.output
                             : ov.burst) = n;
    } else if (a == "--max-level" || a == "--max-ratio") {
      const std::string_view v = value();
      const auto n = parse_uint(v);
      if (!n || (a == "--max-level" && *n > 255) || *n > 0xFFFFFFFFu) bad_value(a, v, "out of range");
      (a == "--max-level" ? ov.level : ov.ratio) = n;
    } else if (a == "-" || !a.starts_with("-")) {
      if (o.input) throw UsageError("more than one input");
      o.input = std::string(a);
    } else {
      throw UsageError("unknown option " + std::string(a));
    }
  }
  if (o.help || o.version) return o;

  // Limits: the profile first, then explicit overrides, whatever their order.
  if (profile == "untrusted") o.limits = hc::DecodeLimits::untrusted();
  if (ov.memory) o.limits.max_memory_bytes = *ov.memory;
  if (ov.window) {
    if (!std::has_single_bit(*ov.window)) throw UsageError("--max-window must be a power of two");
    o.limits.max_window_log = static_cast<hc::u8>(std::countr_zero(*ov.window));
  }
  if (ov.block) o.limits.max_block_bytes = static_cast<hc::u32>(std::min<u64>(*ov.block, 0xFFFFFFFFu));
  if (ov.level) o.limits.max_level = static_cast<hc::u8>(*ov.level);
  if (ov.ratio) o.limits.max_ratio = static_cast<hc::u32>(*ov.ratio);
  if (ov.burst) o.limits.ratio_burst_bytes = *ov.burst;
  if (ov.output) o.limits.max_output_bytes = *ov.output;

  const bool decoding = o.mode != Mode::compress;
  if (!decoding && (o.report || o.recovery != hc::Recovery::strict || profile || ov.window || ov.block || ov.level ||
                    ov.ratio || ov.burst || ov.output))
    throw UsageError("--recovery, --fill, --report and decoder limits apply to -d and -t only");
  if (decoding && !o.record_size) throw UsageError("--no-content-size applies to compression only");
  if (o.mode == Mode::test && o.output) throw UsageError("-t discards its output; -o is not allowed");
  if (o.keep_partial && (!o.output || *o.output == "-")) throw UsageError("--keep-partial needs -o FILE");
  if (o.report && o.output && *o.report == *o.output) throw UsageError("--report and -o name the same file");
  return o;
}

// -------------------------------------------------------------- telemetry ---

std::string num(u64 v) {
  char buf[24];
  const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v);
  return std::string(buf, end);
}

// a * b / c without overflow for the magnitudes telemetry sees.
u64 scale(u64 a, u64 b, u64 c) {
  if (c == 0) return 0;
  if (a <= ~u64{0} / b) return a * b / c;
  return a / c * b;
}

// Integer-only key/value telemetry, as logfmt text or one JSON object.
class Telemetry {
 public:
  void add(std::string_view key, u64 v) { fields_.push_back({std::string(key), num(v), false}); }
  void add(std::string_view key, std::string_view v) { fields_.push_back({std::string(key), std::string(v), true}); }

  std::string render(StatsFormat f) const {
    std::string s = f == StatsFormat::json ? "{" : "hce:";
    for (std::size_t i = 0; i < fields_.size(); ++i) {
      const Field& x = fields_[i];
      if (f == StatsFormat::json) {
        s += (i ? ",\"" : "\"") + x.key + "\":" + (x.quoted ? "\"" + x.value + "\"" : x.value);
      } else {
        s += " " + x.key + "=" + x.value;
      }
    }
    return s + (f == StatsFormat::json ? "}\n" : "\n");
  }

 private:
  struct Field {
    std::string key, value;
    bool quoted;
  };
  std::vector<Field> fields_;
};

// ----------------------------------------------------------------- report ---

// Gap reports as JSON Lines. The decoder calls on_gap from a single thread.
class GapReport {
 public:
  bool open(const std::string& path, bool force, mode_t mode, std::string& err) {
    return file_.open(path, force, mode, err);
  }

  static void on_gap(void* ctx, const hc::GapInfo& g) noexcept { static_cast<GapReport*>(ctx)->add(g); }

  bool finish(std::string& err) {
    if (!flush()) {
      err = "write failed";
      return false;
    }
    return file_.commit(err);
  }

 private:
  void add(const hc::GapInfo& g) noexcept {
    try {
      buf_ += "{\"stream\":" + num(g.stream_index) + ",\"out_offset\":" + num(g.out_offset) +
              ",\"length\":" + (g.length_known ? num(g.length) : "null") +
              ",\"length_known\":" + (g.length_known ? "true" : "false") + ",\"in_offset\":" + num(g.in_offset) +
              ",\"cause\":\"" + hc::to_string(g.cause) + "\"}\n";
      if (buf_.size() >= 64 * 1024) failed_ |= !flush();
    } catch (...) {
      failed_ = true;
    }
  }

  bool flush() {
    if (failed_) return false;
    FdOutput out(file_.fd(), false, {});
    const bool ok = out.write(hc::Bytes(reinterpret_cast<const std::byte*>(buf_.data()), buf_.size()));
    buf_.clear();
    return ok;
  }

  OutputFile file_;
  std::string buf_;
  bool failed_ = false;
};

// ---------------------------------------------------------------- signals ---

std::atomic<int> g_signal{0};
static_assert(std::atomic<int>::is_always_lock_free, "the signal handler needs a lock-free atomic");

void on_signal(int sig) { g_signal.store(sig, std::memory_order_relaxed); }

// SIGINT/SIGTERM/SIGHUP request a clean stop (temporary files are removed);
// SA_RESETHAND makes a second signal act immediately. SIGPIPE is ignored so a
// closed pipe surfaces as EPIPE instead of killing the process.
class SignalGuard {
 public:
  explicit SignalGuard(std::stop_source stop) {
    std::signal(SIGPIPE, SIG_IGN);
    struct sigaction sa {};
    sa.sa_handler = on_signal;
    sa.sa_flags = static_cast<int>(SA_RESETHAND);  // glibc defines it as 0x80000000
    sigemptyset(&sa.sa_mask);
    for (const int s : {SIGINT, SIGTERM, SIGHUP}) sigaction(s, &sa, nullptr);
    watcher_ = std::jthread([stop](std::stop_token st) mutable {
      while (!st.stop_requested()) {
        if (g_signal.load(std::memory_order_relaxed) != 0) {
          stop.request_stop();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
  }
  static int signal() noexcept { return g_signal.load(std::memory_order_relaxed); }

 private:
  std::jthread watcher_;
};

// ------------------------------------------------------------------ files ---

struct Input {
  int fd = STDIN_FILENO;
  bool owned = false, pollable = false;
  std::string name = "<stdin>";
  struct stat st {};
  ~Input() {
    if (owned) ::close(fd);
  }
};

struct Output {
  OutputFile file;
  bool is_file = false, pollable = false;
  int fd = STDOUT_FILENO;
  std::string name = "<stdout>";
};

std::string sys_error(const std::string& what) { return what + ": " + std::strerror(errno); }

bool same_file(const struct stat& a, const struct stat& b) { return a.st_dev == b.st_dev && a.st_ino == b.st_ino; }

void open_input(const Options& o, Input& in) {
  if (o.input && *o.input != "-") {
    in.fd = ::open(o.input->c_str(), O_RDONLY | O_CLOEXEC);
    if (in.fd < 0) throw std::runtime_error(sys_error(*o.input));
    in.owned = true;
    in.name = *o.input;
  }
  if (::fstat(in.fd, &in.st) != 0) throw std::runtime_error(sys_error(in.name));
  if (S_ISDIR(in.st.st_mode)) throw std::runtime_error(in.name + ": is a directory");
  in.pollable = !S_ISREG(in.st.st_mode);
  if (!in.owned && ::isatty(in.fd) && !o.force)
    throw UsageError("refusing to read from a terminal (use -i FILE, a pipe, or -f)");
}

void open_output(const Options& o, const Input& in, mode_t mode, Output& out) {
  if (o.mode == Mode::test) return;
  if (o.output && *o.output != "-") {
    out.name = *o.output;
    struct stat st {};
    if (::stat(o.output->c_str(), &st) == 0) {
      if (same_file(st, in.st)) throw UsageError(out.name + ": input and output are the same file");
      if (S_ISDIR(st.st_mode)) throw UsageError(out.name + ": is a directory");
      if (!o.force) throw UsageError(out.name + ": already exists (use -f to overwrite)");
    }
    std::string err;
    if (!out.file.open(*o.output, o.force, mode, err)) throw std::runtime_error(out.name + ": " + err);
    out.fd = out.file.fd();
    out.is_file = true;
    return;
  }
  struct stat st {};
  if (::fstat(out.fd, &st) != 0) throw std::runtime_error(sys_error(out.name));
  if (S_ISREG(st.st_mode) && st.st_size > 0 && same_file(st, in.st))
    throw UsageError("input and output are the same file");
  out.pollable = !S_ISREG(st.st_mode);
  if (o.mode == Mode::compress && ::isatty(out.fd) && !o.force)
    throw UsageError("refusing to write compressed data to a terminal (use -o FILE, a pipe, or -f)");
}

mode_t creation_mode() {
  const mode_t mask = ::umask(0);  // read before any thread starts: umask() is process-wide
  ::umask(mask);
  return static_cast<mode_t>(0666 & ~mask);
}

unsigned default_threads() { return std::max(1u, std::min(std::thread::hardware_concurrency(), hce::kMaxThreads)); }

// ------------------------------------------------------------------ modes ---

struct Outcome {
  int exit = kSuccess;
  std::string error;    // printed as "hce: error: ..."
  std::string warning;  // printed as "hce: warning: ..."
};

int decode_exit(Errc s, const Options& o, const hce::DecodeResult& r) {
  switch (s) {
    case Errc::ok:
    case Errc::end_of_stream: return o.recovery != hc::Recovery::strict && r.stats.gaps > 0 ? kDataLoss : kSuccess;
    case Errc::truncated:
      return o.recovery != hc::Recovery::strict && r.params_known ? kDataLoss : kCorrupt;
    case Errc::limit_memory:
    case Errc::limit_block:
    case Errc::limit_window:
    case Errc::limit_level:
    case Errc::limit_ratio:
    case Errc::limit_output: return kLimit;
    case Errc::unsupported_version:
    case Errc::unsupported_feature: return kUnsupported;
    case Errc::out_of_memory:
    case Errc::io_error:
    case Errc::cancelled: return kUsageOrIo;
    default: return kCorrupt;
  }
}

std::string limit_hint(Errc s) {
  switch (s) {
    case Errc::limit_memory: return " (raise --max-memory)";
    case Errc::limit_block: return " (raise --max-block)";
    case Errc::limit_window: return " (raise --max-window)";
    case Errc::limit_level: return " (raise --max-level)";
    case Errc::limit_ratio: return " (raise --max-ratio, or 0 to disable)";
    case Errc::limit_output: return " (raise --max-output)";
    default: return "";
  }
}

struct Streams {
  FdInput& in;
  hce::OutputStream& out;
  const FdOutput* out_fd;  // null when the output is discarded (-t)
};

// Describes an io_error or cancellation precisely: which side, and why.
std::string io_failure(const Streams& io, const std::string& in_name, const std::string& out_name) {
  if (SignalGuard::signal() != 0) return "interrupted by signal " + num(static_cast<u64>(SignalGuard::signal()));
  if (io.in.sys_errno() != 0) return in_name + ": read error: " + std::strerror(io.in.sys_errno());
  if (io.out_fd && io.out_fd->broken_pipe()) return out_name + ": output closed early (broken pipe)";
  if (io.out_fd && io.out_fd->sys_errno() != 0) return out_name + ": write error: " + std::strerror(io.out_fd->sys_errno());
  return "interrupted";
}

Outcome run_compress(const Options& o, const Input& in, const std::string& out_name, Streams io,
                     std::stop_token stop, Telemetry& t) {
  hce::EncodeOptions eo;
  eo.level = o.level;
  eo.window_log = o.window_log;
  eo.block_log = o.block_log;
  eo.frame_bytes = o.frame_bytes;
  eo.threads = o.threads ? o.threads : default_threads();
  eo.max_memory = o.limits.max_memory_bytes;
  eo.stop = std::move(stop);
  const std::optional<u64> file_size =
      S_ISREG(in.st.st_mode) ? std::optional<u64>(static_cast<u64>(in.st.st_size)) : std::nullopt;
  if (o.expect_size && file_size && *file_size != *o.expect_size)
    return {kUsageOrIo, in.name + ": size is " + num(*file_size) + " bytes, --expect-size says " + num(*o.expect_size), {}};
  if (o.expect_size) eo.content_size = o.expect_size;
  else if (o.record_size) eo.content_size = file_size;
  if (const char* why = hce::validate(eo)) return {kUsageOrIo, why, {}};

  const hce::EncodeResult r = hce::encode_parallel<hce::NeuralCMCodec>(io.in, io.out, eo);
  t.add("mode", "compress");
  t.add("status", hc::to_string(r.status));
  t.add("in_bytes", r.bytes_in);
  t.add("out_bytes", r.bytes_out);
  t.add("ratio_x1000", scale(r.bytes_in, 1000, r.bytes_out));
  t.add("threads", r.threads);
  t.add("frames", r.frames);
  t.add("blocks", r.blocks);
  t.add("stored_blocks", r.stored_blocks);
  t.add("mem_bound_bytes", r.memory_bound);
  t.add("stream_id", r.stream_id);

  switch (r.status) {
    case Errc::ok: return {};
    case Errc::content_mismatch:
      return {kUsageOrIo,
              in.name + ": input is " + num(r.bytes_in) + (r.bytes_in > eo.content_size.value_or(0) ? "+" : "") +
                  " bytes but " + num(eo.content_size.value_or(0)) +
                  (o.expect_size ? " were expected (--expect-size)"
                                 : " were recorded when it was opened (file changing? use --no-content-size)"),
              {}};
    case Errc::limit_memory:
      return {kLimit, "--max-memory " + num(eo.max_memory) + " cannot hold one worker; it needs " + num(r.memory_bound) + " bytes", {}};
    case Errc::out_of_memory: return {kUsageOrIo, "out of memory", {}};
    case Errc::io_error:
    case Errc::cancelled: return {kUsageOrIo, io_failure(io, in.name, out_name), {}};
    default: return {kUsageOrIo, std::string("compression failed: ") + hc::to_string(r.status), {}};
  }
}

Outcome run_decompress(const Options& o, const std::string& in_name, const std::string& out_name, Streams io,
                       GapReport* report, std::stop_token stop, Telemetry& t) {
  hce::DecodeOptions d;
  d.limits = o.limits;
  d.recovery = o.recovery;
  d.fill = o.fill;
  d.threads = o.threads ? o.threads : default_threads();
  if (report) {
    d.on_gap = &GapReport::on_gap;
    d.gap_ctx = report;
  }
  d.stop = std::move(stop);
  const hce::DecodeResult r = hce::decode_stream<hce::NeuralCMCodec>(io.in, io.out, d);

  const char* recovery = o.recovery == hc::Recovery::strict ? "strict"
                         : o.recovery == hc::Recovery::skip ? "skip"
                                                            : "skip_and_fill";
  t.add("mode", o.mode == Mode::test ? "test" : "decompress");
  t.add("status", hc::to_string(r.status));
  t.add("recovery", recovery);
  t.add("in_bytes", r.bytes_in);
  t.add("out_bytes", r.bytes_out);
  t.add("ratio_x1000", scale(r.bytes_out, 1000, r.bytes_in));
  t.add("threads", r.threads);
  t.add("frames", r.stats.frames);
  t.add("blocks_ok", r.stats.blocks_ok);
  t.add("blocks_salvaged", r.stats.blocks_salvaged);
  t.add("blocks_lost", r.stats.blocks_lost);
  t.add("gaps", r.stats.gaps);
  t.add("bytes_lost", r.stats.bytes_lost);
  t.add("bytes_filled", r.stats.bytes_filled);
  t.add("damage_events", r.stats.damage_events);
  t.add("verified", u64{r.stats.content_verified && r.stats.damage_events == 0});
  t.add("mem_bound_bytes", r.memory_bound);

  const int code = decode_exit(r.status, o, r);
  const std::string what = hc::to_string(r.status);
  switch (code) {
    case kSuccess:
      if (o.expect_size && r.bytes_out != *o.expect_size)
        return {kUsageOrIo, "decoded " + num(r.bytes_out) + " bytes, --expect-size says " + num(*o.expect_size), {}};
      if (r.stats.damage_events > 0)
        return {kSuccess, {}, in_name + ": input was damaged (" + num(r.stats.damage_events) + " events) but no data was lost"};
      return {};
    case kDataLoss: {
      std::string w = in_name + ": recovered with data loss: " + num(r.stats.gaps) + " gap(s)";
      if (r.status == Errc::truncated) w += ", input truncated";
      if (r.stats.bytes_lost) w += ", " + num(r.stats.bytes_lost) + " bytes lost";
      return {kDataLoss, {}, w + (report ? "; see --report" : "; use --report to list the ranges")};
    }
    case kCorrupt:
      if (o.recovery != hc::Recovery::strict) return {kCorrupt, in_name + ": no recognizable hce stream data", {}};
      return {kCorrupt, in_name + ": corrupt input: " + what + " (recover the rest with --recovery skip)", {}};
    case kLimit: return {kLimit, in_name + ": resource limit exceeded: " + what + limit_hint(r.status), {}};
    case kUnsupported: return {kUnsupported, in_name + ": unsupported stream: " + what, {}};
    default:
      if (r.status == Errc::out_of_memory) return {kUsageOrIo, "out of memory", {}};
      return {kUsageOrIo, io_failure(io, in_name, out_name), {}};
  }
}

int run(int argc, char** argv) {
  Options o;
  try {
    o = parse(argc, argv);
  } catch (const UsageError& e) {
    std::fprintf(stderr, "hce: %s\nTry 'hce --help' for more information.\n", e.what());
    return kUsageOrIo;
  }
  if (o.help) {
    std::fwrite(kHelp.data(), 1, kHelp.size(), stdout);
    return kSuccess;
  }
  if (o.version) {
    std::printf("hce %.*s (container v3 FRAMED, neural CM codec 0x%08X, levels 0-%u, %s kernels)\n",
                static_cast<int>(kVersion.size()), kVersion.data(), hce::kNeuralCMCodecId,
                static_cast<unsigned>(hce::kNeuralCMMaxLevel), hce::NeuralCMCodec::simd_path());
    return kSuccess;
  }

  const mode_t mode = creation_mode();
  std::stop_source stop;
  const SignalGuard signals(stop);
  const auto started = std::chrono::steady_clock::now();

  Input in;
  Output out;
  GapReport report;
  try {
    open_input(o, in);
    open_output(o, in, mode, out);
    if (o.report) {
      struct stat st {};
      if (::stat(o.report->c_str(), &st) == 0 && !o.force)
        throw UsageError(*o.report + ": already exists (use -f to overwrite)");
      std::string err;
      if (!report.open(*o.report, o.force, mode, err)) throw std::runtime_error(*o.report + ": " + err);
    }
  } catch (const UsageError& e) {
    if (!o.quiet) std::fprintf(stderr, "hce: %s\n", e.what());
    return kUsageOrIo;
  } catch (const std::runtime_error& e) {
    if (!o.quiet) std::fprintf(stderr, "hce: error: %s\n", e.what());
    return kUsageOrIo;
  }

  FdInput src(in.fd, in.pollable, stop.get_token());
  FdOutput dst(out.fd, out.pollable, stop.get_token());
  hce::NullOutput discard;
  const bool testing = o.mode == Mode::test;
  const Streams io{src, testing ? static_cast<hce::OutputStream&>(discard) : dst, testing ? nullptr : &dst};
  Telemetry t;
  Outcome res = o.mode == Mode::compress
                    ? run_compress(o, in, out.name, io, stop.get_token(), t)
                    : run_decompress(o, in.name, out.name, io, o.report ? &report : nullptr, stop.get_token(), t);
  if (SignalGuard::signal() != 0 && res.exit == kSuccess)  // finished anyway: the output is complete
    res.warning = "signal received after the work was done";

  // Publish files: complete or recovered output; partial output only on request.
  std::string err;
  if (out.is_file) {
    if (res.exit == kSuccess || res.exit == kDataLoss || o.keep_partial) {
      if (!out.file.commit(err)) res = {kUsageOrIo, out.name + ": " + err, {}};
    } else {
      out.file.discard();
    }
  }
  if (o.report && !report.finish(err)) res = {kUsageOrIo, *o.report + ": " + err, {}};

  const auto us = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
  if (!o.quiet) {
    if (!res.warning.empty()) std::fprintf(stderr, "hce: warning: %s\n", res.warning.c_str());
    if (!res.error.empty()) std::fprintf(stderr, "hce: error: %s\n", res.error.c_str());
  }
  if (o.stats != StatsFormat::none) {
    t.add("elapsed_us", us);
    t.add("exit", static_cast<u64>(res.exit));
    std::fputs(t.render(o.stats).c_str(), stderr);
  }
  return res.exit;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hce: error: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "hce: error: unexpected failure\n");
  }
  return kUsageOrIo;
}
