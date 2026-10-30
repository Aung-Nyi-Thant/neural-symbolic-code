// ultra_bench: round-trip benchmark for one ultracodec tier.
//
//   ultra_bench <tier 1-4> [--threads N] [--block BYTES] [--budget MIB]
//               [--prime BYTES] [--runs N] [--kernel auto|scalar|avx2|neon]
//               [--no-primer] [--tables 0-5] [--bypass] [--wide-mixer] [--t3-legacy] [--total-only] FILE...
//
// For every file (and the concatenation of all of them, "TOTAL") prints the
// compressed size, ratio, and compress and decompress throughput over --runs
// repetitions: the median and, labeled as such, the best (MB = 10^6 bytes).
// Every repetition's output is compared with its input byte for byte as soon
// as it is produced; one mismatch fails the run (exit 1). Peak RSS covers the
// whole process (all inputs, buffers and engines), so run one tier per
// process to read that tier's memory.
//
// Numbers are checked in full: --threads 0..1024 (0 = one per CPU), --block
// a power of two 64 KiB..64 MiB, --budget 1..1048576 MiB, --prime 0..64 MiB,
// --runs 1..10000, --tables 0..5. Exit 2 on bad usage.
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#define ULTRA_HAVE_RUSAGE 1
#endif

#include <cstdio>
#include <string>
#include <vector>

#include "rans_static.hpp"  // kernel override (internal header)
#include "tool_util.hpp"
#include "ultracodec/engine.hpp"

namespace {

int usage(const std::string& why) {
  std::fprintf(stderr, "ultra_bench: %s\n", why.c_str());
  std::fprintf(stderr, "usage: ultra_bench <tier 1-4> [--threads N] [--block B] [--budget MiB] [--prime B] [--runs N] FILE...\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ultracodec;
  namespace tools = ultracodec::tools;
  if (argc < 3) return usage("a tier and at least one file are needed");
  const auto tier = tools::parse_uint(argv[1], 1, 4);
  if (!tier) return usage(std::string("tier ") + argv[1] + ": expected 1..4");
  EngineOptions opt;
  int runs = 1;
  bool each = true;
  std::vector<const char*> files;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const char* v = nullptr;
    const auto number = [&](std::uint64_t lo, std::uint64_t hi) -> std::optional<std::uint64_t> {
      if (i + 1 >= argc) return std::nullopt;
      v = argv[++i];
      return tools::parse_uint(v, lo, hi);
    };
    const auto bad = [&](const char* range) { return usage(a + " " + (v ? v : "(missing)") + ": expected " + range); };
    if (a == "--threads") {
      const auto n = number(0, 1024);
      if (!n) return bad("0..1024");
      opt.threads = static_cast<unsigned>(*n);
    } else if (a == "--block") {
      const auto n = number(std::uint64_t{1} << 16, std::uint64_t{1} << 26);
      if (!n || (*n & (*n - 1)) != 0) return bad("a power of two, 65536..67108864");
      opt.block_size = static_cast<std::size_t>(*n);
    } else if (a == "--budget") {
      const auto n = number(1, std::uint64_t{1} << 20);
      if (!n) return bad("1..1048576 MiB");
      opt.memory_budget = static_cast<std::size_t>(*n) << 20;
    } else if (a == "--prime") {
      const auto n = number(0, std::uint64_t{1} << 26);
      if (!n) return bad("0..67108864");
      opt.prime_bytes = static_cast<std::size_t>(*n);
    } else if (a == "--runs") {
      const auto n = number(1, 10000);
      if (!n) return bad("1..10000");
      runs = static_cast<int>(*n);
    } else if (a == "--tables") {
      const auto n = number(0, 5);
      if (!n) return bad("0..5");
      opt.tier4_tables = static_cast<int>(*n);
    } else if (a == "--kernel") {
      if (i + 1 >= argc) return usage("--kernel needs a value");
      const std::string k = argv[++i];
      using detail::RansKernel;
      if (k != "auto" && k != "scalar" && k != "avx2" && k != "neon") return usage("--kernel " + k + ": expected auto, scalar, avx2 or neon");
      detail::rans_set_kernel(k == "scalar" ? RansKernel::Scalar : k == "avx2" ? RansKernel::Avx2
                              : k == "neon" ? RansKernel::Neon : RansKernel::Auto);
    } else if (a == "--total-only") each = false;
    else if (a == "--no-primer") opt.tier4_primer = false;
    else if (a == "--bypass") opt.tier4_bypass = true;
    else if (a == "--wide-mixer") opt.tier4_narrow_mixer = false;
    else if (a == "--t3-legacy") opt.tier3_compact = false;
    else if (a.size() > 1 && a[0] == '-') return usage("unknown option " + a);
    else files.push_back(argv[i]);
  }
  if (files.empty()) return usage("no input files");
  try {
    auto engine = create_engine(static_cast<TierLevel>(*tier), opt);
    std::printf("%s threads=%u runs=%d (speeds: median and best of the runs)\n", std::string(engine->get_name()).c_str(),
                opt.threads, runs);
    std::printf("%-12s %12s %12s %8s %10s %10s %10s %10s\n", "file", "bytes", "packed", "bpc", "comp med", "comp best",
                "dec med", "dec best");
    std::vector<std::uint8_t> all;
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> inputs;
    for (auto* f : files) {
      auto d = tools::read_file(f);
      all.insert(all.end(), d.begin(), d.end());
      std::string name = f;
      if (auto s = name.rfind('/'); s != std::string::npos) name = name.substr(s + 1);
      if (each) inputs.emplace_back(name, std::move(d));
    }
    if (files.size() > 1 || !each) inputs.emplace_back("TOTAL(cat)", std::move(all));
    std::size_t sum_packed = 0, sum_raw = 0;
    double sum_ct = 0, sum_dt = 0;
    const auto mbs = [](std::size_t n, double s) { return s > 0 ? static_cast<double>(n) / s / 1e6 : 0.0; };
    for (auto& [name, data] : inputs) {
      const tools::Repetitions r = tools::run_repetitions(*engine, data, runs);
      if (!r.verified) {
        std::printf("%s: ROUND TRIP FAILED in repetition %d of %d\n", name.c_str(), r.failed_run + 1, runs);
        return 1;
      }
      const double ct = tools::Repetitions::median(r.compress_s), dt = tools::Repetitions::median(r.decompress_s);
      std::printf("%-12s %12zu %12zu %8.4f %10.2f %10.2f %10.2f %10.2f%s\n", name.c_str(), data.size(), r.packed_bytes,
                  data.empty() ? 0.0 : 8.0 * static_cast<double>(r.packed_bytes) / static_cast<double>(data.size()),
                  mbs(data.size(), ct), mbs(data.size(), tools::Repetitions::best(r.compress_s)), mbs(data.size(), dt),
                  mbs(data.size(), tools::Repetitions::best(r.decompress_s)), r.deterministic ? "" : "  (archives differ between runs)");
      if (name != "TOTAL(cat)") {
        sum_packed += r.packed_bytes;
        sum_raw += data.size();
        sum_ct += ct;
        sum_dt += dt;
      }
    }
    if (each && files.size() > 1)
      std::printf("%-12s %12zu %12zu %8.4f %10.2f %10s %10.2f %10s\n", "SUM(files)", sum_raw, sum_packed,
                  sum_raw ? 8.0 * static_cast<double>(sum_packed) / static_cast<double>(sum_raw) : 0.0,
                  mbs(sum_raw, sum_ct), "-", mbs(sum_raw, sum_dt), "-");
  } catch (const tools::IoError& e) {
    std::fprintf(stderr, "ultra_bench: %s\n", e.what.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ultra_bench: %s\n", e.what());
    return 1;
  }
#if defined(ULTRA_HAVE_RUSAGE)
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  std::printf("peak RSS %.1f MiB (whole process)\n", tools::maxrss_mib(ru.ru_maxrss));
#endif
  return 0;
}
