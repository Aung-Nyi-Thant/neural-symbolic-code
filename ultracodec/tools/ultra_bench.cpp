// ultra_bench: round-trip benchmark for one ultracodec tier.
//
//   ultra_bench <tier 1-4> [--threads N] [--block BYTES] [--budget MIB]
//               [--prime BYTES] [--runs N] [--kernel auto|scalar|avx2|neon]
//               [--no-primer] [--tables 0-5] [--bypass] [--wide-mixer] [--total-only] FILE...
//
// For every file (and the concatenation of all of them, "TOTAL") prints the
// compressed size, ratio, compress and decompress throughput (best of
// --runs, MB = 10^6 bytes) and verifies the round trip byte for byte. Peak
// RSS of the whole process is printed at the end, so run one tier per process
// to read that tier's memory.
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#define ULTRA_HAVE_RUSAGE 1
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "rans_static.hpp"  // kernel override (internal header)
#include "ultracodec/engine.hpp"

namespace {

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path);
    std::exit(2);
  }
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ultracodec;
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <tier 1-4> [--threads N] [--block B] [--budget MiB] [--prime B] [--runs N] FILE...\n", argv[0]);
    return 2;
  }
  int tier = std::atoi(argv[1]);
  if (tier < 1 || tier > 4) return 2;
  EngineOptions opt;
  int runs = 1;
  bool each = true;
  std::vector<const char*> files;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return i + 1 < argc ? argv[++i] : (std::exit(2), nullptr); };
    if (a == "--threads") opt.threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--block") opt.block_size = std::strtoull(next(), nullptr, 0);
    else if (a == "--budget") opt.memory_budget = std::strtoull(next(), nullptr, 0) << 20;
    else if (a == "--prime") opt.prime_bytes = std::strtoull(next(), nullptr, 0);
    else if (a == "--runs") runs = std::atoi(next());
    else if (a == "--total-only") each = false;
    else if (a == "--no-primer") opt.tier4_primer = false;
    else if (a == "--tables") opt.tier4_tables = std::atoi(next());
    else if (a == "--bypass") opt.tier4_bypass = true;
    else if (a == "--wide-mixer") opt.tier4_narrow_mixer = false;
    else if (a == "--kernel") {
      std::string k = next();
      using detail::RansKernel;
      detail::rans_set_kernel(k == "scalar" ? RansKernel::Scalar : k == "avx2" ? RansKernel::Avx2
                              : k == "neon" ? RansKernel::Neon : RansKernel::Auto);
    } else files.push_back(argv[i]);
  }
  auto engine = create_engine(static_cast<TierLevel>(tier), opt);
  std::printf("%s threads=%u\n", std::string(engine->get_name()).c_str(), opt.threads);
  std::printf("%-12s %12s %12s %8s %10s %10s\n", "file", "bytes", "packed", "bpc", "comp MB/s", "dec MB/s");
  std::vector<std::uint8_t> all;
  std::vector<std::pair<std::string, std::vector<std::uint8_t>>> inputs;
  for (auto* f : files) {
    auto d = read_file(f);
    all.insert(all.end(), d.begin(), d.end());
    std::string name = f;
    if (auto s = name.rfind('/'); s != std::string::npos) name = name.substr(s + 1);
    if (each) inputs.emplace_back(name, std::move(d));
  }
  if (files.size() > 1 || !each) inputs.emplace_back("TOTAL(cat)", std::move(all));
  std::size_t sum_packed = 0, sum_raw = 0;
  double sum_ct = 0, sum_dt = 0;
  for (auto& [name, data] : inputs) {
    double ct = 1e30, dt = 1e30;
    std::vector<std::uint8_t> packed, back;
    for (int r = 0; r < runs; ++r) {
      double t0 = now();
      packed = engine->compress(data);
      double t1 = now();
      back = engine->decompress(packed);
      double t2 = now();
      ct = std::min(ct, t1 - t0);
      dt = std::min(dt, t2 - t1);
    }
    if (back != data) {
      std::printf("%s: ROUND TRIP FAILED\n", name.c_str());
      return 1;
    }
    std::printf("%-12s %12zu %12zu %8.4f %10.2f %10.2f\n", name.c_str(), data.size(), packed.size(),
                data.empty() ? 0.0 : 8.0 * static_cast<double>(packed.size()) / static_cast<double>(data.size()),
                static_cast<double>(data.size()) / ct / 1e6, static_cast<double>(data.size()) / dt / 1e6);
    if (name != "TOTAL(cat)") {
      sum_packed += packed.size();
      sum_raw += data.size();
      sum_ct += ct;
      sum_dt += dt;
    }
  }
  if (each && files.size() > 1)
    std::printf("%-12s %12zu %12zu %8.4f %10.2f %10.2f\n", "SUM(files)", sum_raw, sum_packed,
                8.0 * static_cast<double>(sum_packed) / static_cast<double>(sum_raw),
                static_cast<double>(sum_raw) / sum_ct / 1e6, static_cast<double>(sum_raw) / sum_dt / 1e6);
#if defined(ULTRA_HAVE_RUSAGE)
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  std::printf("peak RSS %.1f MiB\n", static_cast<double>(ru.ru_maxrss) / 1024.0);
#endif
  return 0;
}
