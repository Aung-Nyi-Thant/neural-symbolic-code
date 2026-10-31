// ultra: compress or decompress one file with ultracodec.
//
//   ultra c [-t TIER] [-T THREADS] INPUT OUTPUT     compress (default tier 2)
//   ultra d [-T THREADS] INPUT OUTPUT               decompress (tier from header)
//   ultra i INPUT                                   print the archive header
//
// TIER is 1..4; THREADS is 0..1024, 0 meaning one per CPU. OUTPUT appears
// (or is replaced) only once complete and synced: on any failure an existing
// OUTPUT keeps its content. Exit status: 0 success, 1 failure, 2 usage.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "tool_util.hpp"
#include "ultracodec/engine.hpp"

namespace {

int usage(const char* why = nullptr) {
  if (why) std::fprintf(stderr, "ultra: %s\n", why);
  std::fprintf(stderr, "usage: ultra c [-t TIER] [-T THREADS] IN OUT | ultra d [-T THREADS] IN OUT | ultra i IN\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ultracodec;
  namespace tools = ultracodec::tools;
  if (argc < 3) return usage();
  const std::string mode = argv[1];
  if (mode != "c" && mode != "d" && mode != "i") return usage("unknown mode");
  int tier = 2;
  EngineOptions opt;
  std::vector<const char*> paths;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-t" || a == "-T") {
      if (i + 1 >= argc) return usage((a + " needs a value").c_str());
      const char* v = argv[++i];
      const auto n = a == "-t" ? tools::parse_uint(v, 1, 4) : tools::parse_uint(v, 0, 1024);
      if (!n) return usage((a + " " + v + ": expected " + (a == "-t" ? "a tier 1..4" : "0..1024 threads")).c_str());
      if (a == "-t") tier = static_cast<int>(*n);
      else opt.threads = static_cast<unsigned>(*n);
      if (a == "-t" && mode != "c") return usage("-t applies to compression only");
    } else if (a.size() > 1 && a[0] == '-') {
      return usage(("unknown option " + a).c_str());
    } else {
      paths.push_back(argv[i]);
    }
  }
  try {
    if (mode == "i") {
      if (paths.size() != 1) return usage();
      const auto a = tools::read_file(paths[0]);
      const ArchiveHeader h = peek_header(a);
      std::printf("%s, %llu bytes, %u blocks of %zu, prefix %u, flags %u, model %u\n",
                  std::string(tier_name(h.tier)).c_str(), static_cast<unsigned long long>(h.uncompressed_size),
                  h.block_count, h.block_size(), h.prime_bytes, h.flags, h.model_param);
      return 0;
    }
    if (paths.size() != 2) return usage();
    std::error_code ec;
    if (std::filesystem::equivalent(paths[0], paths[1], ec) && !ec) {
      std::fprintf(stderr, "ultra: %s and %s are the same file\n", paths[0], paths[1]);
      return 1;
    }
    const auto in = tools::read_file(paths[0]);
    const auto out = mode == "c" ? compress(static_cast<TierLevel>(tier), in, opt) : decompress(in, opt);
    bool durable = true;
    tools::write_file(paths[1], out, &durable);
    if (!durable) {
      std::fprintf(stderr, "ultra: %s written, but its directory could not be synced: it may not survive a crash\n",
                   paths[1]);
      return 1;
    }
  } catch (const tools::IoError& e) {
    std::fprintf(stderr, "ultra: %s\n", e.what.c_str());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ultra: %s\n", e.what());
    return 1;
  }
  return 0;
}
