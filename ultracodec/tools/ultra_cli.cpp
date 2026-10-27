// ultra: compress or decompress one file with ultracodec.
//
//   ultra c [-t TIER] [-T THREADS] INPUT OUTPUT     compress (default tier 2)
//   ultra d [-T THREADS] INPUT OUTPUT               decompress (tier from header)
//   ultra i INPUT                                   print the archive header
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ultracodec/engine.hpp"

namespace {

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw ultracodec::Error(std::string("cannot read ") + path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void write_file(const char* path, const std::vector<std::uint8_t>& d) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size()));
  if (!f) throw ultracodec::Error(std::string("cannot write ") + path);
}

int usage() {
  std::fprintf(stderr, "usage: ultra c [-t TIER] [-T THREADS] IN OUT | ultra d [-T THREADS] IN OUT | ultra i IN\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ultracodec;
  if (argc < 3) return usage();
  const std::string mode = argv[1];
  int tier = 2;
  EngineOptions opt;
  std::vector<const char*> paths;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-t") && i + 1 < argc) tier = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-T") && i + 1 < argc) opt.threads = static_cast<unsigned>(std::atoi(argv[++i]));
    else paths.push_back(argv[i]);
  }
  try {
    if (mode == "i" && paths.size() == 1) {
      auto a = read_file(paths[0]);
      ArchiveHeader h = peek_header(a);
      std::printf("%s, %llu bytes, %u blocks of %zu, prefix %u, flags %u, model %u\n",
                  std::string(tier_name(h.tier)).c_str(), static_cast<unsigned long long>(h.uncompressed_size),
                  h.block_count, h.block_size(), h.prime_bytes, h.flags, h.model_param);
      return 0;
    }
    if (paths.size() != 2) return usage();
    if (mode == "c") {
      if (tier < 1 || tier > 4) return usage();
      write_file(paths[1], compress(static_cast<TierLevel>(tier), read_file(paths[0]), opt));
    } else if (mode == "d") {
      write_file(paths[1], decompress(read_file(paths[0]), opt));
    } else {
      return usage();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ultra: %s\n", e.what());
    return 1;
  }
  return 0;
}
