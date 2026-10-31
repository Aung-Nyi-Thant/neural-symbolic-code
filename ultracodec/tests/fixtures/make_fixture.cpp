// make_fixture: writes one fixture archive with the ultracodec library it is
// linked against. make_fixtures.sh builds it against past commits; it uses
// only the API those commits share, plus the variant options of the newest
// ones when UF_VARIANTS is defined.
//
//   make_fixture TIER OUTPUT [VARIANT]
//   VARIANT: t1-raw, t3-legacy, t4-wide-mixer, t4-bypass, t4-tables-5 (UF_VARIANTS only)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "fixture_input.hpp"
#include "ultracodec/engine.hpp"

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const int tier = std::atoi(argv[1]);
  if (tier < 1 || tier > 4) return 2;
  ultracodec::EngineOptions o;
  o.threads = 1;
  o.block_size = std::size_t{1} << 16;
  const std::string variant = argc > 3 ? argv[3] : "";
#if defined(UF_VARIANTS)
  if (variant == "t1-raw") o.tier1_entropy = false;
  else if (variant == "t3-legacy") o.tier3_compact = false;
  else if (variant == "t4-wide-mixer") o.tier4_narrow_mixer = false;
  else if (variant == "t4-bypass") o.tier4_bypass = true;
  else if (variant == "t4-tables-5") o.tier4_tables = 5;
  else if (!variant.empty()) return 2;
#else
  if (!variant.empty()) return 2;
#endif
  const auto input = ultracodec::fixtures::fixture_input();
  const auto archive = ultracodec::compress(static_cast<ultracodec::TierLevel>(tier), input, o);
  if (ultracodec::decompress(archive) != input) return 1;  // the encoder's own round trip, at that commit
  std::FILE* f = std::fopen(argv[2], "wb");
  if (!f || std::fwrite(archive.data(), 1, archive.size(), f) != archive.size() || std::fclose(f) != 0) return 1;
  return 0;
}
