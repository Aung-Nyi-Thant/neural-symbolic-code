// Backward compatibility: archives written by the encoders of past commits
// (fixtures/, made by fixtures/make_fixtures.sh from those commits' sources,
// never by today's encoder) must decode to fixture_input() with today's
// decoder, on one thread and on four. Separate from the round-trip and
// byte-identical encoder tests in test_ultracodec.cpp. The archives listed in
// fixtures/KNOWN_BREAKS (formats changed in place before any release) must
// instead fail with an error: never decode to other bytes, never crash.
//
//   test_ultra_fixtures FIXTURE_DIR
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures/fixture_input.hpp"
#include "ultracodec/engine.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 2) {
    std::printf("usage: test_ultra_fixtures FIXTURE_DIR\n");
    return 2;
  }
  const fs::path dir = argv[1];
  std::ifstream manifest(dir / "MANIFEST");
  if (!manifest) {
    std::printf("FAIL: no %s\n", (dir / "MANIFEST").string().c_str());
    return 1;
  }
  std::set<std::pair<std::string, std::string>> breaks;  // (commit, tier)
  {
    std::ifstream known(dir / "KNOWN_BREAKS");
    std::string l;
    while (std::getline(known, l)) {
      if (l.empty() || l[0] == '#') continue;
      std::istringstream in(l);
      std::string commit, tier;
      in >> commit >> tier;
      breaks.emplace(commit, tier);
    }
  }
  const std::vector<std::uint8_t> expected = ultracodec::fixtures::fixture_input();
  int failures = 0, checked = 0;
  std::set<std::string> commits;
  std::string line;
  while (std::getline(manifest, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string file, commit, tier, variant;
    std::size_t bytes = 0;
    fields >> file >> commit >> tier >> variant >> bytes;
    std::ifstream f(dir / file, std::ios::binary);
    const std::vector<std::uint8_t> archive((std::istreambuf_iterator<char>(f)), {});
    std::string why;
    const bool known_break = breaks.count({commit, tier}) != 0;
    if (!f.is_open() && archive.empty()) {
      why = "missing";
    } else if (archive.size() != bytes) {
      why = "has " + std::to_string(archive.size()) + " bytes, MANIFEST says " + std::to_string(bytes);
    } else if (known_break) {
      try {
        (void)ultracodec::decompress(archive);
        why = "decoded, but KNOWN_BREAKS says it does not: remove it from there";
      } catch (const ultracodec::Error& e) {
        std::printf("     (rejected as listed in KNOWN_BREAKS: %s)\n", e.what());
      }
    } else {
      for (const unsigned threads : {1u, 4u}) {
        ultracodec::EngineOptions o;
        o.threads = threads;
        try {
          if (ultracodec::peek_header(archive).tier != static_cast<ultracodec::TierLevel>(std::stoi(tier)))
            why = "header names another tier";
          else if (ultracodec::decompress(archive, o) != expected)
            why = "decodes to different bytes (" + std::to_string(threads) + " thread(s))";
        } catch (const std::exception& e) {
          why = std::string(e.what()) + " (" + std::to_string(threads) + " thread(s))";
        }
        if (!why.empty()) break;
      }
    }
    std::printf("%-4s %s tier %s %s: %s\n", !why.empty() ? "FAIL" : known_break ? "rej" : "ok", commit.c_str(), tier.c_str(), variant.c_str(),
                why.empty() ? file.c_str() : (file + ": " + why).c_str());
    failures += !why.empty();
    ++checked;
    commits.insert(commit);
  }
  std::printf("%d archive(s) from %zu commit(s), %zu known break(s), %d failure(s)\n", checked, commits.size(),
              breaks.size(), failures);
  return failures || checked == 0 ? 1 : 0;
}
