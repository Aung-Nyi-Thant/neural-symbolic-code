// replay: runs a fuzz target's LLVMFuzzerTestOneInput() over files and
// directories (recursively), for builds without libFuzzer. ctest uses it to
// run every target over its seed corpus and its saved regressions.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace fs = std::filesystem;

static void run(const fs::path& p, int& n) {
  std::ifstream f(p, std::ios::binary);
  const std::vector<char> d((std::istreambuf_iterator<char>(f)), {});
  std::printf("  %s (%zu bytes)\n", p.string().c_str(), d.size());
  (void)LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(d.data()), d.size());
  ++n;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // the last line names the input that crashed
  int n = 0;
  for (int i = 1; i < argc; ++i) {
    const fs::path arg = argv[i];
    if (fs::is_directory(arg)) {
      std::vector<fs::path> files;
      for (const auto& e : fs::recursive_directory_iterator(arg))
        if (e.is_regular_file()) files.push_back(e.path());
      std::sort(files.begin(), files.end());
      for (const auto& f : files) run(f, n);
    } else if (fs::is_regular_file(arg)) {
      run(arg, n);
    } else {
      std::fprintf(stderr, "replay: %s: no such file or directory\n", argv[i]);
      return 2;
    }
  }
  std::printf("replayed %d input(s)\n", n);
  return n ? 0 : 2;
}
