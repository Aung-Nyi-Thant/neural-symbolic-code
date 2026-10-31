// fixture_input.hpp - the input every archive in this directory was made
// from. Deterministic and self-contained (no library code), so the encoder
// builds of past commits (make_fixtures.sh) and today's test
// (test_fixtures.cpp) produce the same 80,000 bytes: prose-like words,
// numbered log lines, a binary run and a long repeat, over two 64 KiB blocks.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ultracodec::fixtures {

inline std::vector<std::uint8_t> fixture_input() {
  std::uint64_t s = 0x9E3779B97F4A7C15ull;
  const auto next = [&s] {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  };
  static const char* const words[] = {"the", "archive", "block", "stream", "model", "of", "and", "a", "to",
                                      "context", "symbol", "window", "frame", "is", "decoded", "byte"};
  std::string text;
  while (text.size() < 30000) {
    text += words[next() % 16];
    text += (next() % 9 == 0) ? ".\n" : " ";
  }
  for (int i = 0; text.size() < 52000; ++i)
    text += "2026-10-" + std::to_string(10 + i % 20) + " worker=" + std::to_string(next() % 8) + " offset=" +
            std::to_string(i * 4096) + " status=ok\n";
  std::vector<std::uint8_t> out(text.begin(), text.end());
  while (out.size() < 64000) out.push_back(static_cast<std::uint8_t>(next() >> 56));  // incompressible
  const std::size_t repeat_from = 1000;
  while (out.size() < 80000) out.push_back(out[repeat_from + (out.size() - 64000)]);  // a 16,000-byte repeat
  return out;
}

}  // namespace ultracodec::fixtures
