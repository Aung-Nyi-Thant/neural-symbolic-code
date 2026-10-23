#include "corpus.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string_view>

namespace hce::corpus {
namespace {

void append(std::vector<uint8_t>& out, std::string_view s) { out.insert(out.end(), s.begin(), s.end()); }

constexpr std::array<std::string_view, 64> kWords{
    "the",   "of",     "and",    "to",      "in",          "a",       "is",    "that",
    "for",   "it",     "as",     "was",     "with",        "be",      "by",    "on",
    "not",   "he",     "this",   "are",     "or",          "his",     "from",  "at",
    "which", "but",    "have",   "an",      "had",         "they",    "you",   "were",
    "their", "one",    "all",    "we",      "can",         "her",     "has",   "there",
    "been",  "if",     "more",   "when",    "will",        "would",   "who",   "so",
    "no",    "model",  "context", "stream", "probability", "history", "frame", "table",
    "entropy", "coder", "mixer", "weight",  "signal",      "pattern", "byte",  "bit"};

constexpr std::array<std::string_view, 5> kServices{"api-gateway", "auth", "billing", "search", "storage"};
constexpr std::array<std::string_view, 4> kMethods{"GET", "POST", "PUT", "DELETE"};
constexpr std::array<std::string_view, 3> kResources{"items", "users", "orders"};

}  // namespace

std::vector<uint8_t> random_bytes(size_t n, uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<uint8_t> out(n);
    for (uint8_t& b : out) b = static_cast<uint8_t>(rng.next() >> 56);
    return out;
}

std::vector<uint8_t> text(size_t n, uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<uint8_t> out;
    out.reserve(n + 64);
    while (out.size() < n) {
        const uint32_t sentence_words = 5 + rng.below(12);
        for (uint32_t w = 0; w < sentence_words; ++w) {
            const std::string_view word = kWords[std::min(rng.below(64), rng.below(64))];  // Zipf-like skew
            if (w == 0) {
                out.push_back(static_cast<uint8_t>(word[0] - 'a' + 'A'));
                append(out, word.substr(1));
            } else {
                out.push_back(' ');
                append(out, word);
            }
            if (w + 1 < sentence_words && rng.below(10) == 0) out.push_back(',');
        }
        out.push_back('.');
        out.push_back(rng.below(6) == 0 ? '\n' : ' ');
    }
    out.resize(n);
    return out;
}

std::vector<uint8_t> logs(size_t n, uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<uint8_t> out;
    out.reserve(n + 256);
    uint64_t ms = 0;  // milliseconds since 2026-01-01T00:00:00Z
    while (out.size() < n) {
        ms += rng.below(250);
        const uint32_t level_roll = rng.below(100);
        const char* level = level_roll < 70 ? "INFO " : level_roll < 85 ? "DEBUG" : level_roll < 95 ? "WARN " : "ERROR";
        const uint32_t status_roll = rng.below(100);
        const unsigned status = status_roll < 80 ? 200u : status_roll < 88 ? 201u : status_roll < 93 ? 404u
                                : status_roll < 97 ? 400u : 500u;
        const auto service = kServices[rng.below(kServices.size())];
        const auto method = kMethods[std::min(rng.below(4), rng.below(4))];
        const auto resource = kResources[rng.below(kResources.size())];
        char line[256];
        const int len = std::snprintf(
            line, sizeof line,
            "2026-01-01T%02u:%02u:%02u.%03uZ %s [%.*s] req=%08x method=%.*s path=/v1/%.*s/%u status=%u "
            "latency_ms=%u\n",
            static_cast<unsigned>(ms / 3600000 % 24), static_cast<unsigned>(ms / 60000 % 60),
            static_cast<unsigned>(ms / 1000 % 60), static_cast<unsigned>(ms % 1000), level,
            static_cast<int>(service.size()), service.data(), static_cast<unsigned>(rng.next() >> 32),
            static_cast<int>(method.size()), method.data(), static_cast<int>(resource.size()), resource.data(),
            1 + rng.below(5000), status, 1 + rng.below(40) * rng.below(8));
        append(out, std::string_view(line, static_cast<size_t>(len)));
    }
    out.resize(n);
    return out;
}

std::vector<Vector> conformance_vectors() {
    std::vector<Vector> v;
    v.push_back({"empty", {}});
    v.push_back({"one-byte", {0x41}});
    v.push_back({"zeros-4k", std::vector<uint8_t>(4096, 0)});
    std::vector<uint8_t> ramp(4096);
    for (size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<uint8_t>(i);
    v.push_back({"ramp-4k", ramp});
    v.push_back({"random-80k", random_bytes(80000, 1)});
    v.push_back({"text-150k", text(150000, 2)});
    v.push_back({"logs-200k", logs(200000, 3)});
    // Text, then a random block spanning whole 64 KiB frames, then logs: exercises STORED frames
    // between modeled ones (profile 1) and the model carrying state across them.
    std::vector<uint8_t> mixed = text(50000, 4);
    const auto noise = random_bytes(140000, 5);
    const auto tail = logs(60000, 6);
    mixed.insert(mixed.end(), noise.begin(), noise.end());
    mixed.insert(mixed.end(), tail.begin(), tail.end());
    v.push_back({"mixed-250k", mixed});
    // Text and noise, then both again: long-range repeats that pin COPY and match prediction.
    std::vector<uint8_t> repeats = text(40000, 7);
    const auto repeated_noise = random_bytes(60000, 8);
    repeats.insert(repeats.end(), repeated_noise.begin(), repeated_noise.end());
    const std::vector<uint8_t> first_pass = repeats;
    repeats.insert(repeats.end(), first_pass.begin(), first_pass.end());
    v.push_back({"repeats-200k", repeats});
    return v;
}

}  // namespace hce::corpus
