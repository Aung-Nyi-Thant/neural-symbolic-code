// Conformance harness. Compresses the deterministic corpus with every profile, checks the round
// trip and encoder/decoder trace equality, and compares one line per (profile, vector) with the
// golden file:
//
//   hce_conformance --check conformance/golden_v1.txt   exit status 1 on any difference
//   hce_conformance --regen conformance/golden_v1.txt   rewrite it (ONLY for an intentional change,
//                                                        which also requires a new profile id)
//   hce_conformance --print                             print the current lines
#include <cinttypes>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core/kernels/crc32c.hpp"
#include "corpus.hpp"
#include "format/container.hpp"
#include "format/profiles.hpp"

namespace {

std::string hex32(uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%08" PRIx32, v);
    return buf;
}

std::string hex64(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016" PRIx64, v);
    return buf;
}

// Returns the golden lines for the current build; sets `ok` to false on a round-trip failure.
std::vector<std::string> current_lines(bool& ok) {
    std::vector<std::string> lines;
    const auto vectors = hce::corpus::conformance_vectors();
    for (const hce::Profile& profile : hce::profiles()) {
        lines.push_back("model " + std::to_string(profile.id) + " " + profile.name + " " +
                        hex32(hce::model_id(profile)));
        for (const auto& vec : vectors) {
            hce::CodecStats enc;
            hce::CodecStats dec;
            const auto packed = hce::compress(vec.data, profile.id, &enc);
            const auto unpacked = hce::decompress(packed, {}, &dec);
            if (unpacked != vec.data || enc.trace != dec.trace || enc.decisions != dec.decisions) {
                std::cerr << "ROUND TRIP FAILURE: profile " << int{profile.id} << " vector " << vec.name << "\n";
                ok = false;
            }
            lines.push_back(std::to_string(profile.id) + " " + vec.name + " " + std::to_string(vec.data.size()) +
                            " " + hex32(hce::crc32c(vec.data)) + " " + std::to_string(packed.size()) + " " +
                            hex32(hce::crc32c(packed)) + " " + hex64(enc.trace));
        }
    }
    return lines;
}

const char* kHeader =
    "# HCE format v1 conformance vectors. Do not edit by hand.\n"
    "# Regenerate only for an intentional format change (which also needs a new profile id):\n"
    "#   hce_conformance --regen conformance/golden_v1.txt\n"
    "# model <profile> <name> <model_id>\n"
    "# <profile> <vector> <input_size> <input_crc32c> <output_size> <output_crc32c> <trace_fnv1a64>\n";

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc >= 2 ? argv[1] : "";
    if (mode != "--check" && mode != "--regen" && mode != "--print") {
        std::cerr << "usage: hce_conformance --check FILE | --regen FILE | --print\n";
        return 2;
    }
    if (mode != "--print" && argc < 3) {
        std::cerr << "missing golden file path\n";
        return 2;
    }

    try {
        bool ok = true;
        const auto lines = current_lines(ok);
        if (mode == "--print") {
            for (const auto& line : lines) std::cout << line << "\n";
            return ok ? 0 : 1;
        }
        if (mode == "--regen") {
            std::ofstream out(argv[2], std::ios::binary);
            out << kHeader;
            for (const auto& line : lines) out << line << "\n";
            std::cout << "wrote " << lines.size() << " lines to " << argv[2] << "\n";
            return ok && out ? 0 : 1;
        }

        std::ifstream in(argv[2], std::ios::binary);
        if (!in) {
            std::cerr << "cannot open " << argv[2] << "\n";
            return 1;
        }
        std::vector<std::string> golden;
        for (std::string line; std::getline(in, line);) {
            if (!line.empty() && line[0] != '#') golden.push_back(line);
        }
        size_t mismatches = 0;
        for (size_t i = 0; i < std::max(golden.size(), lines.size()); ++i) {
            const std::string want = i < golden.size() ? golden[i] : "<missing>";
            const std::string got = i < lines.size() ? lines[i] : "<missing>";
            if (want != got) {
                std::cerr << "MISMATCH\n  golden: " << want << "\n  actual: " << got << "\n";
                ++mismatches;
            }
        }
        if (mismatches == 0 && ok) {
            std::cout << "conformance: " << lines.size() << " lines match\n";
            return 0;
        }
        std::cerr << "conformance: " << mismatches << " mismatching line(s)\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
