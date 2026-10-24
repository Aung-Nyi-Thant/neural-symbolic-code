// token_stream.hpp - what Phase 2 adds to the Phase 1 interface (structural_parser.hpp).
//
// The Phase 1 header declares tokenize() and serialize_tokens(); this header
// fixes the contracts they implement and adds the inverse operations the
// compressor needs:
//
//   * tokenize() is total and lossless: every byte of the input belongs to
//     exactly one token, so detokenize(tokenize(s)) == s for every s, binary
//     or not. No token is longer than kMaxTokenBytes (long comments and string
//     literals are split). checked_integer_id interns (type, value) in order
//     of first appearance within one call.
//   * serialize_tokens() writes the HST1 image below; deserialize_tokens()
//     parses it with every length and count checked against TokenStreamLimits,
//     so a hostile image costs bounded memory and never reads out of bounds.
//
// HST1 image, little-endian, varints are LEB128 (at most 10 bytes):
//   "HST1"  varint token_count
//   token_count x { u8 type  varint value_len  value  varint zigzag(line delta)  varint checked_integer_id }
#pragma once

#include "hc_framed.hpp"
#include "structural/structural_parser.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hce::structural {

inline constexpr std::size_t kMaxTokenBytes = 4096;
inline constexpr std::size_t kMaxSourceBytes = std::size_t{1} << 30;  // per tokenize() call
inline constexpr std::uint8_t kTokenTypes = 6;
inline constexpr std::array<std::byte, 4> kTokenStreamMagic{std::byte{'H'}, std::byte{'S'}, std::byte{'T'},
                                                            std::byte{'1'}};

// Keyword tables: which identifier-shaped words become KEYWORD tokens. The
// table only shapes the tokens (and so the compression); every table is
// lossless, and decoders never need to know which one was used. `generic`
// is the union of the others and is what StructuralPreprocessor::tokenize() uses.
enum class Language : std::uint8_t { generic = 0, c, cpp, python, javascript, typescript, java, go, rust };
inline constexpr std::uint8_t kLanguages = 9;

const char* to_string(Language lang) noexcept;
// generic, c, cpp (c++, cxx), python (py), javascript (js), typescript (ts),
// java, go, rust (rs). "auto" is not a language: resolve it with language_for_path().
std::optional<Language> parse_language(std::string_view name) noexcept;
// By file extension (.py, .cpp, .hpp, .h, .c, .js, .ts, .java, .go, .rs, ...); generic otherwise.
Language language_for_path(const std::filesystem::path& path) noexcept;

// tokenize() with the keyword table of `lang`; the same contract otherwise.
std::vector<StructuralToken> tokenize(std::string_view source, Language lang);

struct TokenStreamLimits {
  hc::u64 max_tokens = hc::u64{1} << 28;
  hc::u64 max_value_bytes = hc::u64{1} << 30;  // sum of all values
  hc::u32 max_token_bytes = kMaxTokenBytes;
};

// Parses one serialize_tokens() image into `out` (replacing its contents).
// bad_magic, header_corrupt (truncated or malformed), limit_output (a limit
// exceeded), out_of_memory. On error `out` is left empty.
hc::Errc deserialize_tokens(hc::Bytes image, std::vector<StructuralToken>& out,
                            const TokenStreamLimits& limits = {}) noexcept;

// The source the tokens were cut from: their values, concatenated.
std::string detokenize(const std::vector<StructuralToken>& tokens);

// Structural state as the estimators see it, advanced token by token:
//   scope_depth           '{' minus '}', never below 0
//   block_brackets_count  '(' and '[' minus ')' and ']', never below 0
//   indentation_level     width in bytes of the current line's leading whitespace
// at_line_start holds from a newline until the line's first other token.
struct StructuralTracker {
  StructuralContextState state{};
  bool at_line_start = true;

  void advance(TokenType type, std::string_view value) noexcept;
};

// LEB128, shared by the token and model images.
void put_varint(std::vector<std::uint8_t>& out, hc::u64 v);
// Reads one varint at `at`, advancing it. False when truncated or longer than 10 bytes.
bool get_varint(hc::Bytes in, std::size_t& at, hc::u64& v) noexcept;

}  // namespace hce::structural
