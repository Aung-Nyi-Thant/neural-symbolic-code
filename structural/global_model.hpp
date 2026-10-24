// global_model.hpp - Phase 2, pass 1: the frozen global model.
//
// Pass 1 reads the whole token stream once and freezes what it learned into a
// GlobalModel that never changes afterwards:
//   * the dictionary: every distinct (type, value), numbered in order of
//     first appearance over the whole input, so that symbols introduced
//     together get nearby IDs and the codec's gamma code spends few bits on
//     the early, common ones (measured 2-3% smaller overall than numbering by
//     frequency, on Python and C++ corpora);
//   * the global successor table: for every symbol, its most frequent
//     successor over the whole input (a two-slot Misra-Gries count: exact for
//     any successor seen more than a third of the time), kept when it held at
//     least two of its successors: a prediction that reaches across every
//     frame boundary, which the frame-local models cannot;
//   * per-symbol structural traits, so the estimators never inspect strings.
//
// Thread safety: a GlobalModel is immutable once freeze() or parse() returns.
// Every encoder and decoder worker reads the same instance through const
// member functions only; publishing it before the workers start (thread
// creation) is the only synchronization needed.
#pragma once

#include "hc_framed.hpp"
#include "structural/structural_parser.hpp"
#include "structural/token_stream.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hce::structural {

inline constexpr hc::u32 kNoSymbol = 0xFFFFFFFFu;
inline constexpr std::array<std::byte, 4> kGlobalModelMagic{std::byte{'H'}, std::byte{'G'}, std::byte{'M'},
                                                            std::byte{'1'}};

struct GlobalModelLimits {
  hc::u32 max_symbols = hc::u32{1} << 22;           // distinct tokens
  hc::u64 max_dictionary_bytes = hc::u64{1} << 28;  // sum of their values
  hc::u32 max_token_bytes = kMaxTokenBytes;
};

// Structural role of a symbol, precomputed from its type and value.
enum SymbolTrait : std::uint8_t {
  kTraitNewline = 1,       // STRUCTURE holding '\n'
  kTraitOpenBrace = 2,     // {
  kTraitCloseBrace = 4,    // }
  kTraitOpenBracket = 8,   // ( [
  kTraitCloseBracket = 16, // ) ]
  kTraitIndent = 32,       // INDENTATION
};

class GlobalModel {
 public:
  GlobalModel() = default;
  GlobalModel(GlobalModel&&) noexcept = default;
  GlobalModel& operator=(GlobalModel&&) noexcept = default;
  GlobalModel(const GlobalModel&) = delete;  // the encoder index points into blob_
  GlobalModel& operator=(const GlobalModel&) = delete;

  hc::u32 size() const noexcept { return static_cast<hc::u32>(types_.size()); }
  // Preconditions for the accessors below: id < size().
  TokenType type(hc::u32 id) const noexcept { return static_cast<TokenType>(types_[id]); }
  std::string_view value(hc::u32 id) const noexcept {
    return {blob_.data() + offsets_[id], static_cast<std::size_t>(offsets_[id + 1] - offsets_[id])};
  }
  hc::u32 successor(hc::u32 id) const noexcept { return succ_[id]; }  // kNoSymbol: none
  std::uint8_t traits(hc::u32 id) const noexcept { return traits_[id]; }
  std::uint8_t indent(hc::u32 id) const noexcept { return indent_[id]; }  // INDENTATION width, at most 255
  hc::u64 dictionary_bytes() const noexcept { return blob_.size(); }

  // Encoder side: the ID of (type, value), or kNoSymbol. Only models built by
  // GlobalModelBuilder::freeze() carry the index; parsed models do not.
  hc::u32 find(TokenType type, std::string_view value) const noexcept;

  // HGM1 image: "HGM1" varint n, n x {u8 type, varint len, value}, n x varint(successor + 1).
  std::vector<std::uint8_t> serialize() const;
  // Checks every count, length, type and successor against `limits` and the
  // image; out is replaced only on success. bad_magic, header_corrupt,
  // limit_output, out_of_memory.
  static hc::Errc parse(hc::Bytes image, const GlobalModelLimits& limits, GlobalModel& out) noexcept;

 private:
  friend class GlobalModelBuilder;
  struct Key {
    std::string_view value;
    std::uint8_t type;
    bool operator==(const Key&) const noexcept = default;
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
      return std::hash<std::string_view>{}(k.value) ^ (std::size_t{k.type} * 0x9E3779B97F4A7C15ull);
    }
  };
  void derive_traits();

  std::vector<char> blob_;          // values, in ID order
  std::vector<hc::u64> offsets_;    // size() + 1 entries
  std::vector<std::uint8_t> types_;
  std::vector<std::uint8_t> traits_;
  std::vector<std::uint8_t> indent_;
  std::vector<hc::u32> succ_;
  std::unordered_map<Key, hc::u32, KeyHash> index_;
};

// Pass 1 accumulator. One thread feeds it the token stream in order.
class GlobalModelBuilder {
 public:
  explicit GlobalModelBuilder(const GlobalModelLimits& limits = {}) : lim_(limits) {}

  // Adds the next tokens of the stream; successive calls continue one another.
  // limit_output when the dictionary outgrows the limits, out_of_memory.
  hc::Errc add(const std::vector<StructuralToken>& tokens) noexcept;
  // Freezes the symbols in order of first appearance, with the successor
  // table and the encoder index, into `out`.
  hc::Errc freeze(GlobalModel& out) noexcept;

  hc::u64 tokens() const noexcept { return tokens_; }
  hc::u32 symbols() const noexcept { return static_cast<hc::u32>(values_.size()); }

 private:
  GlobalModelLimits lim_;
  std::unordered_map<std::string, hc::u32> ids_;  // type byte, then the value -> provisional ID
  std::vector<const std::string*> values_;        // provisional ID -> its key (stable: node-based map)
  std::vector<hc::u64> counts_;
  std::vector<hc::u32> cand0_, cand1_;  // two-slot Misra-Gries count over each symbol's successors
  std::vector<hc::u32> n0_, n1_;
  std::string key_;
  hc::u32 prev_ = kNoSymbol;
  hc::u64 tokens_ = 0, bytes_ = 0;
};

}  // namespace hce::structural
