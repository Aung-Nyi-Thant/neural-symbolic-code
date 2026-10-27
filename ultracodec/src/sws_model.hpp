// Tier 4 context-mixing model (a port of the HCE "Ultra v4" lean model).
//
// Per bit it mixes 22 hashed contexts (orders 0-6, 8, 10, 12, 16, 24, 32,
// two sparse, three word, two column, two indirect) and a match model in
// four int32 mixer weight sets plus a final mixer, then refines the result
// with four APMs. Context tables use 64-byte buckets of four 16-byte nibble
// slots (8-bit check + 15 bit-history states), so a nibble costs one cache
// line per context.
//
// All arithmetic is integer and identical on every platform; the SIMD mixer
// kernels (AVX2 / NEON) produce the same results as the scalar ones.
//
// The model is copyable. Tier 4 trains one model on the primer and the
// first block, packs that snapshot (Model::pack) and copies it into every
// worker; a copy re-runs the context lookup so its slot pointers refer to its
// own tables.
#pragma once

#include <array>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "common.hpp"

namespace ultracodec::detail::sws {

// Makes dst an exact copy of src, reusing dst's mapping when the sizes match. An empty src is a
// packed model's released array: dst is left alone, and Model::operator= fills it from the pack.
template <class T>
void copy_big(BigArray<T>& dst, const BigArray<T>& src) {
  if (!src.size()) return;
  if (dst.size() != src.size()) dst.reset(src.size());
  std::memcpy(dst.data(), src.data(), src.size() * sizeof(T));
}

inline constexpr std::size_t kOrders = 13, kSparse = 2, kWord = 3, kColumn = 2, kIndirect = 2;
inline constexpr std::size_t kContexts = kOrders + kSparse + kWord + kColumn + kIndirect;  // 22
inline constexpr std::size_t kMatchLane = kContexts, kBiasLane = kContexts + 2;
inline constexpr std::size_t kLanes = 32;  // 22 contexts + 2 match inputs + bias, zero padded
inline constexpr std::size_t kSets = 4;
inline constexpr std::array<std::size_t, kSets> kSetRows{256, 64 * 256, 256 * 8, 2048};
inline constexpr std::size_t kMixRows = kSetRows[0] + kSetRows[1] + kSetRows[2] + kSetRows[3];
inline constexpr std::int32_t kWeightLimit = std::int32_t{1} << 19;
inline constexpr unsigned kBucketBits = 17;  // Full layout's per-context cap: 2^17 x 64 B = 8 MiB

// Per-context table caps. The archive records the layout, because table sizes change the coded
// bits. Measured on 2 MiB blocks after a 2 MiB snapshot (Python, Perl, prose, XML):
//   Full        all contexts 2^17 buckets                        ~194 MiB per model
//   Balanced    orders 2-4 and 16/24/32 at 2^15                   ~158 MiB, +0.20% size
//   Compact     Balanced, plus orders 5-12 at 2^16                ~138 MiB, +0.52% size
//   Small       every context at 2^15                             ~74 MiB,  +2.2% size
//   Rebalanced  bits moved from contexts that do not use them     ~159 MiB, -0.23% vs Balanced
//               (column, indirect, sparse, orders 2/16) to the starved ones (orders 3-5,
//               word pair, two words + byte); 2^20 match table
//   Lean        Balanced minus the unused bits, 2^20 match table  ~123 MiB, +0.07% vs Balanced
// Speed is the same for all of them: the model is bound by instructions, not by cache misses.
enum class TableLayout : u8 { Full = 0, Balanced = 1, Compact = 2, Small = 3, Rebalanced = 4, Lean = 5 };
inline constexpr unsigned kTableLayouts = 6;
inline constexpr std::int32_t kLearningRate = 6;

// Static tables of the model arithmetic, built once by integer-only code.
inline constexpr std::int32_t kLogitMin = -3071, kLogitMax = 3071, kSquashOffset = 3072;
struct Tables {
  std::array<u16, 6144> squash;    // squash[x + 3072] = 65536 / (1 + e^(-x/256)), in [1, 65535]
  std::array<std::int16_t, 4096> stretch;  // inverse of squash, indexed by p16 >> 4
  std::array<u32, 1024> recip;     // 2^16 / (n + 1.5): the StateMap adaptation rate
  std::array<std::array<u8, 2>, 256> next;  // bit-history state machine: next[state][bit]
  std::array<u8, 256> n0, n1;      // counts each state stands for
  std::array<u32, 256> p22_init;   // (2*n1 + 1) / (2*(n0 + n1) + 2), 22-bit fixed point
};
const Tables& tables();

// Adaptive map from a bit-history state to a probability. Entry = p22 << 10 | n.
class StateMap {
 public:
  StateMap() = default;
  explicit StateMap(const Tables& t) {
    for (std::size_t s = 0; s < entries_.size(); ++s) entries_[s] = t.p22_init[s] << 10;
  }
  u32 p16(u8 state) const noexcept { return entries_[state] >> 16; }
  void update(u8 state, u32 bit, const Tables& t) noexcept;

 private:
  std::array<u32, 256> entries_{};
};

// Bytes seen so far; the encoder's copy of the input, the decoder's output.
struct History {
  const u8* data = nullptr;
  std::size_t position = 0;
  u8 at(std::size_t back) const noexcept { return position >= back ? data[position - back] : 0; }
};

// Partial byte state.
struct Bits {
  u32 c0 = 1, bit = 0;
  u8 c1 = 0, c2 = 0;
  bool advance(u32 value) noexcept {
    c0 = c0 * 2 + value;
    ++bit;
    if (c0 < 256) return false;
    c2 = c1;
    c1 = static_cast<u8>(c0);
    c0 = 1;
    bit = 0;
    return true;
  }
};

// Fixed slots: a 64 B bucket holds four 16 B slots (8-bit check + 15 nibble-node states).
class SlotTable {
  friend class Model;

 public:
  struct Slot {
    u8 check = 0;
    std::array<u8, 15> states{};
  };
  struct alignas(64) Bucket {
    std::array<Slot, 4> slots{};
  };
  static_assert(sizeof(Slot) == 16 && sizeof(Bucket) == 64);
  SlotTable() = default;
  SlotTable(const SlotTable& o) { *this = o; }
  SlotTable& operator=(const SlotTable& o) {
    copy_big(buckets_, o.buckets_);
    bits_ = o.bits_;
    return *this;
  }
  void reset(unsigned bits);
  // Replaces the slot with the least evidence at node 1; ties keep the lowest slot index.
  u8* find(u64 hash, const Tables& t) noexcept;
  void prefetch(u64 hash) const noexcept;

 private:
  BigArray<Bucket> buckets_;
  unsigned bits_ = 0;
};

// Byte-level context hashes, including word, column and indirect state.
class Frontend {
 public:
  void reset();
  void observe(const History& h, u8 byte) noexcept;  // h already includes byte
  void contexts(const History& h, std::array<u64, kContexts>& out) const noexcept;

 private:
  u64 word0_ = 0, word1_ = 0, word2_ = 0;
  std::size_t line_start_ = 0, previous_line_ = 0;
  u32 column_ = 0, indent_ = 0;
  bool in_indent_ = true;
  std::vector<u16> indirect1_, indirect2_;
};

class Contexts {
  friend class Model;

 public:
  void reset(std::size_t capacity, TableLayout layout);
  void lookup(const std::array<u64, kContexts>& hashes, u32 c0) noexcept;
  // Writes stretch(p) for each context; returns how many contexts have history.
  u32 predict(std::int32_t* inputs) noexcept;
  void update(u32 bit) noexcept;

 private:
  const Tables* t_ = &tables();
  std::array<SlotTable, kContexts> tables_{};
  std::array<StateMap, kContexts> maps_{};
  std::array<u8*, kContexts> slots_{};
  std::array<u8, kContexts> states_{};
  u32 node_ = 1;
};

class Match {
  friend class Model;

 public:
  Match() = default;
  Match(const Match& o) { *this = o; }
  Match& operator=(const Match& o) {
    copy_big(table_, o.table_);
    maps_ = o.maps_;
    pointer_ = o.pointer_;
    length_ = o.length_;
    expected_ = o.expected_;
    context_ = o.context_;
    bits_ = o.bits_;
    return *this;
  }
  void reset(unsigned bits);
  void predict(const History& h, u32 bit, std::int32_t* inputs) noexcept;
  void update(u32 bit) noexcept;
  void observe(const History& h) noexcept;
  u32 length() const noexcept { return length_; }
  u32 expected() const noexcept { return expected_; }
  // The byte the current match predicts; valid while length() > 0.
  u8 next_byte(const History& h) const noexcept { return h.data[pointer_]; }

 private:
  const Tables* t_ = &tables();
  BigArray<u32> table_;
  std::array<StateMap, 64> maps_{};
  std::size_t pointer_ = 0;
  u32 length_ = 0, expected_ = 0, context_ = 0;
  unsigned bits_ = 20;
};

class Mixer {
  friend class Model;

 public:
  Mixer() = default;
  Mixer(const Mixer& o) { *this = o; }
  Mixer& operator=(const Mixer& o) {
    copy_big(weights_, o.weights_);
    final_weights_ = o.final_weights_;
    rows_ = o.rows_;
    set_probability_ = o.set_probability_;
    final_inputs_ = o.final_inputs_;
    final_row_ = o.final_row_;
    mixed_ = o.mixed_;
    return *this;
  }
  static constexpr std::int32_t kInitialWeight = 65536 / 8;
  void reset();
  u32 predict(const std::int32_t* inputs, u32 hits, u32 match_length, u32 match_bit, const Bits& bits) noexcept;
  void train(const std::int32_t* inputs, u32 bit) noexcept;

 private:
  const Tables* t_ = &tables();
  BigArray<std::int32_t> weights_;
  std::vector<std::int32_t> final_weights_;
  std::array<std::size_t, kSets> rows_{};
  std::array<u32, kSets> set_probability_{};
  std::array<std::int32_t, kSets + 1> final_inputs_{};
  std::size_t final_row_ = 0;
  u32 mixed_ = 32768;
};

class Calibration {
  friend class Model;

 public:
  void reset();
  u16 predict(u32 mixed, u32 match_length, u32 match_bit, const Bits& bits) noexcept;
  void update(u32 bit) noexcept;

 private:
  struct Apm {
    Apm() = default;
    Apm(const Apm& o) { *this = o; }
    Apm& operator=(const Apm& o) {
      copy_big(values, o.values);
      index = o.index;
      return *this;
    }
    BigArray<u16> values;
    std::size_t index = 0;
    static std::array<u16, 33> initial();  // the value of every untouched row
    void reset(std::size_t rows);
    u32 predict(std::size_t row, u32 low, u32 weight) noexcept;
    void update(u32 bit) noexcept;
  };
  Apm apm0_, apm1_, apm2_, apm3_;
};

// The whole predictor plus its history buffer. `capacity` bounds the bytes
// one model ever sees (primer + block 0 + one later block) and sizes its tables.
class Model {
 public:
  explicit Model(std::size_t capacity, TableLayout layout = TableLayout::Full);
  Model(const Model& o);
  Model& operator=(const Model& o);

  // Fresh state, empty history.
  void reset();
  // Models `bytes` without coding them (the primer, or a stored block).
  void train(std::span<const u8> bytes);
  // Codes `raw` after the current history. Returns false when the coded form
  // would not be smaller than raw (out is then unspecified).
  bool encode(std::span<const u8> raw, std::vector<u8>& out);
  // Decodes exactly out.size() bytes; throws Error on a malformed payload.
  void decode(std::span<const u8> payload, std::span<u8> out);

  // Long-match bypass: at a byte boundary where the match model's match is at least
  // `min_length` bytes long, one adaptive binary flag (by match length and predicted byte) says
  // whether the predicted byte is right; a right byte skips the 8 context-mixing steps and the
  // tables do not learn it. 0 = off. Set before the model sees data; the archive records it.
  // Measured at 32 on a 27 MB mixed archive: 1.11x speed, +0.70% size (xml +3.5%, prose 0).
  void set_bypass(u32 min_length) noexcept { bypass_min_ = min_length; }

  // Stores the large arrays packed (per 64-byte chunk: a mask of the non-zero bytes, then those
  // bytes) and releases them. Used for the Tier 4 snapshot, which is only ever copied: its tables
  // are 60-66% zero bytes after block 0, so it shrinks to ~40%. A copy of a packed model is a
  // full, bit-exact model; the packed model itself can no longer code.
  void pack();
  bool packed() const noexcept { return packed_.size() != 0; }
  // Bytes of heap memory this model holds now (less than bytes() once packed).
  std::size_t resident_bytes() const noexcept;

  std::size_t position() const noexcept { return history_.position; }
  // Bytes of heap memory one model holds (tables, history, mixers, APMs).
  static std::size_t bytes(std::size_t capacity, TableLayout layout = TableLayout::Full) noexcept;

 private:
  u16 predict() noexcept;
  void update(u32 bit) noexcept;
  void append(u8 byte) noexcept;
  void next_contexts() noexcept;
  bool bypass_ready() const noexcept { return bypass_min_ && match_.length() >= bypass_min_; }
  u32 bypass_bucket() const noexcept;
  void bypass_update(u32 bucket, u32 hit) noexcept;
  void skip_byte(u8 byte) noexcept;
  void ensure_contexts() noexcept { if (pending_) { next_contexts(); pending_ = false; } }
  void check_unpacked() const;
  template <class F> void for_each_array(F&& f);
  void unpack_into(Model& dst) const;

  BigArray<u8> buffer_;
  TableLayout layout_ = TableLayout::Full;
  History history_;
  Bits bits_;
  Frontend frontend_;
  Contexts contexts_;
  Match match_;
  Mixer mixer_;
  Calibration calibration_;
  std::array<u64, kContexts> hashes_{};
  alignas(64) std::array<std::int32_t, kLanes> inputs_{};
  u32 hits_ = 0;
  u32 bypass_min_ = 0;
  bool pending_ = false;  // bytes were skipped since the last context lookup
  std::array<u16, 16 * 256> bypass_p_{};  // P(predicted byte is right), 16-bit, by log2 match length and that byte
  BigArray<u8> packed_;                    // the large arrays while packed (pack())
  std::vector<std::size_t> packed_sizes_;  // their element counts
};

// The built-in primer (third_party/sws_primer/NOTICE.md) and its CRC-32C.
std::span<const u8> primer() noexcept;
u32 primer_crc() noexcept;

}  // namespace ultracodec::detail::sws
