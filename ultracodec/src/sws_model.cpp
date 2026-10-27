#include "sws_model.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>

#include "range_coder.hpp"

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#define ULTRA_SWS_NEON 1
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define ULTRA_SWS_AVX2 1
#endif

namespace ultracodec::detail::sws {
namespace {

constexpr u64 kBase = 0x100000001B3ull, kSalt = 0x9E3779B97F4A7C15ull, kNibble = 0xA24BAED4963EE407ull,
              kSlot = 0xD6E8FEB86659FD93ull;
constexpr std::array<unsigned, kOrders> kOrderList{0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32};
constexpr unsigned kMatchMin = 6;
constexpr std::size_t kApmRows0 = 256, kApmRows1 = 65536, kApmRows2 = 65536, kApmRows3 = 64 * 256 * 2;
constexpr int kStateCap = 35;
constexpr u32 kNumStates = 254;

constexpr u64 mix64(u64 x) noexcept {  // SplitMix64 finalizer
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return x;
}
constexpr std::int64_t clamp64(std::int64_t v, std::int64_t lo, std::int64_t hi) noexcept {
  return v < lo ? lo : (v > hi ? hi : v);
}
std::int32_t stretch(const Tables& t, u32 p16) noexcept { return t.stretch[p16 >> 4]; }
u32 squash(const Tables& t, std::int64_t logit) noexcept {
  return t.squash[static_cast<std::size_t>(clamp64(logit, kLogitMin, kLogitMax) + kSquashOffset)];
}

// Table sizes follow the bytes a model will see (`capacity`), up to the caps. Order 0 sees 17
// nibble contexts and order 1 at most 4352; larger tables would stay empty. Four slots per bucket:
// 2^(log2(capacity) - 1) buckets hold the two nibble contexts of every byte.
unsigned table_bits(std::size_t capacity, std::size_t context) noexcept {
  const unsigned cap = context == 0 ? 6u : context == 1 ? 12u : kBucketBits;
  const unsigned wanted = capacity <= 1 ? 0u : static_cast<unsigned>(std::bit_width(capacity - 1));
  return std::clamp(wanted ? wanted - 1 : 0u, std::min(10u, cap), cap);
}
unsigned match_bits(std::size_t capacity) noexcept { return capacity < (std::size_t{1} << 20) ? 20u : 22u; }

// ---- static tables ----------------------------------------------------------

// e^(-1/256) in Q62, from its Taylor series.
constexpr u64 exp_neg_1_256_q62() {
  constexpr u64 one = u64{1} << 62;
  const u64 even = one + (one >> 16) / 2 + (one >> 32) / 24 + (one >> 48) / 720;
  const u64 odd = (one >> 8) + (one >> 24) / 6 + (one >> 40) / 120;
  return even - odd;
}

void build_squash_stretch(Tables& t) {
  constexpr u64 kOneQ31 = u64{1} << 31, kHalfQ31 = u64{1} << 30;
  constexpr u64 r = (exp_neg_1_256_q62() + kHalfQ31) >> 31;  // e^(-1/256) in Q31
  std::array<u64, kSquashOffset + 1> e{};
  e[0] = kOneQ31;
  for (std::size_t x = 1; x < e.size(); ++x) e[x] = (e[x - 1] * r + kHalfQ31) >> 31;
  for (std::size_t i = 0; i < t.squash.size(); ++i) {
    const std::int64_t x = static_cast<std::int64_t>(i) - kSquashOffset;
    const u64 den = kOneQ31 + e[static_cast<std::size_t>(x < 0 ? -x : x)];
    u64 p = ((u64{1} << 47) + den / 2) / den;  // 65536 / (1 + e^-|x|), rounded
    if (x < 0) p = 65536 - p;                  // exact symmetry
    t.squash[i] = static_cast<u16>(p < 1 ? 1 : (p > 65535 ? 65535 : p));
  }
  for (std::size_t i = 0; i < t.stretch.size(); ++i) {
    const u32 target = static_cast<u32>(i) * 16 + 8;
    std::size_t lo = 0, hi = t.squash.size() - 1;
    while (lo < hi) {  // squash is non-decreasing
      const std::size_t mid = (lo + hi) / 2;
      if (t.squash[mid] >= target) hi = mid; else lo = mid + 1;
    }
    t.stretch[i] = static_cast<std::int16_t>(static_cast<std::int64_t>(lo) - kSquashOffset);
  }
}

struct Counts {
  u8 n0, n1;
};
// On a bit its count grows (saturating at kStateCap) and an opposite count above 2 fades to n/2 + 1.
Counts step(Counts c, u32 bit) {
  const auto grow = [](u8 n) { return static_cast<u8>(n < kStateCap ? n + 1 : kStateCap); };
  const auto fade = [](u8 n) { return static_cast<u8>(n > 2 ? n / 2 + 1 : n); };
  return bit ? Counts{fade(c.n0), grow(c.n1)} : Counts{grow(c.n0), fade(c.n1)};
}

void build_states(Tables& t) {
  constexpr std::size_t kSide = kStateCap + 1;
  std::array<std::array<bool, kSide>, kSide> reached{};
  std::array<Counts, kSide * kSide> queue{};
  std::size_t head = 0, tail = 0;
  reached[0][0] = true;
  queue[tail++] = Counts{0, 0};
  while (head < tail) {
    const Counts c = queue[head++];
    for (u32 bit = 0; bit < 2; ++bit) {
      const Counts n = step(c, bit);
      if (!reached[n.n0][n.n1]) {
        reached[n.n0][n.n1] = true;
        queue[tail++] = n;
      }
    }
  }
  // Canonical numbering: by total count, then by n1. State 0 is the empty history.
  std::array<std::array<u8, kSide>, kSide> id{};
  u32 count = 0;
  for (std::size_t total = 0; total <= 2 * kStateCap; ++total) {
    for (std::size_t n1 = 0; n1 <= total && n1 < kSide; ++n1) {
      const std::size_t n0 = total - n1;
      if (n0 >= kSide || !reached[n0][n1]) continue;
      if (count >= 256) std::abort();
      id[n0][n1] = static_cast<u8>(count);
      t.n0[count] = static_cast<u8>(n0);
      t.n1[count] = static_cast<u8>(n1);
      ++count;
    }
  }
  if (count != kNumStates) std::abort();  // the state machine definition changed
  for (u32 s = 0; s < 256; ++s) {
    if (s < count) {
      for (u32 bit = 0; bit < 2; ++bit) {
        const Counts n = step(Counts{t.n0[s], t.n1[s]}, bit);
        t.next[s][bit] = id[n.n0][n.n1];
      }
    } else {
      t.next[s] = {0, 0};
      t.n0[s] = t.n1[s] = 0;
    }
    const u64 num = (u64{t.n1[s]} * 2 + 1) << 22;
    const u64 den = (u64{t.n0[s]} + t.n1[s]) * 2 + 2;
    t.p22_init[s] = static_cast<u32>(num / den);
  }
}

Tables build_tables() {
  Tables t{};
  build_squash_stretch(t);
  for (u32 n = 0; n < t.recip.size(); ++n) t.recip[n] = (u32{1} << 17) / (2 * n + 3);
  build_states(t);
  return t;
}

// ---- mixer kernels ------------------------------------------------------------
// |input| <= 3071 and |weight| <= 2^19: each product and the 32-lane sum fit in int64.

std::int64_t dot_scalar(const std::int32_t* x, const std::int32_t* w) noexcept {
  std::int64_t sum = 0;
  for (std::size_t i = 0; i < kLanes; ++i) sum += std::int64_t{x[i]} * w[i];
  return sum;
}
// |input * error| <= 3071 * 4095 * 16 < 2^31.
void train_scalar(const std::int32_t* x, std::int32_t* w, std::int32_t error) noexcept {
  for (std::size_t i = 0; i < kLanes; ++i)
    w[i] = std::clamp(w[i] + ((x[i] * error + 8192) >> 14), -kWeightLimit, kWeightLimit);
}

#if defined(ULTRA_SWS_AVX2)
__attribute__((target("avx2"))) std::int64_t dot_avx2(const std::int32_t* x, const std::int32_t* w) noexcept {
  __m256i even = _mm256_setzero_si256(), odd = even;
  for (std::size_t i = 0; i < kLanes; i += 8) {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    even = _mm256_add_epi64(even, _mm256_mul_epi32(a, b));
    odd = _mm256_add_epi64(odd, _mm256_mul_epi32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32)));
  }
  alignas(32) std::array<std::int64_t, 4> lanes{};
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes.data()), _mm256_add_epi64(even, odd));
  return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
__attribute__((target("avx2"))) void train_avx2(const std::int32_t* x, std::int32_t* w, std::int32_t error) noexcept {
  const __m256i e = _mm256_set1_epi32(error), round = _mm256_set1_epi32(8192), hi = _mm256_set1_epi32(kWeightLimit),
                lo = _mm256_set1_epi32(-kWeightLimit);
  for (std::size_t i = 0; i < kLanes; i += 8) {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<__m256i*>(w + i));
    b = _mm256_add_epi32(b, _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(a, e), round), 14));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(w + i), _mm256_min_epi32(hi, _mm256_max_epi32(lo, b)));
  }
}
bool has_avx2() noexcept {
  static const bool value = __builtin_cpu_supports("avx2");
  return value;
}
#endif

std::int64_t dot(const std::int32_t* x, const std::int32_t* w) noexcept {
#if defined(ULTRA_SWS_NEON)
  int64x2_t a = vdupq_n_s64(0), b = a, c = a, d = a;
  for (std::size_t i = 0; i < kLanes; i += 8) {
    const auto x0 = vld1q_s32(x + i), w0 = vld1q_s32(w + i);
    const auto x1 = vld1q_s32(x + i + 4), w1 = vld1q_s32(w + i + 4);
    a = vmlal_s32(a, vget_low_s32(x0), vget_low_s32(w0));
    b = vmlal_high_s32(b, x0, w0);
    c = vmlal_s32(c, vget_low_s32(x1), vget_low_s32(w1));
    d = vmlal_high_s32(d, x1, w1);
  }
  return vaddvq_s64(vaddq_s64(vaddq_s64(a, b), vaddq_s64(c, d)));
#elif defined(ULTRA_SWS_AVX2)
  return has_avx2() ? dot_avx2(x, w) : dot_scalar(x, w);
#else
  return dot_scalar(x, w);
#endif
}
void train(const std::int32_t* x, std::int32_t* w, std::int32_t error) noexcept {
#if defined(ULTRA_SWS_NEON)
  const auto round = vdupq_n_s32(8192), hi = vdupq_n_s32(kWeightLimit), lo = vdupq_n_s32(-kWeightLimit);
  for (std::size_t i = 0; i < kLanes; i += 4) {
    const auto a = vld1q_s32(x + i);
    auto b = vld1q_s32(w + i);
    b = vaddq_s32(b, vshrq_n_s32(vaddq_s32(vmulq_n_s32(a, error), round), 14));
    vst1q_s32(w + i, vminq_s32(hi, vmaxq_s32(lo, b)));
  }
#elif defined(ULTRA_SWS_AVX2)
  if (has_avx2()) train_avx2(x, w, error); else train_scalar(x, w, error);
#else
  train_scalar(x, w, error);
#endif
}

constexpr u8 kPrimer[] = {
#include "sws_primer.inc"
};

}  // namespace

const Tables& tables() {
  static const Tables t = build_tables();
  return t;
}

std::span<const u8> primer() noexcept { return kPrimer; }
u32 primer_crc() noexcept {
  static const u32 crc = crc32c(kPrimer, sizeof(kPrimer));
  return crc;
}

void StateMap::update(u8 state, u32 bit, const Tables& t) noexcept {
  u32& e = entries_[state];
  const u32 n = e & 1023u;
  const std::int64_t p22 = e >> 10;
  const std::int64_t target = bit ? (std::int64_t{1} << 22) - 1 : 0;
  // |target - p22| < 2^22 and recip <= 43690: the product stays below 2^38, and the new
  // value lies between p22 and target.
  const std::int64_t delta = ((target - p22) * std::int64_t{t.recip[n]} + 32768) >> 16;
  const auto p = static_cast<u32>(p22 + delta);
  e = (p << 10) | (n < 1023 ? n + 1 : 1023u);
}

// ---- SlotTable --------------------------------------------------------------

void SlotTable::reset(unsigned bits) {
  buckets_.reset(std::size_t{1} << bits);  // zeroed: every slot empty
  tags_.reset(std::size_t{1} << bits);
  bits_ = bits;
}

u8* SlotTable::find(u64 hash, const Tables& t) noexcept {
  const auto index = static_cast<std::size_t>(hash >> (64 - bits_));
  auto& bucket = buckets_[index];
  const auto check = static_cast<u8>(static_cast<u8>(hash >> 8) | 1u);  // 0 marks an empty slot
  const u32 difference = tags_[index] ^ (u32{check} * 0x01010101u);
  const u32 equal = (difference - 0x01010101u) & ~difference & 0x80808080u;
  if (equal) return bucket.slots[static_cast<std::size_t>(std::countr_zero(equal)) / 8].states.data();
  std::size_t victim = 0;
  u32 best = UINT32_MAX;
  for (std::size_t i = 0; i < bucket.slots.size(); ++i) {
    const auto& slot = bucket.slots[i];
    if (!slot.check) {
      victim = i;
      break;
    }
    const u32 evidence = u32{t.n0[slot.states[0]]} + t.n1[slot.states[0]];
    if (evidence < best) {
      best = evidence;
      victim = i;
    }
  }
  auto& slot = bucket.slots[victim];
  slot = Slot{};
  slot.check = check;
  const auto shift = static_cast<unsigned>(victim * 8);
  tags_[index] = (tags_[index] & ~(255u << shift)) | (u32{check} << shift);
  return slot.states.data();
}

void SlotTable::prefetch(u64 hash) const noexcept {
  const auto index = static_cast<std::size_t>(hash >> (64 - bits_));
  ULTRA_PREFETCH(&tags_[index]);
  ULTRA_PREFETCH(&buckets_[index]);
}

// ---- Frontend ---------------------------------------------------------------

void Frontend::reset() {
  word0_ = word1_ = word2_ = 0;
  line_start_ = previous_line_ = 0;
  column_ = indent_ = 0;
  in_indent_ = true;
  indirect1_.assign(256, 0);
  indirect2_.assign(65536, 0);
}

void Frontend::observe(const History& h, u8 c) noexcept {
  const std::size_t position = h.position;
  const u8* data = h.data;
  const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128 || (c >= '0' && c <= '9') || c == '_';
  if (alpha) {
    word0_ = (word0_ + u64{c >= 'A' && c <= 'Z' ? static_cast<u8>(c + 32) : c} + 1) * 0x2F0F3D;
  } else if (word0_) {
    word2_ = word1_;
    word1_ = word0_;
    word0_ = 0;
  }
  if (c == '\n') {
    previous_line_ = line_start_;
    line_start_ = position;
    column_ = indent_ = 0;
    in_indent_ = true;
  } else {
    ++column_;
    if (in_indent_ && (c == ' ' || c == '\t')) ++indent_; else in_indent_ = false;
  }
  if (position >= 2) {
    const u8 p1 = data[position - 2];
    indirect1_[p1] = static_cast<u16>(indirect1_[p1] << 8 | c);
    if (position >= 3) {
      const std::size_t key = std::size_t{data[position - 3]} << 8 | p1;
      indirect2_[key] = static_cast<u16>(indirect2_[key] << 8 | c);
    }
  }
}

void Frontend::contexts(const History& h, std::array<u64, kContexts>& out) const noexcept {
  const auto at = [&](std::size_t back) { return h.at(back); };
  std::size_t i = 0;
  unsigned used = 0;
  u64 prefix = 0, power = 1;
  // Every order shares one byte polynomial (exact unsigned wraparound).
  for (const unsigned order : kOrderList) {
    while (used < order) {
      prefix = (prefix + at(++used) + 1) * kBase;
      power *= kBase;
    }
    out[i++] = u64{order} * kSalt * power + prefix;
  }
  {
    u64 hash = 77;
    for (const unsigned offset : {1u, 3u, 5u, 8u}) hash = (hash ^ (u64{at(offset)} + offset)) * kBase;
    out[i++] = hash;
  }
  {
    u64 hash = 78;
    for (const unsigned offset : {1u, 4u, 9u, 16u}) hash = (hash ^ (u64{at(offset)} + offset)) * kBase;
    out[i++] = hash;
  }
  out[i++] = mix64(word0_ * 31 + 1);
  out[i++] = mix64(word0_ * 31 + word1_ * 0x2F0F + 2);
  out[i++] = mix64(word1_ * 131 + word2_ * 0x51 + 3 + u64{at(1)} * 7);
  const std::size_t above = previous_line_ + column_;
  const u8 byte_above = above < line_start_ ? h.data[above] : 0;
  out[i++] = mix64((u64{byte_above} << 8 | at(1)) * 5 + 0x1234);
  out[i++] = mix64((u64{indent_} << 16 | u64{std::min(column_, 255u)} << 8 | (in_indent_ ? 1u : 0u)) * 11 + word0_ + 0x777);
  const u16 h1 = indirect1_[at(1)], h2 = indirect2_[std::size_t{at(2)} << 8 | at(1)];
  out[i++] = mix64((u64{h1} << 8 | at(1)) + 0xABC);
  out[i++] = mix64((u64{h2} << 16 | u64{at(2)} << 8 | at(1)) + 0xDEF);
  for (std::size_t k = 0; k < kContexts; ++k) out[k] = mix64(out[k] + u64{k} * kSlot);
}

// ---- Contexts -----------------------------------------------------------------

void Contexts::reset(std::size_t capacity) {
  for (std::size_t k = 0; k < kContexts; ++k) {
    tables_[k].reset(table_bits(capacity, k));
    maps_[k] = StateMap(*t_);
  }
  slots_.fill(nullptr);
  states_.fill(0);
  node_ = 1;
}

void Contexts::lookup(const std::array<u64, kContexts>& hashes, u32 c0) noexcept {
  std::array<u64, kContexts> keys;
  for (std::size_t k = 0; k < kContexts; ++k) {
    keys[k] = mix64(hashes[k] ^ (u64{c0} * kNibble));
    tables_[k].prefetch(keys[k]);
  }
  for (std::size_t k = 0; k < kContexts; ++k) slots_[k] = tables_[k].find(keys[k], *t_);
  node_ = 1;
}

u32 Contexts::predict(std::int32_t* inputs) noexcept {
  const auto& t = *t_;
  u32 hits = 0;
  for (std::size_t k = 0; k < kContexts; ++k) {
    const u8 state = slots_[k][node_ - 1];
    states_[k] = state;
    inputs[k] = stretch(t, maps_[k].p16(state));
    hits += state ? 1u : 0u;
  }
  return hits;
}

void Contexts::update(u32 bit) noexcept {
  const auto& t = *t_;
  for (std::size_t k = 0; k < kContexts; ++k) {
    maps_[k].update(states_[k], bit, t);
    slots_[k][node_ - 1] = t.next[states_[k]][bit];
  }
  node_ = node_ * 2 + bit;
}

// ---- Match ----------------------------------------------------------------------

void Match::reset(std::size_t capacity) {
  for (auto& map : maps_) map = StateMap(*t_);
  bits_ = match_bits(capacity);
  table_.reset(std::size_t{1} << bits_);
  pointer_ = 0;
  length_ = expected_ = context_ = 0;
}

void Match::predict(const History& h, u32 bit, std::int32_t* inputs) noexcept {
  context_ = 0;
  expected_ = 0;
  if (length_) {
    expected_ = (u32{h.data[pointer_]} >> (7 - bit)) & 1u;
    context_ = std::min(length_, 31u) * 2 + expected_;
    const u32 strength = std::min(std::min(length_, 31u) * 64, 2047u);
    inputs[kMatchLane] = stretch(*t_, maps_[context_].p16(1));
    inputs[kMatchLane + 1] = expected_ ? static_cast<std::int32_t>(strength) : -static_cast<std::int32_t>(strength);
  } else {
    inputs[kMatchLane] = 0;
    inputs[kMatchLane + 1] = 0;
  }
}

void Match::update(u32 bit) noexcept {
  if (length_) {
    maps_[context_].update(1, bit, *t_);
    if (expected_ != bit) length_ = 0;
  }
}

void Match::observe(const History& h) noexcept {
  const std::size_t position = h.position;
  const u8* data = h.data;
  if (length_) {
    ++pointer_;
    length_ = std::min(length_ + 1, 65535u);
  }
  if (position >= kMatchMin) {
    u64 hash = 0;
    for (unsigned j = 1; j <= kMatchMin; ++j) hash = (hash + h.at(j) + 1) * kBase;
    const auto index = static_cast<std::size_t>(mix64(hash) >> (64 - bits_));
    if (!length_) {
      const u32 candidate = table_[index];
      if (candidate) {
        pointer_ = candidate;
        u32 length = 0;
        while (length < 65535 && length < pointer_ && data[pointer_ - 1 - length] == data[position - 1 - length]) ++length;
        length_ = length >= kMatchMin ? length : 0;
      }
    }
    table_[index] = static_cast<u32>(position);
  }
}

// ---- Mixer ----------------------------------------------------------------------

void Mixer::reset() {
  weights_.reset(kMixRows * kLanes);
  std::fill_n(weights_.data(), weights_.size(), 65536 / 8);
  final_weights_.assign(256 * (kSets + 1), 65536 / std::int32_t{kSets});
  rows_.fill(0);
  set_probability_.fill(32768);
  final_inputs_.fill(0);
  final_row_ = 0;
  mixed_ = 32768;
}

u32 Mixer::predict(const std::int32_t* inputs, u32 hits, u32 match_length, u32 match_bit, const Bits& bits) noexcept {
  const auto& t = *t_;
  const u32 bucket = match_length == 0 ? 0u : match_length < 16 ? 1u : match_length < 32 ? 2u : 3u;
  rows_[0] = bits.c0;
  rows_[1] = kSetRows[0] + std::size_t{bits.c1} * 64 + ((bucket * 2 + (match_length ? match_bit : 0u)) * 8 + std::min(bits.bit, 7u)) % 64;
  rows_[2] = kSetRows[0] + kSetRows[1] + std::size_t{bits.c0} * 8 + std::min(hits * 8 / (kContexts + 1), std::size_t{7});
  rows_[3] = kSetRows[0] + kSetRows[1] + kSetRows[2] + ((std::size_t{bits.c2} * 8 + bits.bit) & 2047u);
  for (std::size_t s = 0; s < kSets; ++s) {
    const std::int64_t logit = clamp64(dot(inputs, &weights_[rows_[s] * kLanes]) >> 16, kLogitMin, kLogitMax);
    set_probability_[s] = squash(t, logit);
    final_inputs_[s] = static_cast<std::int32_t>(logit);
  }
  final_inputs_[kSets] = 256;
  final_row_ = std::size_t{bits.c0} * (kSets + 1);
  const std::int32_t* w = &final_weights_[final_row_];
  std::int64_t sum = 0;
  for (std::size_t i = 0; i <= kSets; ++i) sum += std::int64_t{final_inputs_[i]} * w[i];
  mixed_ = squash(t, sum >> 16);
  return mixed_;
}

void Mixer::train(const std::int32_t* inputs, u32 bit) noexcept {
  for (std::size_t s = 0; s < kSets; ++s) {
    const std::int32_t error = ((static_cast<std::int32_t>(bit) << 12) - static_cast<std::int32_t>(set_probability_[s] >> 4)) * kLearningRate;
    sws::train(inputs, &weights_[rows_[s] * kLanes], error);
  }
  const std::int32_t error = ((static_cast<std::int32_t>(bit) << 12) - static_cast<std::int32_t>(mixed_ >> 4)) * 2;
  std::int32_t* w = &final_weights_[final_row_];
  for (std::size_t i = 0; i <= kSets; ++i)
    w[i] = std::clamp(w[i] + ((final_inputs_[i] * error + 8192) >> 14), -kWeightLimit, kWeightLimit);
}

// ---- Calibration (APMs) -------------------------------------------------------

void Calibration::Apm::reset(std::size_t rows) {
  const auto& t = tables();
  std::array<u16, 33> initial{};
  for (std::size_t j = 0; j < 33; ++j) initial[j] = static_cast<u16>(squash(t, (static_cast<std::int64_t>(j) - 16) * 192));
  values.reset(rows * 33);
  for (std::size_t r = 0; r < rows; ++r) std::copy_n(initial.data(), 33, values.data() + r * 33);
  index = 0;
}

u32 Calibration::Apm::predict(std::size_t row, u32 low, u32 weight) noexcept {
  const std::size_t base = row * 33 + low;
  index = base + (weight >= 96 ? 1u : 0u);
  return (u32{values[base]} * (192 - weight) + u32{values[base + 1]} * weight) / 192;
}

void Calibration::Apm::update(u32 bit) noexcept {
  const std::int32_t target = bit ? 65535 : 0, value = values[index];
  values[index] = static_cast<u16>(value + (target - value) / 128);
}

void Calibration::reset() {
  apm0_.reset(kApmRows0);
  apm1_.reset(kApmRows1);
  apm2_.reset(kApmRows2);
  apm3_.reset(kApmRows3);
}

u16 Calibration::predict(u32 p, u32 match_length, u32 match_bit, const Bits& bits) noexcept {
  const auto coordinate = static_cast<u32>(stretch(tables(), p) + 16 * 192);
  const u32 low = std::min(coordinate / 192, 31u), weight = std::min(coordinate - low * 192, 191u);
  const u32 p0 = apm0_.predict(bits.c0, low, weight);
  const u32 p1 = apm1_.predict((bits.c0 | u32{bits.c1} << 8) & 0xFFFFu, low, weight);
  const u32 p2 = apm2_.predict(static_cast<std::size_t>((mix64(u64{bits.c2} << 8 | bits.c1) * 256 + bits.c0) & 0xFFFFu), low, weight);
  const u32 p3 = apm3_.predict((std::size_t{std::min(match_length, 63u)} * 2 + (match_length ? match_bit : 0u)) * 256 + bits.c0, low, weight);
  return static_cast<u16>(std::clamp((2 * p + p0 + 2 * p1 + 2 * p2 + p3 + 4) / 8, 32u, 65503u));
}

void Calibration::update(u32 bit) noexcept {
  apm0_.update(bit);
  apm1_.update(bit);
  apm2_.update(bit);
  apm3_.update(bit);
}

// ---- Model ----------------------------------------------------------------------

Model::Model(std::size_t capacity) : buffer_(capacity) { reset(); }

Model::Model(const Model& o) { *this = o; }

Model& Model::operator=(const Model& o) {
  if (this == &o) return *this;
  // Arrays of equal size copy in place, so a worker's model is refreshed without reallocating.
  copy_big(buffer_, o.buffer_);
  history_ = {buffer_.data(), o.history_.position};
  bits_ = o.bits_;
  frontend_ = o.frontend_;
  contexts_ = o.contexts_;
  match_ = o.match_;
  mixer_ = o.mixer_;
  calibration_ = o.calibration_;
  hashes_ = o.hashes_;
  inputs_ = o.inputs_;
  hits_ = o.hits_;
  // The copied slot pointers still point into o's tables: look them up again in ours. Snapshots
  // are taken on byte boundaries, where the lookup uses c0 = 1.
  contexts_.lookup(hashes_, bits_.c0);
  return *this;
}

void Model::reset() {
  history_ = {buffer_.data(), 0};
  bits_ = {};
  inputs_.fill(0);
  hits_ = 0;
  frontend_.reset();
  contexts_.reset(buffer_.size());
  match_.reset(buffer_.size());
  mixer_.reset();
  calibration_.reset();
  next_contexts();
}

std::size_t Model::bytes(std::size_t capacity) noexcept {
  std::size_t b = capacity + (std::size_t{4} << match_bits(capacity)) + (256 + 65536) * 2;
  for (std::size_t k = 0; k < kContexts; ++k) b += std::size_t{68} << table_bits(capacity, k);
  b += kMixRows * kLanes * 4 + 256 * (kSets + 1) * 4;
  b += (kApmRows0 + kApmRows1 + kApmRows2 + kApmRows3) * 33 * 2;
  return b + sizeof(Model);
}

void Model::next_contexts() noexcept {
  frontend_.contexts(history_, hashes_);
  contexts_.lookup(hashes_, 1);
}

u16 Model::predict() noexcept {
  hits_ = contexts_.predict(inputs_.data());
  match_.predict(history_, bits_.bit, inputs_.data());
  inputs_[kBiasLane] = 256;
  const u32 mixed = mixer_.predict(inputs_.data(), hits_, match_.length(), match_.expected(), bits_);
  return calibration_.predict(mixed, match_.length(), match_.expected(), bits_);
}

void Model::update(u32 bit) noexcept {
  contexts_.update(bit);
  match_.update(bit);
  mixer_.train(inputs_.data(), bit);
  calibration_.update(bit);
  if (bits_.advance(bit)) {
    append(bits_.c1);
  } else if (bits_.bit == 4) {
    contexts_.lookup(hashes_, bits_.c0);
  }
}

void Model::append(u8 byte) noexcept {
  buffer_[history_.position++] = byte;
  frontend_.observe(history_, byte);
  match_.observe(history_);
  next_contexts();
}

void Model::train(std::span<const u8> bytes) {
  if (bytes.size() > buffer_.size() - history_.position) throw Error("ultracodec: Tier 4 history overflow");
  for (const u8 c : bytes)
    for (unsigned b = 0; b < 8; ++b) {
      (void)predict();
      update((u32{c} >> (7 - b)) & 1u);
    }
}

bool Model::encode(std::span<const u8> raw, std::vector<u8>& out) {
  if (raw.size() > buffer_.size() - history_.position) throw Error("ultracodec: Tier 4 history overflow");
  const std::size_t start = out.size();
  out.resize(start + raw.size());  // a coded form this long or longer is stored instead
  RangeEncoder coder(std::span<u8>(out.data() + start, raw.size()));
  for (const u8 c : raw)
    for (unsigned b = 0; b < 8; ++b) {
      const u32 bit = (u32{c} >> (7 - b)) & 1u;
      coder.encode(bit, predict());
      update(bit);
    }
  const std::size_t n = coder.finish();
  if (coder.overflowed() || n >= raw.size()) return false;
  out.resize(start + n);
  return true;
}

void Model::decode(std::span<const u8> payload, std::span<u8> out) {
  if (out.size() > buffer_.size() - history_.position) throw Error("ultracodec: Tier 4 history overflow");
  RangeDecoder coder(payload);
  for (u8& c : out) {
    u32 x = 0;
    for (unsigned b = 0; b < 8; ++b) {
      const u32 bit = coder.decode(predict());
      update(bit);
      x = x * 2 + bit;
    }
    c = static_cast<u8>(x);
  }
  if (!coder.consumed_exactly()) throw Error("ultracodec: Tier 4 stream did not end cleanly (corrupt block)");
}

}  // namespace ultracodec::detail::sws
