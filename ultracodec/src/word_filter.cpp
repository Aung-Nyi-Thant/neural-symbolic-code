#include "word_filter.hpp"

#include <array>
#include <iterator>
#include <cstring>
#include <string_view>

namespace ultracodec::detail {
namespace {

#include "word_dict.inc"

constexpr std::size_t kWords = 64 + 63 * 256;
static_assert(std::size(kProseWords) == kWords && std::size(kCodeWords) == kWords);
constexpr unsigned kSlotLog = 16;

class Dictionary {
 public:
  explicit Dictionary(const char* const* words, bool code) {
    for (unsigned c = 0; c < 256; ++c) {
      bool letter = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
      start_[c] = letter || (code && c == '_');
      cont_[c] = start_[c] || (code && c >= '0' && c <= '9');
    }
    slot_.fill(UINT16_MAX);
    std::size_t total = 0;
    for (std::size_t i = 0; i < kWords; ++i) total += std::strlen(words[i]);
    store_.assign(total + 16, 0);  // 16 bytes of slack for the decoder's fixed-size copies
    for (std::size_t i = 0, at = 0; i < kWords; ++i) {
      const std::size_t len = std::strlen(words[i]);
      std::memcpy(store_.data() + at, words[i], len);
      words_[i] = std::string_view(reinterpret_cast<const char*>(store_.data() + at), len);
      at += len;
      u32 h = hash(words_[i]);
      while (slot_[h] != UINT16_MAX) h = (h + 1) & ((1u << kSlotLog) - 1);
      slot_[h] = static_cast<u16>(i);
    }
  }
  bool starts_word(u8 c) const noexcept { return start_[c]; }
  bool in_word(u8 c) const noexcept { return cont_[c]; }
  std::string_view word(std::size_t id) const noexcept { return words_[id]; }
  // Id of the lower-case word w, or -1.
  int find(std::string_view w) const noexcept {
    for (u32 h = hash(w);; h = (h + 1) & ((1u << kSlotLog) - 1)) {
      u16 id = slot_[h];
      if (id == UINT16_MAX) return -1;
      if (words_[id] == w) return id;
    }
  }

 private:
  static u32 hash(std::string_view w) noexcept {
    u32 h = 2166136261u;
    for (char c : w) h = (h ^ static_cast<u8>(c)) * 16777619u;
    return h >> (32 - kSlotLog);
  }
  std::vector<u8> store_;
  std::array<bool, 256> start_{}, cont_{};
  std::array<std::string_view, kWords> words_{};
  std::array<u16, std::size_t{1} << kSlotLog> slot_{};
};

const Dictionary& dictionary(WordDict d) {
  static const Dictionary prose(kProseWords, false);
  static const Dictionary code(kCodeWords, true);
  return d == WordDict::Code ? code : prose;
}

ULTRA_INLINE bool upper(u8 c) noexcept { return c >= 'A' && c <= 'Z'; }
ULTRA_INLINE bool lower(u8 c) noexcept { return c >= 'a' && c <= 'z'; }

// Number of leading bytes of x (little-endian order) that the decoder copies
// as they are, i.e. that are not >= 0x80, 1 or 2. The zero-byte test can only
// misfire above a true hit, so the lowest set bit is exact.
ULTRA_INLINE unsigned plain_prefix(u64 x) noexcept {
  constexpr u64 kOnes = 0x0101010101010101ull, kHigh = 0x8080808080808080ull;
  auto has_zero = [](u64 v) { return (v - kOnes) & ~v & kHigh; };
  const u64 m = (x & kHigh) | has_zero(x ^ kOnes) | has_zero(x ^ (2 * kOnes));
  return m ? ctz64(m) >> 3 : 8;
}

// Appends the filtered form of src[0, n) at o; returns the new end. o needs 2n bytes.
u8* encode_into(const Dictionary& D, const u8* s, std::size_t n, u8* o) {
  std::size_t i = 0;
  char lw[256];
  while (i < n) {
    const u8 c = s[i];
    if (!D.starts_word(c)) {
      if (c >= 0x80 || c == 1 || c == 2) *o++ = 0xFF;
      *o++ = c;
      ++i;
      continue;
    }
    std::size_t j = i + 1;
    while (j < n && D.in_word(s[j])) ++j;
    const std::size_t len = j - i;
    if (len > 255) {
      std::memcpy(o, s + i, len);
      o += len;
      i = j;
      continue;
    }
    unsigned up = 0, lo = 0;
    for (std::size_t q = 0; q < len; ++q) {
      u8 ch = s[i + q];
      if (upper(ch)) { ++up; ch = static_cast<u8>(ch + 32); }
      else if (lower(ch)) ++lo;
      lw[q] = static_cast<char>(ch);
    }
    if (up) {
      if (upper(c) && up == 1) *o++ = 1;
      else if (!lo && len > 1) *o++ = 2;
      else {  // other mixed case: as is
        std::memcpy(o, s + i, len);
        o += len;
        i = j;
        continue;
      }
    }
    int id = len >= 2 ? D.find(std::string_view(lw, len)) : -1;
    if (id >= 64) {
      *o++ = static_cast<u8>(0xC0 + (id - 64) / 256);
      *o++ = static_cast<u8>((id - 64) % 256);
    } else if (id >= 0) {
      *o++ = static_cast<u8>(0x80 + id);
    } else {
      std::memcpy(o, lw, len);
      o += len;
    }
    i = j;
  }
  return o;
}

}  // namespace

WordDict word_filter_choose(std::span<const u8> raw) {
  constexpr std::size_t kSample = 16 << 10, kPieces = 4;
  if (raw.size() < kSample) return WordDict::None;
  std::vector<u8> tmp(2 * kSample);
  std::size_t best = 0;
  WordDict pick = WordDict::None;
  for (WordDict d : {WordDict::Prose, WordDict::Code}) {
    const Dictionary& D = dictionary(d);
    std::size_t out = 0;
    for (std::size_t k = 0; k < kPieces; ++k) {
      std::size_t at = (raw.size() - kSample) / (kPieces - 1) * k;
      out += static_cast<std::size_t>(encode_into(D, raw.data() + at, kSample, tmp.data()) - tmp.data());
    }
    if (pick == WordDict::None || out < best) { best = out; pick = d; }
  }
  return best * 100 < kSample * kPieces * 85 ? pick : WordDict::None;
}

void word_filter_encode(WordDict dict, std::span<const u8> raw, std::vector<u8>& out) {
  out.resize(2 * raw.size());
  u8* end = encode_into(dictionary(dict), raw.data(), raw.size(), out.data());
  out.resize(static_cast<std::size_t>(end - out.data()));
}

void word_filter_decode(WordDict dict, std::span<const u8> in, std::span<u8> dst) {
  const Dictionary& D = dictionary(dict);
  const u8* p = in.data();
  const u8* const end = p + in.size();
  u8* o = dst.data();
  u8* const oend = o + dst.size();
  auto fail = [] { throw Error("ultracodec: corrupt word-filter data"); };
  while (p < end) {
    if (end - p >= 8 && oend - o >= 8) {
      const u64 x = load64(p);
      const unsigned k = plain_prefix(x);
      store64(o, x);
      p += k;
      o += k;
      if (k == 8) continue;
    }
    u8 c = *p;
    if (c < 0x80 && c != 1 && c != 2) {
      if (o == oend) fail();
      *o++ = c;
      ++p;
      continue;
    }
    if (c == 0xFF) {
      if (end - p < 2 || o == oend) fail();
      *o++ = p[1];
      p += 2;
      continue;
    }
    unsigned cap = 0;
    if (c <= 2) {
      cap = c;
      if (++p == end) fail();
      c = *p;
    }
    u8* const word = o;
    if (c >= 0x80 && c < 0xFF) {
      std::size_t id;
      if (c < 0xC0) {
        id = c - 0x80u;
        ++p;
      } else {
        if (end - p < 2) fail();
        id = 64 + (c - 0xC0u) * 256 + p[1];
        p += 2;
      }
      std::string_view w = D.word(id);
      const std::size_t room = static_cast<std::size_t>(oend - o);
      if (w.size() > room) fail();
      if (w.size() <= 16 && room >= 16) copy16(o, reinterpret_cast<const u8*>(w.data()));
      else std::memcpy(o, w.data(), w.size());
      o += w.size();
    } else {
      while (p < end && D.in_word(*p) && !upper(*p)) {
        if (o == oend) fail();
        *o++ = *p++;
      }
      if (o == word) fail();
    }
    if (cap == 1) {
      if (lower(*word)) *word = static_cast<u8>(*word - 32);
    } else if (cap == 2) {
      for (u8* q = word; q < o; ++q)
        if (lower(*q)) *q = static_cast<u8>(*q - 32);
    }
  }
  if (o != oend) fail();
}

}  // namespace ultracodec::detail
