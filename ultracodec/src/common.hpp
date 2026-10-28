// Internal helpers shared by every tier: little-endian access, CRC-32C,
// 16-byte vector copies, cache-line-aligned huge-page arrays and prefetch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <vector>

#include "ultracodec/core.hpp"

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define ULTRA_SSE2 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define ULTRA_NEON 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ULTRA_INLINE inline __attribute__((always_inline))
#define ULTRA_LIKELY(x) __builtin_expect(!!(x), 1)
#define ULTRA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define ULTRA_PREFETCH(p) __builtin_prefetch(p, 1, 3)
#else
#define ULTRA_INLINE inline
#define ULTRA_LIKELY(x) (x)
#define ULTRA_UNLIKELY(x) (x)
#define ULTRA_PREFETCH(p) ((void)(p))
#endif

namespace ultracodec::detail {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

ULTRA_INLINE u16 load16(const u8* p) noexcept { u16 v; std::memcpy(&v, p, 2); return v; }
ULTRA_INLINE u32 load32(const u8* p) noexcept { u32 v; std::memcpy(&v, p, 4); return v; }
ULTRA_INLINE u64 load64(const u8* p) noexcept { u64 v; std::memcpy(&v, p, 8); return v; }
ULTRA_INLINE void store16(u8* p, u16 v) noexcept { std::memcpy(p, &v, 2); }
ULTRA_INLINE void store32(u8* p, u32 v) noexcept { std::memcpy(p, &v, 4); }
ULTRA_INLINE void store64(u8* p, u64 v) noexcept { std::memcpy(p, &v, 8); }

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "ultracodec assumes a little-endian host");

// One 16-byte unaligned copy in a single vector load/store (SSE2 / NEON).
ULTRA_INLINE void copy16(u8* dst, const u8* src) noexcept {
#if defined(ULTRA_SSE2)
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
#elif defined(ULTRA_NEON)
  vst1q_u8(dst, vld1q_u8(src));
#else
  std::memcpy(dst, src, 16);
#endif
}

ULTRA_INLINE unsigned highbit32(u32 v) noexcept { return 31u - static_cast<unsigned>(__builtin_clz(v)); }  // v > 0
ULTRA_INLINE unsigned ctz64(u64 v) noexcept { return static_cast<unsigned>(__builtin_ctzll(v)); }          // v > 0

// CRC-32C (Castagnoli). SSE4.2 instruction when the CPU has it, else
// slice-by-8 tables.
u32 crc32c(const u8* data, std::size_t n, u32 crc = 0) noexcept;
inline u32 crc32c(std::span<const u8> s) noexcept { return crc32c(s.data(), s.size()); }

// Large allocations (>= 1 MiB) are anonymous mappings on POSIX: zeroing one
// returns its pages to the kernel, so peak RSS tracks the pages a model
// really touched.
void* big_alloc(std::size_t bytes);           // throws std::bad_alloc; memory is zeroed
void big_free(void* p, std::size_t bytes) noexcept;
void big_zero(void* p, std::size_t bytes) noexcept;

// Zero-initialized, 64-byte-aligned array for large model tables. On Linux
// the allocation is advised for transparent huge pages (fewer TLB misses on
// random access to multi-hundred-MiB tables).
template <class T>
class BigArray {
 public:
  BigArray() = default;
  explicit BigArray(std::size_t n) { reset(n); }
  BigArray(const BigArray&) = delete;
  BigArray& operator=(const BigArray&) = delete;
  BigArray(BigArray&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  BigArray& operator=(BigArray&& o) noexcept {
    if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
    return *this;
  }
  ~BigArray() { release(); }

  void reset(std::size_t n);   // reallocates (if the size changes) and zeroes
  void zero() noexcept { if (p_) big_zero(p_, n_ * sizeof(T)); }
  T* data() noexcept { return p_; }
  const T* data() const noexcept { return p_; }
  std::size_t size() const noexcept { return n_; }
  T& operator[](std::size_t i) noexcept { return p_[i]; }
  const T& operator[](std::size_t i) const noexcept { return p_[i]; }

 private:
  void release() noexcept;
  T* p_ = nullptr;
  std::size_t n_ = 0;
};


template <class T>
void BigArray<T>::reset(std::size_t n) {
  if (n != n_) {
    release();
    if (n) p_ = static_cast<T*>(big_alloc(n * sizeof(T)));
    n_ = n;
    return;  // fresh memory is already zero
  }
  zero();
}
template <class T>
void BigArray<T>::release() noexcept {
  if (p_) big_free(p_, n_ * sizeof(T));
  p_ = nullptr;
  n_ = 0;
}

// A bounds-checked little-endian reader over an untrusted byte span.
class Reader {
 public:
  explicit Reader(std::span<const u8> s) noexcept : p_(s.data()), end_(s.data() + s.size()) {}
  std::size_t remaining() const noexcept { return static_cast<std::size_t>(end_ - p_); }
  const u8* pos() const noexcept { return p_; }
  void need(std::size_t n) const {
    if (remaining() < n) throw Error("ultracodec: truncated or corrupt data");
  }
  u8 u8v() { need(1); return *p_++; }
  u32 u32v() { need(4); u32 v = load32(p_); p_ += 4; return v; }
  std::span<const u8> take(std::size_t n) { need(n); auto s = std::span<const u8>(p_, n); p_ += n; return s; }
  // LEB128, at most 5 bytes / 32 bits.
  u32 varint() {
    u32 v = 0;
    for (unsigned shift = 0; shift < 35; shift += 7) {
      u8 b = u8v();
      v |= static_cast<u32>(b & 0x7F) << shift;
      if (!(b & 0x80)) return v;
    }
    throw Error("ultracodec: bad varint");
  }

 private:
  const u8* p_;
  const u8* end_;
};

inline void put_varint(std::vector<u8>& out, u32 v) {
  while (v >= 0x80) { out.push_back(static_cast<u8>(v | 0x80)); v >>= 7; }
  out.push_back(static_cast<u8>(v));
}
inline void put_u32(std::vector<u8>& out, u32 v) {
  u8 b[4];
  store32(b, v);
  out.insert(out.end(), b, b + 4);
}

}  // namespace ultracodec::detail
