#include "dedup.hpp"

#include "lz_util.hpp"
#include "rans_static.hpp"

namespace ultracodec::detail {
namespace {

constexpr std::size_t kWindow = 32;
constexpr std::size_t kSample = 16;
constexpr unsigned kTableLog = 22;
constexpr u32 kEmpty = UINT32_MAX;

ULTRA_INLINE u32 window_hash(const u8* p) noexcept {
  u64 x = (load64(p) * 0x9E3779B97F4A7C15ull) ^ (load64(p + 8) * 0xC2B2AE3D27D4EB4Full) ^
          (load64(p + 16) * 0x165667B19E3779F9ull) ^ (load64(p + 24) * 0x27D4EB2F165667C5ull);
  return static_cast<u32>(((x ^ (x >> 31)) * 0x9E3779B97F4A7C15ull) >> (64 - kTableLog));
}

}  // namespace

std::vector<u8> dedup_encode(std::span<const u8> input, const DedupParams& p, std::vector<u8>& section) {
  const std::size_t n = input.size();
  const u8* in = input.data();
  std::vector<u8> residual;
  residual.reserve(n);
  std::vector<u8> gaps, lens, dists;
  u32 count = 0;
  std::size_t anchor = 0;
  if (n >= 2 * kWindow && n < UINT32_MAX) {
    std::vector<u32> table(std::size_t{1} << kTableLog, kEmpty);
    std::size_t i = 0, ins = 0;
    while (i + kWindow <= n) {
      for (; ins <= i; ++ins)
        if (ins % kSample == 0) table[window_hash(in + ins)] = static_cast<u32>(ins);
      const u32 c = table[window_hash(in + i)];
      if (c != kEmpty && c < i && std::memcmp(in + c, in + i, kWindow) == 0) {
        std::size_t s = i, cs = c;
        while (s > anchor && cs > 0 && in[s - 1] == in[cs - 1]) { --s; --cs; }
        std::size_t e = i + kWindow;
        e += match_length(in + e, in + c + kWindow, in + n);
        const std::size_t len = e - s;
        const bool far = (s / p.segment) != (cs / p.segment);
        if (len >= p.near_min || (far && len >= p.far_min)) {
          residual.insert(residual.end(), in + anchor, in + s);
          put_varint(gaps, static_cast<u32>(s - anchor));
          put_varint(lens, static_cast<u32>(len));
          put_varint(dists, static_cast<u32>(s - cs));
          ++count;
          anchor = i = e;
          continue;
        }
      }
      ++i;
    }
  }
  residual.insert(residual.end(), in + anchor, in + n);
  put_u32(section, count);
  u8 rs[8];
  store64(rs, residual.size());
  section.insert(section.end(), rs, rs + 8);
  put_u32(section, crc32c(input));
  rans_encode(gaps, section);
  rans_encode(lens, section);
  rans_encode(dists, section);
  return residual;
}

void dedup_decode(Reader& r, std::span<const u8> residual, std::span<u8> out) {
  const u32 count = r.u32v();
  auto rs = r.take(8);
  if (load64(rs.data()) != residual.size()) throw Error("ultracodec: dedup residual size mismatch");
  const u32 crc = r.u32v();
  if (count > out.size() / 32 + 1) throw Error("ultracodec: corrupt dedup section");
  std::vector<u8> gs, ls, ds;
  const std::size_t max_sym = std::size_t{count} * 5;
  rans_decode(r, gs, max_sym);
  rans_decode(r, ls, max_sym);
  rans_decode(r, ds, max_sym);
  Reader g(gs), l(ls), d(ds);
  u8* const o = out.data();
  u8* const oend = o + out.size();
  std::size_t op = 0, rp = 0;
  auto fail = [] { throw Error("ultracodec: corrupt dedup references"); };
  for (u32 k = 0; k < count; ++k) {
    const std::size_t gap = g.varint(), len = l.varint(), dist = d.varint();
    if (gap > residual.size() - rp || gap > out.size() - op) fail();
    if (gap) std::memcpy(o + op, residual.data() + rp, gap);
    op += gap;
    rp += gap;
    if (len > out.size() - op || dist == 0 || dist > op) fail();
    copy_match(o + op, dist, len, oend);
    op += len;
  }
  if (g.remaining() || l.remaining() || d.remaining()) fail();
  if (residual.size() - rp != out.size() - op) fail();
  if (out.size() > op) std::memcpy(o + op, residual.data() + rp, out.size() - op);
  if (crc32c(out.data(), out.size()) != crc) throw Error("ultracodec: CRC mismatch after dedup (corrupt archive)");
}

}  // namespace ultracodec::detail
