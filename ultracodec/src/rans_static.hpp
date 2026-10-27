// Static (two-pass) order-0 rANS over bytes with 8 interleaved states.
//
// Symbol i belongs to state i % 8. States are 32-bit, renormalized with
// 16-bit words, probabilities are 12-bit, so a symbol reads at most one word
// and the 8 lanes of a group read their words in lane order. That makes the
// stream identical for the scalar decoder and the vector decoders (AVX2:
// one 8-lane gather + permute per group; NEON: two 4-lane halves), which are
// picked at run time and must agree byte for byte (tests check both).
//
// Stream layout:
//   varint n            symbol count
//   u8 mode             0 raw bytes | 1 one repeated symbol | 2 rANS
//   mode 0: n bytes;  mode 1: u8 symbol
//   mode 2: u8 max_symbol, (max_symbol+1) varint frequencies summing to 4096,
//           varint word_count, word_count u16 words (8 initial states first)
#pragma once

#include <span>
#include <vector>

#include "common.hpp"

namespace ultracodec::detail {

void rans_encode(std::span<const u8> symbols, std::vector<u8>& out);
// Decodes one stream; returns its symbols in `out` (resized). `max_symbols`
// bounds the count a corrupt stream may claim.
void rans_decode(Reader& in, std::vector<u8>& out, std::size_t max_symbols);

enum class RansKernel { Auto, Scalar, Avx2, Neon };
// Forces a decoder kernel (tests and benchmarks); Auto = best available.
void rans_set_kernel(RansKernel k) noexcept;
RansKernel rans_active_kernel() noexcept;

}  // namespace ultracodec::detail
