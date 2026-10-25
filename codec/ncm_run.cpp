// ncm_run.cpp - the block loops of the neural CM codec, one instantiation per
// SIMD kernel policy. Each entry point is compiled for its instruction set and
// flattened: the model's bit loop and the kernels inline into one function,
// so an AVX2 build of the loop runs only where AVX2 was detected. Every
// instantiation computes the same integers (R15) and writes the same bits.
#include "codec/ncm_run.hpp"

#include "codec/ncm_model.hpp"
#include "codec/ncm_simd.hpp"

namespace hce::ncm {
namespace {

template <class K>
inline std::size_t encode_loop(CMModel& m, hc::Bytes frame, std::size_t begin, std::size_t end,
                               hc::MutBytes out) noexcept {
  ArithEncoder enc(out.data(), out.size());
  EncodeIO io{enc};
  const FrameHistory h{frame.data()};
  m.begin_block(h, begin);
  for (std::size_t i = begin; i < end && !enc.full(); ++i) {
    const int c = static_cast<int>(frame[i]);
    m.code_byte<K>(io, c);
    m.byte_update(h, c);
  }
  if (!enc.full()) enc.flush();
  return enc.full() ? 0 : enc.size();
}

template <class K>
inline hc::Errc decode_loop(CMModel& m, hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept {
  ArithDecoder dec(payload);
  DecodeIO io{dec};
  const WindowHistory h{&w, w.pos() - w.frame_offset()};
  m.begin_block(h, w.frame_offset());
  for (hc::u32 i = 0; i < usize; ++i) {
    const int c = m.code_byte<K>(io, 0);
    if (dec.overrun()) return hc::Errc::payload_corrupt;  // ran past the payload: corrupt
    w.put(static_cast<std::byte>(c));
    m.byte_update(h, c);
  }
  return dec.exact() ? hc::Errc::ok : hc::Errc::payload_corrupt;
}

template <class K>
inline void replay_loop(CMModel& m, hc::Bytes frame, const BlockRange* blocks, std::size_t n) noexcept {
  ReplayIO io;
  const FrameHistory h{frame.data()};
  for (std::size_t b = 0; b < n; ++b) {
    m.begin_block(h, blocks[b].first);
    for (std::size_t i = blocks[b].first; i < blocks[b].second; ++i) {
      const int c = static_cast<int>(frame[i]);
      m.code_byte<K>(io, c);
      m.byte_update(h, c);
    }
  }
}

// One Runner per policy; ATTR: the instruction set its loops are compiled for.
#define HCE_NCM_RUNNER(NAME, K, ATTR)                                                                         \
  ATTR __attribute__((flatten)) std::size_t NAME##_encode(CMModel& m, hc::Bytes frame, std::size_t begin,     \
                                                          std::size_t end, hc::MutBytes out) noexcept {       \
    return encode_loop<K>(m, frame, begin, end, out);                                                         \
  }                                                                                                           \
  ATTR __attribute__((flatten)) hc::Errc NAME##_decode(CMModel& m, hc::Bytes payload, hc::u32 usize,          \
                                                       hc::Window& w) noexcept {                              \
    return decode_loop<K>(m, payload, usize, w);                                                              \
  }                                                                                                           \
  ATTR __attribute__((flatten)) void NAME##_replay(CMModel& m, hc::Bytes frame, const BlockRange* blocks,     \
                                                   std::size_t n) noexcept {                                  \
    replay_loop<K>(m, frame, blocks, n);                                                                      \
  }                                                                                                           \
  constexpr Runner k##NAME{NAME##_encode, NAME##_decode, NAME##_replay};

#define HCE_NCM_DEFAULT_TARGET
HCE_NCM_RUNNER(scalar, simd::Scalar, HCE_NCM_DEFAULT_TARGET)
#if HCE_NCM_X86
HCE_NCM_RUNNER(sse2, simd::Sse2, HCE_NCM_DEFAULT_TARGET)
HCE_NCM_RUNNER(avx2, simd::Avx2, HCE_NCM_TARGET_AVX2)
HCE_NCM_RUNNER(avx512, simd::Avx512, HCE_NCM_TARGET_AVX512)
#endif
#if HCE_NCM_NEON
HCE_NCM_RUNNER(neon, simd::Neon, HCE_NCM_DEFAULT_TARGET)
#endif
#undef HCE_NCM_RUNNER
#undef HCE_NCM_DEFAULT_TARGET

}  // namespace

const Runner& runner_for(SimdPath path) noexcept {
  switch (path) {
#if HCE_NCM_X86
    case SimdPath::sse2: return ksse2;
    case SimdPath::avx2: return kavx2;
    case SimdPath::avx512: return kavx512;
#endif
#if HCE_NCM_NEON
    case SimdPath::neon: return kneon;
#endif
    default: return kscalar;
  }
}

}  // namespace hce::ncm
