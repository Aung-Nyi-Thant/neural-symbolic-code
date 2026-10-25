// ncm_run.cpp - the block loops of the neural CM codec, one instantiation per
// SIMD kernel policy. Each entry point is compiled for its instruction set and
// flattened: the model's bit loop and the kernels inline into one function,
// so an AVX2 build of the loop runs only where AVX2 was detected. Every
// instantiation computes the same integers (R15) and writes the same bits.
#include "codec/ncm_run.hpp"

#include "codec/ncm_model.hpp"
#include "codec/ncm_pipe.hpp"
#include "codec/ncm_model2.hpp"
#include "codec/ncm_simd.hpp"

#include <algorithm>

namespace hce::ncm {
namespace {

template <class K, class Model>
inline std::size_t encode_loop(Model& m, hc::Bytes frame, std::size_t begin, std::size_t end,
                               hc::MutBytes out) noexcept {
  ArithEncoder enc(out.data(), out.size());
  EncodeIO io{enc};
  const FrameHistory h{frame.data()};
  m.begin_block(h, begin);
  for (std::size_t i = begin; i < end && !enc.full(); ++i) {
    const int c = static_cast<int>(frame[i]);
    m.template code_byte<K>(io, c);
    m.byte_update(h, c);
  }
  if (!enc.full()) enc.flush();
  return enc.full() ? 0 : enc.size();
}

// Stage one, on the pipe's helper: the counters, contexts and match model.
struct ProduceJob {
  CMModel* m;
  hc::Bytes frame;
  std::size_t begin, end;
};

void produce(void* ctx, Pipe& pipe) noexcept {
  const ProduceJob& j = *static_cast<const ProduceJob*>(ctx);
  CMModel& m = *j.m;
  const FrameHistory h{j.frame.data()};
  m.begin_block(h, j.begin);
  for (std::size_t i = j.begin; i < j.end; ++i) {
    const int c = static_cast<int>(j.frame[i]);
    if (!m.produce_byte(pipe, c)) return;
    m.byte_update(h, c);
  }
}

// Stage two, here: mixing, APMs and the coder, one record per coded decision.
template <class K>
inline std::size_t encode_piped_loop(CMModel& m, Pipe& pipe, hc::Bytes frame, std::size_t begin, std::size_t end,
                                     hc::MutBytes out) noexcept {
  ArithEncoder enc(out.data(), out.size());
  EncodeIO io{enc};
  const CMModel::MixTables mt = m.mix_tables();  // before the helper starts writing the model
  ProduceJob job{&m, frame, begin, end};
  pipe.start(produce, &job);
  while (const Pipe::Rec* r = pipe.next()) {
    const int meta = static_cast<u16>((*r)[kLanes - 1]);
    const int y = (meta >> 3) & 1;
    if (meta & kRecFast) {
      io.bit(static_cast<u16>((*r)[0]), y);
    } else {
      simd::Lanes x = *r;
      const u32 cc = static_cast<u16>(x[kLanes - 2]);
      const int ms = x[kLanes - 3];
      x[kLanes - 3] = x[kLanes - 2] = x[kLanes - 1] = 0;
      CMModel::stretch_nodes(x);
      CMModel::mix_code<K>(mt, io, x, cc & 0xFF, cc >> 8, ms, meta & 7, y);
    }
    pipe.pop();
    if (enc.full()) {
      pipe.abort();  // the caller stores the block and replays the frame (R12)
      return 0;
    }
  }
  pipe.join();
  enc.flush();
  return enc.full() ? 0 : enc.size();
}

// Profile 2 has no pipelined loop yet: its records would not fit one Lanes.
template <class K>
inline std::size_t encode_piped_loop(p2::CMModel2& m, Pipe&, hc::Bytes frame, std::size_t begin,
                                     std::size_t end, hc::MutBytes out) noexcept {
  return encode_loop<K>(m, frame, begin, end, out);
}

template <class K, class Model>
inline hc::Errc decode_loop(Model& m, hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept {
  ArithDecoder dec(payload);
  DecodeIO io{dec};
  const WindowHistory h{&w, w.pos() - w.frame_offset()};
  m.begin_block(h, w.frame_offset());
  for (hc::u32 i = 0; i < usize; ++i) {
    const int c = m.template code_byte<K>(io, 0);
    if (dec.overrun()) return hc::Errc::payload_corrupt;  // ran past the payload: corrupt
    w.put(static_cast<std::byte>(c));
    m.byte_update(h, c);
  }
  return dec.exact() ? hc::Errc::ok : hc::Errc::payload_corrupt;
}

template <class K, class Model>
inline void replay_loop(Model& m, hc::Bytes frame, const BlockRange* blocks, std::size_t n) noexcept {
  ReplayIO io;
  const FrameHistory h{frame.data()};
  for (std::size_t b = 0; b < n; ++b) {
    m.begin_block(h, blocks[b].first);
    for (std::size_t i = blocks[b].first; i < blocks[b].second; ++i) {
      const int c = static_cast<int>(frame[i]);
      m.template code_byte<K>(io, c);
      m.byte_update(h, c);
    }
  }
}

// One Runner per policy and model; ATTR: the instruction set its loops are compiled for.
#define HCE_NCM_RUNNER(NAME, K, ATTR, MODEL)                                                                         \
  ATTR __attribute__((flatten)) std::size_t NAME##_encode(MODEL& m, hc::Bytes frame, std::size_t begin,     \
                                                          std::size_t end, hc::MutBytes out) noexcept {       \
    return encode_loop<K>(m, frame, begin, end, out);                                                         \
  }                                                                                                           \
  ATTR __attribute__((flatten)) std::size_t NAME##_encode_piped(MODEL& m, Pipe& pipe, hc::Bytes frame,        \
                                                                std::size_t begin, std::size_t end,           \
                                                                hc::MutBytes out) noexcept {                  \
    return encode_piped_loop<K>(m, pipe, frame, begin, end, out);                                             \
  }                                                                                                           \
  ATTR __attribute__((flatten)) hc::Errc NAME##_decode(MODEL& m, hc::Bytes payload, hc::u32 usize,          \
                                                       hc::Window& w) noexcept {                              \
    return decode_loop<K>(m, payload, usize, w);                                                              \
  }                                                                                                           \
  ATTR __attribute__((flatten)) void NAME##_replay(MODEL& m, hc::Bytes frame, const BlockRange* blocks,     \
                                                   std::size_t n) noexcept {                                  \
    replay_loop<K>(m, frame, blocks, n);                                                                      \
  }                                                                                                           \
  constexpr RunnerFor<MODEL> k##NAME{NAME##_encode, NAME##_encode_piped, NAME##_decode, NAME##_replay};

#define HCE_NCM_DEFAULT_TARGET
HCE_NCM_RUNNER(scalar, simd::Scalar, HCE_NCM_DEFAULT_TARGET, CMModel)
HCE_NCM_RUNNER(p2_scalar, simd::Scalar, HCE_NCM_DEFAULT_TARGET, p2::CMModel2)
#if HCE_NCM_X86
HCE_NCM_RUNNER(sse2, simd::Sse2, HCE_NCM_DEFAULT_TARGET, CMModel)
HCE_NCM_RUNNER(avx2, simd::Avx2, HCE_NCM_TARGET_AVX2, CMModel)
HCE_NCM_RUNNER(avx512, simd::Avx512, HCE_NCM_TARGET_AVX512, CMModel)
HCE_NCM_RUNNER(p2_sse2, simd::Sse2, HCE_NCM_DEFAULT_TARGET, p2::CMModel2)
HCE_NCM_RUNNER(p2_avx2, simd::Avx2, HCE_NCM_TARGET_AVX2, p2::CMModel2)
HCE_NCM_RUNNER(p2_avx512, simd::Avx512, HCE_NCM_TARGET_AVX512, p2::CMModel2)
#endif
#if HCE_NCM_NEON
HCE_NCM_RUNNER(neon, simd::Neon, HCE_NCM_DEFAULT_TARGET, CMModel)
HCE_NCM_RUNNER(p2_neon, simd::Neon, HCE_NCM_DEFAULT_TARGET, p2::CMModel2)
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

const Runner2& runner2_for(SimdPath path) noexcept {
  switch (path) {
#if HCE_NCM_X86
    case SimdPath::sse2: return kp2_sse2;
    case SimdPath::avx2: return kp2_avx2;
    case SimdPath::avx512: return kp2_avx512;
#endif
#if HCE_NCM_NEON
    case SimdPath::neon: return kp2_neon;
#endif
    default: return kp2_scalar;
  }
}

}  // namespace hce::ncm
