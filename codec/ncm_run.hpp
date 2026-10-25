// ncm_run.hpp - the neural CM codec's block loops, compiled once per SIMD
// kernel path so the kernels inline into them (ncm_run.cpp).
#pragma once

#include "codec/ncm_kernels.hpp"
#include "hc_framed.hpp"

#include <cstddef>
#include <utility>

namespace hce::ncm {

struct CMModel;
class Pipe;
namespace p2 {
struct CMModel2;
}
using BlockRange = std::pair<std::size_t, std::size_t>;  // [begin, end) of a frame

template <class Model>
struct RunnerFor {
  // Codes frame[begin, end) into out, at most out.size() bytes. Returns the
  // payload size, or 0 when it does not fit (the models have still learned it).
  std::size_t (*encode)(Model& m, hc::Bytes frame, std::size_t begin, std::size_t end, hc::MutBytes out) noexcept;
  // encode, with the counter half of the model on the pipe's helper thread
  // (ncm_pipe.hpp): the same bits and the same model state afterwards.
  // Profile 1 only: profile 2's entry runs encode and leaves the pipe idle.
  std::size_t (*encode_piped)(Model& m, Pipe& pipe, hc::Bytes frame, std::size_t begin, std::size_t end,
                              hc::MutBytes out) noexcept;
  // Decodes usize bytes of payload into w; payload_corrupt unless it is consumed exactly.
  hc::Errc (*decode)(Model& m, hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept;
  // Updates the models over frame blocks as encoding them would, without coding (R12 replay).
  void (*replay)(Model& m, hc::Bytes frame, const BlockRange* blocks, std::size_t n) noexcept;
};

using Runner = RunnerFor<CMModel>;          // profile 1
using Runner2 = RunnerFor<p2::CMModel2>;    // profile 2

// The loops for `path`; the caller has checked simd_available(path).
const Runner& runner_for(SimdPath path) noexcept;
const Runner2& runner2_for(SimdPath path) noexcept;

}  // namespace hce::ncm
