// neural_cm_codec.hpp - the engine's neural-symbolic context-mixing codec.
//
// Symbolic experts (order 0-6, word, column, log-field and sparse contexts,
// and a long-range match model) predict every bit; a two-layer integer neural
// mixer combines their logits; two adaptive probability maps refine the
// result; an exact binary arithmetic coder codes it. During long matches the
// coder switches to one decision per byte. All arithmetic is integer with
// fixed tables (R14), the SIMD kernels are bit-exact (R15), every table lives
// in the arena the caller provides (R21) and resets completely at each frame
// (R20), at a cost proportional to what the frame touched (R19).
#pragma once

#include "hc_framed.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hce {

namespace ncm {
struct CMModel;
class Pipe;
}  // namespace ncm

// StreamParams::prior_id = family | profile. Profile 1: text, logs and mixed data.
inline constexpr hc::u32 kNeuralCMFamily = 0x4E434D00u;  // "NCM"
inline constexpr hc::u8 kNeuralCMProfile = 1;
inline constexpr hc::u32 kNeuralCMCodecId = kNeuralCMFamily | kNeuralCMProfile;
inline constexpr hc::u8 kNeuralCMMaxLevel = 4;

class NeuralCMCodec {
 public:
  // Decoder role (hc::BlockCodec): default-constructed, then bound to an arena.
  NeuralCMCodec() noexcept;
  // Encoder role (hce::FrameEncoder): allocates and owns an arena of memory_required(p).
  explicit NeuralCMCodec(const hc::StreamParams& p);
  NeuralCMCodec(NeuralCMCodec&& other) noexcept;
  NeuralCMCodec& operator=(NeuralCMCodec&& other) noexcept;
  NeuralCMCodec(const NeuralCMCodec&) = delete;
  NeuralCMCodec& operator=(const NeuralCMCodec&) = delete;
  ~NeuralCMCodec();

  static hc::u32 codec_id() noexcept { return kNeuralCMCodecId; }
  // Codec family, profile and level; checked before anything is allocated.
  static hc::Errc accepts(const hc::StreamParams& p) noexcept;
  // Exact arena size for these parameters: tables, model state, alignment slack.
  static hc::u64 memory_required(const hc::StreamParams& p) noexcept;

  // ---- decoding: hc::BlockCodec ----
  hc::Errc bind(const hc::StreamParams& p, hc::MutBytes arena) noexcept;
  void reset_frame() noexcept;  // R20: every table back to its initial state
  hc::Errc decode_block(hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept;

  // ---- encoding: hce::FrameEncoder ----
  void reset_frame(hc::Bytes frame) noexcept;
  std::size_t encode_block(std::size_t begin, std::size_t end, hc::MutBytes out) noexcept;
  // Encoder role: code blocks on two threads (ncm_pipe.hpp), the counter half
  // of the model on a helper this codec starts at its first block. Encoder side
  // only: the output is the same bytes either way. Falls back to one thread
  // when the helper cannot be had.
  void set_pipeline(bool on) noexcept { pipeline_ = on; }
  static constexpr hc::u64 kPipelineBytes = hc::u64{256} << 10;  // the helper's ring

  // ---- diagnostics and tests ----
  struct Counters {
    hc::u64 cm_blocks = 0, stored_blocks = 0, replays = 0;
  };
  const Counters& counters() const noexcept { return counters_; }
  static const char* simd_path() noexcept;  // kernels in use: scalar, sse2, avx2 or neon
  // Tests only: try CM even on blocks the pre-check would store, to exercise replay.
  static void test_force_cm(bool on) noexcept;
  // Tests only: kernels for codecs bound from now on (scalar, sse2, avx2, neon;
  // nullptr restores the default). False when that path is not available here.
  static bool test_select_simd(const char* name) noexcept;

 private:
  void replay() noexcept;

  ncm::CMModel* m_ = nullptr;            // placement-constructed inside the arena
  hc::ArenaPtr owned_;                   // encoder role: the arena it owns
  std::unique_ptr<ncm::Pipe> pipe_;      // encoder role, set_pipeline(true): the helper
  bool pipeline_ = false;
  // Encoder role: the frame being coded and the blocks of it coded with CM so far.
  hc::Bytes frame_;
  std::vector<std::pair<std::size_t, std::size_t>> cm_blocks_;
  unsigned replays_ = 0;
  bool store_rest_ = false;
  Counters counters_;
};

}  // namespace hce
