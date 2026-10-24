// structural_cm_codec.hpp - Phase 2, pass 2: token-level context mixing with a global model.
//
// The container content is the token stream as 4-byte records: one ID per
// token, u32 little-endian, IDs of a structural::GlobalModel. Each token
// is coded as at most three kinds of binary decision:
//
//   1. local flag:  "the token is the one that last followed (h2, h1) in this
//      frame" (an adaptive order-2 successor table, reset with the frame);
//   2. global flag: "the token is the global successor of h1" (the frozen
//      model: statistics over the whole input, across every frame);
//   3. otherwise the ID itself, Elias-gamma coded: 5 bits of length, then the
//      mantissa, so early (low-numbered) symbols cost few decisions.
//
// Every decision mixes eight counter experts over wide token contexts (orders
// 0-3 in tokens, a skip-gram, the last keyword, the line's first token and
// the structural state: scope depth, bracket depth, indentation, position in
// the line) through the engine's integer two-layer mixer (ncm kernels: the
// same bit-exact scalar/SSE2/AVX2/NEON code as NeuralCMCodec). All
// arithmetic is integer. Tables live in the caller's arena and reset per
// frame at a cost proportional to what the frame touched (R19-R21); the
// global model is outside the arena, shared read-only by every worker.
#pragma once

#include "hc_framed.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hce {

namespace structural {
class GlobalModel;
}

// StreamParams::prior_id = family | profile.
inline constexpr hc::u32 kStructuralCMFamily = 0x53434D00u;  // "SCM"
inline constexpr hc::u8 kStructuralCMProfile = 1;
inline constexpr hc::u32 kStructuralCMCodecId = kStructuralCMFamily | kStructuralCMProfile;
inline constexpr hc::u8 kStructuralCMMaxLevel = 4;
inline constexpr std::size_t kTokenRecordBytes = 4;

class StructuralCMCodec {
 public:
  // Decoder role (hc::BlockCodec): default-constructed, then bound to an arena.
  StructuralCMCodec() noexcept;
  // Encoder role (hce::FrameEncoder): allocates and owns an arena of memory_required(p).
  explicit StructuralCMCodec(const hc::StreamParams& p);
  StructuralCMCodec(StructuralCMCodec&& other) noexcept;
  StructuralCMCodec& operator=(StructuralCMCodec&& other) noexcept;
  StructuralCMCodec(const StructuralCMCodec&) = delete;
  StructuralCMCodec& operator=(const StructuralCMCodec&) = delete;
  ~StructuralCMCodec();

  static hc::u32 codec_id() noexcept { return kStructuralCMCodecId; }
  static hc::Errc accepts(const hc::StreamParams& p) noexcept;
  static hc::u64 memory_required(const hc::StreamParams& p) noexcept;

  // The frozen global model, as a `const structural::GlobalModel*`: set it
  // before bind() or the first encode_block(). It must outlive the codec and
  // stay unchanged; every worker reads it concurrently through const access
  // only. encode_parallel() and decode_stream() pass their options'
  // codec_context here. Without a model, bind() fails (unsupported_feature)
  // and the encoder stores every block.
  void set_context(const void* global_model) noexcept;
  void set_model(const structural::GlobalModel* model) noexcept { set_context(model); }

  // ---- decoding: hc::BlockCodec ----
  hc::Errc bind(const hc::StreamParams& p, hc::MutBytes arena) noexcept;
  void reset_frame() noexcept;
  // usize and the frame offset must be whole records; every ID must be in the model.
  hc::Errc decode_block(hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept;

  // ---- encoding: hce::FrameEncoder ----
  void reset_frame(hc::Bytes frame) noexcept;
  std::size_t encode_block(std::size_t begin, std::size_t end, hc::MutBytes out) noexcept;

  // ---- diagnostics (encoder role) ----
  struct Counters {
    hc::u64 cm_blocks = 0, stored_blocks = 0, replays = 0;
    hc::u64 tokens = 0, local_hits = 0, global_hits = 0, explicit_ids = 0;
  };
  const Counters& counters() const noexcept { return counters_; }
  // Tests only: mixer kernels for codecs bound from now on (scalar, sse2, avx2,
  // neon; nullptr restores the default). False when that path is not available.
  static bool test_select_simd(const char* name) noexcept;

 private:
  struct Model;
  void replay() noexcept;

  Model* m_ = nullptr;  // placement-constructed inside the arena
  std::unique_ptr<std::byte[]> owned_;
  const structural::GlobalModel* global_ = nullptr;
  hc::Bytes frame_;
  std::vector<std::pair<std::size_t, std::size_t>> cm_blocks_;
  unsigned replays_ = 0;
  bool store_rest_ = false;
  Counters counters_;
};

}  // namespace hce
