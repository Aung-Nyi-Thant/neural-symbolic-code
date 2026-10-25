// parallel.hpp - multi-threaded FRAMED v3 encoder for hce.
#pragma once

#include "hc_framed.hpp"
#include "io/stream.hpp"

#include <concepts>
#include <cstddef>
#include <optional>
#include <stop_token>

namespace hce {

// What the parallel encoder needs from a block codec (NeuralCMCodec, or any
// other frame encoder). Each worker owns one instance (R11).
template <class E>
concept FrameEncoder =
    std::constructible_from<E, const hc::StreamParams&> &&
    requires(E& e, const hc::StreamParams& p, hc::Bytes frame, std::size_t at, hc::MutBytes out) {
      { E::codec_id() } noexcept -> std::same_as<hc::u32>;  // stored in StreamParams::prior_id
      { E::memory_required(p) } noexcept -> std::same_as<hc::u64>;
      { e.reset_frame(frame) } noexcept;  // sync point: forget all state
      // Codes frame[at, end) into out (out.size() == end - at); 0 means "store it".
      { e.encode_block(at, at, out) } noexcept -> std::same_as<std::size_t>;
    };

inline constexpr unsigned kMaxThreads = 256;
inline constexpr hc::u64 kMaxFrameBytes = hc::u64{1} << 30;

struct EncodeOptions {
  hc::u8 level = 2;
  // StreamParams::prior_id to write; 0 means Enc::codec_id(). Selects a codec
  // profile (kNeuralCMCodecId2, say); rejected unless Enc accepts it.
  hc::u32 prior_id = 0;
  hc::u8 window_log = 23;                     // 8 MiB history
  hc::u8 block_log = 18;                      // 256 KiB blocks
  hc::u64 frame_bytes = hc::u64{8} << 20;     // sync interval: the unit of parallelism and recovery
  unsigned threads = 1;                       // worker threads
  std::optional<hc::u64> content_size;        // written to the header; the input must match exactly
  hc::u64 max_memory = hc::u64{1} << 30;      // workers are shed until the pipeline fits
  std::stop_token stop;                       // external cancellation; never requested by the pipeline
  // Read-only state shared by every worker's codec (a frozen
  // structural::GlobalModel for StructuralCMCodec), handed to Enc::set_context()
  // when the codec has one. It must outlive the call and not change during it.
  const void* codec_context = nullptr;
};

struct EncodeResult {
  hc::Errc status = hc::Errc::ok;
  bool size_mismatch = false;  // status is content_mismatch: input length != content_size
  hc::u64 bytes_in = 0, bytes_out = 0;
  hc::u64 frames = 0, blocks = 0, stored_blocks = 0;
  hc::u64 memory_bound = 0;  // upper bound of the pipeline's buffers and codec state
  unsigned threads = 0;      // workers actually used
  hc::u32 stream_id = 0;
};

// Checks option ranges; returns a description of the first problem, or nullptr.
const char* validate(const EncodeOptions& opt) noexcept;

// Compresses `in` into one FRAMED v3 stream on `out`. T workers encode whole
// frames in T+2 preallocated slots under semaphore back-pressure and a writer
// emits them in order, so the output is byte-identical for every thread
// count. On failure no END trailer is written: partial output can never pass
// for a complete stream.
template <FrameEncoder Enc>
EncodeResult encode_parallel(InputStream& in, OutputStream& out, const EncodeOptions& opt);

}  // namespace hce
