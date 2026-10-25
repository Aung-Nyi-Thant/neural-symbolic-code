// pipeline.hpp - multi-threaded FRAMED v3 decoder for hce.
#pragma once

#include "hc_framed.hpp"
#include "io/stream.hpp"

#include <cstddef>
#include <stop_token>

namespace hce {

struct DecodeOptions {
  hc::DecodeLimits limits{};  // R19; max_memory_bytes also bounds the pipeline's buffers
  hc::Recovery recovery = hc::Recovery::strict;
  std::byte fill{0};
  hc::GapFn on_gap = nullptr;  // always called from one thread, in stream order
  void* gap_ctx = nullptr;
  unsigned threads = 1;                        // 1 = sequential decoder, no extra threads
  hc::u64 chunk_bytes = hc::u64{4} << 20;      // decoded bytes per work unit (raised to the block limit)
  std::stop_token stop;                        // external cancellation; never requested by the pipeline
  // Read-only state shared by every worker's codec (see EncodeOptions), handed
  // to Codec::set_context() when the codec has one.
  const void* codec_context = nullptr;
};

struct DecodeResult {
  hc::Errc status = hc::Errc::ok;
  hc::DecodeStats stats;
  hc::u64 bytes_in = 0, bytes_out = 0;
  hc::u64 memory_bound = 0;   // upper bound of buffers, windows and codec state
  unsigned threads = 0;       // pipeline workers used; 0 = sequential decoder
  bool params_known = false;  // some stream or frame parameters were recognized
};

// Decodes every concatenated stream in `in` to `out`.
//
// With threads > 1 and enough memory under limits.max_memory_bytes, frames are
// decoded in parallel: a splitter validates the framing and enforces the R19
// budget in stream order, T workers decode frames with private codec state
// (R11) in T+2 preallocated slots, and a writer emits them in order.
// Otherwise the sequential decoder runs. Both paths produce the same output,
// status, gap reports and statistics; the one difference is that in skip mode
// the pipeline does not refund the budget for blocks lost to damage, so near a
// ratio or output limit it may stop earlier.
template <hc::BlockCodec Codec>
DecodeResult decode_stream(InputStream& in, OutputStream& out, const DecodeOptions& opt);

}  // namespace hce
