// fuzz_framed: FRAMED v3 decoding (decode_stream<NeuralCMCodec>) of hostile
// input under the untrusted limits, as `hce d` and hce.decompress() run it.
// Each input is decoded twice: as given (CRC rejection, framing) and with its
// CRCs recomputed (the parameters, the block shapes and the codec behind them),
// the second time also threaded or in a salvage mode.
#include <cstddef>
#include <cstdint>

#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "fuzz_util.hpp"
#include "io/stream.hpp"

namespace {

void decode(const std::vector<std::byte>& d, unsigned threads, hc::Recovery recovery) {
  hce::MemoryInput in(d, 4093);  // short reads: every parser state sees split input
  hce::NullOutput out;
  hce::DecodeOptions o;
  o.threads = threads;
  o.recovery = recovery;
  o.limits = hc::DecodeLimits::untrusted();
  o.limits.max_memory_bytes = std::uint64_t{512} << 20;  // a level-4 stream with a 32 MiB window fits
  o.limits.max_output_bytes = std::uint64_t{64} << 20;
  (void)hce::decode_stream<hce::NeuralCMCodec>(in, out, o);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  std::vector<std::byte> d = hce::fuzz::as_bytes(data, size);
  decode(d, 1, hc::Recovery::strict);
  hce::fuzz::fix_framed_crcs(d);
  // The threaded decoder and the salvage modes too, chosen by the input's length.
  const hc::Recovery modes[] = {hc::Recovery::strict, hc::Recovery::skip, hc::Recovery::skip_and_fill};
  decode(d, size % 2 ? 2u : 1u, modes[(size / 2) % 3]);
  return 0;
}
