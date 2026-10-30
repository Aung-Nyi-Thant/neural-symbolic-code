// fuzz_structural: HCS archive decoding (structural::decompress_stream) under
// tight limits, as `hce d` runs it on a structural archive. Each input is
// decoded as given and with its archive-header and FRAMED CRCs recomputed.
// The model CRC and the content CRCs cannot be recomputed without decoding;
// fuzz_codec reaches the token codec's block decoder directly.
#include <cstddef>
#include <cstdint>

#include "fuzz_util.hpp"
#include "io/stream.hpp"
#include "structural/archive.hpp"

namespace st = hce::structural;

namespace {

void decode(const std::vector<std::byte>& d, unsigned threads) {
  hce::MemoryInput in(d, 4093);
  hce::NullOutput out;
  st::ArchiveOptions o;
  o.threads = threads;
  o.max_memory = std::uint64_t{256} << 20;
  o.max_output = std::uint64_t{16} << 20;
  o.model_limits.max_symbols = 1u << 18;
  o.model_limits.max_dictionary_bytes = std::uint64_t{16} << 20;
  (void)st::decompress_stream(in, out, o);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  std::vector<std::byte> d = hce::fuzz::as_bytes(data, size);
  decode(d, 1);
  if (d.size() < st::kArchiveHeaderSize) return 0;
  hc::store_le<hc::u32>(&d[40], hc::crc32c(hc::Bytes(d).first(40)));
  hce::fuzz::fix_framed_crcs(d, st::kArchiveHeaderSize);
  decode(d, size % 2 ? 2u : 1u);
  return 0;
}
