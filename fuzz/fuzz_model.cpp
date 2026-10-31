// fuzz_model: the two parsers of decoded structural data, GlobalModel::parse()
// (the HGM1 dictionary image) and deserialize_tokens() (the HST1 token spool
// image). Neither image carries a CRC of its own, so every input reaches the
// parser. A parsed model must serialize back to the same image.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "structural/global_model.hpp"
#include "structural/token_stream.hpp"

namespace st = hce::structural;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const hc::Bytes image(reinterpret_cast<const std::byte*>(data), size);
  st::GlobalModelLimits ml;
  ml.max_symbols = 1u << 18;
  ml.max_dictionary_bytes = std::uint64_t{16} << 20;
  st::GlobalModel m;
  if (st::GlobalModel::parse(image, ml, m) == hc::Errc::ok) {
    const std::vector<std::uint8_t> again = m.serialize();
    st::GlobalModel m2;
    if (st::GlobalModel::parse(hc::Bytes(reinterpret_cast<const std::byte*>(again.data()), again.size()), ml, m2) !=
            hc::Errc::ok ||
        m2.size() != m.size() || m2.serialize() != again)
      std::abort();  // a model that parses must survive its own round trip
  }
  st::TokenStreamLimits tl;
  tl.max_tokens = 1u << 20;
  tl.max_value_bytes = std::uint64_t{16} << 20;
  std::vector<st::StructuralToken> tokens;
  if (st::deserialize_tokens(image, tokens, tl) == hc::Errc::ok) (void)st::detokenize(tokens);
  return 0;
}
