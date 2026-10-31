// fuzz_codec: the block decoders themselves, NeuralCMCodec and
// StructuralCMCodec::decode_block(), on hostile payloads. In a stream these
// sit behind a payload CRC, which an attacker can compute, so every payload
// shape is reachable; this target skips the framing to spend its time there.
//
// Input: byte 0 selects the codec (bit 0: NCM or the structural codec), the
// NCM profile (bit 1) and the level ((byte >> 2) % 5); byte 1 the block size
// (2^(10 + byte % 3): 1 to 4 KiB, small so that a CM decoder under the
// sanitizers runs many inputs a second). Then up to 8 blocks of {u16 usize,
// u16 csize, payload} into one 64 KiB window, as the stream decoder hands them over: usize in
// 1..capacity, 0 < csize < usize, and decoding stops at the first error.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "codec/neural_cm_codec.hpp"
#include "hc/window.hpp"
#include "structural/global_model.hpp"
#include "structural/structural_cm_codec.hpp"
#include "structural/token_stream.hpp"

namespace st = hce::structural;

namespace {

// A small frozen dictionary (a few hundred symbols) for the structural codec.
const st::GlobalModel& model() {
  static const st::GlobalModel m = [] {
    std::string src;
    for (int i = 0; i < 200; ++i)
      src += "def f" + std::to_string(i % 37) + "(x, y):\n    return {\"k\": [x + " + std::to_string(i) + ", y]}\n";
    st::GlobalModelBuilder b;
    st::GlobalModel out;
    if (b.add(st::tokenize(src, st::Language::python)) != hc::Errc::ok || b.freeze(out) != hc::Errc::ok) std::abort();
    return out;
  }();
  return m;
}

// One arena, grown as needed and reused across inputs, as the stream decoder
// reuses its arena across streams.
hc::MutBytes arena(std::size_t n) {
  static std::unique_ptr<std::byte[]> buf;
  static std::size_t cap = 0;
  if (n > cap) {
    buf = std::make_unique<std::byte[]>(n);
    cap = n;
  }
  return {buf.get(), n};
}

template <class Codec>
void run(Codec& codec, const hc::StreamParams& p, const std::uint8_t* data, std::size_t size) {
  if (Codec::accepts(p) != hc::Errc::ok) return;
  const std::size_t window = std::size_t{1} << 16;
  const hc::MutBytes mem = arena(window + static_cast<std::size_t>(Codec::memory_required(p)));
  hc::Window w;
  w.bind(mem.first(window));
  if (codec.bind(p, mem.subspan(window)) != hc::Errc::ok) return;
  codec.reset_frame();
  w.reset_frame();
  std::vector<std::byte> payload;
  for (int block = 0; block < 8 && size >= 4; ++block) {
    const std::uint32_t usize = 1 + (static_cast<std::uint32_t>(data[0] | data[1] << 8) % p.block_capacity());
    std::uint32_t csize = static_cast<std::uint32_t>(data[2] | data[3] << 8);
    data += 4;
    size -= 4;
    csize = static_cast<std::uint32_t>(std::min<std::size_t>(csize, size));
    if (csize == 0 || csize >= usize) return;  // check_block_shape() rejects these first
    payload.assign(reinterpret_cast<const std::byte*>(data), reinterpret_cast<const std::byte*>(data) + csize);
    data += csize;
    size -= csize;
    const std::uint64_t start = w.pos();
    if (codec.decode_block(payload, usize, w) != hc::Errc::ok) return;
    if (w.pos() - start != usize) return;  // the stream decoder rejects the block here
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size < 2) return 0;
  hc::StreamParams p;
  p.level = static_cast<hc::u8>((data[0] >> 2) % 5);
  p.window_log = 16;
  p.block_log = static_cast<hc::u8>(10 + data[1] % 3);
  if (data[0] & 1) {
    p.prior_id = hce::kStructuralCMCodecId;
    hce::StructuralCMCodec codec;
    codec.set_model(&model());
    run(codec, p, data + 2, size - 2);
  } else {
    p.prior_id = (data[0] & 2) ? hce::kNeuralCMCodecId2 : hce::kNeuralCMCodecId;
    hce::NeuralCMCodec codec;
    run(codec, p, data + 2, size - 2);
  }
  return 0;
}
