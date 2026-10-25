// archive.hpp - Phase 2 end to end: source file <-> HCS archive, two passes over the aio layer.
//
//   source --(aio read)--> tokenize() --> serialize_tokens() --(aio write)--> token spool
//   pass 1: spool --(aio read)--> GlobalModelBuilder --> frozen GlobalModel
//   pass 2: spool --(aio read)--> IDs --> encode_parallel<StructuralCMCodec>, T workers
//           sharing the frozen model read-only --(aio write)--> archive
//
// Decoding reads the model section first, freezes the model, then decodes the
// token stream with T workers (decode_stream<StructuralCMCodec>) and writes
// the source back through the aio layer, checking its length and CRC.
//
// HCS archive, little-endian:
//    0 "HCS1"       4 version u8 (1)   5 level u8   6 reserved u16 (0)
//    8 source_bytes u64   16 tokens u64   24 model_bytes u64
//   32 source CRC-32C u32   36 model CRC-32C u32 (of the HGM1 image)   40 header CRC-32C u32 (of 0..39)
//   44 model section: model_bytes of FRAMED v3, NeuralCMCodec, content = the HGM1 image
//      token stream:  FRAMED v3, StructuralCMCodec, content = tokens x u32 LE IDs
#pragma once

#include "aio/async_io.hpp"
#include "hc_framed.hpp"
#include "structural/global_model.hpp"

#include <filesystem>

namespace hce::structural {

inline constexpr std::size_t kArchiveHeaderSize = 44;
inline constexpr hc::u8 kArchiveVersion = 1;

struct ArchiveOptions {
  hc::u8 level = 2;                          // both codecs, 0..4
  unsigned threads = 1;                      // encoder and decoder workers
  hc::u64 frame_tokens = hc::u64{1} << 20;   // tokens per frame: the unit of parallelism (>= 16384)
  hc::u64 chunk_bytes = hc::u64{4} << 20;    // source bytes per tokenize() call, cut after a newline
  aio::Options io{};                         // backend, queue depth, chunk size, O_DIRECT
  GlobalModelLimits model_limits{};
  hc::u64 max_memory = hc::u64{1} << 30;     // decoder memory bound (DecodeLimits::max_memory_bytes)
  std::filesystem::path spool;               // token spool; empty: <archive>.spool, removed afterwards
};

struct ArchiveStats {
  hc::Errc status = hc::Errc::ok;
  const char* stage = "";  // the stage that failed: open, spool, pass1, model, pass2, header, decode, verify, finish
  int sys_error = 0;
  hc::u64 source_bytes = 0, tokens = 0, symbols = 0, dictionary_bytes = 0;
  hc::u64 spool_bytes = 0, model_bytes = 0, stream_bytes = 0, archive_bytes = 0;
  unsigned threads = 0;
  const char* backend = "";
  double seconds[3]{};  // compress: spool, pass 1, pass 2; decompress: model, tokens, -
};

ArchiveStats compress_file(const std::filesystem::path& source, const std::filesystem::path& archive,
                           const ArchiveOptions& opt = {});
// On failure the output file is removed: nothing unverified is left behind.
ArchiveStats decompress_file(const std::filesystem::path& archive, const std::filesystem::path& output,
                             const ArchiveOptions& opt = {});

// The first two stages on their own. The spool is a sequence of records
// [u32 LE length][HST1 image], one per tokenize() call.
ArchiveStats write_token_spool(const std::filesystem::path& source, const std::filesystem::path& spool,
                               const ArchiveOptions& opt, hc::u32& source_crc);
hc::Errc build_global_model(const std::filesystem::path& spool, const ArchiveOptions& opt, GlobalModel& out,
                            hc::u64* tokens = nullptr);

}  // namespace hce::structural
