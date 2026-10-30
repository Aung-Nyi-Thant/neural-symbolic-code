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
//    0 "HCS1"       4 version u8 (1)   5 level u8   6 language u8 (informational)   7 reserved u8 (0)
//    8 source_bytes u64   16 tokens u64   24 model_bytes u64
//   32 source CRC-32C u32   36 model CRC-32C u32 (of the HGM1 image)   40 header CRC-32C u32 (of 0..39)
//   44 model section: model_bytes of FRAMED v3, NeuralCMCodec, content = the HGM1 image
//      token stream:  FRAMED v3, StructuralCMCodec, content = tokens x u32 LE IDs
#pragma once

#include "aio/async_io.hpp"
#include "hc_framed.hpp"
#include "io/stream.hpp"
#include "structural/global_model.hpp"
#include "structural/token_stream.hpp"

#include <array>
#include <filesystem>
#include <stop_token>

namespace hce::structural {

inline constexpr std::size_t kArchiveHeaderSize = 44;
inline constexpr std::array<std::byte, 4> kArchiveMagic{std::byte{'H'}, std::byte{'C'}, std::byte{'S'}, std::byte{'1'}};
inline constexpr hc::u8 kArchiveVersion = 1;

struct ArchiveOptions {
  hc::u8 level = 2;                          // both codecs, 0..4
  unsigned threads = 1;                      // encoder and decoder workers
  hc::u64 frame_tokens = hc::u64{1} << 20;   // tokens per frame: the unit of parallelism (>= 16384)
  hc::u64 chunk_bytes = hc::u64{4} << 20;    // source bytes per tokenize() call, cut after a newline
  Language language = Language::generic;     // keyword table for tokenizing (compression only)
  aio::Options io{};                         // backend, queue depth, chunk size, O_DIRECT
  GlobalModelLimits model_limits{};
  // Decoder memory bound, in bytes. It covers the codec workers of each
  // decoding phase together with what that phase keeps: while the model
  // section decodes, its decoder plus the dictionary image (capped at half
  // the budget); while the token stream decodes, its workers plus the parsed
  // model and the 1 MiB output stage. A budget that cannot hold the parsed
  // model fails with limit_memory before any source byte is written. Not
  // covered: the brief overlap of the image and the parsed model while the
  // model is parsed (at most the image size again plus 15 bytes per
  // symbol), and the caller's own buffers.
  hc::u64 max_memory = hc::u64{1} << 30;
  // Largest source (decompressed) size accepted, in bytes. Checked against
  // the archive header before anything is decoded: a larger archive fails
  // with limit_output and writes nothing. The dictionary has its own bound,
  // model_limits.max_dictionary_bytes.
  hc::u64 max_output = ~hc::u64{0};
  hc::u8 max_level = 255;  // decoding: archives of a higher level fail with limit_level, from the header
  // The token spool (compression), removed afterwards. It is always created
  // new and owner-only (0600): never over an existing file or through a
  // symbolic link. `spool` names it exactly (it must not exist); otherwise a
  // fresh name .hce-spool-<random>.tmp goes in `spool_dir`, or by default
  // next to the archive for compress_file() and in the temporary directory
  // for compress_stream(). Both passes check they read the file they wrote.
  std::filesystem::path spool;
  std::filesystem::path spool_dir;
  std::stop_token stop;                      // cancellation: every stage stops with Errc::cancelled
};

struct ArchiveStats {
  hc::Errc status = hc::Errc::ok;
  // Where it failed: open, alias (the file functions' source and destination
  // are one file), read, spool, pass1, model, pass2, write, header, decode,
  // verify, finish, or sync-dir: the file functions published the output but
  // could not sync its directory, so it may not survive a crash; it is kept.
  const char* stage = "";
  int sys_error = 0;
  hc::u64 source_bytes = 0, tokens = 0, symbols = 0, dictionary_bytes = 0;
  hc::u64 spool_bytes = 0, model_bytes = 0, stream_bytes = 0, archive_bytes = 0;
  unsigned threads = 0;
  Language language = Language::generic;  // decompression: as recorded (to_string() says "unknown" if newer)
  const char* backend = "";
  double seconds[3]{};  // compress: spool, pass 1, pass 2; decompress: model, tokens, -
};

// Streams: the source is read once (into the spool, which both passes then
// read through aio); the archive is written and read strictly in order, so
// pipes work. On failure a stream output holds an incomplete archive or
// unverified source: callers discard it.
ArchiveStats compress_stream(InputStream& source, OutputStream& archive, const ArchiveOptions& opt = {});
ArchiveStats decompress_stream(InputStream& archive, OutputStream& source, const ArchiveOptions& opt = {});

// Files, through aio at both ends. The output is written to a temporary file
// beside it and renamed over `archive` / `output` only once complete and, for
// decompression, verified: on any failure, cancellation included, an existing
// destination keeps its content and only the temporary file is removed. A
// source and destination that are the same file (hard or symbolic links
// included) fail with stage "alias" before anything is written.
ArchiveStats compress_file(const std::filesystem::path& source, const std::filesystem::path& archive,
                           const ArchiveOptions& opt = {});
ArchiveStats decompress_file(const std::filesystem::path& archive, const std::filesystem::path& output,
                             const ArchiveOptions& opt = {});

// True when `prefix` starts with the archive magic: tells HCS archives from FRAMED v3 streams.
bool is_archive(hc::Bytes prefix) noexcept;

// The first two stages on their own. The spool is a sequence of records
// [u32 LE length][HST1 image], one per tokenize() call. write_token_spool()
// creates `spool` new and owner-only (an existing name fails with io_error,
// EEXIST) and removes it again if spooling fails. build_global_model() with
// `expect` fails with content_mismatch unless `spool` is still that file.
ArchiveStats write_token_spool(InputStream& source, const std::filesystem::path& spool, const ArchiveOptions& opt,
                               hc::u32& source_crc);
ArchiveStats write_token_spool(const std::filesystem::path& source, const std::filesystem::path& spool,
                               const ArchiveOptions& opt, hc::u32& source_crc);
hc::Errc build_global_model(const std::filesystem::path& spool, const ArchiveOptions& opt, GlobalModel& out,
                            hc::u64* tokens = nullptr, const aio::FileId* expect = nullptr);

}  // namespace hce::structural
