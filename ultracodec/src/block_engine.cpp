#include "block_engine.hpp"

#include "dedup.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>

namespace ultracodec {

std::string_view tier_name(TierLevel tier) noexcept {
  switch (tier) {
    case TierLevel::Tier1_HyperSpeed: return "tier1-hyperspeed";
    case TierLevel::Tier2_Consumer: return "tier2-consumer";
    case TierLevel::Tier3_Balanced: return "tier3-balanced";
    case TierLevel::Tier4_Enterprise: return "tier4-enterprise";
  }
  return "unknown";
}

void ArchiveHeader::serialize(std::uint8_t out[kSize]) const noexcept {
  using namespace detail;
  std::memcpy(out, kMagic, 4);
  out[4] = version;
  out[5] = static_cast<u8>(tier);
  out[6] = flags;
  out[7] = block_log2;
  store64(out + 8, uncompressed_size);
  store32(out + 16, block_count);
  store32(out + 20, prime_bytes);
  store32(out + 24, model_param);
  store32(out + 28, crc32c(out, 28));
}

ArchiveHeader ArchiveHeader::parse(std::span<const std::uint8_t> bytes) {
  using namespace detail;
  if (bytes.size() < kSize) throw Error("ultracodec: archive shorter than its header");
  const u8* p = bytes.data();
  if (std::memcmp(p, kMagic, 4) != 0) throw Error("ultracodec: not an ultracodec archive (bad magic)");
  if (load32(p + 28) != crc32c(p, 28)) throw Error("ultracodec: header CRC mismatch");
  ArchiveHeader h;
  h.version = p[4];
  if (h.version != kFormatVersion) throw Error("ultracodec: unsupported format version");
  if (p[5] < 1 || p[5] > 4) throw Error("ultracodec: unknown tier");
  h.tier = static_cast<TierLevel>(p[5]);
  h.flags = p[6];
  if (h.flags & ~(kFlagWavefront | kFlagDedup | kFlagSnapshot)) throw Error("ultracodec: unknown header flags");
  // Tier 4 always uses (and alone uses) snapshot warm-start, without prefix or dedup.
  if (((h.flags & kFlagSnapshot) != 0) != (h.tier == TierLevel::Tier4_Enterprise) ||
      ((h.flags & kFlagSnapshot) && (h.flags & (kFlagWavefront | kFlagDedup))))
    throw Error("ultracodec: header flags do not match the tier");
  h.block_log2 = p[7];
  if (h.block_log2 < 16 || h.block_log2 > 26) throw Error("ultracodec: block size out of range");
  h.uncompressed_size = load64(p + 8);
  h.block_count = load32(p + 16);
  h.prime_bytes = load32(p + 20);
  h.model_param = load32(p + 24);
  const u64 bs = u64{1} << h.block_log2;
  if (h.uncompressed_size > (u64{1} << 56)) throw Error("ultracodec: size out of range");
  // With dedup the blocks cover the residual, checked once its size is read.
  if (h.block_count != (h.uncompressed_size + bs - 1) / bs && !(h.flags & kFlagDedup))
    throw Error("ultracodec: block count does not match size");
  if (static_cast<u64>(h.block_count) * bs < h.uncompressed_size && !(h.flags & kFlagDedup))
    throw Error("ultracodec: block count does not match size");
  if (h.prime_bytes > bs || ((h.flags & kFlagWavefront) == 0) != (h.prime_bytes == 0))
    throw Error("ultracodec: bad wavefront prefix");
  return h;
}

namespace detail {

void parallel_blocks(std::size_t n, unsigned workers, const std::function<void(std::size_t, unsigned)>& fn,
                     const std::function<void()>& on_error) {
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr error;
  std::mutex error_mutex;
  auto run = [&](unsigned w) {
    for (;;) {
      if (failed.load(std::memory_order_relaxed)) return;
      std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) return;
      try {
        fn(i, w);
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (!failed.exchange(true)) {
          error = std::current_exception();
          if (on_error) on_error();
        }
        return;
      }
    }
  };
  workers = std::max(1u, std::min<unsigned>(workers, static_cast<unsigned>(std::max<std::size_t>(n, 1))));
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  try {
    for (unsigned w = 1; w < workers; ++w) pool.emplace_back(run, w);
  } catch (...) {
    // Could not start a thread: run with what we have.
  }
  run(0);
  for (auto& t : pool) t.join();
  if (error) std::rethrow_exception(error);
}

BlockEngine::BlockEngine(TierLevel tier, std::string_view name, const EngineOptions& options,
                         TierDefaults defaults)
    : tier_(tier), name_(name), options_(options), dedup_(defaults.dedup) {
  if (options.block_size) {
    std::size_t b = options.block_size;
    if ((b & (b - 1)) != 0 || b < (std::size_t{1} << 16) || b > (std::size_t{1} << 26))
      throw Error("ultracodec: block_size must be a power of two in [64 KiB, 64 MiB]");
    block_log2_ = highbit32(static_cast<u32>(b));
  } else {
    block_log2_ = defaults.block_log2;
  }
  if (defaults.prime_bytes == 0 || options.prime_bytes == 0) {
    prime_bytes_ = 0;
  } else {
    prime_bytes_ = options.prime_bytes == SIZE_MAX ? defaults.prime_bytes : options.prime_bytes;
    prime_bytes_ = std::min(prime_bytes_, (std::size_t{1} << block_log2_) / 2);
  }
  if (options_.threads == 0) options_.threads = std::max(1u, std::thread::hardware_concurrency());
}

unsigned BlockEngine::workers_for(std::size_t blocks, u32 model_param) const noexcept {
  std::size_t w = std::min<std::size_t>({options_.threads, blocks, worker_cap(model_param)});
  return static_cast<unsigned>(std::max<std::size_t>(1, w));
}

void BlockEngine::prepare_codecs(const CodecKey& key, unsigned workers) {
  if (!(key == codec_key_)) {
    codecs_.clear();
    codec_key_ = key;
  }
  if (codecs_.size() < workers) codecs_.resize(workers);
}

BlockCodec& BlockEngine::codec(unsigned worker) {
  auto& slot = codecs_[worker];
  if (!slot) slot = make_codec(codec_key_.param, codec_key_.block, codec_key_.prime);
  return *slot;
}

std::vector<u8> BlockEngine::compress(std::span<const u8> input, EngineStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t bs = std::size_t{1} << block_log2_;
  std::vector<u8> section, residual;
  const std::span<const u8> original = input;
  if (dedup_.far_min) {
    residual = dedup_encode(input, {bs, dedup_.far_min, dedup_.near_min}, section);
    input = residual;
  }
  const std::size_t blocks = (input.size() + bs - 1) / bs;
  if (blocks > UINT32_MAX) throw Error("ultracodec: input too large");
  const u32 param = choose_model_param(input.size());
  const unsigned workers = workers_for(blocks, param);
  prepare_codecs({param, bs, prime_bytes_}, workers);

  std::vector<std::vector<u8>> payloads(blocks);
  std::vector<u32> crcs(blocks);
  std::atomic<std::size_t> stored{0};
  parallel_blocks(
      blocks, workers,
      [&](std::size_t i, unsigned w) {
        auto raw = input.subspan(i * bs, std::min(bs, input.size() - i * bs));
        std::span<const u8> prime;
        if (prime_bytes_ && i > 0) prime = input.subspan((i - 1) * bs, prime_bytes_);
        crcs[i] = crc32c(raw);
        auto& out = payloads[i];
        out.clear();
        out.reserve(raw.size() / 2 + 64);
        out.push_back(1);
        codec(w).encode(prime, raw, out);
        if (out.size() >= raw.size() + 1) {  // incompressible: store
          out.assign(1, 0);
          out.insert(out.end(), raw.begin(), raw.end());
          stored.fetch_add(1, std::memory_order_relaxed);
        }
      },
      {});

  ArchiveHeader h;
  h.tier = tier_;
  h.flags = static_cast<u8>((prime_bytes_ ? ArchiveHeader::kFlagWavefront : 0) |
                            (dedup_.far_min ? ArchiveHeader::kFlagDedup : 0));
  h.block_log2 = static_cast<u8>(block_log2_);
  h.uncompressed_size = original.size();
  h.block_count = static_cast<u32>(blocks);
  h.prime_bytes = static_cast<u32>(prime_bytes_);
  h.model_param = param;

  const std::size_t section_bytes = dedup_.far_min ? 4 + section.size() : 0;
  std::size_t total = ArchiveHeader::kSize + section_bytes + blocks * 8 + 4;
  for (auto& p : payloads) total += p.size();
  std::vector<u8> out(total);
  h.serialize(out.data());
  if (section_bytes) {
    store32(out.data() + ArchiveHeader::kSize, static_cast<u32>(section.size()));
    std::memcpy(out.data() + ArchiveHeader::kSize + 4, section.data(), section.size());
  }
  u8* table = out.data() + ArchiveHeader::kSize + section_bytes;
  for (std::size_t i = 0; i < blocks; ++i) {
    if (payloads[i].size() > UINT32_MAX) throw Error("ultracodec: block payload too large");
    store32(table + i * 8, static_cast<u32>(payloads[i].size()));
    store32(table + i * 8 + 4, crcs[i]);
  }
  store32(table + blocks * 8, crc32c(table, blocks * 8));
  u8* dst = table + blocks * 8 + 4;
  for (auto& p : payloads) {
    if (!p.empty()) std::memcpy(dst, p.data(), p.size());
    dst += p.size();
    std::vector<u8>().swap(p);
  }
  if (stats) {
    stats->blocks = blocks;
    stats->stored_blocks = stored.load();
    stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return out;
}

std::vector<u8> BlockEngine::decompress(std::span<const u8> archive, EngineStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  const ArchiveHeader h = ArchiveHeader::parse(archive);
  if (h.tier != tier_) throw Error("ultracodec: archive belongs to another tier");
  if (h.uncompressed_size > options_.max_output_bytes) throw Error("ultracodec: archive exceeds max_output_bytes");
  validate_model_param(h.model_param);
  const std::size_t blocks = h.block_count;
  const std::size_t bs = h.block_size();

  Reader r(archive.subspan(ArchiveHeader::kSize));
  std::span<const u8> section;
  std::size_t data_size = static_cast<std::size_t>(h.uncompressed_size);
  if (h.flags & ArchiveHeader::kFlagDedup) {
    section = r.take(r.u32v());
    Reader peek(section);
    peek.u32v();
    auto rs = peek.take(8);
    const u64 residual = load64(rs.data());
    if (residual > h.uncompressed_size || h.block_count != (residual + bs - 1) / bs)
      throw Error("ultracodec: dedup residual does not match the block count");
    data_size = static_cast<std::size_t>(residual);
  }
  if (r.remaining() / 8 < blocks) throw Error("ultracodec: truncated block table");
  auto table = r.take(blocks * 8);
  if (r.u32v() != crc32c(table)) throw Error("ultracodec: block table CRC mismatch");
  std::vector<std::size_t> offset(blocks + 1);
  offset[0] = 0;
  for (std::size_t i = 0; i < blocks; ++i) {
    u32 sz = load32(table.data() + i * 8);
    if (sz == 0) throw Error("ultracodec: empty block payload");
    offset[i + 1] = offset[i] + sz;
  }
  if (offset[blocks] != r.remaining()) throw Error("ultracodec: payload sizes do not match the archive");
  const u8* base = r.pos();

  std::vector<u8> out(data_size);
  const unsigned workers = workers_for(blocks, h.model_param);
  prepare_codecs({h.model_param, bs, h.prime_bytes}, workers);

  std::unique_ptr<Progress[]> progress(new Progress[std::max<std::size_t>(blocks, 1)]);
  std::atomic<bool> aborted{false};
  std::atomic<std::size_t> stored{0};
  parallel_blocks(
      blocks, workers,
      [&](std::size_t i, unsigned w) {
        const std::size_t raw_size = std::min<std::size_t>(bs, out.size() - i * bs);
        std::span<u8> dst(out.data() + i * bs, raw_size);
        std::span<const u8> payload(base + offset[i], offset[i + 1] - offset[i]);
        std::span<const u8> prime;
        if (h.prime_bytes && i > 0) {
          progress[i - 1].wait();
          if (aborted.load(std::memory_order_acquire)) return;
          prime = std::span<const u8>(out.data() + (i - 1) * bs, h.prime_bytes);
        }
        const u8 method = payload[0];
        payload = payload.subspan(1);
        if (method == 0) {
          if (payload.size() != raw_size) throw Error("ultracodec: stored block has the wrong size");
          std::memcpy(dst.data(), payload.data(), raw_size);
          stored.fetch_add(1, std::memory_order_relaxed);
        } else if (method == 1) {
          codec(w).decode(prime, payload, dst, h.prime_bytes ? &progress[i] : nullptr,
                                         std::min<std::size_t>(h.prime_bytes, raw_size));
        } else {
          throw Error("ultracodec: unknown block method");
        }
        progress[i].signal();
        if (crc32c(dst.data(), dst.size()) != load32(table.data() + i * 8 + 4))
          throw Error("ultracodec: block CRC mismatch (corrupt archive)");
      },
      [&] {
        aborted.store(true, std::memory_order_release);
        for (std::size_t i = 0; i < blocks; ++i) progress[i].signal();
      });
  if (h.flags & ArchiveHeader::kFlagDedup) {
    std::vector<u8> original(static_cast<std::size_t>(h.uncompressed_size));
    Reader sr(section);
    dedup_decode(sr, out, original);
    if (sr.remaining()) throw Error("ultracodec: trailing dedup data");
    out.swap(original);
  }
  if (stats) {
    stats->blocks = blocks;
    stats->stored_blocks = stored.load();
    stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return out;
}

}  // namespace detail
}  // namespace ultracodec
