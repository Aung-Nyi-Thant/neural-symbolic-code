// Tier 4: Enterprise SWS (snapshot warm-start context mixing).
//
// Splitting a file into independently modelled blocks is what lets a CM
// codec use several cores, but every cold block forgets what the model
// learned: with 1 MiB blocks the 22-context model loses ~7 points of
// density against coding the whole file. Tier 4 keeps most of that:
//
//   1. One model reads the built-in primer (sws_primer.inc), then codes
//      block 0 serially. Its state afterwards is the snapshot S.
//   2. Every later block is coded by a copy of S, so it starts with tables,
//      mixer weights and APMs trained on the primer and block 0, and with
//      block 0 in its history for the match model. These blocks are
//      independent of each other and run on all workers, in encode and in
//      decode alike.
//
// Payloads use the block container layout (core.hpp); the header has
// kFlagSnapshot, and model_param holds the primer's CRC-32C (0 = no primer).
// A block the model cannot shrink is stored; S is the same either way,
// because the model processes block 0's bytes whether or not they are coded.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>

#include "block_engine.hpp"
#include "sws_model.hpp"
#include "tiers.hpp"

namespace ultracodec::detail {
namespace {

constexpr unsigned kDefaultBlockLog2 = 21;                    // 2 MiB blocks
constexpr std::size_t kDefaultBudget = std::size_t{4} << 30;  // all models of one call

// History one model needs: primer, block 0 and the longest later block. It
// also sizes the model's tables, so small inputs get small models.
std::size_t capacity(std::size_t size, std::size_t block, bool primed) noexcept {
  return (primed ? sws::primer().size() : 0) + std::min(size, block) + (size > block ? std::min(size - block, block) : 0);
}

// A model that has read the primer (when enabled), ready for block 0.
sws::Model start_model(std::size_t capacity, bool primed) {
  sws::Model m(capacity);
  if (primed) m.train(sws::primer());
  return m;
}

// Codes `raw` with `model` into `out` (method byte + payload), storing it
// when coding does not help. Returns true when stored.
bool code_block(sws::Model& model, std::span<const u8> raw, std::vector<u8>& out) {
  out.assign(1, 1);
  out.reserve(raw.size() + 1);
  if (model.encode(raw, out)) return false;
  out.assign(1, 0);
  out.insert(out.end(), raw.begin(), raw.end());
  return true;
}

// Makes `slot` a copy of `snapshot`, reusing its memory when it already holds a model.
sws::Model& refresh(std::optional<sws::Model>& slot, const sws::Model& snapshot) {
  if (slot) *slot = snapshot; else slot.emplace(snapshot);
  return *slot;
}

class SwsEngine final : public ICompressionEngine {
 public:
  explicit SwsEngine(const EngineOptions& o) : options_(o) {
    if (o.block_size) {
      const std::size_t b = o.block_size;
      if ((b & (b - 1)) != 0 || b < (std::size_t{1} << 16) || b > (std::size_t{1} << 26))
        throw Error("ultracodec: block_size must be a power of two in [64 KiB, 64 MiB]");
      block_log2_ = highbit32(static_cast<u32>(b));
    }
    if (options_.threads == 0) options_.threads = std::max(1u, std::thread::hardware_concurrency());
    if (options_.memory_budget == 0) options_.memory_budget = kDefaultBudget;
  }

  std::vector<u8> compress(std::span<const u8> input, EngineStats* stats) override;
  std::vector<u8> decompress(std::span<const u8> archive, EngineStats* stats) override;
  TierLevel get_tier() const noexcept override { return TierLevel::Tier4_Enterprise; }
  std::string_view get_name() const noexcept override { return "Tier4 Enterprise SWS (snapshot warm-start CM)"; }

 private:
  // Workers for the parallel stage, so that the snapshot plus one model per
  // worker fit the memory budget. The archive does not depend on this.
  unsigned workers_for(std::size_t later_blocks, std::size_t capacity) const noexcept {
    const std::size_t models = options_.memory_budget / sws::Model::bytes(capacity);
    const std::size_t w = std::min<std::size_t>({options_.threads, later_blocks, models > 1 ? models - 1 : 1});
    return static_cast<unsigned>(std::max<std::size_t>(1, w));
  }

  EngineOptions options_;
  unsigned block_log2_ = kDefaultBlockLog2;
};

std::vector<u8> SwsEngine::compress(std::span<const u8> input, EngineStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t bs = std::size_t{1} << block_log2_;
  const std::size_t blocks = (input.size() + bs - 1) / bs;
  if (blocks > UINT32_MAX) throw Error("ultracodec: input too large");
  const bool primed = options_.tier4_primer;
  const auto block = [&](std::size_t i) { return input.subspan(i * bs, std::min(bs, input.size() - i * bs)); };

  std::vector<std::vector<u8>> payloads(blocks);
  std::vector<u32> crcs(blocks);
  std::atomic<std::size_t> stored{0};
  if (blocks) {
    const std::size_t cap = capacity(input.size(), bs, primed);
    sws::Model snapshot = start_model(cap, primed);
    crcs[0] = crc32c(block(0));
    if (code_block(snapshot, block(0), payloads[0])) stored.fetch_add(1);
    if (blocks > 1) {
      const unsigned workers = workers_for(blocks - 1, cap);
      std::vector<std::optional<sws::Model>> models(workers);
      parallel_blocks(
          blocks - 1, workers,
          [&](std::size_t k, unsigned w) {
            const std::size_t i = k + 1;
            crcs[i] = crc32c(block(i));
            if (code_block(refresh(models[w], snapshot), block(i), payloads[i]))
              stored.fetch_add(1, std::memory_order_relaxed);
          },
          {});
    }
  }

  ArchiveHeader h;
  h.tier = TierLevel::Tier4_Enterprise;
  h.flags = ArchiveHeader::kFlagSnapshot;
  h.block_log2 = static_cast<u8>(block_log2_);
  h.uncompressed_size = input.size();
  h.block_count = static_cast<u32>(blocks);
  h.prime_bytes = 0;
  h.model_param = primed ? sws::primer_crc() : 0;

  std::size_t total = ArchiveHeader::kSize + blocks * 8 + 4;
  for (auto& p : payloads) total += p.size();
  std::vector<u8> out(total);
  h.serialize(out.data());
  u8* table = out.data() + ArchiveHeader::kSize;
  for (std::size_t i = 0; i < blocks; ++i) {
    if (payloads[i].size() > UINT32_MAX) throw Error("ultracodec: block payload too large");
    store32(table + i * 8, static_cast<u32>(payloads[i].size()));
    store32(table + i * 8 + 4, crcs[i]);
  }
  store32(table + blocks * 8, crc32c(table, blocks * 8));
  u8* dst = table + blocks * 8 + 4;
  for (auto& p : payloads) {
    std::memcpy(dst, p.data(), p.size());
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

std::vector<u8> SwsEngine::decompress(std::span<const u8> archive, EngineStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  const ArchiveHeader h = ArchiveHeader::parse(archive);
  if (h.tier != TierLevel::Tier4_Enterprise) throw Error("ultracodec: archive belongs to another tier");
  if (h.uncompressed_size > options_.max_output_bytes) throw Error("ultracodec: archive exceeds max_output_bytes");
  if (h.model_param != 0 && h.model_param != sws::primer_crc()) throw Error("ultracodec: unknown Tier 4 primer");
  const bool primed = h.model_param != 0;
  const std::size_t blocks = h.block_count;
  const std::size_t bs = h.block_size();

  Reader r(archive.subspan(ArchiveHeader::kSize));
  if (r.remaining() / 8 < blocks) throw Error("ultracodec: truncated block table");
  auto table = r.take(blocks * 8);
  if (r.u32v() != crc32c(table)) throw Error("ultracodec: block table CRC mismatch");
  std::vector<std::size_t> offset(blocks + 1);
  for (std::size_t i = 0; i < blocks; ++i) {
    const u32 sz = load32(table.data() + i * 8);
    if (sz == 0) throw Error("ultracodec: empty block payload");
    offset[i + 1] = offset[i] + sz;
  }
  if (offset[blocks] != r.remaining()) throw Error("ultracodec: payload sizes do not match the archive");
  const u8* base = r.pos();

  std::vector<u8> out(static_cast<std::size_t>(h.uncompressed_size));
  std::atomic<std::size_t> stored{0};
  // Decodes block i into out. `model` may be null only for a stored block after block 0.
  const auto decode_block = [&](std::size_t i, sws::Model* model) {
    std::span<u8> dst(out.data() + i * bs, std::min(bs, out.size() - i * bs));
    std::span<const u8> payload(base + offset[i], offset[i + 1] - offset[i]);
    const u8 method = payload[0];
    payload = payload.subspan(1);
    if (method == 0) {
      if (payload.size() != dst.size()) throw Error("ultracodec: stored block has the wrong size");
      std::memcpy(dst.data(), payload.data(), dst.size());
      if (i == 0) model->train(dst);  // the snapshot still learns block 0
      stored.fetch_add(1, std::memory_order_relaxed);
    } else if (method == 1) {
      model->decode(payload, dst);
    } else {
      throw Error("ultracodec: unknown block method");
    }
    if (crc32c(dst.data(), dst.size()) != load32(table.data() + i * 8 + 4))
      throw Error("ultracodec: block CRC mismatch (corrupt archive)");
  };

  if (blocks) {
    const std::size_t cap = capacity(out.size(), bs, primed);
    sws::Model snapshot = start_model(cap, primed);
    decode_block(0, &snapshot);
    if (blocks > 1) {
      const unsigned workers = workers_for(blocks - 1, cap);
      std::vector<std::optional<sws::Model>> models(workers);
      parallel_blocks(
          blocks - 1, workers,
          [&](std::size_t k, unsigned w) {
            const std::size_t i = k + 1;
            const bool coded = base[offset[i]] == 1;  // stored blocks need no model
            decode_block(i, coded ? &refresh(models[w], snapshot) : nullptr);
          },
          {});
    }
  }
  if (stats) {
    stats->blocks = blocks;
    stats->stored_blocks = stored.load();
    stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return out;
}

}  // namespace

std::unique_ptr<ICompressionEngine> make_tier4(const EngineOptions& o) { return std::make_unique<SwsEngine>(o); }

}  // namespace ultracodec::detail
