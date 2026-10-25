// neural_cm_codec.cpp - the neural-symbolic context-mixing codec (see the header).
//
// The models and their bit loops live in ncm_model.hpp (profile 1) and
// ncm_model2.hpp (profile 2), the block loops for each SIMD path in ncm_run.cpp; this file binds the model to its arena and holds
// the encoder-side policy.
//
// R12: STORED blocks reach the history (contexts are re-derived from it at
// every block start) but never the models. A block that would expand is
// stored, and the encoder rebuilds its models by replaying the frame's CM
// blocks; a pre-check stores obviously incompressible blocks without trying.
#include "codec/neural_cm_codec.hpp"

#include "codec/ncm_kernels.hpp"
#include "codec/ncm_model.hpp"
#include "codec/ncm_pipe.hpp"
#include "codec/ncm_model2.hpp"
#include "codec/ncm_run.hpp"
#include "encoder/parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>

namespace hce {
namespace {

using hc::u32;
using hc::u64;
using hc::u8;
using ncm::hash;
using ncm::log2_q16;

// ------------------------------------------------- encoder-side policy ---
// Not part of the format: the decoder never sees these decisions.
constexpr unsigned kMaxReplays = 3;     // per frame; afterwards the rest of the frame is STORED
constexpr std::size_t kMinCmBlock = 16;  // smaller blocks are STORED
constexpr u64 kStoreEntropyQ16 = 511181;  // 7.8 bits/byte, Q16
constexpr u64 kEntropyBiasQ16 = u64{184} << 16;  // small-sample bias of the empirical entropy

std::atomic<bool> g_force_cm{false};
std::atomic<const ncm::Kernels*> g_kernels{nullptr};

}  // namespace

// ------------------------------------------------------------- the codec ---
NeuralCMCodec::NeuralCMCodec() noexcept = default;

NeuralCMCodec::NeuralCMCodec(const hc::StreamParams& p) {
  if (accepts(p) != hc::Errc::ok) return;  // no model: every block is STORED
  const u64 n = memory_required(p);
  owned_ = hc::allocate_arena(static_cast<std::size_t>(n));
  if (!owned_) throw std::bad_alloc();
  (void)bind(p, hc::MutBytes(owned_.get(), static_cast<std::size_t>(n)));
}

NeuralCMCodec::NeuralCMCodec(NeuralCMCodec&& other) noexcept
    : m_(std::exchange(other.m_, nullptr)),
      m2_(std::exchange(other.m2_, nullptr)),
      owned_(std::move(other.owned_)),
      pipe_(std::move(other.pipe_)),
      pipeline_(other.pipeline_),
      frame_(other.frame_),
      cm_blocks_(std::move(other.cm_blocks_)),
      replays_(other.replays_),
      store_rest_(other.store_rest_),
      counters_(other.counters_) {}

NeuralCMCodec& NeuralCMCodec::operator=(NeuralCMCodec&& other) noexcept {
  if (this != &other) {
    m_ = std::exchange(other.m_, nullptr);
    m2_ = std::exchange(other.m2_, nullptr);
    owned_ = std::move(other.owned_);
    pipe_ = std::move(other.pipe_);
    pipeline_ = other.pipeline_;
    frame_ = other.frame_;
    cm_blocks_ = std::move(other.cm_blocks_);
    replays_ = other.replays_;
    store_rest_ = other.store_rest_;
    counters_ = other.counters_;
  }
  return *this;
}

NeuralCMCodec::~NeuralCMCodec() = default;  // the models are trivially destructible

static int profile_of(const hc::StreamParams& p) noexcept {
  if ((p.prior_id & 0xFFFFFF00u) != kNeuralCMFamily) return 0;
  const hc::u32 profile = p.prior_id & 0xFFu;
  return profile == kNeuralCMProfile || profile == kNeuralCMProfile2 ? static_cast<int>(profile) : 0;
}

hc::Errc NeuralCMCodec::accepts(const hc::StreamParams& p) noexcept {
  if (profile_of(p) == 0) return hc::Errc::unsupported_feature;
  if (p.level > kNeuralCMMaxLevel) return hc::Errc::unsupported_feature;
  return hc::Errc::ok;
}

// Depends on the profile and the level only: the decoder pipeline sizes
// workers before it has seen a stream header, with a prior_id of no profile,
// and so for either. Levels above the maximum are rejected by accepts().
hc::u64 NeuralCMCodec::memory_required(const hc::StreamParams& p) noexcept {
  const hc::u64 m1 = ncm::CMModel::layout(ncm::spec_for(p.level)).total;
  const hc::u64 m2 = ncm::p2::CMModel2::layout(ncm::p2::spec2_for(p.level)).total;
  const int profile = profile_of(p);
  return (profile == 1 ? m1 : profile == 2 ? m2 : std::max(m1, m2)) + 63;  // + slack to align the arena to 64 bytes
}

hc::Errc NeuralCMCodec::bind(const hc::StreamParams& p, hc::MutBytes arena) noexcept {
  m_ = nullptr;
  m2_ = nullptr;
  if (const hc::Errc e = accepts(p); e != hc::Errc::ok) return e;
  if (arena.size() < memory_required(p)) return hc::Errc::out_of_memory;
  const auto addr = reinterpret_cast<std::uintptr_t>(arena.data());
  std::byte* const base = arena.data() + ((64 - (addr & 63)) & 63);
  const ncm::Kernels* forced = g_kernels.load(std::memory_order_relaxed);
  const ncm::Kernels& kern = forced ? *forced : ncm::active_kernels();
  if (profile_of(p) == kNeuralCMProfile2) {
    ncm::p2::CMModel2* const m = new (base) ncm::p2::CMModel2{};
    m->carve(base, ncm::p2::spec2_for(p.level), p.window_size(), kern);
    m2_ = m;
    m2_->reset(true);
  } else {
    ncm::CMModel* const m = new (base) ncm::CMModel{};
    m->carve(base, ncm::spec_for(p.level), p.window_size(), kern);
    m_ = m;
    m_->reset(true);
  }
  return hc::Errc::ok;
}

// Lazy: the reset happens at the frame's first CM block, so frames without one
// (STORED data, damaged frames, headers alone) cost nothing.
void NeuralCMCodec::reset_frame() noexcept {
  if (m_) m_->stale = true;
  if (m2_) m2_->stale = true;
}

hc::Errc NeuralCMCodec::decode_block(hc::Bytes payload, hc::u32 usize, hc::Window& w) noexcept {
  if (m2_) {
    ncm::p2::CMModel2& m = *m2_;
    if (m.stale) m.reset(false);
    return ncm::runner2_for(m.k->path).decode(m, payload, usize, w);
  }
  if (!m_) return hc::Errc::payload_corrupt;
  ncm::CMModel& m = *m_;
  if (m.stale) m.reset(false);
  return ncm::runner_for(m.k->path).decode(m, payload, usize, w);
}

void NeuralCMCodec::reset_frame(hc::Bytes frame) noexcept {
  frame_ = frame;
  cm_blocks_.clear();
  replays_ = 0;
  store_rest_ = false;
  // Now: the pre-check reads the match table.
  if (m_) m_->reset(false);
  if (m2_) m2_->reset(false);
}

// Encoder-side guess, never seen by the decoder: is this block hopeless for
// CM? High order-0 entropy and almost no repeats the match model could use.
static bool looks_incompressible(hc::Bytes frame, std::size_t begin, std::size_t end, const u32* match_tab,
                                 int match_shift, u64 window) noexcept {
  const hc::Bytes b = frame.subspan(begin, end - begin);
  std::array<u32, 256> count{};
  for (const std::byte c : b) ++count[static_cast<u8>(c)];
  const u64 n = b.size();
  u64 sum = 0;
  for (const u32 c : count)
    if (c) sum += u64{c} * log2_q16(c);
  const u64 bits = n * log2_q16(n) - sum;
  if (bits + kEntropyBiasQ16 < n * kStoreEntropyQ16) return false;

  // Sampled every 64 bytes: does the 8-byte string ending here occur earlier in
  // this block (local table) or in earlier CM-coded data (the match table)?
  std::array<u32, 4096> local{};
  u64 samples = 0, repeats = 0;
  u64 v = 0;  // the last eight bytes, most recent lowest: the models' c8
  for (std::size_t i = begin; i < std::min(end, begin + 8); ++i) v = (v << 8) | static_cast<u8>(frame[i]);
  for (std::size_t i = begin + 8; i < end; v = (v << 8) | static_cast<u8>(frame[i]), ++i) {
    const u32 slot = static_cast<u32>((v * 0x9E3779B97F4A7C15ull) >> 52);
    if ((i & 63) == 0) {
      ++samples;
      bool hit = false;
      if (const u32 prev = local[slot]; prev && std::memcmp(frame.data() + prev - 8, frame.data() + i - 8, 8) == 0)
        hit = true;
      const u32 cand = match_tab[hash(v & 0xFFFFFFFFFFFF, 9) >> match_shift];  // as in byte_update
      const u64 d = i - cand;
      if (!hit && cand < i && d <= std::min<u64>(window, i) && cand >= 8 &&
          std::memcmp(frame.data() + cand - 8, frame.data() + i - 8, 8) == 0)
        hit = true;
      repeats += hit;
    }
    local[slot] = static_cast<u32>(i);
  }
  return repeats * 16 < samples;
}

std::size_t NeuralCMCodec::encode_block(std::size_t begin, std::size_t end, hc::MutBytes out) noexcept {
  const std::size_t n = end - begin;
  if ((!m_ && !m2_) || store_rest_ || n < kMinCmBlock || out.size() < n) {
    ++counters_.stored_blocks;
    return 0;
  }
  // Both profiles index their first match table alike.
  const u32* const match_tab = m_ ? m_->match_tab : m2_->match_tab;
  const int match_shift = m_ ? m_->match_shift : m2_->match_shift;
  const u64 window = m_ ? m_->window : m2_->window;
  if (!g_force_cm.load(std::memory_order_relaxed) &&
      looks_incompressible(frame_, begin, end, match_tab, match_shift, window)) {
    ++counters_.stored_blocks;
    return 0;
  }
  if (cm_blocks_.size() == cm_blocks_.capacity()) {
    try {
      cm_blocks_.reserve(std::max<std::size_t>(64, cm_blocks_.capacity() * 2));
    } catch (...) {  // cannot record it for a replay: code nothing more with CM
      store_rest_ = true;
      ++counters_.stored_blocks;
      return 0;
    }
  }

  // A CM payload must be strictly smaller than its content.
  std::size_t size;
  if (m_) {
    const ncm::Runner& run = ncm::runner_for(m_->k->path);
    if (pipeline_ && !pipe_) {
      pipe_ = ncm::Pipe::create();
      if (!pipe_) pipeline_ = false;  // no helper: one thread, the same bits
    }
    size = pipe_ && pipeline_ ? run.encode_piped(*m_, *pipe_, frame_, begin, end, out.first(n - 1))
                              : run.encode(*m_, frame_, begin, end, out.first(n - 1));
  } else {  // profile 2 is not pipelined yet
    size = ncm::runner2_for(m2_->k->path).encode(*m2_, frame_, begin, end, out.first(n - 1));
  }
  if (size == 0) {  // expands: store it, and rebuild the models without it (R12)
    replay();
    ++counters_.replays;
    if (++replays_ >= kMaxReplays) store_rest_ = true;
    ++counters_.stored_blocks;
    return 0;
  }
  cm_blocks_.emplace_back(begin, end);
  ++counters_.cm_blocks;
  return size;
}

// Rebuilds the models exactly as the decoder will hold them: reset, then the
// frame's CM blocks again, updating without coding.
void NeuralCMCodec::replay() noexcept {
  if (m_) {
    m_->reset(false);
    ncm::runner_for(m_->k->path).replay(*m_, frame_, cm_blocks_.data(), cm_blocks_.size());
  } else {
    m2_->reset(false);
    ncm::runner2_for(m2_->k->path).replay(*m2_, frame_, cm_blocks_.data(), cm_blocks_.size());
  }
}

const char* NeuralCMCodec::simd_path() noexcept {
  const ncm::Kernels* forced = g_kernels.load(std::memory_order_relaxed);
  return (forced ? *forced : ncm::active_kernels()).name;
}

void NeuralCMCodec::test_force_cm(bool on) noexcept { g_force_cm.store(on, std::memory_order_relaxed); }

bool NeuralCMCodec::test_select_simd(const char* name) noexcept {
  if (!name) {
    g_kernels.store(nullptr, std::memory_order_relaxed);
    return true;
  }
  for (const ncm::SimdPath p : ncm::kAllPaths) {
    if (std::strcmp(ncm::kernels_for(p).name, name) == 0 && ncm::simd_available(p)) {
      g_kernels.store(&ncm::kernels_for(p), std::memory_order_relaxed);
      return true;
    }
  }
  return false;
}

static_assert(hc::BlockCodec<NeuralCMCodec>);
static_assert(FrameEncoder<NeuralCMCodec>);

}  // namespace hce
