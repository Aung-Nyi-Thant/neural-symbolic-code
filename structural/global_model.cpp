// global_model.cpp - pass 1 statistics and the frozen global model (see the header).
#include "structural/global_model.hpp"

#include <algorithm>

namespace hce::structural {

using hc::u32;
using hc::u64;

// A successor is kept when its count held at least this: one-off pairs cost
// more in the model section than they save in the token stream.
constexpr u32 kMinSuccessorCount = 2;

// ------------------------------------------------------------ GlobalModel ---

u32 GlobalModel::find(TokenType type, std::string_view value) const noexcept {
  const auto it = index_.find(Key{value, static_cast<std::uint8_t>(type)});
  return it == index_.end() ? kNoSymbol : it->second;
}

void GlobalModel::derive_traits() {
  const u32 n = size();
  traits_.assign(n, 0);
  indent_.assign(n, 0);
  for (u32 id = 0; id < n; ++id) {
    const std::string_view v = value(id);
    std::uint8_t t = 0;
    if (type(id) == TokenType::STRUCTURE) {
      if (v.find('\n') != std::string_view::npos) t = kTraitNewline;
      else if (v == "{") t = kTraitOpenBrace;
      else if (v == "}") t = kTraitCloseBrace;
      else if (v == "(" || v == "[") t = kTraitOpenBracket;
      else if (v == ")" || v == "]") t = kTraitCloseBracket;
    } else if (type(id) == TokenType::INDENTATION) {
      t = kTraitIndent;
      indent_[id] = static_cast<std::uint8_t>(std::min<std::size_t>(v.size(), 255));
    }
    traits_[id] = t;
  }
}

std::vector<std::uint8_t> GlobalModel::serialize() const {
  std::vector<std::uint8_t> out;
  out.reserve(blob_.size() + 4 * std::size_t{size()} + 16);
  for (const std::byte b : kGlobalModelMagic) out.push_back(static_cast<std::uint8_t>(b));
  put_varint(out, size());
  for (u32 id = 0; id < size(); ++id) {
    const std::string_view v = value(id);
    out.push_back(types_[id]);
    put_varint(out, v.size());
    out.insert(out.end(), v.begin(), v.end());
  }
  for (const u32 s : succ_) put_varint(out, s == kNoSymbol ? 0 : u64{s} + 1);
  return out;
}

hc::Errc GlobalModel::parse(hc::Bytes in, const GlobalModelLimits& lim, GlobalModel& out) noexcept {
  try {
    if (in.size() < kGlobalModelMagic.size() ||
        !std::equal(kGlobalModelMagic.begin(), kGlobalModelMagic.end(), in.begin()))
      return hc::Errc::bad_magic;
    std::size_t at = kGlobalModelMagic.size();
    u64 n = 0;
    if (!get_varint(in, at, n)) return hc::Errc::header_corrupt;
    if (n > lim.max_symbols) return hc::Errc::limit_output;
    if (n > (in.size() - at) / 3) return hc::Errc::header_corrupt;  // type, length and successor: >= 3 bytes
    GlobalModel m;
    const auto count = static_cast<std::size_t>(n);
    m.types_.reserve(count);
    m.offsets_.reserve(count + 1);
    m.succ_.reserve(count);
    m.offsets_.push_back(0);
    for (std::size_t k = 0; k < count; ++k) {
      if (at >= in.size()) return hc::Errc::header_corrupt;
      const auto type = static_cast<std::uint8_t>(in[at++]);
      u64 len = 0;
      if (type >= kTokenTypes || !get_varint(in, at, len)) return hc::Errc::header_corrupt;
      if (len > lim.max_token_bytes) return hc::Errc::limit_output;
      if (len > in.size() - at) return hc::Errc::header_corrupt;
      if (m.blob_.size() + len > lim.max_dictionary_bytes) return hc::Errc::limit_output;
      const auto* p = reinterpret_cast<const char*>(in.data() + at);
      m.blob_.insert(m.blob_.end(), p, p + len);
      at += static_cast<std::size_t>(len);
      m.types_.push_back(type);
      m.offsets_.push_back(m.blob_.size());
    }
    for (std::size_t k = 0; k < count; ++k) {
      u64 s = 0;
      if (!get_varint(in, at, s) || s > n) return hc::Errc::header_corrupt;
      m.succ_.push_back(s == 0 ? kNoSymbol : static_cast<u32>(s - 1));
    }
    if (at != in.size()) return hc::Errc::header_corrupt;
    m.derive_traits();
    out = std::move(m);
    return hc::Errc::ok;
  } catch (const std::bad_alloc&) {
    return hc::Errc::out_of_memory;
  }
}

// ----------------------------------------------------- GlobalModelBuilder ---

hc::Errc GlobalModelBuilder::add(const std::vector<StructuralToken>& tokens) noexcept {
  try {
    for (const StructuralToken& t : tokens) {
      if (t.value.size() > lim_.max_token_bytes) return hc::Errc::limit_output;
      key_.assign(1, static_cast<char>(t.type));
      key_.append(t.value);
      auto it = ids_.find(key_);
      if (it == ids_.end()) {
        if (values_.size() >= lim_.max_symbols || bytes_ + t.value.size() > lim_.max_dictionary_bytes)
          return hc::Errc::limit_output;
        it = ids_.emplace(key_, static_cast<u32>(values_.size())).first;
        values_.push_back(&it->first);
        counts_.push_back(0);
        cand0_.push_back(kNoSymbol);
        cand1_.push_back(kNoSymbol);
        n0_.push_back(0);
        n1_.push_back(0);
        bytes_ += t.value.size();
      }
      const u32 id = it->second;
      ++counts_[id];
      if (prev_ != kNoSymbol) {  // Misra-Gries, two slots, over prev_'s successors
        const u32 p = prev_;
        if (cand0_[p] == id) {
          ++n0_[p];
        } else if (cand1_[p] == id) {
          ++n1_[p];
        } else if (n0_[p] == 0) {
          cand0_[p] = id;
          n0_[p] = 1;
        } else if (n1_[p] == 0) {
          cand1_[p] = id;
          n1_[p] = 1;
        } else {
          --n0_[p];
          --n1_[p];
        }
      }
      prev_ = id;
      ++tokens_;
    }
    return hc::Errc::ok;
  } catch (const std::bad_alloc&) {
    return hc::Errc::out_of_memory;
  }
}

hc::Errc GlobalModelBuilder::freeze(GlobalModel& out) noexcept {
  try {
    const std::size_t n = values_.size();
    GlobalModel m;
    m.blob_.reserve(static_cast<std::size_t>(bytes_));
    m.offsets_.reserve(n + 1);
    m.types_.reserve(n);
    m.succ_.reserve(n);
    m.offsets_.push_back(0);
    for (std::size_t id = 0; id < n; ++id) {  // provisional IDs are already in order of first appearance
      const std::string& key = *values_[id];  // type byte, then the value
      m.types_.push_back(static_cast<std::uint8_t>(key[0]));
      m.blob_.insert(m.blob_.end(), key.begin() + 1, key.end());
      m.offsets_.push_back(m.blob_.size());
      const bool second = n1_[id] > n0_[id];  // ties: the earlier slot
      const u32 best = second ? cand1_[id] : cand0_[id];
      m.succ_.push_back((second ? n1_[id] : n0_[id]) >= kMinSuccessorCount ? best : kNoSymbol);
    }
    m.derive_traits();
    m.index_.reserve(n);
    for (u32 id = 0; id < m.size(); ++id) m.index_.emplace(GlobalModel::Key{m.value(id), m.types_[id]}, id);
    out = std::move(m);  // the index's views follow the blob's heap buffer
    return hc::Errc::ok;
  } catch (const std::bad_alloc&) {
    return hc::Errc::out_of_memory;
  }
}

}  // namespace hce::structural
