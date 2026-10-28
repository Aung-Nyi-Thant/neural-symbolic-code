// Per-tier engine constructors (one translation unit per tier).
#pragma once

#include <memory>

#include "ultracodec/core.hpp"

namespace ultracodec::detail {

std::unique_ptr<ICompressionEngine> make_tier1(const EngineOptions& options);
std::unique_ptr<ICompressionEngine> make_tier2(const EngineOptions& options);
std::unique_ptr<ICompressionEngine> make_tier3(const EngineOptions& options);
std::unique_ptr<ICompressionEngine> make_tier4(const EngineOptions& options);

}  // namespace ultracodec::detail
