// ultracodec engine factory and one-call helpers.
#pragma once

#include <memory>

#include "ultracodec/core.hpp"

namespace ultracodec {

// Creates the engine for a tier. Throws Error on an unknown tier or invalid
// options (block size not a power of two in range, ...).
std::unique_ptr<ICompressionEngine> create_engine(TierLevel tier, const EngineOptions& options = {});

// Reads the header of an archive without decoding it.
ArchiveHeader peek_header(std::span<const std::uint8_t> archive);

// Compresses with a fresh engine of the given tier.
std::vector<std::uint8_t> compress(TierLevel tier, std::span<const std::uint8_t> input,
                                   const EngineOptions& options = {});

// Decompresses any ultracodec archive, picking the tier from its header.
std::vector<std::uint8_t> decompress(std::span<const std::uint8_t> archive,
                                     const EngineOptions& options = {});

}  // namespace ultracodec
