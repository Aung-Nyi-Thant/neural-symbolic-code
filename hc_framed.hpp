// hc_framed.hpp - Container v3 FRAMED layout: reference patterns for R18-R21.
//
// Covers framing (R18), resource limits (R19), resynchronization and partial
// recovery (R20) and the bounded history window (R21). The tiered CM codec plugs
// in through the BlockCodec concept. Integer-only, no exceptions; the only heap
// allocation is in bind_stream(), at most once per stream, and none at all when
// the arena from a previous stream is already large enough.
//
// The library is split by layer under hc/; this header includes all of it.
#pragma once

#include "hc/base.hpp"
#include "hc/crc32c.hpp"
#include "hc/format.hpp"
#include "hc/window.hpp"
#include "hc/arena.hpp"
#include "hc/interfaces.hpp"
#include "hc/stream_decoder.hpp"
#include "hc/framed_writer.hpp"
