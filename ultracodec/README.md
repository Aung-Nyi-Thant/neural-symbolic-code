# ultracodec

A 4-tier lossless compression engine in C++20, built as a separate module of
this repository (it shares no code with the HCE codec). One interface,
four strategies:

| Tier | Name | Backend | Default block | Memory |
|---|---|---|---|---|
| 1 | Hyper-Speed | Byte-level LZ77 in the LZ4 block layout, 64 Ki-entry hash table, 64 KiB window | 1 MiB | ~256 KiB tables per thread |
| 2 | Consumer | Lazy hash-chain LZ77 with repeat offsets + 8-way interleaved static rANS (AVX2 / NEON decode) | 2 MiB | ~25 MiB per thread |
| 3 | Balanced | Long-range dedup pass, then 1 MiB wavefront-primed blocks through a lean 6-context CM + binary rANS | 1 MiB + 64 KiB prefix | 32 MiB table + ~29 MiB per thread, budget 512 MiB |
| 4 | Enterprise TR3 | Long-range dedup pass, then 4 MiB wavefront-primed blocks through a 14-input, two-layer CM + binary rANS | 4 MiB + 256 KiB prefix | 640 MiB table + ~180 MiB per thread, budget 3.5 GiB |

**Results in one line:** every tier round-trips, all archives are
deterministic and the same on x86 and ARM, and RAM stays within budget.
Tier 2 meets its speed target. Tier 1 decompression meets its target, but
Tier 1 compression does not. Tiers 3 and 4 miss their speed targets by a
wide margin, and on the mixed corpus they miss their density targets too.
The measured numbers and the reasons are below.

## API

```cpp
#include "ultracodec/engine.hpp"

auto engine = ultracodec::create_engine(ultracodec::TierLevel::Tier3_Balanced);
std::vector<std::uint8_t> packed = engine->compress(data);
std::vector<std::uint8_t> back = ultracodec::decompress(packed);  // picks the tier from the header
```

`core.hpp` defines `TierLevel`, `ArchiveHeader` (magic `ULTR`, version,
tier, uncompressed size, CRC-protected), `EngineOptions` (threads, block
size, memory budget, wavefront prefix, decompression size cap) and
`ICompressionEngine` (`compress`, `decompress`, `get_tier`, `get_name`).
`engine.hpp` has the factory `create_engine(TierLevel, EngineOptions)` and
the one-call helpers. Every failure throws `ultracodec::Error`.

The `ultra` tool compresses files: `ultra c -t 3 in out`, `ultra d in out`,
`ultra i archive`. `ultra_bench <tier> FILE...` measures round trips.

## Archive format

```
ArchiveHeader   32 bytes: "ULTR", version 1, tier, flags, block_log2,
                u64 size, u32 block count, u32 prefix length, u32 model
                parameter, CRC-32C of the first 28 bytes
[dedup section] (flag 2, Tiers 3/4) u32 size + removed-repeat list, CRC of the original
block table     per block {u32 payload size, u32 CRC-32C of the raw block}, then its own CRC
payloads        one per block: method byte (0 stored, 1 coded) + tier data
```

The decoder validates every length against the archive before it allocates
anything. It caps output at `max_output_bytes`, checks every block CRC, and
uses bounded copies in every tier. The corruption tests truncate archives
and flip random bits, under ASan/UBSan and TSan in CI. A corrupt archive
either throws `Error` or, rarely, decodes to the exact original (when the
flip lands in padding). None has crashed.

Archives do not depend on the thread count (the memory budget limits how
many workers run, never the model size). The x86 build (SSE2/AVX2) and the
AArch64 build (NEON) write byte-identical archives for all four tiers. That
was checked by decoding x86 archives under `qemu-aarch64` and comparing the
archives the AArch64 build writes.

## How the research items are implemented

| Item | Where | Implementation | Measured effect |
|---|---|---|---|
| Confident-prediction skipping (APM bypass) | `cm_model.hpp` | When the mixer logit is above a threshold (\|st\| >= 1900/1950 of 2047), the APM lookup and update are skipped. A second option, `kMatchSkip`, also bypasses the mixer and all counters inside long, reliable matches. | APM bypass: density unchanged (threshold 1000: +0.15% size); speed change within noise. Match skip: +1.3% size (len >= 32) or +3.9% (len >= 16), with no speed gain on code, so it is **off** by default. |
| Vectorized multi-stream rANS | `rans_static.cpp`, `rans_binary.hpp` | Tier 2: 8 interleaved 32-bit states. The AVX2 decoder does one gather plus a permute-based renormalization per 8 symbols, and NEON does two 4-lane halves. The stream is defined so the scalar and vector decoders read the same words (tests compare them). Tiers 3/4: two interleaved binary rANS states. The encoder uses exact reciprocal multiplication instead of division. | Tier 2 end-to-end decode: scalar 275 MB/s, AVX2 306-323 MB/s (+11-17%). For the CM tiers the coder is a few % of the time (as the Phase 8 study found), so rANS there gives a parallel-decodable stream, not speed. |
| Staggered / wavefront scheduling | `block_engine.cpp` | Block k is primed with the first P bytes of block k-1 (64 KiB in Tier 3, 256 KiB in Tier 4). Encoding is fully parallel. In decoding, the worker on block k waits only until block k-1 has published its first P bytes (an atomic flag), so the workers start a few % of a block apart. A serial long-range dedup pass (`dedup.cpp`) also removes repeats whose source is in another block. | 2 × 1 MiB blocks vs one 2 MiB block: +1.5% size. The 64 KiB prefix wins back 0.5%, a 256 KiB prefix 0.9%. Dedup: Tier 3 full corpus -0.9% (prose/docs -4.7%, XML faster). |
| Mixer distillation and LUTs | `cm_model.hpp` | stretch/squash are 4096-entry integer tables built at compile time from lpaq's 33-point curve, so there is no floating point and every platform is identical. The mixer is int16 SIMD (SSE2 `pmaddwd` / NEON `vmlal`): exact 32-bit dot product, multiply-high + saturating-add training. Inputs are built in registers. The adaptive rate 1/(n+1.5) comes from a table. | Building the inputs in registers instead of per-element stores fixed a store-forwarding stall: +38% on the order-1-only model. Moving the int16 SIMD mixer from 32-bit scalar cut ~106 to ~10 instructions per bit for training. |
| Memory-wall mitigation | `cm_model.hpp`, `common.cpp` | Counters sit in 64-byte buckets (one cache line holds a nibble's 15 nodes plus a tag), two-way associative on 128-byte pairs. One bit before each nibble boundary, both possible buckets of every context are prefetched (`__builtin_prefetch`), along with the match-table slot and the next bit's APM row and order-1 counter. Tables are huge-page backed (`MADV_HUGEPAGE`) and reset with `MADV_DONTNEED`, so RSS tracks the pages actually touched. Blocks run on parallel workers. | Prefetching changed speed by under 10% (within VM noise). With a 256 KiB table that fits in L2 the model is barely faster, so these tiers are bound by instructions and dependencies, not memory, on this machine. |

## Measured results

Setup: 4-vCPU Intel Xeon VM at 2.8 GHz (Firecracker), GCC 13 `-O3`,
SSE2 baseline with AVX2 chosen at run time. The corpus is 26.4 MB in five
files: 8 MiB Python stdlib, 6 MiB `/usr/share` XML/HTML/SVG, 4 MiB Perl
modules and docs, 4 MiB ELF binaries, and a 3.4 MB Gutenberg novel.
Baseline: `zstd -19` (python-zstandard 0.25), 5,297,778 bytes summed per
file, compressing at 1-3 MB/s.

Speeds are MB/s (10^6 bytes per second) over the five files, with each
file run on its own (the SUM rows of `ultra_bench`). Timing on this VM
varies by about ±20% from run to run.

| Tier | Size | vs zstd -19 | 1 thread comp / dec | 4 threads comp / dec | Peak RSS (bench, 4 thr.) |
|---|---|---|---|---|---|
| 1 | 10,220,112 | +93% | 240 / 815 | 752 / 1,389 | 152 MiB |
| 2 | 6,162,287 | +16.3% | 13-19 / 275-323 | 33 / 620 | 177 MiB |
| 3 | 4,748,159 | **-10.4%** | 1.5 / 1.6 | 4.5 / 4.9 | 454 MiB |
| 4 | 4,153,233 | **-21.6%** | ~0.12 / ~0.13 | 0.32 / 0.38 (0.66 / 0.78 on the 26 MB concatenation) | 3.6 GiB |

Per-file reduction vs zstd -19 (Tier 3 / Tier 4): Python -10.7% / -21.8%,
XML -4.7% / -16.0%, Perl+docs -13.2% / -23.5%, binaries -5.8% / -18.8%,
novel -17.4% / -26.7%. On four 2 MB text/code samples alone (Python,
novel, XML, docs), Tier 3 is -17.0% and Tier 4 -25.0%.

For reference, LZ4 (python-lz4, 1 MiB blocks, one thread) on the Python
file: 3,026,137 bytes at 363 / 1003 MB/s. Tier 1 is 2,891,213 bytes at
282 / 888 MB/s.

### Against the targets

| Tier | Target | Measured | Verdict |
|---|---|---|---|
| 1 | 1.0-1.8 GB/s | Decompress 1.39 GB/s on 4 threads (0.82 on 1). Compress 0.75-0.82 GB/s on 4 threads (0.24 on 1). | Decompress met on 4 threads; compress missed. One core of this VM runs LZ4 itself at 0.26-0.61 GB/s, so a GB/s per core needs a faster machine. |
| 2 | 15-20+ MB/s, <= 256 MiB | 33 MB/s comp on 4 threads, 13-19 on 1; decode 275-620 MB/s; 177 MiB | Met on 2+ threads; borderline on 1 thread. |
| 3 | ~11 MB/s, 17.2% smaller than zstd -19, ~500 MiB | 4.5-4.8 MB/s on 4 threads; -10.4% on the mixed corpus, -17.0% on text/code; 454 MiB | Speed missed (~2.4x). Density met only on text/code. RAM met. |
| 4 | >= 10 MB/s, 30-33% smaller, 3-4 GB | 0.3-0.7 MB/s on 4 threads; -21.6%; 3.6 GiB | Speed missed by ~15-30x, density missed by ~9 points. RAM met. |

### Why Tiers 3 and 4 miss

* **Per-core speed.** A context-mixing model codes eight binary decisions
  per byte. Each one reads, mixes and updates every context, so the work
  per byte grows with the number of contexts. The lean model runs ~3,500
  instructions per byte at an IPC of ~1.3, which is ~1.5 MB/s per core
  here. Shrinking its table to one that fits in L2 barely helps, so the
  cost is compute, not memory. Each extra hashed context costs about 20-25%
  speed. Tier 3 with three contexts runs 1.8 MB/s per core at -16.5%; with
  five, 1.1 MB/s at -17.7%. The default four-context build sits in between.
* **Parallelism is the only lever left, and it is linear.** Blocks are
  independent apart from the wavefront prefix, so throughput scales with
  cores until memory bandwidth runs out. 11 MB/s for Tier 3 needs about 8
  of these cores. 10 MB/s for Tier 4 needs about 80, and its 3-4 GiB
  budget allows only 4 workers with 640 MiB tables.
* **Density.** Splitting into 1-4 MiB blocks costs ~1.5% per halving of
  the block size, and binaries and markup compress less than prose. The
  30-33% figure is what a full unblocked model reaches on text-heavy data;
  the earlier v4 study measured 29-31% at ~0.6 MB/s. Tier 4 reaches -25%
  on the text/code samples and -21.6% with binaries mixed in.
* **The table budget buys little.** Tier 4 with 64 MiB, 128 MiB and 640 MiB
  tables per worker: +1.4%, +0.6% and baseline size. Most of the 3-4 GiB
  goes to tables that save under 1.5%.

What would close the gap is a different design, not tuning:

* An LZ-plus-CM-literal hybrid (the HCE-LX direction) for 10+ MB/s.
* SIMD across contexts. A structure-of-arrays nibble cache would let one
  vector update all counters per bit. That needs SSSE3/AVX2, not the SSE2
  baseline used here.
* Many more cores for the CM tiers.

## Tuning notes (all measured on the 4 × 2 MB samples)

* Tier 1 hash table: 64 Ki entries is 1.5% smaller than 16 Ki entries on
  the 26 MB corpus at the same speed (the results table above used 16 Ki).

* Counter limit (adaptation floor 1/(n+1.5)): 16 is better than 30 (-0.2%)
  and 127 (-0.8%).
* Tier 3 table per worker: 4 MiB +2.3%, 16 MiB +0.6%, 32 MiB (default)
  about +0.3%, 128 MiB baseline.
* Confidence maps (an adaptive re-estimate of each context's prediction
  from its hit count): Tier 4 -0.7% (on), Tier 3 -0.03% (off, it costs
  speed).
* A two-layer mixer in Tier 3 gives -0.4%, but it is not used, for speed.
  A second, order-2 APM gives -0.08%.
* Tier 2 block size: 1 MiB is +2.5% at 87 MB/s, 2 MiB (default) +0.9% at
  46 MB/s, 4 MiB is the baseline at 29 MB/s (4 threads, concatenated
  corpus). Chain depth 48 instead of 16 gives -1.6% at 2.5x less speed.

## Build and test

Built with the repository (`HCE_BUILD_ULTRACODEC`, on by default), or on
its own with `cmake -S ultracodec -B build-ultra`. The CTest entry
`ultracodec` runs `tests/test_ultracodec.cpp`. It covers header
validation, the CRC-32C known answer, rANS scalar/AVX2/NEON agreement,
round trips of every tier over edge cases (empty input, 1 byte, block
size ± 1, random, zeros, structured, text), byte-identical archives for
1/2/3/4/7 threads, engine rules, corruption fuzzing, and peak RSS per tier
in a forked child (Linux, not under sanitizers). It passes with GCC and
Clang at `-Werror` and under ASan+UBSan and TSan. The AArch64 and MinGW
builds compile cleanly at `-Werror`.

`HCE_FUZZ_ITERS` / `ULTRA_FUZZ_ITERS` scale the corruption rounds
(sanitizer and emulator runs use 24). `ULTRA_SKIP_RSS=1` skips the RSS
checks.

## Layout

```text
include/ultracodec/core.hpp     TierLevel, ArchiveHeader, EngineOptions, ICompressionEngine, Error
include/ultracodec/engine.hpp   create_engine factory, compress/decompress/peek_header helpers
src/engine.cpp                  Factory
src/block_engine.{hpp,cpp}      Block container: framing, validation, worker pool, wavefront waits
src/common.{hpp,cpp}            LE access, CRC-32C (SSE4.2 or slice-by-8), vector copy, huge-page arrays
src/lz_util.hpp                 Match length, wild and overlapping copies
src/tier1_hyperspeed.cpp        Tier 1 LZ77
src/tier2_consumer.cpp          Tier 2 lazy LZ77 + rANS streams
src/rans_static.{hpp,cpp}       8-way static rANS, scalar / AVX2 / NEON decoders
src/rans_binary.hpp             Two-state binary rANS for the CM tiers
src/cm_model.hpp                CM predictor: LUTs, counters, buckets, mixer kernels, APM, match model
src/cm_codec.hpp                CM block codec and engine (table sizing, worker cap)
src/dedup.{hpp,cpp}             Long-range dedup pre-pass
src/tier3_balanced.cpp          Tier 3 profile (contexts and knobs)
src/tier4_enterprise.cpp        Tier 4 profile
tools/ultra_cli.cpp             ultra: compress/decompress/inspect files
tools/ultra_bench.cpp           ultra_bench: round-trip benchmark
tests/test_ultracodec.cpp       Test suite
```
