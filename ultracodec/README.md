# ultracodec

A 4-tier lossless compression engine in C++20, built as a separate module of
this repository (it shares no code with the HCE codec). One interface,
four strategies:

| Tier | Name | Backend | Default block | Memory |
|---|---|---|---|---|
| 1 | Hyper-Speed | Byte-level LZ77 in the LZ4 block layout, 64 Ki-entry hash table, 64 KiB window | 1 MiB | ~256 KiB tables per thread |
| 2 | Consumer | Gated word-dictionary pre-filter, lazy hash-chain LZ77 with repeat offsets, 8-way interleaved static rANS (AVX2 / NEON decode) with literals split into 16 context streams | 2 MiB | ~30 MiB per thread |
| 3 | Balanced | Long-range dedup pass, then 1 MiB wavefront-primed blocks through a lean 6-context CM + binary rANS | 1 MiB + 64 KiB prefix | 32 MiB table + ~29 MiB per thread, budget 512 MiB |
| 4 | Enterprise SWS | 256 KiB built-in primer and block 0 train one model; every later 2 MiB block starts from a copy of it (snapshot warm-start). 22-context CM, 4 mixer sets, 4 APMs, binary range coder | 2 MiB | ~200 MiB per model: snapshot + one per worker (~1.1 GiB at 4 threads), budget 4 GiB |

**Results in one line:** every tier round-trips, all archives are
deterministic and the same on x86 and ARM, and RAM stays within budget.
Tier 2 meets its speed target. Tier 1 decompression meets its target, but
Tier 1 compression does not. Tier 3 misses its speed target by a wide
margin, and on the mixed corpus its density target too. Tier 4 was rebuilt
around snapshot warm-start: -28.8% vs `zstd -19` on a text/code corpus at
1.7 MB/s on 4 cores of an Apple M4 (see "Tier 4" below). The measured
numbers and the reasons are below.

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
[dedup section] (flag 2, Tier 3) u32 size + removed-repeat list, CRC of the original
block table     per block {u32 payload size, u32 CRC-32C of the raw block}, then its own CRC
payloads        one per block: method byte (0 stored, 1 coded) + tier data
```

Tier 4 sets flag 4 (snapshot warm-start) and stores the CRC-32C of its
primer in the model parameter (0 = no primer). Its blocks are coded in the
same layout, but every block after block 0 is decoded by a copy of the
model that has read the primer and block 0.

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
| 2 | 6,162,287 | +16.3% | 13-19 / 275-323 | 33 / 620 | 177 MiB (before the word filter; see below) |
| 3 | 4,748,159 | **-10.4%** | 1.5 / 1.6 | 4.5 / 4.9 | 454 MiB |
| 4 (old TR3 design, replaced) | 4,153,233 | -21.6% | ~0.12 / ~0.13 | 0.32 / 0.38 (0.66 / 0.78 on the 26 MB concatenation) | 3.6 GiB |

The new Tier 4 was measured on a different corpus; see "Tier 4" below.
Per-file reduction vs zstd -19 for the old design (Tier 3 / Tier 4): Python -10.7% / -21.8%,
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
| 4 (old TR3) | >= 10 MB/s, 30-33% smaller, 3-4 GB | 0.3-0.7 MB/s on 4 threads; -21.6%; 3.6 GiB | Speed missed by ~15-30x, density missed by ~9 points. Replaced by SWS. |
| 4 (SWS) | about -29% at about 2 MB/s | -28.8% per file (-26.8% as one mixed archive); 1.67 MB/s on 4 M4 cores; ~1.1 GiB | Density within 0.2 points on separate files; speed ~15% short on 4 cores. |

### Tier 4: snapshot warm-start (SWS)

A CM codec uses several cores by splitting the input into blocks with
their own models, and every cold block forgets what the model learned:
with 1 MiB blocks the 22-context model of `sws_model.cpp` (a port of the
HCE "Ultra v4" lean model) loses ~7 points against coding a whole file.
Tier 4 trains one model on a built-in 256 KiB primer and then on block 0,
coding block 0 as it goes. Every later block is coded by a copy of that
snapshot, so it starts with trained tables, mixer weights and APMs, and
with block 0 in its match history. Those blocks are independent, so encode
and decode both run them on all workers; only block 0 is serial.

Measured 2026-10-27 on a 22.8 MB text/code corpus: 8 MiB Python 3.12
stdlib (`*.py`, no tests), 4 MiB Perl modules, 4 MiB `/usr/share`
XML/HTML/SVG and 5.7 MB of five Gutenberg novels. `zstd -19` gives
4,191,049 bytes summed per file.

| Input | Tier 4 bytes | vs zstd -19 |
|---|---|---|
| Four files, each compressed on its own | 2,985,777 | **-28.8%** |
| Same, without the primer | 3,010,537 | -28.2% |
| One 22.8 MB concatenation (one archive) | 3,066,541 | -26.8% |
| Same, without the primer | 3,087,616 | -26.3% |
| 4 MiB of ELF binaries from `/usr/bin` (zstd -19: 1,563,496) | 1,236,203 | -20.9% |

Per file: Python -30.8%, Perl -29.5%, novels -26.6%, XML -29.3%. The
primer helps mostly the Python file (-1.9%): it holds CPython 3.11 library
code, which is close to this corpus even though the primer excludes the
modules the benchmark corpus was built from. A concatenation compresses
worse than separate files because the snapshot learns only block 0 (here
Python), and the later Perl, prose and XML blocks start from that.

| Machine | Threads | Compress / decompress (22.8 MB concatenation) | Peak RSS (bench) |
|---|---|---|---|
| Apple M4 (Linux VM, 4 cores) | 1 | 0.65 / 0.64 MB/s | 462 MiB |
| Apple M4 (Linux VM, 4 cores) | 4 | 1.67 / 1.58 MB/s | 1,085 MiB |
| x86 VM (Xeon 2.8 GHz) | 1 | 0.20 / 0.19 MB/s | 457 MiB |
| x86 VM (Xeon 2.8 GHz) | 4 | 0.44 / 0.47 MB/s | 1,080 MiB |

Archives are byte-identical on both machines and for any thread count.
Separate files run slower (0.86 MB/s on the M4 at 4 threads) because a
4-8 MiB file has only 1-3 blocks after the serial block 0.

Memory: a call holds the snapshot plus one model per worker. The context
tables dominate, and the archive records one of four table layouts
(`EngineOptions::tier4_tables`). Measured 2026-10-27 on 2 MiB blocks after
a 2 MiB snapshot:

| Layout | Model | Size vs Full | Note |
|---|---|---|---|
| 0 Full | ~194 MiB | 0 | all contexts 2^17 buckets (the original design) |
| 1 Balanced | ~158 MiB | +0.20% | orders 2-4 and 16/24/32 at 2^15 |
| 2 Compact | ~138 MiB | +0.52% | Balanced, plus orders 5-12 at 2^16 |
| 3 Small | ~74 MiB | +2.2% | every context at 2^15 |

Speed does not change with the layout: the model is bound by instructions,
not cache misses. By default (`tier4_tables = -1`) the encoder picks the
largest layout whose snapshot plus four workers fit `memory_budget`
(default 900 MiB), which is Balanced for inputs over ~4 MiB. That choice
does not depend on the thread count, so archives stay identical for any
`threads`. `memory_budget` also caps the workers, as before.

On the 26.97 MB mixed corpus (x86 VM, 4 threads, concatenation): Full under
a 900 MiB budget runs 3 workers, 0.74 MB/s, 889 MiB peak RSS; Balanced runs
4 workers, 0.93 MB/s, 909 MiB, +0.35% size (4,374,909 vs 4,359,652 bytes).

Each bucket's four check bytes are read from the bucket's own cache line
(there used to be a separate tag array), so a lookup touches one line, not
two: +13% speed with byte-identical archives (0.86 -> 0.97 MB/s on the
concatenation, 4 threads), and ~10 MiB less per model.

`tier4_bypass` (off by default, recorded in the archive) adds a long-match
bypass: inside a match of 32+ bytes, one flag (adaptive by match length and
predicted byte) says whether the predicted byte is right, and a right byte
skips the model. Measured on the same corpus: 0.97 -> 1.08 MB/s (1.11x),
+0.70% size (xml +3.5%, code +1.5%, prose 0). Variants that were worse: a
16 or 24 byte threshold (+1.3-3.9% size), and still training the context
tables on skipped bytes (slower and larger).

Sharing the snapshot copy-on-write does not help: after one 256 KiB block a
worker has written 99-100% of its 4 KiB table pages (100% at 16 KiB), because
hashed contexts scatter their writes. Small inputs get smaller tables,
because tables are sized from the bytes a model will see.

Not done: block 0 is serial in decode, so speed grows with the number of
blocks; the encoder could run v4's bit-exact producer pipeline on it, but
the decoder cannot.

### Tiers 1 and 2 against real codecs

Re-measured on a rebuild of the same five-file corpus (26.4 MB; zstd -19 =
5,217,375 bytes summed per file), one thread, peers via python-zstandard,
python-lz4, brotli and lzma:

| codec | size | vs zstd -19 | comp MB/s | dec MB/s |
|---|---|---|---|---|
| lz4 -1 | 10,733,385 | +106% | 380 | 1,130 |
| Tier 1 (16 Ki hash) | 10,137,183 | +94% | 344 | 1,256 |
| **Tier 1 (64 Ki hash)** | 9,980,486 | **+91%** | 327 | 1,366 |
| lz4 -12 | 7,877,407 | +51% | 9.8 | 1,446 |
| zstd -1 | 7,361,338 | +41% | 278 | 754 |
| zstd -3 | 6,568,232 | +26% | 199 | 825 |
| Tier 2 (no filter, order-0 literals) | 6,062,343 | +16.2% | 23-25 | 430-447 |
| zstd -6 | 6,024,509 | +15% | 73 | 792 |
| brotli -6 | 5,773,772 | +11% | 27 | ~380 |
| zstd -12 | 5,759,873 | +10% | 18 | 621 |
| **Tier 2 (filter + 16 literal streams)** | 5,666,549 | **+8.6%** | 24-26 | 270-279 |
| zstd -15 | 5,653,379 | +8% | 4.0 | 731 |
| zstd -18 | 5,233,396 | +0.3% | 2.4 | |
| zstd -19 | 5,217,375 | 0 | 1.9 | 669 |

Tier 2 per file vs zstd -19: bin +10.2%, novel +1.7%, Perl +8.5%, Python
+8.7%, XML +17.8%. On 4 threads it compresses at 32-52 MB/s and decodes at
490-570 MB/s. The filter is what costs decode speed (about 300-460 MB/s
per output byte on its own), plus the per-byte literal loop.

No LZ codec reaches zstd -19 size above about 2.5 MB/s per core, and LZ4's
byte format cannot get below +51% at any speed. Measured levers that did
not go in: an 8 MiB window across blocks (-1.6%, 1.4x slower, blocks stop
decoding independently), hash chain depth 64 (-2.1%, 2.8x slower), a
price-based optimal parser (-3.8%, 4.7x slower; +3.1% vs zstd -19 at
2.9 MB/s together with the rest), per-128 KiB rANS tables (-0.15%).

### Why Tiers 3 and 4 (old design) miss

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
1/2/3/4/7 threads, engine rules, the Tier 4 primer path, corruption
fuzzing, and peak RSS per tier
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
src/word_filter.cpp             Tier 2 word-dictionary pre-filter
src/word_dict.inc               its two built-in dictionaries (tools/word_dict/make_dict.py)
src/rans_static.{hpp,cpp}       8-way static rANS, scalar / AVX2 / NEON decoders
src/rans_binary.hpp             Two-state binary rANS for the CM tiers
src/cm_model.hpp                CM predictor: LUTs, counters, buckets, mixer kernels, APM, match model
src/cm_codec.hpp                CM block codec and engine (table sizing, worker cap)
src/dedup.{hpp,cpp}             Long-range dedup pre-pass
src/tier3_balanced.cpp          Tier 3 profile (contexts and knobs)
src/tier4_enterprise.cpp        Tier 4 engine: primer, block 0, snapshot copies on workers
src/sws_model.{hpp,cpp}         Tier 4 22-context CM (port of HCE Ultra v4), copyable for snapshots
src/sws_primer.inc              Tier 4 built-in 256 KiB primer (sources: third_party/sws_primer/NOTICE.md)
src/range_coder.hpp             Tier 4 binary range coder
tools/ultra_cli.cpp             ultra: compress/decompress/inspect files
tools/ultra_bench.cpp           ultra_bench: round-trip benchmark
tests/test_ultracodec.cpp       Test suite
```
