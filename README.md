<div align="center">

# HCE — Hybrid Compression Engine

**Neural-symbolic context mixing in a hardened, streaming, parallel container.**

21–45% smaller than `zstd -19` on text, source code and logs · bit-exact on x86-64 and AArch64 ·
zero sanitizer findings across GCC and Clang

[![CI](https://github.com/Aung-Nyi-Thant/neural-symbolic-code/actions/workflows/ci.yml/badge.svg)](https://github.com/Aung-Nyi-Thant/neural-symbolic-code/actions/workflows/ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](#quick-start)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Sanitizers](https://img.shields.io/badge/sanitizers-ASan%20%C2%B7%20UBSan%20%C2%B7%20TSan-2ea44f)](#quality-assurance)
[![Architectures](https://img.shields.io/badge/arch-x86--64%20%C2%B7%20AArch64%20NEON-informational)](#determinism-r11r17)
[![Ratio vs zstd -19](https://img.shields.io/badge/vs%20zstd%20--19-21%E2%80%9345%25%20smaller-2ea44f)](#performance-and-benchmarks)
[![CMake](https://img.shields.io/badge/CMake-%E2%89%A5%203.20-064F8C?logo=cmake&logoColor=white)](#quick-start)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)](#python-package)

[Overview](#overview) · [ultracodec](#ultracodec-the-4-tier-engine) · [Architecture](#architecture) · [Hardening](#hardening-and-security-r11r21) ·
[Quality](#quality-assurance) · [Benchmarks](#performance-and-benchmarks) · [Quick start](#quick-start) ·
[Python](#python-package) · [Tooling](#tooling-and-automation) · [License](#license)

</div>

---

## Overview

HCE is a C++20 compression engine built for data that must be **small, verifiable and safe to decode**:
log archives, backups, data-lake cold tiers and bandwidth-bound transfers. It pairs a real
**neural-symbolic context-mixing (CM) codec** with **Container v3 FRAMED**, a streaming format engineered
for hostile input, partial damage and parallelism.

The repository also ships [**ultracodec**](#ultracodec-the-4-tier-engine), a separate 4-tier engine that spans
LZ77 speed (Tier 1, 606–626 MB/s on 4 threads) to CM density (Tier 4, 28.7% smaller than `zstd -19` on
text and code). Its measured throughput, memory bounds and audit results are below.

| Pillar | What it means |
|---|---|
| **Ratio** | Across levels 0–4, 21–45% smaller than `zstd -19` and 19–41% smaller than `xz -9e` on the benchmark corpus, with the largest gains on logs. |
| **Safety** | Memory, output volume and decode work are bounded by declared, validated limits. Nothing is emitted before it is verified. |
| **Resilience** | Every frame is an independent sync point. Damage costs only the frames it touches; three recovery modes report every lost byte range. |
| **Reproducibility** | The output is byte-identical across thread counts, SIMD kernel paths (scalar, SSE2, AVX2, AVX-512, NEON), compilers (GCC, Clang), PGO and non-PGO builds, and architectures (x86-64, AArch64). |
| **Parallelism** | Frames compress and decompress on all cores under counting-semaphore back-pressure and a hard memory bound. |

> [!NOTE]
> **Status:** version 2.2.0. Container v3 FRAMED, codec profiles 1 (`prior_id` `0x4E434D01`) and 2
> (`0x4E434D02`, the CLI's and the Python package's default since 2.2; see [Codec profile 2](#codec-profile-2)).
> Every 2.0 and 2.1 stream decodes unchanged.
> Streams from 1.x, written by the retired LZ stand-in codec, are rejected as unsupported (exit 5),
> never misdecoded.

---

## ultracodec: the 4-tier engine

[`ultracodec/`](ultracodec/README.md) is a second, independent engine in this repository (C++20, no code
shared with the HCE codec). One `ICompressionEngine` interface fronts four strategies, from an LZ77 tier
that runs at hundreds of MB/s to a 22-context CM tier that beats `zstd -19` by 28.7%. Every tier
round-trips, every archive is byte-identical for any thread count and on x86-64 (SSE2/AVX2) and AArch64
(NEON), and the decoder validates every length before it allocates.

> [!NOTE]
> **Every number in this section was measured.** Where a metric has not been measured for the current
> default build, the cell says so. Measurements come from three corpora on 4-vCPU cloud VMs, so read each
> tier's row against its own corpus (listed under [How these numbers were measured](#how-these-numbers-were-measured)).
> Timing on these VMs varies by about ±20% from run to run; ranges show the spread across sessions.

### Tier specification

| Tier | Name | Pipeline (default build on `main`) | Block | Memory model |
|---|---|---|---|---|
| 1 | Hyper-Speed | Greedy two-hash LZ77 (8-byte 2¹⁷ + 5-byte 2¹⁶ hashes, repeat offsets), entropy-coded sequences via rANS. Raw LZ4-layout mode still available (`tier1_entropy = false`) | 1 MiB | 256–768 KiB of hash tables per thread (L2-resident) |
| 2 | Consumer | Gated word-dictionary pre-filter → lazy hash-chain LZ77 → 8-way interleaved static rANS (AVX2 / NEON decode), literals in 16 context streams | 2 MiB | ~30 MiB per thread (computed: 1 MiB head + 4 B chain per position) |
| 3 | Balanced | Long-range dedup pass → wavefront-primed blocks → lean CM with **16-bit counters** → **one rANS stream per 64 KiB** | 1 MiB + 64 KiB prefix | 16 MiB table per worker; 512 MiB default budget |
| 4 | Enterprise SWS | 256 KiB built-in primer + block 0 train one model; every later block starts from a **packed snapshot** of it. 22-context CM in the **Rebalanced** layout, 4 **int16** mixer sets, 4 APMs, binary range coder | 2 MiB | ~159 MiB per worker model + ~61 MiB packed snapshot; 900 MiB default budget |

### Performance floor (current defaults on `main`)

Size is the sum of per-file archives against the sum of per-file `zstd -19` archives. Negative = smaller
than `zstd -19`.

| Tier | Reference corpus | vs `zstd -19` | 1 thread comp / dec (MB/s) | 4 threads comp / dec (MB/s) | Peak RSS, 4 threads |
|---|---|---:|---:|---:|---:|
| 1 | Mixed 26.97 MB | **+33.5%** | 160–168 / 310–356 | 606–626 / 866–937 (concatenation) | not measured for entropy mode (raw mode: 152 MiB) |
| 2 | Mixed 26.4 MB | **+8.6%** | 24–26 / 270–279 | 32–52 / 490–570 | not re-measured since the word filter (177 MiB before it) |
| 3 | Text/code 22.8 MB | **−13.6%** | not measured for the compact format | 4.45 / 5.18 (concatenation) | **213.1 MiB** (131.1 MiB on 1 thread) |
| 4 | Text/code 22.8 MB | **−28.7%** (−26.9% as one archive) | not measured since the memory pass | 0.52–0.74 / 0.51–0.76 (concatenation); 0.25–0.26 per file | **788.1 MiB** (750.3 / 764.2 MiB comp / dec, concatenation only) |

Density on the other corpora, for the same tiers:

| Tier | Mixed corpus (with ELF binaries) | Text/code corpus |
|---|---:|---:|
| 1 | +33.5% (26.97 MB) | not measured |
| 2 | +8.6% (26.4 MB); per file: bin +10.2%, novel +1.7%, Perl +8.5%, Python +8.7%, XML +17.8% | not measured |
| 3 | −10.4% (26.4 MB, measured before the compact format, which changes size by −0.003%) | −13.6% |
| 4 | −24.0% (26.97 MB, measured on the PR #10 build: Balanced layout, int32 mixer) | −28.7%; per file (PR #8 build): Python −30.8%, Perl −29.5%, XML −29.3%, novels −26.6% |

```mermaid
xychart-beta
    title "Size vs zstd -19, % (each tier on its reference corpus; lower is better)"
    x-axis ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
    y-axis "% vs zstd -19" -35 --> 40
    bar [33.5, 8.6, -13.6, -28.7]
```

**Against peers (one thread, 26.4 MB mixed corpus, Tiers 1 and 2):** Tier 2 at +8.6% and 24–26 MB/s is
denser than `zstd -12` (+10%, 18 MB/s) and `brotli -6` (+11%, 27 MB/s). No measured LZ codec reaches
`zstd -19` size above about 2.5 MB/s per core (`zstd -18`: +0.3% at 2.4 MB/s). Tier 1's entropy mode is as
dense as `zstd -2` (+34.7%) and 1.5× slower to compress.

### Memory bounds

Peak RSS ceilings are enforced by the test suite (`test_rss_limits`), which runs each tier in a forked child and reads
`ru_maxrss` (Linux release builds; skipped under sanitizers). The inputs are counted in the RSS.

| Tier | Test input | Test ceiling | Measured (2026-10-30, `main`, x86 VM) | Default model budget |
|---|---:|---:|---:|---:|
| 1 | 32 MiB | 160 MiB | 139 MiB | n/a |
| 2 | 32 MiB | 256 MiB | 144 MiB | n/a |
| 3 | 4 MiB | 256 MiB | 130 MiB | 512 MiB |
| 4 | 4 MiB (snapshot + one worker) | 640 MiB | 263 MiB | 900 MiB |

`memory_budget` caps the worker count, never the model size, so archives do not depend on how many workers
run. Tier 4 picks its table layout (Rebalanced, else Lean, else Small) as the first whose snapshot plus four
workers fits the budget, independent of the thread count.

### Memory audit results (PRs #11–#13)

The 2026-10-27 audit inventoried every allocation in all four tiers. It found no pointers or padding in any
table to strip. The waste was RAM given to contexts that could not use it, a match table 82–92% empty, a
full snapshot kept resident only to be copied, and 32-bit Tier 3 counters that only needed 16 bits. Each
change shipped behind a `model_param` salt or flag, so every older archive still decodes.

All rows: 22.8 MB text/code corpus, x86 VM, 4 threads.

| Step | PR | Size: sum of files | Concatenation | Peak RSS | Speed (concatenation) |
|---|:---:|---:|---:|---:|---:|
| **Tier 4** before (Balanced layout, unpacked snapshot) | #10 | 2,990,482 | 3,071,982 | 883.6 MiB | 0.71–0.76 MB/s comp |
| + Rebalanced layout, packed snapshot (T4-A, T4-C) | #11 | 2,983,124 (−0.25%) | 3,063,104 (−0.29%) | 794.4 MiB | same within noise |
| + int16 mixer weights (T4-D) | #12 | 2,986,164 (+0.10%) | 3,065,845 (+0.09%) | 788.1 MiB | +8% comp and dec (8 interleaved runs) |
| Lean layout instead (T4-A′, `tier4_tables = 5`) | #11 | 2,992,783 (+0.08% vs #10) | 3,074,208 | 638.9 MiB | same within noise |
| **Tier 3** before (32-bit counters, one rANS stream per block) | #10 | 3,621,559 | 3,620,815 | 397.8 MiB | 4.39 / 4.80 MB/s |
| Compact format (T3-A, T3-B) | #13 | 3,621,491 (−0.002%) | 3,620,697 (−0.003%) | **213.1 MiB** (−46 MiB per worker) | 4.45 / 5.18 MB/s |

The 788.1 MiB row comes from the phase 4 gate run of the default build; the other RSS figures are from each
PR's own run.

```mermaid
xychart-beta
    title "Peak RSS at 4 threads, MiB (22.8 MB text/code corpus)"
    x-axis ["T4 before", "T4 #11", "T4 #12", "T4 Lean", "T3 before", "T3 #13"]
    y-axis "MiB" 0 --> 900
    bar [883.6, 794.4, 788.1, 638.9, 397.8, 213.1]
```

### How the wins were achieved

| Technique | Tier | Status | What it does | Measured effect |
|---|:---:|---|---|---|
| **Rebalanced context tables** (T4-A) | 4 | Merged, #11 (default) | The audit halved each context's table on its own and measured the size cost per MiB freed. Bits moved from contexts that do not use them (column −3 bits, indirect 1 −2, orders 2/16 and two others −1) to the starved ones (orders 3/4 +2 bits; order 5, word pair, 2 words + byte +1). Match table 2²² → 2²⁰ entries | −0.25% files / −0.29% concatenation at the same RAM per model |
| Lean layout (T4-A′) | 4 | Merged, #11 (opt-in) | The shrinks only | −35 MiB per model, +0.08% size |
| **Packed snapshot** (T4-C) | 4 | Merged, #11 (default) | After block 0, every large array is XORed with its initial value and stored per 64-byte chunk as a non-zero mask plus the non-zero bytes, in one exactly sized mapping. Workers expand it at each block start instead of a `memcpy` | Snapshot ~159 → ~61 MiB; expansion 60–75 ms per 2 MiB block (under 1% of block time); bit-exact |
| **int16 mixer weights** (T4-D) | 4 | Merged, #12 (default) | Weights at 1.0 = 4096, saturating at ±8. A set's dot product is two `vpmaddwd` (AVX2) or four `smlal` (NEON); training rounds like `vpmulhrsw` / `sqrdmulh`. Scalar, AVX2 and NEON kernels give identical results | +8% comp and dec on 4 threads; +0.09–0.10% size; ~1 MiB less per model |
| **16-bit counters** (T3-A) | 3 | Merged, #13 (default) | Hashed counters store `p12 << 4 \| n` (n ≤ 15). A nibble bucket is 32 bytes, so a two-way pair fits in one cache line instead of two; 16 MiB holds as many buckets as the old 32 MiB table | Part of the −46 MiB per worker below |
| **Chunked rANS** (T3-B) | 3 | Merged, #13 (default) | One rANS stream per 64 KiB of the block. The reverse-coding record plus word buffer drops from 32 MiB to 2 MiB per 1 MiB block, for 12 bytes per stream | With T3-A: 397.8 → 213.1 MiB at 4 threads, −0.003% size |
| Check bytes inside the bucket | 4 | Merged, #10 | A lookup reads one cache line instead of two (the separate tag array is gone) | +13% speed, byte-identical archives, ~10 MiB less per model |
| Snapshot warm-start (SWS) | 4 | Merged, #8 | Later blocks start from a model trained on the primer and block 0, so they run in parallel without starting cold | Tier 4 study: −28.8% with 2 MiB SWS blocks vs −21.8% with cold 1 MiB blocks |
| Shared history, copy-on-write (T4-F) | 4 | **Open, PR #15** | Primer + block 0 history mapped `MAP_PRIVATE` into each worker | −9.5 MiB peak PSS (−2.4 MiB per worker); RSS only −2.8 MiB; archives identical |
| Tier 3 2 MiB blocks + 256 KiB prime | 3 | **Open, PR #14** | Longer blocks and prime in the compact format | −16.2% files / −15.5% concatenation at 5.03 / 5.51 MB/s, 217 MiB |
| Compact path slots (T4-E) | 4 | **Dropped** (lab only) | 16-byte slot quarters holding one full slot or two 7-byte path entries | −0.41% size but −15 to −17% speed and +40 MiB RSS; missed its ≤8% speed gate |

**What memory does not buy.** Doubling every Tier 4 table costs +124 MiB per model (about +620 MiB at
4 threads) and returns −0.47%. One block dirties 99–100% of table pages, so copy-on-write or compressed
working tables do not help; compression fits only the cold snapshot. Memory stalls are about 16% of Tier 4
time, so the CM tiers are bound by instructions, and the next points of density have to come from more
CPU per bit, not more RAM.

### Targets that are not met

| Tier | Target | Measured | Gap |
|---|---|---|---|
| 1 | 752 MB/s compress, 4 threads (entropy mode) | 606–626 MB/s | ~18% short |
| 3 | ~11 MB/s, −17.2% vs `zstd -19` | 4.45 MB/s (4T); −13.6% text/code, −10.4% mixed | ~2.4× on speed; density short without PR #14 |
| 4 | about −29% at about 2 MB/s | −28.7% per file; 0.52–0.74 MB/s on 4 x86 threads; 1.67 MB/s on 4 Apple M4 cores (measured before PRs #10–#12) | Density within 0.3 points; speed short |

### How these numbers were measured

| Corpus | Contents | `zstd -19` (sum of files) | Machine | Used for |
|---|---|---:|---|---|
| Text/code, 22.8 MB | 8 MiB Python 3.12 stdlib, 4 MiB Perl modules, 4 MiB `/usr/share` XML/HTML/SVG, 5.7 MB of five Gutenberg novels | 4,191,049 B | 4-vCPU Xeon VM, 2.8 GHz | Tiers 3 and 4, memory audit |
| Mixed, 26.97 MB | The text/code corpus plus 4 MiB of `/usr/bin` ELF binaries | 5,758,651 B | 4-vCPU Xeon VM, 2.1 GHz | Tier 1 entropy mode, Tier 4 mixed row |
| Mixed, 26.4 MB | 8 MiB Python, 6 MiB XML/HTML/SVG, 4 MiB Perl, 4 MiB ELF, 3.4 MB novel | 5,217,375 B (rebuild) | 4-vCPU Xeon VM, 2.8 GHz | Tier 2, Tier 3 mixed row |

GCC 13 `-O3`, SSE2 baseline with AVX2 chosen at run time; `zstd -19` via python-zstandard; speeds in
MB/s (10⁶ bytes/s) from `ultra_bench`. Per-tier design notes, tuning results and the full history are in
[`ultracodec/README.md`](ultracodec/README.md).

---

## Architecture

```mermaid
flowchart TB
    IN([Input stream]) --> ENC
    subgraph ENC["Parallel encoder"]
        direction LR
        RD["Reader<br/>frames into T+2 slots"] --> E1["Worker 1<br/>NeuralCMCodec"] & E2["Worker 2<br/>NeuralCMCodec"] & EN["Worker T<br/>NeuralCMCodec"]
        E1 & E2 & EN --> EW["Ordered<br/>writer"]
    end
    ENC -- "byte-identical for any T" --> ST[["Container v3 FRAMED stream"]]
    ST --> DEC
    subgraph DEC["Parallel decoder"]
        direction LR
        SP["Splitter<br/>validation · R19 budget<br/>· resync"] --> D1["Worker 1<br/>NeuralCMCodec"] & D2["Worker 2<br/>NeuralCMCodec"] & DN["Worker T<br/>NeuralCMCodec"]
        D1 & D2 & DN --> DW["Ordered writer<br/>gap replay"]
    end
    DEC -- "identical to the sequential decoder" --> OUT([Verified output])
```

### Tiered v3 container: stream → frame → block

```mermaid
flowchart TB
    SH["Stream header · 32 B<br/>codec id · level · window · block · content size"] --> F0
    subgraph F0["Frame 0 · sync point"]
        direction LR
        H0["Frame header · 40 B<br/>marker · params · offset<br/>models reset here"] --> B00["CM block<br/>CRC-32C ×2"] --> B01["STORED block<br/>CRC-32C"]
    end
    F0 --> F1
    subgraph F1["Frame 1 · sync point"]
        direction LR
        H1["Frame header · 40 B<br/>marker · params · offset<br/>models reset here"] --> B10["CM block<br/>CRC-32C ×2"] --> B11["CM block<br/>CRC-32C ×2"]
    end
    F1 --> EN["END trailer · 32 B<br/>totals · CRC over every block checksum"]
```

- **Stream** — a self-describing header carries the codec ID, level, window and block geometry, and the
  content size. The size may be recorded as unknown, so pipes are supported (R18). Concatenated streams
  decode back to back.
- **Frame** — a sync point. An 8-byte marker and a CRC-protected 40-byte header repeat every coding
  parameter plus the frame's output offset. Models reset completely at every frame, so each frame decodes
  on its own: this is what enables parallel decoding, late join and recovery.
- **Block** — up to 8 MiB. A 20-byte header carries the sizes and CRC-32C checksums of both the payload
  and the decoded content. Blocks are `CM` (always strictly smaller than their content), `STORED`, or
  skippable `ancillary` metadata.

### Neural-symbolic CM codec

```mermaid
flowchart TB
    subgraph EXP["Symbolic experts"]
        direction LR
        X1["Orders<br/>0 · 1 · 2 · 3 · 4 · 6"] ~~~ X2["Word · Column<br/>Log field · Sparse"] ~~~ X3["Long-range<br/>match model"]
    end
    EXP -- "13 inputs · Q8 logits" --> NM
    subgraph NM["Integer neural mixer"]
        direction LR
        L1["Layer 1 · 3 gated sets<br/>partial byte · match state<br/>· previous byte"] --> L2["Layer 2 · 1 gated set<br/>bit position"]
    end
    NM --> APM["APM / SSE refinement<br/>order 0 · order 1"] --> AC["Binary arithmetic coder<br/>32-bit · exact range split"]
```

The codec predicts every bit and codes it with a binary arithmetic coder.

- **Symbolic experts.** Ten counter-based context models make the predictions. Orders 0 and 1 use direct
  tables; orders 2, 3, 4 and 6 are hashed. The remaining four are domain contexts: a case-folded **word**
  model, a **column** model that predicts from the byte above in the previous line, a **log-field** model
  keyed by the separator count of the current line and the last two bytes, and a **sparse** model that
  skips the most recent byte. Every context state is a 16-bit counter (12-bit probability, 4-bit
  confidence). The hashed models keep theirs in nibble-bucketed, two-way associative tables with
  deterministic replacement.
- **Match model.** A long-range match model indexes 6-byte contexts. It verifies every candidate inside
  the window before predicting from it.
- **Neural mixer.** A two-layer integer mixer combines the experts' logits. In layer 1, symbolic state
  (the partial byte, the match state and the previous byte) selects three of 544 weight sets; layer 2
  selects one of eight by bit position. Weights are int16 in Q14 format and learn online by integer gradient steps.
- **Refinement and coding.** Two adaptive probability maps (APM/SSE) refine the mixer's output. The
  arithmetic coder splits its range exactly with a 64-bit product, and the decoder rejects any payload it
  does not consume exactly.

**Neuro-symbolic routing.** Deterministic symbolic signals route every decision in the system:

- **Blocks.** An integer entropy-and-repeat estimator routes each block to CM or STORED. A block that
  would expand is stored instead, and the encoder rebuilds the exact decoder state by replay (R12).
- **Bits.** Symbolic context gates which weight sets the neural mixer uses for every bit.
- **Bytes.** Inside long verified matches, coding switches to a single decision per byte. Both sides take
  this switch from state they share (R13).

**Levels** trade model memory for ratio. The decoder needs the same model memory, plus the window and one block buffer.

| Level | Model memory per thread | Buckets per expert | Match-table entries | One-decision path from |
|:---:|---:|---:|---:|---:|
| 0 | 21 MiB | 2¹⁶ | 2¹⁸ | 32-byte matches |
| 1 | 38 MiB | 2¹⁷ | 2¹⁹ | 64-byte matches |
| **2** (default) | **40 MiB** | 2¹⁷ | 2²⁰ | 128-byte matches |
| 3 | 76 MiB | 2¹⁸ | 2²¹ | 256-byte matches |
| 4 | 149 MiB | 2¹⁹ | 2²² | 1,024-byte matches |

### Codec profile 2

Profile 2 (`prior_id` `0x4E434D02`, [`codec/ncm_model2.hpp`](codec/ncm_model2.hpp)) is a separate,
higher-ratio model. Profile 1's code and coded bits are unchanged, so every existing archive still decodes;
a decoder reads the profile from each frame. The CLI and the Python package write profile 2 by default;
`hce --codec-profile 1` and `hce.compress(data, profile=1)` keep writing profile 1, which is about twice as
fast. Profile 2 adds:

- **Seven more hashed experts.** Orders 5 and 8, word pairs, two **indirect** models (the bytes that
  followed the last occurrences of the order-1 and order-2 contexts), and two **structure** models for
  JSON, XML/HTML and C-like source code: one keyed by the enclosing bracket or element, one by the current
  key, attribute or identifier. Table sizes differ per expert, within the level's arena (R21).
- **A second match model** that indexes 16-byte contexts and verifies 16 bytes before it predicts.
- **A wider mixer.** Each expert feeds both its logit and its probability: 48 inputs in three groups of
  sixteen, each group one call of the existing SIMD kernels, summed in 64 bits. Both match models select
  weight sets.
- **A third APM** over a hashed order-2 context, and faster-adapting counters.

| Level | 0 | 1 | **2** | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Model memory per thread, profile 1 | 21 MiB | 38 MiB | **40 MiB** | 76 MiB | 149 MiB |
| Model memory per thread, profile 2 | 25 MiB | 42 MiB | **46 MiB** | 83 MiB | 157 MiB |

Results at level 2, one thread, on the same VM (`release` build, one run; RSS is the process peak):

| File | Size | Profile 1 | **Profile 2** | Change | Compress MB/s, p1 / p2 | Peak RSS, p1 / p2 |
|---|---:|---:|---:|---:|---:|---:|
| JSON (ISO 3166 codes) | 1,372,876 | 77,410 | **72,479** | −6.4% | 1.07 / 0.53 | 48 / 52 MiB |
| XML (shared MIME info) | 2,857,771 | 174,994 | **156,772** | −10.4% | 0.87 / 0.48 | 49 / 53 MiB |
| C++ standard headers | 4,000,000 | 329,911 | **308,820** | −6.4% | 1.02 / 0.49 | 51 / 55 MiB |
| `dpkg` logs | 586,148 | 18,908 | **14,990** | −20.7% | 1.25 / 0.51 | 47 / 51 MiB |
| Changelogs (prose) | 4,000,000 | 204,617 | **203,000** | −0.8% | 2.30 / 0.95 | 51 / 54 MiB |
| Python standard library | 4,000,000 | 408,841 | **391,911** | −4.1% | 1.16 / 0.52 | 51 / 55 MiB |
| **Total** | 16,816,795 | 1,214,681 | **1,147,972** | **−5.5%** | | |

Decompression runs at the same speed as compression. Frames still compress and decompress in parallel,
and the output is byte-identical at every thread count. The benchmark tables below were measured with
profile 1.

---

## Hardening and security (R11–R21)

HCE is specified by eleven hardening rules. Each is enforced in code and exercised by the test suites.

### Determinism (R11–R17)

| Rule | Guarantee | How it is enforced |
|:---:|---|---|
| **R11** | No model state is shared between threads | Each worker owns a codec. Output is byte-identical for 1, 2, 3, 5 and 8 threads (tested). The encoder's two-stage pipeline splits one model between two threads along disjoint tables: the helper owns the counters, the coding thread the mixers and APMs. |
| **R12** | STORED data feeds the history but never the models | When a block would expand, it is stored and the encoder replays the frame's CM blocks to restore the decoder's exact state. Replay is capped at 3 per frame. |
| **R13** | Implicit decisions use shared integer state only | The match fast path, weight-set gating and block-start context derivation all read decoded state, never encoder-only data. |
| **R14** | Declared widths, no overflow | Formats are fixed-point: Q8 logits, Q12/Q16/Q22 probabilities, Q14 weights. Key bounds are checked by `static_assert`, and the tables are generated by `consteval` integer code. |
| **R15** | SIMD is exact | Dot products are int16×int16→int32 sums kept below 2³⁰, so every reduction order agrees. Training is lane-wise and identical on SSE2, AVX2, AVX-512 and NEON. |
| **R16** | No floating point on the decode path | The codec contains no floating point at all. |
| **R17** | Format changes need a new version | Nine golden streams freeze each profile. Changing any coded bit requires a new codec ID: profile 2 is one, and profile 1's code is untouched by it. |

### Streaming I/O (R18)

- **Pipe-native.** The decoder is a push API that never seeks and never reads past what it consumes. The
  encoder emits frames as they complete. Both ends work on pipes, and the content size is recorded when
  it is known.
- **Atomic files.** The CLI writes to a temporary file in the target directory, then fsyncs and renames
  it. A failed or interrupted run leaves no partial file unless `--keep-partial` is given.
- **Clean shutdown.** SIGINT, SIGTERM and SIGHUP stop a run promptly and remove temporary files. A closed
  pipe surfaces as an error (exit 1), never as a hang.

### Bounded resources and compression-bomb defense (R19)

- **Token-bucket expansion budget.** Output may not exceed `ratio × input + burst`. Each input byte earns
  `ratio` bytes of credit, up to the burst, and each block spends its declared size **before** any decode
  work. The `untrusted` profile sets a ratio of 100 with a 64 MiB burst. In the tests, a 6 MiB bomb stops
  at the configured burst with identical results from the sequential and parallel decoders.
- **Declared limits, checked before allocation.** Memory, window, block size, level and total output are
  validated against the stream header before anything is allocated, and each stream allocates at most once.
  The `untrusted` profile caps these at 256 MiB, 32 MiB, 1 MiB, level 3 and 64 GiB respectively.
- **Shape checks.** A CM block must be strictly smaller than its content, and a STORED block exactly the
  size of its content. Anything else is rejected as corrupt before decoding.
- **Bounded work per byte.** CM decoding costs O(declared size) whatever the payload. A frame reset clears
  only the cache lines the previous frame touched, so a stream of tiny frames cannot amplify CPU time: at
  level 4, 390 one-KiB frames decode within about 10% of the time one frame holding the same bytes takes.

### Recovery and integrity (R20)

- **Verify, then emit.** Every block's payload and decoded content are checked with CRC-32C before a single
  byte reaches the output. The END trailer cross-checks totals and the full list of block checksums.
- **Three recovery modes.**
  - `strict` stops at the first damage; its output is always a verified prefix.
  - `skip` drops lost ranges.
  - `skip_and_fill` preserves offsets by filling lost ranges with a chosen byte.

  Every lost range is reported as JSON with its output offset, length, input offset and cause.
- **Resynchronization.** After damage, the decoder scans for the next frame marker, and forged markers fail
  their CRC. STORED blocks inside a damaged frame are salvaged. In the recovery modes, decoding can join a
  stream mid-way at any frame header.
- **Deterministic recovery.** For every damaged input, the parallel decoder produces exactly the sequential
  decoder's output, status, gap reports and statistics, which is checked by differential fuzzing. The full
  suite, recovery included, passes on x86-64 and AArch64.

### Bounded history (R21)

- **Ring window.** The history is a ring of `2^window_log` bytes (64 KiB to 1 GiB). Memory does not depend on
  input size, and a sync point costs O(1).
- **Validated references.** Every back-reference and match candidate is range-checked against the reachable
  window. Table positions are 32-bit and validated by distance, so aliasing is deterministic, never undefined.
- **Exact arena.** All codec tables are carved from one caller-provided arena. `memory_required()` is exact:
  binding into one byte less fails.

---

## Quality assurance

Every configuration builds with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror`,
and every suite it builds passes with **zero compiler warnings and zero sanitizer reports**.
[CI](.github/workflows/ci.yml) runs the whole matrix on every push and pull request; each row is a CMake
preset, so any of them reproduces locally (see [Tooling and automation](#tooling-and-automation)).

| Preset | Toolchain | Instrumentation | Suites |
|---|---|---|:---:|
| `release` | GCC 13 `-O3` | — | 6 |
| `clang-release` | Clang 18 `-O3` | — | 6 |
| `gcc-asan` | GCC 13 | ASan + UBSan | 6 |
| `clang-asan` | Clang 18 | ASan + UBSan | 6 |
| `gcc-tsan` | GCC 13 | TSan | 6 |
| `clang-tsan` | Clang 18 | TSan | 6 |
| `aarch64-qemu` | GCC 13 cross, `armv8.2-a+dotprod` | QEMU user mode, `-cpu max` | 6 |
| `mingw-wine` | MinGW-w64 GCC 13 cross, static | Wine 9 | 5 (the CLI is POSIX-only) |
| `python` | GCC 13 `-O3`, `HCE_BUILD_PYTHON=ON` | — | 7 (the six, plus pytest) |

| Suite | Scope | Highlights |
|---|---|---|
| `framing` | Container (R18–R21) | 3,000-iteration mutation fuzz (bit flips, insertions, deletions, truncation), checked in strict and fill modes. Also: resync, forged markers, bombs, late join, cancellation. |
| `codec` | CM codec | 200,000-case SIMD parity per path. Nine golden streams per profile. 120 CRC-valid hostile payloads per profile, decoded in two modes. The two-stage encoder pipeline writes the same blocks as one thread on every kernel path, through expanding blocks and replays. Also, for both profiles: replay, late join, window bound and reset-cost bound. |
| `pipeline` | Parallel encoder and decoder | Byte-identical output for 1, 2, 3, 5 and 8 threads. Differential fuzzing on damaged streams (pipeline ≡ sequential). Also: cancellation, I/O failure, and a race-regression test (60 repetitions). |
| `aio` | Asynchronous file I/O | Every backend present (io_uring, worker threads): the engine contract, round trips across sizes, queue depths and `O_DIRECT`, interrupts and errors, and the pipelines over async files. |
| `structural` | Structural codec | Tokenizer fuzzing for losslessness, hostile token and model images, deterministic models, round trips across thread counts and SIMD paths, archives on every aio backend, damaged archives. |
| `cli` | `hce` binary | Exit codes, atomic output files, pipes, SIGINT and EPIPE, gap reports, limits, legacy-stream rejection, structural mode and format detection. |
| `python` | Python package ([pytest](#run-the-test-suite)) | Round trips over codecs × corpora × levels × languages, Hypothesis properties, every truncation and sampled bit flips rejected, limits and the exception hierarchy, fourteen golden archives (R17), bindings ≡ CLI byte for byte. |

**Cross-architecture bit-exactness.** The same nine golden streams, compared by size and CRC-32C, are
reproduced on x86-64 with the scalar, SSE2, AVX2 and AVX-512 kernels and on AArch64 with NEON, from GCC and Clang
builds. Streams written on either architecture decode on the other.

> [!TIP]
> Force a kernel path with `HCE_SIMD=scalar|sse2|avx2|avx512|neon`; every path writes identical streams.
> AVX-512 is never picked by default. Its kernels are the AVX2 ones built for AVX-512 (EVEX, 256 bits wide,
> no frequency drop) and measure on par with AVX2; 512-bit registers measured 5–10% slower, since each bit
> mixes one 16-lane vector per weight set on a serial chain.
> Deepen the differential fuzz with `HCE_FUZZ_ITERS=2000 ./build/release/test_pipeline`.

---

## Performance and benchmarks

**Setup.** Intel Xeon @ 2.80 GHz (4 vCPU cloud VM), Linux 6.18, GCC 13.3 `release` build. Each figure is
the best of two runs; tools ran single-threaded unless noted; MB = 10⁶ bytes. Baselines: gzip 1.12,
zstd 1.5.5, xz 5.4.5.

**Corpus.**

| Name | Contents | Size |
|---|---|---:|
| Prose | Package changelogs (English prose) | 4,000,000 B |
| Source code | Python 3.12 standard library | 4,000,000 B |
| System logs | Real `dpkg`/`apt` logs | 1,019,495 B |
| Access logs | Synthetic HTTP access log | 4,000,000 B |

### Compressed size (bytes, lower is better)

| Corpus | gzip -9 | zstd -19 | xz -9e | hce -l0 | **hce -l2** | hce -l4 |
|---|---:|---:|---:|---:|---:|---:|
| Prose | 890,674 | 357,486 | 343,944 | 257,074 | **251,185** | 250,180 |
| Source code | 975,333 | 777,950 | 761,780 | 615,559 | **602,376** | 597,364 |
| System logs | 107,531 | 77,263 | 72,752 | 52,572 | **42,761** | 42,653 |
| Access logs | 500,361 | 376,366 | 359,816 | 240,832 | **231,025** | 230,738 |

### Size reduction vs `zstd -19`

| Corpus | hce -l0 | **hce -l2** | hce -l4 | Ratio, hce -l2 vs zstd -19 |
|---|---:|---:|---:|---:|
| Prose | 28.1% | **29.7%** | 30.0% | 15.9 : 1 vs 11.2 : 1 |
| Source code | 20.9% | **22.6%** | 23.2% | 6.6 : 1 vs 5.1 : 1 |
| System logs | 32.0% | **44.7%** | 44.8% | 23.8 : 1 vs 13.2 : 1 |
| Access logs | 36.0% | **38.6%** | 38.7% | 17.3 : 1 vs 10.6 : 1 |

### Throughput (MB/s, one thread, compress / decompress)

| Tool | Prose | Source code | System logs | Access logs |
|---|---:|---:|---:|---:|
| gzip -9 | 18.8 / 149 | 5.6 / 133 | 19.5 / 111 | 12.9 / 175 |
| zstd -19 | 3.6 / 249 | 1.1 / 223 | 1.6 / 123 | 1.1 / 250 |
| xz -9e | 2.0 / 111 | 1.2 / 62 | 2.0 / 81 | 0.7 / 89 |
| hce -l0 | 1.5 / 1.4 | 0.69 / 0.67 | 0.86 / 1.1 | 0.94 / 1.0 |
| hce -l2 | 1.2 / 1.3 | 0.65 / 0.63 | 0.72 / 0.84 | 0.83 / 0.80 |
| hce -l4 | 0.86 / 1.0 | 0.52 / 0.51 | 0.68 / 0.65 | 0.73 / 0.62 |

### Multi-threaded scaling

The input is 32 MiB of access logs, which forms four 8 MiB frames, at level 2. The output is
**1,932,155 bytes (17.4 : 1), byte-identical at every thread count**; `zstd -19` produces 3,154,033 bytes.

| Compression | Throughput | Speed-up |
|---|---:|:---:|
| `-T 1` | 0.78 MB/s | 1.0× |
| `-T 2` | 1.66 MB/s | 2.1× |
| `-T 4` | 3.00 MB/s | **3.9×** |

| Decompression | Workers | Throughput | Speed-up |
|---|:---:|---:|:---:|
| `-T 1` | sequential | 0.78 MB/s | 1.0× |
| `-T 4`, default `trusted` limits | 3 | 1.54 MB/s | 2.0× |
| `-T 4 --max-window 8M --max-level 2` | 4 | 3.05 MB/s | **3.9×** |

### Source-code corpora (`tools/benchmark.py`)

Produced by [`tools/benchmark.py`](tools/benchmark.py), which [CI](#tooling-and-automation) also runs:
the same Xeon VM, 4 threads for every tool, hce level 2. Every result was decompressed and compared with
the original.

| Codec | Python stdlib (9.9 MB) | libstdc++ headers (12.4 MB) | HCE sources (0.5 MB) |
|---|---:|---:|---:|
| `hce` (byte codec) | **13.12%** | **7.75%** | **16.99%** |
| `hce --structural` | 15.91% | 8.90% | 19.64% |
| `xz -9e` | 16.41% | 9.54% | 21.65% |
| `zstd -19` | 16.78% | 9.99% | 22.08% |

Compressed size as % of the original. Speeds, compress / decompress in MB/s: byte codec 0.84 / 0.75 and
1.14 / 1.10, structural 1.45 / 1.73 and 1.76 / 2.18, on the two large corpora.

> [!IMPORTANT]
> **Cost is symmetric.** Context mixing spends about as much CPU to decompress as to compress: roughly
> 0.5–1.5 MB/s per core, against 60–250 MB/s for zstd or xz decompression. HCE is built for archival,
> retention, backup and bandwidth-bound transfer, where ratio and integrity dominate. It is not a
> replacement for zstd on latency-critical read paths. Parallelism comes from frames, so throughput scales
> with cores on inputs of several frames.

**Encoder pipeline.** When the input size is known (files, `content_size`) and there are at least twice as
many threads as frames, each encoder splits its frame over two threads: a helper runs the counter half of
the model (contexts, buckets, match model) ahead, since every bit is known in advance, and the coding thread
mixes and codes. The stream is byte-identical. On the 4 MB corpora below at level 2, encoding takes 2.0 s
and 1.7 s with `-T 2` against 3.3 s and 2.8 s with `-T 1`, and 2.1 s and 1.8 s for `zstd -19 -T2`.
Decoding cannot be split this way: the decoder learns each bit only by decoding it. Profile 1 only for now:
profile 2 encodes on one thread per frame.

---

## Quick start

### Requirements

- Linux on x86-64 or AArch64 (the CLI uses POSIX I/O).
- A C++20 compiler; GCC 13 and Clang 18 are tested.
- CMake ≥ 3.20 (≥ 3.25 to use the presets) and Ninja.

```bash
sudo apt install build-essential cmake ninja-build                # minimal
sudo apt install clang libclang-rt-18-dev                         # Clang + sanitizer runtimes
sudo apt install g++-aarch64-linux-gnu qemu-user                  # AArch64 cross-testing
sudo apt install g++-mingw-w64-x86-64-posix wine64                # Windows cross-testing
sudo apt install liburing-dev pkg-config                          # io_uring backend (optional)
```

### Build, test, install

```bash
cmake --preset release            # Release: codec at -O3 with loop vectorization
cmake --build --preset release
ctest --preset release            # all six suites
sudo cmake --install build/release   # installs bin/hce
```

Without presets, or tuned for the build machine:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHCE_NATIVE=ON
cmake --build build
ctest --test-dir build -j4 --output-on-failure
```

| CMake option | Default | Effect |
|---|:---:|---|
| `HCE_NATIVE` | `OFF` | Adds `-march=native` to the whole build; the binary may then not run on other CPUs. Mixer kernels are already chosen at run time (AVX2 → SSE2 → scalar on x86-64, NEON on AArch64), so a portable build loses little. |
| `HCE_SANITIZE` | empty | `address,undefined` or `thread`. |
| `HCE_WERROR` | `ON` | Treat warnings as errors; set `OFF` for packaging on newer compilers. |
| `HCE_BUILD_TESTS` | `ON` | Build the test suites. |
| `HCE_AIO_URING` | `AUTO` | io_uring backend of `aio/`: `AUTO` builds it when pkg-config finds liburing ≥ 2.2, `ON` requires it, `OFF` builds the thread backend only. |
| `HCE_PGO` | `OFF` | Profile-guided optimization: `GENERATE` builds instrumented binaries that write profiles to `HCE_PGO_DIR`, `USE` rebuilds from them. [`tools/pgo.sh`](tools/pgo.sh) runs the whole cycle; output is byte-identical either way. |
| `HCE_BUILD_PYTHON` | `OFF` | Build the Python extension `hce._hce` (needs Python ≥ 3.10 with headers and pybind11) and register the pytest suite with CTest. See [Python package](#python-package). |

The codec library always builds with `-O3 -ftree-vectorize`, except in `Debug` builds.

Reproduce the full verification matrix:

```bash
for p in release clang-release gcc-asan clang-asan gcc-tsan clang-tsan aarch64-qemu mingw-wine; do
  cmake --preset "$p" && cmake --build --preset "$p" && ctest --preset "$p" || break
done
```

### Use the CLI

```bash
hce -i app.log -o app.log.hce                     # compress; the file appears only once complete
cat app.log | hce -T 8 > app.log.hce              # streaming, 8 worker threads
hce -d < app.log.hce | grep ERROR                 # decompress to a pipe
hce -t -i app.log.hce --stats=json                # verify every byte, write nothing
hce -d --limits untrusted -i upload.hce -o out    # hostile input: R19 limits
hce -l 4 --window 64M --frame 32M -i big.tar -o big.tar.hce
hce --structural -i server.py -o server.py.hcs    # structural codec; --lang auto-detects Python from .py
hce -d -i server.py.hcs -o server.py              # -d and -t recognize either format by its first bytes
```

`-v` prints one readable line: sizes, ratio, time and speed. This is real output:

```console
$ hce --structural -v -T4 -i repo.cpp -o repo.hcs
hce: repo.cpp -> repo.hcs: 414834 -> 83359 bytes (20.09% of the original, 4.98x) in 0.862 s, 0.48 MB/s; structural, cpp, 140708 tokens, 4571 symbols, 4 threads, level 2, spool via io_uring
```

`--stats=json` writes the same numbers, and more, as one integer-valued JSON object for scripts.

> [!NOTE]
> **Byte codec or structural codec?** The byte codec (the default) compresses source code 15–20%
> smaller; `--structural` runs about twice as fast and still beats `zstd -19` and `xz -9e` (see
> [benchmarks](#source-code-corpora-toolsbenchmarkpy)). `--lang` only changes which words count as
> keywords, which moves the size by less than 0.1%: the codec's contexts key on the tokens themselves.

Recover a damaged archive. This is real output:

```console
$ hce -d --recovery skip_and_fill --fill 0x3F --report gaps.jsonl -i damaged.hce -o recovered.log
hce: warning: damaged.hce: recovered with data loss: 1 gap(s), 495207 bytes lost; see --report
$ echo $?
3
$ cat gaps.jsonl
{"stream":0,"out_offset":524288,"length":495207,"length_known":true,"in_offset":29733,"cause":"payload_corrupt"}
```

> [!TIP]
> The frame is the unit of recovery. Damage loses the rest of its frame (STORED blocks are salvaged).
> A smaller `--frame` bounds the damage radius at some cost in ratio.

| Exit | Meaning |
|:---:|---|
| 0 | Success; with `-d` or `-t`, every output byte was verified |
| 1 | Usage or I/O error, interruption, `--expect-size` mismatch |
| 2 | Corrupt input (strict mode stops at the first damage) |
| 3 | Finished with data loss; a recovery mode reported gaps |
| 4 | An R19 resource limit was exceeded |
| 5 | Unsupported format version, feature or codec |

<details>
<summary><b>Full CLI reference</b> (<code>hce --help</code>)</summary>

```text
Usage: hce [MODE] [OPTIONS] [-i INPUT] [-o OUTPUT]

Modes:
  -z, --compress          compress (default)
  -d, --decompress        decompress (either format, recognized by its first bytes)
  -t, --test              decompress and verify, discarding the output

Input and output:
  -i, --input PATH        input file, or - for stdin (default: stdin)
  -o, --output PATH       output file, or - for stdout (default: stdout);
                          files appear under their name only once complete
  -f, --force             overwrite existing files; allow terminal I/O
      --keep-partial      on failure, keep the partial output file
      --expect-size SIZE  fail unless the uncompressed size is exactly SIZE

Compression:
  -l, --level N           effort 0..4 (default 2); model memory per thread
                          21M, 38M, 40M, 76M, 149M; decoding needs the same
      --window SIZE       history window, power of two 64K..1G (default 8M)
      --block SIZE        block size, power of two 1K..8M (default 256K)
      --frame SIZE        frame length / sync interval, block..1G (default 8M)
      --no-content-size   do not record the input size in the header
  -T, --threads N         worker threads (default: one per CPU)
      --structural        two-pass structural codec for source code: tokens,
                          a frozen global model, token-level context mixing
      --lang LANG         keyword table for --structural: auto (default: from
                          the input's extension), generic, c, cpp, python,
                          javascript, typescript, java, go, rust
      --spool-dir DIR     where --structural keeps its token spool (default:
                          next to the output file, else the temporary directory)

Decompression:
      --recovery MODE     strict | skip | skip_and_fill (default strict)
      --fill BYTE         fill byte for skip_and_fill, 0..255 or 0xNN (default 0)
      --report PATH       write every lost range as a JSON line
      --limits PROFILE    trusted | untrusted (default trusted)
      --max-memory SIZE   memory cap; compression sheds threads to meet it
      --max-window SIZE   largest accepted history window
      --max-block SIZE    largest accepted block
      --max-level N       highest accepted level
      --max-ratio N       expansion budget: output <= N * input + burst (0 = off)
      --ratio-burst SIZE  burst depth of the expansion budget (default 64M)
      --max-output SIZE   largest total output
                          Structural archives are verified as a whole: they take
                          --max-memory, but not --recovery, --report or the
                          other limits.

Reporting:
      --stats[=FORMAT]    integer telemetry on stderr: text (default) or json
  -v, --verbose           a readable summary on stderr: sizes, ratio, time, speed
  -q, --quiet             suppress warnings and error messages
  -h, --help              show this help
  -V, --version           show version information

SIZE takes an optional K, M, G or T suffix (powers of 1024).

Environment:
  HCE_AIO=io_uring|threads        I/O backend of the --structural token spool
  HCE_SIMD=scalar|sse2|avx2|avx512|neon
                                  mixer kernels (default: avx2, else sse2 or neon)

Exit status: 0 success (with -d/-t: every byte verified), 1 usage or I/O error,
2 corrupt input, 3 finished with data loss, 4 resource limit exceeded,
5 unsupported version or feature.
```

</details>

### Embed the library

Link the `hce_pipeline` target (it pulls in `hce_codec`) and stream through `hce::InputStream` /
`hce::OutputStream`. The example below uses the in-memory implementations:

```cpp
#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"

#include <vector>

bool roundtrip(const std::vector<std::byte>& data) {
  // Compress: level 2, four workers; the stream is identical for any thread count.
  hce::EncodeOptions eo;
  eo.level = 2;
  eo.threads = 4;
  hce::MemoryInput src(data);
  hce::MemoryOutput packed;
  if (hce::encode_parallel<hce::NeuralCMCodec>(src, packed, eo).status != hc::Errc::ok) return false;

  // Decompress as untrusted input: R19 limits, recover past damage, count gaps.
  hce::DecodeOptions dopt;
  dopt.limits = hc::DecodeLimits::untrusted();
  dopt.recovery = hc::Recovery::skip_and_fill;
  dopt.threads = 4;
  hce::MemoryInput in(packed.data);
  hce::MemoryOutput out;
  const hce::DecodeResult r = hce::decode_stream<hce::NeuralCMCodec>(in, out, dopt);
  return r.status == hc::Errc::ok && r.stats.gaps == 0 && out.data == data;
}
```

```cmake
set(HCE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/hce)
target_link_libraries(my_app PRIVATE hce_pipeline)
```

## Python package

The `hce` Python package wraps the engine through [pybind11](https://pybind11.readthedocs.io) bindings
([`bindings/hce_python.cpp`](bindings/hce_python.cpp)) and a typed API ([`python/hce/`](python/hce/__init__.py)).
It is built by the same CMake project, through [scikit-build-core](https://scikit-build-core.readthedocs.io),
and a wheel also installs the `hce` command. Python 3.10–3.13 on Linux is tested.

```python
import hce

packed = hce.compress(data, level=2, threads=4)          # byte codec; identical output for any thread count
packed = hce.compress(src, structural=True, language="python")   # structural codec for source code
data = hce.decompress(packed, max_output=64 << 20)         # either format; bound the output on untrusted input
stats = hce.verify(packed)                                 # Stats(format, input_bytes, output_bytes, ..., verified=True)

hce.compress_file("big.log", "big.log.hce", threads=8)    # files go through the io_uring aio layer
hce.decompress_file("big.log.hce", "big.log")             # on failure the output file is removed

try:
    hce.decompress(untrusted, max_output=1 << 30)
except hce.CorruptInputError as e:                         # also a ValueError
    print(e.code)                                          # "payload_corrupt", "truncated", "bad_magic", ...
except hce.LimitExceededError:                             # max_output / max_memory
    ...
```

Every function decodes completely and checks every CRC before it returns, releases the GIL while it works,
and raises an `HceError` subclass rather than returning damaged bytes: `CorruptInputError`,
`LimitExceededError`, `UnsupportedFormatError`, or `HceIOError` (an `OSError` with `errno`). Bad arguments
raise `ValueError` or `TypeError`. `hce.tokenize()` exposes the structural tokenizer, and `hce.info()`
reports the version, codec IDs, SIMD kernels and I/O backend.

### Set up a development environment

Every Python dependency is pinned to an exact version: the direct ones in
[`pyproject.toml`](pyproject.toml), every transitive one in [`constraints.txt`](constraints.txt), and the
whole conda toolchain in [`environment.yml`](environment.yml). Pick one of three routes.

**A. pip and a virtual environment** (system compiler)

1. Install a C++20 compiler, Ninja and, for the io_uring backend, liburing:
   ```bash
   sudo apt install build-essential ninja-build python3-dev python3-venv liburing-dev pkg-config
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv && . .venv/bin/activate
   ```
3. Build and install the package with its test tools. `PIP_CONSTRAINT` pins every package, including
   those of the isolated build environment (scikit-build-core, pybind11, and CMake if yours is older than 3.25):
   ```bash
   PIP_CONSTRAINT=constraints.txt python -m pip install ".[test]"     # or ".[dev]" for build and pyflakes too
   ```
4. Check it: `python -c "import hce; print(hce.info())"` and `hce --version`.

**B. conda** (compiler, CMake, liburing and Python packages from conda-forge)

1. Create and activate the environment (`mamba` or `micromamba` work the same):
   ```bash
   conda env create -f environment.yml
   conda activate hce
   ```
2. Install the package from the checkout, using the environment's packages only:
   ```bash
   python -m pip install --no-deps --no-build-isolation -e .
   ```
   The editable install rebuilds the extension on the next `pip install`, not on import.

**C. CMake and CTest** (no installation; with route A's or B's packages)

```bash
cmake --preset python             # Release + HCE_BUILD_PYTHON=ON
cmake --build --preset python     # stages the package in build/python/python/hce
ctest --preset python             # the six C++ suites, then pytest against the staged package
```

CMake finds the first Python on `PATH`; pass `-DPython_EXECUTABLE=/path/to/python` to choose another.
The pytest suite is registered with CTest only when that Python can import pytest and Hypothesis.

### Run the test suite

```bash
python -m pytest                                  # the installed package (routes A and B)
PYTHONPATH=build/python/python python -m pytest   # the CMake build tree (route C)
python -m pytest -m "not slow"                    # skip the largest inputs
python -m pytest -k golden -v                     # one area
HYPOTHESIS_PROFILE=thorough python -m pytest      # 1,000 random examples per property instead of 60 fixed ones
```

The suite ([`tests/python/`](tests/python)) runs in about 75 seconds on four cores. Warnings are errors,
markers are strict, and every test has a 10-minute timeout. Tests marked `cli` compare the bindings with
the `hce` command: they look for `HCE_CLI`, then the interpreter's scripts directory, `PATH` and
`build/release/hce`, and are skipped when none exists.

| Module | What it checks |
|---|---|
| `test_api.py` | Exports, stubs and `py.typed`, `info()`, `Stats`, format detection; the version agrees across the module, `pyproject.toml` and `CMakeLists.txt` |
| `test_roundtrip.py` | Both codecs × five corpora × five levels × nine languages; output independent of thread count; buffer types; concurrent calls from threads; Hypothesis properties on arbitrary bytes and text |
| `test_errors.py` | Argument validation; every truncation, trailing bytes, sampled single-bit flips and random damage must raise; foreign formats; `max_output` and `max_memory` limits; the exception hierarchy and its `code`, `stage` and `errno` |
| `test_golden.py` | Fourteen golden archives (five per byte-codec profile, four of them non-empty for profile 2, and five structural): each decompresses and verifies to its regenerated input, and re-encoding reproduces it byte for byte with one and three threads |
| `test_files.py` | `compress_file` / `decompress_file`: language by extension, Unicode paths, replacement of existing files, no output left behind on failure, OS errors with `errno` |
| `test_tokenize.py` | Lossless tokenization in every language, keyword tables, line numbers and token IDs |
| `test_cli.py` | The bindings and the CLI write identical archives, decode each other's, and agree on exit status and exception for corrupt, limited and unsupported input |
| `test_benchmark_tool.py` | `tools/benchmark.py`: corpus assembly, argument errors, an end-to-end run with Markdown and CSV output |

### Golden vectors

[`tests/python/golden/`](tests/python/golden) holds fourteen archives with a `manifest.json` of their options,
sizes and SHA-256 digests; their inputs come from the deterministic generators in
[`tests/python/testdata.py`](tests/python/testdata.py). Under R17 they are never edited:

```bash
tests/python/golden/regenerate.py --check    # exit 1 if the encoder no longer reproduces every vector
tests/python/golden/regenerate.py            # add vectors newly listed in VECTORS; refuses to change recorded ones
```

A change to any coded bit ships as a new codec profile with new vectors; `--force` (rewrite everything) is
reserved for that.

### Updating the pinned versions

1. Edit the versions in `pyproject.toml` (direct dependencies) and `environment.yml`.
2. Resolve the transitive set in a fresh environment and copy every version into `constraints.txt`:
   ```bash
   python3 -m venv /tmp/pins && /tmp/pins/bin/pip install "scikit-build-core==X" "pybind11==Y" ".[dev]"
   /tmp/pins/bin/pip freeze --exclude hce
   ```
   Keep the `python_version < "3.11"` markers for the backports Python 3.10 needs.
3. Run routes A and B and the full suite; CI repeats them on Python 3.10 and 3.13 and with the conda environment.

## Tooling and automation

### Benchmark script

[`tools/benchmark.py`](tools/benchmark.py) (Python 3.8+, standard library only) compresses each corpus
with `hce --structural`, `hce`, `zstd -19` and `xz -9e`, decompresses every result and compares it with
the original byte for byte, and writes Markdown and CSV tables of size and speed.

```bash
tools/benchmark.py                                   # every built-in corpus, every tool found
tools/benchmark.py --corpus hce-repo --runs 3 --md bench.md --csv bench.csv
tools/benchmark.py --corpus logs=/var/log/syslog --codecs ncm,zstd --threads 8
```

| Option | Default | Meaning |
|---|---|---|
| `--corpus NAME[=PATH]` | all built-ins found | `python-stdlib`, `cxx-headers`, `hce-repo`, or any file or directory; repeatable |
| `--codecs LIST` | `structural,ncm,zstd,xz` | Subset to run; missing tools are skipped unless named explicitly |
| `--hce PATH` | `build/release/hce` | The binary under test |
| `--level N`, `--threads N` | `2`, all CPUs | hce level; threads for every tool |
| `--runs N` | `1` | Repetitions; the fastest time is reported |
| `--max-bytes N` | none | Cut every corpus, for quick runs |
| `--md FILE`, `--csv FILE` | none | Also write the report to files (Markdown always goes to stdout) |

Corpora are concatenations of their files in sorted order, so every run measures the same bytes. Exit
status: 0 when every round trip verified, 1 when a command failed or an output differed, 2 for usage errors.

### Profile-guided build

[`tools/pgo.sh`](tools/pgo.sh) builds an instrumented `hce` (`HCE_PGO=GENERATE`), trains it by compressing
and decompressing a corpus with both codecs at levels 0, 2 and 4 (default: this repository's C++ sources,
every round trip checked), then rebuilds with the profiles (`HCE_PGO=USE`). GCC needs nothing else;
with `CXX=clang++` it also needs `llvm-profdata`. Normal builds and CI do not use PGO.

```bash
tools/pgo.sh                          # build/pgo, trained on the repository's sources
tools/pgo.sh build/pgo-logs app.log   # train on your own data instead
```

On the benchmark VM a GCC profile-guided build decodes about 15% faster than the plain release build and
encodes at the same speed. Streams are byte-identical to every other build.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request:

| Job | What it runs |
|---|---|
| `linux (preset)` | `release`, `clang-release`, `gcc-asan`, `clang-asan`, `gcc-tsan`, `clang-tsan`: build with `-Werror`, then all six suites; io_uring compiled in (`HCE_AIO_URING=ON`) |
| `aarch64 (QEMU)` | The `aarch64-qemu` preset: a static cross build, every suite under `qemu-aarch64 -cpu max` |
| `windows (MinGW-w64 + Wine)` | The `mingw-wine` preset: a static Windows build, five suites under Wine |
| `benchmark (smoke)` | `tools/benchmark.py` on the three corpora; tables in the job summary and as an artifact |
| `python (3.10, 3.13)` | An sdist, then a wheel built from it with `-Werror`, installed with the pinned test tools; pyflakes, `regenerate.py --check`, and pytest against the installed wheel and its `hce` command. The sdist and wheel are kept as artifacts |
| `conda` | `environment.yml` as written: `ctest --preset python` with `-Werror`, then an editable install and pytest |

Failed test jobs upload `LastTest.log`. Each C++ job is a preset, so `cmake --preset P && cmake --build
--preset P && ctest --preset P` reproduces it locally; the Python jobs follow
[Set up a development environment](#set-up-a-development-environment).

---

## Wire format

<details>
<summary><b>Container v3 FRAMED item layout</b></summary>

| Item | Size | Fields |
|---|---:|---|
| Stream header | 32 B | Magic, version 3, feature bits, level, `window_log`, `block_log`, codec ID (`prior_id`), content size + 1 (0 = unknown), stream ID, CRC-32C |
| Frame header | 40 B | 8-byte sync marker, stream ID, codec ID, level, `window_log`, `block_log`, frame index, output offset, CRC-32C |
| Block header | 20 B | Tag (CM, STORED or ancillary), uncompressed size (u24), payload size (u24), payload CRC-32C, content CRC-32C, header CRC-32C |
| END trailer | 32 B | Total output, frame count, CRC-32C over every block checksum in order, seek-index back-pointer, CRC-32C |

The first byte decides the item kind: the marker byte for a frame header, `0x07` for END, anything else
for a block. Block tags can never equal the marker byte. All integers are little-endian.

</details>

---

## Repository layout

```text
hc_framed.hpp                     Container v3 FRAMED, header-only; includes the layers under hc/:
hc/                                 base types, CRC-32C, wire format, window and budget (R19, R21),
                                    arenas, interfaces, the push decoder (R18-R20), the framed writer
codec/neural_cm_codec.{hpp,cpp}   Neural-symbolic CM codec (BlockCodec + FrameEncoder): arena, encoder policy
codec/ncm_model.hpp               Profile 1's model: contexts, match model, mixer, APMs, the bit and byte steps
codec/ncm_model2.hpp              Profile 2's model: more experts, structure models, two match models, 48-input mixer
codec/ncm_run.{hpp,cpp}           Block loops, compiled once per SIMD path and profile with the kernels inlined
codec/ncm_pipe.{hpp,cpp}          The encoder's two-stage pipeline (profile 1): helper thread and record ring
codec/ncm_coder.hpp               Binary arithmetic coder and history views
codec/ncm_simd.hpp                Exact mixer kernels as inline policies: scalar, SSE2, AVX2, AVX-512, NEON
codec/ncm_kernels.{hpp,cpp}       The kernel table and run-time path selection (tests, diagnostics)
codec/ncm_tables.{hpp,cpp}        Fixed-point tables generated at compile time
encoder/parallel.{hpp,cpp}        Multi-threaded encoder: T workers, T+2 slots, ordered writer, pipelines
decoder/pipeline.{hpp,cpp}        Multi-threaded decoder: splitter, workers, ordered writer
io/stream.hpp                     Stream interfaces and in-memory implementations
aio/                              Asynchronous file I/O: io_uring and worker-thread backends, stream adapters
ultracodec/                       The 4-tier ultracodec engine (see ultracodec/README.md)
structural/                       Structural codec: tokenizer, frozen global model, token-level CM, HCS archives
cli/hce.cpp, cli/posix_io.hpp     The hce command-line tool
bindings/hce_python.cpp           pybind11 bindings: the hce._hce extension module
python/hce/                       The typed Python package (__init__.py, _hce.pyi stubs, py.typed)
test_hc_framed.cpp                Container suite with mutation fuzzing
tests/                            Codec, pipeline, aio, structural and CLI suites; legacy-stream fixture
tests/python/                     pytest suite; golden/ holds the frozen archives and regenerate.py
tools/benchmark.py                Multi-corpus benchmark against zstd and xz, Markdown and CSV output
tools/pgo.sh                      Profile-guided build: instrument, train on a corpus, rebuild
pyproject.toml, constraints.txt   Python packaging (scikit-build-core) and the exact pins of every dependency
environment.yml                   Pinned conda-forge development environment
cmake/aarch64-linux-gnu.cmake     AArch64 cross toolchain; tests run under QEMU
cmake/x86_64-w64-mingw32.cmake    Windows cross toolchain (MinGW-w64); tests run under Wine
.github/workflows/ci.yml          CI: the preset matrix, cross targets, the benchmark, the Python package
```

---

## Trade-offs and current limitations

- **Symmetric speed.** About 0.5–1.5 MB/s per core in both directions (see
  [benchmarks](#performance-and-benchmarks)).
- **Memory.** Each worker needs 21–149 MiB of model memory with profile 1 and 25–157 MiB with profile 2,
  depending on level, plus its window. A decoder sizes its workers for profile 2, the larger one.
- **Profile 2 speed.** Profile 2 does about twice the work per bit and runs at roughly 55% of profile 1's
  speed on one thread. Frames still compress and decompress in parallel, and `--codec-profile 1` keeps the
  faster model.
- **Decoder planning.** The decoder sizes every worker for the largest stream its limits accept, so the
  default 1 GiB `trusted` cap allows 3 workers. Tighter `--max-window` and `--max-level` values, or a larger
  `--max-memory`, allow more.
- **R12 consequence.** Incompressible data is stored and never enters the models, so a repeated
  incompressible region is not deduplicated.
- **Fuzzing.** Fuzzing is randomized and runs inside the test suites (mutation and differential). There is
  no coverage-guided harness (libFuzzer/OSS-Fuzz) yet.
- **Platforms.** Linux x86-64 and AArch64 are tested. Windows builds (MinGW-w64) pass the library suites
  under Wine; native Windows (MSVC) and macOS have not been validated, and the CLI requires POSIX.
- **Structural codec.** `--structural` is faster but compresses source code 15–20% larger than the byte
  codec: its dictionary of distinct tokens costs about a third of every archive.
- **Format.** Profile 1 is frozen by golden vectors, but the 4-byte container magic is still provisional.

## Roadmap

- Coverage-guided fuzz targets for the container and codec decoders (libFuzzer, OSS-Fuzz).
- Decoder worker planning driven by the stream header rather than the limits' worst case.
- Seek-index ancillary blocks for random access; the container already reserves them.
- A final container magic number, then a formal format specification document.

---

## Contributing

Changes are welcome. The bar matches the guarantees above:

1. **Zero warnings** under the project's `-Werror` flag set, on GCC and Clang.
2. **All nine configurations pass:** `release`, `clang-release`, `gcc-asan`, `clang-asan`, `gcc-tsan`,
   `clang-tsan`, `aarch64-qemu`, `mingw-wine` and `python` (with the pytest suite). CI checks them, and
   the Python package on 3.10 and 3.13, on every pull request.
3. **No floating point in the codec**, and every kernel path stays bit-exact with the scalar reference.
4. **R17:** golden streams are never edited (`tests/python/golden/regenerate.py --check` must pass). A
   change to any coded bit ships as a new codec profile with new golden vectors.
5. **Python changes** keep `python -m pyflakes python/hce tests/python tools` clean and the
   `_hce.pyi` stubs in step with the bindings.

Please report suspected security issues privately, for example through the repository's private
vulnerability reporting, rather than in a public issue.

## Acknowledgments

HCE's modeling approach follows the context-mixing tradition of the PAQ and lpaq family of compressors.
Its fixed-point formats, tables, kernels and container are independent implementations.

## License

Copyright © 2026 Aung Nyi Nyi Thant.

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for the full text.

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
