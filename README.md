# Hybrid Compression Engine (hce)

A bit-exact lossless compression engine. A deterministic, integer-only core codes every bit with a
binary range coder, and models supply the probabilities. This repository implements **steps 1–2** of the build
order:
- the container format;
- STORED frames;
- CRC-32C;
- an LZMA-style range coder with 16-bit probabilities;
- an order-N context-mixing baseline;
- content-defined Gear anchors over the whole stream, implicit match prediction and verified long-range COPY;
- pybind11 bindings;
- a conformance harness.

The format is specified normatively in [SPEC.md](SPEC.md).

## Layout

| Path | Contents | Rules |
|---|---|---|
| `core/kernels/` | CRC-32C, generated tables, bit-history states, StateMap, hash table, mixer, CM model | decode path: integer-only, single-threaded, no allocation in per-bit code, `<cstdint>` types |
| `format/` | container (header, frames), profiles, range coder, little-endian I/O | decode path: same rules |
| `encoder/` | integer cost estimation and byte-verified COPY planner | encoder-only, integer arithmetic |
| `bindings/` | pybind11 module `hce` and its Python tests | orchestration, testing, analysis |
| `conformance/` | deterministic corpus, `golden_v1.txt`, `hce_conformance` harness | |
| `tests/` | C++ unit tests (dependency-free harness) | |
| `cmake/` | `CheckDecodePath.cmake`, the decode-path rule check | |

## Build and test

Requires CMake 3.20+ and a C++20 compiler.

```bash
cmake -S . -B build -G Ninja
```

```bash
cmake --build build
```

```bash
ctest --test-dir build --output-on-failure
```

To build the Python module, create a virtual environment with pybind11 and point CMake at it:

```bash
python3 -m venv .venv && .venv/bin/pip install pybind11
```

```bash
cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$PWD/.venv/bin/python
```

Other useful configurations:
- `-DHCE_SANITIZE=address,undefined -DCMAKE_BUILD_TYPE=Debug` runs the suite under ASan and UBSan.
  Any undefined behaviour aborts the test.
- `-DHCE_WERROR=ON` makes warnings fatal. The build is clean under `-Wall -Wextra -Wpedantic
  -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast`.

```python
import hce

blob = hce.compress(data)                    # default profile 2 (cm6-64m)
assert hce.decompress(blob) == data
blob, stats = hce.compress(data, 1, stats=True)   # frames, decisions, probability-trace hash
blob, stats = hce.compress(data, 3, stats=True)   # hybrid: anchors, implicit matches and COPY
hce.profiles(), hce.model_id(2), hce.crc32c(data)
```

`hce.FormatError` is a subclass of `ValueError`. It is raised for any invalid, corrupted or truncated
stream.

## Determinism: how it is enforced

- **No floating point in the decode path.** Every `.cpp` in `core/` and `format/` ends its includes
  with `no_float.hpp` (`#pragma GCC poison float double`). The `decode_path_rules` test also scans
  every source and header for `float`, `double`, `<cmath>`, `<thread>`, `<random>` and `<chrono>`.
  - The tables (squash, stretch, state machine) are generated at startup by integer code. That
    includes `e^(-1/256)`, which comes from an exact Q62 Taylor series.
- **No undefined behaviour.**
  - Signed arithmetic that could overflow goes through `core/kernels/arith.hpp`, which wraps via
    unsigned types. C++20 defines both the wraparound and arithmetic right shifts.
  - Hashes and the coder are unsigned throughout.
  - The full suite passes under ASan and UBSan.
- **Bit-exact byte order.** Serialization is byte-by-byte little-endian; structs are never `memcpy`'d.
- **No hidden inputs.** There is no `std::hash`, no iteration over unordered containers, and no
  `<random>` (its distributions differ between standard libraries). Table sizes come from the profile,
  never from available RAM.
- **Model identity.** The header carries a model id: a CRC over the configuration, the arithmetic
  version and every generated table. A decoder refuses streams written by a different model.
- **Golden vectors.**
  - `conformance/golden_v1.txt` pins output size, output CRC and a hash of every probability for
    each profile and corpus vector. The Debug (`-O0`) and Release (`-O3`) builds reproduce it.
  - Unit tests check that encoder and decoder probability traces are identical.

## Changing the format

Shipped profiles are frozen. To change the model arithmetic or a profile parameter:
1. Add a new profile id. If code-defined behaviour changed, also bump `kModelArithmeticVersion`.
2. Regenerate the vectors with `build/conformance/hce_conformance --regen conformance/golden_v1.txt`.
3. Review the diff: only the new profile's lines may change.

## Baseline numbers

Measured on an Apple M4, single thread, on 8 MiB of *synthetic* data from `conformance/corpus.cpp`.
That data is simple enough for order-N contexts to capture, so it is not representative of real
files. Measure on your own corpus before drawing conclusions.

| Data | hce `cm6-64m` | zstd -19 | xz -9e | bzip2 -9 | gzip -9 |
|---|---|---|---|---|---|
| logs | 748,210 | 1,119,576 | 1,088,612 | 940,609 | 1,307,544 |
| text | 1,678,349 | 2,114,520 | 2,112,604 | 1,892,315 | 2,553,207 |

On this data, hce's output is 0.67× the size of zstd -19's on logs and 0.79× on text. It encodes and
decodes at about 4 MB/s.

## Step 2: anchors and long-range COPY

Profiles 3 (`hybrid-1m`) and 4 (`hybrid-64m`) add the following on top of profiles 1 and 2:
- **Content-defined Gear anchors** over the whole stream.
- **An exact-fingerprint anchor index** of 1 MiB / 16 MiB.
- **Implicit match/divergence inputs** to the mixer.
- **Unlimited COPY reach.** Anchor ids are stream-wide ordinals and history is read from the stream
  buffer itself, so a COPY can reach any earlier anchor regardless of distance.
- **No cost on data without repeats.** The COPY flag is coded only at anchors that matched an earlier
  one, so on such data profiles 3 and 4 produce byte-identical sizes and probability traces to
  profiles 1 and 2.
- **Verified COPY on both sides.**
  - The encoder byte-verifies every COPY and chooses between COPY and literals with an integer
    running cost estimate.
  - The decoder checks the anchor id, the subtraction, the source extent (no overlap) and the frame
    bounds before expanding any COPY.

| `repeats-200k` (text + noise, repeated) | profile 1 | profile 3 | profile 2 | profile 4 |
|---|---|---|---|---|
| bytes | 125,237 | 68,997 | 74,449 | 68,893 |

The existing profiles, their outputs and the default (profile 2) are unchanged. Use profile 3 or 4
explicitly to enable hybrid coding. Python stats include `copies` and `copied_bytes`, and profile
descriptions expose `copy`. The [format specification](SPEC.md#6-hybrid-anchors-and-copy-profiles-3-and-4)
defines the grammar and model identity.

Unit tests cover:
- COPY from more than 2 MiB back;
- id stability across index evictions;
- cost rejection and delta verification;
- malformed and overlapping tokens;
- stored-frame rollback.

## Known limits

- **Encoder snapshot cost.** The encoder copies the model state once per frame (64 MiB for profile 2),
  so it can roll back frames that end up STORED. An undo journal would make this proportional to the
  slots touched.
- **Small inputs.** A modeled profile allocates its full hash table even for tiny inputs, and the
  28-byte header has no compact small-file form yet.
- **COPY search.**
  - The planner examines 16 bytes before each matching anchor and up to 64 KiB of non-overlapping
    source.
  - Direct-mapped index collisions can lose candidates. The reach is unlimited, but the index
    remembers at most 2^index_bits distinct fingerprints.
  - The running cost estimate can miss a better literal/COPY choice. Corpus tuning and trial coding
    remain future work.
- **Anchor memory.** The anchor list keeps 8 bytes per anchor for the whole stream: about N/8 on
  typical data, and at most N/2 on adversarial input thanks to the minimum anchor gap.
- **Compression target.** Step 2 has no demonstrated 50%+ improvement over zstd on real corpora.
- **Not yet built.** APM/SSE, templates, keyed delta and the integer NN expert remain future work.
