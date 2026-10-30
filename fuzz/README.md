# Fuzz targets

Coverage-guided targets for the decoders that read untrusted bytes. Each is
one `LLVMFuzzerTestOneInput()` and runs under libFuzzer, or under
[`replay.cpp`](replay.cpp) in builds without it.

| Target | What it decodes | Seeds |
|---|---|---|
| `fuzz_framed` | FRAMED v3 streams, `decode_stream<NeuralCMCodec>` under the `untrusted` limits; each input as given, then with its header and payload CRCs recomputed, threaded or in a salvage mode | `tests/python/golden/*.hce`, `corpus/framed` |
| `fuzz_structural` | HCS archives, `structural::decompress_stream` under tight limits; as given and with the archive-header and FRAMED CRCs recomputed | `tests/python/golden/*.hcs`, `corpus/structural` |
| `fuzz_codec` | `NeuralCMCodec` (profiles 1 and 2) and `StructuralCMCodec` `decode_block()` directly, levels 0-4, on hostile payloads, without the framing | `corpus/codec` |
| `fuzz_model` | `GlobalModel::parse()` (HGM1 dictionaries) and `deserialize_tokens()` (HST1 spools); a model that parses must survive its own round trip | `corpus/model` |
| `fuzz_ultra` | `ultracodec::decompress()`, Tiers 1-3; as given and with the header and block-table CRCs recomputed | `corpus/ultra` |
| `fuzz_ultra_t4` | the same for Tier 4, whose decoder trains on a built-in primer first (seconds per input under the sanitizers) | `corpus/ultra` |

Content CRCs are never recomputed (that needs the decoded bytes), so a
mutated block still reaches its decoder and is rejected afterwards.

## Build and run

```bash
cmake --preset clang-fuzz            # Clang, HCE_FUZZ=ON, ASan + UBSan
cmake --build --preset clang-fuzz
ctest --preset clang-fuzz -R '^fuzz_' # every target over its seeds and regressions, once

mkdir -p work/framed
build/clang-fuzz/fuzz/fuzz_framed -max_total_time=3600 -rss_limit_mb=3072 -max_len=12000 \
  work/framed tests/python/golden fuzz/corpus/framed
```

Other presets build the same targets with `replay.cpp`, and their `fuzz_*`
ctests replay the same inputs without coverage feedback.

## Seeds and regressions

`corpus/` is written by [`make_seeds.py`](make_seeds.py) from the golden
vectors and the current tools (`PYTHONPATH=build/python/python fuzz/make_seeds.py
--hce build/python/hce --ultra build/python/ultracodec/ultra`). The seeds only
steer the fuzzer; they are not compatibility fixtures.

An input that once crashed, hung or ran out of memory goes into
`regressions/<target>/` with a name saying what it found; every build's
ctest then replays it.

## What a clean run means

A campaign that finds nothing shows only that the inputs it reached did not
trip the sanitizers or the targets' own checks. It is not proof that the
decoders are free of bugs. Report each campaign with its duration, executions
and coverage.
