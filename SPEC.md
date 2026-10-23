# HCE format v1: normative specification

This document, together with the conformance vectors in `conformance/golden_v1.txt`, defines the
bitstream. If a behaviour is not described here, it must not affect the output.

Conventions: all multi-byte fields are little-endian. All arithmetic is on integers. `>>` on a signed
value is an arithmetic shift (floor division by a power of two). `mix64` is the SplitMix64 finalizer
(`core/kernels/hash.hpp`), and `crc32c` is CRC-32C (reflected polynomial `0x82F63B78`, initial value
and final XOR `0xFFFFFFFF`).

## 1. Container

### Header (28 bytes)

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `48 43 45 1A` (`"HCE\x1A"`) |
| 4 | 1 | format version, `1` |
| 5 | 1 | profile id (section 2) |
| 6 | 2 | flags, must be `0` |
| 8 | 4 | model id, must equal `model_id(profile)` |
| 12 | 8 | original size `N` |
| 20 | 4 | `crc32c` of the original data |
| 24 | 4 | `crc32c` of header bytes `0..23` |

### Frames

Frames follow until their raw sizes add up to `N`. Nothing may follow the last frame.

| Size | Field |
|---|---|
| 1 | type: `0` = STORED, `1` = MODELED (profiles 1–2), `2` = HYBRID (profiles 3–4) |
| 4 | raw size, `1..frame_size` |
| 4 | payload size |
| 4 | `crc32c` of the frame's raw bytes |
| n | payload |

- **STORED.** The payload is the raw bytes, so payload size equals raw size. In a modeled profile, the
  model then *observes* each byte (section 4.6): its history advances but nothing is learned.
- **MODELED.** The payload is range-coded (section 3), with `4 <= payload size < raw size`. Each byte
  is coded as 8 binary decisions, most significant bit first, each with probability `model.predict()`
  followed by `model.update(bit)`.
- The model state carries over from frame to frame. The encoder may choose either frame type; the
  reference encoder codes a frame as MODELED and keeps it only if the payload is smaller than the raw
  bytes.

A decoder must reject the stream if any of the following hold:
- bad magic, header CRC, version or flags;
- unknown profile, or a model id that does not match;
- `N` above the caller's limit;
- an invalid frame size or type;
- a modeled payload that is not consumed exactly;
- a frame or content CRC mismatch;
- trailing bytes after the last frame.

## 2. Profiles

| id | name | model | hash table | frame size |
|---|---|---|---|---|
| 0 | `store` | none | none | 1 MiB |
| 1 | `cm6-1m` | orders {0,1,2,3,4,6}, `mixer_lr` 7, `statemap_limit` 255 | 2^16 slots (1 MiB) | 64 KiB |
| 2 | `cm6-64m` | same as profile 1 | 2^22 slots (64 MiB) | 1 MiB |
| 3 | `hybrid-1m` | profile 1 + section 6 | 2^16 slots (1 MiB) | 64 KiB |
| 4 | `hybrid-64m` | profile 2 + section 6 | 2^22 slots (64 MiB) | 1 MiB |

`model_id` is `crc32c("hce-store-1")` for profile 0. For modeled profiles it is `model_fingerprint`:
a CRC-32C over the configuration, the arithmetic version, model constants, hash probes and every
generated table (`core/kernels/cm_model.cpp`).

Shipped profiles are frozen. Any change to a profile or to the arithmetic needs a new profile id.

## 3. Binary range coder

The coder is LZMA-style with carry propagation. State: `low` (64-bit, 33 bits used), `range`
(32-bit, starting at `0xFFFFFFFF`), `cache` (starting at `0`), `pending` (starting at `1`).

```text
p1    = clamp(p1, 1, 65535)                     probability that the bit is 1, 16 bits
bound = (range * p1) >> 16                       64-bit product
bit 1: range = bound
bit 0: low += bound; range -= bound
while range < 2^24: range <<= 8; shift_low()
shift_low: if (low mod 2^32) < 0xFF000000 or low >= 2^32:
               emit (cache + carry), then (pending - 1) bytes of (0xFF + carry), carry = low >> 32
               cache = (low >> 24) & 0xFF; pending = 0
           pending += 1; low = (low & 0x00FFFFFF) << 8
finish: 5 x shift_low()
```

The first emitted byte is always `0`, so it is not stored. The decoder initializes `code` from the
first 4 payload bytes, big-endian. To decode a bit: if `code < bound` then the bit is 1 and
`range = bound`; otherwise the bit is 0, `code -= bound` and `range -= bound`. It renormalizes the
same way, shifting in one payload byte per step. A valid payload is consumed exactly: it is
`4 + (number of renormalizations)` bytes long.

## 4. Context-mixing model (profiles 1 and 2)

### 4.1 Tables, generated with integer arithmetic only

- `r = round(2^31 * e^(-1/256))`. It is computed from the Taylor series of `e^(-u)` with `u = 2^-8`,
  terms k = 0..6, in Q62 (`u^k = 2^(62-8k)` exactly; each term is divided by `k!` with truncation),
  then `(q62 + 2^30) >> 31`.
- `E[0] = 2^31`, and `E[x] = (E[x-1] * r + 2^30) >> 31` for `x = 1..3072`.
- `squash[x + 3072]` for `x` in `-3072..3071`: `p = (2^47 + d/2) / d` with `d = 2^31 + E[|x|]`. If
  `x < 0`, `p = 65536 - p`. Finally `p` is clamped to `[1, 65535]`.
- `stretch[i]` for `i` in `0..4095`: the smallest `x` in `-3072..3071` with `squash[x + 3072] >= 16i + 8`.
- `recip[n] = floor(2^17 / (2n + 3))` for `n` in `0..1023`.
- **Bit-history states.** States are pairs `(n0, n1)`. On bit `b`, the count `n_b` becomes
  `min(n_b + 1, 35)`, and the opposite count becomes `n/2 + 1` if it is above 2.
  - The states are those reachable from `(0, 0)`: 254 of them.
  - They are numbered by `n0 + n1`, then by `n1`, so state 0 is `(0, 0)`.
  - `next[s][b]` is the state reached by bit `b`.
  - `p22_init[s] = ((2*n1 + 1) << 22) / (2*(n0 + n1) + 2)`.

### 4.2 Contexts

- `H` is the previous 8 bytes, with the newest in the low byte. It starts at `0`.
- After every byte (decoded or observed), each order `k` gets
  `ctx_k = mix64((H & mask_k) XOR ((k + 1) << 56))`, where `mask_k` keeps the low `8k` bits.

### 4.3 Slots

The hash table holds `2^b` slots of 16 bytes, grouped in buckets of 4 slots (64 bytes). It starts
zeroed. Each slot is one check byte followed by states for nodes 1..15 of a nibble's binary tree.

At bit positions 0 and 4 of each byte, every order looks up
`h = mix64(ctx_k + c0 * 0x9E3779B97F4A7C15)`. Here `c0` is the partial byte with a leading 1: it is
`1` at bit 0, and `1hhhh` at bit 4.
- The bucket is `h >> (64 - (b - 2))`, and the check byte is `h & 0xFF`.
- The first of the bucket's 4 slots whose check byte matches is used.
- Otherwise the slot with the smallest `n0 + n1` of its node-1 state is cleared and takes the check
  byte; ties go to the lowest index.

### 4.4 Prediction

`nib` holds the bits of the current nibble behind a leading 1 (1..15).

```text
s_i = slot_i[nib]                                state of the current node, per order i
x_i = stretch[(StateMap_i[s_i] >> 16) >> 4]      StateMap entry = p22 << 10 | count
x_n = 256                                        bias input
dot = sum_i x_i * w[c0][i]                       64-bit; weights Q16
p   = squash[clamp(dot >> 16, -3071, 3071) + 3072]
```

The initial weights are `16384` for each order and `0` for the bias, in each of the 256 gates (one
per `c0`). The StateMap starts at `p22_init[s] << 10` (count 0).

### 4.5 Update with bit `y`

```text
err = (y << 16) - p;  w[c0][i] = clamp(w + ((x_i * err * lr + 2^19) >> 20), -(2^20 - 1), 2^20 - 1)
StateMap_i[s_i]: n = count; p22 += ((target - p22) * recip[n] + 2^15) >> 16,
                 target = y ? 2^22 - 1 : 0; count = min(n + 1, limit)
slot_i[nib] = next[s_i][y]
c0 = c0*2 + y; nib = nib*2 + y
after bit 4: nib = 1, lookup again (section 4.3)
after bit 8: H = H*256 + byte, c0 = 1, nib = 1, recompute contexts (section 4.2)
```

### 4.6 Observing a byte (STORED frames)

`H = H*256 + byte` and the contexts are recomputed. Tables, StateMaps and weights are untouched.

## 5. Conformance

`conformance/golden_v1.txt` pins, for every profile and corpus vector, the input size and CRC, the
output size and CRC, and an FNV-1a hash over `(p1 << 1 | bit)` of every modeled decision. An
implementation conforms if `hce_conformance --check` passes.

## 6. Hybrid anchors and COPY (profiles 3 and 4)

The container version remains 1. Profiles 0–2, their model ids and their encoded bytes remain
unchanged. Hybrid profiles accept STORED and HYBRID frames only. HYBRID payloads use the same
range coder and the same size/consumption rules as MODELED frames. `frames_modeled` statistics
include HYBRID frames. The default profile remains 2.

### 6.1 Configuration, Gear hash and anchors

The model configuration K pins `gear_seed = 0x4843452D47454152`, `density_bits = 6`, and
`index_bits = 16` (profile 3) or `20` (profile 4).
Starting with `s = gear_seed`, generate the fixed 256-word table in byte order:
`s += 0x9E3779B97F4A7C15; Gear[i] = mix64(s)`, with unsigned 64-bit wrapping arithmetic.
Start `g = 0` and, after every literal, copied or stored byte b, compute `g = (g << 1) + Gear[b]`.
A boundary t (exclusive byte position) is an anchor when all of these hold:
- `t >= 64`;
- `(g & ((1 << density_bits) - 1)) == 0`;
- `t - last_anchor >= 2^(density_bits - 2)`, where `last_anchor` starts at 0. This minimum gap bounds
  the anchor list on adversarial input.

History is the stream itself, so every earlier byte stays addressable.
- **Anchor ids** are ordinals over the whole stream. Id 0 is reserved (position 0, never resolves),
  and each new anchor takes the next id, so ids never expire.
- **Lookup buckets.** A direct-mapped bucket `mix64(g) & ((1 << index_bits) - 1)` holds the latest
  anchor's id and its full 64-bit fingerprint. A later anchor in the same bucket replaces it.
- **At an anchor,** first query the bucket. It matches when it is non-empty and its fingerprint equals
  g. Then append the current anchor and store it in the bucket.

### 6.2 Implicit match model

At a matching anchor, the earlier anchor position becomes the predicted next-byte position.
Keep the consecutive match length when it is already the active predicted position; otherwise
reset it to zero. Before observing a byte, compare it with the predicted history byte, if valid:
save their XOR as the divergence byte; on equality increment the match length (saturating at
`UINT32_MAX`) and predicted position; on inequality deactivate prediction and reset the length.
Anchor lookup and insertion follow this comparison. Stored and copied bytes update this state
identically.

The mixer adds two inputs after its bias, initially weighted 16384, using the same learning rule.
If a history prediction exists and its high bits agree with the partial current byte, input one
is signed toward its next bit with magnitude `256 * min(8, bit_width(match_length | 1))`.
Input two has magnitude 256, signed toward the prediction bit XOR the corresponding bit of the
last divergence byte. Both inputs are zero when there is no valid, prefix-consistent prediction.
An anchor can reactivate prediction after a divergence. The divergence byte initially equals zero.

### 6.3 Token grammar and safety

A token flag is decoded only at a byte boundary where the last observed byte produced a matching
anchor (section 6.1). Its probability is a fixed 256, a COPY prior of 1/256. Everywhere else no flag
is coded and the next byte is a literal. All token fields after the flag are unsigned, most
significant bit first, with probability 32768 for every bit. These control decisions do not train
the context model.

- Flag 0: decode one literal with eight context-model decisions, as in section 4.
- Flag 1: decode three fields:
  - `anchor_id`: `bit_width(anchor_count - 1)` bits, where `anchor_count` includes the reserved id 0;
  - `delta`: 5 bits;
  - `length`: 32 bits.

  Set `source = anchor.position - delta`. Copy `length` bytes from history and observe each byte as
  in section 4.6, including anchor and implicit match updates. Copied bytes do not train the context
  model.

Before reading or writing any COPY byte, reject unless all of these hold:

- `1 <= anchor_id < anchor_count`.
- `delta <= anchor.position`, so the subtraction cannot underflow.
- `8 <= length <= remaining_raw_bytes_in_frame`.
- `length <= t - source`: the entire source already exists, and overlapping expansion is forbidden.

Invalid references, truncation, extra payload bytes and checksum failures throw `FormatError`.

### 6.4 Reference encoder planner and identity

The encoder queries the preceding byte's anchor match and tests source deltas 0..16, comparing
every byte for up to 64 KiB, bounded also by frame end and existing source history. It chooses the
longest verified candidate of at least 8 bytes that passes the cost test; delta ties go to the
smallest delta. This byte comparison is mandatory even on an exact fingerprint match.

For each frame, initialize estimated literal-byte cost C to 2048 (Q8 bits). After every encoded
literal of cost B, set `C = (31*C + B) / 32`. For each literal bit, B adds
an estimate of `-log2(P(observed_bit)) * 256`, calculated by eight normalized integer squarings
(`encoder/planner.cpp`). Emit COPY only when
`length * (C + 2) > (8 + id_bits + 5 + 32 + 8) * 256`, with `id_bits = bit_width(anchor_count - 1)`.
This includes token flag costs (8 bits for COPY, 2 Q8 units per literal, rounded up) and an
eight-bit savings margin. The decoder does not use C.
Frames that do not compress are rolled back and then treated as STORED:
1. Restore the model and lookup-bucket snapshot.
2. Truncate the anchor list to its length at the start of the frame. The list is shared with the
   snapshot, because anchors depend only on content.
3. Observe the frame's bytes as STORED. Probability trace statistics include control decisions, and COPY counts from
discarded trials are discarded too.

The hybrid model fingerprint extends the legacy fingerprint input immediately after the hash
probes with:
- the hybrid arithmetic/token version, 3 (4 bytes);
- index and density bits (one byte each);
- the COPY flag probability (4 bytes) and the delta width (1 byte);
- the Gear seed (8 bytes);
- all 256 Gear words (8 bytes each). Generated baseline tables follow as
before. This pins the table and configuration without changing legacy fingerprints.
