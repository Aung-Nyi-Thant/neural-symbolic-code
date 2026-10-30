#!/usr/bin/env python3
"""Writes the seed corpora of the fuzz targets that tests/python/golden does not cover.

  fuzz/make_seeds.py [--hce PATH] [--ultra PATH]   # needs hce importable (PYTHONPATH=<build>/python)

corpus/codec  {selector, block size, blocks} inputs of fuzz_codec, cut from the CM blocks
              (4 KiB at most) of corpus/framed and of the archives' token streams
corpus/model  HGM1 images (the golden archives' decoded model sections) and HST1 images
              (hce.tokenize() of small sources), for fuzz_model
corpus/framed small FRAMED streams (1 KiB blocks, 2 KiB frames: many items) of every level and
              profile, written by the hce tool at PATH, for fuzz_framed
corpus/structural  small HCS archives of every level (--hce), for fuzz_structural
corpus/ultra  one small archive per ultracodec tier, written by PATH (the ultra tool)

Seeds only steer the fuzzer: they are produced by the current encoder and are not
compatibility fixtures. fuzz_framed and fuzz_structural also start from tests/python/golden.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import hce

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "tests" / "python" / "golden"
TYPES = {"KEYWORD": 0, "IDENTIFIER": 1, "LITERAL": 2, "OPERATOR": 3, "STRUCTURE": 4, "INDENTATION": 5}
SCM_FAMILY = 0x53434D00
MAGIC = b"\x89HCZ"
SYNC = bytes([0xC3, 0x5A, 0x9E, 0x27, 0xB1, 0x6D, 0x04, 0xF8])


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def cm_blocks(data: bytes, pos: int = 0):
    """(level, prior_id, usize, payload) of each CM block of the FRAMED stream at pos."""
    level = prior = 0
    while pos < len(data):
        if data[pos:pos + 4] == MAGIC:
            level, prior = data[pos + 7], struct.unpack_from("<I", data, pos + 12)[0]
            pos += 32
        elif data[pos:pos + 8] == SYNC:
            pos += 40
        elif data[pos] == 0x07:
            return
        else:
            tag = data[pos]
            usize = int.from_bytes(data[pos + 1:pos + 4], "little")
            csize = int.from_bytes(data[pos + 4:pos + 7], "little")
            if tag == 0 and 0 < usize <= 4096:
                yield level, prior, usize, data[pos + 20:pos + 20 + csize]
            pos += 20 + csize


def codec_seed(level: int, prior: int, blocks) -> bytes:
    structural = (prior & 0xFFFFFF00) == SCM_FAMILY
    sel = (level << 2) | (1 if structural else 0) | (2 if (not structural and prior & 0xFF) == 2 else 0)
    out = bytearray([sel, 2])  # 4 KiB blocks: fuzz_codec takes usize modulo 4096
    for usize, payload in blocks:
        out += struct.pack("<HH", usize - 1, len(payload)) + payload
    return bytes(out)


def hst1(source: str, language: str) -> bytes:
    toks = hce.tokenize(source, language=language)
    out = bytearray(b"HST1" + varint(len(toks)))
    line = 1
    for t in toks:
        out += bytes([TYPES[t.type]]) + varint(len(t.value)) + t.value + varint(zigzag(t.line - line)) + varint(t.id)
        line = t.line
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hce", type=Path, help="the hce tool, for corpus/framed")
    ap.add_argument("--ultra", type=Path, help="the ultra tool, for corpus/ultra")
    args = ap.parse_args()
    codec, model, ultra = HERE / "corpus" / "codec", HERE / "corpus" / "model", HERE / "corpus" / "ultra"
    framed, structural = HERE / "corpus" / "framed", HERE / "corpus" / "structural"
    for d in (codec, model, ultra, framed, structural):
        d.mkdir(parents=True, exist_ok=True)

    if args.hce:
        sys.path.insert(0, str(GOLDEN.parent))
        import testdata

        with tempfile.TemporaryDirectory() as tmp:
            small = ["--block", "1K", "--frame", "2K", "--window", "64K"]
            src, log = Path(tmp) / "src", Path(tmp) / "log"
            src.write_bytes(testdata.source(11, 3000))
            log.write_bytes(testdata.log(12, 3000))
            for level in range(5):
                for profile in (1, 2):
                    out = framed / f"src-l{level}-p{profile}.hce"
                    subprocess.run([str(args.hce), "-z", "-f", "-l", str(level), "--codec-profile", str(profile), *small,
                                    "-i", str(src), "-o", str(out)], check=True)
            subprocess.run([str(args.hce), "-z", "-f", "-l", "1", *small, "-i", str(log), "-o", str(framed / "log-l1.hce")],
                           check=True)
            src.write_bytes(testdata.source(13, 900))  # under 1,024 tokens: token blocks under 4 KiB
            for level in range(5):
                subprocess.run([str(args.hce), "-z", "-f", "--structural", "--lang", "python", "-l", str(level),
                                "-i", str(src), "-o", str(structural / f"src-l{level}.hcs")], check=True)

    for f in sorted(framed.glob("*.hce")):
        blocks = list(cm_blocks(f.read_bytes()))
        if blocks:
            level, prior = blocks[0][0], blocks[0][1]
            (codec / f"{f.stem}.bin").write_bytes(codec_seed(level, prior, [(u, p) for _, _, u, p in blocks[:8]]))
    for f in sorted(GOLDEN.glob("*.hcs")) + sorted(structural.glob("*.hcs")):
        a = f.read_bytes()
        model_bytes = struct.unpack_from("<Q", a, 24)[0]
        image = hce.decompress(a[44:44 + model_bytes])
        (model / f"{f.stem}.hgm1").write_bytes(image)
        blocks = list(cm_blocks(a, 44 + model_bytes))
        if blocks:
            level, prior = blocks[0][0], blocks[0][1]
            (codec / f"{f.stem}.bin").write_bytes(codec_seed(level, prior, [(u, p) for _, _, u, p in blocks[:8]]))
    (model / "python.hst1").write_bytes(hst1("def f(x):\n    return {'a': [x, 1.5]}  # c\n", "python"))
    (model / "c.hst1").write_bytes(hst1("int main(void) {\n\treturn 0x1f;\n}\n", "c"))
    (model / "empty.hst1").write_bytes(hst1("", "generic"))

    if args.ultra:
        src = ("".join(f"line {i * 7919 % 1000} of some repetitive text\n" for i in range(600))).encode()
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.txt"
            inp.write_bytes(src)
            for tier in (1, 2, 3, 4):
                out = ultra / f"tier{tier}.u"
                subprocess.run([str(args.ultra), "c", "-t", str(tier), "-T", "1", str(inp), str(out)], check=True)
            (ultra / "empty.u").unlink(missing_ok=True)
            empty = Path(tmp) / "empty"
            empty.write_bytes(b"")
            subprocess.run([str(args.ultra), "c", "-t", "1", str(empty), str(ultra / "empty.u")], check=True)
    for d in (codec, model, ultra, framed, structural):
        print(f"{d.relative_to(HERE)}: {len(list(d.iterdir()))} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
