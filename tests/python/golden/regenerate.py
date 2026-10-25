#!/usr/bin/env python3
"""Writes the golden vectors that tests/python/test_golden.py checks.

Each vector is a deterministic input (tests/python/testdata.py), the options it
is compressed with, and the exact compressed bytes, stored next to this script
and listed with their SHA-256 in manifest.json.

The vectors freeze the formats (rule R17): a change that alters any coded bit
must ship as a new codec profile, never as edited golden files. So this script
refuses to change a recorded vector unless --force is given, and reports which
ones would change. Typical uses:

  tests/python/golden/regenerate.py            # add vectors new in VECTORS; verify the rest
  tests/python/golden/regenerate.py --check    # change nothing; exit 1 on any difference
  tests/python/golden/regenerate.py --force    # a new profile: rewrite everything

It needs the hce package importable (installed, or PYTHONPATH=<build>/python).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import hce  # noqa: E402
import testdata  # noqa: E402

MANIFEST = HERE / "manifest.json"

# name: (generator, seed, size, compress options)
VECTORS = {
    "framed-source-l2": ("source", 1, 32768, {"level": 2}),
    "framed-log-l0": ("log", 2, 32768, {"level": 0}),
    "framed-text-l4": ("text", 3, 16384, {"level": 4}),
    "framed-binary-l1": ("binary", 4, 8192, {"level": 1}),
    "framed-empty": ("empty", 0, 0, {"level": 2}),
    "structural-source-python": ("source", 5, 32768, {"level": 2, "structural": True, "language": "python"}),
    "structural-source-generic-l1": ("source", 6, 16384, {"level": 1, "structural": True, "language": "generic"}),
    "structural-log-generic": ("log", 7, 16384, {"level": 2, "structural": True, "language": "generic"}),
    "structural-binary": ("binary", 8, 4096, {"level": 2, "structural": True, "language": "generic"}),
    "structural-empty": ("empty", 0, 0, {"level": 2, "structural": True, "language": "generic"}),
}


def build(name: str) -> dict:
    kind, seed, size, opts = VECTORS[name]
    data = testdata.generate(kind, seed, size)
    packed = hce.compress(data, **opts)
    ext = ".hcs" if opts.get("structural") else ".hce"
    return {
        "name": name,
        "file": name + ext,
        "input": {"generator": kind, "seed": seed, "size": size, "sha256": testdata.sha256(data)},
        "options": opts,
        "compressed_size": len(packed),
        "compressed_sha256": testdata.sha256(packed),
        "format": "structural" if opts.get("structural") else "framed",
        "_bytes": packed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="change nothing; exit 1 when a vector differs")
    mode.add_argument("--force", action="store_true", help="rewrite every vector (a new codec profile)")
    args = ap.parse_args()

    old = {v["name"]: v for v in json.loads(MANIFEST.read_text())["vectors"]} if MANIFEST.exists() else {}
    fresh = {name: build(name) for name in VECTORS}
    changed = [n for n in fresh if n in old and old[n]["compressed_sha256"] != fresh[n]["compressed_sha256"]]
    added = [n for n in fresh if n not in old]
    for n in changed:
        print(f"CHANGED {n}: {old[n]['compressed_sha256'][:16]}... -> {fresh[n]['compressed_sha256'][:16]}...")
    for n in added:
        print(f"new     {n}: {fresh[n]['compressed_size']} bytes")
    if args.check:
        return 1 if changed or added else 0
    if changed and not args.force:
        print("refusing to change recorded vectors (R17): a new profile needs --force and new golden files")
        return 1

    vectors = []
    for name, v in fresh.items():
        (HERE / v["file"]).write_bytes(v.pop("_bytes"))
        vectors.append(v)
    MANIFEST.write_text(json.dumps({"hce_version": hce.__version__, "vectors": vectors}, indent=2) + "\n")
    print(f"wrote {len(vectors)} vectors to {MANIFEST.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
