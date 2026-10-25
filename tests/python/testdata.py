"""Deterministic inputs for the tests and the golden vectors.

Every generator is a pure function of (seed, size) built on random.Random,
whose integer-seeded sequence is stable across Python versions, so the golden
manifest can record the SHA-256 of each input and catch any drift.
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable, Dict

_IDENTS = ["request", "response", "items", "total", "index", "buffer", "config", "result", "value", "count",
           "handler", "session", "payload", "offset", "limit", "token", "cache", "stream", "frame", "block"]
_WORDS = ["the", "engine", "compresses", "every", "frame", "with", "context", "mixing", "and", "verifies",
          "each", "block", "before", "it", "writes", "output", "to", "disk", "while", "workers", "share", "a",
          "frozen", "global", "model"]


def source(seed: int, size: int) -> bytes:
    """Python-like source code: functions, branches, calls, comments, strings."""
    rng = random.Random(seed)
    out = []
    n = 0
    while n < size:
        name = rng.choice(_IDENTS)
        args = ", ".join(rng.sample(_IDENTS, rng.randrange(1, 4)))
        lines = [f"def {name}_{rng.randrange(100)}({args}):",
                 f'    """{" ".join(rng.choice(_WORDS) for _ in range(rng.randrange(4, 10)))}."""']
        for _ in range(rng.randrange(2, 7)):
            a, b = rng.choice(_IDENTS), rng.choice(_IDENTS)
            kind = rng.randrange(4)
            if kind == 0:
                lines.append(f"    if {a} > {rng.randrange(1000)}:\n        return {b}[{rng.randrange(8)}]")
            elif kind == 1:
                lines.append(f"    {a} = {b}.get('{rng.choice(_IDENTS)}', {rng.randrange(64)})")
            elif kind == 2:
                lines.append(f"    # {' '.join(rng.choice(_WORDS) for _ in range(rng.randrange(3, 9)))}")
            else:
                lines.append(f"    for {a} in range({rng.randrange(1, 32)}):\n        {b} += {a} * {rng.randrange(9)}")
        lines.append(f"    return {rng.choice(_IDENTS)}\n\n")
        chunk = "\n".join(lines).encode()
        out.append(chunk)
        n += len(chunk)
    return b"".join(out)[:size]


def log(seed: int, size: int) -> bytes:
    """HTTP-style access log lines."""
    rng = random.Random(seed)
    out = []
    n = 0
    i = 0
    while n < size:
        line = (f"2026-10-25T{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}:{i % 60:02d}Z host-{rng.randrange(8)} "
                f"GET /api/v1/{rng.choice(_IDENTS)}/{rng.randrange(100000)} status={200 if rng.randrange(13) else 503} "
                f"bytes={rng.randrange(1, 65536)} ms={rng.randrange(1, 900)}\n").encode()
        out.append(line)
        n += len(line)
        i += 1
    return b"".join(out)[:size]


def binary(seed: int, size: int) -> bytes:
    """Incompressible bytes: the codecs must store them, not expand them much."""
    return random.Random(seed).randbytes(size)


def text(seed: int, size: int) -> bytes:
    """Prose-like sentences."""
    rng = random.Random(seed)
    out = []
    n = 0
    while n < size:
        s = " ".join(rng.choice(_WORDS) for _ in range(rng.randrange(5, 16)))
        chunk = (s[0].upper() + s[1:] + ". ").encode()
        out.append(chunk)
        n += len(chunk)
    return b"".join(out)[:size]


def empty(seed: int, size: int) -> bytes:
    return b""


GENERATORS: Dict[str, Callable[[int, int], bytes]] = {
    "source": source,
    "log": log,
    "binary": binary,
    "text": text,
    "empty": empty,
}


def generate(kind: str, seed: int, size: int) -> bytes:
    return GENERATORS[kind](seed, size)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
