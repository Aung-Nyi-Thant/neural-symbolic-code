"""HCE: lossless neuro-symbolic hybrid compression.

Two codecs share one container family:

* the **byte codec** (default): neural-symbolic context mixing over bytes,
  written as Container v3 FRAMED streams;
* the **structural codec** (``structural=True``): source code tokenized,
  a frozen global model and token-level context mixing, written as HCS
  archives.

Decompression recognizes either format by its first bytes. Every function
verifies what it decodes (CRCs and sizes) before returning it.

>>> import hce
>>> packed = hce.compress(b"hello hello hello hello")
>>> hce.decompress(packed)
b'hello hello hello hello'

Errors derive from :class:`HceError` and carry ``code`` (the engine's error
name), ``stage`` and ``errno``. Damaged input raises
:class:`CorruptInputError` (also a ``ValueError``), resource limits raise
:class:`LimitExceededError`, unknown formats or versions
:class:`UnsupportedFormatError`, and file problems :class:`HceIOError` (also an
``OSError``). Bad arguments raise ``ValueError`` or ``TypeError``.

The functions release the GIL while they work, so threads can compress in
parallel; ``threads=N`` also parallelizes one call across frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Union

from . import _hce
from ._hce import (
    CorruptInputError,
    HceError,
    HceIOError,
    LimitExceededError,
    UnsupportedFormatError,
)

__version__: str = _hce.__version__

__all__ = [
    "__version__",
    "DEFAULT_MAX_MEMORY",
    "LANGUAGES",
    "compress",
    "decompress",
    "verify",
    "compress_file",
    "decompress_file",
    "detect_format",
    "tokenize",
    "detokenize",
    "info",
    "Stats",
    "Token",
    "HceError",
    "CorruptInputError",
    "LimitExceededError",
    "UnsupportedFormatError",
    "HceIOError",
]

#: Default decoder memory bound (bytes): window, model and buffers of all workers.
DEFAULT_MAX_MEMORY: int = _hce.DEFAULT_MAX_MEMORY

#: Keyword tables the structural tokenizer knows; ``"auto"`` (files only) picks one by extension.
LANGUAGES: tuple = tuple(_hce.info()["languages"])

BytesLike = Union[bytes, bytearray, memoryview]
PathLike = Union[str, "os.PathLike[str]"]

_COMMON = ("format", "input_bytes", "output_bytes", "threads", "verified")


@dataclass(frozen=True)
class Stats:
    """What one compression, decompression or verification did.

    ``input_bytes`` and ``output_bytes`` are the sizes read and written;
    ``verified`` is True when every output byte was checked against the
    stream's CRCs. ``details`` holds the codec-specific counters (frames and
    blocks for the byte codec; tokens, symbols, dictionary and stream sizes
    and the language for the structural codec).
    """

    format: str
    input_bytes: int
    output_bytes: int
    threads: int
    verified: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Uncompressed size divided by compressed size (0 for empty data)."""
        raw, packed = (self.output_bytes, self.input_bytes) if self.verified else (self.input_bytes, self.output_bytes)
        return raw / packed if packed else 0.0

    @classmethod
    def _from(cls, d: Dict[str, Any]) -> "Stats":
        return cls(**{k: d[k] for k in _COMMON}, details={k: v for k, v in d.items() if k not in _COMMON})


class Token(NamedTuple):
    """One token of the structural tokenizer.

    ``value`` is the exact source bytes, so ``detokenize(tokenize(s)) == s``.
    ``id`` interns (type, value) in order of first appearance within one call.
    """

    type: str
    value: bytes
    line: int
    id: int


def compress(
    data: BytesLike,
    *,
    level: int = 2,
    threads: int = 1,
    structural: bool = False,
    language: str = "generic",
) -> bytes:
    """Compresses ``data``; the output is identical for every ``threads`` value.

    ``level`` (0..4) trades memory for ratio. ``structural=True`` selects the
    structural codec, with ``language`` choosing its keyword table.
    """
    return _hce.compress(data, level=level, threads=threads, structural=structural, language=language)


def decompress(
    data: BytesLike,
    *,
    threads: int = 1,
    max_memory: int = DEFAULT_MAX_MEMORY,
    max_output: Optional[int] = None,
) -> bytes:
    """Decompresses and verifies an hce stream or archive.

    ``max_output`` bounds the decompressed size (``LimitExceededError`` beyond
    it): use it on untrusted input. ``max_memory`` bounds the decoder's memory.
    """
    return _hce.decompress(data, threads=threads, max_memory=max_memory, max_output=max_output)


def verify(data: BytesLike, *, threads: int = 1, max_memory: int = DEFAULT_MAX_MEMORY) -> Stats:
    """Decodes ``data`` without keeping the output; raises unless every byte verifies."""
    return Stats._from(_hce.verify(data, threads=threads, max_memory=max_memory))


def compress_file(
    source: PathLike,
    dest: PathLike,
    *,
    level: int = 2,
    threads: int = 1,
    structural: bool = False,
    language: str = "auto",
) -> Stats:
    """Compresses a regular file into ``dest`` (created or replaced).

    Both files go through the engine's asynchronous I/O layer (io_uring where
    available). ``language="auto"`` picks the keyword table by extension.
    """
    return Stats._from(
        _hce.compress_file(os.fspath(source), os.fspath(dest), level=level, threads=threads,
                           structural=structural, language=language)
    )


def decompress_file(
    source: PathLike, dest: PathLike, *, threads: int = 1, max_memory: int = DEFAULT_MAX_MEMORY
) -> Stats:
    """Decompresses either format into ``dest``; on failure ``dest`` is removed."""
    return Stats._from(_hce.decompress_file(os.fspath(source), os.fspath(dest), threads=threads, max_memory=max_memory))


def detect_format(data: BytesLike) -> Optional[str]:
    """``"framed"`` (byte codec), ``"structural"``, or None, from the first bytes."""
    return _hce.detect_format(data)


def tokenize(source: Union[str, BytesLike], *, language: str = "generic") -> List[Token]:
    """The structural tokenizer: lossless, total, never fails on any input."""
    return [Token(*t) for t in _hce.tokenize(source, language=language)]


def detokenize(tokens: List[Token]) -> bytes:
    """The source the tokens came from."""
    return b"".join(t.value for t in tokens)


def info() -> Dict[str, Any]:
    """Build facts: version, codec IDs, SIMD kernels, I/O backend, languages."""
    return dict(_hce.info())
