#!/usr/bin/env python3
"""Multi-corpus compression benchmark for HCE.

Compresses the same inputs with

  structural  hce --structural    the two-pass structural codec
  ncm         hce                 the byte-level neural CM codec (its default profile)
  zstd        zstd -19
  xz          xz -9e

decompresses every result and compares it with the original byte for byte
after every repetition (one mismatch fails the codec, whatever the other
repetitions did), and writes the sizes and speeds as Markdown, CSV and JSON.

Corpora are built under the work directory, with files in sorted order so
every run sees the same bytes:

  python-stdlib  the .py files of this interpreter's standard library
                 (top level and one directory down)
  cxx-headers    the newest libstdc++ headers under /usr/include/c++
  hce-repo       this repository's C++ sources
  synthetic-*    generated from fixed seeds, identical on every machine:
                 logs (text), binary (fixed-layout records), random
                 (incompressible) and small (300 files of 0.2-8 KB)
  NAME=PATH      any file, or every file under a directory

Every corpus is listed in the manifest (--manifest, and in --json) with each
file's size and SHA-256 and the corpus's own SHA-256, so a later run can show
it measured the same bytes. --per-file also compresses each file on its own
and reports the sums next to the concatenated result.

Speeds are in MB/s of original data (1 MB = 10^6 bytes), measured as wall
time around each command: the median of --runs, with the fastest run shown
separately and labeled "best". Peak memory is the peak resident set size of
the one command (getrusage of that child), not of this script. Size ratios
are compressed size as a percentage of the original. --json keeps every raw
repetition with the source revision, build settings, machine and commands.

Examples:
  tools/benchmark.py                                  # every corpus, every codec found
  tools/benchmark.py --corpus hce-repo --runs 3 --md bench.md --csv bench.csv
  tools/benchmark.py --corpus logs=/var/log/syslog --codecs ncm,zstd

Exit status: 0 when every round trip verified, 1 when a command failed or a
round trip did not match, 2 for usage errors (unknown corpus, missing tool).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import filecmp
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HCE = REPO_ROOT / "build" / "release" / "hce"
HCE_SOURCE_DIRS = ("codec", "encoder", "decoder", "aio", "structural", "cli", "io", "tests")
ALL_CODECS = ("structural", "ncm", "zstd", "xz")


class UsageError(Exception):
    """A problem with the command line or the environment: exit status 2."""


# ----------------------------------------------------------------- corpora ---


@dataclass
class Corpus:
    name: str
    path: Path
    files: int
    sources: List[Path] = field(default_factory=list)  # the files, in order (synthetic: under the work directory)
    origin: str = ""

    @property
    def size(self) -> int:
        return self.path.stat().st_size


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_of(c: Corpus, root: Path) -> Dict[str, object]:
    """Exact content of a corpus: every file's size and hash, and the concatenation's."""
    def name(p: Path) -> str:
        for base in (root, REPO_ROOT):
            try:
                return str(p.relative_to(base))
            except ValueError:
                pass
        return str(p)
    return {"name": c.name, "origin": c.origin, "bytes": c.size, "sha256": sha256_of(c.path), "files": [
        {"path": name(p), "bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in c.sources]}


def concatenate(files: Sequence[Path], dest: Path, max_bytes: Optional[int]) -> int:
    """Writes the files to dest in order, stopping at max_bytes; returns how many were used."""
    used = 0
    written = 0
    with dest.open("wb") as out:
        for f in files:
            if max_bytes is not None and written >= max_bytes:
                break
            data = f.read_bytes()
            if max_bytes is not None:
                data = data[: max_bytes - written]
            out.write(data)
            written += len(data)
            used += 1
    return used


def python_stdlib_files() -> List[Path]:
    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.glob("*.py") if p.is_file())
    files += sorted(p for p in root.glob("*/*.py") if p.is_file() and "site-packages" not in p.parts)
    return files


def cxx_header_files() -> List[Path]:
    base = Path("/usr/include/c++")
    versions = [d for d in base.iterdir() if d.is_dir() and re.fullmatch(r"\d+(\.\d+)*", d.name)] if base.is_dir() else []
    if not versions:
        return []
    newest = max(versions, key=lambda d: tuple(int(x) for x in d.name.split(".")))
    return sorted(p for p in newest.rglob("*") if p.is_file())


def hce_repo_files() -> List[Path]:
    files = [REPO_ROOT / "hc_framed.hpp", REPO_ROOT / "test_hc_framed.cpp"]
    for d in HCE_SOURCE_DIRS:
        files += sorted(p for p in (REPO_ROOT / d).glob("*") if p.suffix in (".cpp", ".hpp"))
    return sorted(p for p in files if p.is_file())


# Synthetic corpora: random.Random's Mersenne Twister gives the same bytes for
# the same seed on every Python and platform.
def synthetic_logs(rng: random.Random) -> List[bytes]:
    hosts, paths = [f"web-{i:02d}" for i in range(12)], ["/api/v1/items", "/api/v1/users", "/login", "/static/app.js"]
    lines = []
    for i in range(30000):
        lines.append(f"2026-10-{1 + i // 2000:02d}T{i // 120 % 24:02d}:{i // 2 % 60:02d}:{i % 60:02d}Z "
                     f"{rng.choice(hosts)} {rng.choice(('GET', 'POST'))} {rng.choice(paths)}/{rng.randrange(100000)} "
                     f"status={rng.choice((200, 200, 200, 304, 404, 503))} ms={rng.randrange(900)}\n")
    return ["".join(lines).encode()]


def synthetic_binary(rng: random.Random) -> List[bytes]:
    out = bytearray()
    for i in range(40000):  # fixed 28-byte records: id, timestamp, float, flags, small counters
        out += i.to_bytes(4, "little") + (1_700_000_000 + 7 * i).to_bytes(8, "little")
        out += rng.getrandbits(32).to_bytes(4, "little") + bytes([rng.randrange(4), 0, 0, 0])
        out += rng.randrange(1000).to_bytes(4, "little") + rng.randrange(16).to_bytes(4, "little")
    return [bytes(out)]


def synthetic_random(rng: random.Random) -> List[bytes]:
    return [rng.randbytes(1 << 20)]


def synthetic_small(rng: random.Random) -> List[bytes]:
    words = [w.encode() for w in "the of and to in is that for it as with was on be by this are from".split()]
    files = []
    for _ in range(300):
        n = rng.randrange(200, 8192)
        files.append(b" ".join(rng.choice(words) for _ in range(n // 4))[:n] + b"\n")
    return files


# The SHA-256 of each concatenated synthetic corpus: a change to a generator
# (or to random.Random) shows up as a mismatch in tests/python/test_benchmark_tool.py.
SYNTHETIC_SHA256 = {
    "synthetic-logs": "f5683402c51419bf3062c4ce331b3573fb829e4c9e8e819ffaf5a8312449c591",
    "synthetic-binary": "71ce789676505cb35895f565d9018172661dd860edeccaedee594f9cfd26904d",
    "synthetic-random": "09c2588d3024d22d3b02c58c8a656378d42a80cf6ed51a773c58bb1489a911dc",
    "synthetic-small": "da32254e5aeaa2844dd0ce3ce87d83a10fd3ee332fe4b90cb0e4fcca0a7d2979",
}

SYNTHETIC: Dict[str, tuple] = {
    "synthetic-logs": (synthetic_logs, ".log", 11),
    "synthetic-binary": (synthetic_binary, ".bin", 12),
    "synthetic-random": (synthetic_random, ".bin", 13),
    "synthetic-small": (synthetic_small, ".txt", 14),
}

# name -> (file lister, extension). The extension lets hce --structural pick
# the right keyword table (--lang auto).
BUILTIN_CORPORA: Dict[str, tuple] = {
    "python-stdlib": (python_stdlib_files, ".py"),
    "cxx-headers": (cxx_header_files, ".hpp"),
    "hce-repo": (hce_repo_files, ".cpp"),
    **{name: (lambda: [Path("synthetic")], ext) for name, (_, ext, _) in SYNTHETIC.items()},
}


def synthetic_files(name: str, workdir: Path) -> List[Path]:
    make, ext, seed = SYNTHETIC[name]
    d = workdir / f"{name}.files"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, data in enumerate(make(random.Random(seed))):
        p = d / f"{i:04d}{ext}"
        p.write_bytes(data)
        paths.append(p)
    return paths


def build_corpus(spec: str, workdir: Path, max_bytes: Optional[int]) -> Corpus:
    """Builds one corpus from a built-in name or NAME=PATH."""
    if "=" in spec:
        name, raw = spec.split("=", 1)
        src = Path(raw).expanduser()
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise UsageError(f"corpus name {name!r}: use letters, digits, '.', '_' or '-'")
        origin = str(src)
        if src.is_file():
            files, ext = [src], src.suffix or ".txt"
        elif src.is_dir():
            files, ext = sorted(p for p in src.rglob("*") if p.is_file()), ".txt"
        else:
            raise UsageError(f"corpus {name}: {src} does not exist")
    elif spec in SYNTHETIC:
        name, ext = spec, SYNTHETIC[spec][1]
        files = synthetic_files(spec, workdir)
        origin = f"generated by tools/benchmark.py, seed {SYNTHETIC[spec][2]}"
    elif spec in BUILTIN_CORPORA:
        name = spec
        lister, ext = BUILTIN_CORPORA[spec]
        files = lister()
        origin = {"python-stdlib": f"Python {platform.python_version()} standard library ({sysconfig.get_paths()['stdlib']})",
                  "cxx-headers": "newest libstdc++ headers under /usr/include/c++",
                  "hce-repo": "this checkout's C++ sources"}[spec]
    else:
        raise UsageError(f"unknown corpus {spec!r} (built in: {', '.join(BUILTIN_CORPORA)}; or NAME=PATH)")
    if not files:
        raise UsageError(f"corpus {name}: no files found")
    dest = workdir / f"{name}{ext}"
    used = concatenate(files, dest, max_bytes)
    if dest.stat().st_size == 0:
        raise UsageError(f"corpus {name}: the files are empty")
    return Corpus(name, dest, used, list(files[:used]), origin)


# ------------------------------------------------------------------ codecs ---


@dataclass
class Codec:
    name: str
    label: str
    extension: str
    compress: Callable[[Path, Path], List[str]]    # (source, packed) -> argv
    decompress: Callable[[Path, Path], List[str]]  # (packed, restored) -> argv
    stdout: bool = False  # the tool writes to stdout (xz): redirect it to the output file
    version: str = ""


def first_line(argv: Sequence[str]) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        text = (out.stdout or out.stderr).strip()
        return text.splitlines()[0].strip("* ") if text else "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def make_codecs(hce: Path, level: int, threads: int) -> Dict[str, Codec]:
    h, t = str(hce), str(threads)
    hce_version = first_line([h, "--version"])
    return {
        "structural": Codec(
            "structural", f"hce --structural -l {level}", ".hcs",
            lambda s, o: [h, "--structural", "-l", str(level), "-T", t, "-f", "-i", str(s), "-o", str(o)],
            lambda p, o: [h, "-d", "-T", t, "-f", "-i", str(p), "-o", str(o)],
            version=hce_version),
        "ncm": Codec(
            "ncm", f"hce -l {level} (byte codec, default profile)", ".hce",
            lambda s, o: [h, "-l", str(level), "-T", t, "-f", "-i", str(s), "-o", str(o)],
            lambda p, o: [h, "-d", "-T", t, "-f", "-i", str(p), "-o", str(o)],
            version=hce_version),
        "zstd": Codec(
            "zstd", "zstd -19", ".zst",
            lambda s, o: ["zstd", "-19", f"-T{t}", "-q", "-f", str(s), "-o", str(o)],
            lambda p, o: ["zstd", "-d", "-q", "-f", str(p), "-o", str(o)],
            version=first_line(["zstd", "--version"])),
        "xz": Codec(
            "xz", "xz -9e", ".xz",
            lambda s, o: ["xz", "-9e", f"-T{t}", "-c", str(s)],
            lambda p, o: ["xz", "-d", f"-T{t}", "-c", str(p)],
            stdout=True,
            version=first_line(["xz", "--version"])),
    }


def available(codec: Codec, hce: Path) -> bool:
    if codec.name in ("structural", "ncm"):
        return hce.is_file() and os.access(hce, os.X_OK)
    return shutil.which(codec.name) is not None


# ------------------------------------------------------------- measurement ---


@dataclass
class Result:
    corpus: str
    corpus_bytes: int
    codec: str
    label: str
    mode: str = "concatenated"  # or "per-file": every file on its own, sizes and times summed
    packed_bytes: int = 0
    compress_runs: List[float] = field(default_factory=list)    # seconds, every repetition
    decompress_runs: List[float] = field(default_factory=list)
    compress_rss: int = 0    # bytes, peak RSS of the slowest-memory compress command
    decompress_rss: int = 0
    verified: bool = False
    error: str = ""
    commands: List[List[str]] = field(default_factory=list)

    @property
    def compress_s(self) -> float:
        return median(self.compress_runs)

    @property
    def decompress_s(self) -> float:
        return median(self.decompress_runs)

    @property
    def ratio_pct(self) -> float:
        return 100.0 * self.packed_bytes / self.corpus_bytes if self.corpus_bytes else 0.0

    @property
    def saved_pct(self) -> float:
        return 100.0 - self.ratio_pct

    @staticmethod
    def mb_s(nbytes: int, seconds: float) -> float:
        return nbytes / seconds / 1e6 if seconds > 0 else 0.0

    @property
    def compress_mb_s(self) -> float:
        return self.mb_s(self.corpus_bytes, self.compress_s)

    @property
    def decompress_mb_s(self) -> float:
        return self.mb_s(self.corpus_bytes, self.decompress_s)

    @property
    def best_compress_mb_s(self) -> float:
        return self.mb_s(self.corpus_bytes, min(self.compress_runs, default=0.0))

    @property
    def best_decompress_mb_s(self) -> float:
        return self.mb_s(self.corpus_bytes, min(self.decompress_runs, default=0.0))


def median(v: Sequence[float]) -> float:
    if not v:
        return 0.0
    s = sorted(v)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def max_rss_bytes(ru_maxrss: int) -> int:
    """getrusage's ru_maxrss in bytes: macOS reports bytes, Linux and the BSDs KiB."""
    return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


def timed(argv: List[str], stdout_to: Optional[Path], timeout: float) -> tuple:
    """Runs argv; returns (wall seconds, peak RSS bytes or 0); raises RuntimeError with stderr on failure."""
    start = time.perf_counter()
    out = stdout_to.open("wb") if stdout_to is not None else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.PIPE)
    except OSError as e:
        raise RuntimeError(f"cannot run {argv[0]}: {e}") from None
    finally:
        if stdout_to is not None:
            out.close()
    rss = 0
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    try:
        if hasattr(os, "wait4"):
            # stderr first, so a chatty child cannot block on a full pipe; then its own rusage.
            err = proc.stderr.read()
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            rss = max_rss_bytes(usage.ru_maxrss)
        else:
            _, err = proc.communicate()
    finally:
        killer.cancel()
        proc.stderr.close()
    elapsed = time.perf_counter() - start
    if elapsed >= timeout:
        raise RuntimeError(f"timed out after {timeout:.0f} s: {' '.join(argv)}")
    if proc.returncode != 0:
        lines = err.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(argv)}" + (f": {lines[-1]}" if lines else ""))
    return elapsed, rss


def round_trip(source: Path, codec: Codec, workdir: Path, tag: str, runs: int, timeout: float, r: Result) -> None:
    """Adds `runs` verified repetitions of one input to r; raises RuntimeError on any failure."""
    packed = workdir / f"{tag}{codec.extension}"
    restored = workdir / f"{tag}.{codec.name}.restored"
    try:
        size = None
        for i in range(runs):
            restored.unlink(missing_ok=True)  # each repetition must produce its own output
            cargv, dargv = codec.compress(source, packed), codec.decompress(packed, restored)
            if len(r.commands) < 2:
                r.commands += [cargv, dargv]
            t, rss = timed(cargv, packed if codec.stdout else None, timeout)
            r.compress_rss = max(r.compress_rss, rss)
            n = packed.stat().st_size
            if size is not None and n != size:
                raise RuntimeError(f"run {i + 1}: compressed size changed from {size} to {n} bytes")
            size = n
            d, rss = timed(dargv, restored if codec.stdout else None, timeout)
            r.decompress_rss = max(r.decompress_rss, rss)
            if not restored.exists() or not filecmp.cmp(source, restored, shallow=False):
                raise RuntimeError(f"run {i + 1}: decompressed output differs from the original")
            if len(r.compress_runs) <= i:
                r.compress_runs.append(0.0)
                r.decompress_runs.append(0.0)
            r.compress_runs[i] += t
            r.decompress_runs[i] += d
        r.packed_bytes += size or 0
    finally:
        for p in (packed, restored):
            p.unlink(missing_ok=True)


def measure(corpus: Corpus, codec: Codec, workdir: Path, runs: int, timeout: float, per_file: bool = False) -> Result:
    r = Result(corpus.name, corpus.size, codec.name, codec.label, mode="per-file" if per_file else "concatenated")
    try:
        if per_file:
            r.corpus_bytes = sum(p.stat().st_size for p in corpus.sources)
            for i, src in enumerate(corpus.sources):
                round_trip(src, codec, workdir, f"{corpus.name}.{i:05d}", runs, timeout, r)
        else:
            round_trip(corpus.path, codec, workdir, corpus.name, runs, timeout, r)
        r.verified = True
    except RuntimeError as e:
        r.error = str(e)
        r.verified = False
    return r


# ----------------------------------------------------------------- reports ---


def git_revision() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (rev or "?") + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def build_settings(hce: Path) -> Dict[str, str]:
    """Compiler, build type and flags from the CMake cache beside the hce binary, if there is one."""
    keys = ("CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
            "CMAKE_CXX_FLAGS_RELWITHDEBINFO", "HCE_NATIVE", "HCE_PGO", "HCE_SANITIZE")
    out: Dict[str, str] = {}
    cache = hce.resolve().parent / "CMakeCache.txt" if hce.exists() else None
    if cache and cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            k, _, v = line.partition("=")
            k = k.split(":", 1)[0]
            if k in keys:
                out[k] = v
    if "CMAKE_CXX_COMPILER" in out:
        out["compiler_version"] = first_line([out["CMAKE_CXX_COMPILER"], "--version"])
    return out


def machine() -> Dict[str, object]:
    cpu = platform.processor() or "?"
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        if sys.platform == "darwin":
            cpu = first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version()}


def environment(codecs: Dict[str, Codec], args: argparse.Namespace) -> List[str]:
    m = machine()
    lines = [
        f"- Date: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Source revision: {git_revision()}",
        f"- Machine: {m['cpu']}, {m['logical_cpus']} logical CPUs; {m['platform']}",
        f"- Settings: {args.threads} threads, hce level {args.level}, {args.runs} run(s) per measurement: "
        "speeds are medians, \"best\" the fastest run; peak memory is the command's own peak RSS",
    ]
    build = build_settings(args.hce)
    if build:
        lines.append("- hce build: " + ", ".join(f"{k}={v}" for k, v in build.items() if v))
    seen = set()
    for c in codecs.values():
        if c.version not in seen:
            seen.add(c.version)
            lines.append(f"- {c.version}")
    return lines


def mib(n: int) -> str:
    return f"{n / 1048576:.1f}" if n else "-"


def markdown(corpora: List[Corpus], results: List[Result], env: List[str]) -> str:
    out = ["# HCE compression benchmark", "", *env, ""]
    for c in corpora:
        rows = [r for r in results if r.corpus == c.name]
        out += [f"## {c.name}: {c.size:,} bytes from {c.files} file(s)", "",
                "| Codec | Mode | Compressed bytes | Size | Saved | Compress MB/s (best) | Decompress MB/s (best) "
                "| Peak MiB c/d | Verified |",
                "|:--|:--|--:|--:|--:|--:|--:|--:|:--|"]
        for r in rows:
            if r.error:
                out.append(f"| {r.label} | {r.mode} | - | - | - | - | - | - | **failed**: {r.error} |")
            else:
                out.append(f"| {r.label} | {r.mode} | {r.packed_bytes:,} | {r.ratio_pct:.2f}% | {r.saved_pct:.2f}% | "
                           f"{r.compress_mb_s:.2f} ({r.best_compress_mb_s:.2f}) | "
                           f"{r.decompress_mb_s:.2f} ({r.best_decompress_mb_s:.2f}) | "
                           f"{mib(r.compress_rss)} / {mib(r.decompress_rss)} | yes, every run |")
        out.append("")
    if len(corpora) > 1:
        labels = list(dict.fromkeys(r.label for r in results))
        out += ["## Summary: compressed size as % of the original (lower is better; concatenated)", "",
                "| Codec | " + " | ".join(c.name for c in corpora) + " |",
                "|:--|" + "--:|" * len(corpora)]
        for label in labels:
            cells = []
            for c in corpora:
                r = next((x for x in results if x.corpus == c.name and x.label == label and x.mode == "concatenated"), None)
                cells.append(f"{r.ratio_pct:.2f}%" if r and not r.error else "-")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


CSV_FIELDS = ["corpus", "corpus_bytes", "codec", "label", "mode", "compressed_bytes", "size_pct", "saved_pct",
              "compress_s", "decompress_s", "compress_mb_s", "decompress_mb_s", "best_compress_mb_s",
              "best_decompress_mb_s", "compress_peak_rss_bytes", "decompress_peak_rss_bytes", "runs", "verified",
              "error"]


def write_csv(path: Path, results: List[Result]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in results:
            w.writerow([r.corpus, r.corpus_bytes, r.codec, r.label, r.mode, r.packed_bytes, f"{r.ratio_pct:.4f}",
                        f"{r.saved_pct:.4f}", f"{r.compress_s:.4f}", f"{r.decompress_s:.4f}",
                        f"{r.compress_mb_s:.4f}", f"{r.decompress_mb_s:.4f}", f"{r.best_compress_mb_s:.4f}",
                        f"{r.best_decompress_mb_s:.4f}", r.compress_rss, r.decompress_rss, len(r.compress_runs),
                        int(r.verified), r.error])


def write_json(path: Path, corpora: List[Corpus], results: List[Result], codecs: Dict[str, Codec],
               args: argparse.Namespace, workdir: Path) -> None:
    doc = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source_revision": git_revision(),
        "machine": machine(),
        "hce_build": build_settings(args.hce),
        "settings": {"threads": args.threads, "level": args.level, "runs": args.runs, "max_bytes": args.max_bytes,
                     "per_file": args.per_file},
        "codecs": {c.name: {"label": c.label, "version": c.version} for c in codecs.values()},
        "peak_rss_scope": "peak resident set size of each compress or decompress command (getrusage of the child)",
        "corpora": [manifest_of(c, workdir) for c in corpora],
        "results": [{"corpus": r.corpus, "codec": r.codec, "label": r.label, "mode": r.mode,
                     "corpus_bytes": r.corpus_bytes, "compressed_bytes": r.packed_bytes,
                     "compress_s": r.compress_runs, "decompress_s": r.decompress_runs,
                     "compress_s_median": r.compress_s, "decompress_s_median": r.decompress_s,
                     "compress_peak_rss_bytes": r.compress_rss, "decompress_peak_rss_bytes": r.decompress_rss,
                     "verified_every_run": r.verified, "error": r.error, "commands": r.commands}
                    for r in results],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# -------------------------------------------------------------------- main ---


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                epilog="See the module docstring (or README) for details.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hce", type=Path, default=DEFAULT_HCE, help=f"hce binary (default: {DEFAULT_HCE})")
    p.add_argument("--corpus", action="append", default=[], metavar="NAME[=PATH]",
                   help="corpus to run; repeatable (default: every built-in corpus present)")
    p.add_argument("--codecs", default=",".join(ALL_CODECS),
                   help=f"comma-separated subset of {','.join(ALL_CODECS)} (default: all found)")
    p.add_argument("--level", type=int, default=2, choices=range(0, 5), metavar="0..4", help="hce level (default 2)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="threads for every tool (default: all CPUs)")
    p.add_argument("--runs", type=int, default=1,
                   help="repetitions, each verified; the median is reported, the fastest labeled best (default 1)")
    p.add_argument("--per-file", action="store_true",
                   help="also compress every file of each corpus on its own and report the sums")
    p.add_argument("--max-bytes", type=int, default=None, help="cut each corpus to this many bytes")
    p.add_argument("--timeout", type=float, default=3600, help="seconds allowed per command (default 3600)")
    p.add_argument("--workdir", type=Path, default=None, help="where corpora and outputs go (default: a temporary directory)")
    p.add_argument("--keep", action="store_true", help="keep the work directory")
    p.add_argument("--md", type=Path, help="write the Markdown report here (it is always printed)")
    p.add_argument("--csv", type=Path, help="write the CSV results here")
    p.add_argument("--json", type=Path, help="write every raw repetition, the manifest and the environment here")
    p.add_argument("--manifest", type=Path, help="write the corpus manifest (files, sizes, SHA-256) here")
    args = p.parse_args(argv)
    if args.threads < 1 or args.runs < 1 or (args.max_bytes is not None and args.max_bytes < 1):
        p.error("--threads, --runs and --max-bytes must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    requested = [c.strip() for c in args.codecs.split(",") if c.strip()]
    explicit_codecs = args.codecs != ",".join(ALL_CODECS)
    try:
        unknown = [c for c in requested if c not in ALL_CODECS]
        if unknown:
            raise UsageError(f"unknown codec(s): {', '.join(unknown)}")
        all_codecs = make_codecs(args.hce, args.level, args.threads)
        codecs: Dict[str, Codec] = {}
        for name in requested:
            if available(all_codecs[name], args.hce):
                codecs[name] = all_codecs[name]
            elif explicit_codecs:
                raise UsageError(f"{name}: tool not found" + (f" ({args.hce})" if name in ("structural", "ncm") else ""))
            else:
                print(f"note: skipping {name}: tool not found", file=sys.stderr)
        if not codecs:
            raise UsageError("no codec available")

        tmp = None
        if args.workdir is None:
            tmp = tempfile.mkdtemp(prefix="hce-bench-")
            workdir = Path(tmp)
        else:
            workdir = args.workdir
            workdir.mkdir(parents=True, exist_ok=True)
        try:
            specs = args.corpus or [n for n, (lister, _) in BUILTIN_CORPORA.items() if lister()]
            corpora = [build_corpus(s, workdir, args.max_bytes) for s in specs]
            if args.manifest:
                args.manifest.write_text(json.dumps([manifest_of(c, workdir) for c in corpora], indent=2) + "\n",
                                         encoding="utf-8")
            results: List[Result] = []
            for c in corpora:
                for codec in codecs.values():
                    for per_file in (False, True) if args.per_file else (False,):
                        print(f"{c.name} ({c.size:,} bytes, {'per file' if per_file else 'concatenated'}): "
                              f"{codec.label} ...", file=sys.stderr, flush=True)
                        r = measure(c, codec, workdir, args.runs, args.timeout, per_file)
                        results.append(r)
                        if r.error:
                            print(f"  FAILED: {r.error}", file=sys.stderr)
                        else:
                            print(f"  {r.packed_bytes:,} bytes ({r.ratio_pct:.2f}%), {r.compress_mb_s:.2f} / "
                                  f"{r.decompress_mb_s:.2f} MB/s (median)", file=sys.stderr)
            report = markdown(corpora, results, environment(codecs, args))
            print(report)
            if args.md:
                args.md.write_text(report + "\n", encoding="utf-8")
            if args.csv:
                write_csv(args.csv, results)
            if args.json:
                write_json(args.json, corpora, results, codecs, args, workdir)
        finally:
            if tmp and not args.keep:
                shutil.rmtree(tmp, ignore_errors=True)
        return 0 if all(r.verified and not r.error for r in results) else 1
    except UsageError as e:
        print(f"benchmark.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
