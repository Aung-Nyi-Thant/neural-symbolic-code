#!/usr/bin/env python3
"""Multi-corpus compression benchmark for HCE.

Compresses the same inputs with

  structural  hce --structural    the two-pass structural codec
  ncm         hce                 the v2.1.0 byte-level neural CM codec
  zstd        zstd -19
  xz          xz -9e

decompresses every result and compares it with the original byte for byte,
and writes the sizes and speeds as Markdown and CSV tables.

Corpora are built under the work directory, with files in sorted order so
every run sees the same bytes:

  python-stdlib  the .py files of this interpreter's standard library
                 (top level and one directory down)
  cxx-headers    the newest libstdc++ headers under /usr/include/c++
  hce-repo       this repository's C++ sources
  NAME=PATH      any file, or every file under a directory

Speeds are in MB/s of original data (1 MB = 10^6 bytes), measured as wall
time around each command, fastest of --runs. Size ratios are compressed
size as a percentage of the original.

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
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass
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

    @property
    def size(self) -> int:
        return self.path.stat().st_size


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


# name -> (file lister, extension). The extension lets hce --structural pick
# the right keyword table (--lang auto).
BUILTIN_CORPORA: Dict[str, tuple] = {
    "python-stdlib": (python_stdlib_files, ".py"),
    "cxx-headers": (cxx_header_files, ".hpp"),
    "hce-repo": (hce_repo_files, ".cpp"),
}


def build_corpus(spec: str, workdir: Path, max_bytes: Optional[int]) -> Corpus:
    """Builds one corpus from a built-in name or NAME=PATH."""
    if "=" in spec:
        name, raw = spec.split("=", 1)
        src = Path(raw).expanduser()
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise UsageError(f"corpus name {name!r}: use letters, digits, '.', '_' or '-'")
        if src.is_file():
            files, ext = [src], src.suffix or ".txt"
        elif src.is_dir():
            files, ext = sorted(p for p in src.rglob("*") if p.is_file()), ".txt"
        else:
            raise UsageError(f"corpus {name}: {src} does not exist")
    elif spec in BUILTIN_CORPORA:
        name = spec
        lister, ext = BUILTIN_CORPORA[spec]
        files = lister()
    else:
        raise UsageError(f"unknown corpus {spec!r} (built in: {', '.join(BUILTIN_CORPORA)}; or NAME=PATH)")
    if not files:
        raise UsageError(f"corpus {name}: no files found")
    dest = workdir / f"{name}{ext}"
    used = concatenate(files, dest, max_bytes)
    if dest.stat().st_size == 0:
        raise UsageError(f"corpus {name}: the files are empty")
    return Corpus(name, dest, used)


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
            "ncm", f"hce -l {level} (v2.1.0 byte codec)", ".hce",
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
    packed_bytes: int = 0
    compress_s: float = 0.0
    decompress_s: float = 0.0
    verified: bool = False
    error: str = ""

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


def timed(argv: List[str], stdout_to: Optional[Path], timeout: float) -> float:
    """Runs argv, returns its wall time; raises RuntimeError with stderr on failure."""
    start = time.perf_counter()
    try:
        if stdout_to is not None:
            with stdout_to.open("wb") as out:
                proc = subprocess.run(argv, stdout=out, stderr=subprocess.PIPE, timeout=timeout)
        else:
            proc = subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timed out after {timeout:.0f} s: {' '.join(argv)}") from None
    except OSError as e:
        raise RuntimeError(f"cannot run {argv[0]}: {e}") from None
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(argv)}" + (f": {err[-1]}" if err else ""))
    return elapsed


def measure(corpus: Corpus, codec: Codec, workdir: Path, runs: int, timeout: float) -> Result:
    r = Result(corpus.name, corpus.size, codec.name, codec.label)
    packed = workdir / f"{corpus.name}{codec.extension}"
    restored = workdir / f"{corpus.name}.{codec.name}.restored"
    try:
        comp, decomp = [], []
        for _ in range(runs):
            comp.append(timed(codec.compress(corpus.path, packed), packed if codec.stdout else None, timeout))
            decomp.append(timed(codec.decompress(packed, restored), restored if codec.stdout else None, timeout))
        r.packed_bytes = packed.stat().st_size
        r.compress_s, r.decompress_s = min(comp), min(decomp)
        r.verified = filecmp.cmp(corpus.path, restored, shallow=False)
        if not r.verified:
            r.error = "decompressed output differs from the original"
    except RuntimeError as e:
        r.error = str(e)
    finally:
        for p in (packed, restored):
            p.unlink(missing_ok=True)
    return r


# ----------------------------------------------------------------- reports ---


def environment(codecs: Dict[str, Codec], args: argparse.Namespace) -> List[str]:
    cpu = platform.processor() or "?"
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    lines = [
        f"- Date: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Machine: {cpu}, {os.cpu_count()} logical CPUs; {platform.platform()}",
        f"- Settings: {args.threads} threads, hce level {args.level}, fastest of {args.runs} run(s)",
    ]
    seen = set()
    for c in codecs.values():
        if c.version not in seen:
            seen.add(c.version)
            lines.append(f"- {c.version}")
    return lines


def markdown(corpora: List[Corpus], results: List[Result], env: List[str]) -> str:
    out = ["# HCE compression benchmark", "", *env, ""]
    for c in corpora:
        rows = [r for r in results if r.corpus == c.name]
        out += [f"## {c.name}: {c.size:,} bytes from {c.files} file(s)", "",
                "| Codec | Compressed bytes | Size | Saved | Compress MB/s | Decompress MB/s | Verified |",
                "|:--|--:|--:|--:|--:|--:|:--|"]
        for r in rows:
            if r.error:
                out.append(f"| {r.label} | - | - | - | - | - | **failed**: {r.error} |")
            else:
                out.append(f"| {r.label} | {r.packed_bytes:,} | {r.ratio_pct:.2f}% | {r.saved_pct:.2f}% | "
                           f"{r.compress_mb_s:.2f} | {r.decompress_mb_s:.2f} | yes |")
        out.append("")
    if len(corpora) > 1:
        labels = list(dict.fromkeys(r.label for r in results))
        out += ["## Summary: compressed size as % of the original (lower is better)", "",
                "| Codec | " + " | ".join(c.name for c in corpora) + " |",
                "|:--|" + "--:|" * len(corpora)]
        for label in labels:
            cells = []
            for c in corpora:
                r = next((x for x in results if x.corpus == c.name and x.label == label), None)
                cells.append(f"{r.ratio_pct:.2f}%" if r and not r.error else "-")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


CSV_FIELDS = ["corpus", "corpus_bytes", "codec", "label", "compressed_bytes", "size_pct", "saved_pct",
              "compress_s", "decompress_s", "compress_mb_s", "decompress_mb_s", "verified", "error"]


def write_csv(path: Path, results: List[Result]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in results:
            w.writerow([r.corpus, r.corpus_bytes, r.codec, r.label, r.packed_bytes, f"{r.ratio_pct:.4f}",
                        f"{r.saved_pct:.4f}", f"{r.compress_s:.4f}", f"{r.decompress_s:.4f}",
                        f"{r.compress_mb_s:.4f}", f"{r.decompress_mb_s:.4f}", int(r.verified), r.error])


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
    p.add_argument("--runs", type=int, default=1, help="repetitions; the fastest is reported (default 1)")
    p.add_argument("--max-bytes", type=int, default=None, help="cut each corpus to this many bytes")
    p.add_argument("--timeout", type=float, default=3600, help="seconds allowed per command (default 3600)")
    p.add_argument("--workdir", type=Path, default=None, help="where corpora and outputs go (default: a temporary directory)")
    p.add_argument("--keep", action="store_true", help="keep the work directory")
    p.add_argument("--md", type=Path, help="write the Markdown report here (it is always printed)")
    p.add_argument("--csv", type=Path, help="write the CSV results here")
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
            results: List[Result] = []
            for c in corpora:
                for codec in codecs.values():
                    print(f"{c.name} ({c.size:,} bytes): {codec.label} ...", file=sys.stderr, flush=True)
                    r = measure(c, codec, workdir, args.runs, args.timeout)
                    results.append(r)
                    if r.error:
                        print(f"  FAILED: {r.error}", file=sys.stderr)
                    else:
                        print(f"  {r.packed_bytes:,} bytes ({r.ratio_pct:.2f}%), {r.compress_mb_s:.2f} / "
                              f"{r.decompress_mb_s:.2f} MB/s", file=sys.stderr)
            report = markdown(corpora, results, environment(codecs, args))
            print(report)
            if args.md:
                args.md.write_text(report + "\n", encoding="utf-8")
            if args.csv:
                write_csv(args.csv, results)
        finally:
            if tmp and not args.keep:
                shutil.rmtree(tmp, ignore_errors=True)
        return 0 if all(r.verified and not r.error for r in results) else 1
    except UsageError as e:
        print(f"benchmark.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
