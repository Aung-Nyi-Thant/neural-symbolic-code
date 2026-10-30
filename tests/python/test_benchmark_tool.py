"""tools/benchmark.py: corpus building, argument checks and an end-to-end run."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

import testdata

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "benchmark.py"

if not SCRIPT.is_file():
    pytest.skip("tools/benchmark.py is only in a source checkout", allow_module_level=True)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("hce_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


def test_corpus_from_file(bench, tmp_path):
    src = tmp_path / "in.log"
    src.write_bytes(testdata.log(1, 5000))
    work = tmp_path / "work"
    work.mkdir()
    corpus = bench.build_corpus(f"logs={src}", work, None)
    assert corpus.name == "logs" and corpus.files == 1
    assert corpus.path.read_bytes() == src.read_bytes()
    assert corpus.path.suffix == ".log"


def test_corpus_from_directory_is_sorted_and_cut(bench, tmp_path):
    src = tmp_path / "tree"
    (src / "sub").mkdir(parents=True)
    (src / "b.txt").write_bytes(b"B" * 100)
    (src / "a.txt").write_bytes(b"A" * 100)
    (src / "sub" / "c.txt").write_bytes(b"C" * 100)
    work = tmp_path / "work"
    work.mkdir()
    corpus = bench.build_corpus(f"tree={src}", work, 250)
    assert corpus.path.read_bytes() == b"A" * 100 + b"B" * 100 + b"C" * 50


@pytest.mark.parametrize("spec, message", [("bad name=/tmp", "corpus name"), ("x=/does/not/exist", "does not exist"),
                                           ("no-such-builtin", "unknown corpus")])
def test_corpus_errors(bench, tmp_path, spec, message):
    with pytest.raises(bench.UsageError, match=message):
        bench.build_corpus(spec, tmp_path, None)


def test_argument_validation(bench, capsys):
    with pytest.raises(SystemExit) as info:
        bench.parse_args(["--threads", "0"])
    assert info.value.code == 2
    assert "must be positive" in capsys.readouterr().err


def test_unknown_codec_is_usage_error(bench, capsys):
    assert bench.main(["--codecs", "gzip", "--corpus", "x=/dev/null"]) == 2
    assert "unknown codec" in capsys.readouterr().err


@pytest.mark.cli
def test_end_to_end(bench, cli, tmp_path, capsys):
    src = tmp_path / "corpus.py"
    src.write_bytes(testdata.source(9, 20000))
    md, out_csv = tmp_path / "bench.md", tmp_path / "bench.csv"
    status = bench.main(["--hce", str(cli), "--corpus", f"code={src}", "--codecs", "ncm,structural",
                         "--threads", "2", "--workdir", str(tmp_path / "work"), "--md", str(md),
                         "--csv", str(out_csv)])
    assert status == 0
    assert "## code: 20,000 bytes from 1 file(s)" in md.read_text()
    with out_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["codec"] for r in rows] == ["ncm", "structural"]
    for row in rows:
        assert row["verified"] == "1" and not row["error"]
        assert 0 < int(row["compressed_bytes"]) < 20000
    capsys.readouterr()


def _fake_codec(bench, tmp_path, bad_runs):
    """A codec that copies its input; its decompressor corrupts the output on the listed runs (1-based)."""
    counter = tmp_path / "runs"
    counter.write_text("0")
    decomp = tmp_path / "fake_decompress.py"
    decomp.write_text(
        "import sys, pathlib\n"
        f"c = pathlib.Path({str(counter)!r}); n = int(c.read_text()) + 1; c.write_text(str(n))\n"
        "data = pathlib.Path(sys.argv[1]).read_bytes()\n"
        f"if n in {tuple(bad_runs)!r}: data = data[:-1] + bytes([data[-1] ^ 1])\n"
        "pathlib.Path(sys.argv[2]).write_bytes(data)\n")
    copy = ["-c", "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])"]
    return bench.Codec("fake", "fake", ".fk",
                       lambda s, o: [sys.executable, *copy, str(s), str(o)],
                       lambda p, o: [sys.executable, str(decomp), str(p), str(o)])


def test_every_repetition_is_verified(bench, tmp_path):
    src = tmp_path / "in.txt"
    src.write_bytes(testdata.text(4, 3000))
    work = tmp_path / "work"
    work.mkdir()
    corpus = bench.build_corpus(f"t={src}", work, None)
    # A wrong first repetition fails the measurement although the second is right.
    r = bench.measure(corpus, _fake_codec(bench, tmp_path, [1]), work, 2, 60)
    assert not r.verified and "run 1" in r.error
    # A wrong last repetition too; three right ones pass, with every run recorded.
    assert not bench.measure(corpus, _fake_codec(bench, tmp_path, [3]), work, 3, 60).verified
    ok = bench.measure(corpus, _fake_codec(bench, tmp_path, []), work, 3, 60)
    assert ok.verified and not ok.error and len(ok.compress_runs) == 3 and ok.packed_bytes == src.stat().st_size


def test_median_best_and_rss_units(bench, monkeypatch):
    assert bench.median([3.0, 1.0, 2.0]) == 2.0 and bench.median([4.0, 1.0, 2.0, 3.0]) == 2.5
    r = bench.Result("c", 10_000_000, "x", "x", compress_runs=[1.0, 4.0, 2.0], decompress_runs=[0.5, 0.5])
    assert r.compress_mb_s == 5.0 and r.best_compress_mb_s == 10.0
    monkeypatch.setattr(bench.sys, "platform", "darwin")
    assert bench.max_rss_bytes(1 << 20) == 1 << 20  # macOS: bytes
    monkeypatch.setattr(bench.sys, "platform", "linux")
    assert bench.max_rss_bytes(1024) == 1 << 20  # Linux: KiB


def test_synthetic_corpora_are_fixed(bench, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for name in bench.SYNTHETIC:
        ca, cb = bench.build_corpus(name, a, None), bench.build_corpus(name, b, None)
        ma, mb = bench.manifest_of(ca, a), bench.manifest_of(cb, b)
        assert ma["sha256"] == mb["sha256"] == bench.SYNTHETIC_SHA256[name]
        assert ma["files"] == mb["files"] and ma["bytes"] > 0
    assert bench.build_corpus("synthetic-small", a, None).files == 300
