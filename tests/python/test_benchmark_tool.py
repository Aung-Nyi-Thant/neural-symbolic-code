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
