"""Archive verification against the golden vectors in tests/python/golden/.

The vectors freeze both formats (R17). For each one the suite checks that
 * the stored archive decompresses and verifies to the regenerated input,
 * compressing that input with the recorded options reproduces the archive
   byte for byte, with one thread and with several,
 * the manifest's hashes match the files and the regenerated inputs.
A failure here means a change altered the coded bits: ship it as a new codec
profile (and new vectors), never by rewriting these files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hce
import testdata

GOLDEN = Path(__file__).resolve().parent / "golden"
MANIFEST = json.loads((GOLDEN / "manifest.json").read_text())
VECTORS = {v["name"]: v for v in MANIFEST["vectors"]}

each_vector = pytest.mark.parametrize("name", sorted(VECTORS))


def _input(vector: dict) -> bytes:
    spec = vector["input"]
    return testdata.generate(spec["generator"], spec["seed"], spec["size"])


def _archive(vector: dict) -> bytes:
    return (GOLDEN / vector["file"]).read_bytes()


@each_vector
def test_input_generator_is_stable(name):
    vector = VECTORS[name]
    data = _input(vector)
    assert len(data) == vector["input"]["size"]
    assert testdata.sha256(data) == vector["input"]["sha256"], "testdata.py changed: the golden inputs drifted"


@each_vector
def test_archive_file_matches_manifest(name):
    vector = VECTORS[name]
    archive = _archive(vector)
    assert len(archive) == vector["compressed_size"]
    assert testdata.sha256(archive) == vector["compressed_sha256"]
    assert hce.detect_format(archive) == vector["format"]


@each_vector
def test_golden_archive_decompresses(name):
    vector = VECTORS[name]
    assert hce.decompress(_archive(vector)) == _input(vector)


@each_vector
def test_golden_archive_verifies(name):
    vector = VECTORS[name]
    stats = hce.verify(_archive(vector))
    assert stats.verified
    assert stats.format == vector["format"]
    assert stats.input_bytes == vector["compressed_size"]
    assert stats.output_bytes == vector["input"]["size"]
    if vector["format"] == "structural":
        assert stats.details["language"] == vector["options"].get("language", "generic")


@each_vector
@pytest.mark.parametrize("threads", [1, 3])
def test_encoder_reproduces_golden_archive(name, threads):
    vector = VECTORS[name]
    assert hce.compress(_input(vector), threads=threads, **vector["options"]) == _archive(vector)


@each_vector
def test_golden_archive_decompresses_from_file(name, tmp_path):
    vector = VECTORS[name]
    out = tmp_path / "restored"
    stats = hce.decompress_file(GOLDEN / vector["file"], out)
    assert stats.verified
    assert out.read_bytes() == _input(vector)


@pytest.mark.cli
@each_vector
def test_golden_archive_through_cli(name, run_cli):
    vector = VECTORS[name]
    proc = run_cli("-d", "-q", "-f", stdin=_archive(vector))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == _input(vector)


def test_manifest_covers_every_golden_file():
    assert MANIFEST["hce_version"]
    on_disk = {p.name for p in GOLDEN.iterdir() if p.suffix in (".hce", ".hcs")}
    assert on_disk == {v["file"] for v in VECTORS.values()}
    assert {v["format"] for v in VECTORS.values()} == {"framed", "structural"}
