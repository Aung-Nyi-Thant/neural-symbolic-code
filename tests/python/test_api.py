"""The public surface of the hce package: names, types, metadata, versions."""

from __future__ import annotations

import array
import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import pytest

import hce
from hce import _hce

REPO = Path(__file__).resolve().parents[2]


def test_all_names_exist():
    for name in hce.__all__:
        assert hasattr(hce, name), name
    assert len(set(hce.__all__)) == len(hce.__all__)


def test_extension_is_compiled_module():
    # (An editable install keeps _hce in site-packages, apart from __init__.py.)
    assert _hce.__name__ == "hce._hce"
    assert Path(_hce.__file__).suffix in (".so", ".pyd")
    assert Path(_hce.__file__).parent.name == "hce"


def test_typed_package_markers_ship():
    pkg = Path(hce.__file__).parent
    assert (pkg / "py.typed").is_file()
    assert (pkg / "_hce.pyi").is_file()


def test_stub_declares_every_extension_function():
    stub = (Path(hce.__file__).parent / "_hce.pyi").read_text()
    for name in dir(_hce):
        obj = getattr(_hce, name)
        if callable(obj) and not name.startswith("_") and not isinstance(obj, type):
            assert re.search(rf"^def {name}\(", stub, re.M), f"_hce.pyi lacks {name}"


def test_version_matches_pyproject_and_cmake():
    pyproject = REPO / "pyproject.toml"
    cmake = REPO / "CMakeLists.txt"
    if not (pyproject.is_file() and cmake.is_file()):
        pytest.skip("not running from a source checkout")
    if sys.version_info >= (3, 11):
        import tomllib

        py_version = tomllib.loads(pyproject.read_text())["project"]["version"]
    else:
        py_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M).group(1)
    cmake_version = re.search(r"project\(\s*\w+\s+VERSION\s+([0-9.]+)", cmake.read_text()).group(1)
    assert hce.__version__ == py_version == cmake_version == hce.info()["version"]


def test_info_facts():
    facts = hce.info()
    assert set(facts) >= {"version", "neural_cm_codec_id", "structural_codec_id", "max_level", "simd_path",
                          "aio_backend", "languages"}
    assert facts["max_level"] == 4
    assert facts["simd_path"] in ("scalar", "sse2", "avx2", "neon")
    assert facts["aio_backend"] in ("io_uring", "threads")
    assert facts["neural_cm_codec_id"] != facts["structural_codec_id"]
    # info() hands out a copy: callers cannot change what the next call returns.
    facts["version"] = "mutated"
    assert hce.info()["version"] == hce.__version__


def test_codec_profiles():
    facts = hce.info()
    assert facts["max_profile"] == 2
    assert facts["neural_cm_codec_ids"] == (0x4E434D01, 0x4E434D02)
    data = b"2026-10-25 host=a path=/api/v1/items status=200\n" * 400 + bytes(range(256))
    p1 = hce.compress(data, profile=1)
    p2 = hce.compress(data, profile=2)
    assert p1 != p2
    assert hce.compress(data) == p2  # profile 2 is the default
    for packed in (p1, p2):
        assert hce.decompress(packed) == data
        assert hce.verify(packed).verified
    for bad in (0, 3, -1):
        with pytest.raises(ValueError):
            hce.compress(data, profile=bad)


def test_compress_file_profile(tmp_path):
    src = tmp_path / "a.log"
    src.write_bytes(b"key=value; other=thing\n" * 300)
    for profile in (1, 2):
        dest = tmp_path / f"a{profile}.hce"
        hce.compress_file(src, dest, profile=profile)
        assert dest.read_bytes() == hce.compress(src.read_bytes(), profile=profile)
        assert hce.decompress(dest.read_bytes()) == src.read_bytes()


def test_languages():
    assert hce.LANGUAGES[0] == "generic"
    assert set(hce.LANGUAGES) >= {"c", "cpp", "python", "javascript", "typescript", "java", "go", "rust"}
    assert "auto" not in hce.LANGUAGES
    assert isinstance(hce.LANGUAGES, tuple)


def test_default_max_memory():
    assert hce.DEFAULT_MAX_MEMORY == 1 << 30


@pytest.mark.parametrize(
    "data, expected",
    [(b"", None), (b"\x89HC", None), (b"\x89HCZ", "framed"), (b"HCS1", "structural"), (b"hello", None),
     (b"\x89HCZ" + bytes(100), "framed")],
)
def test_detect_format_magic(data, expected):
    assert hce.detect_format(data) == expected


@pytest.mark.parametrize("make", [bytes, bytearray, memoryview, lambda b: array.array("B", b),
                                  lambda b: memoryview(bytearray(b))[0:len(b)]])
def test_detect_format_buffer_types(make):
    assert hce.detect_format(make(b"HCS1" + bytes(50))) == "structural"
    assert hce.detect_format(make(b"\x89HCZ")) == "framed"
    assert hce.detect_format(make(b"HC")) is None


def test_detect_format_multi_byte_items():
    assert hce.detect_format(array.array("I", [int.from_bytes(b"HCS1", "little"), 7])) == "structural"


def test_detect_format_rejects_non_contiguous():
    with pytest.raises(ValueError, match="contiguous"):
        hce.detect_format(memoryview(b"HHCCSS11")[::2])
    with pytest.raises(TypeError):
        hce.detect_format("HCS1")  # type: ignore[arg-type]


@pytest.mark.skipif(sys.platform == "win32", reason="getrusage")
def test_detect_format_reads_only_the_prefix():
    # Peak memory must not grow by the buffer's size: only four bytes are copied.
    script = (
        "import resource, hce\n"
        "data = bytearray(256 << 20); data[:4] = b'HCS1'\n"
        "before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "assert hce.detect_format(data) == 'structural'\n"
        "after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "scale = 1 if __import__('sys').platform == 'darwin' else 1024\n"
        "print((after - before) * scale)\n")
    grew = int(subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout)
    assert grew < 32 << 20


def test_detect_format_on_real_output():
    assert hce.detect_format(hce.compress(b"x" * 100)) == "framed"
    assert hce.detect_format(hce.compress(b"x" * 100, structural=True)) == "structural"


def test_stats_is_frozen_and_typed():
    stats = hce.verify(hce.compress(b"abc" * 100))
    assert isinstance(stats, hce.Stats)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.verified = False  # type: ignore[misc]
    assert stats.format == "framed"
    assert stats.input_bytes > 0 and stats.output_bytes == 300
    assert stats.threads == 1 and stats.verified is True
    assert stats.details["frames"] == 1
    assert stats.ratio == pytest.approx(300 / stats.input_bytes)


def test_stats_ratio_of_empty_is_zero():
    assert hce.verify(hce.compress(b"")).ratio == 0.0


def test_structural_stats_details():
    src = b"def f(x):\n    return x + 1\n" * 40
    stats = hce.verify(hce.compress(src, structural=True, language="python"))
    assert stats.format == "structural"
    assert stats.output_bytes == len(src)
    assert stats.details["language"] == "python"
    assert stats.details["tokens"] > 0
    assert {"symbols", "dictionary_bytes", "model_bytes", "stream_bytes"} <= set(stats.details)


def test_doctest_in_module_docstring():
    import doctest

    failures, tried = doctest.testmod(hce, verbose=False, report=False)
    assert tried > 0 and failures == 0


def test_exceptions_are_exported_from_extension():
    for name in ("HceError", "CorruptInputError", "LimitExceededError", "UnsupportedFormatError", "HceIOError"):
        assert getattr(hce, name) is getattr(_hce, name)
        assert getattr(hce, name).__doc__
