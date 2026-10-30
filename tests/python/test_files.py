"""compress_file / decompress_file: the asynchronous file paths."""

from __future__ import annotations

import errno
import os
import subprocess
import sys

import pytest

import hce
import testdata


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "module.py"
    path.write_bytes(testdata.source(21, 60000))
    return path


@pytest.mark.parametrize("structural", [False, True])
def test_file_roundtrip(tmp_path, source_file, structural):
    packed, restored = tmp_path / "packed", tmp_path / "restored"
    c = hce.compress_file(source_file, packed, structural=structural)
    assert c.format == ("structural" if structural else "framed")
    assert c.input_bytes == source_file.stat().st_size
    assert c.output_bytes == packed.stat().st_size
    d = hce.decompress_file(packed, restored)
    assert d.verified and d.format == c.format
    assert d.input_bytes == c.output_bytes and d.output_bytes == c.input_bytes
    assert restored.read_bytes() == source_file.read_bytes()


def test_framed_file_matches_in_memory(tmp_path, source_file):
    packed = tmp_path / "packed"
    for level in (0, 3):
        hce.compress_file(source_file, packed, level=level, threads=2)
        assert packed.read_bytes() == hce.compress(source_file.read_bytes(), level=level)


def test_structural_language_auto_uses_extension(tmp_path, source_file):
    packed = tmp_path / "packed"
    stats = hce.compress_file(source_file, packed, structural=True)
    assert stats.details["language"] == "python"
    assert packed.read_bytes() == hce.compress(source_file.read_bytes(), structural=True, language="python")


@pytest.mark.parametrize("name, language", [("a.c", "c"), ("a.hpp", "cpp"), ("a.rs", "rust"), ("a.go", "go"),
                                            ("a.ts", "typescript"), ("a.java", "java"), ("a.txt", "generic")])
def test_structural_language_by_extension(tmp_path, name, language):
    src = tmp_path / name
    src.write_bytes(testdata.source(1, 4000))
    assert hce.compress_file(src, tmp_path / "out", structural=True).details["language"] == language


def test_explicit_language_overrides_extension(tmp_path, source_file):
    stats = hce.compress_file(source_file, tmp_path / "out", structural=True, language="rust")
    assert stats.details["language"] == "rust"


def test_str_and_pathlike_paths(tmp_path, source_file):
    packed = tmp_path / "packed"
    hce.compress_file(str(source_file), str(packed))
    hce.decompress_file(os.fsencode(packed).decode(), tmp_path / "restored")
    assert (tmp_path / "restored").read_bytes() == source_file.read_bytes()


def test_unicode_file_names(tmp_path):
    src = tmp_path / "données-λ.txt"
    src.write_bytes(testdata.text(2, 3000))
    hce.compress_file(src, tmp_path / "压缩.hce")
    hce.decompress_file(tmp_path / "压缩.hce", tmp_path / "復元")
    assert (tmp_path / "復元").read_bytes() == src.read_bytes()


@pytest.mark.parametrize("structural", [False, True])
def test_empty_file(tmp_path, structural):
    src = tmp_path / "empty"
    src.write_bytes(b"")
    hce.compress_file(src, tmp_path / "packed", structural=structural)
    stats = hce.decompress_file(tmp_path / "packed", tmp_path / "restored")
    assert stats.output_bytes == 0 and (tmp_path / "restored").read_bytes() == b""


def test_existing_destination_is_replaced(tmp_path, source_file):
    packed, restored = tmp_path / "packed", tmp_path / "restored"
    packed.write_bytes(b"old contents that are longer than nothing" * 1000)
    restored.write_bytes(b"stale" * 100000)
    hce.compress_file(source_file, packed)
    hce.decompress_file(packed, restored)
    assert restored.read_bytes() == source_file.read_bytes()


def test_decompress_file_reads_memory_archives(tmp_path):
    data = testdata.log(3, 20000)
    for opts in ({}, {"structural": True}):
        (tmp_path / "a").write_bytes(hce.compress(data, **opts))
        hce.decompress_file(tmp_path / "a", tmp_path / "b")
        assert (tmp_path / "b").read_bytes() == data


@pytest.mark.parametrize("fn", ["compress_file", "decompress_file"])
def test_missing_source(tmp_path, fn):
    with pytest.raises(hce.HceIOError) as info:
        getattr(hce, fn)(tmp_path / "nope", tmp_path / "out")
    assert info.value.errno == errno.ENOENT
    assert isinstance(info.value, OSError)
    assert "nope" in str(info.value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fn", ["compress_file", "decompress_file"])
def test_directory_as_source(tmp_path, fn):
    with pytest.raises(hce.HceIOError, match="not a regular file"):
        getattr(hce, fn)(tmp_path, tmp_path / "out")


@pytest.mark.parametrize("structural", [False, True])
def test_unwritable_destination(tmp_path, source_file, structural):
    with pytest.raises(hce.HceIOError) as info:
        hce.compress_file(source_file, tmp_path / "missing-dir" / "out", structural=structural)
    assert info.value.errno == errno.ENOENT


def test_corrupt_archive_leaves_no_output(tmp_path, source_file):
    packed, restored = tmp_path / "packed", tmp_path / "restored"
    for structural in (False, True):
        hce.compress_file(source_file, packed, structural=structural)
        damaged = bytearray(packed.read_bytes())
        damaged[len(damaged) // 2] ^= 0x10
        packed.write_bytes(bytes(damaged))
        with pytest.raises(hce.CorruptInputError):
            hce.decompress_file(packed, restored)
        assert not restored.exists()


def test_non_archive_leaves_no_output(tmp_path, source_file):
    with pytest.raises(hce.CorruptInputError) as info:
        hce.decompress_file(source_file, tmp_path / "restored")
    assert info.value.code == "bad_magic"
    assert not (tmp_path / "restored").exists()


def test_file_argument_validation(tmp_path, source_file):
    with pytest.raises(ValueError, match="level"):
        hce.compress_file(source_file, tmp_path / "o", level=9)
    with pytest.raises(ValueError, match="threads"):
        hce.compress_file(source_file, tmp_path / "o", threads=0)
    with pytest.raises(ValueError, match="language"):
        hce.compress_file(source_file, tmp_path / "o", structural=True, language="cobol")
    with pytest.raises(ValueError, match="max_memory"):
        hce.decompress_file(source_file, tmp_path / "o", max_memory=1)
    with pytest.raises(TypeError):
        hce.compress_file(12, tmp_path / "o")  # type: ignore[arg-type]
    assert not (tmp_path / "o").exists()


def test_decompress_file_memory_limit(tmp_path, source_file):
    packed = tmp_path / "packed"
    hce.compress_file(source_file, packed, level=4)
    with pytest.raises(hce.LimitExceededError):
        hce.decompress_file(packed, tmp_path / "restored", max_memory=16 << 20)
    assert not (tmp_path / "restored").exists()


# ------------------------------------------------- transactional outputs ---


@pytest.mark.parametrize("structural", [False, True])
def test_failed_decompress_keeps_existing_destination(tmp_path, source_file, structural):
    packed, dest = tmp_path / "packed", tmp_path / "dest"
    hce.compress_file(source_file, packed, structural=structural)
    damaged = bytearray(packed.read_bytes())
    damaged[len(damaged) // 2] ^= 0x10
    (tmp_path / "bad").write_bytes(bytes(damaged))
    (tmp_path / "short").write_bytes(packed.read_bytes()[:-9])
    dest.write_bytes(b"PRECIOUS")
    for bad in ("bad", "short"):
        with pytest.raises(hce.CorruptInputError):
            hce.decompress_file(tmp_path / bad, dest)
        assert dest.read_bytes() == b"PRECIOUS"
    with pytest.raises(hce.CorruptInputError):
        hce.decompress_file(source_file, dest)  # not an archive
    assert dest.read_bytes() == b"PRECIOUS"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad", "dest", "module.py", "packed", "short"]


@pytest.mark.parametrize("fn", ["compress_file", "decompress_file"])
def test_same_file_is_refused(tmp_path, source_file, fn):
    target = source_file
    if fn == "decompress_file":
        target = tmp_path / "packed"
        hce.compress_file(source_file, target)
    before = target.read_bytes()
    aliases = [target, tmp_path / "." / target.name]
    os.link(target, tmp_path / "hard")
    aliases.append(tmp_path / "hard")
    if hasattr(os, "symlink"):
        os.symlink(target.name, tmp_path / "soft")
        aliases.append(tmp_path / "soft")
    for alias in aliases:
        for structural in (False, True):
            kwargs = {"structural": structural} if fn == "compress_file" else {}
            with pytest.raises(ValueError, match="same file"):
                getattr(hce, fn)(target, alias, **kwargs)
            assert target.read_bytes() == before


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symbolic links")
def test_symlink_destination_is_replaced_not_followed(tmp_path, source_file):
    (tmp_path / "target").write_bytes(b"TARGET")
    os.symlink("target", tmp_path / "link")
    hce.compress_file(source_file, tmp_path / "link")
    assert not (tmp_path / "link").is_symlink()
    assert (tmp_path / "target").read_bytes() == b"TARGET"
    assert hce.decompress(bytes((tmp_path / "link").read_bytes())) == source_file.read_bytes()


def test_destination_keeps_permissions(tmp_path, source_file):
    dest = tmp_path / "dest"
    dest.write_bytes(b"x")
    os.chmod(dest, 0o640)
    hce.compress_file(source_file, dest)
    assert dest.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("fn, structural", [("compress_file", False), ("compress_file", True),
                                            ("decompress_file", False), ("decompress_file", True)])
def test_write_failure_keeps_existing_destination(tmp_path, source_file, fn, structural):
    src = source_file
    if fn == "decompress_file":
        src = tmp_path / "packed"
        hce.compress_file(source_file, src, structural=structural)
    dest = tmp_path / "dest"
    dest.write_bytes(b"PRECIOUS")
    kwargs = "structural=True" if fn == "compress_file" and structural else ""
    script = (
        "import resource, signal, sys, hce\n"
        "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (2048, 2048))\n"
        "try:\n"
        f"    hce.{fn}({str(src)!r}, {str(dest)!r}, {kwargs})\n"
        "except hce.HceIOError as e:\n"
        "    sys.exit(0 if e.errno else 3)\n"
        "sys.exit(2)\n")
    rc = subprocess.run([sys.executable, "-c", script], env=dict(os.environ)).returncode
    assert rc == 0
    assert dest.read_bytes() == b"PRECIOUS"
    assert not [p for p in tmp_path.iterdir() if ".hce-" in p.name]


# ----------------------------------------------------------- max_output ---


@pytest.mark.parametrize("structural", [False, True])
def test_max_output_boundaries(tmp_path, source_file, structural):
    packed, dest = tmp_path / "packed", tmp_path / "dest"
    hce.compress_file(source_file, packed, structural=structural)
    size = source_file.stat().st_size
    assert hce.decompress_file(packed, dest, max_output=size).output_bytes == size
    dest.write_bytes(b"PRECIOUS")
    with pytest.raises(hce.LimitExceededError) as info:
        hce.decompress_file(packed, dest, max_output=size - 1)
    assert info.value.code == "limit_output"
    assert dest.read_bytes() == b"PRECIOUS"
    data = packed.read_bytes()
    assert len(hce.decompress(data, max_output=size)) == size
    with pytest.raises(hce.LimitExceededError):
        hce.decompress(data, max_output=10)


@pytest.mark.parametrize("structural", [False, True])
def test_max_output_empty(structural):
    assert hce.decompress(hce.compress(b"", structural=structural), max_output=0) == b""
    with pytest.raises(hce.LimitExceededError):
        hce.decompress(hce.compress(b"x", structural=structural), max_output=0)
