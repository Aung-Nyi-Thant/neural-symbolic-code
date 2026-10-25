"""The bindings and the hce command-line tool agree byte for byte and on errors."""

from __future__ import annotations

import pytest

import hce
import testdata

pytestmark = pytest.mark.cli

# Exit statuses documented in `hce --help`.
EXIT_OK, EXIT_USAGE, EXIT_CORRUPT, EXIT_LIMIT, EXIT_UNSUPPORTED = 0, 1, 2, 4, 5


def test_cli_version(run_cli):
    proc = run_cli("--version")
    assert proc.returncode == 0
    assert hce.__version__ in proc.stdout.decode()


@pytest.mark.parametrize("level", [0, 2, 4])
@pytest.mark.parametrize("threads", [1, 3])
def test_framed_output_identical(run_cli, tmp_path, level, threads):
    # From a file the CLI records the content size in the header, as compress() does.
    data = testdata.log(level, 40000)
    src, out = tmp_path / "in.log", tmp_path / "out.hce"
    src.write_bytes(data)
    proc = run_cli("-l", str(level), "-T", str(threads), "-i", str(src), "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.read_bytes() == hce.compress(data, level=level, threads=threads)


def test_streamed_input_differs_only_by_content_size(run_cli):
    # From a pipe the size is unknown: the CLI omits it, and either stream decodes.
    data = testdata.log(1, 20000)
    piped = run_cli(stdin=data).stdout
    assert piped != hce.compress(data)
    assert hce.decompress(piped) == data


@pytest.mark.parametrize("language", ["generic", "python", "cpp"])
def test_structural_output_identical(run_cli, language):
    data = testdata.source(2, 30000)
    proc = run_cli("--structural", "--lang", language, stdin=data)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == hce.compress(data, structural=True, language=language)


@pytest.mark.parametrize("opts", [[], ["--structural", "--lang", "python"]])
def test_cli_decodes_binding_output_and_back(run_cli, opts):
    data = testdata.source(3, 20000)
    proc = run_cli("-d", stdin=hce.compress(data, structural=bool(opts), language="python"))
    assert proc.returncode == 0 and proc.stdout == data
    packed = run_cli(*opts, stdin=data).stdout
    assert hce.decompress(packed) == data


def test_cli_and_binding_agree_on_files(run_cli, tmp_path):
    src = tmp_path / "x.py"
    src.write_bytes(testdata.source(4, 30000))
    run_cli("--structural", "-i", str(src), "-o", str(tmp_path / "cli.hcs"))
    hce.compress_file(src, tmp_path / "py.hcs", structural=True)
    assert (tmp_path / "cli.hcs").read_bytes() == (tmp_path / "py.hcs").read_bytes()


def test_verify_mode(run_cli):
    packed = hce.compress(testdata.text(1, 5000))
    assert run_cli("-t", stdin=packed).returncode == EXIT_OK


def _corrupt(packed: bytes) -> bytes:
    b = bytearray(packed)
    b[len(b) // 2] ^= 0x01
    return bytes(b)


@pytest.mark.parametrize("structural", [False, True])
def test_corrupt_input_status_matches_exception(run_cli, structural):
    damaged = _corrupt(hce.compress(testdata.source(5, 20000), structural=structural))
    assert run_cli("-d", "-q", stdin=damaged).returncode == EXIT_CORRUPT
    with pytest.raises(hce.CorruptInputError):
        hce.decompress(damaged)


def test_bad_magic_status(run_cli):
    assert run_cli("-d", "-q", stdin=b"plain text, not an archive").returncode == EXIT_CORRUPT
    with pytest.raises(hce.CorruptInputError):
        hce.decompress(b"plain text, not an archive")


def test_limit_status_matches_exception(run_cli):
    packed = hce.compress(testdata.text(2, 50000))
    assert run_cli("-d", "-q", "--max-output", "1K", stdin=packed).returncode == EXIT_LIMIT
    with pytest.raises(hce.LimitExceededError):
        hce.decompress(packed, max_output=1024)


def test_unsupported_status_matches_exception(run_cli):
    future = b"\x89HCZ" + bytes(60)
    assert run_cli("-d", "-q", stdin=future).returncode == EXIT_UNSUPPORTED
    with pytest.raises(hce.UnsupportedFormatError):
        hce.decompress(future)


def test_usage_error_status(run_cli):
    assert run_cli("-l", "9", "-q").returncode == EXIT_USAGE
    with pytest.raises(ValueError):
        hce.compress(b"", level=9)
