"""Error handling: bad arguments, damaged and hostile input, resource limits.

Every failure must raise, never return wrong bytes: the decoders verify CRCs
and sizes before handing anything back.
"""

from __future__ import annotations

import errno
import random

import pytest
from hypothesis import given, strategies as st

import hce
import testdata

CODECS = {"framed": {}, "structural": {"structural": True, "language": "python"}}


@pytest.fixture(scope="module", params=list(CODECS))
def sample(request):
    data = testdata.source(31, 6000)
    return request.param, data, hce.compress(data, **CODECS[request.param])


# ------------------------------------------------------------- hierarchy ---

def test_exception_hierarchy():
    assert issubclass(hce.HceError, Exception)
    for cls in (hce.CorruptInputError, hce.LimitExceededError, hce.UnsupportedFormatError, hce.HceIOError):
        assert issubclass(cls, hce.HceError)
    assert issubclass(hce.CorruptInputError, ValueError)
    assert issubclass(hce.HceIOError, OSError)
    assert not issubclass(hce.LimitExceededError, ValueError)


def test_exception_attributes():
    with pytest.raises(hce.CorruptInputError) as info:
        hce.decompress(b"not an archive")
    e = info.value
    assert e.code == "bad_magic"
    assert e.errno == 0
    assert "bad_magic" in str(e) and str(e).startswith("decompress")


def test_exceptions_are_catchable_as_builtin_types():
    with pytest.raises(ValueError):
        hce.decompress(b"junk")
    with pytest.raises(OSError):
        hce.compress_file("/nonexistent/hce/input", "/nonexistent/hce/output")


# ------------------------------------------------------------- arguments ---

@pytest.mark.parametrize("kwargs, match", [
    ({"level": -1}, "level"), ({"level": 5}, "level"),
    ({"threads": 0}, "threads"), ({"threads": 257}, "threads"),
    ({"structural": True, "language": "cobol"}, "unknown language"),
    ({"structural": True, "language": "auto"}, "auto"),
])
def test_compress_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hce.compress(b"data", **kwargs)


@pytest.mark.parametrize("fn", [hce.decompress, hce.verify])
@pytest.mark.parametrize("kwargs, match", [
    ({"threads": 0}, "threads"), ({"max_memory": 0}, "max_memory"), ({"max_memory": (16 << 20) - 1}, "max_memory"),
])
def test_decompress_bad_arguments(fn, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fn(hce.compress(b"x"), **kwargs)


@pytest.mark.parametrize("value", ["text", 123, None, [1, 2], 1.5])
def test_non_buffer_input(value):
    with pytest.raises(TypeError):
        hce.compress(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        hce.decompress(value)  # type: ignore[arg-type]


def test_keyword_only_options():
    with pytest.raises(TypeError):
        hce.compress(b"x", 3)  # type: ignore[misc]
    with pytest.raises(TypeError):
        hce.compress(b"x", lvl=3)  # type: ignore[call-arg]


def test_non_contiguous_buffer():
    strided = memoryview(bytes(range(200)))[::2]
    with pytest.raises(ValueError, match="contiguous"):
        hce.compress(strided)
    with pytest.raises(ValueError, match="contiguous"):
        hce.decompress(strided)


def test_negative_max_output_is_rejected():
    with pytest.raises((ValueError, TypeError, OverflowError)):
        hce.decompress(hce.compress(b"x"), max_output=-1)


# ---------------------------------------------------------- damaged input ---

@pytest.mark.parametrize("data", [b"", b"\x89", b"\x89HC", b"HCS"])
def test_short_input_is_truncated(data):
    with pytest.raises(hce.CorruptInputError) as info:
        hce.decompress(data)
    assert info.value.code in ("truncated", "bad_magic")


def test_empty_input():
    with pytest.raises(hce.CorruptInputError) as info:
        hce.decompress(b"")
    assert info.value.code == "truncated"
    with pytest.raises(hce.CorruptInputError):
        hce.verify(b"")


@pytest.mark.parametrize("data", [b"PK\x03\x04" + bytes(40), b"\x1f\x8b\x08" + bytes(40), b"\xfd7zXZ\x00" + bytes(40),
                                  b"hello world, plain text"])
def test_foreign_format_is_bad_magic(data):
    assert hce.detect_format(data) is None
    with pytest.raises(hce.CorruptInputError) as info:
        hce.decompress(data)
    assert info.value.code == "bad_magic"


def test_every_truncation_raises(sample):
    codec, data, packed = sample
    # Every prefix of a small stream, and a spread of prefixes of the larger one.
    lengths = range(len(packed)) if len(packed) < 600 else sorted(set(range(0, 64)) |
                                                                  set(range(0, len(packed), 37)))
    for n in lengths:
        with pytest.raises(hce.CorruptInputError):
            hce.decompress(packed[:n])


def test_trailing_bytes_are_rejected(sample):
    codec, data, packed = sample
    with pytest.raises(hce.CorruptInputError):
        hce.decompress(packed + b"\0")
    with pytest.raises(hce.CorruptInputError):
        hce.decompress(packed + packed[:16])


def test_single_bit_flips_are_rejected(sample):
    # Every coded bit is covered by a CRC or a checked field, so any single flip
    # must raise. (An exhaustive sweep over every bit of both codecs' streams
    # holds; this samples it: every header bit, then random payload bits.)
    codec, data, packed = sample
    rng = random.Random(1234)
    bits = list(range(48 * 8)) + rng.sample(range(48 * 8, len(packed) * 8), 400)
    for bit in bits:
        damaged = bytearray(packed)
        damaged[bit // 8] ^= 1 << (bit % 8)
        with pytest.raises((hce.CorruptInputError, hce.UnsupportedFormatError)):
            hce.decompress(bytes(damaged))


def test_payload_damage_is_corrupt_input(sample):
    codec, data, packed = sample
    for pos in (len(packed) // 3, len(packed) // 2, len(packed) - 8):
        damaged = bytearray(packed)
        damaged[pos] ^= 0xFF
        with pytest.raises(hce.CorruptInputError):
            hce.decompress(bytes(damaged))
        with pytest.raises(hce.CorruptInputError):
            hce.verify(bytes(damaged))


def test_swapping_codecs_magic_is_detected():
    data = testdata.source(2, 4000)
    framed, structural = hce.compress(data), hce.compress(data, structural=True)
    with pytest.raises(hce.HceError):
        hce.decompress(structural[:4] + framed[4:])
    with pytest.raises(hce.HceError):
        hce.decompress(framed[:4] + structural[4:])


def test_unsupported_version():
    with pytest.raises(hce.UnsupportedFormatError) as info:
        hce.decompress(b"\x89HCZ" + bytes(60))
    assert info.value.code == "unsupported_version"


@given(st.binary(max_size=512))
def test_property_garbage_never_crashes(blob):
    try:
        hce.decompress(blob)
    except hce.HceError:
        pass


@given(st.binary(min_size=1, max_size=256))
def test_property_garbage_after_valid_magic(tail):
    for magic in (b"\x89HCZ", b"HCS1"):
        try:
            hce.decompress(magic + tail)
        except hce.HceError:
            pass


@given(st.integers(0, 10**6), st.integers(1, 255))
def test_property_random_byte_damage(seed, xor):
    data = testdata.log(seed, 1500)
    packed = bytearray(hce.compress(data, level=0))
    packed[seed % len(packed)] ^= xor
    try:
        assert hce.decompress(bytes(packed)) == data
    except hce.HceError:
        pass


# ---------------------------------------------------------------- limits ---

@pytest.mark.parametrize("structural", [False, True])
def test_max_output(structural):
    data = testdata.text(4, 20000)
    packed = hce.compress(data, structural=structural)
    with pytest.raises(hce.LimitExceededError) as info:
        hce.decompress(packed, max_output=len(data) - 1)
    assert info.value.code == "limit_output"
    with pytest.raises(hce.LimitExceededError):
        hce.decompress(packed, max_output=0)


def test_max_memory_for_high_level():
    packed = hce.compress(testdata.text(5, 5000), level=4)
    with pytest.raises(hce.LimitExceededError) as info:
        hce.decompress(packed, max_memory=16 << 20)
    assert info.value.code == "limit_memory"
    with pytest.raises(hce.LimitExceededError):
        hce.verify(packed, max_memory=16 << 20)
    assert hce.decompress(packed, max_memory=1 << 30) == testdata.text(5, 5000)


def test_io_error_carries_errno(tmp_path):
    with pytest.raises(hce.HceIOError) as info:
        hce.decompress_file(tmp_path / "missing.hce", tmp_path / "out")
    assert info.value.errno == errno.ENOENT
    assert info.value.code == "io_error"
    assert info.value.stage == "open"
