"""Round-trip fidelity: decompress(compress(x)) == x for every codec, level and input shape."""

from __future__ import annotations

import array
import concurrent.futures

import pytest
from hypothesis import given, strategies as st

import hce
import testdata

CODECS = {
    "framed": {},
    "structural": {"structural": True, "language": "generic"},
    "structural-python": {"structural": True, "language": "python"},
}


@pytest.mark.parametrize("codec", list(CODECS))
@pytest.mark.parametrize("kind", ["source", "log", "text", "binary", "empty"])
def test_corpora(codec, kind):
    data = testdata.generate(kind, 11, 20000)
    packed = hce.compress(data, **CODECS[codec])
    assert hce.decompress(packed) == data
    stats = hce.verify(packed)
    assert stats.verified and stats.output_bytes == len(data) and stats.input_bytes == len(packed)


@pytest.mark.parametrize("structural", [False, True])
@pytest.mark.parametrize("level", range(5))
def test_every_level(structural, level):
    data = testdata.source(level, 12000)
    packed = hce.compress(data, level=level, structural=structural)
    assert hce.decompress(packed) == data


@pytest.mark.parametrize("language", hce.LANGUAGES)
def test_every_language(language):
    data = testdata.source(3, 8000)
    assert hce.decompress(hce.compress(data, structural=True, language=language)) == data


@pytest.mark.parametrize("kind", ["source", "log", "text"])
def test_compressible_input_shrinks(kind):
    data = testdata.generate(kind, 5, 50000)
    for opts in CODECS.values():
        assert len(hce.compress(data, **opts)) < len(data) // 3


def test_incompressible_input_barely_expands():
    data = testdata.binary(9, 100000)
    assert len(hce.compress(data)) < len(data) * 1.01 + 256
    assert len(hce.compress(data, structural=True)) < len(data) * 1.25 + 1024


@pytest.mark.parametrize("codec", list(CODECS))
def test_output_is_deterministic(codec):
    data = testdata.log(4, 30000)
    assert hce.compress(data, **CODECS[codec]) == hce.compress(data, **CODECS[codec])


@pytest.mark.parametrize("structural", [False, True])
def test_output_independent_of_threads(structural, cpus):
    # Several frames so the parallel path really splits the work.
    data = testdata.source(6, 300000) if structural else testdata.log(6, 300000)
    ref = hce.compress(data, structural=structural, threads=1)
    for threads in sorted({2, 3, cpus}):
        assert hce.compress(data, structural=structural, threads=threads) == ref
        assert hce.decompress(ref, threads=threads) == data


@pytest.mark.slow
def test_multi_frame_stream():
    # Beyond the default 8 MiB frame: the stream has several independent frames.
    data = testdata.log(8, 9 << 20)
    packed = hce.compress(data, level=0, threads=4)
    stats = hce.verify(packed, threads=4)
    assert stats.details["frames"] == 2
    assert hce.decompress(packed, threads=4) == data


@pytest.mark.parametrize("codec", list(CODECS))
def test_buffer_types(codec):
    data = testdata.text(2, 5000)
    opts = CODECS[codec]
    ref = hce.compress(data, **opts)
    assert hce.compress(bytearray(data), **opts) == ref
    assert hce.compress(memoryview(data), **opts) == ref
    assert hce.compress(memoryview(b"xx" + data + b"yy")[2:-2], **opts) == ref
    assert hce.decompress(bytearray(ref)) == data
    assert hce.decompress(memoryview(ref)) == data
    assert hce.verify(memoryview(bytearray(ref))).verified


def test_multibyte_items_are_compressed_as_raw_bytes():
    values = array.array("I", range(1000))
    assert hce.decompress(hce.compress(values)) == values.tobytes()


def test_returns_bytes():
    assert type(hce.compress(bytearray(b"abc"))) is bytes
    assert type(hce.decompress(hce.compress(b"abc"))) is bytes


def test_input_buffer_is_not_modified_and_can_change_afterwards():
    buf = bytearray(testdata.text(1, 4000))
    snapshot = bytes(buf)
    packed = hce.compress(buf)
    assert buf == snapshot
    buf[:] = b"\0" * len(buf)
    assert hce.decompress(packed) == snapshot


def test_max_output_exact_size_is_allowed():
    data = testdata.text(3, 3000)
    assert hce.decompress(hce.compress(data), max_output=len(data)) == data
    assert hce.decompress(hce.compress(data, structural=True), max_output=len(data)) == data


def test_concurrent_calls_from_threads():
    inputs = [testdata.generate(k, s, 30000) for s in range(3) for k in ("source", "log", "text", "binary")]

    def work(data: bytes) -> bool:
        return hce.decompress(hce.compress(data)) == data and \
            hce.decompress(hce.compress(data, structural=True)) == data

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        assert all(pool.map(work, inputs))


# ------------------------------------------------------------- properties ---

@given(st.binary(max_size=4096), st.integers(0, 4))
def test_property_framed(data, level):
    assert hce.decompress(hce.compress(data, level=level)) == data


@given(st.binary(max_size=4096))
def test_property_structural_bytes(data):
    assert hce.decompress(hce.compress(data, structural=True)) == data


@given(st.text(max_size=2000), st.sampled_from(hce.LANGUAGES))
def test_property_structural_text(text, language):
    data = text.encode("utf-8", "surrogatepass")
    assert hce.decompress(hce.compress(data, structural=True, language=language)) == data


@given(st.lists(st.sampled_from([b"if", b" ", b"\n", b"(", b")", b"{", b"}", b"x", b"42", b"\"s\"", b"//c",
                                 b"/*", b"*/", b"\t", b"'", b"#", b"\\", b"\r\n"]), max_size=400))
def test_property_structural_token_soup(parts):
    data = b"".join(parts)
    assert hce.decompress(hce.compress(data, structural=True, language="c")) == data
