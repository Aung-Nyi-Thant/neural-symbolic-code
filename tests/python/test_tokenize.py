"""The structural tokenizer exposed by hce.tokenize()."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

import hce
import testdata

# The six token types of hce::structural::TokenType, in enum order.
TYPES = {"KEYWORD", "IDENTIFIER", "LITERAL", "OPERATOR", "STRUCTURE", "INDENTATION"}


def test_simple_python():
    tokens = hce.tokenize("def f(x):\n    return x\n", language="python")
    assert all(isinstance(t, hce.Token) for t in tokens)
    assert tokens[0] == hce.Token("KEYWORD", b"def", 1, 0)
    by_value = {t.value: t for t in tokens}
    assert by_value[b"f"].type == "IDENTIFIER"
    assert by_value[b"("].type == "STRUCTURE"
    assert by_value[b"return"].type == "KEYWORD" and by_value[b"return"].line == 2
    assert by_value[b"    "].type == "INDENTATION"
    assert by_value[b"\n"].type == "STRUCTURE"


def test_comments_are_single_literals():
    tokens = hce.tokenize(b"/* block */ x // line\n", language="c")
    assert tokens[0] == hce.Token("LITERAL", b"/* block */", 1, 0)
    assert hce.Token("LITERAL", b"// line", 1, 3) in tokens


@pytest.mark.parametrize("language, keywords", [
    ("python", {b"def", b"elif"}), ("rust", {b"fn", b"let", b"impl"}), ("go", {b"func"}),
    ("javascript", {b"let"}), ("c", set()),
    ("generic", {b"def", b"elif", b"fn", b"let", b"impl", b"func"}),  # generic knows every table's words
])
def test_keywords_depend_on_language(language, keywords):
    types = {t.value: t.type for t in hce.tokenize(b"def elif fn let impl func", language=language)}
    assert {v for v, kind in types.items() if kind == "KEYWORD"} == keywords


def test_str_is_utf8():
    text = "s = 'héllo ✓'\n"
    assert hce.tokenize(text) == hce.tokenize(text.encode("utf-8"))
    assert hce.detokenize(hce.tokenize(text)) == text.encode("utf-8")


def test_buffer_inputs():
    src = testdata.source(4, 3000)
    ref = hce.tokenize(src)
    assert hce.tokenize(bytearray(src)) == ref
    assert hce.tokenize(memoryview(src)) == ref


def test_empty():
    assert hce.tokenize(b"") == []
    assert hce.detokenize([]) == b""


@pytest.mark.parametrize("language", hce.LANGUAGES)
@pytest.mark.parametrize("kind", ["source", "log", "binary"])
def test_lossless_on_corpora(language, kind):
    data = testdata.generate(kind, 7, 6000)
    tokens = hce.tokenize(data, language=language)
    assert hce.detokenize(tokens) == data
    assert {t.type for t in tokens} <= TYPES


def test_line_numbers_and_ids():
    data = testdata.source(8, 8000)
    tokens = hce.tokenize(data, language="python")
    assert tokens[0].line == 1
    assert all(a.line <= b.line for a, b in zip(tokens, tokens[1:]))
    # Line numbers count the newlines before each token.
    offset = 0
    for t in tokens:
        assert t.line == data.count(b"\n", 0, offset) + 1
        offset += len(t.value)
    # Ids intern (type, value) in order of first appearance.
    seen = {}
    for t in tokens:
        expected = seen.setdefault((t.type, t.value), len(seen))
        assert t.id == expected


def test_unknown_language():
    with pytest.raises(ValueError, match="unknown language 'cobol'"):
        hce.tokenize(b"x", language="cobol")
    with pytest.raises(ValueError):
        hce.tokenize(b"x", language="auto")


def test_wrong_type():
    with pytest.raises(TypeError):
        hce.tokenize(42)  # type: ignore[arg-type]


@given(st.binary(max_size=3000), st.sampled_from(hce.LANGUAGES))
def test_property_lossless_bytes(data, language):
    assert hce.detokenize(hce.tokenize(data, language=language)) == data


@given(st.text(max_size=1000))
def test_property_lossless_text(text):
    assert hce.detokenize(hce.tokenize(text, language="python")) == text.encode("utf-8", "surrogatepass")
