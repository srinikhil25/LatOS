"""Tests for `latos.ingestion.labeling.normalize`."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from latos.ingestion.labeling.normalize import normalize, tokens

# ---------------------------------------------------------------------------
# Headline cases — what 2C will ultimately compare
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # The Dhivya regression: same sample written with parentheses
        # vs. just a space. Both must collapse to the same form.
        ("CS Pure", "CS (Pure)"),
        ("CS Pure", "cs_pure"),
        ("CS Pure", "CS-Pure"),
        ("CS Pure", "cs.pure"),
        ("CS-1", "cs1"),
        ("CS-1", "CS_1"),
        ("CS-1", "CS 1"),
        ("CSCBI 1", "CS-CBI-1"),
        ("CSCBI-1", "cscbi1"),
        ("MX-001", "mx001"),
        ("Sample: MX-1", "MX-1"),
        ("specimen_007", "MX-007"),  # not equal — different prefix words
    ],
)
def test_pairs_with_known_relationship(a: str, b: str):
    """Equality matrix for the canonical Dhivya/MXene cases."""
    na, nb = normalize(a), normalize(b)
    if a in {"specimen_007"}:
        # Sanity: these intentionally don't normalize together because
        # the *content* after the prefix is genuinely different.
        assert na != nb
    else:
        assert na == nb, f"normalize({a!r}) != normalize({b!r}): {na!r} vs {nb!r}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CS Pure", "cspure"),
        ("CS-1", "cs1"),
        ("Sample: MX-1", "mx1"),
        ("specimen 007", "007"),  # bare specimen prefix gone
        ("Cs3Bi2I9", "cs3bi2i9"),  # chemistry-style — digits preserved
        # Full-width Japanese-IME-pasted sample name. NFKC folds to
        # ASCII before lowercasing.
        ("ＣＳ１", "cs1"),  # noqa: RUF001
        # Subscript digits (chemistry rendering): Cs₃Bi₂I₉ → cs3bi2i9
        ("Cs₃Bi₂I₉", "cs3bi2i9"),
        ("", ""),
        ("   ", ""),
        ("---", ""),
        ("___", ""),
        ("()()", ""),
    ],
)
def test_explicit_outputs(raw: str, expected: str):
    assert normalize(raw) == expected


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(st.text())
def test_idempotent(s: str):
    """`normalize` is idempotent — applying it twice gives the same result."""
    once = normalize(s)
    twice = normalize(once)
    assert once == twice


@given(st.text())
def test_output_is_lowercase(s: str):
    out = normalize(s)
    assert out == out.lower()


@given(st.text())
def test_output_has_no_separator_class_chars(s: str):
    out = normalize(s)
    forbidden = " \t\n\r\f\v-_().:[]/\\"
    for ch in forbidden:
        assert ch not in out, f"normalize({s!r}) → {out!r} contains {ch!r}"


@given(st.text())
def test_digits_in_input_survive_unless_in_prefix(s: str):
    """Every ASCII digit in the input shows up in the output, except
    digits inside a leading "sample…" prefix that gets stripped.

    We can't compute "what survived the prefix strip" from the input
    alone, so we phrase the property as an upper bound: the count of
    digits in `normalize(s)` is ≤ the count in NFKC-folded `s`.
    """
    import unicodedata as ud

    folded = ud.normalize("NFKC", s)
    in_count = sum(1 for ch in folded if ch.isdigit())
    out_count = sum(1 for ch in normalize(s) if ch.isdigit())
    assert out_count <= in_count


# ---------------------------------------------------------------------------
# Type-error contract
# ---------------------------------------------------------------------------


class TestTypeContract:
    def test_non_str_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize(123)  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Prefix-stripping nuances
# ---------------------------------------------------------------------------


class TestPrefixStripping:
    def test_sample_prefix_with_colon(self):
        assert normalize("Sample: MX-1") == "mx1"

    def test_specimen_prefix_with_underscore(self):
        assert normalize("Specimen_07") == "07"

    def test_sample_prefix_in_middle_not_stripped(self):
        # "data_sample_5" should NOT have the inner "sample" stripped.
        assert normalize("data_sample_5") == "datasample5"

    def test_prefix_word_only_yields_empty(self):
        assert normalize("Sample") == ""
        assert normalize("specimen:") == ""

    def test_prefix_case_insensitive(self):
        assert normalize("SAMPLE-007") == "007"
        assert normalize("sAmPle 007") == "007"


# ---------------------------------------------------------------------------
# Unicode handling
# ---------------------------------------------------------------------------


class TestUnicode:
    def test_subscript_digits_fold_to_ascii(self):
        # Cs₃Bi₂I₉ — NFKC compatibility decomposition handles this.
        assert normalize("Cs₃Bi₂I₉") == "cs3bi2i9"

    def test_fullwidth_letters_fold_to_ascii(self):
        # Excel imports from Japanese keyboards often use full-width.
        assert normalize("ＣＳ１") == "cs1"  # noqa: RUF001

    def test_combining_accent_folds_into_precomposed(self):
        # 'é' as e + combining acute → precomposed é. Both should
        # normalize equally (and the cluster phase will pick a canonical).
        decomposed = "é"
        precomposed = "é"
        assert normalize(decomposed) == normalize(precomposed)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokens:
    def test_basic_split(self):
        assert tokens("MX-001 batch 3") == ("mx001", "batch", "3")

    def test_empty_input(self):
        assert tokens("") == ()
        assert tokens("   ") == ()

    def test_all_separators(self):
        assert tokens("---___") == ()

    def test_leading_sample_word_dropped(self):
        assert tokens("Sample MX 1") == ("mx", "1")
        assert tokens("specimen 7") == ("7",)

    def test_inline_sample_word_dropped(self):
        # `normalize("sample")` returns "" because the prefix regex
        # matches the whole string. So when a "sample" word appears
        # mid-string and `tokens()` normalizes each piece, that piece
        # collapses to "" and is filtered out. This is the right
        # behavior for the cluster phase — bare "sample" is noise.
        assert tokens("data sample 5") == ("data", "5")

    def test_tokens_concatenated_equals_normalize_when_no_sample_word(self):
        # Property: joining tokens reproduces normalize *only* when the
        # input has no inline `sample`/`specimen` words — otherwise the
        # token-side normalization filters them out while the string-
        # side normalization preserves them in the concatenated form.
        for raw in ["CS Pure", "MX-001 batch", "ＣＳ_１_2"]:  # noqa: RUF001
            assert "".join(tokens(raw)) == normalize(raw)
