"""Tests for `latos.ingestion.labeling.suggestions.suggest_merges`."""

from __future__ import annotations

from dataclasses import dataclass

from latos.ingestion.labeling.suggestions import MergeSuggestion, suggest_merges


@dataclass(frozen=True)
class _Sample:
    id: str
    canonical_name: str


def _samples(*names: str) -> list[_Sample]:
    return [_Sample(id=f"id{i}", canonical_name=n) for i, n in enumerate(names)]


def _pairs(suggestions: list[MergeSuggestion]) -> set[frozenset[str]]:
    return {frozenset({s.target_name, s.source_name}) for s in suggestions}


class TestDigitGuard:
    def test_doping_series_never_suggested(self):
        # The single most important guard: a doping series must never be
        # proposed for merging, however similar the prefixes are.
        out = suggest_merges(_samples("CS-1", "CS-3", "CS-5"))
        assert out == []

    def test_cscbi_series_never_suggested(self):
        out = suggest_merges(_samples("CSCBI-1", "CSCBI-3", "CSCBI 5"))
        assert out == []

    def test_pure_vs_numbered_not_suggested(self):
        # "CS (Pure)" (no digit) vs "CS-1" (digit 1) — different samples.
        out = suggest_merges(_samples("CS (Pure)", "CS-1"))
        assert out == []


class TestPrefixNoise:
    def test_researcher_prefix_suggested(self):
        out = suggest_merges(_samples("CSCBI-1", "Dr.MN-dhivya-cscbi1"))
        assert len(out) == 1
        s = out[0]
        # Cleaner (shorter) name is the merge target.
        assert s.target_name == "CSCBI-1"
        assert s.source_name == "Dr.MN-dhivya-cscbi1"
        assert s.confidence == "high"  # exact substring containment

    def test_typo_inside_prefix_suggested_as_medium(self):
        # "cskbi3" (typo) wrapped in a researcher prefix vs "CSCBI-3".
        out = suggest_merges(_samples("CSCBI-3", "Dr.MN-dhivya-cskbi3"))
        assert len(out) == 1
        assert _pairs(out) == {frozenset({"CSCBI-3", "Dr.MN-dhivya-cskbi3"})}


class TestNoFalsePositives:
    def test_distinct_materials_not_suggested(self):
        out = suggest_merges(_samples("CS (Pure)", "cUsE3", "Images"))
        assert out == []

    def test_distinct_short_name_not_suggested(self):
        # "CS" vs "cUsE3" — not a substring, different digits → no suggestion.
        out = suggest_merges(_samples("CS", "cUsE3"))
        assert out == []


class TestShortNameContainment:
    def test_cs_pure_suggested(self):
        # "CS" ⊂ "CS (Pure)" — same undoped sample, two spellings.
        out = suggest_merges(_samples("CS", "CS (Pure)"))
        assert len(out) == 1
        assert _pairs(out) == {frozenset({"CS", "CS (Pure)"})}
        # The shorter, cleaner "CS" is the merge target.
        assert out[0].target_name == "CS"

    def test_cs_not_merged_into_doping_series(self):
        # "CS" must NOT be suggested against CS-1/CS-3 (digit guard).
        out = suggest_merges(_samples("CS", "CS-1", "CS-3", "CS (Pure)"))
        pairs = _pairs(out)
        assert frozenset({"CS", "CS (Pure)"}) in pairs
        assert frozenset({"CS", "CS-1"}) not in pairs
        assert frozenset({"CS", "CS-3"}) not in pairs

    def test_empty_and_single(self):
        assert suggest_merges([]) == []
        assert suggest_merges(_samples("OnlyOne")) == []


class TestRanking:
    def test_sorted_by_score_descending(self):
        out = suggest_merges(
            _samples("CSCBI-1", "Dr.MN-dhivya-cscbi1", "CSCBI-3", "Dr.MN-dhivya-cskbi3")
        )
        scores = [s.score for s in out]
        assert scores == sorted(scores, reverse=True)
        # The exact-substring pair outranks the typo pair.
        assert out[0].score >= out[-1].score


class TestWordGuard:
    """The alphabetic counterpart of the digit guard.

    `normalize` strips separators, so "MX_Ti3C2Tx_Air_50" and
    "MX_Ti3C2Tx_Ar_50" reach the scorer as a one-character difference in a
    twenty-character string and score 99%. Merging them would silently collapse
    two different etching atmospheres into one sample.
    """

    def test_air_and_argon_are_never_merged(self):
        out = suggest_merges(
            _samples(
                "MX_Ti3C2Tx_Air_50_CAF_ESCA",
                "MX_Ti3C2Tx_Ar_50_CAF_ESCA",
                "MX_Ti3C2Tx_N2_50_CAF_ESCA",
                "MX_Ti3C2Tx_Air_40_CAF_ESCA",
                "MX_Ti3C2Tx_Ar_40_CAF_ESCA",
            )
        )
        assert out == []

    def test_same_sample_different_technique_still_suggested(self):
        # ESCA and RS describe how the specimen was measured, not which
        # specimen it is, so this pair must survive the guard.
        out = suggest_merges(_samples("MX_Ti3C2Tx_Air_50_CAF_ESCA", "MX_Ti3C2Tx_Air_50_CAF_RS"))
        assert _pairs(out) == {
            frozenset({"MX_Ti3C2Tx_Air_50_CAF_ESCA", "MX_Ti3C2Tx_Air_50_CAF_RS"})
        }

    def test_condition_and_technique_both_differ_is_vetoed(self):
        out = suggest_merges(
            _samples(
                "MX_Ti3C2Tx_Air_50_CAF_ESCA",
                "MX_Ti3C2Tx_Ar_50_CAF_RS",
                "MX_Ti3C2Tx_Air_50_CAF_RS",
                "MX_Ti3C2Tx_Ar_50_CAF_ESCA",
            )
        )
        for pair in _pairs(out):
            names = " ".join(pair).lower().replace("_", " ").split()
            assert not ("air" in names and "ar" in names)

    def test_a_typo_is_still_caught(self):
        # "cskbi" appears once, so it is a misspelling rather than vocabulary
        # and the pair must still be offered — that is the feature's whole job.
        out = suggest_merges(_samples("Dr.MN-dhivya-cskbi3", "CSCBI-3"))
        assert _pairs(out) == {frozenset({"Dr.MN-dhivya-cskbi3", "CSCBI-3"})}

    def test_a_researcher_prefix_is_still_caught(self):
        # Words added on one side only are not a substitution, so the guard
        # must not fire however common those words are.
        out = suggest_merges(
            _samples("Dr.MN-dhivya-cscbi1", "CSCBI-1", "Dr.MN-dhivya-cscbi2", "CSCBI-2")
        )
        assert frozenset({"Dr.MN-dhivya-cscbi1", "CSCBI-1"}) in _pairs(out)

    def test_a_word_seen_once_does_not_veto(self):
        # Vocabulary needs two sightings. "argonn" is written once, so it reads
        # as a misspelling of "argon" rather than a second atmosphere, and the
        # pair is still surfaced for a human to judge.
        out = suggest_merges(
            _samples("MX_Ti3C2Tx_Argon_50_CAF_ESCA", "MX_Ti3C2Tx_Argonn_50_CAF_ESCA")
        )
        assert _pairs(out) == {
            frozenset({"MX_Ti3C2Tx_Argon_50_CAF_ESCA", "MX_Ti3C2Tx_Argonn_50_CAF_ESCA"})
        }

    def test_the_same_typo_repeated_becomes_vocabulary(self):
        # Deliberate and conservative: once a spelling is used consistently it
        # is indistinguishable from a real factor level, so the pair stops being
        # suggested. Failing to merge is undone by hand; a wrong merge is not.
        out = suggest_merges(
            _samples(
                "MX_Ti3C2Tx_Argon_50_CAF_ESCA",
                "MX_Ti3C2Tx_Argonn_50_CAF_ESCA",
                "MX_Ti3C2Tx_Argon_40_CAF_ESCA",
                "MX_Ti3C2Tx_Argonn_40_CAF_ESCA",
            )
        )
        assert out == []
