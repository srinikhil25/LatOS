"""Tests for `latos.ingestion.labeling.cluster`.

The cluster phase is where the user-facing behaviour of Stage 2
crystallizes: the Dhivya regression must collapse, but related-but-
distinct samples (CS-1 / CS-3, CS / CSCBI) must stay separate. Tests
are organized roughly by surface area:

- `similarity()` — pure metric, exercised on its own so mistakes in
  the rapidfuzz wiring surface immediately and don't only show up as
  weird cluster outputs.
- `pick_canonical()` — small, deterministic, tie-breakable.
- `cluster_samples()` — the integration: hint → graph → component →
  `SampleCluster`. Headline cases anchor the regression; the rest
  drill into edge cases (empty inputs, threshold boundaries, fallback
  for files with no normalizable hints).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from latos.ingestion.labeling.cluster import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SampleCluster,
    cluster_samples,
    pick_canonical,
    similarity,
)
from latos.ingestion.labeling.hints import SampleHints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hints(
    path: str,
    *,
    metadata: dict[str, str] | None = None,
    filename: str | None = None,
    cleaned: str | None = None,
    path_segments: tuple[str, ...] = (),
    confidences: dict[str, float] | None = None,
) -> SampleHints:
    """Build a `SampleHints` for tests without going through `extract_hints`.

    Lets each test focus on the cluster phase's behaviour without
    coupling to the hint-extractor's specific weights — every test
    explicitly states which signals carry which weight.
    """
    return SampleHints(
        file_path=Path(path),
        from_path_segments=path_segments,
        from_filename=filename,
        from_filename_cleaned=cleaned,
        from_file_metadata=dict(metadata or {}),
        from_file_content=None,
        from_excel_sheet=None,
        confidence_per_source=dict(confidences or {}),
    )


# ---------------------------------------------------------------------------
# similarity()
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_identical_strings_score_one(self):
        assert similarity("CS-1", "CS-1") == 1.0

    def test_normalized_equality_scores_one(self):
        # Both normalize to "cs1"; the fast-path inside similarity()
        # short-circuits to 1.0 once normalization equates them.
        assert similarity("CS-1", "cs_1") == 1.0
        assert similarity("CS Pure", "CS (Pure)") == 1.0

    def test_empty_inputs_return_zero(self):
        assert similarity("", "anything") == 0.0
        assert similarity("anything", "") == 0.0
        assert similarity("", "") == 0.0

    def test_whitespace_only_inputs_return_zero(self):
        # Stripped to empty → normalize() returns "" → score 0.
        assert similarity("   ", "CS-1") == 0.0

    def test_completely_different_strings_score_low(self):
        # Different domains — should be well below the default threshold.
        assert similarity("apple", "rocketship") < DEFAULT_SIMILARITY_THRESHOLD

    def test_returns_within_unit_interval(self):
        # Spot-check the range invariant for a handful of inputs.
        for a, b in [("MX-001", "MX-002"), ("Cs3Bi2I9", "CsBi"), ("a", "b")]:
            score = similarity(a, b)
            assert 0.0 <= score <= 1.0

    def test_typo_one_char_off_scores_high(self):
        # Levenshtein metric carries this case.
        assert similarity("CSCBI-1", "CSCBI-2") >= 0.8

    def test_short_prefix_difference_scores_below_one(self):
        # Jaro-Winkler is generous on common-prefix matches but a
        # one-character suffix difference must score < 1 so the
        # cluster phase can still pull rank with the threshold.
        score = similarity("CS-1", "CS-2")
        assert 0.0 < score < 1.0

    def test_max_of_three_metrics(self):
        # `similarity` returns max(lev, tok, jaro). At least one of
        # them must dominate for the headline regression cases.
        assert similarity("CS Pure", "CS (Pure)") >= 0.95
        assert similarity("MX-001", "MX001") >= 0.95


# ---------------------------------------------------------------------------
# pick_canonical()
# ---------------------------------------------------------------------------


class TestPickCanonical:
    def test_shortest_wins(self):
        assert pick_canonical(["CS Pure", "CS Pure (extra)"]) == "CS Pure"

    def test_alphabetical_tiebreak_among_equally_short(self):
        # Equal length 4 — alphabetical first wins.
        assert pick_canonical(["CS-1", "CS-3", "CS-2"]) == "CS-1"

    def test_strips_whitespace_before_compare(self):
        # Trailing whitespace shouldn't affect the choice — both have
        # the same effective length after strip.
        result = pick_canonical(["  CS-1  ", "CS-2"])
        assert result == "CS-1"

    def test_ignores_empty_and_whitespace_only_aliases(self):
        # Empty / whitespace-only get filtered before the choice.
        assert pick_canonical(["", "   ", "MX-1"]) == "MX-1"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty alias set"):
            pick_canonical([])

    def test_all_whitespace_raises(self):
        with pytest.raises(ValueError, match="empty alias set"):
            pick_canonical(["", "   ", "\t"])

    def test_accepts_any_iterable(self):
        # Generators, sets, tuples — all should work.
        assert pick_canonical({"foo", "fo"}) == "fo"
        assert pick_canonical(("alpha", "beta", "ab")) == "ab"
        assert pick_canonical(x for x in ["loooong", "mid", "x"]) == "x"


# ---------------------------------------------------------------------------
# cluster_samples() — headline regression
# ---------------------------------------------------------------------------


class TestDhivyaRegression:
    """The user-facing motivation for Stage 2 — these must merge."""

    def test_cs_pure_variants_merge_into_one_cluster(self):
        # Two files in different technique folders, sample written
        # cosmetically differently. Stage 1 produced two samples; Stage
        # 2 must produce one.
        h1 = _hints(
            "/proj/XRD/CS Pure/run.xrdml",
            path_segments=("CS Pure", "XRD"),
            confidences={"path_segment_d0": 0.60},
        )
        h2 = _hints(
            "/proj/XPS/CS (Pure)/run.csv",
            path_segments=("CS (Pure)", "XPS"),
            confidences={"path_segment_d0": 0.60},
        )

        clusters = cluster_samples([h1, h2])

        assert len(clusters) == 1
        cluster = clusters[0]
        # Both raw aliases preserved for the UI.
        assert "CS Pure" in cluster.aliases
        assert "CS (Pure)" in cluster.aliases
        # Both files routed into the same cluster.
        assert set(cluster.file_paths) == {h1.file_path, h2.file_path}
        # Canonical is the shorter of the two.
        assert cluster.canonical == "CS Pure"

    def test_distinct_samples_stay_separate(self):
        # CS-1 and CS-3 share a prefix but are different samples — the
        # threshold must keep them apart.
        h1 = _hints(
            "/proj/XRD/CS-1/run.xrdml",
            path_segments=("CS-1",),
            confidences={"path_segment_d0": 0.60},
        )
        h2 = _hints(
            "/proj/XRD/CS-3/run.xrdml",
            path_segments=("CS-3",),
            confidences={"path_segment_d0": 0.60},
        )

        clusters = cluster_samples([h1, h2])

        assert len(clusters) == 2
        canonicals = {c.canonical for c in clusters}
        assert canonicals == {"CS-1", "CS-3"}

    def test_cs_versus_cscbi_stay_separate(self):
        # Different chemistry — these must not merge despite a prefix
        # overlap.
        h1 = _hints(
            "/proj/XRD/CS/run.xrdml",
            path_segments=("CS",),
            confidences={"path_segment_d0": 0.60},
        )
        h2 = _hints(
            "/proj/XRD/CSCBI/run.xrdml",
            path_segments=("CSCBI",),
            confidences={"path_segment_d0": 0.60},
        )

        clusters = cluster_samples([h1, h2])
        canonicals = {c.canonical for c in clusters}
        assert canonicals == {"CS", "CSCBI"}


# ---------------------------------------------------------------------------
# cluster_samples() — voting / file assignment
# ---------------------------------------------------------------------------


class TestFileVoting:
    def test_file_assigned_to_highest_confidence_vote(self):
        # A file with two contradictory hints — strong metadata for
        # `MX-1` and weak path-segment for `Other`. Metadata wins.
        h = _hints(
            "/proj/Other/run.csv",
            metadata={"metadata_sample_name": "MX-1"},
            path_segments=("Other",),
            confidences={
                "metadata_sample_name": 1.00,
                "path_segment_d0": 0.60,
            },
        )

        clusters = cluster_samples([h])

        # File ends up in the MX-1 cluster (the higher-weighted hint).
        mx1 = next(c for c in clusters if c.canonical == "MX-1")
        assert h.file_path in mx1.file_paths

    def test_multiple_files_vote_independently(self):
        # Two files, each with a clear single-hint sample assignment.
        h1 = _hints(
            "/proj/A/run.csv",
            metadata={"metadata_sample_name": "MX-1"},
            confidences={"metadata_sample_name": 1.00},
        )
        h2 = _hints(
            "/proj/B/run.csv",
            metadata={"metadata_sample_name": "MX-2"},
            confidences={"metadata_sample_name": 1.00},
        )

        clusters = cluster_samples([h1, h2])

        canon = {c.canonical: c for c in clusters}
        assert h1.file_path in canon["MX-1"].file_paths
        assert h2.file_path in canon["MX-2"].file_paths

    def test_confidences_aggregate_per_file(self):
        # A file with three weak hints all pointing at the same
        # cluster should still land there even if a single rival hint
        # has a higher individual weight.
        h = _hints(
            "/proj/Aa/Aa/Aa/run.csv",
            path_segments=("Aa", "Aa", "Aa"),
            metadata={"metadata_name": "Bb"},
            confidences={
                "path_segment_d0": 0.60,
                "path_segment_d1": 0.50,
                "path_segment_d2": 0.40,
                "metadata_name": 0.85,  # Higher than any single segment.
            },
        )

        clusters = cluster_samples([h])

        # Aa accumulates 0.60 + 0.50 + 0.40 = 1.50 vs Bb's 0.85.
        aa = next(c for c in clusters if c.canonical == "Aa")
        assert h.file_path in aa.file_paths


# ---------------------------------------------------------------------------
# cluster_samples() — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input_returns_empty_tuple(self):
        assert cluster_samples([]) == ()

    def test_single_file_single_hint(self):
        h = _hints(
            "/proj/MX-1/run.csv",
            path_segments=("MX-1",),
            confidences={"path_segment_d0": 0.60},
        )

        clusters = cluster_samples([h])

        assert len(clusters) == 1
        assert clusters[0].canonical == "MX-1"
        assert clusters[0].aliases == ("MX-1",)
        assert clusters[0].file_paths == (h.file_path,)

    def test_file_with_no_hints_falls_back_to_filename(self):
        # No metadata, no path segments — fallback to filename stem.
        h = _hints(
            "/proj/lonely.csv",
            filename="lonely",
            confidences={"filename_stem": 0.70},
        )

        clusters = cluster_samples([h])

        assert len(clusters) == 1
        assert clusters[0].canonical == "lonely"
        assert h.file_path in clusters[0].file_paths

    def test_file_with_no_normalizable_hints_uses_fallback_label(self):
        # No metadata, no path segments, no filename either — fallback
        # is forced to use the path string. Such a file shouldn't
        # crash the pipeline.
        h = _hints("/proj/unnamed")

        clusters = cluster_samples([h])

        # Each leftover file gets a single-file cluster — not empty.
        assert len(clusters) == 1
        # Path-derived label is non-empty.
        assert clusters[0].canonical
        assert h.file_path in clusters[0].file_paths

    def test_all_separator_string_skipped(self):
        # A hint that normalizes to "" must not pull a file into a
        # cluster — the file should land in a fallback instead.
        h = _hints(
            "/proj/run.csv",
            metadata={"metadata_sample_name": "---"},
            filename="run",
            confidences={
                "metadata_sample_name": 1.00,
                "filename_stem": 0.70,
            },
        )

        clusters = cluster_samples([h])

        # Filename stem ("run") survives normalization, so the cluster
        # is keyed off "run" not "---".
        assert any("run" in c.aliases for c in clusters)
        # The all-separator alias must not appear as a canonical.
        assert all(c.canonical != "---" for c in clusters)


# ---------------------------------------------------------------------------
# cluster_samples() — threshold behaviour
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_invalid_threshold_below_zero(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cluster_samples([], similarity_threshold=-0.1)

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cluster_samples([], similarity_threshold=1.1)

    def test_threshold_one_keeps_only_exact_matches(self):
        # At threshold 1.0 only normalized-equal strings link. CS Pure
        # and CS (Pure) normalize to the same string so they still
        # merge; CSCBI-1 and CSCBI-2 differ in the last char and stay
        # separate.
        h1 = _hints(
            "/p/a.csv",
            path_segments=("CSCBI-1",),
            confidences={"path_segment_d0": 0.60},
        )
        h2 = _hints(
            "/p/b.csv",
            path_segments=("CSCBI-2",),
            confidences={"path_segment_d0": 0.60},
        )

        clusters = cluster_samples([h1, h2], similarity_threshold=1.0)

        # Distinct.
        canonicals = {c.canonical for c in clusters}
        assert canonicals == {"CSCBI-1", "CSCBI-2"}

    def test_threshold_zero_collapses_everything(self):
        # At threshold 0 every pair gets an edge. Two genuinely-
        # different samples should merge into a single cluster.
        h1 = _hints(
            "/p/a.csv",
            path_segments=("Apple",),
            confidences={"path_segment_d0": 0.60},
        )
        h2 = _hints(
            "/p/b.csv",
            path_segments=("Banana",),
            confidences={"path_segment_d0": 0.60},
        )

        clusters = cluster_samples([h1, h2], similarity_threshold=0.0)

        assert len(clusters) == 1

    def test_default_threshold_constant_is_reasonable(self):
        # The default is the published Dhivya-tuned constant. Guard
        # against accidental edits.
        assert 0.7 <= DEFAULT_SIMILARITY_THRESHOLD <= 0.95


# ---------------------------------------------------------------------------
# SampleCluster dataclass smoke
# ---------------------------------------------------------------------------


class TestSampleClusterShape:
    def test_dataclass_fields(self):
        cluster = SampleCluster(
            canonical="CS-1",
            aliases=("CS-1", "cs_1"),
            file_paths=(Path("/proj/a.csv"),),
            normalized_forms=("cs1",),
        )
        assert cluster.canonical == "CS-1"
        assert cluster.aliases == ("CS-1", "cs_1")
        assert cluster.file_paths == (Path("/proj/a.csv"),)
        assert cluster.normalized_forms == ("cs1",)

    def test_dataclass_is_frozen(self):
        cluster = SampleCluster(canonical="CS-1")
        with pytest.raises((AttributeError, TypeError)):
            cluster.canonical = "Other"  # type: ignore[misc]

    def test_default_factory_fields(self):
        cluster = SampleCluster(canonical="CS-1")
        assert cluster.aliases == ()
        assert cluster.file_paths == ()
        assert cluster.normalized_forms == ()


# ---------------------------------------------------------------------------
# Integration: realistic Dhivya-shaped scenarios
# ---------------------------------------------------------------------------


class TestRealisticScenarios:
    def test_three_techniques_one_sample(self):
        # XRD + XPS + UV-DRS all on the same sample written three ways.
        h_xrd = _hints(
            "/proj/XRD/CS-1/run.xrdml",
            path_segments=("CS-1", "XRD"),
            confidences={"path_segment_d0": 0.60},
        )
        h_xps = _hints(
            "/proj/XPS/cs_1/run.csv",
            path_segments=("cs_1", "XPS"),
            confidences={"path_segment_d0": 0.60},
        )
        h_uvdrs = _hints(
            "/proj/UV-DRS/CS 1/run.csv",
            path_segments=("CS 1", "UV-DRS"),
            confidences={"path_segment_d0": 0.60},
        )

        clusters = cluster_samples([h_xrd, h_xps, h_uvdrs])

        # Single cluster covering all three files.
        assert len(clusters) == 1
        cluster = clusters[0]
        assert len(cluster.file_paths) == 3
        # All three raw forms preserved.
        assert set(cluster.aliases) >= {"CS-1", "cs_1", "CS 1"}

    def test_results_sorted_by_canonical(self):
        # The orchestrator and UI both rely on stable iteration order.
        h_b = _hints(
            "/p/b.csv",
            metadata={"metadata_sample_name": "Beta"},
            confidences={"metadata_sample_name": 1.00},
        )
        h_a = _hints(
            "/p/a.csv",
            metadata={"metadata_sample_name": "Alpha"},
            confidences={"metadata_sample_name": 1.00},
        )
        h_g = _hints(
            "/p/g.csv",
            metadata={"metadata_sample_name": "Gamma"},
            confidences={"metadata_sample_name": 1.00},
        )

        clusters = cluster_samples([h_a, h_b, h_g])
        canonicals = [c.canonical for c in clusters]
        assert canonicals == sorted(canonicals)

    def test_aliases_and_file_paths_sorted(self):
        # Stable test output relies on every collection inside a
        # cluster being sorted.
        h1 = _hints(
            "/p/z.csv",
            metadata={"metadata_sample_name": "Zebra"},
            confidences={"metadata_sample_name": 1.00},
        )
        h2 = _hints(
            "/p/a.csv",
            metadata={"metadata_sample_name": "zebra"},
            confidences={"metadata_sample_name": 1.00},
        )

        clusters = cluster_samples([h1, h2])

        assert len(clusters) == 1
        cluster = clusters[0]
        # Aliases sorted alphabetically.
        assert list(cluster.aliases) == sorted(cluster.aliases)
        # File paths sorted.
        assert list(cluster.file_paths) == sorted(cluster.file_paths)
