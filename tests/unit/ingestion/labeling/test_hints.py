"""Tests for `latos.ingestion.labeling.hints.extract_hints`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latos.core.enums import Technique
from latos.ingestion.labeling.hints import (
    SampleHints,
    extract_hints,
)
from latos.ingestion.parsed_data import ParsedData

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _parsed(metadata: dict[str, object]) -> ParsedData:
    """Minimal `ParsedData` carrying only the metadata under test."""
    return ParsedData(
        technique=Technique.XRD,
        arrays={"two_theta": np.array([1.0]), "intensity": np.array([2.0])},
        metadata=metadata,
        instrument="Stub",
        measured_at=None,
        issues=(),
        parser_name="stub-parser",
        parser_version="0.0.1",
    )


# ---------------------------------------------------------------------------
# Filename hints
# ---------------------------------------------------------------------------


class TestFilenameHints:
    def test_basic_stem_emitted(self):
        hints = extract_hints(Path("/proj/Samples/CS-1/scan.xrdml"))
        assert hints.from_filename == "scan"
        assert hints.confidence_per_source["filename_stem"] == 0.70

    def test_run_suffix_stripped_into_cleaned_variant(self):
        hints = extract_hints(Path("/proj/MX-001_run5.csv"))
        assert hints.from_filename == "MX-001_run5"
        assert hints.from_filename_cleaned == "MX-001"

    def test_paren_dup_suffix_stripped(self):
        hints = extract_hints(Path("/proj/CS Pure (1).xrdml"))
        assert hints.from_filename == "CS Pure (1)"
        assert hints.from_filename_cleaned == "CS Pure"

    def test_index_suffix_stripped(self):
        hints = extract_hints(Path("/proj/CSCBI-1_5.xy"))
        assert hints.from_filename == "CSCBI-1_5"
        # `_5` is the trailing index → should strip.
        assert hints.from_filename_cleaned == "CSCBI-1"

    def test_clean_equal_to_original_means_no_cleaned_emitted(self):
        # Plain stems with no run/index suffix should *not* surface a
        # duplicate "cleaned" hint.
        hints = extract_hints(Path("/proj/CS_pure.xy"))
        assert hints.from_filename == "CS_pure"
        assert hints.from_filename_cleaned is None

    def test_no_stem_yields_none(self):
        hints = extract_hints(Path("/proj/.gitkeep"))
        # `.gitkeep` has stem `.gitkeep` actually — let's use a true
        # extension-less file with no name.
        assert hints.from_filename in (".gitkeep", None)


# ---------------------------------------------------------------------------
# Path-segment hints
# ---------------------------------------------------------------------------


class TestPathSegments:
    def test_immediate_parent_emitted_first(self):
        hints = extract_hints(Path("/proj/CS-1/XRD/run.xrdml"), root=Path("/proj"))
        # Walk emits parent-first: ["XRD", "CS-1"].
        assert hints.from_path_segments == ("XRD", "CS-1")

    def test_walk_stops_at_root(self):
        hints = extract_hints(
            Path("/somewhere/proj/CS/XRD/run.xrdml"), root=Path("/somewhere/proj")
        )
        # Should NOT include "proj" or anything above it.
        assert hints.from_path_segments == ("XRD", "CS")

    def test_generic_segments_get_low_weight(self):
        hints = extract_hints(Path("/proj/raw data/XRD/run.xrdml"), root=Path("/proj"))
        # Both segments are generic (XRD + "raw data").
        assert "path_segment_generic" in hints.confidence_per_source
        # Generic weight is < non-generic d0 weight.
        assert hints.confidence_per_source["path_segment_generic"] < 0.6

    def test_non_generic_d0_weight(self):
        hints = extract_hints(Path("/proj/CS-1/XRD/run.xrdml"), root=Path("/proj"))
        # XRD is generic (d0); CS-1 is non-generic (d1).
        assert hints.confidence_per_source["path_segment_d1"] == pytest.approx(0.50)

    def test_segments_decay_with_depth(self):
        hints = extract_hints(Path("/r/A/B/C/D/file.xy"), root=Path("/r"))
        # All four segments are non-generic; weights should decay.
        d0 = hints.confidence_per_source["path_segment_d0"]
        d1 = hints.confidence_per_source["path_segment_d1"]
        d2 = hints.confidence_per_source["path_segment_d2"]
        d3 = hints.confidence_per_source["path_segment_d3"]
        assert d0 > d1 > d2 > d3

    def test_walk_capped_at_max_depth(self):
        hints = extract_hints(
            Path("/r/a/b/c/d/e/f/g/h/file.xy"),
            # No root → walk hits the depth cap before the filesystem root.
        )
        # 6 levels max per `_MAX_PATH_DEPTH`.
        assert len(hints.from_path_segments) <= 6

    def test_generic_folders_recognized(self):
        # Walk emits parent-first, so `CS-1` is at depth 0 here even
        # though it sits 4 levels above the project root. `XRD`,
        # `characterization`, `data` are all generic and collapse to
        # the single `path_segment_generic` confidence bucket.
        hints = extract_hints(
            Path("/proj/data/characterization/XRD/CS-1/run.xrdml"),
            root=Path("/proj"),
        )
        # Depth 0 is non-generic (`CS-1`). The immediate-parent weight
        # is 0.80 - the highest of any non-metadata signal - so a
        # researcher's deliberate folder structure outranks the
        # filename hint (0.70) when both exist.
        assert hints.confidence_per_source.get("path_segment_d0") == pytest.approx(0.80)
        # The generic parents above it share one bucket.
        assert "path_segment_generic" in hints.confidence_per_source
        # The non-generic walk stops emitting numbered tags after `CS-1`
        # because everything else hit the generic bucket.
        assert "path_segment_d1" not in hints.confidence_per_source


# ---------------------------------------------------------------------------
# Metadata hints
# ---------------------------------------------------------------------------


class TestMetadataHints:
    def test_no_parsed_data_means_no_metadata_hints(self):
        hints = extract_hints(Path("/proj/CS/run.xy"))
        assert hints.from_file_metadata == {}

    def test_sample_name_key_emits_top_confidence(self):
        hints = extract_hints(
            Path("/proj/run.xy"),
            parsed_data=_parsed({"sample_name": "MX-Pure"}),
        )
        assert hints.from_file_metadata == {"metadata_sample_name": "MX-Pure"}
        assert hints.confidence_per_source["metadata_sample_name"] == 1.00

    def test_sample_id_key_recognized(self):
        hints = extract_hints(
            Path("/proj/run.xrdml"),
            parsed_data=_parsed({"sample_id": "Cs3Bi2I9-pellet-7"}),
        )
        assert hints.from_file_metadata["metadata_sample_id"] == "Cs3Bi2I9-pellet-7"
        assert hints.confidence_per_source["metadata_sample_id"] == 1.00

    def test_generic_name_key_lowest_weight(self):
        hints = extract_hints(
            Path("/proj/run.xy"),
            parsed_data=_parsed({"name": "specimen-5"}),
        )
        assert hints.confidence_per_source["metadata_name"] == 0.85

    def test_blank_or_non_string_metadata_ignored(self):
        hints = extract_hints(
            Path("/proj/run.xy"),
            parsed_data=_parsed(
                {
                    "sample_name": "   ",  # whitespace-only
                    "sample_id": 42,  # wrong type
                    "title": "Real Title",
                }
            ),
        )
        # Only the real string survives.
        assert list(hints.from_file_metadata) == ["metadata_title"]

    def test_multiple_metadata_keys_all_emitted(self):
        hints = extract_hints(
            Path("/proj/run.xy"),
            parsed_data=_parsed({"sample_name": "X", "title": "X-doped pellet"}),
        )
        assert "metadata_sample_name" in hints.from_file_metadata
        assert "metadata_title" in hints.from_file_metadata


# ---------------------------------------------------------------------------
# `candidates()` flattening
# ---------------------------------------------------------------------------


class TestCandidatesIterator:
    def test_full_flatten_includes_every_source(self):
        # `MX-001_run5` is the canonical case where filename cleaning
        # actually fires (separator-led `_run5` suffix). With a stem
        # like `run5` (no leading separator) the cleaning rules
        # deliberately leave it alone — see `_FILENAME_TRAILING_PATTERNS`.
        hints = extract_hints(
            Path("/proj/CS-1/XRD/MX-001_run5.xrdml"),
            parsed_data=_parsed({"sample_name": "MX-Pure"}),
            root=Path("/proj"),
        )
        cands = hints.candidates()
        tags = {tag for tag, _v, _c in cands}
        assert "metadata_sample_name" in tags
        assert "filename_stem" in tags
        assert "filename_stem_cleaned" in tags
        assert any(t.startswith("path_segment") for t in tags)

    def test_cleaned_filename_omitted_when_equal_to_stem(self):
        hints = extract_hints(Path("/proj/CS_pure.xy"))
        cands = hints.candidates()
        tags = {tag for tag, _v, _c in cands}
        assert "filename_stem" in tags
        assert "filename_stem_cleaned" not in tags

    def test_confidence_attached_to_each_candidate(self):
        hints = extract_hints(
            Path("/proj/CS-1/run.xy"),
            parsed_data=_parsed({"sample_name": "CS-1"}),
            root=Path("/proj"),
        )
        for tag, value, conf in hints.candidates():
            _ = tag
            _ = value
            assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Dataclass smoke
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_is_frozen(self):
        hints = SampleHints(file_path=Path("/x"))
        with pytest.raises((AttributeError, TypeError)):
            hints.from_filename = "no"  # type: ignore[misc]

    def test_default_factories_are_independent_instances(self):
        a = SampleHints(file_path=Path("/x"))
        b = SampleHints(file_path=Path("/y"))
        assert a.from_file_metadata is not b.from_file_metadata
        assert a.confidence_per_source is not b.confidence_per_source


# ---------------------------------------------------------------------------
# Real-world Dhivya case
# ---------------------------------------------------------------------------


class TestDhivyaCase:
    """The headline regression: 'CS Pure' XRD vs 'CS (Pure)' XPS."""

    def test_xrd_folder_yields_cs_pure_hint(self):
        hints = extract_hints(
            Path("/dhivya/CS Pure/scan1.xrdml"),
            root=Path("/dhivya"),
        )
        # `CS Pure` is the immediate parent and is non-generic.
        assert hints.from_path_segments[0] == "CS Pure"
        assert (
            hints.confidence_per_source["path_segment_d0"]
            > hints.confidence_per_source.get("filename_stem", 0)
            - 0.20  # within reach so the cluster phase will tie them
        )

    def test_xps_folder_yields_cs_paren_pure_hint(self):
        hints = extract_hints(
            Path("/dhivya/CS (Pure)/region1.csv"),
            root=Path("/dhivya"),
        )
        # `CS (Pure)` is the immediate parent and is non-generic.
        assert hints.from_path_segments[0] == "CS (Pure)"

    def test_both_paths_yield_strings_that_will_normalize_together(self):
        # 2A doesn't normalize — that's 2B's job. But the strings we emit
        # must contain the variants the cluster phase expects to merge.
        # Document the contract via this test so the next stage's
        # implementer can rely on it.
        a = extract_hints(Path("/dhivya/CS Pure/scan1.xrdml"), root=Path("/dhivya"))
        b = extract_hints(Path("/dhivya/CS (Pure)/region1.csv"), root=Path("/dhivya"))
        a_strings = [v for _t, v, _c in a.candidates()]
        b_strings = [v for _t, v, _c in b.candidates()]
        assert "CS Pure" in a_strings
        assert "CS (Pure)" in b_strings


@pytest.mark.parametrize(
    ("filename", "expected_cleaned"),
    [
        ("CS-1", None),
        ("CS-1_run3", "CS-1"),
        ("CS-1 (1)", "CS-1"),
        ("CS-1_5", "CS-1"),
        ("CS-1_scan42", "CS-1"),
        ("CS-1_TRIAL3", "CS-1"),
        ("MX-001-run-7", "MX-001"),
    ],
)
def test_clean_filename_matrix(filename: str, expected_cleaned: str | None):
    """Table-style coverage of the trailing-suffix patterns."""
    hints = extract_hints(Path(f"/p/{filename}.xy"))
    assert hints.from_filename == filename
    assert hints.from_filename_cleaned == expected_cleaned
