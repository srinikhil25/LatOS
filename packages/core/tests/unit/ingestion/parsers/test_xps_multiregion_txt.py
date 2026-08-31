"""Tests for `MultiRegionXpsTxtParser`.

Exports are synthesised so each region's acquisition settings, and therefore
the expected per-sweep rate, are known exactly.

The central test is `test_per_sweep_divides_by_sweeps_and_dwell`. Regions in one
file are acquired with different sweep counts AND different dwell times — a real
export here runs the survey at 1 sweep and the core lines at 25 — so raw counts
are not comparable between regions. Nothing in the file says so and the numbers
look ordinary, which is exactly why the corrected array is published rather than
left for the caller to remember.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.xps_multiregion_txt import MultiRegionXpsTxtParser

COLUMNS = "Kinetic Energy(eV)\tBinding Energy(eV)\tIntensity(cps)\tTransmission Value"


def region_block(
    name: str,
    scan: int,
    sample: str,
    *,
    n: int = 20,
    be_lo: float = 450.0,
    be_hi: float = 470.0,
    sweeps: float | None = 25.0,
    dwell: float | None = 0.2,
    counts: float = 5000.0,
    descending: bool = False,
) -> list[str]:
    """One `Dataset` block with its own acquisition settings."""
    lines = [f"Dataset {name}:{scan}({sample})"]
    if dwell is not None:
        lines.append(f"Dwell Time (s)\t{dwell:.6f}")
    if sweeps is not None:
        lines.append(f"Number of sweeps\t{sweeps:.6f}")
    lines.append(COLUMNS)
    be = np.linspace(be_hi, be_lo, n) if descending else np.linspace(be_lo, be_hi, n)
    for value in be:
        lines.append(f"{1486.6 - value:.6f}\t{value:.6f}\t{counts:.6f}\t1.000000")
    return lines


def write(tmp_path: Path, blocks: list[list[str]], *, name: str = "esca.txt") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(line for block in blocks for line in block), encoding="utf8")
    return path


def messages(parsed) -> str:
    return " ".join(i.message for i in parsed.issues).lower()


def severities(parsed) -> set[Severity]:
    return {i.severity for i in parsed.issues}


class TestSniffing:
    def test_recognises_the_dataset_marker(self, tmp_path):
        path = write(tmp_path, [region_block("Ti 2p", 14, "S1")])
        assert MultiRegionXpsTxtParser().can_parse(path) == 1.0

    def test_rejects_a_file_that_does_not_open_with_a_dataset_line(self, tmp_path):
        path = tmp_path / "other.txt"
        path.write_text("angle,intensity\n3,100\nDataset X:1(y)\n", encoding="utf8")
        assert MultiRegionXpsTxtParser().can_parse(path) == 0.0

    def test_missing_file_scores_zero_rather_than_raising(self, tmp_path):
        assert MultiRegionXpsTxtParser().can_parse(tmp_path / "absent.txt") == 0.0


class TestRegionSplitting:
    def test_every_region_is_found_with_its_own_settings(self, tmp_path):
        path = write(
            tmp_path,
            [
                region_block(
                    "wide", 13, "S1", n=40, be_lo=0.0, be_hi=1200.0, sweeps=1.0, dwell=0.1
                ),
                region_block("Ti 2p", 14, "S1", n=20, sweeps=25.0, dwell=0.2985),
                region_block(
                    "C 1s", 19, "S1", n=25, be_lo=277.0, be_hi=300.0, sweeps=15.0, dwell=0.2597
                ),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert parsed.technique is Technique.XPS
        assert parsed.metadata["n_regions"] == 3
        assert parsed.metadata["region_names"] == ["wide", "Ti 2p", "C 1s"]
        by_name = {d["name"]: d for d in parsed.metadata["regions"]}
        assert by_name["wide"]["sweeps"] == pytest.approx(1.0)
        assert by_name["Ti 2p"]["sweeps"] == pytest.approx(25.0)
        assert by_name["C 1s"]["dwell_s"] == pytest.approx(0.2597)

    def test_regions_of_different_length_share_one_flat_table(self, tmp_path):
        """`ParsedData` requires co-indexed arrays, so regions concatenate."""
        path = write(
            tmp_path,
            [
                region_block("wide", 13, "S1", n=40),
                region_block("Ti 2p", 14, "S1", n=20),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        lengths = {arr.shape[0] for arr in parsed.arrays.values()}
        assert lengths == {60}
        assert parsed.metadata["n_points"] == 60

    def test_region_index_selects_the_right_rows(self, tmp_path):
        path = write(
            tmp_path,
            [
                region_block("wide", 13, "S1", n=40, be_lo=0.0, be_hi=1200.0),
                region_block("Ti 2p", 14, "S1", n=20, be_lo=450.0, be_hi=470.0),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        index = parsed.arrays["region_index"]
        be = parsed.arrays["binding_energy_ev"]
        assert be[index == 1].min() == pytest.approx(450.0)
        assert be[index == 1].max() == pytest.approx(470.0)

    def test_row_spans_in_metadata_match_the_index(self, tmp_path):
        path = write(
            tmp_path,
            [
                region_block("wide", 13, "S1", n=40),
                region_block("Ti 2p", 14, "S1", n=20),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        for described in parsed.metadata["regions"]:
            span = parsed.arrays["region_index"][described["row_start"] : described["row_end"]]
            assert np.all(span == described["index"])

    def test_the_column_caption_row_is_not_read_as_data(self, tmp_path):
        path = write(tmp_path, [region_block("Ti 2p", 14, "S1", n=20)])
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert parsed.metadata["n_points"] == 20

    def test_the_sample_id_is_captured(self, tmp_path):
        path = write(tmp_path, [region_block("Ti 2p", 14, "1-MX-NO-50")])
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert parsed.metadata["sample_id"] == "1-MX-NO-50"


class TestPerSweepNormalisation:
    def test_per_sweep_divides_by_sweeps_and_dwell(self, tmp_path):
        """The correction is sweeps x dwell, and BOTH vary between regions.

        A real export runs the survey at 1 sweep / 0.1 s and the core lines at
        25 sweeps / 0.2985 s. Comparing raw counts across those two is wrong by
        a factor of 75 before any chemistry enters.
        """
        path = write(
            tmp_path,
            [
                region_block("wide", 13, "S1", n=10, sweeps=1.0, dwell=0.1, counts=1000.0),
                region_block("Ti 2p", 14, "S1", n=10, sweeps=25.0, dwell=0.2, counts=1000.0),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        index = parsed.arrays["region_index"]
        rate = parsed.arrays["intensity_per_sweep"]
        assert rate[index == 0][0] == pytest.approx(1000.0 / (1.0 * 0.1))
        assert rate[index == 1][0] == pytest.approx(1000.0 / (25.0 * 0.2))

    def test_raw_counts_are_kept_for_provenance(self, tmp_path):
        path = write(tmp_path, [region_block("Ti 2p", 14, "S1", counts=5000.0)])
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert np.all(parsed.arrays["intensity_cps"] == pytest.approx(5000.0))

    def test_equal_raw_counts_become_unequal_rates(self, tmp_path):
        """The failure this exists to prevent: equal raw counts are NOT equal signal."""
        path = write(
            tmp_path,
            [
                region_block("A", 1, "S1", n=10, sweeps=25.0, dwell=0.2, counts=3000.0),
                region_block("B", 2, "S1", n=10, sweeps=15.0, dwell=0.2, counts=3000.0),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        index = parsed.arrays["region_index"]
        rate = parsed.arrays["intensity_per_sweep"]
        assert rate[index == 1][0] > rate[index == 0][0]
        assert rate[index == 1][0] / rate[index == 0][0] == pytest.approx(25.0 / 15.0)

    def test_differing_sweep_counts_are_reported(self, tmp_path):
        path = write(
            tmp_path,
            [
                region_block("A", 1, "S1", sweeps=25.0),
                region_block("B", 2, "S1", sweeps=15.0),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert "not\ncomparable" in messages(parsed) or "not comparable" in messages(parsed)
        assert Severity.INFO in severities(parsed)

    def test_identical_sweep_counts_are_not_reported(self, tmp_path):
        path = write(
            tmp_path,
            [
                region_block("A", 1, "S1", sweeps=25.0),
                region_block("B", 2, "S1", sweeps=25.0),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert "not comparable" not in messages(parsed)

    def test_a_region_without_settings_yields_nan_and_a_warning(self, tmp_path):
        path = write(tmp_path, [region_block("Ti 2p", 14, "S1", sweeps=None, dwell=None)])
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert np.all(np.isnan(parsed.arrays["intensity_per_sweep"]))
        assert "must not be compared" in messages(parsed)


class TestIssues:
    def test_a_clean_single_region_export_raises_no_warning(self, tmp_path):
        path = write(tmp_path, [region_block("Ti 2p", 14, "S1")])
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert Severity.WARNING not in severities(parsed)
        assert Severity.ERROR not in severities(parsed)

    def test_an_empty_region_is_named_and_dropped(self, tmp_path):
        path = write(
            tmp_path,
            [
                region_block("Ti 2p", 14, "S1", n=20),
                region_block("Ghost", 15, "S1", n=0),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert parsed.metadata["n_regions"] == 1
        assert "'ghost' contains no data rows" in messages(parsed)

    def test_descending_binding_energy_is_accepted(self, tmp_path):
        """XPS is conventionally recorded high-to-low; that is not an error."""
        path = write(tmp_path, [region_block("Ti 2p", 14, "S1", descending=True)])
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert Severity.WARNING not in severities(parsed)

    def test_mixed_sample_ids_are_reported(self, tmp_path):
        path = write(
            tmp_path,
            [
                region_block("Ti 2p", 14, "SAMPLE-A"),
                region_block("C 1s", 19, "SAMPLE-B"),
            ],
        )
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert "mixes 2 sample ids" in messages(parsed)

    def test_a_file_with_no_dataset_markers_is_an_error(self, tmp_path):
        path = tmp_path / "plain.txt"
        path.write_text("470.0\t1000.0\n469.9\t1010.0\n", encoding="utf8")
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert parsed.arrays == {}
        assert Severity.ERROR in severities(parsed)

    def test_all_regions_empty_is_an_error(self, tmp_path):
        path = write(tmp_path, [region_block("Ghost", 1, "S1", n=0)])
        parsed = MultiRegionXpsTxtParser().parse(path)
        assert parsed.arrays == {}
        assert Severity.ERROR in severities(parsed)


class TestRobustness:
    def test_never_raises_on_junk(self, tmp_path):
        for content in ("", "Dataset", "Dataset x:1(y)\n", "\n\n\n", "garbage"):
            path = tmp_path / "junk.txt"
            path.write_text(content, encoding="utf8")
            parsed = MultiRegionXpsTxtParser().parse(path)
            assert parsed.parser_name == "xps-multiregion-txt"

    def test_never_raises_on_a_missing_file(self, tmp_path):
        parsed = MultiRegionXpsTxtParser().parse(tmp_path / "absent.txt")
        assert Severity.ERROR in severities(parsed)

    def test_rows_with_the_wrong_column_count_are_skipped(self, tmp_path):
        block = region_block("Ti 2p", 14, "S1", n=10)
        block.append("470.0\t1000.0")  # two columns, not four
        parsed = MultiRegionXpsTxtParser().parse(write(tmp_path, [block]))
        assert parsed.metadata["n_points"] == 10
