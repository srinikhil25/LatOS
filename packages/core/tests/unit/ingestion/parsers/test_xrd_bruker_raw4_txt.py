"""Tests for `BrukerRaw4TxtParser`.

Exports are synthesised, so every structural trap is present deliberately.

Two tests carry most of the weight, because both describe silent corruption
rather than a crash:

* ``Time`` means different things in ``[RawHeader]`` and ``[RangeHeader]``. A
  flat key/value parse takes whichever comes last and produces a plausible
  wrong answer either way.
* ``[VarInfo]`` repeats, one block per field. Overwriting instead of pairing
  loses the sample id and keeps whatever happened to be read last.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import numpy as np
import pytest

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.xrd_bruker_raw4_txt import (
    BrukerRaw4TxtParser,
    max_visible_d_spacing_nm,
)


def write(
    tmp_path: Path,
    *,
    name: str = "scan.txt",
    magic: str = ";RAW4.00",
    start: float = 3.0,
    step: float = 0.0204724,
    n: int = 200,
    declared_steps: int | None = None,
    wavelength: str | None = "1.5418",
    date: str | None = "06/17/2026",
    clock: str | None = "12:11:21",
    range_duration: str = "192",
    var_info: tuple[tuple[str, str], ...] = (
        ("USER", "Lab Manager"),
        ("SAMPLEID", "Commander Sample ID"),
        ("COMMENT", ""),
        ("CREATOR", "V5Converter"),
    ),
    angles: np.ndarray | None = None,
) -> Path:
    """A RAW4 text export with the requested structure."""
    lines = [magic, "[RawHeader]"]
    if date:
        lines.append(f"Date={date}")
    if clock:
        lines.append(f"Time={clock}")
    lines.append("NumberOfRanges=1")
    for kind, value in var_info:
        lines += ["", "[VarInfo]", "Type=" + kind, "Flags=0", f"Value={value}"]
    lines += ["", "[HardwareConfiguration]", "Anode=Cu", "WaveUnit=A"]
    lines += ["", "[RangeHeader]"]
    if wavelength:
        lines.append(f"ActuallyUsedLambda={wavelength}")
    lines += [
        f"Start={start}",
        f"Increment={step}",
        f"Steps={declared_steps if declared_steps is not None else n}",
        "ScanMode=1",
        "GeneratorCurrent=40",
        "GeneratorVoltage=40",
        # Same key as the clock above, different section, different meaning.
        f"Time={range_duration}",
    ]
    lines += ["", "[Data]", "     Angle,       PSD,"]
    x = angles if angles is not None else start + step * np.arange(n)
    rng = np.random.default_rng(0)
    for angle in x:
        lines.append(f"{angle:>10.5f}, {int(1000 + rng.integers(0, 200)):>9d},")
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf8")
    return path


def messages(parsed) -> str:
    return " ".join(i.message for i in parsed.issues).lower()


def severities(parsed) -> set[Severity]:
    return {i.severity for i in parsed.issues}


class TestMaxVisibleDSpacing:
    def test_a_three_degree_start_reaches_almost_three_nanometres(self):
        assert max_visible_d_spacing_nm(3.0, 1.5418) == pytest.approx(2.945, abs=0.005)

    def test_a_five_degree_start_reaches_much_less(self):
        """The distinction that decides whether a 'phase absent' claim is evidence.

        A wide scan opening at 5 deg cannot see beyond ~1.77 nm, so a basal
        reflection at 1.6 nm is only marginally in range and one at 2 nm was
        never scanned at all.
        """
        assert max_visible_d_spacing_nm(5.0, 1.5418) == pytest.approx(1.767, abs=0.005)

    def test_a_lower_start_angle_always_reaches_further(self):
        assert max_visible_d_spacing_nm(2.0, 1.5418) > max_visible_d_spacing_nm(8.0, 1.5418)

    @pytest.mark.parametrize("angle", [0.0, -1.0, 180.0, 200.0])
    def test_an_impossible_angle_is_rejected(self, angle):
        with pytest.raises(ValueError, match="2theta"):
            max_visible_d_spacing_nm(angle, 1.5418)

    @pytest.mark.parametrize("lam", [0.0, -1.5])
    def test_a_non_positive_wavelength_is_rejected(self, lam):
        with pytest.raises(ValueError, match="wavelength"):
            max_visible_d_spacing_nm(3.0, lam)


class TestSniffing:
    def test_recognises_the_raw4_magic(self, tmp_path):
        assert BrukerRaw4TxtParser().can_parse(write(tmp_path)) == 1.0

    def test_recognises_a_later_raw4_revision(self, tmp_path):
        assert BrukerRaw4TxtParser().can_parse(write(tmp_path, magic=";RAW4.01")) == 1.0

    def test_rejects_a_file_without_the_magic(self, tmp_path):
        assert BrukerRaw4TxtParser().can_parse(write(tmp_path, magic="[RawHeader]")) == 0.0

    def test_missing_file_scores_zero_rather_than_raising(self, tmp_path):
        assert BrukerRaw4TxtParser().can_parse(tmp_path / "absent.txt") == 0.0


class TestSectionScoping:
    def test_the_two_meanings_of_time_are_kept_apart(self, tmp_path):
        """`Time` is a clock in [RawHeader] and a duration in [RangeHeader].

        Parsed flat, one silently overwrites the other and the result looks
        entirely reasonable whichever way it lands.
        """
        parsed = BrukerRaw4TxtParser().parse(
            write(tmp_path, clock="12:11:21", range_duration="192")
        )
        assert parsed.metadata["range_duration_s"] == pytest.approx(192.0)
        assert parsed.measured_at is not None
        assert (parsed.measured_at.hour, parsed.measured_at.minute) == (12, 11)

    def test_repeated_var_info_blocks_are_paired_not_overwritten(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path))
        assert parsed.metadata["sample_id"] == "Commander Sample ID"
        assert parsed.metadata["operator"] == "Lab Manager"
        assert parsed.metadata["creator"] == "V5Converter"

    def test_an_empty_var_info_value_is_skipped(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path))
        assert "comment" not in parsed.metadata


class TestArrays:
    def test_reads_angles_and_intensities(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, n=200))
        assert parsed.technique is Technique.XRD
        assert parsed.arrays["two_theta_deg"].shape == (200,)
        assert parsed.arrays["intensity"].shape == (200,)
        assert parsed.metadata["n_points"] == 200

    def test_the_caption_row_is_not_read_as_data(self, tmp_path):
        """`     Angle,       PSD,` sits inside [Data] and must be skipped."""
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, n=50))
        assert parsed.metadata["n_points"] == 50
        assert np.all(np.isfinite(parsed.arrays["two_theta_deg"]))

    def test_the_trailing_comma_does_not_break_a_row(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, n=30))
        assert parsed.arrays["intensity"].min() >= 1000

    def test_instrument_and_scan_metadata_are_captured(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, start=3.0, step=0.02))
        assert parsed.metadata["anode"] == "Cu"
        assert parsed.metadata["wavelength_angstrom"] == pytest.approx(1.5418)
        assert parsed.metadata["step_deg"] == pytest.approx(0.02)
        assert parsed.metadata["generator_current_ma"] == pytest.approx(40.0)
        assert "Cu" in parsed.instrument

    def test_max_d_spacing_is_derived_from_the_start_angle(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, start=5.0))
        assert parsed.metadata["max_d_spacing_nm"] == pytest.approx(1.767, abs=0.01)


class TestIssues:
    def test_a_clean_scan_raises_no_warning(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path))
        assert Severity.WARNING not in severities(parsed)
        assert Severity.ERROR not in severities(parsed)

    def test_a_missing_wavelength_is_reported(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, wavelength=None))
        assert "no wavelength" in messages(parsed)
        assert "max_d_spacing_nm" not in parsed.metadata

    def test_a_step_count_mismatch_is_reported(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, n=100, declared_steps=587))
        assert "declares 587 steps but 100 points" in messages(parsed)

    def test_a_matching_step_count_is_not_reported(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, n=200, declared_steps=200))
        assert "steps but" not in messages(parsed)

    def test_non_monotonic_angles_are_reported(self, tmp_path):
        angles = np.concatenate([np.linspace(3, 10, 50), np.linspace(3, 10, 50)])
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, angles=angles))
        assert "not strictly increasing" in messages(parsed)

    def test_the_timestamp_carries_a_timezone_caveat(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path))
        assert parsed.measured_at is not None
        assert parsed.measured_at.tzinfo is UTC
        assert "no timezone" in messages(parsed)

    def test_a_missing_timestamp_is_not_invented(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, date=None, clock=None))
        assert parsed.measured_at is None

    def test_an_export_with_no_data_is_an_error(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(write(tmp_path, n=0))
        assert parsed.arrays == {}
        assert Severity.ERROR in severities(parsed)


class TestRobustness:
    def test_never_raises_on_junk(self, tmp_path):
        for content in ("", ";RAW4.00", ";RAW4.00\n[Data]\n", "garbage\nmore garbage"):
            path = tmp_path / "junk.txt"
            path.write_text(content, encoding="utf8")
            parsed = BrukerRaw4TxtParser().parse(path)
            assert parsed.parser_name == "bruker-raw4-txt"

    def test_never_raises_on_a_missing_file(self, tmp_path):
        parsed = BrukerRaw4TxtParser().parse(tmp_path / "absent.txt")
        assert Severity.ERROR in severities(parsed)
