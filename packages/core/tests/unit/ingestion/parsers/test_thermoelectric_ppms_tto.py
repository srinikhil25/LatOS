"""Tests for `PpmsTtoParser`.

Files are synthesised rather than fixtured, so each failure mode is present by
construction and the expected verdict is known exactly. The failure modes are
not hypothetical: every one below was found in a real dataset, and every one is
silent — the export looks well-formed and the numbers look plausible in all of
them. That is what the parser exists to catch.

Two layouts are covered, because the instrument writes both: the single-channel
TTO export, and the two-channel export used when a second sample is mounted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.thermoelectric_ppms_tto import (
    PpmsTtoParser,
    geometry_correction,
)

SINGLE_HEADER = (
    "time(s)\tT(K)\tB(T)\tPosition\trho(micr\tS(microV\tkappa(W/\tdT(K)\t"
    "heaterI(\tJq(W)\tZT\tPF(mW/mK\tL/L0"
)
TWO_CHANNEL_HEADER = (
    "Time_s\tTR_K\tTS1_K\tTS2_K\tField_Oe\tPosition\tR1_Ohm\tR2_Ohm\t"
    "rho1_mic\tS1\tS2\tdT1\tdT2\tPF1\tkappa\tHeaterI_\tZT"
)


def write_single(
    tmp_path: Path,
    *,
    name: str = "run.txt",
    geometry: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    n: int = 12,
    seebeck: float = 15.4,
    kappa: float = -1.6e-5,
    rho: float = 8.66e4,
    lorenz: float = -1.8e-3,
    heater: float = 10.0,
) -> Path:
    """A single-channel TTO export with the requested pathology."""
    lines = []
    if geometry is not None:
        t, w, length = geometry
        lines.append(f"thickness: {t:.6f}, width: {w:.6f}, length: {length:.6f}")
    lines.append(SINGLE_HEADER)
    for i in range(n):
        temp = 300.0 + i
        lines.append(
            "\t".join(
                f"{v:.8E}"
                for v in (
                    i * 300.0,
                    temp,
                    6.0e-6,
                    0.0,
                    rho,
                    seebeck,
                    kappa,
                    2.0,
                    heater,
                    -3.4e-8,
                    -1.9,
                    2.7e-4,
                    lorenz,
                )
            )
        )
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf8")
    return path


def write_two_channel(tmp_path: Path, *, name: str = "two.txt", heater: float = 0.0) -> Path:
    """A two-channel export whose second channel was never wired."""
    lines = ["thickness: 1.000000, width: 1.000000, length: 1.000000", TWO_CHANNEL_HEADER]
    for i in range(10):
        temp = 270.0 + 2 * i
        lines.append(
            "\t".join(
                f"{v:.8E}"
                for v in (
                    i * 100.0,
                    temp,
                    temp,
                    0.0,
                    0.0,
                    0.0,
                    7.1e-4,
                    0.0,
                    0.0,
                    -3.4e-5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    95.7,
                    heater,
                    0.0,
                )
            )
        )
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf8")
    return path


def messages(parsed) -> str:
    return " ".join(i.message for i in parsed.issues).lower()


def severities(parsed) -> set[Severity]:
    return {i.severity for i in parsed.issues}


class TestGeometryCorrection:
    def test_the_measured_dataset_factor(self):
        """10 x 5 x 0.25 mm against a 1/1/1 default is a factor of 0.125."""
        assert geometry_correction(0.25, 5.0, 10.0) == pytest.approx(0.125)

    def test_identity_when_the_entered_geometry_was_right(self):
        assert geometry_correction(
            0.25, 5.0, 10.0, entered_thickness_mm=0.25, entered_width_mm=5.0, entered_length_mm=10.0
        ) == pytest.approx(1.0)

    def test_rho_and_kappa_corrections_are_reciprocal(self):
        """This is why ZT and L/L0 cannot be repaired by entering geometry.

        rho scales with (w.t)/L and kappa with L/(w.t), so their product - and
        so ZT and the Lorenz ratio built on it - is geometry-free.
        """
        k = geometry_correction(0.25, 5.0, 10.0)
        assert k * (1.0 / k) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [(0, 5, 10), (-1, 5, 10), (0.25, 0, 10), (0.25, 5, 0)])
    def test_non_positive_dimensions_are_rejected(self, bad):
        with pytest.raises(ValueError, match="positive"):
            geometry_correction(*bad)


class TestSniffing:
    def test_recognises_the_single_channel_export(self, tmp_path):
        assert PpmsTtoParser().can_parse(write_single(tmp_path)) >= 0.9

    def test_recognises_the_processed_export_that_has_no_geometry_line(self, tmp_path):
        """The processed file starts at the header; only its `_raw` sibling has geometry.

        Requiring the geometry line here would reject exactly the files a
        caller normally hands over.
        """
        assert PpmsTtoParser().can_parse(write_single(tmp_path, geometry=None)) >= 0.9

    def test_recognises_the_two_channel_export(self, tmp_path):
        assert PpmsTtoParser().can_parse(write_two_channel(tmp_path)) >= 0.9

    def test_rejects_an_unrelated_file(self, tmp_path):
        other = tmp_path / "notes.txt"
        other.write_text("wavelength,intensity\n400,0.1\n", encoding="utf8")
        assert PpmsTtoParser().can_parse(other) == 0.0

    def test_missing_file_scores_zero_rather_than_raising(self, tmp_path):
        assert PpmsTtoParser().can_parse(tmp_path / "absent.txt") == 0.0


class TestArrays:
    def test_reads_the_single_channel_columns(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, n=12))
        assert parsed.technique is Technique.THERMOELECTRIC
        for name in (
            "temperature_k",
            "seebeck_microv_per_k",
            "resistivity_microohm_cm",
            "thermal_conductivity_w_per_mk",
            "zt",
            "lorenz_ratio",
        ):
            assert name in parsed.arrays, name
            assert parsed.arrays[name].shape == (12,)
        assert parsed.metadata["n_points"] == 12

    def test_reads_the_two_channel_columns_despite_different_names(self, tmp_path):
        """`TR_K`, `S1` and `HeaterI_` carry the same quantities under other names."""
        parsed = PpmsTtoParser().parse(write_two_channel(tmp_path))
        assert "temperature_k" in parsed.arrays
        assert "seebeck_microv_per_k" in parsed.arrays
        assert "heater_current_ma" in parsed.arrays
        assert parsed.arrays["temperature_k"][0] == pytest.approx(270.0)

    def test_seebeck_is_surfaced_as_a_feature(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, seebeck=15.4))
        assert parsed.features["seebeck_median_microv_per_k"] == pytest.approx(15.4)


class TestSilentFailureModes:
    def test_default_geometry_is_flagged(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, geometry=(1.0, 1.0, 1.0)))
        assert parsed.metadata["geometry_entered"] is False
        assert "1/1/1 default" in messages(parsed)
        assert "cannot be repaired" in messages(parsed)

    def test_real_geometry_is_not_flagged(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, geometry=(0.25, 5.0, 10.0)))
        assert parsed.metadata["geometry_entered"] is True
        assert "default" not in messages(parsed)

    def test_geometry_is_recovered_from_the_raw_sibling(self, tmp_path):
        """The processed export omits geometry; the `_raw` file written alongside has it."""
        write_single(tmp_path, name="run_raw.txt", geometry=(0.25, 5.0, 10.0))
        parsed = PpmsTtoParser().parse(write_single(tmp_path, name="run.txt", geometry=None))
        assert parsed.metadata["geometry_source"] == "run_raw.txt"
        assert parsed.metadata["entered_width_mm"] == pytest.approx(5.0)
        assert parsed.metadata["geometry_entered"] is True

    def test_negative_thermal_conductivity_is_flagged(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, kappa=-1.6e-5))
        assert "negative at every point" in messages(parsed)
        assert "seebeck is unaffected" in messages(parsed)

    def test_physical_thermal_conductivity_is_not_flagged(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, kappa=1.2, lorenz=1.0))
        assert "negative at every point" not in messages(parsed)

    def test_lorenz_ratio_far_from_unity_is_flagged(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, lorenz=-1.8e-3))
        assert "lorenz" in messages(parsed)
        assert "geometry-free" in messages(parsed)

    def test_plausible_lorenz_ratio_is_not_flagged(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, kappa=1.2, lorenz=1.05))
        assert "lorenz" not in messages(parsed)

    def test_negative_resistivity_is_flagged_with_a_count(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_single(tmp_path, n=13, rho=-18.0))
        assert "resistivity is negative for 13 of 13" in messages(parsed)

    def test_heater_never_energised_is_flagged(self, tmp_path):
        """A resistance sweep with the heater off is not a thermoelectric measurement.

        Its Seebeck column still contains numbers, which is the trap: they are
        an offset artefact, not a measurement.
        """
        parsed = PpmsTtoParser().parse(write_two_channel(tmp_path, heater=0.0))
        assert "no thermal gradient was ever applied" in messages(parsed)
        assert "offset artefact" in messages(parsed)

    def test_energised_heater_is_not_flagged(self, tmp_path):
        parsed = PpmsTtoParser().parse(write_two_channel(tmp_path, heater=9.5))
        assert "no thermal gradient" not in messages(parsed)

    def test_empty_second_channel_is_reported(self, tmp_path):
        """A sample was mounted and never wired; its data does not exist elsewhere."""
        parsed = PpmsTtoParser().parse(write_two_channel(tmp_path))
        assert parsed.metadata["second_channel_empty"] is True
        assert "never wired" in messages(parsed)
        assert Severity.INFO in severities(parsed)

    def test_aborted_run_is_an_error_not_a_crash(self, tmp_path):
        path = tmp_path / "aborted.txt"
        path.write_text(
            "thickness: 1.000000, width: 1.000000, length: 1.000000\n" + SINGLE_HEADER,
            encoding="utf8",
        )
        parsed = PpmsTtoParser().parse(path)
        assert parsed.arrays == {}
        assert Severity.ERROR in severities(parsed)
        assert "no data rows" in messages(parsed)


class TestRobustness:
    def test_never_raises_on_junk(self, tmp_path):
        for content in ("", "\n\n\n", "garbage", SINGLE_HEADER, "a\tb\nc\td"):
            path = tmp_path / "junk.txt"
            path.write_text(content, encoding="utf8")
            parsed = PpmsTtoParser().parse(path)
            assert parsed.parser_name == "ppms-tto-txt"

    def test_never_raises_on_a_missing_file(self, tmp_path):
        parsed = PpmsTtoParser().parse(tmp_path / "absent.txt")
        assert Severity.ERROR in severities(parsed)

    def test_unparseable_rows_are_counted_not_fatal(self, tmp_path):
        path = write_single(tmp_path, n=5)
        path.write_text(path.read_text(encoding="utf8") + "\nnot\tnumbers\there\n", encoding="utf8")
        parsed = PpmsTtoParser().parse(path)
        assert parsed.metadata["n_points"] == 5
        assert "could not be read as numbers" in messages(parsed)

    def test_ragged_rows_are_padded_not_dropped(self, tmp_path):
        path = write_single(tmp_path, n=4)
        path.write_text(path.read_text(encoding="utf8") + "\n1.0\t305.0\n", encoding="utf8")
        parsed = PpmsTtoParser().parse(path)
        assert parsed.metadata["n_points"] == 5
        assert np.isnan(parsed.arrays["seebeck_microv_per_k"][-1])
