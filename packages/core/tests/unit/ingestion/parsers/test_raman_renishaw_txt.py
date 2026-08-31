"""Tests for `RenishawRamanTxtParser`.

Spectra are synthesised so the answer is known by construction: a Lorentzian
band of known width sits at a known position, and any spike or saturation is
put there deliberately.

The load-bearing test is the pair around cosmic rays. Flagging a spike is easy;
NOT flagging the summit of a real band is the hard half, and a detector that
failed it would strip peaks out of every spectrum it touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.raman_renishaw_txt import (
    RenishawRamanTxtParser,
    find_cosmic_rays,
)

HEADER = "#Wave\t\t#Intensity"


def lorentzian(x: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    return height / (1.0 + ((x - centre) / width) ** 2)


def spectrum(
    n: int = 400,
    lo: float = 100.0,
    hi: float = 1800.0,
    *,
    bands: tuple[tuple[float, float, float], ...] = ((157.0, 8.0, 1000.0),),
    noise: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ascending wavenumber and a spectrum with the requested bands."""
    x = np.linspace(lo, hi, n)
    y = np.full_like(x, 50.0)
    for centre, width, height in bands:
        y = y + lorentzian(x, centre, width, height)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, x.size)
    return x, y


def write(
    tmp_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    name: str = "s.txt",
    descending: bool = True,
    header: str = HEADER,
    newline: str = "\r\n",
) -> Path:
    """Write a WiRE-style export. Renishaw writes descending, with CRLF."""
    order = np.argsort(-x if descending else x)
    rows = [header] + [f"{x[i]:.6f}\t{y[i]:.6f}" for i in order]
    path = tmp_path / name
    path.write_bytes(newline.join(rows).encode("utf8"))
    return path


def messages(parsed) -> str:
    return " ".join(i.message for i in parsed.issues).lower()


def severities(parsed) -> set[Severity]:
    return {i.severity for i in parsed.issues}


class TestCosmicRays:
    def test_a_single_point_spike_is_found(self):
        _x, y = spectrum(noise=3.0)
        y[200] = y[200] + 3000.0
        found = find_cosmic_rays(y)
        assert 200 in found.tolist()

    def test_a_real_band_apex_is_not_flagged(self):
        """The discriminator is WIDTH, not height.

        A Raman band spans many points, so its apex sits on a shoulder nearly
        as high. A cosmic ray lands on one pixel with nothing beside it. Were
        this test to fail, the parser would delete real peaks.
        """
        _x, y = spectrum(bands=((600.0, 12.0, 5000.0),), noise=3.0)
        assert find_cosmic_rays(y).size == 0

    def test_a_tall_narrow_band_is_still_not_flagged(self):
        _x, y = spectrum(n=800, bands=((157.0, 6.0, 20000.0),), noise=5.0)
        assert find_cosmic_rays(y).size == 0

    def test_several_spikes_are_all_found(self):
        _x, y = spectrum(noise=3.0)
        for i in (80, 190, 330):
            y[i] = y[i] + 4000.0
        assert set(find_cosmic_rays(y).tolist()) >= {80, 190, 330}

    def test_a_flat_spectrum_yields_none(self):
        assert find_cosmic_rays(np.full(300, 100.0)).size == 0

    def test_a_spectrum_shorter_than_the_window_yields_none(self):
        assert find_cosmic_rays(np.arange(4.0)).size == 0

    @pytest.mark.parametrize("window", [2, 4, 8, 1])
    def test_an_even_or_tiny_window_is_rejected(self, window):
        with pytest.raises(ValueError, match="odd"):
            find_cosmic_rays(np.zeros(100), window=window)


class TestSniffing:
    def test_recognises_the_wire_header(self, tmp_path):
        x, y = spectrum()
        assert RenishawRamanTxtParser().can_parse(write(tmp_path, x, y)) == 1.0

    def test_tolerates_a_single_tab_in_the_header(self, tmp_path):
        """The tab count has been seen to vary between WiRE versions."""
        x, y = spectrum()
        path = write(tmp_path, x, y, header="#Wave\t#Intensity")
        assert RenishawRamanTxtParser().can_parse(path) == 1.0

    def test_rejects_a_file_without_the_header(self, tmp_path):
        x, y = spectrum()
        path = write(tmp_path, x, y, header="wavelength,intensity")
        assert RenishawRamanTxtParser().can_parse(path) == 0.0

    def test_missing_file_scores_zero_rather_than_raising(self, tmp_path):
        assert RenishawRamanTxtParser().can_parse(tmp_path / "absent.txt") == 0.0


class TestArrays:
    def test_descending_input_is_published_ascending(self, tmp_path):
        """Renishaw writes high-to-low; every other spectroscopy parser here is low-to-high."""
        x, y = spectrum()
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y, descending=True))
        shifts = parsed.arrays["raman_shift_cm1"]
        assert np.all(np.diff(shifts) > 0)
        assert parsed.metadata["stored_descending"] is True

    def test_ascending_input_is_left_alone_and_reported(self, tmp_path):
        x, y = spectrum()
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y, descending=False))
        assert np.all(np.diff(parsed.arrays["raman_shift_cm1"]) > 0)
        assert parsed.metadata["stored_descending"] is False

    def test_intensity_follows_the_reordering(self, tmp_path):
        """Sorting the axis must carry the intensities with it."""
        x, y = spectrum(bands=((157.0, 8.0, 5000.0),), noise=0.0)
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        shifts, counts = parsed.arrays["raman_shift_cm1"], parsed.arrays["intensity"]
        assert shifts[int(np.argmax(counts))] == pytest.approx(157.0, abs=10.0)

    def test_crlf_line_endings_are_handled(self, tmp_path):
        x, y = spectrum()
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y, newline="\r\n"))
        assert parsed.metadata["n_points"] == x.size

    def test_metadata_describes_the_axis(self, tmp_path):
        x, y = spectrum(n=400, lo=100.0, hi=1800.0)
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        assert parsed.technique is Technique.RAMAN
        assert parsed.metadata["n_points"] == 400
        assert parsed.metadata["wavenumber_min_cm1"] == pytest.approx(100.0)
        assert parsed.metadata["wavenumber_max_cm1"] == pytest.approx(1800.0)
        assert parsed.metadata["median_step_cm1"] > 0


class TestIssues:
    def test_absent_acquisition_settings_are_always_reported(self, tmp_path):
        """Laser power is not in the file, so intensities cannot be compared across files.

        Replicates of one sample have been measured differing 1.9x in absolute
        counts while agreeing on band ratios, so this warning is not academic.
        """
        x, y = spectrum()
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        assert "not comparable between files" in messages(parsed)
        assert Severity.INFO in severities(parsed)

    def test_a_cosmic_ray_is_reported_with_its_position(self, tmp_path):
        x, y = spectrum(noise=3.0)
        y[200] += 4000.0
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        assert "cosmic-ray spike" in messages(parsed)
        assert f"{x[200]:.0f}" in " ".join(i.message for i in parsed.issues)

    def test_a_clean_spectrum_raises_no_warning(self, tmp_path):
        x, y = spectrum(bands=((600.0, 12.0, 4000.0),), noise=3.0)
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        assert Severity.WARNING not in severities(parsed)

    def test_detector_saturation_is_reported(self, tmp_path):
        x, y = spectrum(n=400, noise=0.0)
        y[180:200] = 65535.0
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        assert "saturated" in messages(parsed)

    def test_a_sparse_spectrum_is_reported(self, tmp_path):
        x, y = spectrum(n=20, noise=0.0)
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        assert "too sparse" in messages(parsed)

    def test_duplicate_wavenumbers_are_reported(self, tmp_path):
        x, y = spectrum(n=200, noise=0.0)
        x[100] = x[99]
        parsed = RenishawRamanTxtParser().parse(write(tmp_path, x, y))
        assert "duplicates" in messages(parsed)

    def test_an_empty_export_is_an_error(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text(HEADER, encoding="utf8")
        parsed = RenishawRamanTxtParser().parse(path)
        assert parsed.arrays == {}
        assert Severity.ERROR in severities(parsed)

    def test_unreadable_rows_are_counted_not_fatal(self, tmp_path):
        x, y = spectrum(n=200, noise=0.0)
        path = write(tmp_path, x, y)
        path.write_bytes(path.read_bytes() + b"\r\nnot\tanumber\r\n")
        parsed = RenishawRamanTxtParser().parse(path)
        assert parsed.metadata["n_points"] == 200
        assert "could not be read" in messages(parsed)


class TestRobustness:
    def test_never_raises_on_junk(self, tmp_path):
        for content in (b"", b"\r\n\r\n", b"#Wave\t\t#Intensity\r\n", b"garbage"):
            path = tmp_path / "junk.txt"
            path.write_bytes(content)
            parsed = RenishawRamanTxtParser().parse(path)
            assert parsed.parser_name == "renishaw-raman-txt"

    def test_never_raises_on_a_missing_file(self, tmp_path):
        parsed = RenishawRamanTxtParser().parse(tmp_path / "absent.txt")
        assert Severity.ERROR in severities(parsed)
