"""Tests for `latos.analysis.microscopy.lattice_spacing`.

The kernel this analyzer wraps is already covered by `test_lattice`, and the
info-bar decoder by `test_calibration`. What is untested until here is the
wiring between them, which is exactly what was missing from the product: a
registered analyzer that opens a frame, recovers its pixel size, and either
returns a length or refuses to.

So these tests are about the seams:

1. **Recovery** — a synthetic frame carrying fringes of a known period, behind
   a synthetic info bar of a known field of view, comes back with that period
   in nanometres. This is the only assertion that proves calibration and FFT
   were composed the right way round; a pixel-size error would still produce a
   plausible number.
2. **Refusal** — the three ways this dataset's frames fail (no bar, no matching
   template, an impossible "2 m" field of view) each produce no length and a
   message naming that specific cause, because the three need different fixes.
3. **Registration** — the analyzer is in `default_registry` and claims TEM.
   The whole point of this module is that a tested kernel nothing dispatched to
   is not a feature, so the dispatch is the thing to pin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.microscopy.lattice_spacing import MicroscopyLatticeAnalyzer
from latos.analysis.registry import default_registry
from latos.core.enums import FileRole, Severity, Technique
from latos.core.models import FileRef, Measurement, new_id, utc_now

from .test_calibration import make_frame, templates_for

# The field of view the synthetic bar prints, and the frame it prints it on.
FIELD_TEXT = "21.7 nm"
FIELD_NM = 21.7
WIDTH = 1024
NM_PER_PX = FIELD_NM / WIDTH

# Well inside DEFAULT_D_WINDOW_NM, and ~17 repeats across the window — far
# above the min_repeats floor, so the FFT can place the peak precisely.
PLANTED_D_NM = 1.25


def _fringed_frame(d_nm: float = PLANTED_D_NM, *, field_text: str = FIELD_TEXT) -> np.ndarray:
    """A JEOL-style frame whose image area is fringes of period `d_nm`."""
    frame = make_frame(field_text, width=WIDTH)
    coords = np.arange(WIDTH, dtype=np.float64)
    wave = np.sin(2.0 * np.pi * NM_PER_PX * coords / d_nm)
    frame[:WIDTH, :] = (128 + 100 * np.tile(wave, (WIDTH, 1))).astype(np.uint8)
    return frame


def _measurement(path: Path, *, technique: Technique = Technique.TEM) -> Measurement:
    return Measurement(
        id=new_id(),
        sample_id=new_id(),
        technique=technique,
        instrument="JEOL JEM-2100F (synthetic)",
        measured_at=utc_now(),
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(
            FileRef(
                path=path,
                sha256="a" * 64,
                size_bytes=path.stat().st_size if path.exists() else 1,
                role=FileRole.RAW,
                scanned_at=utc_now(),
            ),
        ),
    )


@pytest.fixture
def templates_npz(tmp_path: Path) -> Path:
    path = tmp_path / "jeol_synthetic_templates.npz"
    templates_for(
        [FIELD_TEXT, "2 m", "2.3 um", "14.4 nm"],
        widths=(WIDTH,),
    ).save(path)
    return path


def _run(
    frame: np.ndarray,
    tmp_path: Path,
    templates: Path | None,
    **overrides: object,
) -> tuple:
    image_path = tmp_path / "frame.tif"
    tifffile.imwrite(image_path, frame)
    analyzer = MicroscopyLatticeAnalyzer()
    params = analyzer.merge_params(
        {"templates_path": str(templates) if templates else None, **overrides},
    )
    output = analyzer.analyze(
        AnalyzerInputs(measurement=_measurement(image_path), arrays={}, params=params),
    )
    return output, analyzer


class TestRecovery:
    """A known spacing behind a known info bar comes back as a length."""

    def test_planted_spacing_is_recovered_in_nanometres(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        output, _ = _run(_fringed_frame(), tmp_path, templates_npz)
        assert output.outputs["n_windows_detected"] >= 1
        measured = output.outputs["lattice_d_nm"]
        # Tolerance is the FFT's own resolution floor, not a tuned constant.
        assert measured == pytest.approx(
            PLANTED_D_NM,
            abs=max(output.outputs["lattice_d_err_nm"], 0.02),
        )

    def test_pixel_size_comes_from_the_info_bar(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        output, _ = _run(_fringed_frame(), tmp_path, templates_npz)
        assert output.outputs["nm_per_px"] == pytest.approx(NM_PER_PX, abs=5e-7)
        assert output.outputs["field_of_view_label"] == FIELD_TEXT
        assert output.outputs["image_area_px"] == WIDTH

    def test_a_different_planted_spacing_moves_the_answer(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """Guards against a constant being returned regardless of the image."""
        wide, _ = _run(_fringed_frame(2.00), tmp_path, templates_npz)
        assert wide.outputs["lattice_d_nm"] == pytest.approx(2.00, abs=0.1)
        assert wide.outputs["lattice_d_nm"] > PLANTED_D_NM

    def test_no_issues_raised_on_a_clean_frame(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        output, _ = _run(_fringed_frame(), tmp_path, templates_npz)
        assert output.issues == ()


class TestRefusal:
    """Each way a frame can fail says which way it failed."""

    def test_frame_without_an_info_bar_yields_no_length(self, tmp_path: Path) -> None:
        square = np.full((WIDTH, WIDTH), 128, dtype=np.uint8)
        templates = tmp_path / "t.npz"
        templates_for([FIELD_TEXT], widths=(WIDTH,)).save(templates)
        output, _ = _run(square, tmp_path, templates)
        assert "lattice_d_nm" not in output.outputs
        assert output.issues[0].severity is Severity.WARNING
        assert "without its info bar" in output.issues[0].message

    def test_unmatched_strip_asks_for_a_label_rather_than_guessing(
        self,
        tmp_path: Path,
    ) -> None:
        templates = tmp_path / "t.npz"
        templates_for(["44.7 nm"], widths=(WIDTH,)).save(templates)
        output, _ = _run(_fringed_frame(), tmp_path, templates)
        assert "lattice_d_nm" not in output.outputs
        assert "matches no template" in output.issues[0].message

    def test_impossible_field_of_view_is_refused_by_name(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """'2 m' is what the instrument writes when magnification was lost."""
        output, _ = _run(_fringed_frame(field_text="2 m"), tmp_path, templates_npz)
        assert "lattice_d_nm" not in output.outputs
        assert "'2 m'" in output.issues[0].message

    def test_missing_templates_is_an_error_not_a_warning(self, tmp_path: Path) -> None:
        """Nothing can be measured at all, as opposed to not from this frame."""
        output, _ = _run(_fringed_frame(), tmp_path, None)
        assert output.outputs == {}
        assert output.issues[0].severity is Severity.ERROR

    def test_calibrated_but_featureless_frame_reports_zero_not_a_refusal(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """A survey frame is normal, so it keeps its calibration and says INFO."""
        flat = make_frame(FIELD_TEXT, width=WIDTH)
        flat[:WIDTH, :] = 128
        output, _ = _run(flat, tmp_path, templates_npz)
        assert output.outputs["n_windows_detected"] == 0
        assert output.outputs["nm_per_px"] == pytest.approx(NM_PER_PX, abs=5e-7)
        assert output.issues[0].severity is Severity.INFO


class TestResolvability:
    """The two limits that bracket what a frame can measure.

    Both were found by running the registered analyzer over the real dataset,
    not reasoned about in advance. At a 2.3 um field it reported a 2.28 nm
    "spacing" from a period of two pixels, and across the whole set the median
    spacing rose monotonically with the field of view — the signature of a
    detector reporting its own resolution rather than the sample.
    """

    def test_coarse_pixels_are_refused_rather_than_measured(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """2.3 um over 1024 px samples 0.70 nm over a third of a pixel."""
        output, _ = _run(_fringed_frame(field_text="2.3 um"), tmp_path, templates_npz)
        assert "lattice_d_nm" not in output.outputs
        assert output.issues[0].severity is Severity.WARNING
        assert "below the 25 px floor" in output.issues[0].message

    def test_the_floor_is_the_parameter_not_a_constant(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """The same frame passes at 25 and fails at 100, so the knob is wired."""
        passes, _ = _run(_fringed_frame(), tmp_path, templates_npz)
        assert passes.outputs["n_windows_detected"] >= 1
        fails, _ = _run(
            _fringed_frame(),
            tmp_path,
            templates_npz,
            min_px_per_period=100.0,
        )
        assert "lattice_d_nm" not in fails.outputs

    def test_gate_is_measured_against_the_smallest_sought_spacing(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """Not against the spacing found, which would be circular."""
        output, _ = _run(_fringed_frame(), tmp_path, templates_npz)
        expected = output.outputs["d_window_nm_used"][0] / NM_PER_PX
        assert output.outputs["px_per_smallest_period"] == pytest.approx(expected, abs=0.1)

    def test_high_magnification_narrows_the_window_and_says_so(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """A 14.4 nm view holds only 6 repeats of 2.4 nm, not of the 2.8 asked for."""
        frame = make_frame("14.4 nm", width=WIDTH)
        fine_nm_per_px = 14.4 / WIDTH
        coords = np.arange(WIDTH, dtype=np.float64)
        wave = np.sin(2.0 * np.pi * fine_nm_per_px * coords / 1.0)
        frame[:WIDTH, :] = (128 + 100 * np.tile(wave, (WIDTH, 1))).astype(np.uint8)
        output, _ = _run(frame, tmp_path, templates_npz)

        used = output.outputs["d_window_nm_used"]
        assert used[1] == pytest.approx(14.4 / 6.0, abs=0.01)
        assert used[1] < 2.80
        narrowing = [i for i in output.issues if i.field == "d_window_nm"]
        assert narrowing and "narrowed to" in narrowing[0].message

    def test_window_that_cannot_hold_its_smallest_spacing_is_refused(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        """A tile too small for even the bottom of the range measures nothing."""
        output, _ = _run(
            _fringed_frame(),
            tmp_path,
            templates_npz,
            tile_size=128,
            d_window_nm=[0.70, 2.80],
        )
        assert "lattice_d_nm" not in output.outputs
        assert "cannot hold" in output.issues[0].message

    def test_a_frame_within_both_limits_reports_the_full_window(
        self,
        tmp_path: Path,
        templates_npz: Path,
    ) -> None:
        output, _ = _run(_fringed_frame(), tmp_path, templates_npz)
        assert output.outputs["d_window_nm_used"] == [0.7, 2.8]
        assert output.issues == ()


class TestDispatch:
    """The gap this module closed was dispatch, so dispatch is pinned."""

    def test_registered_in_the_default_registry(self) -> None:
        names = {a.name for a in default_registry()._analyzers}
        assert "microscopy-lattice" in names

    def test_registry_dispatches_tem_to_it(self, tmp_path: Path) -> None:
        image_path = tmp_path / "frame.tif"
        tifffile.imwrite(image_path, _fringed_frame())
        found = default_registry().find_for(_measurement(image_path))
        assert "microscopy-lattice" in {a.name for a in found}

    def test_accepts_needs_an_openable_image(self, tmp_path: Path) -> None:
        analyzer = MicroscopyLatticeAnalyzer()
        assert analyzer.accepts(_measurement(tmp_path / "frame.tif"))
        # .bmp exports carry no info bar, so borrowing a calibration for them
        # would manufacture one; the parser refuses them for the same reason.
        assert not analyzer.accepts(_measurement(tmp_path / "frame.bmp"))

    def test_declines_techniques_it_cannot_calibrate(self, tmp_path: Path) -> None:
        found = default_registry().find_for(
            _measurement(tmp_path / "frame.tif", technique=Technique.SEM),
        )
        assert "microscopy-lattice" not in {a.name for a in found}
