"""Tests for `latos.analysis.microscopy.lattice`.

Ground truth is synthetic: a sinusoidal fringe pattern of known period is built
at a known pixel size, so the spacing the detector should return is known
exactly rather than assumed. Tolerances are tied to the FFT's own resolution -
a window holding R repeats cannot localise a peak better than about 1/R - so
the assertions scale with the window rather than being hand-tuned constants.

Three tiers:

1. **Recovery** - known spacings, angles and window sizes come back correctly,
   and the reported `n_repeats` really does track achievable precision.
2. **Discrimination** - amorphous texture, flat fields, and spacings outside
   the requested window all return None rather than a plausible-looking
   number. Silent false positives are the dangerous failure here.
3. **Harmonics** - the regression that motivated the design. A second-order
   reflection stronger than the first must not be reported as the spacing,
   and the documented limit of the window rule is pinned so a future change
   to it is a deliberate one.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.analysis.microscopy.lattice import (
    aggregate_frames,
    analyse_tile,
    iter_tiles,
    scan_frame,
)

NM_PER_PX = 0.05


def fringes(
    n: int,
    d_nm: float,
    *,
    nm_per_px: float = NM_PER_PX,
    angle_deg: float = 0.0,
    amplitude: float = 1.0,
    harmonics: dict[int, float] | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A square field of parallel fringes with period `d_nm`.

    `harmonics` maps order -> amplitude, so ``{2: 3.0}`` puts a second-order
    component three times stronger than the fundamental into the image - the
    situation that makes "take the strongest peak" report half the spacing.
    """
    yy, xx = np.indices((n, n)).astype(float)
    theta = np.radians(angle_deg)
    projection = xx * np.cos(theta) + yy * np.sin(theta)
    k = 2.0 * np.pi * nm_per_px / d_nm
    image = amplitude * np.sin(k * projection)
    for order, amp in (harmonics or {}).items():
        image = image + amp * np.sin(order * k * projection)
    if noise:
        image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return image


def amorphous(n: int, *, seed: int = 0, blur: int = 4) -> np.ndarray:
    """Isotropic blobby texture, of the kind a support film gives."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, (n, n))
    kernel = np.ones(blur) / blur
    for _ in range(3):
        raw = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 0, raw)
        raw = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, raw)
    return raw


class TestRecovery:
    @pytest.mark.parametrize("d_nm", [0.80, 1.20, 1.60, 2.20])
    def test_known_spacing_is_recovered(self, d_nm: float) -> None:
        peak = analyse_tile(fringes(512, d_nm), NM_PER_PX)
        assert peak is not None
        # The FFT can only localise the peak to ~1/n_repeats.
        assert peak.d_nm == pytest.approx(d_nm, rel=1.5 / peak.n_repeats)

    @pytest.mark.parametrize("angle", [0.0, 23.0, 45.0, 67.0, 90.0, 152.0])
    def test_recovery_is_independent_of_fringe_orientation(self, angle: float) -> None:
        peak = analyse_tile(fringes(512, 1.20, angle_deg=angle), NM_PER_PX)
        assert peak is not None
        assert peak.d_nm == pytest.approx(1.20, rel=0.10)

    def test_reported_angle_tracks_the_fringe_normal(self) -> None:
        for angle in (0.0, 30.0, 60.0, 120.0):
            peak = analyse_tile(fringes(512, 1.20, angle_deg=angle), NM_PER_PX)
            assert peak is not None
            delta = abs(peak.angle_deg - angle) % 180.0
            assert min(delta, 180.0 - delta) <= 8.0

    def test_survives_noise_comparable_to_the_signal(self) -> None:
        peak = analyse_tile(fringes(512, 1.20, noise=1.0), NM_PER_PX)
        assert peak is not None
        assert peak.d_nm == pytest.approx(1.20, rel=0.10)

    def test_repeats_and_error_scale_with_window_size(self) -> None:
        """A bigger window holds more repeats, so it measures more precisely."""
        small = analyse_tile(fringes(256, 1.20), NM_PER_PX)
        large = analyse_tile(fringes(1024, 1.20), NM_PER_PX)
        assert small is not None and large is not None
        assert large.n_repeats > small.n_repeats * 3
        assert large.d_err_nm < small.d_err_nm
        # The stated uncertainty must actually bracket the truth.
        assert abs(large.d_nm - 1.20) <= 2 * large.d_err_nm
        assert abs(small.d_nm - 1.20) <= 2 * small.d_err_nm

    def test_contrast_is_far_above_unity_for_a_real_fringe(self) -> None:
        peak = analyse_tile(fringes(512, 1.20), NM_PER_PX)
        assert peak is not None
        assert peak.contrast > 50


class TestDiscrimination:
    def test_amorphous_texture_yields_nothing(self) -> None:
        assert analyse_tile(amorphous(512), NM_PER_PX) is None

    def test_flat_field_yields_nothing(self) -> None:
        assert analyse_tile(np.full((512, 512), 128.0), NM_PER_PX) is None

    def test_pure_noise_yields_nothing(self) -> None:
        rng = np.random.default_rng(7)
        assert analyse_tile(rng.normal(0, 1, (512, 512)), NM_PER_PX) is None

    def test_spacing_outside_the_window_is_not_reported(self) -> None:
        """A 0.30 nm fringe must not be squeezed into a 0.70-2.80 window."""
        assert analyse_tile(fringes(512, 0.30), NM_PER_PX) is None

    def test_spacing_above_the_window_is_not_reported(self) -> None:
        assert analyse_tile(fringes(512, 1.20), NM_PER_PX, d_window_nm=(0.3, 0.6)) is None

    def test_raising_min_contrast_suppresses_a_weak_fringe(self) -> None:
        weak = fringes(512, 1.20, amplitude=0.05, noise=1.0)
        assert analyse_tile(weak, NM_PER_PX, min_contrast=1e6) is None

    def test_tiny_tile_yields_nothing(self) -> None:
        assert analyse_tile(np.zeros((8, 8)), NM_PER_PX) is None

    @pytest.mark.parametrize("nm_per_px", [0.007031, 0.010596, 0.02, 0.05])
    def test_illumination_gradient_is_not_a_lattice(self, nm_per_px: float) -> None:
        """Regression: a smooth gradient reported a confident large spacing.

        Uneven illumination piles power near the origin, and the shoulder of
        that pile contains local maxima a few bins out. Measured against an
        azimuthal median that is essentially zero out there, such a maximum
        scores a contrast in the thousands - far above anything a real fringe
        needs - and maps to a d near the top of the window. Nothing about the
        result looks suspicious, which is what made it dangerous.
        """
        yy, xx = np.indices((512, 512)).astype(float)
        ramp = (xx / 512) * 2.0 + (yy / 512) * 1.2 + 3.0 * ((xx / 512) ** 2)
        tile = ramp * 40 + np.random.default_rng(0).normal(0, 1.0, (512, 512))
        assert analyse_tile(tile, nm_per_px) is None

    def test_a_spacing_seen_too_few_times_is_refused(self) -> None:
        """Below `min_repeats` the FFT cannot resolve the peak meaningfully."""
        # 512 px at 0.05 nm/px spans 25.6 nm, so a 6.4 nm spacing fits 4 times.
        assert analyse_tile(fringes(512, 6.40), NM_PER_PX, d_window_nm=(4.0, 8.0)) is None
        # The same fringe in a window holding four times as many repeats is fine.
        peak = analyse_tile(fringes(2048, 6.40), NM_PER_PX, d_window_nm=(4.0, 8.0))
        assert peak is not None
        assert peak.d_nm == pytest.approx(6.40, rel=0.10)

    def test_every_reported_peak_clears_the_repeat_floor(self) -> None:
        for d_nm in (0.80, 1.20, 2.20):
            peak = analyse_tile(fringes(512, d_nm), NM_PER_PX)
            assert peak is not None
            assert peak.n_repeats >= 6.0


class TestHarmonics:
    def test_stronger_second_order_does_not_halve_the_reported_spacing(self) -> None:
        """The regression this design exists for.

        With a second-order component three times the fundamental, picking the
        globally strongest peak reports d/2. Restricting the fundamental to a
        physical window keeps the answer right.
        """
        image = fringes(512, 1.20, harmonics={2: 3.0})
        peak = analyse_tile(image, NM_PER_PX)
        assert peak is not None
        assert peak.d_nm == pytest.approx(1.20, rel=0.10)
        assert peak.d_nm > 0.9, "reported the second order rather than the fundamental"

    def test_harmonics_are_counted_as_corroboration(self) -> None:
        plain = analyse_tile(fringes(512, 1.60), NM_PER_PX, d_window_nm=(1.0, 2.5))
        laddered = analyse_tile(
            fringes(512, 1.60, harmonics={2: 0.6, 3: 0.4}),
            NM_PER_PX,
            d_window_nm=(1.0, 2.5),
        )
        assert plain is not None and laddered is not None
        assert laddered.n_orders > plain.n_orders

    def test_all_peaks_are_exposed_for_audit(self) -> None:
        peak = analyse_tile(fringes(512, 1.60, harmonics={2: 0.6}), NM_PER_PX)
        assert peak is not None
        assert len(peak.all_d_nm) >= 2
        assert peak.all_d_nm == tuple(sorted(peak.all_d_nm, reverse=True))
        assert any(abs(d - 0.80) < 0.15 for d in peak.all_d_nm), "second order missing"

    def test_window_spanning_more_than_an_octave_can_be_ambiguous(self) -> None:
        """Pinning a documented limitation, so changing it is deliberate.

        When the caller's window is wide enough to contain both a fundamental
        and its second order, and the second order is the stronger, the rule
        selects the second order. This is why the window is meant to bracket
        the expected spacing rather than span the whole plausible range.
        """
        image = fringes(512, 2.40, harmonics={2: 4.0})
        peak = analyse_tile(image, NM_PER_PX, d_window_nm=(0.7, 2.8))
        assert peak is not None
        assert peak.d_nm == pytest.approx(1.20, rel=0.15)
        # Narrowing the window to bracket the true spacing recovers it.
        narrowed = analyse_tile(image, NM_PER_PX, d_window_nm=(1.8, 2.8))
        assert narrowed is not None
        assert narrowed.d_nm == pytest.approx(2.40, rel=0.15)


class TestValidation:
    def test_non_square_tile_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            analyse_tile(np.zeros((64, 128)), NM_PER_PX)

    def test_three_dimensional_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            analyse_tile(np.zeros((8, 8, 3)), NM_PER_PX)

    @pytest.mark.parametrize("bad", [0.0, -0.05, float("nan"), float("inf")])
    def test_bad_pixel_size_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="nm_per_px"):
            analyse_tile(np.zeros((64, 64)), bad)

    @pytest.mark.parametrize("window", [(2.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (1.0, 1.0)])
    def test_bad_window_is_rejected(self, window: tuple[float, float]) -> None:
        with pytest.raises(ValueError, match="d_window_nm"):
            analyse_tile(np.zeros((64, 64)), NM_PER_PX, d_window_nm=window)


class TestTiling:
    def test_tiles_cover_the_image_at_the_expected_stride(self) -> None:
        tiles = list(iter_tiles(np.zeros((2048, 2048)), 1024, 512))
        assert len(tiles) == 9
        assert all(t.shape == (1024, 1024) for _r, _c, t in tiles)
        assert [r for r, _c, _t in tiles][:3] == [0, 0, 0]

    def test_window_is_clipped_when_the_image_is_smaller(self) -> None:
        tiles = list(iter_tiles(np.zeros((300, 300)), 1024, 512))
        assert len(tiles) == 1
        assert tiles[0][2].shape == (300, 300)

    def test_tiles_are_views_onto_the_source(self) -> None:
        image = np.arange(64 * 64, dtype=float).reshape(64, 64)
        _r, _c, tile = next(iter(iter_tiles(image, 32, 32)))
        assert np.array_equal(tile, image[:32, :32])

    @pytest.mark.parametrize(("size", "stride"), [(0, 1), (1, 0), (-4, 4)])
    def test_bad_tiling_parameters_are_rejected(self, size: int, stride: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            list(iter_tiles(np.zeros((16, 16)), size, stride))

    def test_non_two_dimensional_image_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            list(iter_tiles(np.zeros((4, 4, 4)), 2, 2))


class TestScanFrame:
    def test_a_frame_of_fringes_yields_a_detection_per_window(self) -> None:
        peaks = scan_frame(fringes(1024, 1.20), NM_PER_PX, tile_size=512, stride=256)
        assert len(peaks) == 9
        assert all(p.d_nm == pytest.approx(1.20, rel=0.12) for p in peaks)

    def test_a_frame_of_support_film_yields_nothing(self) -> None:
        assert scan_frame(amorphous(1024), NM_PER_PX, tile_size=512, stride=256) == ()

    def test_a_crystalline_patch_in_an_amorphous_frame_is_found(self) -> None:
        """The reason scanning is required rather than measuring the centre."""
        frame = amorphous(1024, seed=3) * 0.5
        frame[:512, :512] += fringes(512, 1.20) * 3.0
        peaks = scan_frame(frame, NM_PER_PX, tile_size=512, stride=512)
        assert len(peaks) >= 1
        assert any(p.d_nm == pytest.approx(1.20, rel=0.12) for p in peaks)


class TestAggregation:
    def test_medians_within_then_across_frames(self) -> None:
        est = aggregate_frames({"a": [1.0, 1.2, 1.4], "b": [2.0], "c": [3.0]})
        assert est is not None
        assert est.d_nm == pytest.approx(2.0)  # medians 1.2, 2.0, 3.0
        assert est.n_frames == 3
        assert est.n_tiles == 5

    def test_one_densely_tiled_frame_cannot_dominate(self) -> None:
        """A frame with fifty windows still counts once, like every other."""
        est = aggregate_frames({"dense": [5.0] * 50, "b": [1.0], "c": [1.1]})
        assert est is not None
        assert est.d_nm == pytest.approx(1.1)
        assert est.n_frames == 3
        assert est.n_tiles == 52

    def test_spread_describes_variation_between_frames(self) -> None:
        tight = aggregate_frames({f"f{i}": [1.20] for i in range(6)})
        loose = aggregate_frames({f"f{i}": [1.0 + 0.1 * i] for i in range(6)})
        assert tight is not None and loose is not None
        assert tight.spread_nm == pytest.approx(0.0)
        assert loose.spread_nm > 0.1

    def test_quartiles_bracket_the_centre(self) -> None:
        est = aggregate_frames({f"f{i}": [1.0 + 0.1 * i] for i in range(9)})
        assert est is not None
        assert est.q1_nm <= est.d_nm <= est.q3_nm

    def test_no_detections_gives_none(self) -> None:
        assert aggregate_frames({}) is None
        assert aggregate_frames({"a": [], "b": []}) is None

    def test_frames_without_detections_are_ignored_not_counted(self) -> None:
        est = aggregate_frames({"a": [1.2], "b": [], "c": [1.4]})
        assert est is not None
        assert est.n_frames == 2

    def test_end_to_end_from_image_to_estimate(self) -> None:
        frames = {
            f"frame-{i}": [
                p.d_nm
                for p in scan_frame(
                    fringes(512, 1.15 + 0.02 * i, seed=i, noise=0.3),
                    NM_PER_PX,
                    tile_size=512,
                )
            ]
            for i in range(5)
        }
        est = aggregate_frames(frames)
        assert est is not None
        assert est.n_frames == 5
        assert est.d_nm == pytest.approx(1.19, abs=0.12)
