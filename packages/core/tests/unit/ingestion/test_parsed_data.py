"""Tests for `latos.ingestion.parsed_data.ParsedData`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

from latos.core.enums import Severity, Technique
from latos.core.exceptions import ValidationError
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.parsed_data import ParsedData


def _good_kwargs(**override: Any) -> dict[str, Any]:
    """Build a valid ParsedData kwargs dict; override individual fields for negative tests."""
    base: dict[str, Any] = {
        "technique": Technique.XRD,
        "arrays": {
            "two_theta": np.array([10.0, 20.0, 30.0]),
            "intensity": np.array([100.0, 250.0, 80.0]),
        },
        "metadata": {"wavelength_ka1": 1.5406, "sample": "MX-01"},
        "instrument": "Rigaku Ultima IV",
        "measured_at": utc_now() - timedelta(days=1),
        "issues": (),
        "parser_name": "rigaku-xrd-txt",
        "parser_version": "1.0.0",
    }
    base.update(override)
    return base


# ─── Happy path ─────────────────────────────────────────────────────
class TestValidConstruction:
    def test_constructs_with_valid_args(self):
        pd = ParsedData(**_good_kwargs())
        assert pd.technique is Technique.XRD
        assert pd.parser_name == "rigaku-xrd-txt"
        assert pd.parser_version == "1.0.0"

    def test_empty_arrays_allowed(self):
        # .tif metadata-only parsers return no arrays.
        ParsedData(**_good_kwargs(arrays={}))

    def test_single_array_allowed(self):
        # One array (just intensity, no x-axis) is fine.
        ParsedData(**_good_kwargs(arrays={"intensity": np.array([1.0, 2.0, 3.0])}))

    def test_none_instrument_allowed(self):
        ParsedData(**_good_kwargs(instrument=None))

    def test_none_measured_at_allowed(self):
        ParsedData(**_good_kwargs(measured_at=None))

    def test_with_issues(self):
        issue = ValidationIssue(
            field="wavelength",
            severity=Severity.WARNING,
            message="missing",
            detected_at=utc_now(),
        )
        pd = ParsedData(**_good_kwargs(issues=(issue,)))
        assert len(pd.issues) == 1


# ─── Immutability ───────────────────────────────────────────────────
class TestImmutability:
    def test_frozen(self):
        pd = ParsedData(**_good_kwargs())
        with pytest.raises(FrozenInstanceError):
            pd.parser_name = "different"  # type: ignore[misc]

    def test_no_slot_dict(self):
        # `slots=True` means there's no per-instance __dict__ — saves memory
        # and prevents typo-attributes from silently sticking. We verify this
        # via __slots__ presence rather than by attempting a write, because
        # Python's frozen+slots combination uses a super() call in the
        # generated __setattr__ that is brittle across stdlib versions when
        # tested directly.
        pd = ParsedData(**_good_kwargs())
        assert hasattr(type(pd), "__slots__")
        assert not hasattr(pd, "__dict__")


# ─── technique validation ───────────────────────────────────────────
class TestTechniqueValidation:
    def test_string_technique_rejected(self):
        with pytest.raises(ValidationError, match="technique"):
            ParsedData(**_good_kwargs(technique="XRD"))

    def test_none_technique_rejected(self):
        with pytest.raises(ValidationError, match="technique"):
            ParsedData(**_good_kwargs(technique=None))


# ─── arrays validation ──────────────────────────────────────────────
class TestArraysValidation:
    def test_non_dict_rejected(self):
        with pytest.raises(ValidationError, match="arrays"):
            ParsedData(**_good_kwargs(arrays=[1, 2, 3]))

    def test_non_string_key_rejected(self):
        with pytest.raises(ValidationError, match="arrays keys"):
            ParsedData(**_good_kwargs(arrays={42: np.array([1.0])}))

    def test_empty_string_key_rejected(self):
        with pytest.raises(ValidationError, match="arrays keys"):
            ParsedData(**_good_kwargs(arrays={"": np.array([1.0])}))

    def test_list_value_rejected(self):
        with pytest.raises(ValidationError, match=r"np\.ndarray"):
            ParsedData(**_good_kwargs(arrays={"x": [1.0, 2.0, 3.0]}))

    def test_2d_array_rejected(self):
        # Stage 1C is 1-D only. 2-D image content is deferred to Stage 5.
        with pytest.raises(ValidationError, match="1-D"):
            ParsedData(**_good_kwargs(arrays={"image": np.zeros((10, 10))}))

    def test_3d_array_rejected(self):
        with pytest.raises(ValidationError, match="1-D"):
            ParsedData(**_good_kwargs(arrays={"cube": np.zeros((2, 2, 2))}))

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValidationError, match="same length"):
            ParsedData(
                **_good_kwargs(
                    arrays={
                        "two_theta": np.array([10.0, 20.0, 30.0]),
                        "intensity": np.array([100.0, 250.0]),  # length 2 vs. 3
                    },
                ),
            )

    def test_three_arrays_same_length_allowed(self):
        # Thermoelectric: temperature + seebeck + conductivity, all co-indexed.
        ParsedData(
            **_good_kwargs(
                arrays={
                    "temperature": np.array([300.0, 400.0, 500.0]),
                    "seebeck": np.array([-50.0, -60.0, -70.0]),
                    "conductivity": np.array([1e3, 1.2e3, 1.5e3]),
                },
            ),
        )


# ─── metadata validation ────────────────────────────────────────────
class TestMetadataValidation:
    def test_non_dict_rejected(self):
        with pytest.raises(ValidationError, match="metadata"):
            ParsedData(**_good_kwargs(metadata=["a", "b"]))

    def test_set_value_rejected(self):
        with pytest.raises(ValidationError, match="JSON-safe"):
            ParsedData(**_good_kwargs(metadata={"x": {1, 2, 3}}))

    def test_numpy_scalar_rejected(self):
        # Parsers must convert np.int64 → int before storing in metadata.
        with pytest.raises(ValidationError, match="JSON-safe"):
            ParsedData(**_good_kwargs(metadata={"x": np.int64(42)}))

    def test_bytes_value_rejected(self):
        with pytest.raises(ValidationError, match="JSON-safe"):
            ParsedData(**_good_kwargs(metadata={"raw": b"\x00\x01"}))

    def test_non_string_key_rejected(self):
        with pytest.raises(ValidationError, match="JSON-safe"):
            ParsedData(**_good_kwargs(metadata={42: "value"}))

    def test_nested_dict_ok(self):
        ParsedData(**_good_kwargs(metadata={"a": {"b": {"c": 1}}}))

    def test_nested_list_ok(self):
        ParsedData(**_good_kwargs(metadata={"peaks": [[1, 2], [3, 4]]}))

    def test_empty_metadata_ok(self):
        ParsedData(**_good_kwargs(metadata={}))


# ─── instrument validation ──────────────────────────────────────────
class TestInstrumentValidation:
    def test_non_string_rejected(self):
        with pytest.raises(ValidationError, match="instrument"):
            ParsedData(**_good_kwargs(instrument=123))


# ─── measured_at validation ─────────────────────────────────────────
class TestMeasuredAtValidation:
    def test_naive_datetime_rejected(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            ParsedData(**_good_kwargs(measured_at=datetime(2024, 1, 1, 12, 0, 0)))

    def test_non_datetime_rejected(self):
        with pytest.raises(ValidationError, match="datetime"):
            ParsedData(**_good_kwargs(measured_at="2024-01-01"))


# ─── issues validation ──────────────────────────────────────────────
class TestIssuesValidation:
    def test_list_rejected(self):
        # Tuples enforce immutability; lists are caught.
        with pytest.raises(ValidationError, match="tuple"):
            ParsedData(**_good_kwargs(issues=[]))

    def test_non_validation_issue_rejected(self):
        with pytest.raises(ValidationError, match="ValidationIssue"):
            ParsedData(**_good_kwargs(issues=("not an issue",)))


# ─── parser_name validation ─────────────────────────────────────────
class TestParserNameValidation:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "",
            "Rigaku-XRD-Txt",  # uppercase
            "rigaku_xrd_txt",  # snake_case
            "rigaku xrd txt",  # spaces
            "-rigaku",  # leading dash
            "rigaku-",  # trailing dash
            "1rigaku",  # leading digit
            123,  # not a string
        ],
    )
    def test_invalid_names_rejected(self, bad_name: Any):
        with pytest.raises(ValidationError, match="parser_name"):
            ParsedData(**_good_kwargs(parser_name=bad_name))


# ─── parser_version validation ──────────────────────────────────────
class TestParserVersionValidation:
    @pytest.mark.parametrize(
        "bad_version",
        [
            "",
            "1.0",  # missing patch
            "v1.0.0",  # leading "v"
            "1.0.0-rc1",  # pre-release suffix
            "1.0.0+build",  # build metadata
            "1.0.0.0",  # too many components
            123,  # not a string
        ],
    )
    def test_invalid_versions_rejected(self, bad_version: Any):
        with pytest.raises(ValidationError, match="parser_version"):
            ParsedData(**_good_kwargs(parser_version=bad_version))

    @pytest.mark.parametrize("good_version", ["0.0.0", "1.0.0", "12.34.56"])
    def test_valid_versions_accepted(self, good_version: str):
        ParsedData(**_good_kwargs(parser_version=good_version))


# ─── Properties ─────────────────────────────────────────────────────
class TestProperties:
    def _issue(self, severity: Severity) -> ValidationIssue:
        return ValidationIssue(
            field="x",
            severity=severity,
            message="msg",
            detected_at=utc_now(),
        )

    def test_has_errors_true_on_error(self):
        pd = ParsedData(**_good_kwargs(issues=(self._issue(Severity.ERROR),)))
        assert pd.has_errors
        assert not pd.has_warnings

    def test_has_warnings_true_on_warning_only(self):
        pd = ParsedData(**_good_kwargs(issues=(self._issue(Severity.WARNING),)))
        assert pd.has_warnings
        assert not pd.has_errors

    def test_no_issues_means_clean(self):
        pd = ParsedData(**_good_kwargs(issues=()))
        assert not pd.has_errors
        assert not pd.has_warnings

    def test_array_names_sorted(self):
        pd = ParsedData(
            **_good_kwargs(
                arrays={
                    "z": np.array([1.0]),
                    "a": np.array([2.0]),
                    "m": np.array([3.0]),
                },
            ),
        )
        assert pd.array_names() == ("a", "m", "z")
