"""Tests for `latos.core.enums`."""

from __future__ import annotations

import pytest

from latos.core.enums import FileRole, Severity, Technique


class TestTechnique:
    def test_values_are_lowercase_strings(self) -> None:
        for t in Technique:
            assert t.value.islower()
            assert " " not in t.value

    def test_round_trip_via_value(self) -> None:
        for t in Technique:
            assert Technique(t.value) is t

    def test_display_name_present_for_all(self) -> None:
        for t in Technique:
            assert t.display_name
            assert isinstance(t.display_name, str)

    def test_display_name_is_human_readable(self) -> None:
        assert Technique.XRD.display_name == "X-Ray Diffraction"
        assert Technique.UV_DRS.display_name == "UV-Vis Diffuse Reflectance"

    def test_unknown_technique_exists(self) -> None:
        assert Technique.UNKNOWN.value == "unknown"


class TestFileRole:
    def test_all_roles_defined(self) -> None:
        assert FileRole.RAW.value == "raw"
        assert FileRole.PROCESSED.value == "processed"
        assert FileRole.DERIVED.value == "derived"
        assert FileRole.METADATA.value == "metadata"


class TestSeverity:
    def test_order_strict_monotonic(self) -> None:
        assert Severity.INFO.order < Severity.WARNING.order < Severity.ERROR.order

    @pytest.mark.parametrize(
        ("sev", "expected"),
        [
            (Severity.INFO, 0),
            (Severity.WARNING, 1),
            (Severity.ERROR, 2),
        ],
    )
    def test_order_values(self, sev: Severity, expected: int) -> None:
        assert sev.order == expected
