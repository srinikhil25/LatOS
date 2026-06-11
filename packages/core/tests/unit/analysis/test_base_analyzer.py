"""Tests for `latos.analysis.base_analyzer`.

Mirrors the structure of `tests/unit/ingestion/test_base_parser.py`:
- Class-attribute validation fires at import time of malformed subclasses.
- AnalyzerInputs / AnalyzerOutput are frozen and have working defaults.
- merge_params behaves as a shallow override.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, new_id, utc_now


def _measurement() -> Measurement:
    """Build a minimal Measurement for analyzer-input fixtures."""
    from pathlib import Path

    from latos.core.enums import FileRole
    from latos.core.models import FileRef

    return Measurement(
        id=new_id(),
        sample_id=new_id(),
        technique=Technique.UV_DRS,
        instrument="UV-DRS",
        measured_at=utc_now(),
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(
            FileRef(
                path=Path("/data/uv.xlsx"),
                sha256="a" * 64,
                size_bytes=100,
                role=FileRole.RAW,
                scanned_at=utc_now(),
            ),
        ),
    )


class _Concrete(BaseAnalyzer):
    """A minimal valid analyzer for testing the contract types."""

    name: ClassVar[str] = "test-analyzer"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput(outputs={"value": 1.0})


class TestAnalyzerInputs:
    def test_constructs_with_defaults(self) -> None:
        m = _measurement()
        inp = AnalyzerInputs(measurement=m)
        assert inp.arrays == {}
        assert inp.params == {}

    def test_frozen(self) -> None:
        inp = AnalyzerInputs(measurement=_measurement())
        with pytest.raises((AttributeError, TypeError)):
            inp.params = {"x": 1}  # type: ignore[misc]


class TestAnalyzerOutput:
    def test_constructs_with_defaults(self) -> None:
        out = AnalyzerOutput()
        assert out.outputs == {}
        assert out.derived_arrays == {}
        assert out.issues == ()

    def test_frozen(self) -> None:
        out = AnalyzerOutput()
        with pytest.raises((AttributeError, TypeError)):
            out.outputs = {"x": 1}  # type: ignore[misc]


class TestSubclassValidation:
    def test_concrete_analyzer_is_valid(self) -> None:
        analyzer = _Concrete()
        assert analyzer.name == "test-analyzer"
        assert analyzer.version == "1.0.0"

    def test_repr(self) -> None:
        r = repr(_Concrete())
        assert "test-analyzer" in r
        assert "1.0.0" in r

    def test_name_must_be_kebab_case(self) -> None:
        with pytest.raises(TypeError, match="name must be lowercase kebab-case"):

            class _Bad(BaseAnalyzer):
                name: ClassVar[str] = "BadName"
                version: ClassVar[str] = "1.0.0"
                accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

                def accepts(self, measurement: Measurement) -> bool:
                    return True

                def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                    return AnalyzerOutput()

    def test_version_must_be_semver(self) -> None:
        with pytest.raises(TypeError, match="version must be semver"):

            class _Bad(BaseAnalyzer):
                name: ClassVar[str] = "x-analyzer"
                version: ClassVar[str] = "v1"
                accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

                def accepts(self, measurement: Measurement) -> bool:
                    return True

                def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                    return AnalyzerOutput()

    def test_accepts_techniques_must_be_non_empty(self) -> None:
        with pytest.raises(TypeError, match="accepts_techniques must be a non-empty tuple"):

            class _Bad(BaseAnalyzer):
                name: ClassVar[str] = "x-analyzer"
                version: ClassVar[str] = "1.0.0"
                accepts_techniques: ClassVar[tuple[Technique, ...]] = ()

                def accepts(self, measurement: Measurement) -> bool:
                    return True

                def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                    return AnalyzerOutput()

    def test_accepts_techniques_entries_must_be_technique(self) -> None:
        with pytest.raises(TypeError, match="accepts_techniques entries must be Technique"):

            class _Bad(BaseAnalyzer):
                name: ClassVar[str] = "x-analyzer"
                version: ClassVar[str] = "1.0.0"
                accepts_techniques: ClassVar[tuple[Any, ...]] = ("not-a-technique",)

                def accepts(self, measurement: Measurement) -> bool:
                    return True

                def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                    return AnalyzerOutput()

    def test_default_params_must_be_dict(self) -> None:
        with pytest.raises(TypeError, match="default_params must be a dict"):

            class _Bad(BaseAnalyzer):
                name: ClassVar[str] = "x-analyzer"
                version: ClassVar[str] = "1.0.0"
                accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)
                default_params: ClassVar[Any] = ["not", "a", "dict"]

                def accepts(self, measurement: Measurement) -> bool:
                    return True

                def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                    return AnalyzerOutput()


class TestMergeParams:
    def test_merge_with_none_returns_defaults(self) -> None:
        class _Defaults(BaseAnalyzer):
            name: ClassVar[str] = "defaults-analyzer"
            version: ClassVar[str] = "1.0.0"
            accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)
            default_params: ClassVar[dict[str, Any]] = {"a": 1, "b": 2}

            def accepts(self, measurement: Measurement) -> bool:
                return True

            def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                return AnalyzerOutput()

        merged = _Defaults().merge_params(None)
        assert merged == {"a": 1, "b": 2}

    def test_overrides_replace_keys(self) -> None:
        class _Defaults2(BaseAnalyzer):
            name: ClassVar[str] = "defaults2-analyzer"
            version: ClassVar[str] = "1.0.0"
            accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)
            default_params: ClassVar[dict[str, Any]] = {"a": 1, "b": 2}

            def accepts(self, measurement: Measurement) -> bool:
                return True

            def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                return AnalyzerOutput()

        merged = _Defaults2().merge_params({"a": 99, "c": "added"})
        assert merged == {"a": 99, "b": 2, "c": "added"}

    def test_merge_does_not_mutate_defaults(self) -> None:
        analyzer = _Concrete()
        original = dict(analyzer.default_params)
        analyzer.merge_params({"foo": "bar"})
        assert analyzer.default_params == original


class TestAnalyzeContract:
    """Spot-check that a concrete subclass can actually be called."""

    def test_analyzer_runs(self) -> None:
        analyzer = _Concrete()
        m = _measurement()
        inputs = AnalyzerInputs(
            measurement=m,
            arrays={"x": np.array([1.0, 2.0, 3.0])},
            params={},
        )
        out = analyzer.analyze(inputs)
        assert out.outputs == {"value": 1.0}

    def test_issues_pattern(self) -> None:
        """AnalyzerOutput with issues round-trips its tuple."""
        issue = ValidationIssue(
            field="x",
            severity=Severity.WARNING,
            message="x is suspicious",
            detected_at=utc_now(),
        )
        out = AnalyzerOutput(issues=(issue,))
        assert out.issues == (issue,)
