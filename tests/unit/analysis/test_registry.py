"""Tests for `latos.analysis.registry.AnalyzerRegistry`."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.analysis.registry import AnalyzerRegistry, default_registry
from latos.core.enums import FileRole, Technique
from latos.core.models import FileRef, Measurement, new_id, utc_now


def _measurement(*, technique: Technique = Technique.UV_DRS) -> Measurement:
    return Measurement(
        id=new_id(),
        sample_id=new_id(),
        technique=technique,
        instrument="x",
        measured_at=utc_now(),
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(
            FileRef(
                path=Path("/data/x.xlsx"),
                sha256="a" * 64,
                size_bytes=1,
                role=FileRole.RAW,
                scanned_at=utc_now(),
            ),
        ),
    )


class _UvDrsAnalyzer(BaseAnalyzer):
    name: ClassVar[str] = "uv-test"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput()


class _XrdAnalyzer(BaseAnalyzer):
    name: ClassVar[str] = "xrd-test"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.XRD,)

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput()


class _PickyAnalyzer(BaseAnalyzer):
    """Accepts only measurements with at least 2 files."""

    name: ClassVar[str] = "picky-test"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

    def accepts(self, measurement: Measurement) -> bool:
        return len(measurement.files) >= 2

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput()


class _RaisingAnalyzer(BaseAnalyzer):
    """Buggy analyzer whose accepts() raises — registry must not crash."""

    name: ClassVar[str] = "raising-test"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

    def accepts(self, measurement: Measurement) -> bool:
        raise RuntimeError("boom")

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput()


# ─── Construction ───────────────────────────────────────────────────
class TestConstruction:
    def test_empty_registry(self) -> None:
        reg = AnalyzerRegistry()
        assert len(reg) == 0
        assert reg.analyzers == ()

    def test_prepopulated(self) -> None:
        reg = AnalyzerRegistry([_UvDrsAnalyzer(), _XrdAnalyzer()])
        assert len(reg) == 2
        assert "uv-test" in reg
        assert "xrd-test" in reg

    def test_membership_only_works_for_strings(self) -> None:
        reg = AnalyzerRegistry([_UvDrsAnalyzer()])
        assert 42 not in reg  # type: ignore[operator]


# ─── Registration ───────────────────────────────────────────────────
class TestRegister:
    def test_register_adds_analyzer(self) -> None:
        reg = AnalyzerRegistry()
        reg.register(_UvDrsAnalyzer())
        assert len(reg) == 1

    def test_duplicate_name_rejected(self) -> None:
        reg = AnalyzerRegistry([_UvDrsAnalyzer()])
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_UvDrsAnalyzer())

    def test_get_by_name(self) -> None:
        reg = AnalyzerRegistry([_UvDrsAnalyzer(), _XrdAnalyzer()])
        analyzer = reg.get("uv-test")
        assert analyzer is not None
        assert analyzer.name == "uv-test"

    def test_get_missing_returns_none(self) -> None:
        reg = AnalyzerRegistry([_UvDrsAnalyzer()])
        assert reg.get("nope") is None


# ─── Dispatch ───────────────────────────────────────────────────────
class TestFindFor:
    def test_returns_matching_technique(self) -> None:
        reg = AnalyzerRegistry([_UvDrsAnalyzer(), _XrdAnalyzer()])
        m = _measurement(technique=Technique.UV_DRS)
        matches = reg.find_for(m)
        assert len(matches) == 1
        assert matches[0].name == "uv-test"

    def test_skips_non_matching_technique(self) -> None:
        reg = AnalyzerRegistry([_XrdAnalyzer()])
        m = _measurement(technique=Technique.UV_DRS)
        assert reg.find_for(m) == ()

    def test_multiple_matches_preserve_order(self) -> None:
        class _Other(BaseAnalyzer):
            name: ClassVar[str] = "uv-test-2"
            version: ClassVar[str] = "1.0.0"
            accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

            def accepts(self, measurement: Measurement) -> bool:
                return True

            def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                return AnalyzerOutput()

        reg = AnalyzerRegistry([_UvDrsAnalyzer(), _Other()])
        names = [a.name for a in reg.find_for(_measurement())]
        assert names == ["uv-test", "uv-test-2"]

    def test_accepts_can_reject(self) -> None:
        """`accepts()` returning False filters the analyzer out."""
        reg = AnalyzerRegistry([_PickyAnalyzer()])
        m = _measurement()  # only one file
        assert reg.find_for(m) == ()

    def test_raising_accepts_is_skipped(self) -> None:
        """A buggy analyzer that raises doesn't take down dispatch."""
        reg = AnalyzerRegistry([_RaisingAnalyzer(), _UvDrsAnalyzer()])
        matches = reg.find_for(_measurement())
        assert [a.name for a in matches] == ["uv-test"]


# ─── default_registry ───────────────────────────────────────────────
def test_default_registry_includes_tauc() -> None:
    """Smoke test: default registry contains the Tauc analyzer."""
    reg = default_registry()
    assert "uvdrs-tauc" in reg
    analyzer = reg.get("uvdrs-tauc")
    assert analyzer is not None
    assert Technique.UV_DRS in analyzer.accepts_techniques
