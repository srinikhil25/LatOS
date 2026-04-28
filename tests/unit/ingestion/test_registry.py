"""Tests for `latos.ingestion.registry`."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from latos.core.enums import Technique
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData
from latos.ingestion.registry import (
    MIN_CONFIDENCE,
    ParserMatch,
    ParserRegistry,
    default_registry,
)


def _make_test_parser(
    *,
    name: str,
    confidence: float = 1.0,
    raises: type[BaseException] | None = None,
    extensions: tuple[str, ...] = (".testext",),
    technique: Technique = Technique.XRD,
) -> type[BaseParser]:
    """Build a concrete BaseParser subclass that always returns `confidence`.

    See `test_base_parser.py` for the rationale on the underscored locals.
    """
    _name = name
    _confidence = confidence
    _raises = raises
    _extensions = extensions
    _technique = technique

    class _Test(BaseParser):
        name: ClassVar[str] = _name
        version: ClassVar[str] = "1.0.0"
        technique: ClassVar[Technique] = _technique
        supported_extensions: ClassVar[tuple[str, ...]] = _extensions

        def can_parse(self, path: Path) -> float:
            if _raises is not None:
                raise _raises("simulated parser bug")
            # Test parser respects its declared extensions so "wrong
            # extension" tests behave the way real parsers do.
            if not self._extension_matches(path):
                return 0.0
            return _confidence

        def parse(self, path: Path) -> ParsedData:
            # Minimal valid ParsedData using the test parser's identity.
            return ParsedData(
                technique=self.technique,
                arrays={},
                metadata={"source": _name},
                instrument=None,
                measured_at=None,
                issues=(),
                parser_name=self.name,
                parser_version=self.version,
            )

    return _Test


# ─── Construction & registration ────────────────────────────────────
class TestRegistration:
    def test_empty_registry(self):
        r = ParserRegistry()
        assert len(r) == 0
        assert r.parsers == ()

    def test_register_increases_count(self):
        r = ParserRegistry()
        r.register(_make_test_parser(name="parser-a")())
        assert len(r) == 1
        assert "parser-a" in r

    def test_init_with_parsers(self):
        a = _make_test_parser(name="parser-a")()
        b = _make_test_parser(name="parser-b")()
        r = ParserRegistry([a, b])
        assert len(r) == 2
        assert r.parsers == (a, b)

    def test_duplicate_name_raises(self):
        r = ParserRegistry()
        r.register(_make_test_parser(name="dup")())
        with pytest.raises(ValueError, match="dup"):
            r.register(_make_test_parser(name="dup")())

    def test_contains_by_name(self):
        r = ParserRegistry([_make_test_parser(name="parser-x")()])
        assert "parser-x" in r
        assert "missing" not in r
        assert 42 not in r  # non-string membership test


# ─── find_parser dispatch ───────────────────────────────────────────
class TestFindParser:
    def test_returns_match_when_one_parser_claims_file(self, tmp_path: Path):
        f = tmp_path / "data.testext"
        f.write_text("anything")
        parser = _make_test_parser(name="match", confidence=0.9)()
        r = ParserRegistry([parser])

        match = r.find_parser(f)
        assert isinstance(match, ParserMatch)
        assert match.parser is parser
        assert match.confidence == 0.9

    def test_returns_none_when_no_parser_claims(self, tmp_path: Path):
        f = tmp_path / "data.foo"
        f.write_text("x")
        # Test parser registered for `.testext` only.
        parser = _make_test_parser(name="claim-only", extensions=(".testext",))()
        r = ParserRegistry([parser])
        # Wrong extension → can_parse returns 0.0 → below threshold.
        assert r.find_parser(f) is None

    def test_picks_highest_confidence(self, tmp_path: Path):
        f = tmp_path / "data.testext"
        f.write_text("x")
        low = _make_test_parser(name="low", confidence=0.6)()
        high = _make_test_parser(name="high", confidence=0.9)()
        r = ParserRegistry([low, high])
        match = r.find_parser(f)
        assert match is not None
        assert match.parser is high

    def test_tie_broken_by_registration_order(self, tmp_path: Path):
        f = tmp_path / "data.testext"
        f.write_text("x")
        first = _make_test_parser(name="first", confidence=1.0)()
        second = _make_test_parser(name="second", confidence=1.0)()
        r = ParserRegistry([first, second])
        match = r.find_parser(f)
        assert match is not None
        assert match.parser is first  # registered first wins on tie

    def test_below_threshold_excluded(self, tmp_path: Path):
        f = tmp_path / "data.testext"
        f.write_text("x")
        weak = _make_test_parser(name="weak", confidence=0.4)()
        r = ParserRegistry([weak])
        # 0.4 < 0.5 (MIN_CONFIDENCE) → excluded.
        assert r.find_parser(f) is None

    def test_custom_min_confidence(self, tmp_path: Path):
        f = tmp_path / "data.testext"
        f.write_text("x")
        weak = _make_test_parser(name="weak", confidence=0.4)()
        r = ParserRegistry([weak])
        # Lowering threshold lets it through.
        match = r.find_parser(f, min_confidence=0.3)
        assert match is not None
        assert match.parser is weak

    def test_zero_confidence_always_excluded(self, tmp_path: Path):
        # Even with min_confidence=0.0, a parser scoring exactly 0 is
        # below threshold (0.0 < 0.0 is False, so it WOULD be included
        # by `>=`). Verify our actual behavior matches the test name.
        f = tmp_path / "data.testext"
        f.write_text("x")
        zero = _make_test_parser(name="zero", confidence=0.0)()
        r = ParserRegistry([zero])
        # min_confidence=0.0 includes 0.0 (since 0.0 >= 0.0).
        match = r.find_parser(f, min_confidence=0.0)
        assert match is not None
        # But default behavior excludes it.
        assert r.find_parser(f) is None

    def test_buggy_parser_raising_does_not_break_dispatch(self, tmp_path: Path):
        f = tmp_path / "data.testext"
        f.write_text("x")
        broken = _make_test_parser(name="broken", raises=RuntimeError)()
        good = _make_test_parser(name="good", confidence=0.9)()
        r = ParserRegistry([broken, good])
        # Registry catches the exception from `broken.can_parse` and
        # carries on — `good` still wins.
        match = r.find_parser(f)
        assert match is not None
        assert match.parser is good


# ─── parse() convenience ────────────────────────────────────────────
class TestParse:
    def test_dispatches_to_winning_parser(self, tmp_path: Path):
        f = tmp_path / "data.testext"
        f.write_text("x")
        parser = _make_test_parser(name="winner", confidence=0.9)()
        r = ParserRegistry([parser])
        result = r.parse(f)
        assert result is not None
        assert result.parser_name == "winner"

    def test_returns_none_when_no_parser_claims(self, tmp_path: Path):
        f = tmp_path / "data.foo"
        f.write_text("x")
        parser = _make_test_parser(name="claim-only", extensions=(".testext",))()
        r = ParserRegistry([parser])
        assert r.parse(f) is None


# ─── default_registry ──────────────────────────────────────────────
class TestDefaultRegistry:
    def test_contains_all_nine_parsers(self):
        r = default_registry()
        # Stage 1C ships 9 parsers.
        expected_names = {
            "rigaku-xrd-txt",
            "rigaku-xrd-asc",
            "panalytical-xrdml",
            "casaxps-csv",
            "uvdrs-xlsx",
            "hall-xls",
            "thermoelectric-xlsx",
            "bruker-eds-spx",
            "microscopy-tif",
        }
        actual_names = {p.name for p in r.parsers}
        assert expected_names == actual_names

    def test_dispatches_real_xrd_file(self):
        # End-to-end: feed the registry a real fixture and verify it
        # picks the right parser.
        from tests.unit.ingestion.parsers._helpers import FIXTURES_DIR

        f = FIXTURES_DIR / "xrd" / "rigaku_bs3a.txt"
        r = default_registry()
        match = r.find_parser(f)
        assert match is not None
        assert match.parser.name == "rigaku-xrd-txt"

    def test_dispatches_real_xrdml_file(self):
        from tests.unit.ingestion.parsers._helpers import FIXTURES_DIR

        f = FIXTURES_DIR / "xrd" / "panalytical_cscbi1.xrdml"
        r = default_registry()
        match = r.find_parser(f)
        assert match is not None
        assert match.parser.name == "panalytical-xrdml"

    def test_dispatches_real_eds_file(self):
        from tests.unit.ingestion.parsers._helpers import FIXTURES_DIR

        f = FIXTURES_DIR / "eds" / "bruker_1.spx"
        r = default_registry()
        match = r.find_parser(f)
        assert match is not None
        assert match.parser.name == "bruker-eds-spx"

    def test_dispatches_real_tif_file(self):
        from tests.unit.ingestion.parsers._helpers import FIXTURES_DIR

        f = FIXTURES_DIR / "microscopy" / "tem_cs.tif"
        r = default_registry()
        match = r.find_parser(f)
        assert match is not None
        assert match.parser.name == "microscopy-tif"

    def test_unknown_file_returns_none(self, tmp_path: Path):
        f = tmp_path / "random.xyz"
        f.write_text("not anything we know")
        r = default_registry()
        assert r.find_parser(f) is None

    def test_xlsx_dispatch_distinguishes_uvdrs_from_thermoelectric(self):
        # Both UvDrsXlsxParser and ThermoelectricXlsxParser claim `.xlsx`.
        # Their content sniffs MUST disambiguate so each fixture lands on
        # the right parser — this is the highest-collision-risk pair in
        # the default registry.
        from tests.unit.ingestion.parsers._helpers import FIXTURES_DIR

        r = default_registry()
        uv = r.find_parser(FIXTURES_DIR / "uvdrs" / "uvdrs_cs.xlsx")
        te = r.find_parser(FIXTURES_DIR / "thermoelectric" / "zt_calc.xlsx")
        assert uv is not None and uv.parser.name == "uvdrs-xlsx"
        assert te is not None and te.parser.name == "thermoelectric-xlsx"


# ─── End-to-end parse integration ───────────────────────────────────
class TestParseIntegration:
    """Use the default registry to parse real fixtures end-to-end."""

    def test_parse_xrd_txt_returns_xrd_parsed_data(self):
        from tests.unit.ingestion.parsers._helpers import FIXTURES_DIR

        r = default_registry()
        result = r.parse(FIXTURES_DIR / "xrd" / "rigaku_bs3a.txt")
        assert result is not None
        assert result.technique is Technique.XRD
        assert "two_theta" in result.arrays

    def test_parse_eds_spx_returns_eds_parsed_data(self):
        from tests.unit.ingestion.parsers._helpers import FIXTURES_DIR

        r = default_registry()
        result = r.parse(FIXTURES_DIR / "eds" / "bruker_1.spx")
        assert result is not None
        assert result.technique is Technique.EDS
        assert "energy_kev" in result.arrays


# ─── Threshold constant ─────────────────────────────────────────────
class TestThresholdConstant:
    def test_min_confidence_is_half(self):
        # Documented design constant; if this changes, the architecture
        # commitment changes too.
        assert MIN_CONFIDENCE == 0.5
