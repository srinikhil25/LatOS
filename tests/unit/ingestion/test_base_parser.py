"""Tests for `latos.ingestion.base_parser.BaseParser`."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from latos.core.enums import Technique
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData


def _make_concrete_parser_class(
    *,
    name: str = "demo-parser",
    version: str = "1.0.0",
    technique: Technique = Technique.XRD,
    extensions: tuple[str, ...] = (".txt",),
) -> type[BaseParser]:
    """Build a fully-concrete BaseParser subclass for testing.

    Defined inside a function so each test that calls it gets a fresh class —
    avoids namespace clashes when several tests register a class with the
    same `name`.

    Note: we rebind the kwargs to underscored locals before defining the
    inner class. Class bodies don't see enclosing function scope when the
    class attribute name shadows the parameter — `name = name` raises
    `NameError` because the LHS creates a class-scope binding that masks
    the parameter on the RHS.
    """
    _name = name
    _version = version
    _technique = technique
    _extensions = extensions

    class _Demo(BaseParser):
        # ClassVar annotations satisfy mypy strict mode.
        name: ClassVar[str] = _name
        version: ClassVar[str] = _version
        technique: ClassVar[Technique] = _technique
        supported_extensions: ClassVar[tuple[str, ...]] = _extensions

        def can_parse(self, path: Path) -> float:
            return 1.0 if self._extension_matches(path) else 0.0

        def parse(self, path: Path) -> ParsedData:
            raise AssertionError("parse() should not be called in this test")

    return _Demo


# ─── Abstract-ness ──────────────────────────────────────────────────
class TestAbstract:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseParser()  # type: ignore[abstract]

    def test_subclass_missing_can_parse_is_still_abstract(self):
        # An intermediate subclass that hasn't implemented `can_parse` must
        # remain uninstantiable — and __init_subclass__ must NOT fire metadata
        # validation on intermediates (otherwise abstract-base utilities can't
        # exist).
        class Intermediate(BaseParser):
            # Note: no name/version set, no can_parse/parse implemented.
            pass

        with pytest.raises(TypeError, match="abstract"):
            Intermediate()  # type: ignore[abstract]


# ─── Valid concrete subclass ────────────────────────────────────────
class TestValidConcreteSubclass:
    def test_instantiates_cleanly(self):
        cls = _make_concrete_parser_class()
        parser = cls()
        assert parser.name == "demo-parser"
        assert parser.version == "1.0.0"
        assert parser.technique is Technique.XRD
        assert parser.supported_extensions == (".txt",)

    def test_can_parse_returns_float(self):
        cls = _make_concrete_parser_class()
        parser = cls()
        assert parser.can_parse(Path("file.txt")) == 1.0
        assert parser.can_parse(Path("file.bin")) == 0.0


# ─── name validation ────────────────────────────────────────────────
class TestNameValidation:
    @pytest.mark.parametrize(
        "bad_name",
        ["", "Demo", "demo_parser", "demo parser", "1demo", "demo-"],
    )
    def test_invalid_name_raises_at_class_definition(self, bad_name: str):
        with pytest.raises(TypeError, match="name"):
            _make_concrete_parser_class(name=bad_name)


# ─── version validation ─────────────────────────────────────────────
class TestVersionValidation:
    @pytest.mark.parametrize(
        "bad_version",
        ["", "1.0", "v1.0.0", "1.0.0-rc1", "1.0.0.0"],
    )
    def test_invalid_version_raises_at_class_definition(self, bad_version: str):
        with pytest.raises(TypeError, match="version"):
            _make_concrete_parser_class(version=bad_version)


# ─── technique validation ───────────────────────────────────────────
class TestTechniqueValidation:
    def test_unset_technique_raises(self):
        with pytest.raises(TypeError, match="technique"):

            class _NoTechnique(BaseParser):
                name: ClassVar[str] = "no-technique"
                version: ClassVar[str] = "1.0.0"
                # technique deliberately not set
                supported_extensions: ClassVar[tuple[str, ...]] = (".txt",)

                def can_parse(self, path: Path) -> float:
                    return 0.0

                def parse(self, path: Path) -> ParsedData:
                    raise NotImplementedError


# ─── supported_extensions validation ────────────────────────────────
class TestExtensionsValidation:
    def test_list_instead_of_tuple_rejected(self):
        with pytest.raises(TypeError, match="tuple"):
            _make_concrete_parser_class(extensions=[".txt"])  # type: ignore[arg-type]

    def test_extension_without_dot_rejected(self):
        with pytest.raises(TypeError, match="dot"):
            _make_concrete_parser_class(extensions=("txt",))

    def test_uppercase_extension_rejected(self):
        with pytest.raises(TypeError, match="lowercase"):
            _make_concrete_parser_class(extensions=(".TXT",))

    def test_mixed_case_extension_rejected(self):
        with pytest.raises(TypeError, match="lowercase"):
            _make_concrete_parser_class(extensions=(".Xy",))

    def test_empty_tuple_allowed(self):
        # Empty means "match anything by content" — used by binary sniffers.
        cls = _make_concrete_parser_class(extensions=())
        parser = cls()
        assert parser._extension_matches(Path("anything.foo"))


# ─── _extension_matches behavior ────────────────────────────────────
class TestExtensionMatches:
    def test_case_insensitive_match(self):
        cls = _make_concrete_parser_class(extensions=(".txt",))
        parser = cls()
        assert parser._extension_matches(Path("file.txt"))
        assert parser._extension_matches(Path("file.TXT"))
        assert parser._extension_matches(Path("file.Txt"))

    def test_no_match_returns_false(self):
        cls = _make_concrete_parser_class(extensions=(".txt",))
        parser = cls()
        assert not parser._extension_matches(Path("file.csv"))

    def test_no_extension_no_match(self):
        cls = _make_concrete_parser_class(extensions=(".txt",))
        parser = cls()
        assert not parser._extension_matches(Path("README"))

    def test_multi_extension_class(self):
        cls = _make_concrete_parser_class(extensions=(".txt", ".asc"))
        parser = cls()
        assert parser._extension_matches(Path("a.txt"))
        assert parser._extension_matches(Path("a.asc"))
        assert not parser._extension_matches(Path("a.xy"))


# ─── repr ───────────────────────────────────────────────────────────
class TestRepr:
    def test_repr_includes_name_and_version(self):
        cls = _make_concrete_parser_class(name="my-parser", version="2.3.4")
        r = repr(cls())
        assert "my-parser" in r
        assert "2.3.4" in r
