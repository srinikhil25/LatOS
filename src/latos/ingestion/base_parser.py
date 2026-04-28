"""`BaseParser` — the contract every Latos parser must satisfy.

Every parser implements this ABC. The dispatcher (Stage 1C.5) iterates
all registered parsers, picks the one with the highest `can_parse()`
confidence above threshold, and calls `parse()`.

Design constraints enforced here:
- Parsers declare their `name`, `version`, `technique`, and supported
  extensions as **class attributes**, not properties — they're constants.
- `__init_subclass__` validates these attributes when the class is defined,
  so a malformed parser blows up at import time, not at parse time.
- `can_parse()` must be **cheap**: read header bytes only, no full parse.
  This is called for every (file x parser) candidate during dispatch.
- `parse()` must **never raise**. It returns `ParsedData` with any
  problems described in `issues`. The orchestrator decides what to do
  with errored measurements.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from latos.core.enums import Technique
from latos.ingestion.parsed_data import ParsedData

__all__ = ["BaseParser"]


# Same patterns enforced inside `ParsedData`. Duplicated here so the
# class-level error fires at import time of the parser module, not at
# first parse — this catches typos on day one.
_PARSER_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]*[a-z0-9]$")
_PARSER_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


class BaseParser(ABC):
    """Abstract contract for a Latos file parser.

    Concrete subclasses MUST set these class attributes:
        name: lowercase kebab-case identifier (e.g. "rigaku-xrd-txt").
        version: semver string MAJOR.MINOR.PATCH. Bump on any change to
            the structure or values of `ParsedData`.
        technique: which `Technique` enum value this parser produces.
        supported_extensions: tuple of lowercase extensions including
            the dot, e.g. (".txt", ".xrd"). Used for pre-filtering before
            calling `can_parse()`. Empty tuple means "match all extensions"
            (rare — typically only for binary sniffers).

    Concrete subclasses MUST implement:
        can_parse(path) -> float in [0, 1]
        parse(path) -> ParsedData

    Confidence convention:
        0.0  — definitely not this parser's format.
        0.5  — extension matches; structure unverified.
        0.8  — header magic bytes/keywords confirmed.
        1.0  — file is unambiguously this format (e.g. unique XML namespace).
    """

    # Class attributes — subclasses override.
    # Defaults are sentinel values that fail validation in __init_subclass__.
    name: ClassVar[str] = ""
    version: ClassVar[str] = ""
    technique: ClassVar[Technique]
    supported_extensions: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate parser metadata as soon as the subclass is defined.

        A typo in `name`, an unset `technique`, or an extension missing its
        leading dot raises here — before the parser is ever instantiated.
        """
        super().__init_subclass__(**kwargs)

        # Skip validation for intermediate abstract classes (i.e. classes
        # that still have unimplemented abstract methods). Only concrete
        # parsers must have all metadata set.
        #
        # Note: we cannot rely on `cls.__abstractmethods__` here because
        # ABCMeta sets that attribute AFTER `__init_subclass__` runs.
        # Instead, we check for the `__isabstractmethod__` marker on the
        # methods themselves, which `@abstractmethod` set when the base
        # class was defined.
        for method_name in ("can_parse", "parse"):
            method = getattr(cls, method_name, None)
            if method is None or getattr(method, "__isabstractmethod__", False):
                return

        if not isinstance(cls.name, str) or not _PARSER_NAME_RE.match(cls.name):
            raise TypeError(
                f"{cls.__name__}.name must be lowercase kebab-case "
                f"(e.g. 'rigaku-xrd-txt'), got {cls.name!r}",
            )
        if not isinstance(cls.version, str) or not _PARSER_VERSION_RE.match(cls.version):
            raise TypeError(
                f"{cls.__name__}.version must be semver MAJOR.MINOR.PATCH, got {cls.version!r}",
            )
        if not hasattr(cls, "technique") or not isinstance(cls.technique, Technique):
            raise TypeError(
                f"{cls.__name__}.technique must be set to a Technique enum value",
            )
        if not isinstance(cls.supported_extensions, tuple):
            raise TypeError(
                f"{cls.__name__}.supported_extensions must be a tuple, "
                f"got {type(cls.supported_extensions).__name__}",
            )
        for ext in cls.supported_extensions:
            if not isinstance(ext, str) or not ext.startswith(".") or ext != ext.lower():
                raise TypeError(
                    f"{cls.__name__}.supported_extensions entries must be lowercase "
                    f"and start with a dot (e.g. '.txt'), got {ext!r}",
                )

    # ─── Abstract API ────────────────────────────────────────────────
    @abstractmethod
    def can_parse(self, path: Path) -> float:
        """Return confidence in [0, 1] that this parser handles `path`.

        Must be cheap: read no more than the first few KB of the file.
        Implementations can call `self._extension_matches(path)` first
        and short-circuit to 0.0 on a miss.

        Must not raise on malformed/unreadable files — return 0.0 instead.
        """

    @abstractmethod
    def parse(self, path: Path) -> ParsedData:
        """Parse `path` and return a `ParsedData`.

        Must NOT raise on malformed input. Instead, return a `ParsedData`
        with `issues` describing what went wrong. Empty arrays are
        permissible when nothing could be salvaged.

        The returned `ParsedData.parser_name` MUST equal `self.name` and
        `parser_version` MUST equal `self.version`. The base class does
        not check this — the contract is verified by tests.
        """

    # ─── Helpers for subclasses ──────────────────────────────────────
    def _extension_matches(self, path: Path) -> bool:
        """True if `path`'s suffix is in `supported_extensions`.

        Comparison is case-insensitive: `.TXT` matches `.txt`. An empty
        `supported_extensions` tuple means "match anything" — useful for
        binary sniffers that identify by content alone.
        """
        if not self.supported_extensions:
            return True
        return path.suffix.lower() in self.supported_extensions

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} version={self.version!r}>"
