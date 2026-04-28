"""Ingestion layer — file hashing, parsers, crawler, orchestrator.

This package owns the path from "raw file on disk" to "Measurement in DB":
- `hashing` — SHA-256 file hashing with mtime/size cache (Stage 1C.1)
- `parsed_data` — uniform output shape `ParsedData` (Stage 1C.1)
- `base_parser` — `BaseParser` ABC every parser implements (Stage 1C.1)
- `array_store` — Parquet I/O for measurement arrays (Stage 1C.2)
- `parsers/` — concrete parsers per technique x format (Stage 1C.3+)
- `registry` — parser dispatcher with confidence-pick (Stage 1C.5)
- `crawler` — folder walker that classifies files (Stage 1D)
- `orchestrator` — turns a folder into a Project (Stage 1D)
"""

from __future__ import annotations

from latos.ingestion.array_store import ArrayStore
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.hashing import (
    FileFingerprint,
    HashCache,
    fingerprint,
    hash_bytes,
    hash_file,
)
from latos.ingestion.parsed_data import ParsedData

__all__ = [
    # Parser API
    "BaseParser",
    "ParsedData",
    # Storage
    "ArrayStore",
    # Hashing
    "FileFingerprint",
    "HashCache",
    "fingerprint",
    "hash_bytes",
    "hash_file",
]
