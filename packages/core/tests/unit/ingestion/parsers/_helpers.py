"""Shared test helpers for parser tests.

Each parser test module shares the same need: serialize a `ParsedData`
into a deterministic JSON-safe dict for snapshot comparison, and locate
the fixtures/snapshots directories. Centralizing here avoids drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from latos.ingestion.parsed_data import ParsedData

# Project layout:
#   tests/fixtures/parsers/<technique>/<file>
#   tests/unit/ingestion/parsers/snapshots/<technique>_<id>.json
TESTS_DIR = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = TESTS_DIR / "fixtures" / "parsers"
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def parsed_to_snapshot(parsed: ParsedData) -> dict[str, Any]:
    """Serialize a `ParsedData` into a deterministic JSON-safe dict.

    Arrays are summarized as (length, dtype, sha256-of-bytes, head, tail) —
    storing thousands of floats in the snapshot would be noisy, but the
    SHA-256 catches byte-level drift exactly. Head/tail give a
    human-readable sanity check when reviewing diffs.
    """
    return {
        "technique": parsed.technique.value,
        "instrument": parsed.instrument,
        "measured_at": parsed.measured_at.isoformat() if parsed.measured_at else None,
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "metadata": parsed.metadata,
        "arrays": {
            name: {
                "length": int(arr.shape[0]),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "head": [float(x) for x in arr[:5]],
                "tail": [float(x) for x in arr[-5:]],
            }
            for name, arr in sorted(parsed.arrays.items())
        },
        "issues": [
            {
                "field": i.field,
                "severity": i.severity.value,
                "message": i.message,
                "acknowledged": i.acknowledged,
            }
            for i in parsed.issues
        ],
    }
