"""One canonical JSON form, shared by every digest this package computes.

A digest is only worth recording if someone else can recompute it, so the bytes
that are hashed have to be fixed by rule rather than by whatever `json.dumps`
happens to default to. Keys are sorted, separators carry no whitespace, text is
UTF-8 and not ASCII-escaped, and NaN or infinity are refused: neither is JSON,
and a digest over a value that cannot round-trip is not checkable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["canonical_json", "sha256_hex"]


def canonical_json(value: Any) -> str:
    """Serialise `value` to the canonical JSON text every digest is taken over.

    Raises:
        TypeError: If `value` holds something JSON cannot represent.
        ValueError: If `value` holds NaN or infinity.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    """SHA-256 of `text` encoded as UTF-8, as a lowercase hex string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
