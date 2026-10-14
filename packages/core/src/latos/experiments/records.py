"""The flat table a record producer hands to the generic layer.

A producer turns one pipeline's data into rows, and the experiment definition
names the columns it uses. Nothing here guesses.

A **specimen** is the experimental unit represented by one `RecordTable` row:
one independently prepared object, measured as the producer's contract says.
Repeated measurements of one specimen are reduced by the producer and never
become independent specimen rows. That reduction is technique-specific, so it
lives in exactly one place per pipeline, and this layer never learns what the
repeats were.

The producer contract:

1. One row per specimen, keyed by the column the definition declares as
   `sample_key`.
2. `columns` lists every column the producer can emit, whether or not a given
   row has a value in it.
3. A missing value is None. A value the producer could not compute is None
   **plus** an ERROR-severity issue on the row saying why; those issues become
   the exclusion reasons.
4. `provenance[column]` says where that value came from.
5. `producer_detail` is the producer's own record of how the row was obtained.
   It is opaque: it is carried to the frozen record and digested there, and no
   generic code ever reads inside it.

`RecordTable.producer` names the producer's **contract**, not its software
version: a name plus a revision, such as ``example-reader/1``. The revision
changes exactly when the scientific transformation from raw measurement to
reported value changes. It is part of an experiment's identity, so a table made
under one contract is refused by a definition declared against another.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from latos.core.models import ValidationIssue
from latos.experiments._canonical import canonical_json, sha256_hex

__all__ = [
    "PRODUCER_CONTRACT_RE",
    "Record",
    "RecordTable",
    "Scalar",
    "producer_detail_sha256",
]

# What a cell of the table may hold.
Scalar = float | int | str | None

# `<name>/<revision>`: lower-case name, positive integer revision.
PRODUCER_CONTRACT_RE = re.compile(r"^[a-z][a-z0-9-]*/[1-9][0-9]*$")


def producer_detail_sha256(detail: Mapping[str, Any] | None) -> str | None:
    """Deterministic digest of an opaque `producer_detail`, or None when absent.

    The digest is taken over the canonical JSON form, so key order does not
    matter and the same detail always gives the same digest.
    """
    if detail is None:
        return None
    return sha256_hex(canonical_json(detail))


@dataclass(frozen=True, slots=True)
class Record:
    """One specimen, as its producer reported it.

    Attributes:
        record_id: The producer's own handle for the row, used in messages.
        values: Column name to value. A column the row has nothing for may be
            absent or None.
        provenance: Column name to a statement of where that value came from.
        issues: Problems the producer found with this row.
        producer_detail: Opaque, JSON-safe detail about how the row was
            obtained. Never interpreted here.
    """

    record_id: str
    values: Mapping[str, Scalar]
    provenance: Mapping[str, str] = field(default_factory=dict)
    issues: tuple[ValidationIssue, ...] = ()
    producer_detail: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("Record.record_id cannot be empty")
        if self.producer_detail is None:
            return
        # Checked here rather than at freeze time: a detail that cannot be
        # serialised is a producer bug, and it should fail where it was made.
        try:
            canonical_json(self.producer_detail)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"producer_detail of record {self.record_id!r} is not JSON-safe: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class RecordTable:
    """Every specimen one producer reported, one row each.

    Attributes:
        producer: The producer's contract identifier, `<name>/<revision>`.
        columns: Every column the producer can emit.
        rows: One `Record` per specimen.
    """

    producer: str
    columns: tuple[str, ...]
    rows: tuple[Record, ...]

    def __post_init__(self) -> None:
        if not PRODUCER_CONTRACT_RE.match(self.producer):
            raise ValueError(
                f"RecordTable.producer {self.producer!r} is not a contract identifier; "
                "expected '<name>/<revision>', for example 'example-reader/1'"
            )
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("RecordTable.columns lists a column more than once")
        known = set(self.columns)
        for row in self.rows:
            undeclared = sorted(set(row.values) - known)
            if undeclared:
                raise ValueError(
                    f"record {row.record_id!r} carries columns the table does not "
                    f"declare: {', '.join(undeclared)}"
                )
