"""`ArrayStore` — Parquet-backed persistence for measurement arrays.

The metadata side of a measurement (technique, instrument, parser version,
issues, file refs) lives in SQLite. The numeric arrays — typically big enough
that round-tripping them through SQLite would be wasteful — live as Parquet
files under `<project_root>/.latos/arrays/<measurement_id>.parquet`.

Layout
------
- One file per measurement.
- One column per array. All arrays in a single `ParsedData` are co-indexed
  (enforced by `ParsedData._check_arrays`), so a flat columnar table is the
  natural fit. pandas, DuckDB, Power Query — anything that reads Parquet —
  can open these files without nested-type machinery.
- Empty `arrays` dict (e.g. metadata-only TIF parser) → no file written.
  `load()` for a non-existent measurement returns `{}`.

Atomicity
---------
Writes go via `<id>.parquet.tmp` then `os.replace()`, which is atomic at the
filesystem level on both POSIX (rename) and Windows (MoveFileEx with
REPLACE_EXISTING). A crash mid-write leaves the `.tmp` file behind but the
real file is either absent (first write) or the previous valid version. The
orphan `.tmp` is harmless and gets cleaned up on the next `ArrayStore`
construction. This protects researchers who Ctrl+C a long ingestion: we
never poison the parse cache with a half-written Parquet file.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latos.ingestion.parsed_data import ParsedData

__all__ = ["ArrayStore"]

# Suffix for in-flight writes. Chosen to be visually obvious and to avoid
# colliding with any extension a real measurement file might have.
_TEMP_SUFFIX = ".parquet.tmp"


class ArrayStore:
    """Read/write Parquet array files for a single project's measurements.

    Construct with the project's `<root>/.latos/arrays/` directory. The
    constructor creates the directory if missing and sweeps any orphan
    `.tmp` files left over from a previous crash.

    All public methods are keyed on `measurement_id` — the same 32-char
    hex UUID used in the SQLite `measurements.id` column. The store does
    not know about the SQL DB; the orchestrator (Stage 1D) coordinates
    SQL row writes with `ArrayStore.write()` calls.
    """

    def __init__(self, arrays_dir: Path) -> None:
        """Bind the store to `arrays_dir`, creating it and cleaning orphans.

        Args:
            arrays_dir: Directory to store Parquet files in. Created with
                `parents=True, exist_ok=True` if missing. Typically
                `<project_root>/.latos/arrays/`.
        """
        self._dir = Path(arrays_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_orphan_temps()

    @property
    def directory(self) -> Path:
        """The arrays directory this store is bound to."""
        return self._dir

    # ─── Path helpers ────────────────────────────────────────────────
    def _path_for(self, measurement_id: str) -> Path:
        """Final on-disk path for a measurement's arrays."""
        return self._dir / f"{measurement_id}.parquet"

    def _temp_path_for(self, measurement_id: str) -> Path:
        """In-flight path used during atomic write."""
        return self._dir / f"{measurement_id}{_TEMP_SUFFIX}"

    # ─── Public API ──────────────────────────────────────────────────
    def write(self, measurement_id: str, data: ParsedData) -> Path | None:
        """Persist `data.arrays` for `measurement_id`. Returns the final path.

        Atomicity: writes go to a `.tmp` file, then `os.replace()` swings
        it into place. A crash before the swap leaves the real file (if
        any) untouched; a crash after the swap leaves the new file.

        Args:
            measurement_id: 32-char hex ID; used as the filename stem.
            data: ParsedData. Only `data.arrays` is persisted here —
                metadata lives in SQLite via the mappers.

        Returns:
            The final path on disk, or `None` if `data.arrays` was empty
            (no file written; metadata-only measurements don't need one).
        """
        if not data.arrays:
            return None

        target = self._path_for(measurement_id)
        tmp = self._temp_path_for(measurement_id)
        table = self._build_table(data.arrays)

        # Write to .tmp first — never let a partial write hit `target`.
        # If write_table itself raises, the .tmp is removed so we don't
        # accumulate stale orphans on every failed attempt.
        try:
            # `pq.write_table` is untyped in pyarrow's stubs; the runtime
            # behavior is exactly what we want.
            pq.write_table(table, tmp)  # type: ignore[no-untyped-call]
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        # os.replace is atomic on POSIX (rename) and Windows
        # (MoveFileEx with REPLACE_EXISTING).
        os.replace(tmp, target)
        return target

    def load(self, measurement_id: str) -> dict[str, np.ndarray]:
        """Load `measurement_id`'s arrays back as a `{name: ndarray}` dict.

        Returns an empty dict if no file exists for the measurement
        (legitimate — metadata-only measurements never wrote one). Use
        `exists()` if you need to distinguish "no arrays" from "missing".
        """
        path = self._path_for(measurement_id)
        if not path.is_file():
            return {}
        # `pq.read_table` is untyped in pyarrow's stubs.
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        return {
            name: np.asarray(table.column(name).to_numpy(zero_copy_only=False))
            for name in table.column_names
        }

    def exists(self, measurement_id: str) -> bool:
        """True if a Parquet file is on disk for `measurement_id`."""
        return self._path_for(measurement_id).is_file()

    def delete(self, measurement_id: str) -> bool:
        """Remove the Parquet file for `measurement_id`. Returns whether one was deleted.

        Idempotent: calling delete on a non-existent measurement returns
        False but does not raise.
        """
        path = self._path_for(measurement_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # ─── Internals ───────────────────────────────────────────────────
    @staticmethod
    def _build_table(arrays: dict[str, np.ndarray]) -> pa.Table:
        """Construct a pyarrow `Table` from a 1-D-arrays dict.

        Relies on `ParsedData` having already validated that every array
        is 1-D and that all arrays have the same length. We don't re-check
        here — that would just produce duplicate error messages.
        """
        # `pa.array(numpy_arr)` infers the Arrow dtype from numpy and shares
        # the underlying buffer when possible (zero-copy for primitive dtypes).
        return pa.table({name: pa.array(arr) for name, arr in arrays.items()})

    def _cleanup_orphan_temps(self) -> None:
        """Delete any `.parquet.tmp` files left over from previous crashes.

        Called once on construction. Failures to remove individual files are
        silently ignored — an orphan tmp file is harmless and we'd rather
        construct successfully than crash on a stale lock or permission edge.
        """
        for tmp in self._dir.glob(f"*{_TEMP_SUFFIX}"):
            # Don't make construction fail over leftovers we can't clean —
            # an orphan .tmp file is harmless until the next crash overwrites
            # it. Common reasons for failure: file locked by AV scanner,
            # permission edge on a network share.
            with contextlib.suppress(OSError):
                tmp.unlink()
