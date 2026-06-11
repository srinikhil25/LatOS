"""Tests for `latos.ingestion.array_store.ArrayStore`."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest

from latos.core.enums import Technique
from latos.core.models import utc_now
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.parsed_data import ParsedData


def _make_parsed(arrays: dict[str, np.ndarray]) -> ParsedData:
    """Build a minimal valid ParsedData with the given arrays."""
    return ParsedData(
        technique=Technique.XRD,
        arrays=arrays,
        metadata={},
        instrument="test",
        measured_at=utc_now() - timedelta(days=1),
        issues=(),
        parser_name="test-parser",
        parser_version="1.0.0",
    )


# A 32-char hex ID that mimics the real `MeasurementRow.id` format.
_FAKE_ID = "abcdef0123456789abcdef0123456789"
_OTHER_ID = "fedcba9876543210fedcba9876543210"


# ─── Construction ───────────────────────────────────────────────────
class TestConstruction:
    def test_creates_directory_if_missing(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "arrays"
        assert not target.exists()
        ArrayStore(target)
        assert target.is_dir()

    def test_existing_directory_ok(self, tmp_path: Path):
        d = tmp_path / "arrays"
        d.mkdir()
        ArrayStore(d)  # must not raise

    def test_directory_property(self, tmp_path: Path):
        d = tmp_path / "arrays"
        store = ArrayStore(d)
        assert store.directory == d

    def test_orphan_temp_files_cleaned_on_construction(self, tmp_path: Path):
        # Simulate a previous crash leaving a .tmp behind.
        d = tmp_path / "arrays"
        d.mkdir()
        orphan = d / f"{_FAKE_ID}.parquet.tmp"
        orphan.write_bytes(b"corrupt-half-written-parquet")
        # Real file beside it should NOT be touched.
        survivor = d / f"{_OTHER_ID}.parquet"
        survivor.write_bytes(b"valid")

        ArrayStore(d)

        assert not orphan.exists(), "orphan .tmp should be deleted on construction"
        assert survivor.exists(), "non-tmp files must be preserved"


# ─── Write ──────────────────────────────────────────────────────────
class TestWrite:
    def test_writes_parquet_file(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        data = _make_parsed(
            {
                "two_theta": np.array([10.0, 20.0, 30.0]),
                "intensity": np.array([100.0, 250.0, 80.0]),
            },
        )
        path = store.write(_FAKE_ID, data)
        assert path is not None
        assert path == tmp_path / f"{_FAKE_ID}.parquet"
        assert path.is_file()

    def test_empty_arrays_writes_no_file(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        data = _make_parsed({})
        path = store.write(_FAKE_ID, data)
        assert path is None
        assert not (tmp_path / f"{_FAKE_ID}.parquet").exists()

    def test_overwrite_replaces_existing(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"x": np.array([1.0, 2.0, 3.0])}))
        store.write(_FAKE_ID, _make_parsed({"y": np.array([4.0, 5.0])}))

        loaded = store.load(_FAKE_ID)
        assert "x" not in loaded
        assert "y" in loaded
        np.testing.assert_array_equal(loaded["y"], [4.0, 5.0])

    def test_no_temp_file_left_after_successful_write(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"x": np.array([1.0, 2.0])}))
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []


# ─── Atomicity ──────────────────────────────────────────────────────
class TestAtomicity:
    def test_crash_during_write_does_not_corrupt_existing_file(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        # Establish a valid existing file.
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0, 2.0, 3.0])}))

        # Now simulate a crash mid-write of a NEW value.
        with (
            patch(
                "latos.ingestion.array_store.pq.write_table",
                side_effect=RuntimeError("simulated crash"),
            ),
            pytest.raises(RuntimeError, match="simulated crash"),
        ):
            store.write(_FAKE_ID, _make_parsed({"v": np.array([99.0, 99.0])}))

        # Original data must be intact — atomic write means the .tmp never
        # got swung into place.
        loaded = store.load(_FAKE_ID)
        np.testing.assert_array_equal(loaded["v"], [1.0, 2.0, 3.0])

    def test_crash_cleans_up_tmp_file(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        with (
            patch(
                "latos.ingestion.array_store.pq.write_table",
                side_effect=RuntimeError("simulated crash"),
            ),
            pytest.raises(RuntimeError),
        ):
            store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0, 2.0])}))

        # No .tmp file should be left lying around.
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []

    def test_temp_file_name_does_not_collide_with_real_file(self, tmp_path: Path):
        # If the temp suffix were just `.tmp`, a measurement_id ending in
        # `.parquet` would collide. Our suffix is `.parquet.tmp` so there's
        # no overlap with the final `.parquet` filename.
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0])}))
        real = tmp_path / f"{_FAKE_ID}.parquet"
        tmp = tmp_path / f"{_FAKE_ID}.parquet.tmp"
        assert real.exists()
        assert not tmp.exists()
        assert real != tmp


# ─── Load (round-trip) ──────────────────────────────────────────────
class TestLoad:
    def test_round_trip_preserves_values(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        original = {
            "two_theta": np.array([10.0, 20.0, 30.0, 40.0]),
            "intensity": np.array([100.0, 250.0, 80.0, 30.0]),
        }
        store.write(_FAKE_ID, _make_parsed(original))

        loaded = store.load(_FAKE_ID)
        assert set(loaded.keys()) == {"two_theta", "intensity"}
        np.testing.assert_array_equal(loaded["two_theta"], original["two_theta"])
        np.testing.assert_array_equal(loaded["intensity"], original["intensity"])

    def test_round_trip_preserves_float64_dtype(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0, 2.0], dtype=np.float64)}))
        loaded = store.load(_FAKE_ID)
        assert loaded["v"].dtype == np.float64

    def test_round_trip_preserves_int_dtype(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"counts": np.array([1, 2, 3], dtype=np.int64)}))
        loaded = store.load(_FAKE_ID)
        # Parquet int64 → np int64 round-trip.
        assert np.issubdtype(loaded["counts"].dtype, np.integer)
        np.testing.assert_array_equal(loaded["counts"], [1, 2, 3])

    def test_load_missing_returns_empty_dict(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        assert store.load("missing-id-not-on-disk") == {}

    def test_load_returns_real_ndarrays(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0, 2.0])}))
        loaded = store.load(_FAKE_ID)
        assert isinstance(loaded["v"], np.ndarray)


# ─── exists / delete ────────────────────────────────────────────────
class TestExistsDelete:
    def test_exists_false_when_never_written(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        assert not store.exists(_FAKE_ID)

    def test_exists_true_after_write(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0])}))
        assert store.exists(_FAKE_ID)

    def test_exists_false_when_arrays_empty(self, tmp_path: Path):
        # Metadata-only measurements never write a file → exists() is False.
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({}))
        assert not store.exists(_FAKE_ID)

    def test_delete_returns_true_when_existed(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0])}))
        assert store.delete(_FAKE_ID) is True
        assert not store.exists(_FAKE_ID)

    def test_delete_returns_false_when_missing(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        # Never written → delete is a no-op that returns False, doesn't raise.
        assert store.delete(_FAKE_ID) is False

    def test_delete_idempotent(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0])}))
        assert store.delete(_FAKE_ID) is True
        assert store.delete(_FAKE_ID) is False  # second call: nothing to delete


# ─── Multiple measurements isolated ─────────────────────────────────
class TestIsolation:
    def test_two_measurements_independent(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        a = _make_parsed({"x": np.array([1.0, 2.0])})
        b = _make_parsed({"y": np.array([10.0, 20.0])})
        store.write(_FAKE_ID, a)
        store.write(_OTHER_ID, b)

        loaded_a = store.load(_FAKE_ID)
        loaded_b = store.load(_OTHER_ID)
        np.testing.assert_array_equal(loaded_a["x"], [1.0, 2.0])
        np.testing.assert_array_equal(loaded_b["y"], [10.0, 20.0])

    def test_delete_one_leaves_others(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"x": np.array([1.0])}))
        store.write(_OTHER_ID, _make_parsed({"y": np.array([2.0])}))
        store.delete(_FAKE_ID)
        assert not store.exists(_FAKE_ID)
        assert store.exists(_OTHER_ID)


# ─── Parquet file shape (so anything else can read it) ──────────────
class TestParquetShape:
    def test_parquet_file_has_one_column_per_array(self, tmp_path: Path):
        store = ArrayStore(tmp_path)
        store.write(
            _FAKE_ID,
            _make_parsed(
                {
                    "two_theta": np.array([10.0, 20.0, 30.0]),
                    "intensity": np.array([100.0, 250.0, 80.0]),
                },
            ),
        )
        path = tmp_path / f"{_FAKE_ID}.parquet"
        table = pq.read_table(path)
        # Schema is flat (no nested list types) — pandas/DuckDB friendly.
        assert sorted(table.column_names) == ["intensity", "two_theta"]
        assert table.num_rows == 3

    def test_pandas_can_read_back(self, tmp_path: Path):
        # Sanity: a third party tool — pandas — can open the file.
        import pandas as pd

        store = ArrayStore(tmp_path)
        store.write(_FAKE_ID, _make_parsed({"v": np.array([1.0, 2.0, 3.0])}))
        df = pd.read_parquet(tmp_path / f"{_FAKE_ID}.parquet")
        assert list(df.columns) == ["v"]
        assert list(df["v"]) == [1.0, 2.0, 3.0]


# ─── Hypothesis-style sanity round-trip with varied dtypes ──────────
@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int32, np.int64],
)
def test_round_trip_various_dtypes(tmp_path: Path, dtype: Any):
    store = ArrayStore(tmp_path)
    arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
    store.write(_FAKE_ID, _make_parsed({"v": arr}))
    loaded = store.load(_FAKE_ID)
    np.testing.assert_array_equal(loaded["v"], arr)
