"""Tests for the researcher distrust sidecar (`latos.server.trust_store`)."""

from __future__ import annotations

import json
from pathlib import Path

from latos.server import trust_store


class TestRoundTrip:
    def test_no_file_is_an_empty_set_not_an_error(self, tmp_path: Path):
        # A fresh project has trusted everything by default.
        assert trust_store.load_distrusted(tmp_path) == set()

    def test_save_then_load(self, tmp_path: Path):
        trust_store.save_distrusted(tmp_path, {"s1", "s2"})
        assert trust_store.load_distrusted(tmp_path) == {"s1", "s2"}

    def test_written_under_dot_latos(self, tmp_path: Path):
        trust_store.save_distrusted(tmp_path, {"s1"})
        path = tmp_path / ".latos" / "distrusted_samples.json"
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8")) == ["s1"]

    def test_stored_sorted_for_stable_diffs(self, tmp_path: Path):
        trust_store.save_distrusted(tmp_path, {"c", "a", "b"})
        path = tmp_path / ".latos" / "distrusted_samples.json"
        assert json.loads(path.read_text(encoding="utf-8")) == ["a", "b", "c"]

    def test_no_temp_file_left_behind(self, tmp_path: Path):
        trust_store.save_distrusted(tmp_path, {"s1"})
        assert list((tmp_path / ".latos").glob("*.tmp")) == []


class TestSetDistrusted:
    def test_marking_adds(self, tmp_path: Path):
        assert trust_store.set_distrusted(tmp_path, "s1", True) == {"s1"}

    def test_clearing_removes(self, tmp_path: Path):
        trust_store.set_distrusted(tmp_path, "s1", True)
        assert trust_store.set_distrusted(tmp_path, "s1", False) == set()

    def test_clearing_an_untracked_sample_is_a_no_op(self, tmp_path: Path):
        assert trust_store.set_distrusted(tmp_path, "never-seen", False) == set()

    def test_marking_twice_is_idempotent(self, tmp_path: Path):
        trust_store.set_distrusted(tmp_path, "s1", True)
        assert trust_store.set_distrusted(tmp_path, "s1", True) == {"s1"}

    def test_others_are_preserved(self, tmp_path: Path):
        trust_store.set_distrusted(tmp_path, "s1", True)
        trust_store.set_distrusted(tmp_path, "s2", True)
        assert trust_store.set_distrusted(tmp_path, "s1", False) == {"s2"}


class TestCorruptFile:
    """A damaged sidecar must never take the optimizer down with it.

    The safe reading of an unreadable trust file is "nothing is distrusted":
    it restores the default state, and the researcher can see the empty
    checkboxes and re-tick.
    """

    def _write(self, root: Path, text: str) -> None:
        path = root / ".latos" / "distrusted_samples.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_invalid_json_reads_as_empty(self, tmp_path: Path):
        self._write(tmp_path, "{not json")
        assert trust_store.load_distrusted(tmp_path) == set()

    def test_wrong_shape_reads_as_empty(self, tmp_path: Path):
        self._write(tmp_path, '{"s1": true}')
        assert trust_store.load_distrusted(tmp_path) == set()

    def test_non_string_members_are_dropped(self, tmp_path: Path):
        self._write(tmp_path, '["s1", 42, null, "s2"]')
        assert trust_store.load_distrusted(tmp_path) == {"s1", "s2"}

    def test_a_corrupt_file_can_be_overwritten(self, tmp_path: Path):
        self._write(tmp_path, "{not json")
        assert trust_store.set_distrusted(tmp_path, "s1", True) == {"s1"}
