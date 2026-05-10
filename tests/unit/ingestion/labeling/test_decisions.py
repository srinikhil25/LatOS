"""Tests for `latos.ingestion.labeling.decisions`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latos.ingestion.labeling.cluster import SampleCluster
from latos.ingestion.labeling.decisions import (
    DECISIONS_FILENAME,
    ClusterDecisions,
    apply_decisions,
    load_decisions,
    save_decisions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster(canonical: str, *, aliases: tuple[str, ...] = (), files: tuple[str, ...] = ()):
    """Build a `SampleCluster` for tests with sensible defaults."""
    return SampleCluster(
        canonical=canonical,
        aliases=aliases or (canonical,),
        file_paths=tuple(Path(f) for f in files),
        normalized_forms=(),
    )


# ---------------------------------------------------------------------------
# Dataclass — `with_*` helpers return new instances
# ---------------------------------------------------------------------------


class TestWithRename:
    def test_adds_a_new_rename(self):
        d = ClusterDecisions().with_rename("auto-1", "MX-1")
        assert d.renames == {"auto-1": "MX-1"}

    def test_overwrites_an_existing_rename(self):
        d = ClusterDecisions(renames={"auto-1": "old"}).with_rename("auto-1", "new")
        assert d.renames == {"auto-1": "new"}

    def test_blank_target_clears_the_rename(self):
        # An empty / whitespace-only rename means "undo this rename" —
        # otherwise editing the field to "" would persist a blank
        # canonical and break sample lookup.
        d = ClusterDecisions(renames={"auto-1": "X"}).with_rename("auto-1", "   ")
        assert d.renames == {}

    def test_renaming_to_the_same_string_is_a_noop(self):
        d = ClusterDecisions().with_rename("auto-1", "auto-1")
        assert d.renames == {}

    def test_does_not_mutate_input(self):
        original = ClusterDecisions()
        new = original.with_rename("auto-1", "X")
        assert original.renames == {}
        assert new.renames == {"auto-1": "X"}


class TestWithMerge:
    def test_records_a_two_canonical_merge(self):
        d = ClusterDecisions().with_merge(["a", "b"])
        assert d.merges == (("a", "b"),)

    def test_first_canonical_is_the_survivor(self):
        # Apply order is documented to use group[0]. The data layer
        # just records the order the caller supplied.
        d = ClusterDecisions().with_merge(["b", "a", "c"])
        assert d.merges[0][0] == "b"

    def test_single_item_merge_is_dropped(self):
        d = ClusterDecisions().with_merge(["alone"])
        assert d.merges == ()

    def test_empty_merge_is_dropped(self):
        d = ClusterDecisions().with_merge([])
        assert d.merges == ()

    def test_duplicate_canonicals_collapse(self):
        d = ClusterDecisions().with_merge(["a", "a", "b", "a"])
        assert d.merges == (("a", "b"),)

    def test_new_merge_supersedes_overlapping_prior_group(self):
        d = ClusterDecisions(merges=(("a", "b"),)).with_merge(["b", "c"])
        # The prior {a,b} gets dropped because b moves to the new
        # group; we don't auto-collapse a + b + c.
        assert d.merges == (("b", "c"),)


class TestWithSplit:
    def test_records_a_split_from_path_keys(self):
        # Path stringification is platform-dependent (Windows uses
        # backslashes), so derive the expected keys via `str(Path(...))`
        # rather than hard-coding forward slashes.
        pa = Path("/p/a.csv")
        pb = Path("/p/b.csv")
        d = ClusterDecisions().with_split("auto-1", {pa: "MX-1", pb: "MX-2"})
        assert d.splits == {"auto-1": {str(pa): "MX-1", str(pb): "MX-2"}}

    def test_accepts_string_keys_too(self):
        # The UI may already have stringified its paths.
        d = ClusterDecisions().with_split("auto-1", {"/p/a.csv": "MX-1"})
        assert d.splits == {"auto-1": {"/p/a.csv": "MX-1"}}

    def test_empty_mapping_clears_prior_split(self):
        prior = ClusterDecisions(splits={"auto-1": {"/p/a.csv": "MX-1"}})
        cleared = prior.with_split("auto-1", {})
        assert cleared.splits == {}

    def test_blank_target_drops_that_assignment(self):
        d = ClusterDecisions().with_split(
            "auto-1",
            {"/p/a.csv": "MX-1", "/p/b.csv": "  "},
        )
        assert d.splits == {"auto-1": {"/p/a.csv": "MX-1"}}


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_load_returns_empty_when_file_missing(self, tmp_path: Path):
        loaded = load_decisions(tmp_path)
        assert loaded == ClusterDecisions()

    def test_save_then_load_recovers_renames(self, tmp_path: Path):
        d = ClusterDecisions(renames={"auto-1": "MX-1"})
        save_decisions(tmp_path, d)
        assert load_decisions(tmp_path).renames == {"auto-1": "MX-1"}

    def test_save_then_load_recovers_merges(self, tmp_path: Path):
        d = ClusterDecisions(merges=(("a", "b"), ("c", "d")))
        save_decisions(tmp_path, d)
        assert load_decisions(tmp_path).merges == (("a", "b"), ("c", "d"))

    def test_save_then_load_recovers_splits(self, tmp_path: Path):
        d = ClusterDecisions(splits={"auto-1": {"/p/a.csv": "MX-1"}})
        save_decisions(tmp_path, d)
        assert load_decisions(tmp_path).splits == {"auto-1": {"/p/a.csv": "MX-1"}}

    def test_save_creates_latos_dir_if_missing(self, tmp_path: Path):
        # Project root with no `.latos/` yet — save() must create it.
        save_decisions(tmp_path, ClusterDecisions(renames={"a": "b"}))
        assert (tmp_path / ".latos" / DECISIONS_FILENAME).exists()

    def test_save_overwrites_existing_file(self, tmp_path: Path):
        save_decisions(tmp_path, ClusterDecisions(renames={"a": "old"}))
        save_decisions(tmp_path, ClusterDecisions(renames={"a": "new"}))
        assert load_decisions(tmp_path).renames == {"a": "new"}


class TestPersistenceRobustness:
    def test_corrupt_json_raises_value_error(self, tmp_path: Path):
        path = tmp_path / ".latos" / DECISIONS_FILENAME
        path.parent.mkdir(parents=True)
        path.write_text("{ this is not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupt cluster decisions"):
            load_decisions(tmp_path)

    def test_non_object_root_raises(self, tmp_path: Path):
        path = tmp_path / ".latos" / DECISIONS_FILENAME
        path.parent.mkdir(parents=True)
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be an object"):
            load_decisions(tmp_path)

    def test_save_writes_pretty_json(self, tmp_path: Path):
        # The user can hand-edit the file, so it should be readable.
        save_decisions(tmp_path, ClusterDecisions(renames={"a": "b"}))
        text = (tmp_path / ".latos" / DECISIONS_FILENAME).read_text(encoding="utf-8")
        # Indentation and a trailing newline-ish layout.
        assert "  " in text
        # Parses back as JSON.
        json.loads(text)


# ---------------------------------------------------------------------------
# apply_decisions()
# ---------------------------------------------------------------------------


class TestApplyRenames:
    def test_rename_changes_canonical(self):
        clusters = (_cluster("auto-1", files=("/p/a.csv",)),)
        decisions = ClusterDecisions(renames={"auto-1": "MX-1"})
        result = apply_decisions(clusters, decisions)
        assert len(result) == 1
        assert result[0].canonical == "MX-1"

    def test_rename_preserves_files(self):
        clusters = (_cluster("auto-1", files=("/p/a.csv", "/p/b.csv")),)
        decisions = ClusterDecisions(renames={"auto-1": "MX-1"})
        result = apply_decisions(clusters, decisions)
        assert len(result[0].file_paths) == 2

    def test_rename_adds_new_name_to_aliases(self):
        clusters = (_cluster("auto-1", aliases=("auto-1", "alt"), files=("/p/a.csv",)),)
        decisions = ClusterDecisions(renames={"auto-1": "MX-1"})
        result = apply_decisions(clusters, decisions)
        assert "MX-1" in result[0].aliases
        # Old aliases preserved.
        assert "auto-1" in result[0].aliases

    def test_rename_with_no_match_leaves_clusters_unchanged(self):
        clusters = (_cluster("auto-1", files=("/p/a.csv",)),)
        decisions = ClusterDecisions(renames={"phantom": "X"})
        result = apply_decisions(clusters, decisions)
        assert result[0].canonical == "auto-1"


class TestApplyMerges:
    def test_two_clusters_merge_into_one(self):
        clusters = (
            _cluster("CS-1", files=("/p/a.csv",)),
            _cluster("cs_1", files=("/p/b.csv",)),
        )
        decisions = ClusterDecisions(merges=(("CS-1", "cs_1"),))
        result = apply_decisions(clusters, decisions)
        assert len(result) == 1
        assert result[0].canonical == "CS-1"
        assert len(result[0].file_paths) == 2

    def test_merge_aggregates_aliases(self):
        clusters = (
            _cluster("A", aliases=("A", "a"), files=("/p/a.csv",)),
            _cluster("B", aliases=("B", "b"), files=("/p/b.csv",)),
        )
        decisions = ClusterDecisions(merges=(("A", "B"),))
        result = apply_decisions(clusters, decisions)
        assert set(result[0].aliases) >= {"A", "a", "B", "b"}

    def test_merge_with_one_existing_cluster_is_noop(self):
        # If the user merges A + ghost, only A exists. The merge is
        # silently ignored — applying decisions shouldn't fail because
        # data shifted under the user.
        clusters = (_cluster("A", files=("/p/a.csv",)),)
        decisions = ClusterDecisions(merges=(("A", "ghost"),))
        result = apply_decisions(clusters, decisions)
        assert len(result) == 1
        assert result[0].canonical == "A"


class TestApplySplits:
    def test_split_pulls_one_file_into_a_new_cluster(self):
        pa, pb = Path("/p/a.csv"), Path("/p/b.csv")
        clusters = (_cluster("source", files=(str(pa), str(pb))),)
        decisions = ClusterDecisions(splits={"source": {str(pa): "MX-1"}})
        result = apply_decisions(clusters, decisions)

        canon = {c.canonical: c for c in result}
        assert "source" in canon
        assert "MX-1" in canon
        assert canon["MX-1"].file_paths == (pa,)
        assert canon["source"].file_paths == (pb,)

    def test_split_into_existing_cluster_extends_it(self):
        pa, pc = Path("/p/a.csv"), Path("/p/c.csv")
        clusters = (
            _cluster("source", files=(str(pa),)),
            _cluster("MX-1", files=(str(pc),)),
        )
        decisions = ClusterDecisions(splits={"source": {str(pa): "MX-1"}})
        result = apply_decisions(clusters, decisions)

        canon = {c.canonical: c for c in result}
        # Source vanishes - all its files left.
        assert "source" not in canon
        # MX-1 picked up the new file.
        assert set(canon["MX-1"].file_paths) == {pa, pc}

    def test_full_split_removes_source_cluster(self):
        pa, pb = Path("/p/a.csv"), Path("/p/b.csv")
        clusters = (_cluster("source", files=(str(pa), str(pb))),)
        decisions = ClusterDecisions(splits={"source": {str(pa): "MX-1", str(pb): "MX-2"}})
        result = apply_decisions(clusters, decisions)
        assert {c.canonical for c in result} == {"MX-1", "MX-2"}


class TestApplyOrder:
    def test_split_then_merge_then_rename(self):
        # Splits run first - file leaves "source" into "extra".
        # Merges run next - "A" + "B" - still keyed under "A".
        # Renames last - "A" -> "MX-final".
        px, py = Path("/p/x.csv"), Path("/p/y.csv")
        pa, pb = Path("/p/a.csv"), Path("/p/b.csv")
        clusters = (
            _cluster("source", files=(str(px), str(py))),
            _cluster("A", files=(str(pa),)),
            _cluster("B", files=(str(pb),)),
        )
        decisions = ClusterDecisions(
            renames={"A": "MX-final"},
            merges=(("A", "B"),),
            splits={"source": {str(px): "extra"}},
        )
        result = apply_decisions(clusters, decisions)

        canon = {c.canonical: c for c in result}
        assert set(canon) == {"MX-final", "source", "extra"}
        # "extra" got the split file.
        assert canon["extra"].file_paths == (px,)
        # "source" kept the un-reassigned file.
        assert canon["source"].file_paths == (py,)
        # MX-final has the merged + renamed cluster's files.
        assert set(canon["MX-final"].file_paths) == {pa, pb}

    def test_results_sorted_by_canonical(self):
        clusters = (
            _cluster("Z", files=("/p/z.csv",)),
            _cluster("A", files=("/p/a.csv",)),
        )
        result = apply_decisions(clusters, ClusterDecisions())
        canonicals = [c.canonical for c in result]
        assert canonicals == sorted(canonicals)
