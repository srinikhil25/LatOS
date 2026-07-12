"""Tests for the synthesis-log ingestion (`latos.server.synthesis_log`)."""

from __future__ import annotations

from pathlib import Path

from latos.core.models import Project, Sample, new_id, utc_now
from latos.ingestion.orchestrator import IngestionResult
from latos.server import synthesis_log, synthesis_store
from latos.server.state import ServerState


def _project(root: Path, names_aliases: list[tuple[str, tuple[str, ...]]]) -> Project:
    pid = new_id()
    samples = tuple(
        Sample(
            id=new_id(),
            project_id=pid,
            canonical_name=name,
            aliases=aliases,
            measurements=(),
        )
        for name, aliases in names_aliases
    )
    return Project(
        id=pid,
        name=root.name,
        root_path=root,
        created_at=utc_now(),
        schema_version=4,
        samples=samples,
        unassigned_files=(),
    )


def _write_log(root: Path, text: str, name: str = "synthesis.csv") -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


class TestParse:
    def test_happy_path_with_spaces(self, tmp_path: Path):
        path = _write_log(tmp_path, "sample, doping_pct, anneal_c\nCS-1, 1, 450\nCS-3, 3, 450\n")
        rows, variables, problems = synthesis_log.parse_log(path)
        assert variables == ("doping_pct", "anneal_c")
        assert rows["CS-1"] == {"doping_pct": 1.0, "anneal_c": 450.0}
        assert not problems

    def test_bom_tolerated(self, tmp_path: Path):
        path = tmp_path / "synthesis.csv"
        path.write_bytes(b"\xef\xbb\xbfsample,doping_pct\nCS,0\n")
        rows, _, problems = synthesis_log.parse_log(path)
        assert rows["CS"] == {"doping_pct": 0.0}
        assert not problems

    def test_non_numeric_cell_reported_and_skipped(self, tmp_path: Path):
        path = _write_log(tmp_path, "sample,doping_pct\nCS-1,one percent\nCS-3,3\n")
        rows, _, problems = synthesis_log.parse_log(path)
        assert "CS-1" not in rows  # its only cell was invalid
        assert rows["CS-3"] == {"doping_pct": 3.0}
        assert any("one percent" in p for p in problems)

    def test_empty_cells_mean_not_given(self, tmp_path: Path):
        path = _write_log(tmp_path, "sample,doping_pct,anneal_c\nCS-1,1,\n")
        rows, _, problems = synthesis_log.parse_log(path)
        assert rows["CS-1"] == {"doping_pct": 1.0}
        assert not problems

    def test_header_only_sample_column_is_a_problem(self, tmp_path: Path):
        path = _write_log(tmp_path, "sample\nCS-1\n")
        rows, variables, problems = synthesis_log.parse_log(path)
        assert rows == {} and variables == ()
        assert problems


class TestFind:
    def test_none_when_absent(self, tmp_path: Path):
        assert synthesis_log.find_log(tmp_path) is None

    def test_root_level_beats_nested(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        nested = _write_log(tmp_path / "sub", "sample,x\nA,1\n")
        top = _write_log(tmp_path, "sample,x\nA,2\n")
        assert synthesis_log.find_log(tmp_path) == top
        assert nested.exists()

    def test_latos_store_never_searched(self, tmp_path: Path):
        (tmp_path / ".latos").mkdir()
        _write_log(tmp_path / ".latos", "sample,x\nA,1\n")
        assert synthesis_log.find_log(tmp_path) is None


class TestApply:
    def test_matches_by_normalized_name_and_alias(self, tmp_path: Path):
        project = _project(tmp_path, [("CS-1", ()), ("CS-3", ("CS-CBI-3",))])
        _write_log(tmp_path, "sample,doping_pct\ncs_1,1\nCS-CBI-3,3\n")
        report = synthesis_log.apply_log(tmp_path, project)
        assert report is not None and report.applied == 2
        params = synthesis_store.load_params(tmp_path)
        by_name = {s.canonical_name: s.id for s in project.samples}
        assert params[by_name["CS-1"]] == {"doping_pct": 1.0}
        assert params[by_name["CS-3"]] == {"doping_pct": 3.0}

    def test_log_wins_for_named_vars_preserves_others(self, tmp_path: Path):
        project = _project(tmp_path, [("CS-1", ())])
        sid = project.samples[0].id
        synthesis_store.set_sample_params(tmp_path, sid, {"doping_pct": 9.0, "ball_mill_h": 12.0})
        _write_log(tmp_path, "sample,doping_pct\nCS-1,1\n")
        synthesis_log.apply_log(tmp_path, project)
        params = synthesis_store.load_params(tmp_path)
        assert params[sid] == {"doping_pct": 1.0, "ball_mill_h": 12.0}

    def test_unmatched_rows_reported_not_applied(self, tmp_path: Path):
        project = _project(tmp_path, [("CS-1", ())])
        _write_log(tmp_path, "sample,doping_pct\nCS-1,1\nNO-SUCH,5\n")
        report = synthesis_log.apply_log(tmp_path, project)
        assert report is not None
        assert report.unmatched_rows == ("NO-SUCH",)
        assert report.applied == 1

    def test_no_log_returns_none(self, tmp_path: Path):
        project = _project(tmp_path, [("CS-1", ())])
        assert synthesis_log.apply_log(tmp_path, project) is None


class TestIngestHook:
    def test_log_applied_at_end_of_ingest(self, tmp_path: Path):
        """The ServerState worker applies the log once samples exist."""
        project = _project(tmp_path, [("CS-1", ()), ("CS-3", ())])
        _write_log(tmp_path, "sample,doping_pct\nCS-1,1\nCS-3,3\n")

        class _FakeOrchestrator:
            def ingest(self, root, *, project_name=None, on_progress=None):
                return IngestionResult(project=project, outcomes=())

        state = ServerState(orchestrator_factory=_FakeOrchestrator)
        assert state.start_ingest(tmp_path)
        state.join(timeout=10)
        params = synthesis_store.load_params(tmp_path)
        by_name = {s.canonical_name: s.id for s in project.samples}
        assert params[by_name["CS-1"]] == {"doping_pct": 1.0}
        assert params[by_name["CS-3"]] == {"doping_pct": 3.0}
