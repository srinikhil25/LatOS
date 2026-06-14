"""Tests for the pure project-edit transforms in `latos.server.edits`."""

from __future__ import annotations

from pathlib import Path

import pytest

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project, Sample, new_id, utc_now
from latos.server import edits
from latos.server.edits import EditError


def _measurement(sample_id: str, technique: Technique = Technique.TEM) -> Measurement:
    return Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=technique,
        instrument=None,
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="t-1",
        files=(),
        issues=(),
        parsed_data_path=None,
        analysis_results=(),
    )


def _project(confirmed: bool = False) -> Project:
    """Two samples: CS (2 measurements) and CS-3 (1)."""
    pid = new_id()
    s1_id, s2_id = new_id(), new_id()
    s1 = Sample(
        id=s1_id,
        project_id=pid,
        canonical_name="CS",
        aliases=(),
        measurements=(_measurement(s1_id), _measurement(s1_id, Technique.TEM)),
    )
    s2 = Sample(
        id=s2_id,
        project_id=pid,
        canonical_name="CS-3",
        aliases=(),
        measurements=(_measurement(s2_id),),
    )
    return Project(
        id=pid,
        name="proj",
        root_path=Path("/tmp/x"),
        created_at=utc_now(),
        schema_version=4,
        samples=(s1, s2),
        unassigned_files=(),
        review_status=ReviewStatus.CONFIRMED if confirmed else ReviewStatus.NEEDS_REVIEW,
        confirmed_at=utc_now() if confirmed else None,
    )


class TestStatusTransitions:
    def test_confirm_sets_confirmed(self):
        p = edits.confirm(_project())
        assert p.review_status is ReviewStatus.CONFIRMED
        assert p.confirmed_at is not None

    def test_reopen_clears_confirmation(self):
        p = edits.reopen(_project(confirmed=True))
        assert p.review_status is ReviewStatus.NEEDS_REVIEW
        assert p.confirmed_at is None


class TestRename:
    def test_rename_changes_name(self):
        proj = _project()
        sid = proj.samples[0].id
        out = edits.rename_sample(proj, sid, "CuSe")
        assert {s.canonical_name for s in out.samples} == {"CuSe", "CS-3"}

    def test_rename_resets_confirmation(self):
        proj = _project(confirmed=True)
        out = edits.rename_sample(proj, proj.samples[0].id, "X")
        assert out.review_status is ReviewStatus.NEEDS_REVIEW
        assert out.confirmed_at is None

    def test_rename_unknown_sample_raises(self):
        with pytest.raises(EditError):
            edits.rename_sample(_project(), "nope", "X")

    def test_rename_empty_raises(self):
        with pytest.raises(EditError):
            edits.rename_sample(_project(), _project().samples[0].id, "  ")


class TestSetTechnique:
    def test_overrides_one_measurement(self):
        proj = _project()
        mid = proj.samples[0].measurements[0].id
        out = edits.set_measurement_technique(proj, mid, Technique.STEM)
        techniques = {m.id: m.technique for s in out.samples for m in s.measurements}
        assert techniques[mid] is Technique.STEM

    def test_unknown_measurement_raises(self):
        with pytest.raises(EditError):
            edits.set_measurement_technique(_project(), "nope", Technique.STEM)


class TestMerge:
    def test_merge_folds_source_into_target(self):
        proj = _project()
        target = proj.samples[0].id  # CS (2)
        source = proj.samples[1].id  # CS-3 (1)
        out = edits.merge_samples(proj, [source], target)
        assert len(out.samples) == 1
        merged = out.samples[0]
        assert merged.id == target
        assert len(merged.measurements) == 3
        # Every measurement now points at the target sample.
        assert all(m.sample_id == target for m in merged.measurements)
        # The source name is retained as an alias.
        assert "CS-3" in merged.aliases

    def test_target_in_sources_raises(self):
        proj = _project()
        sid = proj.samples[0].id
        with pytest.raises(EditError):
            edits.merge_samples(proj, [sid], sid)


class TestSplit:
    def test_split_moves_measurements_to_new_sample(self):
        proj = _project()
        cs = proj.samples[0]
        move_id = cs.measurements[0].id
        out = edits.move_measurements_to_new_sample(proj, [move_id], "CS-pure")
        names = {s.canonical_name for s in out.samples}
        assert "CS-pure" in names
        new_sample = next(s for s in out.samples if s.canonical_name == "CS-pure")
        assert len(new_sample.measurements) == 1
        assert new_sample.measurements[0].id == move_id
        assert new_sample.measurements[0].sample_id == new_sample.id

    def test_split_drops_emptied_source(self):
        proj = _project()
        cs3 = proj.samples[1]  # has exactly 1 measurement
        out = edits.move_measurements_to_new_sample(proj, [cs3.measurements[0].id], "Moved")
        # CS-3 had its only measurement moved → it disappears.
        assert "CS-3" not in {s.canonical_name for s in out.samples}

    def test_split_unknown_measurement_raises(self):
        with pytest.raises(EditError):
            edits.move_measurements_to_new_sample(_project(), ["nope"], "X")

    def test_split_empty_name_raises(self):
        proj = _project()
        mid = proj.samples[0].measurements[0].id
        with pytest.raises(EditError):
            edits.move_measurements_to_new_sample(proj, [mid], "  ")
