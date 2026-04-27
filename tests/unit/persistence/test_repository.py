"""Tests for `latos.persistence.repository.ProjectRepository`."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from latos.core.enums import Severity, Technique
from latos.core.exceptions import ProjectNotFoundError
from latos.core.models import new_id
from latos.persistence.repository import ProjectRepository

from .conftest import make_issue, make_measurement, make_project, make_sample


# ─── Empty DB ───────────────────────────────────────────────────────
class TestEmpty:
    def test_list_returns_empty(self, repo: ProjectRepository) -> None:
        assert repo.list_projects() == []

    def test_load_first_returns_none(self, repo: ProjectRepository) -> None:
        assert repo.load_first() is None

    def test_load_missing_raises(self, repo: ProjectRepository) -> None:
        with pytest.raises(ProjectNotFoundError):
            repo.load(new_id())

    def test_exists_false(self, repo: ProjectRepository) -> None:
        assert repo.exists(new_id()) is False

    def test_delete_missing_is_noop(self, repo: ProjectRepository) -> None:
        # Should not raise.
        repo.delete(new_id())


# ─── Save + Load round-trip ─────────────────────────────────────────
class TestRoundTrip:
    def test_empty_project(self, repo: ProjectRepository) -> None:
        p = make_project()
        repo.save(p)
        loaded = repo.load(p.id)
        assert loaded == p

    def test_project_with_samples(self, repo: ProjectRepository) -> None:
        p_skeleton = make_project()
        s1 = make_sample(p_skeleton.id, name="CS")
        s2 = make_sample(p_skeleton.id, name="CS-1")
        # Rebuild project with samples
        from latos.core.models import Project

        p = Project(
            id=p_skeleton.id,
            name=p_skeleton.name,
            root_path=p_skeleton.root_path,
            created_at=p_skeleton.created_at,
            schema_version=p_skeleton.schema_version,
            samples=(s1, s2),
        )
        repo.save(p)
        loaded = repo.load(p.id)
        assert {s.canonical_name for s in loaded.samples} == {"CS", "CS-1"}

    def test_sample_with_measurements_and_files(self, repo: ProjectRepository) -> None:
        from latos.core.models import Project, Sample

        p_skel = make_project()
        sid = new_id()
        m_xrd = make_measurement(sid, technique=Technique.XRD, file_sha="a" * 64)
        m_xps = make_measurement(sid, technique=Technique.XPS, file_sha="b" * 64)
        s = Sample(
            id=sid,
            project_id=p_skel.id,
            canonical_name="CS",
            aliases=("cs", "CS Pure"),
            measurements=(m_xrd, m_xps),
        )
        p = Project(
            id=p_skel.id,
            name=p_skel.name,
            root_path=p_skel.root_path,
            created_at=p_skel.created_at,
            schema_version=p_skel.schema_version,
            samples=(s,),
        )
        repo.save(p)
        loaded = repo.load(p.id)
        assert len(loaded.samples) == 1
        loaded_s = loaded.samples[0]
        assert loaded_s.aliases == ("cs", "CS Pure")
        assert {m.technique for m in loaded_s.measurements} == {
            Technique.XRD,
            Technique.XPS,
        }

    def test_measurement_with_issues(self, repo: ProjectRepository) -> None:
        from latos.core.models import Project, Sample

        p_skel = make_project()
        sid = new_id()
        m = make_measurement(
            sid,
            technique=Technique.THERMOELECTRIC,
            file_sha="c" * 64,
            issues=(
                make_issue(severity=Severity.WARNING),
                make_issue(severity=Severity.ERROR),
            ),
        )
        s = Sample(
            id=sid,
            project_id=p_skel.id,
            canonical_name="CS",
            measurements=(m,),
        )
        p = Project(
            id=p_skel.id,
            name=p_skel.name,
            root_path=p_skel.root_path,
            created_at=p_skel.created_at,
            schema_version=p_skel.schema_version,
            samples=(s,),
        )
        repo.save(p)
        loaded = repo.load(p.id)
        loaded_m = loaded.samples[0].measurements[0]
        assert len(loaded_m.issues) == 2
        assert loaded_m.has_errors
        assert loaded_m.has_warnings

    def test_unassigned_files(self, repo: ProjectRepository) -> None:
        from .conftest import make_file_ref

        p_skel = make_project(unassigned=(make_file_ref(sha256="d" * 64),))
        repo.save(p_skel)
        loaded = repo.load(p_skel.id)
        assert len(loaded.unassigned_files) == 1


# ─── Re-save semantics ──────────────────────────────────────────────
class TestReSave:
    def test_resave_replaces_aggregate(self, repo: ProjectRepository) -> None:
        from latos.core.models import Project

        p_skel = make_project()
        s_v1 = make_sample(p_skel.id, name="CS")
        p_v1 = Project(
            id=p_skel.id,
            name=p_skel.name,
            root_path=p_skel.root_path,
            created_at=p_skel.created_at,
            schema_version=p_skel.schema_version,
            samples=(s_v1,),
        )
        repo.save(p_v1)

        # Now re-save with a totally different sample structure.
        s_v2_a = make_sample(p_skel.id, name="ALPHA")
        s_v2_b = make_sample(p_skel.id, name="BETA")
        p_v2 = Project(
            id=p_skel.id,
            name="Renamed",
            root_path=p_skel.root_path,
            created_at=p_skel.created_at,
            schema_version=p_skel.schema_version,
            samples=(s_v2_a, s_v2_b),
        )
        repo.save(p_v2)

        loaded = repo.load(p_skel.id)
        assert loaded.name == "Renamed"
        assert {s.canonical_name for s in loaded.samples} == {"ALPHA", "BETA"}

    def test_resave_doesnt_leak_old_rows(
        self,
        repo: ProjectRepository,
        session_factory: sessionmaker[Session],
    ) -> None:
        """After re-save, old samples must be cascade-deleted."""
        from latos.core.models import Project

        p_skel = make_project()
        s_v1 = make_sample(p_skel.id, name="CS")
        p_v1 = Project(
            id=p_skel.id,
            name=p_skel.name,
            root_path=p_skel.root_path,
            created_at=p_skel.created_at,
            schema_version=p_skel.schema_version,
            samples=(s_v1,),
        )
        repo.save(p_v1)
        repo.save(make_project())  # different project ID; should not affect p_v1
        # Save a v2 with new samples
        s_v2 = make_sample(p_skel.id, name="NEW")
        p_v2 = Project(
            id=p_skel.id,
            name=p_skel.name,
            root_path=p_skel.root_path,
            created_at=p_skel.created_at,
            schema_version=p_skel.schema_version,
            samples=(s_v2,),
        )
        repo.save(p_v2)

        # Direct DB check: only NEW sample remains for p_v2
        with session_factory() as sess:
            count = sess.execute(
                text("SELECT COUNT(*) FROM samples WHERE project_id = :pid"),
                {"pid": p_skel.id},
            ).scalar()
        assert count == 1


# ─── Delete + cascade ───────────────────────────────────────────────
class TestDelete:
    def test_delete_removes_project(self, repo: ProjectRepository) -> None:
        p = make_project()
        repo.save(p)
        repo.delete(p.id)
        assert not repo.exists(p.id)
        with pytest.raises(ProjectNotFoundError):
            repo.load(p.id)

    def test_delete_cascades_to_samples_and_measurements(
        self,
        repo: ProjectRepository,
        session_factory: sessionmaker[Session],
    ) -> None:
        from latos.core.models import Project, Sample

        p_skel = make_project()
        sid = new_id()
        m = make_measurement(sid, file_sha="e" * 64)
        s = Sample(
            id=sid,
            project_id=p_skel.id,
            canonical_name="CS",
            measurements=(m,),
        )
        p = Project(
            id=p_skel.id,
            name=p_skel.name,
            root_path=p_skel.root_path,
            created_at=p_skel.created_at,
            schema_version=p_skel.schema_version,
            samples=(s,),
        )
        repo.save(p)
        repo.delete(p.id)

        with session_factory() as sess:
            assert sess.execute(text("SELECT COUNT(*) FROM samples")).scalar() == 0
            assert sess.execute(text("SELECT COUNT(*) FROM measurements")).scalar() == 0
            assert sess.execute(text("SELECT COUNT(*) FROM files")).scalar() == 0


# ─── List + summaries ───────────────────────────────────────────────
class TestList:
    def test_summaries_match_saved_projects(self, repo: ProjectRepository) -> None:
        p1 = make_project()
        p2 = make_project()
        repo.save(p1)
        repo.save(p2)
        ids = {s.id for s in repo.list_projects()}
        assert ids == {p1.id, p2.id}

    def test_summary_fields_correct(self, repo: ProjectRepository) -> None:
        p = make_project()
        repo.save(p)
        [summary] = repo.list_projects()
        assert summary.id == p.id
        assert summary.name == p.name
        assert summary.root_path == p.root_path
        assert summary.schema_version == p.schema_version

    def test_load_first_returns_only_project(self, repo: ProjectRepository) -> None:
        p = make_project()
        repo.save(p)
        assert repo.load_first() == p


# ─── Foreign-key constraint ─────────────────────────────────────────
def test_unique_canonical_name_per_project(repo: ProjectRepository) -> None:
    """Two samples with same canonical_name in same project should be rejected
    by the DB constraint. Domain prevents this earlier, but we want the
    DB-level safety net too."""
    from sqlalchemy.exc import IntegrityError

    from latos.persistence.schema import SampleRow

    pid = new_id()
    p = make_project()
    repo.save(p)
    # Drop the saved project then bypass the repo to insert duplicates
    repo.delete(p.id)
    with pytest.raises(IntegrityError), repo._sessions() as sess:  # type: ignore[attr-defined]
        sess.add(
            SampleRow(
                id=new_id(),
                project_id=pid,
                canonical_name="CS",
                aliases=[],
            )
        )
        sess.add(
            SampleRow(
                id=new_id(),
                project_id=pid,
                canonical_name="CS",
                aliases=[],
            )
        )
        sess.commit()
