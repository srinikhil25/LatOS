"""Tests for `latos.ingestion.labeling.pipeline`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from latos.core.enums import FileRole, Technique
from latos.core.models import FileRef, Measurement, Project, Sample
from latos.ingestion.labeling.pipeline import cluster_project, hints_for_project

# ---------------------------------------------------------------------------
# Builders — minimal but valid domain objects for the pipeline.
# ---------------------------------------------------------------------------


def _id(seed: int) -> str:
    return f"{seed:032x}"


def _fileref(path: str) -> FileRef:
    return FileRef(
        path=Path(path),
        sha256="0" * 64,
        size_bytes=1,
        role=FileRole.RAW,
        scanned_at=datetime.now(UTC),
    )


def _measurement(seed: int, sample_id: str, *, files: tuple[FileRef, ...]) -> Measurement:
    return Measurement(
        id=_id(seed),
        sample_id=sample_id,
        technique=Technique.XRD,
        instrument="Stub",
        measured_at=None,
        parsed_at=datetime.now(UTC),
        parser_version="0.0.1",
        files=files,
    )


def _sample(seed: int, project_id: str, name: str, files: tuple[str, ...]) -> Sample:
    return Sample(
        id=_id(seed + 1000),
        project_id=project_id,
        canonical_name=name,
        measurements=(
            _measurement(seed, _id(seed + 1000), files=tuple(_fileref(f) for f in files)),
        ),
    )


def _project(root: Path, *, samples: tuple[Sample, ...]) -> Project:
    return Project(
        id=_id(42),
        name="StubProj",
        root_path=root,
        created_at=datetime.now(UTC),
        schema_version=1,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Headline integration: Dhivya-shaped regression
# ---------------------------------------------------------------------------


class TestDhivyaShape:
    def test_cs_pure_variants_collapse(self, tmp_path: Path):
        # Two samples written cosmetically different in two technique
        # folders. Stage 1's heuristic produced two `Sample`s; the
        # pipeline should collapse them.
        s1 = _sample(
            1,
            _id(42),
            name="CS Pure",
            files=(str(tmp_path / "XRD" / "CS Pure" / "run.xrdml"),),
        )
        s2 = _sample(
            2,
            _id(42),
            name="CS (Pure)",
            files=(str(tmp_path / "XPS" / "CS (Pure)" / "run.csv"),),
        )
        project = _project(tmp_path, samples=(s1, s2))

        clusters = cluster_project(project)

        assert len(clusters) == 1
        cluster = clusters[0]
        # Both file paths land in the single cluster.
        assert len(cluster.file_paths) == 2

    def test_distinct_samples_stay_separate(self, tmp_path: Path):
        # Samples that the Stage 2C threshold should keep apart must
        # stay apart through the project-level pipeline, too.
        s1 = _sample(1, _id(42), "CS-1", files=(str(tmp_path / "XRD" / "CS-1" / "a.xrdml"),))
        s2 = _sample(2, _id(42), "CS-3", files=(str(tmp_path / "XRD" / "CS-3" / "b.xrdml"),))
        project = _project(tmp_path, samples=(s1, s2))

        clusters = cluster_project(project)

        canonicals = {c.canonical for c in clusters}
        assert canonicals == {"CS-1", "CS-3"}


# ---------------------------------------------------------------------------
# hints_for_project()
# ---------------------------------------------------------------------------


class TestHintsForProject:
    def test_one_hint_per_file(self, tmp_path: Path):
        s = _sample(
            1,
            _id(42),
            "CS-1",
            files=(
                str(tmp_path / "XRD" / "CS-1" / "a.xrdml"),
                str(tmp_path / "XRD" / "CS-1" / "b.xrdml"),
            ),
        )
        project = _project(tmp_path, samples=(s,))

        hints = hints_for_project(project)
        assert len(hints) == 2

    def test_duplicate_file_paths_deduplicated(self, tmp_path: Path):
        # If the same path appears under two measurements (rare but
        # legal) the pipeline should not produce two hints for it -
        # otherwise the file double-votes in cluster_samples.
        path = tmp_path / "XRD" / "CS-1" / "a.xrdml"
        m1 = _measurement(1, _id(1001), files=(_fileref(str(path)),))
        m2 = _measurement(2, _id(1001), files=(_fileref(str(path)),))
        sample = Sample(
            id=_id(1001),
            project_id=_id(42),
            canonical_name="CS-1",
            measurements=(m1, m2),
        )
        project = _project(tmp_path, samples=(sample,))

        hints = hints_for_project(project)
        assert len(hints) == 1

    def test_empty_project_returns_empty(self, tmp_path: Path):
        project = _project(tmp_path, samples=())
        assert hints_for_project(project) == ()
        # And cluster_project is empty for an empty project.
        assert cluster_project(project) == ()

    def test_root_is_forwarded_to_extract_hints(self, tmp_path: Path):
        # The path walker should stop at `project.root_path`, so the
        # project's parent folder name doesn't get pulled in as a
        # candidate sample name.
        s = _sample(1, _id(42), "CS-1", files=(str(tmp_path / "CS-1" / "a.xrdml"),))
        project = _project(tmp_path, samples=(s,))

        hints = hints_for_project(project)
        # Root is `tmp_path`; the walker collects only the segment
        # between the file and the root - no `tmp_path.name`.
        for h in hints:
            assert tmp_path.name not in h.from_path_segments


# ---------------------------------------------------------------------------
# Threshold passthrough
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_threshold_zero_collapses_everything(self, tmp_path: Path):
        s1 = _sample(1, _id(42), "Apple", files=(str(tmp_path / "Apple" / "a.csv"),))
        s2 = _sample(2, _id(42), "Banana", files=(str(tmp_path / "Banana" / "b.csv"),))
        project = _project(tmp_path, samples=(s1, s2))

        clusters = cluster_project(project, similarity_threshold=0.0)
        assert len(clusters) == 1

    def test_threshold_one_only_merges_normalized_equals(self, tmp_path: Path):
        # Same name spelled two ways - both normalize to the same
        # string so they merge even at threshold 1.0.
        s1 = _sample(1, _id(42), "CS-1", files=(str(tmp_path / "CS-1" / "a.csv"),))
        s2 = _sample(2, _id(42), "cs_1", files=(str(tmp_path / "cs_1" / "b.csv"),))
        project = _project(tmp_path, samples=(s1, s2))

        clusters = cluster_project(project, similarity_threshold=1.0)
        assert len(clusters) == 1
