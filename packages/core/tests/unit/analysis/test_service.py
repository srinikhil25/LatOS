"""Tests for `latos.analysis.service.AnalysisService`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.analysis.service import AnalysisService, _fingerprint
from latos.core.enums import FileRole, Severity, Technique
from latos.core.exceptions import AnalysisError
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
    new_id,
    utc_now,
)
from latos.ingestion.array_store import ArrayStore
from latos.persistence.db import (
    create_memory_engine,
    init_schema,
    make_session_factory,
)
from latos.persistence.repository import ProjectRepository


# ─── Fixtures ───────────────────────────────────────────────────────
@pytest.fixture
def engine() -> Iterator[Engine]:
    """In-memory SQLite with schema applied. Disposed at teardown."""
    eng = create_memory_engine()
    init_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


@pytest.fixture
def repo(session_factory: sessionmaker[Session]) -> ProjectRepository:
    return ProjectRepository(session_factory)


@pytest.fixture
def array_store(tmp_path: Path) -> ArrayStore:
    """ArrayStore in a tmp directory."""
    return ArrayStore(tmp_path / "arrays")


@pytest.fixture
def service(repo: ProjectRepository, array_store: ArrayStore) -> AnalysisService:
    return AnalysisService(repository=repo, array_store=array_store)


def _make_project_with_measurement(
    repo: ProjectRepository,
    array_store: ArrayStore,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    technique: Technique = Technique.UV_DRS,
) -> tuple[Project, Sample, Measurement]:
    """Seed the DB and ArrayStore with one project, one sample, one measurement."""
    project_id = new_id()
    sample_id = new_id()
    measurement_id = new_id()

    file_ref = FileRef(
        path=Path("/data/x.xlsx"),
        sha256="a" * 64,
        size_bytes=100,
        role=FileRole.RAW,
        scanned_at=utc_now(),
    )
    m = Measurement(
        id=measurement_id,
        sample_id=sample_id,
        technique=technique,
        instrument="UV-DRS",
        measured_at=utc_now(),
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(file_ref,),
    )
    s = Sample(
        id=sample_id,
        project_id=project_id,
        canonical_name="CS",
        measurements=(m,),
    )
    p = Project(
        id=project_id,
        name="Test",
        root_path=Path("/data"),
        created_at=utc_now(),
        schema_version=3,
        samples=(s,),
    )
    repo.save(p)

    if arrays is not None:
        # ArrayStore.write needs a ParsedData; build a minimal one.
        # Use pyarrow directly here to bypass the ParsedData hop.
        table = pa.table({name: pa.array(arr) for name, arr in arrays.items()})
        # ParquetWriter writes to `<id>.parquet` directly.
        target = array_store.directory / f"{measurement_id}.parquet"
        pq.write_table(table, target)  # type: ignore[no-untyped-call]

    return p, s, m


# ─── Stub analyzers ─────────────────────────────────────────────────
class _DoublingAnalyzer(BaseAnalyzer):
    """Multiplies the input `x` array by 2 and reports its mean.

    Used to assert input → output flow, cache behaviour, and that
    derived arrays get written to Parquet.
    """

    name: ClassVar[str] = "doubler"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)
    default_params: ClassVar[dict[str, Any]] = {"multiplier": 2.0}

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        multiplier = float(inputs.params.get("multiplier", 2.0))
        x = inputs.arrays.get("x", np.array([], dtype=np.float64))
        scaled = x * multiplier
        return AnalyzerOutput(
            outputs={"mean": float(np.mean(scaled)) if scaled.size else 0.0},
            derived_arrays={"scaled": scaled},
        )


class _ScalarOnlyAnalyzer(BaseAnalyzer):
    """Returns a scalar outputs dict and no derived arrays."""

    name: ClassVar[str] = "scalar-only"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput(outputs={"answer": 42})


class _WrongTechniqueAnalyzer(BaseAnalyzer):
    name: ClassVar[str] = "xrd-only"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.XRD,)

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput()


class _RejectingAnalyzer(BaseAnalyzer):
    name: ClassVar[str] = "always-no"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

    def accepts(self, measurement: Measurement) -> bool:
        return False

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput()


# ─── Fingerprint helper ─────────────────────────────────────────────
class TestFingerprint:
    def test_same_dict_same_hash(self) -> None:
        assert _fingerprint({"a": 1, "b": 2}) == _fingerprint({"a": 1, "b": 2})

    def test_key_order_does_not_matter(self) -> None:
        assert _fingerprint({"a": 1, "b": 2}) == _fingerprint({"b": 2, "a": 1})

    def test_different_values_differ(self) -> None:
        assert _fingerprint({"a": 1}) != _fingerprint({"a": 2})


# ─── Service: basic run ─────────────────────────────────────────────
class TestRun:
    def test_run_persists_analysis_result(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        arrays = {"x": np.array([1.0, 2.0, 3.0])}
        _, _, m = _make_project_with_measurement(repo, array_store, arrays=arrays)
        analyzer = _DoublingAnalyzer()

        outcome = service.run(analyzer, m)

        assert outcome.from_cache is False
        assert outcome.result.analyzer_name == "doubler"
        assert outcome.result.outputs["mean"] == pytest.approx(4.0)
        # The persisted project now carries the AnalysisResult.
        loaded = repo.load_first()
        assert loaded is not None
        loaded_m = loaded.samples[0].measurements[0]
        assert len(loaded_m.analysis_results) == 1
        assert loaded_m.analysis_results[0].outputs["mean"] == pytest.approx(4.0)

    def test_run_writes_derived_arrays_to_parquet(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        arrays = {"x": np.array([1.0, 2.0, 3.0])}
        _, _, m = _make_project_with_measurement(repo, array_store, arrays=arrays)
        outcome = service.run(_DoublingAnalyzer(), m)
        # Path stamped on the result.
        assert outcome.result.derived_arrays_path is not None
        assert outcome.result.derived_arrays_path.is_file()
        assert outcome.result.derived_arrays_path.suffix == ".parquet"
        # Round-trip the file: the `scaled` array should be 2 * input.
        table = pq.read_table(outcome.result.derived_arrays_path)  # type: ignore[no-untyped-call]
        scaled = np.asarray(table.column("scaled").to_numpy(zero_copy_only=False))
        np.testing.assert_array_equal(scaled, np.array([2.0, 4.0, 6.0]))

    def test_run_scalar_only_no_parquet_file(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        _, _, m = _make_project_with_measurement(repo, array_store)
        outcome = service.run(_ScalarOnlyAnalyzer(), m)
        assert outcome.result.derived_arrays_path is None
        # No file should have been created for derived arrays.
        files = list(array_store.directory.glob(f"{m.id}.scalar-only.*.parquet"))
        assert files == []


# ─── Service: cache behaviour ───────────────────────────────────────
class TestCache:
    def test_second_run_hits_cache(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        arrays = {"x": np.array([1.0, 2.0, 3.0])}
        _, _, m = _make_project_with_measurement(repo, array_store, arrays=arrays)
        analyzer = _DoublingAnalyzer()

        first = service.run(analyzer, m)
        # Re-load the measurement so its in-memory copy carries the
        # AnalysisResult that the service persisted.
        loaded_project = repo.load_first()
        assert loaded_project is not None
        m_reloaded = loaded_project.samples[0].measurements[0]
        second = service.run(analyzer, m_reloaded)
        assert second.from_cache is True
        assert second.result.id == first.result.id

    def test_different_params_miss_cache(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        arrays = {"x": np.array([1.0, 2.0, 3.0])}
        _, _, m = _make_project_with_measurement(repo, array_store, arrays=arrays)
        analyzer = _DoublingAnalyzer()

        service.run(analyzer, m, params={"multiplier": 2.0})
        loaded_project = repo.load_first()
        assert loaded_project is not None
        m_reloaded = loaded_project.samples[0].measurements[0]
        second = service.run(analyzer, m_reloaded, params={"multiplier": 3.0})
        assert second.from_cache is False
        assert second.result.outputs["mean"] == pytest.approx(6.0)

    def test_force_bypasses_cache(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        arrays = {"x": np.array([1.0, 2.0, 3.0])}
        _, _, m = _make_project_with_measurement(repo, array_store, arrays=arrays)
        analyzer = _DoublingAnalyzer()

        first = service.run(analyzer, m)
        loaded_project = repo.load_first()
        assert loaded_project is not None
        m_reloaded = loaded_project.samples[0].measurements[0]
        second = service.run(analyzer, m_reloaded, force=True)
        assert second.from_cache is False
        # A fresh ID was minted.
        assert second.result.id != first.result.id

    def test_rerun_replaces_prior_same_key_result(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        """Re-running with same params (and force=True) shouldn't pile up rows."""
        arrays = {"x": np.array([1.0, 2.0, 3.0])}
        _, _, m = _make_project_with_measurement(repo, array_store, arrays=arrays)
        analyzer = _DoublingAnalyzer()

        service.run(analyzer, m)
        loaded_p = repo.load_first()
        assert loaded_p is not None
        m_reloaded = loaded_p.samples[0].measurements[0]
        service.run(analyzer, m_reloaded, force=True)

        # The measurement should hold exactly one analysis_result for
        # this (analyzer_name, params_fp).
        loaded_p2 = repo.load_first()
        assert loaded_p2 is not None
        final = loaded_p2.samples[0].measurements[0]
        assert len(final.analysis_results) == 1


# ─── Service: rejection paths ───────────────────────────────────────
class TestRejection:
    def test_wrong_technique_raises(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        _, _, m = _make_project_with_measurement(repo, array_store)
        with pytest.raises(AnalysisError, match="does not accept technique"):
            service.run(_WrongTechniqueAnalyzer(), m)

    def test_accepts_returning_false_raises(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        _, _, m = _make_project_with_measurement(repo, array_store)
        with pytest.raises(AnalysisError, match="rejected measurement"):
            service.run(_RejectingAnalyzer(), m)


# ─── Service: analyzer-issue preservation ───────────────────────────
class TestIssuePreservation:
    def test_analyzer_issues_round_trip(
        self,
        repo: ProjectRepository,
        array_store: ArrayStore,
        service: AnalysisService,
    ) -> None:
        class _Noisy(BaseAnalyzer):
            name: ClassVar[str] = "noisy"
            version: ClassVar[str] = "1.0.0"
            accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)

            def accepts(self, measurement: Measurement) -> bool:
                return True

            def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
                return AnalyzerOutput(
                    outputs={"x": 1.0},
                    issues=(
                        ValidationIssue(
                            field="x",
                            severity=Severity.WARNING,
                            message="x is suspicious",
                            detected_at=utc_now(),
                        ),
                    ),
                )

        _, _, m = _make_project_with_measurement(repo, array_store)
        outcome = service.run(_Noisy(), m)
        # Both in-memory and round-tripped from DB should carry the issue.
        assert len(outcome.result.issues) == 1
        loaded_p = repo.load_first()
        assert loaded_p is not None
        loaded = loaded_p.samples[0].measurements[0]
        assert len(loaded.analysis_results[0].issues) == 1
        assert loaded.analysis_results[0].issues[0].severity is Severity.WARNING
