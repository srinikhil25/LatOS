"""Tests for `latos.persistence.mappers` — domain ↔ ORM round-trips."""

from __future__ import annotations

from pathlib import Path

from latos.core.enums import FileRole, Severity, Technique
from latos.core.models import (
    AnalysisResult,
    FileRef,
    Measurement,
    Sample,
    ValidationIssue,
    new_id,
    utc_now,
)
from latos.persistence.mappers import (
    analysis_result_to_row,
    file_to_row,
    issue_to_row,
    measurement_to_row,
    project_to_row,
    row_to_analysis_result,
    row_to_file_ref,
    row_to_issue,
    row_to_measurement,
    row_to_project,
    row_to_sample,
    sample_to_row,
)
from latos.persistence.schema import (
    AnalysisResultRow,
    FileRow,
    MeasurementRow,
    ProjectRow,
    SampleRow,
    ValidationIssueRow,
)

from .conftest import make_file_ref, make_issue, make_measurement, make_project, make_sample


# ─── Project ────────────────────────────────────────────────────────
class TestProjectMapper:
    def test_to_row_preserves_fields(self) -> None:
        p = make_project()
        row = project_to_row(p)
        assert isinstance(row, ProjectRow)
        assert row.id == p.id
        assert row.name == p.name
        assert row.root_path == str(p.root_path)
        assert row.created_at == p.created_at
        assert row.schema_version == p.schema_version

    def test_path_serialized_as_string(self) -> None:
        p = make_project()
        row = project_to_row(p)
        assert isinstance(row.root_path, str)


# ─── Sample ─────────────────────────────────────────────────────────
class TestSampleMapper:
    def test_to_row_preserves_fields(self) -> None:
        pid = new_id()
        s = make_sample(pid, name="CS", aliases=("cs", "CS Pure"))
        row = sample_to_row(s)
        assert isinstance(row, SampleRow)
        assert row.id == s.id
        assert row.project_id == pid
        assert row.canonical_name == "CS"
        assert row.aliases == ["cs", "CS Pure"]  # JSON column = list

    def test_aliases_tuple_becomes_list(self) -> None:
        s = make_sample(new_id(), aliases=("a", "b"))
        row = sample_to_row(s)
        assert isinstance(row.aliases, list)


# ─── Measurement ────────────────────────────────────────────────────
class TestMeasurementMapper:
    def test_to_row_preserves_fields(self) -> None:
        sid = new_id()
        m = make_measurement(sid, technique=Technique.XPS, instrument="PHI-VS")
        row = measurement_to_row(m)
        assert isinstance(row, MeasurementRow)
        assert row.sample_id == sid
        assert row.technique == "xps"  # stored as string
        assert row.instrument == "PHI-VS"
        assert row.parser_version == "1.0.0"

    def test_optional_fields(self) -> None:
        m = Measurement(
            id=new_id(),
            sample_id=new_id(),
            technique=Technique.HALL,
            instrument=None,
            measured_at=None,
            parsed_at=utc_now(),
            parser_version="1.0",
            files=(),
        )
        row = measurement_to_row(m)
        assert row.instrument is None
        assert row.measured_at is None
        assert row.parsed_data_path is None


# ─── FileRef ────────────────────────────────────────────────────────
class TestFileRefMapper:
    def test_to_row_with_measurement(self) -> None:
        ref = make_file_ref()
        row = file_to_row(ref, project_id=new_id(), measurement_id=new_id())
        assert isinstance(row, FileRow)
        # str(Path) is platform-dependent (POSIX vs Windows separators); compare
        # against str(ref.path) so this works on both.
        assert row.path == str(ref.path)
        assert row.sha256 == ref.sha256
        assert row.role == "raw"
        assert row.measurement_id is not None

    def test_to_row_unassigned(self) -> None:
        ref = make_file_ref()
        row = file_to_row(ref, project_id=new_id(), measurement_id=None)
        assert row.measurement_id is None


# ─── ValidationIssue ────────────────────────────────────────────────
class TestIssueMapper:
    def test_to_row_preserves_fields(self) -> None:
        issue = make_issue()
        mid = new_id()
        iid = new_id()
        row = issue_to_row(issue, issue_id=iid, measurement_id=mid)
        assert isinstance(row, ValidationIssueRow)
        assert row.id == iid
        assert row.measurement_id == mid
        assert row.field == issue.field
        assert row.severity == "warning"  # enum-as-string
        assert row.message == issue.message
        assert row.acknowledged is False


# ─── Round-trip via plain row construction ──────────────────────────
class TestRoundTrip:
    """ORM rows constructed manually then mapped back must equal the originals."""

    def test_file_ref_round_trip(self) -> None:
        ref = make_file_ref()
        row = file_to_row(ref, project_id=new_id(), measurement_id=new_id())
        # Simulate what comes back from the DB:
        recovered = row_to_file_ref(row)
        assert recovered == ref

    def test_issue_round_trip(self) -> None:
        issue = make_issue(severity=Severity.ERROR)
        row = issue_to_row(issue, issue_id=new_id(), measurement_id=new_id())
        recovered = row_to_issue(row)
        assert recovered == issue

    def test_measurement_round_trip(self) -> None:
        sid = new_id()
        original = make_measurement(sid, technique=Technique.UV_DRS)
        row = measurement_to_row(original)
        # Simulate loaded relationships
        row.files = [
            file_to_row(f, project_id=new_id(), measurement_id=original.id) for f in original.files
        ]
        row.issues = []
        recovered = row_to_measurement(row)
        assert recovered.id == original.id
        assert recovered.technique is Technique.UV_DRS
        assert recovered.files == original.files


def test_row_to_project_with_empty_aggregate() -> None:
    """Mapping an empty project (no samples) must not crash."""
    p = make_project()
    row = project_to_row(p)
    row.samples = []
    row.unassigned_files = []
    recovered = row_to_project(row)
    assert recovered.id == p.id
    assert recovered.samples == ()


def test_row_to_sample_propagates_measurements() -> None:
    pid = new_id()
    sid = new_id()
    s = Sample(
        id=sid,
        project_id=pid,
        canonical_name="CS",
        measurements=(make_measurement(sid, technique=Technique.XRD),),
    )
    row = sample_to_row(s)
    # Hand-build the relationship the way the ORM would
    m = s.measurements[0]
    m_row = measurement_to_row(m)
    m_row.files = [file_to_row(f, project_id=pid, measurement_id=m.id) for f in m.files]
    m_row.issues = []
    row.measurements = [m_row]
    recovered = row_to_sample(row)
    assert recovered.canonical_name == "CS"
    assert len(recovered.measurements) == 1
    assert recovered.measurements[0].technique is Technique.XRD


# ─── AnalysisResult ─────────────────────────────────────────────────
def _make_analysis_result(
    *,
    measurement_id: str | None = None,
    issues: tuple[ValidationIssue, ...] = (),
    derived_arrays_path: Path | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        id=new_id(),
        measurement_id=measurement_id if measurement_id is not None else new_id(),
        analyzer_name="uvdrs-tauc",
        analyzer_version="1.0.0",
        params={"band_gap_type": "direct", "fit_range_ev": [1.5, 3.0]},
        outputs={"band_gap_ev": 2.05, "r_squared": 0.998},
        derived_arrays_path=derived_arrays_path,
        issues=issues,
    )


class TestAnalysisResultMapper:
    def test_to_row_preserves_fields(self) -> None:
        mid = new_id()
        r = _make_analysis_result(measurement_id=mid)
        row = analysis_result_to_row(r)
        assert isinstance(row, AnalysisResultRow)
        assert row.id == r.id
        assert row.measurement_id == mid
        assert row.analyzer_name == "uvdrs-tauc"
        assert row.analyzer_version == "1.0.0"
        assert row.params["band_gap_type"] == "direct"
        assert row.outputs["band_gap_ev"] == 2.05
        assert row.derived_arrays_path is None
        assert row.issues_json == []
        assert row.computed_at == r.computed_at

    def test_to_row_serializes_path(self) -> None:
        r = _make_analysis_result(derived_arrays_path=Path("/arrays/foo.parquet"))
        row = analysis_result_to_row(r)
        assert isinstance(row.derived_arrays_path, str)
        assert row.derived_arrays_path == str(Path("/arrays/foo.parquet"))

    def test_to_row_serializes_issues_into_json(self) -> None:
        issue = make_issue(severity=Severity.ERROR)
        r = _make_analysis_result(issues=(issue,))
        row = analysis_result_to_row(r)
        assert len(row.issues_json) == 1
        payload = row.issues_json[0]
        assert payload["field"] == issue.field
        assert payload["severity"] == "error"
        assert payload["message"] == issue.message
        # detected_at is serialized as ISO 8601
        assert isinstance(payload["detected_at"], str)


class TestAnalysisResultRoundTrip:
    def test_round_trip_minimal(self) -> None:
        original = _make_analysis_result()
        row = analysis_result_to_row(original)
        recovered = row_to_analysis_result(row)
        assert recovered == original

    def test_round_trip_with_path_and_issues(self) -> None:
        issue = make_issue(severity=Severity.WARNING)
        original = _make_analysis_result(
            derived_arrays_path=Path("/arrays/tauc.parquet"),
            issues=(issue,),
        )
        row = analysis_result_to_row(original)
        recovered = row_to_analysis_result(row)
        assert recovered == original
        assert isinstance(recovered.derived_arrays_path, Path)
        assert recovered.issues[0].severity is Severity.WARNING


class TestMeasurementWithAnalysisResults:
    def test_measurement_round_trip_with_analysis_results(self) -> None:
        sid = new_id()
        m = make_measurement(sid, technique=Technique.UV_DRS)
        # Attach an AnalysisResult on a fresh Measurement (frozen dataclass).
        result = _make_analysis_result(measurement_id=m.id)
        m_with = Measurement(
            id=m.id,
            sample_id=m.sample_id,
            technique=m.technique,
            instrument=m.instrument,
            measured_at=m.measured_at,
            parsed_at=m.parsed_at,
            parser_version=m.parser_version,
            files=m.files,
            issues=m.issues,
            parsed_data_path=m.parsed_data_path,
            analysis_results=(result,),
        )
        row = measurement_to_row(m_with)
        # Simulate loaded relationships (the ORM does this normally).
        row.files = [file_to_row(f, project_id=new_id(), measurement_id=m.id) for f in m_with.files]
        row.issues = []
        row.analysis_results = [analysis_result_to_row(result)]
        recovered = row_to_measurement(row)
        assert len(recovered.analysis_results) == 1
        assert recovered.analysis_results[0] == result


def test_path_round_trips_as_pathlib() -> None:
    """str(Path) on save, Path(str) on load — type must come back as Path."""
    ref = FileRef(
        path=Path("D:/data/foo.xy"),
        sha256="0" * 64,
        size_bytes=1,
        role=FileRole.PROCESSED,
        scanned_at=utc_now(),
    )
    row = file_to_row(ref, project_id=new_id(), measurement_id=None)
    recovered = row_to_file_ref(row)
    assert isinstance(recovered.path, Path)
    assert recovered.role is FileRole.PROCESSED
