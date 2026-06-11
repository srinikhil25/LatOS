"""Tests for `latos.core.models` — Project, Sample, Measurement, FileRef."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from latos.core.enums import FileRole, Severity, Technique
from latos.core.exceptions import ValidationError
from latos.core.models import (
    AnalysisResult,
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
    new_id,
    utc_now,
)


# ─── Helpers / fixtures ─────────────────────────────────────────────
def _file_ref(
    *,
    path: Path | None = None,
    sha256: str | None = None,
    size_bytes: int = 1024,
    role: FileRole = FileRole.RAW,
) -> FileRef:
    # Use explicit None check so an empty-string sha256 still hits the validator.
    return FileRef(
        path=path if path is not None else Path("/data/sample.xy"),
        sha256=sha256 if sha256 is not None else "a" * 64,
        size_bytes=size_bytes,
        role=role,
        scanned_at=utc_now(),
    )


def _measurement(
    *,
    sample_id: str,
    technique: Technique = Technique.XRD,
    issues: tuple[ValidationIssue, ...] = (),
) -> Measurement:
    return Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=technique,
        instrument="JEOL JEM-2100F",
        measured_at=utc_now(),
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(_file_ref(),),
        issues=issues,
    )


def _sample(
    project_id: str,
    name: str = "CS",
    aliases: tuple[str, ...] = (),
    measurements: tuple[Measurement, ...] = (),
) -> Sample:
    return Sample(
        id=new_id(),
        project_id=project_id,
        canonical_name=name,
        aliases=aliases,
        measurements=measurements,
    )


def _project(
    *,
    samples: tuple[Sample, ...] = (),
    unassigned: tuple[FileRef, ...] = (),
) -> Project:
    return Project(
        id=new_id(),
        name="Demo",
        root_path=Path("/data/demo"),
        created_at=utc_now(),
        schema_version=1,
        samples=samples,
        unassigned_files=unassigned,
    )


# ─── new_id / utc_now ───────────────────────────────────────────────
class TestIdHelpers:
    def test_new_id_format(self) -> None:
        for _ in range(50):
            uid = new_id()
            assert isinstance(uid, str)
            assert len(uid) == 32
            assert all(c in "0123456789abcdef" for c in uid)

    def test_new_id_uniqueness(self) -> None:
        ids = {new_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_utc_now_is_timezone_aware(self) -> None:
        ts = utc_now()
        assert ts.tzinfo is not None
        assert ts.utcoffset() == timedelta(0)


# ─── FileRef ────────────────────────────────────────────────────────
class TestFileRef:
    def test_constructs_with_valid_inputs(self) -> None:
        ref = _file_ref()
        assert isinstance(ref.path, Path)
        assert ref.size_bytes == 1024

    def test_immutable(self) -> None:
        ref = _file_ref()
        with pytest.raises((AttributeError, TypeError)):
            ref.size_bytes = 2048  # type: ignore[misc]

    def test_path_must_be_pathlib(self) -> None:
        with pytest.raises(ValidationError):
            FileRef(
                path="/tmp/x.xy",  # type: ignore[arg-type]
                sha256="a" * 64,
                size_bytes=1,
                role=FileRole.RAW,
                scanned_at=utc_now(),
            )

    @pytest.mark.parametrize("bad", ["", "abc", "g" * 64, "A" * 64, "a" * 63])
    def test_sha256_validation(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _file_ref(sha256=bad)

    def test_size_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            _file_ref(size_bytes=-1)

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FileRef(
                path=Path("/x"),
                sha256="a" * 64,
                size_bytes=1,
                role=FileRole.RAW,
                scanned_at=datetime.now(),
            )


# ─── ValidationIssue ────────────────────────────────────────────────
class TestValidationIssue:
    def test_constructs_with_valid_inputs(self) -> None:
        issue = ValidationIssue(
            field="zT",
            severity=Severity.WARNING,
            message="zT out of range",
            detected_at=utc_now(),
        )
        assert not issue.acknowledged

    def test_empty_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ValidationIssue(
                field="",
                severity=Severity.INFO,
                message="x",
                detected_at=utc_now(),
            )

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ValidationIssue(
                field="zT",
                severity=Severity.INFO,
                message="",
                detected_at=utc_now(),
            )

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ValidationIssue(
                field="zT",
                severity=Severity.INFO,
                message="x",
                detected_at=datetime.now(),
            )


# ─── Measurement ────────────────────────────────────────────────────
class TestMeasurement:
    def test_constructs_with_valid_inputs(self) -> None:
        sid = new_id()
        m = _measurement(sample_id=sid)
        assert m.sample_id == sid
        assert m.technique is Technique.XRD

    def test_invalid_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Measurement(
                id="not-a-uuid",
                sample_id=new_id(),
                technique=Technique.XRD,
                instrument=None,
                measured_at=None,
                parsed_at=utc_now(),
                parser_version="1.0",
                files=(),
            )

    def test_invalid_sample_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Measurement(
                id=new_id(),
                sample_id="bad",
                technique=Technique.XRD,
                instrument=None,
                measured_at=None,
                parsed_at=utc_now(),
                parser_version="1.0",
                files=(),
            )

    def test_empty_parser_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Measurement(
                id=new_id(),
                sample_id=new_id(),
                technique=Technique.XRD,
                instrument=None,
                measured_at=None,
                parsed_at=utc_now(),
                parser_version="",
                files=(),
            )

    def test_files_must_be_tuple(self) -> None:
        with pytest.raises(ValidationError):
            Measurement(
                id=new_id(),
                sample_id=new_id(),
                technique=Technique.XRD,
                instrument=None,
                measured_at=None,
                parsed_at=utc_now(),
                parser_version="1.0",
                files=[_file_ref()],  # type: ignore[arg-type] -- list, not tuple
            )

    def test_has_errors_true_when_error_issue(self) -> None:
        sid = new_id()
        err = ValidationIssue(
            field="zT", severity=Severity.ERROR, message="x", detected_at=utc_now()
        )
        m = _measurement(sample_id=sid, issues=(err,))
        assert m.has_errors
        assert m.has_warnings

    def test_has_errors_false_when_only_warning(self) -> None:
        sid = new_id()
        warn = ValidationIssue(
            field="zT", severity=Severity.WARNING, message="x", detected_at=utc_now()
        )
        m = _measurement(sample_id=sid, issues=(warn,))
        assert not m.has_errors
        assert m.has_warnings

    def test_has_warnings_false_when_only_info(self) -> None:
        sid = new_id()
        info = ValidationIssue(
            field="zT", severity=Severity.INFO, message="x", detected_at=utc_now()
        )
        m = _measurement(sample_id=sid, issues=(info,))
        assert not m.has_errors
        assert not m.has_warnings


# ─── Sample ─────────────────────────────────────────────────────────
class TestSample:
    def test_constructs_with_valid_inputs(self) -> None:
        pid = new_id()
        s = _sample(pid)
        assert s.canonical_name == "CS"
        assert s.aliases == ()
        assert s.measurements == ()

    def test_blank_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _sample(new_id(), name="   ")

    def test_aliases_must_be_unique(self) -> None:
        with pytest.raises(ValidationError):
            _sample(new_id(), aliases=("CS", "CS"))

    def test_aliases_must_be_non_empty_strings(self) -> None:
        with pytest.raises(ValidationError):
            _sample(new_id(), aliases=("CS", ""))

    def test_measurements_must_belong_to_sample(self) -> None:
        pid = new_id()
        wrong_sample_id = new_id()
        bad_meas = _measurement(sample_id=wrong_sample_id)
        with pytest.raises(ValidationError):
            Sample(
                id=new_id(),
                project_id=pid,
                canonical_name="CS",
                measurements=(bad_meas,),
            )

    def test_techniques_property(self) -> None:
        pid = new_id()
        sid = new_id()
        m1 = Measurement(
            id=new_id(),
            sample_id=sid,
            technique=Technique.XRD,
            instrument=None,
            measured_at=None,
            parsed_at=utc_now(),
            parser_version="1.0",
            files=(),
        )
        m2 = Measurement(
            id=new_id(),
            sample_id=sid,
            technique=Technique.XPS,
            instrument=None,
            measured_at=None,
            parsed_at=utc_now(),
            parser_version="1.0",
            files=(),
        )
        s = Sample(id=sid, project_id=pid, canonical_name="CS", measurements=(m1, m2))
        assert s.techniques == frozenset({Technique.XRD, Technique.XPS})

    def test_has_technique(self) -> None:
        pid = new_id()
        sid = new_id()
        m = Measurement(
            id=new_id(),
            sample_id=sid,
            technique=Technique.HALL,
            instrument=None,
            measured_at=None,
            parsed_at=utc_now(),
            parser_version="1.0",
            files=(),
        )
        s = Sample(id=sid, project_id=pid, canonical_name="CS", measurements=(m,))
        assert s.has_technique(Technique.HALL)
        assert not s.has_technique(Technique.RAMAN)

    def test_measurements_for_filters_correctly(self) -> None:
        pid = new_id()
        sid = new_id()
        ms = tuple(
            Measurement(
                id=new_id(),
                sample_id=sid,
                technique=t,
                instrument=None,
                measured_at=None,
                parsed_at=utc_now(),
                parser_version="1.0",
                files=(),
            )
            for t in (Technique.XRD, Technique.XRD, Technique.XPS)
        )
        s = Sample(id=sid, project_id=pid, canonical_name="CS", measurements=ms)
        assert len(s.measurements_for(Technique.XRD)) == 2
        assert len(s.measurements_for(Technique.XPS)) == 1
        assert len(s.measurements_for(Technique.RAMAN)) == 0


# ─── Project ────────────────────────────────────────────────────────
class TestProject:
    def test_constructs_with_valid_inputs(self) -> None:
        p = _project()
        assert p.name == "Demo"
        assert p.schema_version == 1

    def test_blank_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Project(
                id=new_id(),
                name="   ",
                root_path=Path("/x"),
                created_at=utc_now(),
                schema_version=1,
            )

    def test_path_must_be_pathlib(self) -> None:
        with pytest.raises(ValidationError):
            Project(
                id=new_id(),
                name="Demo",
                root_path="/x",  # type: ignore[arg-type]
                created_at=utc_now(),
                schema_version=1,
            )

    def test_naive_created_at_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Project(
                id=new_id(),
                name="Demo",
                root_path=Path("/x"),
                created_at=datetime.now(),
                schema_version=1,
            )

    @pytest.mark.parametrize("bad_version", [0, -1])
    def test_schema_version_must_be_positive(self, bad_version: int) -> None:
        with pytest.raises(ValidationError):
            Project(
                id=new_id(),
                name="Demo",
                root_path=Path("/x"),
                created_at=utc_now(),
                schema_version=bad_version,
            )

    def test_samples_must_belong_to_project(self) -> None:
        wrong_project_id = new_id()
        bad_sample = _sample(wrong_project_id)
        with pytest.raises(ValidationError):
            Project(
                id=new_id(),
                name="Demo",
                root_path=Path("/x"),
                created_at=utc_now(),
                schema_version=1,
                samples=(bad_sample,),
            )

    def test_total_files_counts_attributed_and_unassigned(self) -> None:
        pid = new_id()
        sid = new_id()
        m = Measurement(
            id=new_id(),
            sample_id=sid,
            technique=Technique.XRD,
            instrument=None,
            measured_at=None,
            parsed_at=utc_now(),
            parser_version="1.0",
            files=(_file_ref(sha256="a" * 64), _file_ref(sha256="b" * 64)),
        )
        s = Sample(id=sid, project_id=pid, canonical_name="CS", measurements=(m,))
        p = Project(
            id=pid,
            name="Demo",
            root_path=Path("/x"),
            created_at=utc_now(),
            schema_version=1,
            samples=(s,),
            unassigned_files=(_file_ref(sha256="c" * 64),),
        )
        assert p.total_files == 3
        assert p.total_measurements == 1

    def test_all_techniques_aggregates(self) -> None:
        pid = new_id()
        sid_a = new_id()
        sid_b = new_id()
        s_a = Sample(
            id=sid_a,
            project_id=pid,
            canonical_name="A",
            measurements=(
                Measurement(
                    id=new_id(),
                    sample_id=sid_a,
                    technique=Technique.XRD,
                    instrument=None,
                    measured_at=None,
                    parsed_at=utc_now(),
                    parser_version="1.0",
                    files=(),
                ),
            ),
        )
        s_b = Sample(
            id=sid_b,
            project_id=pid,
            canonical_name="B",
            measurements=(
                Measurement(
                    id=new_id(),
                    sample_id=sid_b,
                    technique=Technique.XPS,
                    instrument=None,
                    measured_at=None,
                    parsed_at=utc_now(),
                    parser_version="1.0",
                    files=(),
                ),
            ),
        )
        p = Project(
            id=pid,
            name="Demo",
            root_path=Path("/x"),
            created_at=utc_now(),
            schema_version=1,
            samples=(s_a, s_b),
        )
        assert p.all_techniques == frozenset({Technique.XRD, Technique.XPS})

    def test_lookup_by_id(self) -> None:
        pid = new_id()
        s = _sample(pid, name="CS")
        p = Project(
            id=pid,
            name="Demo",
            root_path=Path("/x"),
            created_at=utc_now(),
            schema_version=1,
            samples=(s,),
        )
        assert p.sample_by_id(s.id) is s
        assert p.sample_by_id(new_id()) is None
        assert p.sample_by_name("CS") is s
        assert p.sample_by_name("nope") is None

    def test_summary_shape(self) -> None:
        p = _project()
        summary = p.summary()
        assert set(summary.keys()) == {
            "id",
            "name",
            "root_path",
            "created_at",
            "schema_version",
            "n_samples",
            "n_measurements",
            "n_files",
            "n_unassigned",
            "techniques",
        }
        assert summary["n_samples"] == 0
        assert summary["techniques"] == []


# ─── Time invariance / timezone safety ──────────────────────────────
def test_all_timestamps_round_trip_with_utc() -> None:
    p = _project()
    assert p.created_at.utcoffset() == UTC.utcoffset(p.created_at)


# ─── AnalysisResult ─────────────────────────────────────────────────
def _analysis_result(
    *,
    measurement_id: str | None = None,
    analyzer_name: str = "uvdrs-tauc",
    analyzer_version: str = "1.0.0",
    params: dict | None = None,
    outputs: dict | None = None,
    issues: tuple[ValidationIssue, ...] = (),
) -> AnalysisResult:
    return AnalysisResult(
        id=new_id(),
        measurement_id=measurement_id if measurement_id is not None else new_id(),
        analyzer_name=analyzer_name,
        analyzer_version=analyzer_version,
        params=params if params is not None else {"band_gap_type": "direct"},
        outputs=outputs if outputs is not None else {"band_gap_ev": 2.05},
        issues=issues,
    )


class TestAnalysisResult:
    def test_constructs_with_valid_inputs(self) -> None:
        r = _analysis_result()
        assert r.analyzer_name == "uvdrs-tauc"
        assert r.outputs == {"band_gap_ev": 2.05}
        assert r.has_errors is False
        assert r.derived_arrays_path is None

    def test_immutable(self) -> None:
        r = _analysis_result()
        with pytest.raises((AttributeError, TypeError)):
            r.analyzer_name = "other"  # type: ignore[misc]

    def test_empty_analyzer_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _analysis_result(analyzer_name="")

    def test_empty_analyzer_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _analysis_result(analyzer_version="")

    def test_invalid_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(
                id="not-a-uuid",
                measurement_id=new_id(),
                analyzer_name="x",
                analyzer_version="1",
            )

    def test_naive_computed_at_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(
                id=new_id(),
                measurement_id=new_id(),
                analyzer_name="x",
                analyzer_version="1",
                computed_at=datetime.now(),  # naive
            )

    def test_params_must_be_dict(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(
                id=new_id(),
                measurement_id=new_id(),
                analyzer_name="x",
                analyzer_version="1",
                params=[("k", "v")],  # type: ignore[arg-type]
            )

    def test_outputs_must_be_dict(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(
                id=new_id(),
                measurement_id=new_id(),
                analyzer_name="x",
                analyzer_version="1",
                outputs="not a dict",  # type: ignore[arg-type]
            )

    def test_issues_must_be_tuple(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(
                id=new_id(),
                measurement_id=new_id(),
                analyzer_name="x",
                analyzer_version="1",
                issues=[],  # type: ignore[arg-type]
            )

    def test_has_errors_true_when_any_issue_is_error(self) -> None:
        issue_e = ValidationIssue(
            field="band_gap_ev",
            severity=Severity.ERROR,
            message="negative",
            detected_at=utc_now(),
        )
        issue_w = ValidationIssue(
            field="band_gap_ev",
            severity=Severity.WARNING,
            message="extrapolated",
            detected_at=utc_now(),
        )
        r = _analysis_result(issues=(issue_w, issue_e))
        assert r.has_errors is True

    def test_has_errors_false_for_warnings_only(self) -> None:
        issue_w = ValidationIssue(
            field="band_gap_ev",
            severity=Severity.WARNING,
            message="extrapolated",
            detected_at=utc_now(),
        )
        r = _analysis_result(issues=(issue_w,))
        assert r.has_errors is False


class TestMeasurementAnalysisResults:
    def test_default_is_empty_tuple(self) -> None:
        sid = new_id()
        m = _measurement(sample_id=sid)
        assert m.analysis_results == ()

    def test_accepts_results(self) -> None:
        sid = new_id()
        m = Measurement(
            id=new_id(),
            sample_id=sid,
            technique=Technique.UV_DRS,
            instrument=None,
            measured_at=utc_now(),
            parsed_at=utc_now(),
            parser_version="1.0.0",
            files=(_file_ref(),),
            analysis_results=(_analysis_result(), _analysis_result()),
        )
        assert len(m.analysis_results) == 2

    def test_results_must_be_tuple(self) -> None:
        sid = new_id()
        with pytest.raises(ValidationError):
            Measurement(
                id=new_id(),
                sample_id=sid,
                technique=Technique.UV_DRS,
                instrument=None,
                measured_at=utc_now(),
                parsed_at=utc_now(),
                parser_version="1.0.0",
                files=(_file_ref(),),
                analysis_results=[_analysis_result()],  # type: ignore[arg-type]
            )
