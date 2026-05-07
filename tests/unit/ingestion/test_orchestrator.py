"""Tests for `latos.ingestion.orchestrator.Orchestrator`."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from latos.core.enums import Severity, Technique
from latos.core.models import (
    ValidationIssue,
    utc_now,
)
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.orchestrator import Orchestrator, Outcome
from latos.ingestion.parsed_data import ParsedData
from latos.ingestion.registry import ParserRegistry

# ─── Fixture paths ──────────────────────────────────────────────────
_PROJ_FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "parsers"
_RIGAKU_TXT = _PROJ_FIXTURES / "xrd" / "rigaku_bs3a.txt"
_PANALYTICAL = _PROJ_FIXTURES / "xrd" / "panalytical_cscbi1.xrdml"
_XPS_CSV = _PROJ_FIXTURES / "xps" / "casaxps_c1s.csv"


# ─── Helper: synthetic parser for controlled tests ──────────────────
class _FakeParser(BaseParser):
    """Synthetic parser used for controlled orchestrator tests.

    `arrays_to_emit` lets each test customize the output without depending
    on a real fixture file — keeps these tests independent of parser logic.
    """

    name = "fake-parser"
    version = "1.0.0"
    technique = Technique.XRD
    supported_extensions = (".fake",)

    def __init__(
        self,
        *,
        arrays: dict[str, np.ndarray] | None = None,
        emit_error: bool = False,
        raise_on_parse: bool = False,
    ) -> None:
        self._arrays = arrays if arrays is not None else {"x": np.array([1.0, 2.0])}
        self._emit_error = emit_error
        self._raise_on_parse = raise_on_parse

    def can_parse(self, path: Path) -> float:
        return 1.0 if path.suffix == ".fake" else 0.0

    def parse(self, path: Path) -> ParsedData:
        if self._raise_on_parse:
            raise RuntimeError("simulated parser crash")
        issues: tuple[ValidationIssue, ...] = ()
        if self._emit_error:
            issues = (
                ValidationIssue(
                    field="data",
                    severity=Severity.ERROR,
                    message="synthetic error",
                    detected_at=utc_now(),
                ),
            )
        return ParsedData(
            technique=self.technique,
            arrays=self._arrays,
            metadata={"path": str(path)},
            instrument="FakeInstr",
            measured_at=None,
            issues=issues,
            parser_name=self.name,
            parser_version=self.version,
        )


def _registry_with(*parsers: BaseParser) -> ParserRegistry:
    return ParserRegistry(parsers)


def _orchestrator(registry: ParserRegistry) -> Orchestrator:
    return Orchestrator(registry=registry)


# ─── Empty / missing root ───────────────────────────────────────────
class TestRootValidation:
    def test_missing_root_raises(self, tmp_path: Path):
        o = _orchestrator(_registry_with(_FakeParser()))
        with pytest.raises(NotADirectoryError):
            o.ingest(tmp_path / "does-not-exist")

    def test_root_is_a_file_raises(self, tmp_path: Path):
        f = tmp_path / "afile.txt"
        f.write_text("hi")
        o = _orchestrator(_registry_with(_FakeParser()))
        with pytest.raises(NotADirectoryError):
            o.ingest(f)


# ─── Empty folder ───────────────────────────────────────────────────
class TestEmptyFolder:
    def test_empty_folder_creates_empty_project(self, tmp_path: Path):
        o = _orchestrator(_registry_with(_FakeParser()))
        result = o.ingest(tmp_path)

        assert result.project.samples == ()
        assert result.outcomes == ()
        assert result.parsed_count == 0
        assert result.failed_count == 0
        assert result.unclassified_count == 0

    def test_empty_folder_creates_latos_dir(self, tmp_path: Path):
        o = _orchestrator(_registry_with(_FakeParser()))
        o.ingest(tmp_path)
        assert (tmp_path / ".latos").is_dir()
        assert (tmp_path / ".latos" / "arrays").is_dir()


# ─── Single classified file ─────────────────────────────────────────
class TestSingleFile:
    def test_one_file_in_named_folder(self, tmp_path: Path):
        # Folder name is NOT generic → used as sample name.
        sample_dir = tmp_path / "MX-12"
        sample_dir.mkdir()
        (sample_dir / "scan.fake").write_text("payload")

        o = _orchestrator(_registry_with(_FakeParser()))
        result = o.ingest(tmp_path)

        assert len(result.project.samples) == 1
        sample = result.project.samples[0]
        assert sample.canonical_name == "MX-12"
        assert len(sample.measurements) == 1
        assert sample.measurements[0].technique is Technique.XRD
        assert result.parsed_count == 1

    def test_parquet_written_for_measurement(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "data.fake").write_text("payload")

        o = _orchestrator(_registry_with(_FakeParser()))
        result = o.ingest(tmp_path)

        meas = result.project.samples[0].measurements[0]
        assert meas.parsed_data_path is not None
        assert meas.parsed_data_path.exists()
        assert meas.parsed_data_path.parent == tmp_path / ".latos" / "arrays"

    def test_file_ref_carries_sha256_and_size(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        f = d / "data.fake"
        f.write_text("hello world")

        o = _orchestrator(_registry_with(_FakeParser()))
        result = o.ingest(tmp_path)

        file_ref = result.project.samples[0].measurements[0].files[0]
        assert file_ref.path == f
        assert len(file_ref.sha256) == 64
        assert file_ref.size_bytes == len(b"hello world")


# ─── Sample-name inference ──────────────────────────────────────────
class TestSampleInference:
    def test_non_generic_parent_folder_used(self, tmp_path: Path):
        d = tmp_path / "MyAwesomeSample"
        d.mkdir()
        (d / "x.fake").write_text("hi")
        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        assert result.project.samples[0].canonical_name == "MyAwesomeSample"

    def test_walks_up_through_generic_parent(self, tmp_path: Path):
        # Structure: <root>/SampleA/XRD/scan.fake
        # Parent "XRD" is generic → walk up → use "SampleA".
        d = tmp_path / "SampleA" / "XRD"
        d.mkdir(parents=True)
        (d / "scan.fake").write_text("hi")
        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        assert result.project.samples[0].canonical_name == "SampleA"

    def test_falls_back_to_filename_stem_when_all_parents_generic(self, tmp_path: Path):
        # Structure: <root>/Data/XRD/Raw/scan.fake — all 3 levels generic.
        d = tmp_path / "Data" / "XRD" / "Raw"
        d.mkdir(parents=True)
        (d / "MX-7-scan.fake").write_text("hi")

        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        sample = result.project.samples[0]
        assert sample.canonical_name == "MX-7-scan"
        # Warning issue attached to the measurement.
        meas = sample.measurements[0]
        warnings = [i for i in meas.issues if i.field == "sample_name"]
        assert len(warnings) == 1
        assert warnings[0].severity is Severity.WARNING

    def test_two_files_in_same_sample_folder_merge(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "a.fake").write_text("aa")
        (d / "b.fake").write_text("bb")

        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        # One sample, two measurements.
        assert len(result.project.samples) == 1
        assert len(result.project.samples[0].measurements) == 2
        assert result.parsed_count == 2

    def test_two_different_samples_separate(self, tmp_path: Path):
        (tmp_path / "S1").mkdir()
        (tmp_path / "S1" / "a.fake").write_text("aa")
        (tmp_path / "S2").mkdir()
        (tmp_path / "S2" / "b.fake").write_text("bb")

        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        names = sorted(s.canonical_name for s in result.project.samples)
        assert names == ["S1", "S2"]


# ─── Outcomes ───────────────────────────────────────────────────────
class TestOutcomes:
    def test_unclassified_file_is_skipped(self, tmp_path: Path):
        # `.unknown` extension → no parser claims it.
        (tmp_path / "garbage.unknown").write_text("???")

        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        assert result.parsed_count == 0
        assert result.unclassified_count == 1
        assert result.outcomes[0].outcome == Outcome.SKIPPED_UNCLASSIFIED
        assert result.project.samples == ()

    def test_parser_crash_records_parse_failed(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "data.fake").write_text("hi")

        crashing_parser = _FakeParser(raise_on_parse=True)
        result = _orchestrator(_registry_with(crashing_parser)).ingest(tmp_path)

        assert result.failed_count == 1
        assert result.parsed_count == 0
        assert result.outcomes[0].outcome == Outcome.PARSE_FAILED
        assert "simulated parser crash" in (result.outcomes[0].error or "")
        # Crashed file produces no measurement.
        assert result.project.samples == ()

    def test_parser_emits_error_yields_parsed_with_issues(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "data.fake").write_text("hi")

        result = _orchestrator(_registry_with(_FakeParser(emit_error=True))).ingest(tmp_path)
        assert result.outcomes[0].outcome == Outcome.PARSED_WITH_ISSUES
        # Measurement still saved.
        assert len(result.project.samples) == 1
        # Error issue is attached.
        meas = result.project.samples[0].measurements[0]
        assert any(i.severity is Severity.ERROR for i in meas.issues)


# ─── Project metadata ───────────────────────────────────────────────
class TestProjectMetadata:
    def test_default_name_is_folder_basename(self, tmp_path: Path):
        d = tmp_path / "MyProject"
        d.mkdir()
        result = _orchestrator(_registry_with(_FakeParser())).ingest(d)
        assert result.project.name == "MyProject"

    def test_explicit_name_overrides(self, tmp_path: Path):
        result = _orchestrator(_registry_with(_FakeParser())).ingest(
            tmp_path,
            project_name="MyOverride",
        )
        assert result.project.name == "MyOverride"

    def test_root_path_is_resolved_absolute(self, tmp_path: Path):
        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        assert result.project.root_path.is_absolute()


# ─── Idempotent re-ingestion ────────────────────────────────────────
class TestReIngestion:
    def test_second_ingest_no_duplicates(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "a.fake").write_text("aa")

        o = _orchestrator(_registry_with(_FakeParser()))
        first = o.ingest(tmp_path)
        second = o.ingest(tmp_path)

        # Same number of measurements.
        assert len(second.project.samples) == 1
        assert len(second.project.samples[0].measurements) == 1
        # Project ID preserved across runs.
        assert first.project.id == second.project.id

    def test_second_ingest_records_cache_hit(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "a.fake").write_text("aa")

        o = _orchestrator(_registry_with(_FakeParser()))
        o.ingest(tmp_path)
        second = o.ingest(tmp_path)

        # Second run: file is cached, not re-parsed.
        assert second.cached_count == 1
        assert second.parsed_count == 0
        assert second.outcomes[0].outcome == Outcome.SKIPPED_CACHED

    def test_new_file_added_in_second_run_is_parsed(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "a.fake").write_text("aa")

        o = _orchestrator(_registry_with(_FakeParser()))
        o.ingest(tmp_path)

        # Add a new file, re-ingest.
        (d / "b.fake").write_text("bb")
        second = o.ingest(tmp_path)

        assert second.parsed_count == 1  # b.fake
        assert second.cached_count == 1  # a.fake
        assert len(second.project.samples[0].measurements) == 2

    def test_parser_version_bump_invalidates_cache(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "a.fake").write_text("aa")

        # First ingest with v1.0.0.
        o1 = _orchestrator(_registry_with(_FakeParser()))
        o1.ingest(tmp_path)

        # Second ingest with a "v2.0.0" parser → cache should miss.
        class FakeParserV2(_FakeParser):
            version = "2.0.0"

        o2 = _orchestrator(_registry_with(FakeParserV2()))
        result = o2.ingest(tmp_path)
        assert result.parsed_count == 1
        assert result.cached_count == 0


# ─── Skipped/system files ───────────────────────────────────────────
class TestSkipped:
    def test_dotfiles_silently_skipped(self, tmp_path: Path):
        # Hidden files don't appear in outcomes at all (crawler filters them).
        (tmp_path / ".hidden_data.fake").write_text("ignore me")
        (tmp_path / "real.fake").write_text("real")
        # Need a non-generic parent for real.fake to get a sample.
        d = tmp_path / "S1"
        d.mkdir()
        shutil.move(str(tmp_path / "real.fake"), str(d / "real.fake"))

        result = _orchestrator(_registry_with(_FakeParser())).ingest(tmp_path)
        # Only the real file shows up.
        assert len(result.outcomes) == 1
        assert result.outcomes[0].path.name == "real.fake"

    def test_latos_subdir_not_recrawled(self, tmp_path: Path):
        d = tmp_path / "S1"
        d.mkdir()
        (d / "a.fake").write_text("aa")

        o = _orchestrator(_registry_with(_FakeParser()))
        o.ingest(tmp_path)

        # `.latos/` now exists with our DB and Parquet files. Re-ingesting
        # must not include those files in the outcomes.
        second = o.ingest(tmp_path)
        for out in second.outcomes:
            assert ".latos" not in out.path.parts


# ─── Real-fixture integration ───────────────────────────────────────
class TestRealFixtures:
    """End-to-end against a copy of the parser fixtures."""

    def test_all_fixture_techniques_recognized(self, tmp_path: Path):
        # Copy the entire parser-fixtures tree to a temp project root.
        proj = tmp_path / "RealProj"
        shutil.copytree(_PROJ_FIXTURES, proj)

        from latos.ingestion.registry import default_registry

        result = Orchestrator(registry=default_registry()).ingest(proj)

        # One measurement per fixture file (9 fixtures total).
        all_measurements = [m for s in result.project.samples for m in s.measurements]
        assert len(all_measurements) == 9
        techniques = {m.technique for m in all_measurements}
        # We have: XRD (3 files), XPS, UV-DRS, Hall, Thermoelectric, EDS, SEM/TEM
        assert Technique.XRD in techniques
        assert Technique.XPS in techniques
        assert Technique.UV_DRS in techniques
        assert Technique.HALL in techniques
        assert Technique.THERMOELECTRIC in techniques
        assert Technique.EDS in techniques

    def test_real_fixture_parquet_files_exist(self, tmp_path: Path):
        proj = tmp_path / "RealProj"
        shutil.copytree(_PROJ_FIXTURES, proj)

        from latos.ingestion.registry import default_registry

        result = Orchestrator(registry=default_registry()).ingest(proj)

        # Every measurement that produced arrays should have a Parquet file.
        for s in result.project.samples:
            for m in s.measurements:
                if m.parsed_data_path is not None:
                    assert m.parsed_data_path.exists()
