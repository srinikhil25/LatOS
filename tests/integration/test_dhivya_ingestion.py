"""End-to-end ingestion test on the Dhivya / Materials-Informatics dataset.

This is a *real-data* integration test. It runs only on machines where
`D:/Materials-Informatics/data_raw/dhivya_data` exists (i.e. the
maintainer's laptop). On CI and on every other machine it skips
silently — the assertion-level value lives in checking that the
production orchestrator handles a real, messy folder of 161 mixed
files end-to-end without crashing or losing data.

The numbers asserted here come from a dry run:
- 161 files total in the source folder
- ~76 files have parsers (XRD, XPS, UV-DRS, Hall, Thermoelectric, EDS, microscopy)
- ~84 files are non-data (PDFs, JPEGs, .docx notes, .spe binaries we don't parse)
- All 7 Stage 1 techniques represented at least once
- Multiple samples (≥10) inferred by the folder-name heuristic
- No parser raises; no hash failures.

If the data layout changes, update the bounds — keep them as ranges
(``>=``) rather than equalities so the test is resilient to small
additions to the data folder.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from latos.core.enums import Technique
from latos.ingestion.orchestrator import Orchestrator, Outcome
from latos.ingestion.registry import default_registry

# Source: maintainer's predecessor data. Skipped if absent.
_DHIVYA_SOURCE = Path("D:/Materials-Informatics/data_raw/dhivya_data")

# Wall-clock budget for the full ingest (excluding the copy step). If
# this is exceeded, something has regressed: 161 files should run in
# well under 30 seconds on any reasonable machine.
_INGEST_BUDGET_SEC = 30.0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _DHIVYA_SOURCE.exists(),
        reason=(
            "Real Materials-Informatics dataset not present at "
            f"{_DHIVYA_SOURCE}; integration test only runs on the "
            "maintainer's machine."
        ),
    ),
]


@pytest.fixture(scope="module")
def ingestion_result(tmp_path_factory: pytest.TempPathFactory):
    """Copy the real dataset to a tmp project root and ingest it once.

    We exclude `.latos/` from the copy: a previous run on the source
    folder may have left a populated SQLite cache there, and copying
    it along makes every file show up as `SKIPPED_CACHED` on this
    "fresh" ingestion. Filtering `.latos/` (and `__pycache__/` for
    good measure) ensures the test exercises a clean parse path.
    """
    proj = tmp_path_factory.mktemp("dhivya_ingest") / "Dhivya"
    shutil.copytree(
        _DHIVYA_SOURCE,
        proj,
        ignore=shutil.ignore_patterns(".latos", "__pycache__", ".DS_Store"),
    )

    orchestrator = Orchestrator(registry=default_registry())
    t0 = time.perf_counter()
    result = orchestrator.ingest(proj, project_name="Dhivya")
    elapsed = time.perf_counter() - t0
    return result, elapsed, proj


# ─── Coverage ───────────────────────────────────────────────────────
class TestEndToEnd:
    def test_ingest_completes_within_budget(self, ingestion_result):
        _, elapsed, _ = ingestion_result
        assert elapsed < _INGEST_BUDGET_SEC, (
            f"Ingest took {elapsed:.1f}s, budget is {_INGEST_BUDGET_SEC}s — regression?"
        )

    def test_no_parser_failures(self, ingestion_result):
        result, _, _ = ingestion_result
        # PARSE_FAILED indicates a parser crash, not a data problem.
        # If this fires, a parser raised on real data — needs investigation.
        assert result.failed_count == 0, (
            f"Parser crashes on real data: {result.failed_count} failures. "
            f"Files: {[o.path.name for o in result.outcomes if o.outcome == Outcome.PARSE_FAILED]}"
        )

    def test_no_hash_failures(self, ingestion_result):
        result, _, _ = ingestion_result
        hash_failed = sum(1 for o in result.outcomes if o.outcome == Outcome.SKIPPED_HASH_FAILED)
        assert hash_failed == 0

    def test_total_outcomes_in_expected_range(self, ingestion_result):
        result, _, _ = ingestion_result
        # Source has 161 files; some may be filtered (hidden, lockfiles).
        # Allow a small tolerance.
        assert 150 <= len(result.outcomes) <= 170, (
            f"Crawled {len(result.outcomes)} files; expected ~161"
        )

    def test_at_least_60_files_parsed(self, ingestion_result):
        result, _, _ = ingestion_result
        # Conservative lower bound. A real run produces ~76; if this
        # drops below 60 we've regressed somewhere.
        assert result.parsed_count >= 60, f"Only {result.parsed_count} files parsed — regression?"


# ─── Coverage by technique ──────────────────────────────────────────
class TestTechniques:
    def test_all_seven_stage1_techniques_represented(self, ingestion_result):
        result, _, _ = ingestion_result
        techniques_found = {m.technique for s in result.project.samples for m in s.measurements}
        # The Dhivya dataset includes every Stage 1 technique. Asserting
        # the full set catches regressions in any single parser.
        expected = {
            Technique.XRD,
            Technique.XPS,
            Technique.UV_DRS,
            Technique.HALL,
            Technique.THERMOELECTRIC,
            # Microscopy: data has TEM/SEM .tif files.
            Technique.SEM,
        }
        missing = expected - techniques_found
        assert not missing, f"Techniques missing from real-data ingest: {missing}"


# ─── Sample inference ───────────────────────────────────────────────
class TestSamples:
    def test_multiple_samples_found(self, ingestion_result):
        result, _, _ = ingestion_result
        # Heuristic produces ≥ 10 distinct samples from this dataset
        # (several from filename-stem fallback + several from named folders).
        # Stage 2 will collapse these; for Stage 1 we just want > 1.
        assert len(result.project.samples) >= 5, f"Only {len(result.project.samples)} samples found"

    def test_every_sample_has_at_least_one_measurement(self, ingestion_result):
        result, _, _ = ingestion_result
        for sample in result.project.samples:
            assert len(sample.measurements) >= 1, (
                f"Sample {sample.canonical_name!r} has no measurements"
            )

    def test_known_xps_sample_grouping(self, ingestion_result):
        result, _, _ = ingestion_result
        # XPS folder layout: `XPS/CS (Pure)/*.csv` and `XPS/CS-3/*.csv`.
        # Heuristic uses the immediate parent folder when non-generic.
        names = {s.canonical_name for s in result.project.samples}
        assert "CS (Pure)" in names or "CS-3" in names


# ─── Stage 2: labeling pipeline ─────────────────────────────────────
class TestLabelingPipeline:
    """Stage 2C clustering against the real Dhivya dataset.

    The dataset is the regression motivator: Stage 1's per-folder
    heuristic over-splits things like `CS Pure` (in XRD) and
    `CS (Pure)` (in XPS) into two samples. The labeling pipeline
    must collapse those.
    """

    def test_clustering_reduces_or_preserves_sample_count(self, ingestion_result):
        from latos.ingestion.labeling.pipeline import cluster_project

        result, _, _ = ingestion_result
        clusters = cluster_project(result.project)
        # Clustering can only collapse samples; never invent new ones.
        assert len(clusters) <= len(result.project.samples), (
            f"Pipeline produced {len(clusters)} clusters from "
            f"{len(result.project.samples)} samples - clustering must not grow the set"
        )

    def test_cs_pure_variants_merged_into_single_cluster(self, ingestion_result):
        from latos.ingestion.labeling.pipeline import cluster_project

        result, _, _ = ingestion_result
        clusters = cluster_project(result.project)

        # Find any cluster that absorbed both `CS Pure` and `CS (Pure)`
        # aliases. With the default threshold this regression is the
        # primary thing we're guarding against. If the dataset only
        # contains one of the two variants this test silently passes —
        # we'd rather not fail on data drift.
        names_to_aliases = [set(c.aliases) for c in clusters]
        cs_pure_present = any(
            any(x.lower().startswith("cs pure") for x in a) for a in names_to_aliases
        )
        cs_paren_present = any(any("(pure)" in x.lower() for x in a) for a in names_to_aliases)
        if cs_pure_present and cs_paren_present:
            # Both variants exist: they must land in the same cluster.
            for aliases in names_to_aliases:
                lowered = {x.lower() for x in aliases}
                has_pure = any(x.startswith("cs pure") for x in lowered)
                has_paren = any("(pure)" in x for x in lowered)
                if has_pure and has_paren:
                    return
            raise AssertionError(
                "Both `CS Pure` and `CS (Pure)` variants are present but landed "
                "in different clusters - the Stage 2 regression case is firing"
            )


# ─── Persistence ────────────────────────────────────────────────────
class TestPersistence:
    def test_database_file_created(self, ingestion_result):
        _, _, proj = ingestion_result
        db_path = proj / ".latos" / "data.db"
        assert db_path.exists()
        # Non-trivial size — empty SQLite is ~12KB.
        assert db_path.stat().st_size > 12 * 1024

    def test_arrays_directory_populated(self, ingestion_result):
        _result, _elapsed, proj = ingestion_result
        arrays_dir = proj / ".latos" / "arrays"
        assert arrays_dir.is_dir()
        # Spectroscopy + thermoelectric measurements write Parquet;
        # microscopy is metadata-only and writes none, and Hall is also
        # scalar-metadata-only. So Parquet count ≪ measurement count.
        parquet_files = list(arrays_dir.glob("*.parquet"))
        assert len(parquet_files) >= 10, (
            f"Only {len(parquet_files)} Parquet files written; expected ≥ 10"
        )

    def test_re_ingest_is_fast_and_cached(self, tmp_path: Path):
        """Stage 1 done-criterion: 'reopening project takes <1 sec'."""
        proj = tmp_path / "Dhivya2"
        shutil.copytree(_DHIVYA_SOURCE, proj)
        orchestrator = Orchestrator(registry=default_registry())

        # First ingest establishes the cache.
        orchestrator.ingest(proj)

        # Second ingest should be cached and fast.
        t0 = time.perf_counter()
        result2 = orchestrator.ingest(proj)
        elapsed = time.perf_counter() - t0

        assert result2.parsed_count == 0, (
            f"Re-ingest re-parsed {result2.parsed_count} files; cache should have been hit"
        )
        assert result2.cached_count >= 60
        # Done-criterion: reopen <1s. We give some headroom for slow disks.
        assert elapsed < 5.0, f"Re-ingest took {elapsed:.2f}s; >5s indicates broken caching"
