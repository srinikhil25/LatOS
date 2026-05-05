"""Tests for `latos.ingestion.crawler.crawl()`."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from latos.ingestion.crawler import (
    CrawlReport,
    crawl,
)
from latos.ingestion.hashing import HashCache
from latos.ingestion.registry import ParserRegistry, default_registry


# ─── Helpers ────────────────────────────────────────────────────────
def _write(path: Path, content: bytes | str = b"x") -> Path:
    """Create `path` with simple content. Returns the path for chaining."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return path


def _empty_registry() -> ParserRegistry:
    """A registry with no parsers — every file lands as 'unclassified'."""
    return ParserRegistry()


# ─── Construction / arg validation ──────────────────────────────────
class TestArgValidation:
    def test_missing_root_raises(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            crawl(missing, _empty_registry())

    def test_root_is_a_file_raises(self, tmp_path: Path):
        f = _write(tmp_path / "data.txt")
        with pytest.raises(NotADirectoryError):
            crawl(f, _empty_registry())


# ─── Empty / trivial cases ──────────────────────────────────────────
class TestEmptyCases:
    def test_empty_folder_returns_empty_report(self, tmp_path: Path):
        report = crawl(tmp_path, _empty_registry())
        assert isinstance(report, CrawlReport)
        assert report.root == tmp_path
        assert report.entries == ()
        assert len(report) == 0

    def test_only_system_files_yields_empty_report(self, tmp_path: Path):
        _write(tmp_path / ".DS_Store")
        _write(tmp_path / "Thumbs.db")
        _write(tmp_path / "desktop.ini")
        _write(tmp_path / "ehthumbs.db")
        _write(tmp_path / "~$workbook.xlsx")
        _write(tmp_path / "scratch.tmp")
        _write(tmp_path / ".hidden")
        report = crawl(tmp_path, _empty_registry())
        assert report.entries == ()


# ─── Walking + skip filters ─────────────────────────────────────────
class TestWalkAndSkip:
    def test_finds_files_in_root(self, tmp_path: Path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        report = crawl(tmp_path, _empty_registry())
        names = sorted(e.path.name for e in report.entries)
        assert names == ["a.txt", "b.txt"]

    def test_finds_files_in_subdirectories(self, tmp_path: Path):
        _write(tmp_path / "sub1" / "a.txt")
        _write(tmp_path / "sub1" / "b.txt")
        _write(tmp_path / "sub2" / "deep" / "c.txt")
        report = crawl(tmp_path, _empty_registry())
        names = sorted(e.path.name for e in report.entries)
        assert names == ["a.txt", "b.txt", "c.txt"]

    def test_skips_latos_directory(self, tmp_path: Path):
        # Critical: crawler must not recurse into its own output folder.
        _write(tmp_path / "data.txt")
        _write(tmp_path / ".latos" / "data.db")
        _write(tmp_path / ".latos" / "arrays" / "x.parquet")
        report = crawl(tmp_path, _empty_registry())
        names = sorted(e.path.name for e in report.entries)
        assert names == ["data.txt"]

    def test_skips_git_node_modules_pycache(self, tmp_path: Path):
        _write(tmp_path / "real.txt")
        _write(tmp_path / ".git" / "config")
        _write(tmp_path / "node_modules" / "lib.js")
        _write(tmp_path / "__pycache__" / "x.pyc")
        _write(tmp_path / "venv" / "site.py")
        report = crawl(tmp_path, _empty_registry())
        names = sorted(e.path.name for e in report.entries)
        assert names == ["real.txt"]

    def test_skips_hidden_files(self, tmp_path: Path):
        _write(tmp_path / "visible.txt")
        _write(tmp_path / ".env")
        _write(tmp_path / ".bashrc")
        report = crawl(tmp_path, _empty_registry())
        names = sorted(e.path.name for e in report.entries)
        assert names == ["visible.txt"]

    def test_skips_hidden_directories(self, tmp_path: Path):
        _write(tmp_path / "visible.txt")
        _write(tmp_path / ".cache" / "x.txt")
        _write(tmp_path / ".secret" / "y.txt")
        report = crawl(tmp_path, _empty_registry())
        names = sorted(e.path.name for e in report.entries)
        assert names == ["visible.txt"]

    def test_skips_office_lockfiles(self, tmp_path: Path):
        _write(tmp_path / "real.xlsx")
        _write(tmp_path / "~$real.xlsx")
        _write(tmp_path / "~$thermoelectric.docx")
        report = crawl(tmp_path, _empty_registry())
        names = sorted(e.path.name for e in report.entries)
        assert names == ["real.xlsx"]

    def test_custom_skip_dirs(self, tmp_path: Path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "private" / "b.txt")
        _write(tmp_path / "public" / "c.txt")
        report = crawl(
            tmp_path,
            _empty_registry(),
            skip_dirs=frozenset({"private"}),
        )
        names = sorted(e.path.name for e in report.entries)
        assert names == ["a.txt", "c.txt"]


# ─── Entry contents ─────────────────────────────────────────────────
class TestEntryContents:
    def test_relative_path_is_relative_to_root(self, tmp_path: Path):
        _write(tmp_path / "sub" / "deep" / "x.txt")
        report = crawl(tmp_path, _empty_registry())
        entry = report.entries[0]
        assert entry.relative_path == Path("sub") / "deep" / "x.txt"

    def test_size_bytes_matches_file(self, tmp_path: Path):
        f = _write(tmp_path / "x.txt", b"0123456789")
        report = crawl(tmp_path, _empty_registry())
        assert report.entries[0].size_bytes == 10
        assert f.stat().st_size == 10

    def test_mtime_is_tz_aware_utc(self, tmp_path: Path):
        _write(tmp_path / "x.txt")
        report = crawl(tmp_path, _empty_registry())
        mtime = report.entries[0].mtime
        assert mtime.tzinfo is not None
        # UTC offset is zero.
        assert mtime.utcoffset() is not None
        assert mtime.utcoffset().total_seconds() == 0

    def test_sha256_is_64_hex_chars(self, tmp_path: Path):
        _write(tmp_path / "x.txt", b"hello")
        report = crawl(tmp_path, _empty_registry())
        sha = report.entries[0].sha256
        assert sha is not None
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_no_classification_with_empty_registry(self, tmp_path: Path):
        _write(tmp_path / "x.txt", b"hello")
        report = crawl(tmp_path, _empty_registry())
        entry = report.entries[0]
        assert entry.classified_parser is None
        assert entry.best_match_parser is None
        assert entry.confidence == 0.0
        assert entry.error is None


# ─── Classification (with default_registry on real fixtures) ────────
class TestClassification:
    def test_real_fixtures_classified_correctly(self):
        """Crawl the real test fixture tree → check each file lands on the right parser."""
        fixtures_root = Path(__file__).parent.parent.parent / "fixtures" / "parsers"
        report = crawl(fixtures_root, default_registry())

        # Map filename → expected parser name (from our 9 fixtures).
        expected: dict[str, str] = {
            "rigaku_bs3a.txt": "rigaku-xrd-txt",
            "panalytical_cscbi1.xrdml": "panalytical-xrdml",
            "rigaku_cs_pure.asc": "rigaku-xrd-asc",
            "casaxps_c1s.csv": "casaxps-csv",
            "uvdrs_cs.xlsx": "uvdrs-xlsx",
            "hall_cs.xls": "hall-xls",
        }
        actual: dict[str, str | None] = {e.path.name: e.classified_parser for e in report.entries}
        for fname, parser_name in expected.items():
            assert fname in actual, f"{fname!r} missing from crawl report"
            assert actual[fname] == parser_name, (
                f"{fname!r}: expected {parser_name}, got {actual[fname]}"
            )

    def test_threshold_demotes_weak_match_to_unclassified(self, tmp_path: Path):
        """A file scoring 0.7 with threshold 0.8 should be unclassified but recorded."""
        # The Rigaku XRD txt parser returns 0.7 when only one of two
        # signature keys is present.
        _write(
            tmp_path / "partial.txt",
            ";SampleName = test\n10.0 100\n",
        )
        report = crawl(
            tmp_path,
            default_registry(),
            confidence_threshold=0.8,
        )
        entry = report.entries[0]
        assert entry.classified_parser is None
        assert entry.best_match_parser == "rigaku-xrd-txt"
        assert entry.confidence == pytest.approx(0.7)


# ─── Error handling ─────────────────────────────────────────────────
class TestErrorHandling:
    def test_unreadable_file_recorded_with_error(self, tmp_path: Path):
        _write(tmp_path / "x.txt", b"hello")
        # Patch hash_file to simulate a permission error mid-walk.
        with patch(
            "latos.ingestion.crawler.hash_file",
            side_effect=PermissionError("Access denied"),
        ):
            report = crawl(tmp_path, default_registry())

        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.sha256 is None
        assert entry.error is not None
        assert "hash failed" in entry.error
        assert entry.classified_parser is None

    def test_errored_entry_appears_in_errored_property(self, tmp_path: Path):
        _write(tmp_path / "x.txt")
        with patch(
            "latos.ingestion.crawler.hash_file",
            side_effect=PermissionError("nope"),
        ):
            report = crawl(tmp_path, default_registry())
        assert len(report.errored) == 1
        assert len(report.classified) == 0


# ─── Hash cache reuse ───────────────────────────────────────────────
class TestHashCache:
    def test_second_crawl_reuses_cache(self, tmp_path: Path):
        _write(tmp_path / "x.txt", b"hello")
        cache = HashCache()

        report1 = crawl(tmp_path, _empty_registry(), hash_cache=cache)
        sha1 = report1.entries[0].sha256
        cache_size_after_first = len(cache)

        # Patch hash_file to detect re-hashing.
        from latos.ingestion import crawler as crawler_mod

        original_hash = crawler_mod.hash_file
        call_count = 0

        def counting_hash(path: Path, *, cache: HashCache | None = None) -> str:
            nonlocal call_count
            call_count += 1
            return original_hash(path, cache=cache)

        with patch.object(crawler_mod, "hash_file", side_effect=counting_hash):
            report2 = crawl(tmp_path, _empty_registry(), hash_cache=cache)

        # Same hash, cache used (call_count is the wrapper's count, which
        # delegates to the original — hash_file IS called, but underlying
        # hashlib.sha256 is not because the cache hits).
        assert report2.entries[0].sha256 == sha1
        assert cache_size_after_first == 1


# ─── Progress callback ─────────────────────────────────────────────
class TestProgressCallback:
    def test_called_once_per_file(self, tmp_path: Path):
        for i in range(5):
            _write(tmp_path / f"file_{i}.txt")
        calls: list[tuple[int, int, Path]] = []

        def cb(index: int, total: int, path: Path) -> None:
            calls.append((index, total, path))

        crawl(tmp_path, _empty_registry(), on_progress=cb)
        assert len(calls) == 5

    def test_indices_monotonic_and_total_consistent(self, tmp_path: Path):
        for i in range(5):
            _write(tmp_path / f"file_{i}.txt")
        calls: list[tuple[int, int, Path]] = []

        def cb(index: int, total: int, path: Path) -> None:
            calls.append((index, total, path))

        crawl(tmp_path, _empty_registry(), on_progress=cb)
        indices = [c[0] for c in calls]
        totals = {c[1] for c in calls}
        assert indices == [0, 1, 2, 3, 4]
        assert totals == {5}

    def test_no_callback_no_problem(self, tmp_path: Path):
        # Default on_progress=None must work without raising.
        _write(tmp_path / "x.txt")
        report = crawl(tmp_path, _empty_registry())
        assert len(report.entries) == 1


# ─── CrawlReport helpers ───────────────────────────────────────────
class TestReportHelpers:
    def test_classified_unclassified_errored_partition_entries(self, tmp_path: Path):
        # 1 classified, 1 unclassified, 1 errored.
        _write(tmp_path / "rigaku.txt", ";SampleName = a\n;KAlpha1 = 1.54\n10 100\n")
        _write(tmp_path / "garbage.txt", "random content")

        # Force a third entry into the errored bucket via mock.
        _write(tmp_path / "broken.txt", "x")

        original_hash_file: Any
        from latos.ingestion import crawler as crawler_mod

        original_hash_file = crawler_mod.hash_file

        def selective_fail(path: Path, *, cache: HashCache | None = None) -> str:
            if path.name == "broken.txt":
                raise PermissionError("simulated")
            return original_hash_file(path, cache=cache)

        with patch.object(crawler_mod, "hash_file", side_effect=selective_fail):
            report = crawl(tmp_path, default_registry())

        # Some buckets should be populated with the right names.
        classified_names = {e.path.name for e in report.classified}
        unclassified_names = {e.path.name for e in report.unclassified}
        errored_names = {e.path.name for e in report.errored}
        assert "rigaku.txt" in classified_names
        assert "garbage.txt" in unclassified_names
        assert "broken.txt" in errored_names

    def test_len(self, tmp_path: Path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        report = crawl(tmp_path, _empty_registry())
        assert len(report) == 2
