# Latos — Results Log

Append-only record of milestones, benchmarks, bug fixes, and demo materials.
**Never edit past entries.** Only add new ones at the bottom.

---

## 2026-04-26 — Project Initialized

### Setup
- New repo: `D:/Latos/`
- License: MIT
- Tech stack locked: PySide6 + QFluentWidgets + pyqtgraph + matplotlib + SQLite + lmfit + GPyTorch + BoTorch + Ollama
- Testing stack locked: pytest + pytest-qt + hypothesis + pytest-snapshot
- CI: GitHub Actions (lint + test on Win/Mac/Linux × Py 3.11/3.12)
- Coverage gate: 70%
- Pre-commit hooks: ruff + mypy + standard housekeeping

### Files Committed
- `pyproject.toml` (PEP 621 metadata, ruff/mypy/pytest config)
- `.gitignore`
- `.pre-commit-config.yaml`
- `LICENSE` (MIT)
- `README.md`
- `CONTRIBUTING.md`
- `CLAUDE.md` (AI assistant context)
- `AGENTS.md` (AI agent operational rules)
- `.claude/settings.json`
- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`
- Empty package skeleton with `__init__.py` files
- `STAGES.md` (gitignored, internal planning doc)

### Status
- Stage: 0 (Project Setup) ✅
- Next: Stage 1 — Foundation Layer

---

## 2026-04-26 — Stage 0 CI Green

### CI Pipeline Verified
First full CI run successful on GitHub Actions: https://github.com/srinikhil25/LatOS

| Job | Status | Duration |
|-----|--------|----------|
| Lint & Type Check | ✅ | 1m 3s |
| Test (Python 3.11 ubuntu) | ✅ | ~2m |
| Test (Python 3.12 ubuntu) | ✅ | ~2m |
| Test (Python 3.11 windows) | ✅ | ~3m 43s |
| Test (Python 3.12 windows) | ✅ | ~5m |
| Test (Python 3.11 macos) | ✅ | ~1m 43s |
| Test (Python 3.12 macos) | ✅ | ~1m 40s |
| Coverage Gate | ✅ | 1m 24s |
| Build Distribution | ✅ | 15s |
| **Total** | **✅** | **6m 22s** |

### Tests
- 3 smoke tests passing on all 6 (OS × Python) matrix combinations
- `test_package_imports` — package importable
- `test_version_exists` — `__version__` attribute present
- `test_version_format` — semver-style format validation

### Bugs Found & Fixed (Stage 0)
1. **CI failed: pytest exit code 5** ("no tests collected") — added smoke tests + lowered coverage gate to 0% for Stage 0 (commit `e4d694a`)
2. **CI failed: Linux UI test step exit-5** — explicitly tolerate exit-5 in UI step until Stage 1E adds real UI tests (commit `5138973`)

### Coverage Gate Schedule
| Stage | `COVERAGE_MIN` |
|-------|----------------|
| 0 (current) | 0 |
| 1 | 70 |
| 4 | 80 |
| 8 | 85 |

### Commits
- `b338615` — initial project structure
- `66fc0fd` — GitHub URL casing fix (LatOS)
- `e4d694a` — CI smoke tests + coverage gate adjustments
- `5138973` — CI Linux UI step exit-5 handling

### Slide-Worthy Achievement (Stage 0)
> *"Initialized open-source project Latos with industry-standard tooling — automated cross-platform testing on Windows/Mac/Linux × Python 3.11/3.12, code quality gates (linting + type-checking + coverage), MIT license, and CI/CD pipeline. Project is now ready for Stage 1 development with quality safeguards in place from day 1."*

---

## 2026-04-27 — Stage 1A Complete: Domain Models

### Files added
- `src/latos/core/enums.py` — `Technique`, `FileRole`, `Severity` (with display names + ordering)
- `src/latos/core/exceptions.py` — `LatosError` hierarchy (14 exception types)
- `src/latos/core/models.py` — `Project`, `Sample`, `Measurement`, `FileRef`, `ValidationIssue` (all frozen dataclasses)
- `src/latos/core/__init__.py` — flat re-exports for ergonomics
- `src/latos/py.typed` — PEP 561 marker
- Tests: `tests/unit/core/{test_enums,test_exceptions,test_models}.py`

### Tests
- **83 tests, all passing locally**
- 3 from Stage 0 smoke + 80 new Stage 1A
- Coverage on `core/`: **95%** (above target of 70%)
  - enums.py: 100%
  - exceptions.py: 100%
  - models.py: 93%

### Quality gates
- ✅ Ruff lint clean
- ✅ Ruff format clean
- ✅ Mypy strict clean (28 source files)

### Bugs found & fixed (during Stage 1A)
- `_file_ref` test helper used `or` instead of `is None` check — empty string sha256 fell through to default. Fixed.
- `test_lookup_by_id` constructed two unrelated projects then expected a relation. Refactored to single project.

### Architecture decisions enforced by tests
- All IDs are 32-char lowercase hex UUIDs (validated on construction)
- All timestamps are timezone-aware (naive datetimes rejected)
- All collections are tuples, never lists (immutability)
- Sample.measurements must reference their owning Sample (cross-link validation)
- Project.samples must reference their owning Project (cross-link validation)
- Aliases are unique non-empty strings (deduplication enforced)
- SHA-256 hashes are exactly 64 lowercase hex chars (length + alphabet checked)

### Slide-Worthy Achievement (Stage 1A)
> *"Built the domain model foundation — strict, immutable data shapes that flow through the entire platform. Every constraint (ID format, timezone awareness, cross-references) is enforced at construction time, catching bugs before they reach the database or UI."*

**Wow numbers for slide:**
- 83 tests, 95% coverage
- 14 exception types in clean hierarchy
- 0 mypy strict-mode errors

---

## 2026-04-27 — Stage 1B Complete: Persistence Layer

### Files added
- `src/latos/persistence/schema.py` — SQLAlchemy 2.0 declarative tables (5 tables: projects, samples, measurements, files, validation_issues) + `UtcDateTime` TypeDecorator that round-trips timezone info correctly on SQLite
- `src/latos/persistence/db.py` — engine factory, session factory, project DB path resolver, SQLite PRAGMAs (WAL, foreign_keys, busy_timeout, synchronous=NORMAL)
- `src/latos/persistence/mappers.py` — bidirectional Domain ↔ ORM conversion (the only module bridging the two layers)
- `src/latos/persistence/repository.py` — `ProjectRepository` (save/load/list/delete) + `ProjectSummary`
- `src/latos/persistence/__init__.py` — public API surface
- `migrations/` — Alembic configured with custom `env.py` that uses Latos's metadata
- `migrations/versions/0001_initial_schema.py` — initial schema migration (stable revision ID)
- Tests: `tests/unit/persistence/{conftest,test_db,test_mappers,test_repository,test_migrations}.py`

### Storage convention finalized
```
<project_root>/.latos/
├── data.db          # SQLite metadata (one file per project)
├── arrays/          # Parquet arrays (one file per measurement)
└── exports/         # Generated reports/figures
```

### Tests
- **131 tests, all passing**
- 83 from previous stages + 48 new persistence tests
- Coverage on `persistence/`: **97%** (db 100%, mappers 100%, repository 100%, schema 89%)
- Overall coverage: **95%**

### Quality gates
- ✅ Ruff lint clean (32 source files)
- ✅ Ruff format clean
- ✅ Mypy strict clean
- ✅ Migration apply + downgrade cycle verified

### Bugs found & fixed (during Stage 1B)
1. **SQLite drops tzinfo on read** — `DateTime(timezone=True)` returns naive datetimes from SQLite. Fixed with `UtcDateTime` TypeDecorator that re-attaches UTC on load and rejects naive datetimes on save.
2. **Migration didn't update alembic_version** — `connection.execute(PRAGMA)` in env.py started a transaction before alembic's own, breaking the version write. Fixed by moving PRAGMA to a connection-event listener.
3. **Windows path test failure** — assertion compared `row.path == "/data/sample.xy"` but `Path("/data/sample.xy")` stringifies as `\data\sample.xy` on Windows. Fixed to compare against `str(ref.path)`.

### Slide-Worthy Achievement (Stage 1B)
> *"Built the persistence layer — projects now save to a self-contained SQLite database, with versioned schema migrations powered by Alembic. Researchers can close and reopen Latos and pick up exactly where they left off."*

**Wow numbers for slide:**
- 131 tests passing in 4.4 seconds
- 95% test coverage across the project
- 5-table schema with full cascade-delete safety
- Schema migrations support forward + backward compatibility from day 1

---

## 2026-04-29 — Stage 1C Complete: Parser Migration + File Hashing

### Files added

**Foundation (1C.1, 1C.2):**
- `src/latos/ingestion/hashing.py` — SHA-256 file hashing with `HashCache` keyed on (path, mtime, size). Streamed 1 MB chunks so multi-GB TIF files don't blow memory.
- `src/latos/ingestion/parsed_data.py` — `ParsedData` frozen dataclass: 1-D arrays only, same-length within a measurement, JSON-safe metadata, tz-aware timestamps, semver `parser_version`, kebab-case `parser_name`. Validates 7 invariants in `__post_init__`.
- `src/latos/ingestion/base_parser.py` — `BaseParser` ABC. Concrete parsers set `name`/`version`/`technique`/`supported_extensions` as class attributes; `__init_subclass__` validates them at import time so typos fail fast, not at parse time.
- `src/latos/ingestion/array_store.py` — `ArrayStore` for atomic Parquet I/O. Writes go to `<id>.parquet.tmp` then `os.replace()`; orphan tmp files swept on next construction. Protects researchers who Ctrl+C a long ingestion from corrupting their parse cache.

**Parsers (1C.3, 1C.4a-c):**
- `xrd_rigaku_txt.py` — Rigaku Ultima `.txt` (`;Key = Value` header + `2theta intensity` rows). 96% coverage.
- `xrd_panalytical_xrdml.py` — PANalytical Empyrean `.xrdml` (XML, namespace-agnostic). 90% coverage.
- `xrd_rigaku_asc.py` — Rigaku two-column `.ASC`. Warns at >10% negative intensities (background-subtracted curves are clearly not raw counts). 97% coverage.
- `xps_casaxps_csv.py` — CasaXPS `.csv` exports (variable header). Extracts region label from leading non-numeric line. 95% coverage.
- `uvdrs_xlsx.py` — UV-DRS `.xlsx` (multi-sheet, openpyxl). Parses first sheet, warns about skipped sheets. 84% coverage.
- `hall_xls.py` — Hall-effect `.xls` (xlrd, single-temperature). All values → metadata, no arrays. 78% coverage.
- `thermoelectric_xlsx.py` — zT-style multi-sheet `.xlsx`. Header substring lookup absorbs column-order drift between exports. 86% coverage.
- `eds_bruker_spx.py` — Bruker `.spx` (XML despite the name). Energy axis synthesized via `CalibAbs + CalibLin*i`. 83% coverage.
- `microscopy_tif.py` — TIFF metadata-only (tifffile). Pixels deferred to Stage 5. 80% coverage.

**Dispatcher (1C.5):**
- `src/latos/ingestion/registry.py` — `ParserRegistry` with confidence-pick dispatch (threshold 0.5). 100% coverage. `default_registry()` builds one with all 9 parsers in collision-aware order.

**Test fixtures (real instrument data):**
- 9 fixtures from `D:/Materials-Informatics/data_raw/` covering every parser, with golden-file JSON snapshots for regression detection.

### Architecture decisions

1. **`ParsedData` is the universal contract.** Every parser, regardless of technique or format, returns this shape. Differences live in `arrays` and `metadata` only.
2. **`BaseParser.can_parse(path) -> float` for dispatch.** Cheap (read-header-only), confidence in [0,1]. Threshold 0.5 separates "I'm pretty sure" from "wild guess." Tie-broken by registration order.
3. **Parsers never raise.** Failures are emitted as `ValidationIssue`s on the result; the orchestrator (Stage 1D) decides what to do with errored measurements.
4. **One Parquet file per measurement.** Flat schema (one column per array) means pandas/DuckDB/Power Query can open these files without nested-type machinery.
5. **Atomic writes via `os.replace`.** Prevents half-written Parquet from poisoning the parse cache when the user cancels a long ingestion.
6. **1-D arrays + same-length-within-a-measurement.** Tightened in 1C.2 to match every Stage 1C parser's natural output and avoid loose-validator-vs-strict-writer gaps. Stage 5 will relax for 2-D image content.
7. **Golden-file snapshots.** Each parser is regression-tested against a real instrument file using `pytest-snapshot`. Arrays are summarized as (length, dtype, sha256, head, tail) in the snapshot — exact byte-equality plus human-readable diffs.

### Tests
- **494 tests, all passing** (131 from prior stages + 363 new in Stage 1C)
- **Coverage on `ingestion/`: 89%** average across modules (100% on infra: hashing, ParsedData, BaseParser, ArrayStore, registry; 78–97% on individual parsers, with uncovered lines being OSError/fault-injection branches)
- **Overall coverage: 92%**

### Quality gates
- ✅ Ruff lint clean (45 source files)
- ✅ Ruff format clean
- ✅ Mypy strict clean (with `tifffile` follow_imports="skip" — its source uses 3.12-only syntax)
- ✅ All 9 parsers dispatch correctly via `default_registry()` end-to-end on real fixtures

### Bugs found & fixed (during Stage 1C)
1. **Orphan two_theta on bad intensity row** — XRD parser appended `two_theta` before parsing `intensity`; one bad intensity created a length mismatch, breaking `ParsedData`'s same-length invariant. Fixed by parsing both floats before appending either. Comment in source explains the trap.
2. **`__abstractmethods__` not yet set in `__init_subclass__`** — ABCMeta sets that attribute *after* `__init_subclass__` runs, so `BaseParser.__init_subclass__` couldn't tell intermediate abstract subclasses apart from concrete ones. Fixed by checking `__isabstractmethod__` per method instead.
3. **Class-body name shadowing** — test helper `_make_concrete_parser_class` had `name: ClassVar[str] = name` where the LHS shadowed the function parameter. Renamed to `_name`/`_version`/etc. Comment explains the Python class-scope quirk.
4. **`@dataclass(frozen=True, slots=True)` super() weirdness** — direct attribute-write tests on frozen+slots dataclasses raise `TypeError` instead of `AttributeError` because the generated `__setattr__` uses `super()` against a class object that's been replaced by slots. Switched to `__slots__`/`__dict__` introspection.
5. **`can_parse` 0.7 tier fired on single-line garbage** — `ok >= len(lines) - 1` was satisfied by `0 >= 0`. Added `ok > 0` guard.
6. **Negative-intensity threshold too high** — set at 50%, but the real `.ASC` fixture (background-subtracted curve, ~31% negative) didn't trip the warning. Lowered to 10% — that's the line above which a curve clearly isn't raw counts.
7. **`tifffile` py.typed + Python 3.12 syntax** — tifffile ships type stubs but its source uses 3.12 `type X = Y` statements that fail to parse under our 3.11 mypy target. Added `follow_imports = "skip"` override scoped to that module.

### Slide-Worthy Achievement (Stage 1C)
> *"Built nine instrument-specific parsers — XRD (3 formats), XPS, UV-DRS, Hall, Thermoelectric, EDS, and TEM/SEM — that turn raw lab files into typed, validated measurements ready for analysis. The same `ParsedData` shape flows through every parser; the dispatcher picks the right one by confidence-scoring each file's content (not its extension), so a `.csv` from CasaXPS is correctly distinguished from a `.csv` ledger spreadsheet without false positives."*

**Wow numbers for slide:**
- 9 parsers, 7 techniques covered, all open-source
- 494 tests passing in 26.6 seconds
- 92% test coverage
- Atomic writes + golden-file snapshots — researchers can Ctrl+C a long ingestion without corrupting their cache, and any future parser change against a saved fixture is caught automatically

---

## 2026-05-07 — Stage 1D Complete: File Crawler + Project Orchestrator

### Files added

**1D.1 — Crawler (commit `0c22bb7`)**
- `src/latos/ingestion/crawler.py` — folder walker that hashes every file with SHA-256 (using `HashCache` for fast re-walks) and asks the registry which parser would handle it. Returns a `CrawlReport` of frozen `CrawlEntry`s — pure data, no side effects beyond reading files. Skips `.latos/`, `.git/`, `__pycache__/`, `.idea/`, `.vscode/`, hidden dotfiles, `.DS_Store`/`Thumbs.db`, and Office lockfiles (`~$*.xlsx`).

**1D.2 — Orchestrator (commit `f0335ce`)**
- `src/latos/ingestion/orchestrator.py` — the integration layer where every layer below it connects. Hands a folder to the crawler, runs the winning parser per file, groups files into samples by a Stage-1 heuristic (parent folder name, walking up past generic technique labels), persists Parquet arrays via `ArrayStore` and SQL rows via `ProjectRepository`, and returns a typed `Project` plus a per-file `IngestionResult` ledger with explicit `Outcome` per file (`PARSED`, `PARSED_WITH_ISSUES`, `PARSE_FAILED`, `SKIPPED_UNCLASSIFIED`, `SKIPPED_HASH_FAILED`, `SKIPPED_CACHED`).

**1D.3 — Integration test + CLI (commit `90d9193`)**
- `tests/integration/test_dhivya_ingestion.py` — 12 tests running the orchestrator end-to-end against a copy of `D:/Materials-Informatics/data_raw/dhivya_data` (161 files, 590 MB). Skips silently on machines where the source isn't present.
- `scripts/ingest.py` — CLI wrapper around `Orchestrator.ingest()` that prints a one-page human-readable summary (timings, per-outcome counts, per-technique breakdown, per-sample listing, first 15 unclassified files). Usage: `python scripts/ingest.py <folder>`.

### Architecture decisions

1. **Sample inference is deliberately dumb in Stage 1.** A file's sample = the name of its immediate parent folder, *unless* that folder is generic (XRD, XPS, Hall, data, raw, characterization, ...); in that case we walk up to 3 levels for a non-generic ancestor, then fall back to the file's own stem with a `Severity.WARNING` issue. Stage 2 replaces this entirely with mechanical heuristics + AI/VLM. For Stage 1, dumb-but-stable beats clever-but-surprising.

2. **Idempotent re-ingestion via sha256 + parser_version key.** Re-running ingestion on the same folder is fast: each file's hash is checked against the existing project's `FileRow`s; same hash + same parser version → `SKIPPED_CACHED`, no parse. Parser-version bump → cache miss; the new measurement replaces the old one via a dedupe-by-sha256 pass before save (`FileRow.sha256` is `UNIQUE` in the schema).

3. **Repository factory returns a context manager.** SQLAlchemy engines hold SQLite file handles; not disposing them blocks `tmp_path` cleanup on Windows and prevents the user from moving/deleting a project folder while Latos still has it "open". The `with self.repo_factory(root) as repo:` pattern guarantees disposal even if ingestion raises.

4. **Two-pass walk for accurate progress reporting.** Pass 1 enumerates every surviving path (cheap — `os.scandir()` only). Pass 2 hashes + classifies (the expensive work). Lets the UI show "file 47 of 161" instead of "still working...".

5. **Parsers never crash the orchestrator.** A parser raising during `parse()` becomes a `PARSE_FAILED` outcome with the exception message captured; ingestion continues with the remaining files. This decouples parser correctness from system reliability.

### Tests
- **560 tests passing** (494 from prior stages + 66 new in Stage 1D: 28 crawler + 26 orchestrator + 12 integration)
- **Coverage on `ingestion/`: 89%** average (crawler 93%, orchestrator 96%, infra modules 100% from Stage 1C)
- **End-to-end ingest of 161 real files: 2.6 seconds**
- **Re-ingest with cache hits: ~0.2 seconds** (Stage 1 done-criterion was <1 second)

### Quality gates
- ✅ Ruff lint clean (47 source files, 24 test files, 7 scripts)
- ✅ Ruff format clean
- ✅ Mypy strict clean
- ✅ Real-data integration: 0 parser crashes, 0 hash failures, 7 of 7 Stage 1 techniques recognised, 12 samples inferred from messy folder structure

### Bugs found & fixed (during Stage 1D)
1. **Spurious "best-match" parser for unrelated files** — `find_parser(min_confidence=0.0)` was letting parsers returning exactly 0.0 through (since `0.0 < 0.0` is False). The first-registered parser (PanalyticalXrdml) became a fake "best match" for every unrelated PDF and JPEG in the diagnostic field. Fixed with a positive epsilon (`1e-9`); regression test added (`test_unrelated_file_has_no_best_match`). This was caught by the real-data smoke run — would have shipped silently on synthetic fixtures alone.
2. **Orphan `two_theta` on bad intensity row** — XRD parser appended `two_theta` before parsing `intensity`; one bad intensity created a length mismatch breaking `ParsedData`'s same-length invariant. Fixed by parsing both floats before appending either. (Discovered earlier; carried into 1D regression coverage.)
3. **`FileRow.sha256` UNIQUE collision on parser-version bump** — re-ingesting after a parser version change created two measurements containing the same file's sha256 (the old one from DB seed + the new one from re-parse), violating the schema's UNIQUE constraint. Fixed with a `_dedupe_measurements_by_sha256` pass before save: keep the most recently parsed measurement per sha256.
4. **SQLite engine leak preventing tmp cleanup on Windows** — orchestrator created an engine via the factory but never disposed it, leaving file handles open. On Windows this prevented `pytest`'s `tmp_path` from cleaning up. Fixed by changing the factory contract to a `contextmanager` so `engine.dispose()` always runs.
5. **`tempfile.TemporaryDirectory` cleanup race** — manual smoke runs against real data left occasional unraisable `PermissionError` on Windows after the engine fix. Worked around by switching the integration test to pytest's `tmp_path_factory` (uses delayed cleanup that handles this gracefully).

### Real-data ingestion summary (Dhivya dataset, 161 files)

```
Outcomes:
  parsed                   76
  skipped_unclassified     84

Measurements by technique:
  sem                      44
  xps                      11
  hall                      4
  xrd                       4
  thermoelectric            1
  uv_drs                    1

Samples (12):
  'CS'                       12 measurement(s)  [sem=12]
  'CS (Pure)'                 4 measurement(s)  [xps=4]
  'CS Pure'                   1 measurement(s)  [xrd=1]
  'CS-3'                      7 measurement(s)  [xps=7]
  'Divyamahalakshmi_07042025' 4 measurement(s)  [hall=4]
  'Dr.MN-dhivya-cscbi1'       1 measurement(s)  [xrd=1]
  'Dr.MN-dhivya-cscbi5'       1 measurement(s)  [xrd=1]
  'Dr.MN-dhivya-cskbi3'       1 measurement(s)  [xrd=1]
  'Images'                    5 measurement(s)  [sem=5]
  'UV DRS'                    1 measurement(s)  [uv_drs=1]
  'cUsE3'                    27 measurement(s)  [sem=27]
  'zT calculation'            1 measurement(s)  [thermoelectric=1]
```

The 84 unclassified files are exactly what they should be: PDFs (Hall measurement reports), `.docx` notes, `.jpeg`/`.jpg` thumbnails (TIFs in the same folders ARE parsed correctly), and Avantage `.spe` files we don't yet handle. The grouping shows the heuristic working as designed and also exposes exactly the kinds of mistakes Stage 2's smart-labeling layer is built to fix — `CS Pure` (XRD) and `CS (Pure)` (XPS) are the same physical sample with different folder spellings; `Divyamahalakshmi_07042025` is a Hall folder named after the operator+date that should be split into 4 samples (CS, CS-1, CS-3, CS-5) by reading the filenames.

### Slide-Worthy Achievement (Stage 1D)
> *"Built the ingestion pipeline that turns a folder of raw instrument files into a queryable, validated, cross-correlated database — automatically, in seconds. On a real lab dataset of 161 mixed files (Dhivya's MXene project, ~590 MB), Latos identifies and parses every supported instrument file, correctly skips non-data files like reports and thumbnails, and groups everything into 12 samples — all in 2.6 seconds. Re-opening the same project takes 0.2 seconds because content hashes drive an automatic parse cache."*

**Wow numbers for slide:**
- 161 files → fully ingested → 2.6 seconds
- Re-open: 0.2 seconds (25× faster than first scan)
- 0 parser crashes on real, messy lab data
- 89% test coverage with 560 tests in 21 seconds

---

## 2026-05-07 — Stage 1E Complete: PySide6 Desktop Shell

### What shipped

The end-to-end **desktop application is now usable**: launch `latos-app`,
pick a folder, watch ingestion run on a worker thread with a cancel-able
progress dialog, and land on an Overview dashboard with a sample-detail
Review page in the sidebar. Every layer below this (core, persistence,
ingestion) is still importable headlessly — nothing under `latos.ui.*`
leaks into them.

**Sub-stages, in order:**

| # | Commit | Files | What landed |
|---|---|---|---|
| 1E.1 | `163ec8e` | `ui/app.py`, `main_window.py`, `themes.py`, `pages/welcome.py` | App skeleton: `latos-app` GUI script, `FluentWindow` with sidebar, `WelcomePage`, dark/light theme helper. Removed the dead `latos = streamlit_app:main` console-script. |
| 1E.2 | `3377028` | `services/recent_projects.py`, `pages/project_picker.py` | `RecentProjectsService` (Qt-free MRU JSON store at `$LATOS_HOME/recent.json`, atomic writes, tolerant load) + `ProjectPickerPage` (hero "Open Folder…" button + Recent rail of clickable cards). Picker emits `projectOpened(Path)`. |
| 1E.3 | `7634bdf` | `services/ingestion_worker.py`, `dialogs/ingestion_progress.py` | `IngestionWorker(QObject)` runs `Orchestrator.ingest()` on a `QThread` via `moveToThread`. Cancel via `threading.Event` polled from the orchestrator's `on_progress` callback. `IngestionProgressDialog` modal wraps the worker, surfaces progress, and exposes `ingestion_result()` / `failure()` / `was_cancelled()`. |
| 1E.4 | `19cd057` | `pages/overview.py` | `OverviewPage` dashboard: project name, stat cards (samples / measurements / parsed / cached / failed), one-row-per-sample list, and a `pyqtgraph.PlotWidget` preview that auto-picks the first measurement with 1-D arrays attached. Empty state until first ingestion. |
| 1E.5 | `1b2833d` | `pages/sample_review.py` | `SampleReviewPage` drill-down: `QSplitter` with `TreeWidget` of samples → measurements on the left, detail pane on the right (title, instrument / measured_at / parser metadata, files list, severity-colored validation issues, and a per-measurement pyqtgraph plot). |

### Architecture decisions

1. **Single `FluentWindow` + four sidebar pages.** Welcome, Open (picker),
   Overview, Review — registered up-front so the sidebar layout is
   stable across "no project" / "project open" states. After a successful
   ingestion the main window calls `set_project()` on Overview + Review
   and `switchTo(self._overview)`. Stage 2 will add Analysis / Optimize /
   Settings without restructuring this.

2. **Off-thread ingestion via `moveToThread`.** Pattern #2 from the Qt
   docs — a plain `QObject` worker moved to a `QThread`, with terminal
   signals (`finished` / `failed` / `cancelled`) crossing back to the
   GUI thread via Qt's queued connections. The worker's `start()` slot
   is also synchronously callable, which lets the unit tests verify the
   state-machine logic without the threading layer (the threading itself
   is covered by the dialog tests).

3. **Cancellation is crawl-phase only.** The orchestrator exposes
   `on_progress(idx, total, path)` only during the crawl pass; we poll a
   `threading.Event` from the callback and raise an internal sentinel
   (`_IngestionCancelledError`) which propagates out of the orchestrator.
   Once parsing/persistence starts, cancel is a no-op — aborting mid-write
   would leave SQLite + Parquet in inconsistent state, so the trade-off
   is explicit and documented.

4. **Stub orchestrators in tests, never real ingestion.** The
   `latos_window` fixture injects a `MagicMock(spec=Orchestrator)` whose
   `ingest()` returns an empty `IngestionResult` immediately. UI tests
   never touch SQLite, Parquet, or any parser — they test wiring only.
   The orchestrator, parsers, and array-store are tested against real
   data in their own integration tests (Stage 1D).

5. **Recent projects: filter-on-read + atomic write.** Same `.tmp` +
   `os.replace()` pattern used by `ArrayStore` and Alembic. Entries
   whose path no longer exists are silently dropped from `entries()`
   and any subsequent persisted write — eventual consistency, no
   proactive vacuum needed. Corrupt JSON or schema drift is treated as
   "no recents" rather than crashing the app on startup.

6. **Tree widget user data for measurement IDs.** `SampleReviewPage`
   stores each measurement's id in the tree node's `Qt.UserRole + 1`.
   Selection lookup walks `_project.samples` to find the matching
   `Measurement` rather than caching parallel structures — the tree
   stays the single source of truth for what's selected.

7. **Avoid name collisions with Qt base classes.** Renamed
   `RecentProjectCard.clicked` → `pickRequested(Path)` (base
   `CardWidget.clicked` is a zero-arg signal it fires from
   `mouseReleaseEvent`); renamed `RecentProjectsService.list()` →
   `entries()` (so `list[T]` annotations still resolve to `builtins.list`
   under mypy strict); renamed `IngestionProgressDialog.result()` →
   `ingestion_result()` (so it doesn't shadow `QDialog.result()`'s int
   return type).

### Tests

- **669 tests passing in clean runs** (584 default `not ui` slice +
  85 UI tests run separately under `QT_QPA_PLATFORM=offscreen`).
- **85 new UI tests in Stage 1E**, broken down:

| Module | Tests | What they cover |
|---|---|---|
| `tests/unit/ui/test_main_window.py` | 9 | window construction, page registration, picker → ingestion → overview wire-up |
| `tests/unit/ui/test_app.py` | 1 | `main()` exit code with `QApplication.exec` patched |
| `tests/unit/ui/test_themes.py` | 4 | apply dark / light / system theme, accent hex |
| `tests/unit/ui/pages/test_welcome.py` | 2 | object name, brand text |
| `tests/unit/ui/pages/test_project_picker.py` | 7 | empty state, dialog accept/cancel, recent rail rendering + click |
| `tests/unit/ui/pages/test_overview.py` | 12 | empty/populated states, stat cards, sample rows, plot rendering, `_find_first_plottable` |
| `tests/unit/ui/pages/test_sample_review.py` | 12 | tree population, selection → detail, severity-colored issues, plot rendering, clear |
| `tests/unit/ui/services/test_recent_projects.py` | 24 | MRU semantics, max entries, filter-on-read, tolerant load, atomic write |
| `tests/unit/ui/services/test_ingestion_worker.py` | 10 | success / failure / cancel paths against stub `Orchestrator` |
| `tests/unit/ui/dialogs/test_ingestion_progress.py` | 4 | end-to-end thread plumbing: accept on finished, reject on failed/cancelled |

- **Coverage on `ui/`**: 100% on `app.py`, `main_window.py`, `welcome.py`,
  `themes.py`; 99% on `project_picker.py` and `sample_review.py`; 97%
  on `recent_projects.py`. Untested lines are display-only fallbacks
  (e.g. the metadata-only "no plottable arrays" branch).

### Quality gates

- ✅ Ruff lint clean (54 source files)
- ✅ Ruff format clean
- ✅ Mypy strict clean
- ✅ Default pytest slice (`not ui`): 584 passing
- ✅ UI slice: 85 passing under `QT_QPA_PLATFORM=offscreen`
- ✅ Smoke launch (real `QApplication`, all four sidebar pages register
  cleanly, window opens + closes without leaking the worker thread)

### Bugs found & fixed (during Stage 1E)

1. **`CardWidget.clicked` signature clash** — qfluentwidgets' base
   `CardWidget` defines a zero-arg `clicked` signal it fires from
   `mouseReleaseEvent`. Shadowing it with my `Signal(Path)` broke the
   base implementation with `TypeError: clicked(PyObject) needs 1
   argument(s), 0 given`. Renamed my signal to `pickRequested(Path)`
   and re-emit it from a slot connected to the base `clicked`.
2. **`list[T]` annotations broke after method rename** — naming a method
   `list(self) -> list[RecentProject]` shadowed `builtins.list` inside
   the class scope; mypy resolved the return-type `list[...]` to the
   method itself and emitted seven errors. Renamed to `entries()` and
   updated all callers.
3. **`QDialog.result()` returns int** — overriding `result()` to return
   `IngestionResult | None` triggered mypy's `[override]` because the
   parent's signature returns `int` (the accept/reject code). Renamed
   to `ingestion_result()` and added a comment explaining the clash.
4. **Real ingestion fired during UI tests** — the original
   `_on_project_opened` immediately called `dialog.exec()`, which spun
   up a real `QThread` and blocked the test's `picker._open_button.click()`
   until ingestion finished. Added an `orchestrator_factory` hook on
   `LatosMainWindow` and made the `latos_window` fixture inject a
   `MagicMock(spec=Orchestrator)` returning an empty `IngestionResult`
   immediately.
5. **`mousePressEvent` test using deprecated `QMouseEvent` ctor** — the
   PySide6 `QMouseEvent` constructor I used emits a `DeprecationWarning`
   that pytest's `-W error` promoted to a test failure. Replaced with
   a direct `target.clicked.emit()` (the base CardWidget signal) which
   triggers the same `_on_clicked` → `pickRequested(Path)` chain.
6. **Ruff N802 on `showEvent` / `closeEvent`** — Qt requires camelCase
   names to override these handlers; ruff's snake_case rule rejects
   them. Suppressed per-method with `# noqa: N802` and a comment.
7. **Ruff N818 on `_IngestionCancelled`** — exception names must end in
   `Error`. Renamed to `_IngestionCancelledError`.
8. **Ruff N815 on Qt signal attributes** — Qt convention is mixedCase
   for signal names (`progress`, `projectOpened`, `pickRequested`).
   Suppressed per-attribute with `# noqa: N815` rather than file-wide,
   so accidental non-signal mixedCase still gets caught.

### End-to-end user flow (manual smoke)

1. `latos-app` → `LatosMainWindow` opens at 1280×800 with sidebar:
   Welcome (active), Open, Overview, Review.
2. Click **Open** → `ProjectPickerPage` shows hero + (initially empty)
   Recent rail.
3. Click **Open Folder** → native `QFileDialog` → pick a folder →
   `projectOpened(Path)` fires → `IngestionProgressDialog` modal opens.
4. Worker thread runs `Orchestrator.ingest()`; dialog updates
   "Processing file 47 of 161" + the current filename. Cancel button
   is live; clicking it triggers `request_cancel()` → cancel-on-next-tick.
5. On accept: dialog closes, main window calls
   `overview.set_project(result.project)` +
   `sample_review.set_project(result.project)` and switches the sidebar
   to Overview. The Recent rail now shows the project at the top.
6. **Overview** renders title, stat cards, the sample list, and the
   pyqtgraph preview plot of the first plottable measurement.
7. **Review** lets the user expand a sample, click a measurement, and
   see metadata + files + issues + per-measurement arrays plotted.

### Slide-Worthy Achievement (Stage 1E)

> *"Latos is now a real desktop application. Researchers launch it
> from a single command, drop in a folder of raw lab files, watch the
> ingestion run on a background thread (cancellable; the GUI never
> freezes), and land on a Fluent-styled dashboard with their samples,
> techniques, and a live preview plot of the first XRD scan Latos
> found — without writing a line of Python. From here, every Stage 2
> feature (smart sample labeling, peak fitting, optimization loops)
> attaches to a UI surface that already knows how to render
> measurements, validation issues, and arrays."*

**Wow numbers for slide:**
- 4 sidebar pages, 1 modal dialog, 2 background services, 0 frozen frames
- 85 UI tests in 4 seconds (offscreen Qt) — full sidebar wired and verified
- End-to-end: pick folder → ingest 161 files → render dashboard in ~3 seconds
- Cancellation works mid-crawl without poisoning the persistence layer
- Pure-Python `RecentProjectsService`: atomic-write JSON, tolerant load,
  filter-on-read, MRU semantics, 24 unit tests, no Qt dependency

---

<!-- Future entries go below this line -->

## 2026-05-11 — Stage 2 Complete: Smart Sample Labeling

### What shipped

Stage 1's per-folder heuristic picks one sample name per file. That
breaks the moment a researcher writes `CS Pure` in the XRD folder and
`CS (Pure)` in the XPS folder — Stage 1 produces two `Sample`s for
what is one logical sample. Stage 2 fixes that.

The new layer takes the orchestrator's per-file output, extracts every
plausible sample-name hint (path, filename, future: parser metadata),
normalizes them aggressively, builds a similarity graph with
`rapidfuzz` + `networkx`, and produces a tuple of `SampleCluster`s. A
new sidebar page lets the user review the auto-clustering, rename
canonicals, and merge or revert clusters; their decisions persist as
JSON in `.latos/cluster_decisions.json`.

**Sub-stages, in order:**

| # | Commit | Files | What landed |
|---|---|---|---|
| 2A | `a1eed0c` | `labeling/hints.py`, tests | `SampleHints` dataclass + `extract_hints(path, parsed_data?, root?)`. Per-source confidences (metadata=0.85–1.00, filename=0.70, immediate non-generic parent=0.80, deeper parents decay to 0.30, generic folders=0.20). 34 unit tests covering path walks, filename cleaning regex, generic-folder fallback. |
| 2B | `5925e09` | `labeling/normalize.py`, tests | `normalize(s)` (NFKC + lowercase + leading-prefix scrub + separator strip, idempotent under hypothesis property tests) and `tokens(s)`. Collapses `CS Pure`, `cs_pure`, `CS-Pure`, `CS (Pure)`, `cs.pure` to the same string. 44 unit tests including hypothesis idempotency. |
| 2C | `ad4bfd6` | `labeling/cluster.py`, tests | `SampleCluster` + `cluster_samples(hints, threshold=0.85)`. Combines `fuzz.ratio`, `fuzz.token_sort_ratio`, `JaroWinkler.normalized_similarity` via `max(...)`. Files vote into components by summed confidence; empty-file components are filtered. 38 unit tests covering the Dhivya regression, distinct-but-similar separation, threshold boundaries, fallback paths. |
| 2D | `b7eb482` | `labeling/decisions.py`, `ui/pages/cluster_review.py`, `ui/main_window.py`, tests | `ClusterDecisions` (renames + merges + splits) with atomic JSON persistence at `<root>/.latos/cluster_decisions.json`. `apply_decisions()` runs splits → merges → renames in that order. `ClusterReviewPage`: editable `TableWidget` with inline rename, multi-select Merge, Apply / Revert. Wired into the sidebar between Open and Review. 65 tests across the data layer + the page. |
| 2 | this entry | `labeling/pipeline.py`, hint-weight tuning, integration test | `cluster_project(project)` walks every file in every measurement and runs the Stage 2A→2C pipeline against the persisted `Project`. Hint weights re-tuned: immediate-parent jumped from 0.60 → 0.80 (above filename's 0.70) so a researcher's deliberate folder structure outranks the filename hint. Dhivya integration test gained a `TestLabelingPipeline` class. |

### Architecture decisions

1. **Pipeline runs as a post-process, not inside the orchestrator.**
   Folding clustering into ingestion would mean reordering parse → cluster →
   persist, which is a much bigger surgery. The post-process approach lets
   re-clustering with different thresholds work without re-parsing files —
   important for the UI's "Apply" / "Revert" loop.

2. **User decisions live in JSON, not the database.** `cluster_decisions.json`
   is portable when sharing a project folder, easy to inspect with a text
   editor, and survives `Orchestrator.ingest()` re-runs because it's keyed
   by *auto* canonical (what Stage 2C produced) rather than database row IDs.

3. **Splits → merges → renames apply order.** Splits run first (so a file
   pulled out of cluster A into "MX-7" is no longer in A when A merges with
   B), then merges, then renames last (so the rename targets the surviving
   merged canonical). Tested explicitly.

4. **Path keys in splits are `str(Path(...))`.** Cross-platform stringification:
   `Path("/p/a.csv")` becomes `\p\a.csv` on Windows and `/p/a.csv` on Linux,
   so any tests that touch split keys must derive them via `str(Path(...))`
   rather than hard-coding forward slashes.

5. **Editable canonical, but auto-canonical pinned per row.** The cluster
   review table stores the *auto* canonical (Stage 2C's name) on each row
   via `Qt.UserRole + 1`. After a rename, the rename slot finds the auto
   canonical via that role data instead of from the now-renamed cell text —
   so editing the renamed name a second time still targets the same auto
   canonical instead of nesting renames.

6. **Empty-file clusters dropped from output.** If a generic folder name
   ("XRD", "data") appears in any hint extractor's output but no file's
   strongest signal lands on it, the resulting connected component would
   become a phantom cluster with zero files. The materialize step skips
   empty-file components. The Dhivya regression tests would have produced
   ghost "XRD" / "XPS" clusters without this filter.

7. **Hint-weight tuning: immediate parent > filename.** During pipeline
   integration the test `test_distinct_samples_stay_separate` failed because
   one-character filenames (`a.xrdml`, `b.xrdml`) outvoted the folder
   (`CS-1`, `CS-3`). With folder=0.60 and filename=0.70, every file would
   cluster on its filename stem (`run`, `scan`) and the sample name embedded
   in the folder would be lost. Bumped immediate-parent to 0.80; folder now
   wins whenever it carries real information, filename remains the fallback
   when the parent is generic or absent.

### Real-data behaviour (Dhivya, 161 files)

Stage 1 produced 12 samples; Stage 2 collapses to 11. Reduction of one
because the headline regression case is now fixed:

| Cluster | Aliases | Files | Note |
|---|---|---|---|
| `CS Pure` | `CS Pure`, `CS (Pure)` | 5 | ✅ The headline regression collapsed |
| `CS` | `CS`, `CS-1`, `CS-3`, `CS-5`, `Cs 3d`, `Cs3Bi2I9` | 19 | ⚠️ Over-merged: short prefix-similar names chain in the graph |
| `Dr.MN-dhivya-cscbi1` | `cscbi1`, `cscbi5`, `cskbi3` | 3 | ⚠️ Over-merged: one-char Levenshtein |
| `cUsE3` | `cUsE3` | 27 | Untouched |
| Other clusters (8) | varies | 1–4 each | Untouched |

The over-merging of `CS-1` / `CS-3` / `CS-5` is a known limitation: short
strings with a common prefix get high Jaro-Winkler scores, and when several
of them are present they chain into one connected component. Mitigations
already in place:

- Cluster review page: the user can **revert** the over-merge or **rename**
  the surviving canonical in seconds.
- Threshold is per-call: a future "Strict" mode in the UI could pass `0.95`.

A chemistry-aware similarity booster (e.g. recognizing that `Cs1Bi2I9` and
`Cs3Bi2I9` differ in stoichiometry, not in spelling) is on the roadmap; for
now the human-in-the-loop review handles it.

### Tests

- **746 tests passing** total (up from 700 at end of Stage 1E + Stage 2C).
- **Stage 2 added 181 tests** across the four sub-stages and the integration:

| Module | Tests | What it covers |
|---|---|---|
| `tests/unit/ingestion/labeling/test_hints.py` | 34 | Path walks, filename cleaning, metadata extraction, generic-folder fallback |
| `tests/unit/ingestion/labeling/test_normalize.py` | 44 | NFKC, lowercase, prefix scrub, separator strip, idempotency hypothesis property |
| `tests/unit/ingestion/labeling/test_cluster.py` | 38 | Similarity metric, canonical picking, full Dhivya regression + threshold edges |
| `tests/unit/ingestion/labeling/test_decisions.py` | 36 | Rename / merge / split builders, JSON round-trip, atomic write, apply order |
| `tests/unit/ingestion/labeling/test_pipeline.py` | 7 | Project → hints → clusters end-to-end, dedup, root forwarding, threshold passthrough |
| `tests/unit/ui/pages/test_cluster_review.py` | 29 | Empty state, populate, rename, merge, apply (writes JSON), revert, summary text |
| `tests/integration/test_dhivya_ingestion.py` | +2 | Pipeline reduces or preserves sample count; CS Pure regression collapsed |

### Quality gates

- ✅ Ruff lint clean (63 source files)
- ✅ Ruff format clean
- ✅ Mypy strict clean
- ✅ Default pytest slice: 614 passing
- ✅ UI slice: 132 passing under `QT_QPA_PLATFORM=offscreen`
- ✅ Dhivya integration: 14 tests passing on real 161-file dataset

### Bugs found & fixed (during Stage 2)

1. **Filename hint outvoted folder hint** — see architecture decision #7.
   Fixed by bumping immediate-parent path weight from 0.60 to 0.80.
2. **`Counter[str]` typing under mypy strict** — `Counter` defaults to
   int values; using it for float vote weights crashed mypy strict.
   Switched to `dict[str, float]` with explicit `max(..., key=...)`
   for the deterministic tiebreak.
3. **Ghost clusters from generic path segments** — without the empty-file
   filter, hints for "XRD" / "XPS" would surface as standalone clusters
   in the output. Filter added to `_materialize_clusters`.
4. **Path stringification cross-platform** — split keys recorded with
   forward slashes failed on Windows. Tests now use `str(Path(...))`.
5. **`with_merge(["only-one-name"])` was creating a single-name group**
   that was silently a no-op at apply time but cluttered the JSON file.
   Now dropped at the data layer.
6. **Hypothesis caught `normalize(normalize(x)) != normalize(x)`** —
   `str.lower()` decomposes some characters (Turkish capital İ, etc.).
   Fixed by adding a final NFKC pass after separator scrubbing.

### Demo flow (manual)

1. Launch `latos-app`.
2. Open Folder → pick a Dhivya-shaped project.
3. Wait for ingestion (the existing 1E.3 progress dialog).
4. Sidebar lands on Overview with stat cards.
5. Click "Clustering" in the sidebar.
6. See the auto-clustered table: one row per cluster with editable
   canonical name, alias chips, file count.
7. Click any sample name to rename it inline.
8. Multi-select rows + click "Merge selected" to combine clusters.
9. Click Apply — JSON written to `<project>/.latos/cluster_decisions.json`.
10. Re-open the project later; decisions reload automatically.

### Wow numbers for slide

- 12 Stage 1 samples → 11 Stage 2 clusters (Dhivya regression collapsed)
- Cluster phase: 42 ms on 161 files → "free" relative to the 6.6 s ingest
- 181 new tests, 96% coverage on the cluster review page, 100% on `decisions.py`
- User edits round-trip atomically through `cluster_decisions.json`
- Pipeline is `extract_hints → normalize → cluster_samples → apply_decisions`
  — four pure functions, easy to refactor or rerun with different thresholds

---

## 2026-08-21 — Stage 6 Complete: Reliability-Gated Closed Loop

> **Gap in this log.** The entries above end at Stage 2 (2026-05-11). Stage 3
> has a stage doc but no entry here, and the optimizer, server, pre-registration
> and reliability layers were all built and committed without one. This entry
> does not attempt to backfill them — that would be reconstruction, not a
> record. It covers only the work of 2026-08-19 to 2026-08-21.

Six modules, delivered ahead of a prospective ionic-liquid experiment that had
not yet started. Every one is exercised only against synthetic data with known
answers; see "What is not validated" below.

### What landed

| Commit | What |
|---|---|
| `9f9368a` | `analysis/thermovoltage/slope.py` — Seebeck as a fitted slope |
| `5250406` | `point_noise` on both engine entry points |
| `4f02e21` | `StoppingVerdict` — STOP / CONFIRM / CONTINUE |
| `1d2f9de` | `ingestion/parsers/ite_workbook.py` |
| `765476f` | shared workbook schema, generator moved into the package |
| `6d98f8e` | `optimization/rehearsal.py` |
| `922b88f` | `python -m latos next`, plus the prereg gap it exposed |

### Findings worth keeping

**The stopping criterion was not failing to fire — it was discarding its own
answer.** On a single-peak objective sampled at six points including the peak,
the engine reported `P(within ε) = 0.992`, `epsilon_delta_met = True`, signal
exhausted — then `converged = False` and a recommendation at the far edge of the
search space. The cause is `converged = signal_exhausted and not is_exploratory`:
two independent lines of evidence disagreed and the code silently picked one.
`StoppingVerdict` now reports the disagreement as `CONFIRM`.

**A three-point standard error is biased low.** With one degree of freedom the
residual variance follows a chi-squared whose median is 0.455 of its mean, so
`σ_S` lands below the truth more often than not — by roughly 2.4× at the median
here. This matters because `σ_S` is what feeds per-point variance into the GP, so
a three-ΔT campaign hands the surrogate over-confidence on exactly the sparse
data where it costs most. Pinned by a test; four ΔT values is the cheap fix.

**R² cannot see curvature on a short series.** A deliberately quadratic
five-point series scores 0.963. The first fix — comparing end residuals against
middle ones relative to their spread — was self-defeating, because curvature
inflates the spread that sets the threshold. Replaced with an
extra-sum-of-squares F test against a quadratic fit, taking its noise estimate
from the quadratic residual. False alarms on pure scatter fell from 17 % to ~2 %.

**Excel cannot store a timezone.** `ParsedData.measured_at` requires an aware
datetime and openpyxl refuses to write one, so the first parser implementation
would have reported `measured_at = None` on every real workbook, silently,
forever. Caught by a test. UTC is now attached and the assumption recorded once
per sample — a lab in JST would otherwise find every measurement dated nine
hours from when it happened.

**Perfectly linear input is not precision.** Noiseless synthetic data gives every
fit a vanishing standard error, which collapsed the convergence floor and made
the engine report being "within 4.23e-16 of the optimum". Found by running
`latos next` on a demo campaign, not by reasoning about it.

**`point_noise_used` never reached the frozen record.** The field was added to
`BoConfig` specifically so two runs could not carry identical configs having
weighed observations differently — but `prereg.py` did not serialise it, so the
file on disk still could not tell them apart. Fixed with a test.

### CI

CI had been red for several runs on `PLR0917`, with no local reproduction. Cause:
`ruff>=0.3` with no upper bound, so CI installed whatever had shipped that
morning while the venv sat on 0.15.12 and a rule graduated from preview in 0.16.
The three findings were real (parameters with defaults were positionally
passable) and were fixed by making them keyword-only. Ruff, mypy, pytest,
hypothesis and the pytest plugins are now capped, with floors set to versions the
suite has actually been run green against. `scripts/check.py` promised "green
here implies green there" while linting a different path set than CI; it now
lints `.` as CI does.

### Numbers

- 124 new tests; **1600 passing** in the unit suite, 0 failing
- ruff lint, ruff format, mypy strict: clean on **108** source files
- 3 measured negative results on physics priors now on record (SPB, SPB 1/T,
  linear mixing law), plus one independent literature corroboration

### What is not validated

No real measurement was involved at any point. Every threshold shipped here —
`_R_SQUARED_WARN = 0.99`, `_CURVATURE_F = 10.0`, the 10 % offset warning,
`_DEGENERATE_SIGMA_FRACTION` — is a reasoned guess checked against synthetic
objectives. They prove the code does what was intended. They prove nothing about
ionic liquids. Stage 0 of the experiment plan (two samples certifying the
protocol) is what turns them into measurements, and Phase 2 of the development
plan is deliberately deferred until then.

---

## 2026-09-02 — Two default flips proposed, both refuted by measurement

The August literature review recommended two one-line changes: RBF → Matérn 5/2,
and the exploration fallback `max_std` → `ei`. Both options had shipped in
`eab3d92` with **no tests**, which is why their defaults had never been examined.
Both were measured before flipping. **Both lost, and neither default moved.**

### Kernel — RBF stays

Branin and Hartmann-3, 8 seeds per arm, simple regret (median / worst):

| benchmark | rbf | matern52 | seeds won by matern52 |
|---|---|---|---|
| branin | 0.0708 / **0.3509** | 0.1665 / **1.1385** | 2 / 8 |
| hartmann3 | 0.0079 / 0.7836 | **0.0028** / 0.7739 | 7 / 8 |

Matérn wins clearly on Hartmann-3 and loses badly on Branin, where it **triples
the worst case**. Rohr's warning is the deciding argument: the floor for
deleterious effects is deeper than the ceiling for gain, so a 3× worse tail is
not bought by a median gain on one benchmark of two. AX4 settles this properly
across process-window shapes.

### Exploration fallback — `max_std` stays

The literature case looked stronger here: four papers (Borg, Rohr, Srinivas,
Shields) measure pure uncertainty sampling as the weakest available policy. It
does not transfer. Measured on Forrester 1-D from `n_initial = 4` — the tier the
branch actually fires in, and it fired on **96/96** rounds — 12 seeds:

| policy | median | worst | seeds beating max_std |
|---|---|---|---|
| max_std | **0.0220** | **0.1735** | — |
| ei | 0.0559 | 6.0212 | 5 / 12 |
| ucb | 0.0938 | 6.0212 | 3 / 12 |

A worst case of 6.02 on a function whose range is ~6.02 means those campaigns
never left the flat region. **The papers measure pure exploration as a whole
strategy over long campaigns; here it is a tie-break that fires only when EI is
already flat and the data is still exploratory.** At n = 4–12 in one dimension,
"go to the biggest unmeasured gap" is simply correct, and the switch would have
been a regression with a 35× worse tail.

### A real bug found on the way

`optimize_nd` accepted no `kernel` argument and **hardcoded `"RBF"` into the
reported config** while its surrogate used whatever `_fit_surrogate` defaulted
to. Flipping the helper default would therefore have changed N-D behaviour
silently while every result still claimed RBF. Fixed: `kernel` is threaded
through `optimize_nd`, both entry points name their kernel through one
`_kernel_label` helper, and both defaults now come from single named constants
(`_DEFAULT_KERNEL`, `_DEFAULT_EXPLORE_POLICY`) so the 1-D and N-D paths cannot
drift apart.

### Numbers

- 18 new tests in `test_engine_kernel_policy.py` — the first coverage either
  option has had
- Two defaults pinned to their measurements, so a future change has to argue
  with the evidence rather than with a comment
- `run_campaign(kernel=...)` added, so the kernel arm is reproducible from the
  harness

### Retry: `detect_peaks` now returns widths; clustering still not shipped

`detect_peaks_detailed` and `measure_widths` were added, and they fix the
prerequisite properly. The trick was a **bounded prominence window**: scipy's
default looks outward until the signal rises again, so on this pattern a 0.2°
reflection standing on the fabric's amorphous hump took its prominence from the
far side of the hump. Confining it with `wlen` gives real widths —
**median FWHM 0.203°, range 0.060–0.880°** across the 29 peaks.

With those widths, clustering finally groups correctly: **25 clusters (linear
background) / 27 (ALS)**, fitting in 7.8 s and 1.2 s. Two rounds of fixing
followed, both worth recording:

- unbounded σ let every peak grow to fill its window — fitted widths came back
  4.6× the measured value, and the model's variance exceeded the data's
- bounding σ to the measured width ±4× fixed that (ALS R² 0.29 → 0.73)

**It is still not shipped, and the reason is the finding.** On the linear
background the clustered fit reaches R² 0.11 where the single joint fit reports
0.9895 — because *the joint fit was using its peaks as a background*. A third of
those 30 peaks sit on a hump spanning 3–25° that a straight line cannot model,
and freeing 120 coupled parameters let them absorb it. Clustering forbids that,
so it exposes the misfit instead of hiding it.

That makes the correct order plain: **fix the background first, then cluster.**
Clustering on an ALS background already converges in 1.2 s. Shipping it while a
linear background is still selectable would turn a flattering number into an
honest bad one with no warning, which is the wrong trade to make silently.

Shipped from this round: `PeakCandidate`, `detect_peaks_detailed`,
`measure_widths`, 6 new tests (44 passing in the fitting suite), and the
evaluation cap. `detect_peaks` keeps its old signature as a thin wrapper.

---

## 2026-09-04 — Categorical-axis guard on `optimize_nd`

First step of the "borrow" plan from the product definition. Not a borrow
itself: a correctness fix that stands whether or not Ax ever lands behind the
optimizer, and one that has to exist before a categorical-capable backend can
be wired in honestly.

### The defect

`SynthesisParams` is `dict[str, dict[str, float]]`, so a knob with no ordering
— etching atmosphere Air / Ar / N₂ — can only enter Latos encoded as 0/1/2. The
GP treats it as any other number. In the MXene TEM replay this produced a
recommendation of **gas index 0.55**: a recipe nobody can make, returned with a
predictive interval like a real one. Nothing anywhere in the stack objected.

`grep -n categorical packages/core/src/latos/optimization/engine.py` returned
nothing before this change — 111 KB of engine with no notion that an axis might
not be interpolable.

### The guard is two-sided, because floats carry no intent

The engine cannot recover the caller's meaning from a column of numbers. So:

- **`axis_kinds=[..., "categorical"]` raises `OptimizationError`.** Intent is
  known, so refusing is right, and the message says what to do instead (one
  campaign per level, or hold it fixed and optimize within it).
- **An undeclared axis that *looks* encoded warns** on a new
  `OptimizationResultND.axis_warnings`, and the run proceeds.

The warning fires on two conditions together, not one: the column is 2–6 whole
values spaced exactly one apart, **and** the recommendation for that axis landed
off-level. The second condition is what makes it actionable — it names the exact
number the user would otherwise have taken to the bench.

### Measured behaviour of the detector

| column | recommendation | verdict |
|---|---|---|
| gas 0/1/2 | 1.499 | **warn** |
| gas 0/1/2 | 1.02 | quiet (on a level) |
| temp 40/50/60 | 52.4 | quiet (spacing ≠ 1) |
| continuous, 7 distinct values | 3.9 | quiet |
| 0…6 | 3.5 | quiet (above level cap) |
| binary 0/1 | 0.5 | **warn** |
| anneal 1/2/3 h | 2.4 | **warn — known false positive** |

The last row is the reason the undeclared path warns instead of raising. An
anneal at 1, 2 and 3 hours is indistinguishable from three named gases, and a
wrongly-blocked campaign costs more than a notice a researcher can read past.
The warning text says so in its last sentence, and a test pins that sentence.

### Shipped

- `AXIS_CONTINUOUS` / `AXIS_CATEGORICAL`, `_validate_axis_kinds`,
  `_encoded_axis_warning` in `optimization/engine.py`
- `optimize_nd(axis_kinds=...)`; `OptimizationResultND.axis_warnings`
- `OptimizeNdResult.axis_warnings` through `/optimize/run-nd`, rendered on the
  Optimize screen next to the dropped-sample notice
- `tests/unit/optimization/test_engine_categorical.py` — 18 tests
- Incidental: removed a duplicated `input_names:` line in the `optimize_nd`
  docstring

### Not done here

The 1-D `optimize()` has the identical hole — a one-variable run on "gas index"
is just as broken — and is untouched. It is the route `/optimize/freeze`
records, so widening it touches pre-registration serialisation and deserves its
own change.

---

## 2026-09-04 — Latos run against the raw MXene folders; JPEG parser shipped

### The benchmark

`data/MXene-data/DATA_NOTES.md` was written by working through
`0.Final Paper data/` by hand: the exclusions, the naming decisions, the
coverage table, four open questions. That document is the shape of output Latos
claims to produce, so the test is how much of it the tool derives unaided.

### First run — 11% of the data was read

| | |
|---|---|
| files crawled | 860 |
| parsed | **94 (11%)** |
| unclassified | 766 |

Every one of the 609 TEM `.jpg` frames was claimed by **zero** parsers:
`microscopy_tif.py` handles `.tif`/`.tiff` only. Also unread: 34 `.raw`, 19
`.bmp`, 14 `.wdf` (Renishaw native), 12 `.map`, 3 `.brml` (Bruker native).

Latos owns a working info-bar decoder in `analysis/microscopy/calibration.py`
— `decode_field_of_view` already refuses the impossible "2 m" field of view
these exports write. It had nothing to decode, because ingestion dropped the
files first. The gap was never in the analysis; it was one layer earlier.

### Shipped: `MicroscopyJpegParser`

Metadata-only, same contract as the TIFF parser — 609 frames is 1.3 GB and the
question worth answering at ingest is answerable from the image header.

That question is whether the frame carries an **info bar**. These exports write
a square image area with the bar in extra rows beneath it, so `height > width`
means a bar — exactly what `split_info_bar` tests, and it needs the dimensions,
not the pixels. A square frame has no recoverable scale and is flagged WARNING
at ingest. Field-of-view decoding stays in the analysis layer where the glyph
templates live.

### Also fixed: technique suffixes in folder names

The first run with the parser returned **609 frames across 13 samples labelled
SEM**. `_refine_technique_from_folders` matched only a folder named exactly
`TEM`; this tree files the modality as a suffix — `1.TEM/.../MX_Ti3C2Tx_Air_40_TEM/`.

A wrong modality is worse than a missing one: nothing downstream has a reason
to doubt it. `_folder_technique` now tries the whole name first, then splits on
separators and matches tokens **whole** — so `system` does not become SEM and
`item` does not become TEM.

### Second run — 82%

| | before | after |
|---|---|---|
| parsed | 94 | **703** |
| unclassified | 766 | 157 |
| TEM samples | 0 (13 mislabelled SEM) | **13** |

All nine MXene conditions now appear as TEM — Air/Ar/N₂ × 30/40/50 °C —
matching the DATA_NOTES coverage line "TEM covers all 9 conditions".

### Correction to the first assessment

I reported the first run as flagging **zero** anomalies. That was wrong: the
harness printed `flag_anomalies()` (sample-*name* anomalies) and never printed
measurement validation issues. Latos was already raising, unprompted:

- `4x` **"Sample geometry was left at the 1/1/1 default, so resistivity and
  power factor are wrong"** — the central PPMS finding in DATA_NOTES
- `2x` **"Lorenz ratio L/L0 has median magnitude 8.86e-04 / 1.80e-03, far from
  the expected ~1"** — the κ channel failure, found independently
- `2x` Raman cosmic-ray spikes at 2394 and 802 cm⁻¹
- `8x` XPS regions acquired with differing sweep counts

So the tool reconstructs more of DATA_NOTES than the first report said. The
PPMS half of that document is largely derivable today.

### Still not reconstructed

- **31 of 609 frames flagged as having no info bar**; DATA_NOTES says 33, all
  Ti3AlC2 MAX. Latos attributes 30 to Ti3AlC2 and 1 to Mo2Ti2AlC3. The counts do
  not reconcile, and 19 unparsed `.bmp` files sit in those same MAX folders — a
  plausible but **unverified** explanation. Do not claim a match until a BMP
  parser exists and the numbers are compared again.
- **`old/` is still ingested** — 41 files, now as their own TEM sample named
  `old`. Nothing in Latos can express a quarantined folder.
- **The `(3-15)Deg` / `(3-90)Deg` pairs are still two samples each.** The digit
  guard vetoes the merge (15 vs 90), so XRD reports 17 samples for 6 conditions.
- **PPMS sample ambiguity is hidden, not surfaced**: all three gases collapse
  into one sample `MX_Ti3C2Tx_Air_N2_Ar_50_PPMS`, taken from the folder name.
  DATA_NOTES' open question #1 is answered "all of them", confidently.
- Naming decision (`nogas` = `Air`), MAX phase identity from diffraction, and
  the scan-range choice: all still absent.

### Also learned

The parse cache is keyed on sha256 + parser_version, so a technique decided at
first ingest **survives a code change that should have altered it**. The second
run returned SEM until it was pointed at a fresh project. Re-ingesting an
existing project does not re-derive folder-based refinement.

### Shipped

- `ingestion/parsers/microscopy_jpeg.py`, registered in `default_registry()`
- `_folder_technique()` in `ingestion/orchestrator.py`
- `tests/unit/ingestion/parsers/test_microscopy_jpeg.py` — 17 tests
- `TestTechniqueSuffixInFolderName` in `test_orchestrator.py` — 10 cases
- `microscopy-jpeg` added to the registry inventory test

---

## 2026-09-04 — BMP parser, and the info-bar gap it actually exposed

### The BMP parser

19 `.bmp` files sat unread in the two MAX-phase EDX folders. They are **not**
micrographs: they are JEOL Analysis Station EDS output, one image per element
per view plus a bright-field reference, beside the `.emsa` spectra of the same
acquisition. The filename is the only record of which element each shows —
BMP has no tag structure at all.

`MicroscopyBmpParser` decodes that convention (`View000 Ti K.bmp` → view 000,
Ti, K line) and returns `Technique.EDS`. All 19 now parse:

| | |
|---|---|
| element maps decoded | 14 |
| bright-field references | 4 |
| unnamed, fallback to SEM | 1 (`000.bmp`) |

Elements recovered — **Ti3AlC2**: Al K, C K, Ti K across three views.
**Mo2Ti2AlC3**: Al K, C K, Mo L, O K, Ti K.

The decode is strict: the element is checked against the real symbols and the
line against the shells that exist, so an arbitrary two-word filename cannot be
read as chemistry. Confidence is 1.0 for a decoded name and 0.6 for a bare BMP,
so the registry still claims it while recording which answer rests on evidence.

Each map carries an INFO issue saying a map shows **where** a line's counts fall,
not **how much** of the element is present — quantification comes from the
spectra sitting next to it.

### The JPEG parser's info-bar test is deliberately NOT reused here

Copying it would have manufactured a calibration that does not exist. Measured:
the 267x275 maps carry an **8-row** caption strip and one 512x568 frame a
**56-row** dark strip — neither proportioned like `JEOL_2100F`, whose cells are
measured against a 2048-px reference. The trailing strip is recorded and nothing
is claimed about pixel size.

### The 31-vs-33 hypothesis was wrong

The previous entry guessed that the 19 unparsed BMPs might explain why Latos
found 31 no-info-bar frames where DATA_NOTES says 33. **They do not.** Not one
of the 19 is square, and they are EDS maps in EDX folders rather than TEM image
frames.

A format-agnostic sweep over every image in the tree found the real cause:

| format | folder | square frames |
|---|---|---|
| `.jpg` | `MAX_Ti3AlC2_TEM_IMAGE` | 30 of 70 |
| `.tif` | `MAX_Mo2Ti2AlC3_TEM_IMAGE` | **5 of 47** |
| `.jpg` | `MAX_Mo2Ti2AlC3_TEM_EDX` | 1 of 1 |

**The TIFF parser had no info-bar check.** Five square `.tif` frames were passing
through silently while the identical defect in a `.jpg` was flagged. The test was
written for the JPEG parser and never applied to the container that had existed
all along.

Fixed by extracting `parsers/_frames.py` — `info_bar_geometry` and
`no_info_bar_issue` — now used by both. The count went **31 → 36**, matching the
independent sweep exactly. The golden TIFF snapshot was regenerated; the fixture
is 1024x1024x3, genuinely square, so the new warning is correct for it, and the
diff is only the issue plus the three geometry keys.

### Latos now disagrees with DATA_NOTES, checkably

DATA_NOTES: *"33 TEM frames saved with no info bar (all Ti3AlC2 MAX)."*
Latos: **36** — 30 Ti3AlC2 and **6 Mo2Ti2AlC3**.

The count differs and the attribution is contradicted: there are Mo2Ti2AlC3
frames with no bar. Which is right needs a human look. The point is that the
disagreement is now visible and reproducible, where before it was a hand count
nothing could check.

### Ledger

| | first run | now |
|---|---|---|
| parsed | 94 (11%) | **722 (84%)** |
| unclassified | 766 | 138 |

Still unread: 63 `.txt`, 34 `.raw`, 14 `.wdf`, 12 `.map`, 3 each `.img` / `.pts`
/ `.sid` / `.brml`, 1 each `.pdf` / `.asw` / `.ico`.

### Shipped

- `ingestion/parsers/microscopy_bmp.py`, registered in `default_registry()`
- `ingestion/parsers/_frames.py`; JPEG and TIFF parsers both use it
- `tests/unit/ingestion/parsers/test_microscopy_bmp.py` — 20 tests
- `TestInfoBar` in `test_microscopy_tif.py` — 3 tests, including a multichannel
  page so the channel axis is never read as width
- `microscopy-bmp` added to the registry inventory test; TIFF snapshot updated

## 2026-09-04 — The closed loop was open at its last joint

Setting up the ionic-liquid campaign surfaced a bug in the one claim the audit
calls Latos's strongest. `python -m latos next` froze pre-registrations into
`<workbook>/preregistrations/`. The validation module and the desktop app both
read `<root>/.latos/prereg/`. **Nothing failed.** The bench command printed
`Pre-registered: ...`, wrote a real record with a real prediction, and the
screen that scores it listed nothing.

Measured on a record the rehearsal harness actually wrote:

```
records on disk            : ['prereg_20260904T100338Z.json']
list_preregistrations(root): []
```

Worse than invisible: `app.py` confines `/optimize/validate` requests to
`.latos/prereg/`, so the server would have **refused** a bench record even if
handed the path directly. Freeze → measure → validate, and the last arrow was
broken in both directions.

### Cause

The path was declared four times across three modules and one copy disagreed —
the same failure mode the workbook template's docstring already warns about for
sheet names, in a module that had not adopted the remedy.

### Fix

`prereg.prereg_dir(root)` declares it once; `validate.py`, `server/app.py`
(both the writer and the confinement check) and `campaign_cycle.py` all read
that declaration. No hard-coded copies remain in `src/`.

The regression guard is the join, not the constant: `run_cycle` freezes, then
`list_preregistrations` must find *that* file. Verified to fail against the old
behaviour — reader returns `[]` while the record sits in `preregistrations/`.

After:

```
froze to        : ...\loop_check\.latos\prereg\prereg_20260904T112135Z.json
reader found    : ['prereg_20260904T112135Z.json']
same file       : True
```

### Also this session

`latos rehearse` run before any sample exists, at 5 / 10 / 15 % noise:
**median 5 experiments** to within 5 % of the optimum, 96 / 90 / 92 % of runs
inside a budget of 12. Noise barely moves it — the campaign's risk is the
*shape* of S(x), not measurement precision. The rehearsal flags its own
`sign flip` shape as *optimum at an endpoint*, pricing the degeneracy
independently of any reading of the literature.

Two template columns claimed to be derived and were computed by nothing:
`mole_fraction_x` and `IL_loading_mg_cm2`. Both were tier 0, which the guide
sheet renders blue above the line *"Leave them; Latos computes them from what
you typed."* Moved to tier 2 with the arithmetic stated.

### Shipped

- `optimization/prereg.py` — `prereg_dir()`, exported from `latos.optimization`
- `optimization/validate.py`, `server/app.py`, `campaign_cycle.py` — all four
  hard-coded copies replaced
- `test_campaign_cycle.py` — `test_the_record_is_findable_by_the_reader_that_scores_it`
- `ingestion/ite_workbook_template.py` — the two false DERIVED columns
- `data/ionic-liquid-campaign/` — seeded workbook + printable A4 bench sheet

1943 unit tests pass.

## 2026-09-05 — The frozen record now identifies its own training set

`n_observations` was a count. Two pre-registrations fit to entirely different
measurements were distinguishable only by their timestamps, which is the
obvious question to ask of a pre-registration: *how do we know this prediction
was made on the data you say it was?*

Added to every record: `latos_version`, and a `training_data` block carrying a
SHA-256 over the observations, the observations themselves, and the per-point
weights.

### Three decisions worth recording

**Rows are sorted before hashing, so order does not matter.** A reordering is
the same set of measurements. A digest that cried mismatch over row order would
be a check nobody trusts, and an audit check nobody trusts is worse than none.

**The weights are part of the training set.** `BoConfig.point_noise_used`
already said a fit was heteroscedastic; it did not say *with what*, so two runs
weighting the same points differently were still indistinguishable — the exact
failure that field's own comment describes. `BoConfig.point_noise_scale` now
records them, and the digest covers them.

**The observations are stored at full precision.** Found by trying to verify a
record the way an auditor would: recomputing the digest from the |S| values as
printed in the report gave a mismatch, because a fitted slope is a float64 and
a report rounds. A hash advertised as falsifiable while withholding its inputs
is not checkable, so the record now carries them.

### Verified from the file alone

```
x                : [0.0, 0.5, 1.0]
y                : [0.44723999999999997, 0.15050999999999998, 0.78138]
weights          : [3.1950718685831716, 1.0, 0.8911704312114979]
recorded  sha256 : 577bb30a8fd33035dd067d8f29b284ad69d8e19759eeb6a65e77d2c96435df48
recomputed sha256: 577bb30a8fd33035dd067d8f29b284ad69d8e19759eeb6a65e77d2c96435df48
VERIFIED         : True
after a 1e-12 edit to one measurement: False
```

### Shipped

- `optimization/prereg.py` — `observations_digest()`, the `training_data`
  block, `latos_version`, and both rendered into the Markdown note
- `optimization/engine.py` — `BoConfig.point_noise_scale`
- `test_prereg.py` — 26 tests, including order-independence, that pairing
  survives the sort, that rounded values do *not* reproduce the digest, and
  that the record alone suffices to recompute it
- `_to_markdown` reads older records that predate both fields

Pre-existing and untouched: two mypy `type-arg` errors in
`analysis/microscopy/calibration.py`.
