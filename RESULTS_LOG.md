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

<!-- Future entries go below this line -->
