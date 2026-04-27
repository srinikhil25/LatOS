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

<!-- Future entries go below this line -->
