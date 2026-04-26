# Latos — AI Assistant Working Notes

This file gives any AI assistant (Claude, Codex, Cursor, etc.) the context needed to work effectively on Latos.

## What is Latos?

Latos is a desktop application for **multi-modal materials characterization**. Researchers drop a folder of raw instrument data (XRD, XPS, UV-DRS, Hall, TEM, EDS, etc.), and the platform auto-detects techniques, identifies samples, parses files, fits peaks, and ultimately suggests next experiments via Bayesian optimization.

**Name origin:** *Lat* (lattice) + *OS* (operating system) — an OS for materials science.

**Goal:** Replace Origin for daily fitting work, plus enable autonomous closed-loop discovery.

**License:** MIT
**Predecessor:** [Materials-Informatics](https://github.com/srinikhil25/Materials-Informatics) (Streamlit prototype, archived)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| UI | PySide6 + QFluentWidgets (Windows 11 Fluent Design) |
| Plotting | pyqtgraph (interactive) + matplotlib (publication exports) |
| Database | SQLite + Parquet (arrays) |
| ORM | SQLAlchemy 2.0 |
| Migrations | Alembic |
| Fitting | lmfit |
| Optimization | BoTorch + GPyTorch (Stage 7) |
| ML/AI (local) | Qwen3-VL via Ollama (Stage 5) |
| Testing | pytest + pytest-qt + hypothesis + pytest-snapshot |
| Linting | ruff |
| Type checking | mypy (strict mode) |
| Distribution | PyInstaller |
| CI | GitHub Actions |

## Project Layout

```
src/latos/
├── core/           # Domain models (Project, Sample, Measurement) — no UI/DB deps
├── persistence/    # SQLite + Parquet, repository pattern
├── ingestion/      # Crawler, parsers, sample labeling, validation
├── analysis/       # Pure analysis functions per technique
├── fitting/        # Universal fit engine (lmfit-based)
├── optimization/   # Bayesian Optimization (Stage 7)
├── visualization/  # Plot builders + style templates
├── reporting/      # Markdown/LaTeX export
└── ui/             # PySide6 application (only Qt-aware code lives here)

tests/
├── unit/           # Fast, isolated
├── integration/    # Full pipelines
├── ui/             # pytest-qt
├── snapshot/       # Plot/output regression
└── fixtures/       # Test data
```

**Critical rule:** No code outside `src/latos/ui/` may import PySide6. The analysis core must be importable in a Jupyter notebook with no UI.

## Development Workflow

### Setup
```bash
cd D:/Latos
python -m venv venv
venv/Scripts/activate
pip install -e ".[all]"
pre-commit install
```

### Running tests
```bash
pytest                              # all tests with coverage
pytest -m "not slow"                # fast subset
pytest tests/unit                   # unit only
pytest --cov-report=html           # detailed coverage
```

### Running the app
```bash
latos-app                           # GUI entry point
python -m latos.ui.app              # alternative
```

### Code quality (run before commits)
```bash
ruff check . --fix                  # lint + autofix
ruff format .                       # format
mypy src/                           # type check
pre-commit run --all-files          # everything
```

## Architecture Principles

1. **UI-agnostic core.** The `analysis`, `fitting`, `optimization`, and `core` packages must work without PySide6 imported. UI is a thin layer.
2. **Repository pattern for persistence.** SQL never leaves `persistence/`. UI code calls `ProjectRepository.load(id)`, not raw queries.
3. **Pure functions for analysis.** Every analysis takes inputs and returns a result. No side effects, no hidden state.
4. **Versioned parsers.** Every parser declares a version. Re-parsing happens automatically when the version changes.
5. **File hashing for change detection.** Parse cache keyed by `(file_hash, parser_version)`. Same file + same parser = no re-parse.
6. **Validation at every stage.** Parsers validate, analyses validate, cross-technique checks flag inconsistencies.
7. **Tests from day 1.** Coverage gate starts at 70%, increases over time. No PR merges without tests for new code.
8. **Conservative migrations.** Never break existing project DBs. Use Alembic migrations for schema changes.

## Coding Conventions

- **Line length:** 100 chars (ruff enforces)
- **Type hints:** Required everywhere (mypy strict)
- **Docstrings:** Google style, on every public function/class
- **Imports:** Grouped (stdlib / third-party / first-party / relative) — ruff handles
- **String quotes:** Double quotes
- **Path handling:** `pathlib.Path` always, never raw strings
- **Logging:** Use `loguru`, not `print()` or stdlib `logging` directly
- **Config:** `pydantic-settings`, never bare `os.environ.get()`
- **Exceptions:** Custom exception hierarchy in `latos/core/exceptions.py`

## What Goes Where (Quick Reference)

| Question | Answer |
|----------|--------|
| Adding a new file format parser? | `src/latos/ingestion/parsers/` + register in dispatcher |
| Adding a new technique analysis? | `src/latos/analysis/<technique>/` |
| Adding a new fit peak shape? | `src/latos/fitting/peak_shapes.py` |
| Adding a new UI page? | `src/latos/ui/pages/` + register in app navigation |
| Adding a domain model? | `src/latos/core/models.py` |
| SQL schema change? | Alembic migration in `migrations/versions/` |

## What NOT To Do

- **Don't put SQL queries in UI code.** Always go through `persistence.repository`.
- **Don't import PySide6 outside `ui/`.** Breaks the headless capability.
- **Don't skip tests "for now."** Test as you write, not after.
- **Don't use `print()` for logging.** Use `loguru`.
- **Don't commit large data files.** `.gitignore` enforces 5 MB limit.
- **Don't commit `STAGES.md`.** It's internal planning — gitignored.
- **Don't add a dependency without updating `pyproject.toml`.**
- **Don't write functions over ~50 lines.** Decompose.
- **Don't write parsers that crash on malformed input.** They flag `ValidationIssue` and return what they could parse.

## Stages (High-Level)

See `STAGES.md` (gitignored, internal planning) for full details. Quick summary:

| Stage | Focus | AI/ML used? |
|-------|-------|-------------|
| 1 | Persistence + desktop app shell | None |
| 2 | Smart sample labeling (mechanical) | None — fuzzy strings only |
| 3 | Validation engine | None — rules |
| 4 | Universal fit engine (replaces Origin) | Light: CWT peak detection |
| 5 | Smart labeling AI layer | **Qwen3-VL via Ollama** (vision + LLM, all local) |
| 6 | Cross-correlation + reporting | Optional LLM for auto-summary |
| 7 | Bayesian Optimization | **GPyTorch + BoTorch** (closed-loop discovery) |
| 8 | Polish + paper launch | Optional LLM for docstrings |

**All ML/AI is open-source and runs locally on the user's machine. No API costs, no data leaves the user's computer.**

## When Stuck

- Read `docs/algorithms/` for technique-specific math
- Read `tests/` for usage examples (tests are documentation)
- Check `RESULTS_LOG.md` for known performance benchmarks
- The Streamlit predecessor at `D:/Materials-Informatics/` has working reference implementations of parsers and analyses — copy & adapt, don't bulk-port
