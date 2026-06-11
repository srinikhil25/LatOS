# latos-core

The headless science core of **Latos** — multi-modal materials
characterization with closed-loop Bayesian optimization.

This package contains everything except a UI:

- **Ingestion** — file crawler + 9 instrument parsers across 7
  characterization techniques (XRD, XPS, UV-DRS, Hall, thermoelectric,
  EDS, microscopy), with content-hash parse caching.
- **Labeling** — sample-identity resolution (normalization, fuzzy
  clustering, persistent user decisions).
- **Persistence** — SQLite metadata + Parquet arrays per project,
  Alembic migrations.
- **Analysis** — analyzer framework with UV-DRS Tauc band-gap and XRD
  peak-fit (SNIP baseline + pseudo-Voigt) analyzers; results cached and
  persisted with full provenance.
- **Server** — a localhost-only FastAPI sidecar exposing the core to
  the desktop UI (`apps/desktop`).

The desktop application lives in `apps/desktop` (Tauri + React) and
talks to this package over `127.0.0.1`. No data ever leaves the
machine.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not ui and not gpu and not ollama"
ruff check . && ruff format --check . && mypy src/
```

See the repository root for project-wide documentation.
