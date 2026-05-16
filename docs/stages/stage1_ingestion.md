# Stage 1 — Multi-modal ingestion pipeline

**Status:** ✅ complete
**Sub-stages covered:** 1A (domain models), 1B (persistence), 1C (parsers + hashing), 1D (crawler + orchestrator), 1E (PySide6 desktop shell), 1F (multi-sheet workbooks + folder-aware microscopy)
**Date range:** 2026-04-27 – 2026-05-11

## 1. Goal

Turn a folder of raw, heterogeneous lab files (XRD, XPS, UV-DRS, Hall,
thermoelectric, EDS, microscopy) into a typed, validated, queryable
`Project` — automatically, idempotently, and without requiring the
researcher to write Python.

## 2. Motivation

Materials research generates files in instrument-specific formats that
share *nothing*: a Rigaku XRD scan is a semicolon-headed text file, a
PANalytical XRD scan is XML, a Bruker EDS spectrum is XML-despite-its-
extension, a CasaXPS export is CSV with a variable-length header, a
Hall measurement is an old-style binary `.xls`. A researcher with two
years of accumulated data on their laptop has no way to ask
*"give me every XRD scan of every CsBiI sample I measured in 2024"*
without either a manual spreadsheet or a custom one-off script per
question.

Latos's first task is to make that question answerable. Everything
downstream — clustering, analysis, ML, optimization — depends on a
uniform internal representation. Stage 1 builds that representation
and the machinery to populate it.

## 3. Design decisions

- **Decision:** All domain objects are frozen dataclasses.
  - Alternatives considered: ORM rows used directly, Pydantic models.
  - Why this won: Immutability is enforced by the runtime, validation
    happens once at construction, hashability is free (useful for set
    membership in tests), and the layer is completely decoupled from
    the persistence library (we could swap SQLAlchemy for DuckDB
    without touching domain code). [\[glossary: frozen dataclass\]](../glossary.md#software--tooling)

- **Decision:** One Parquet file per measurement; SQLite for metadata.
  - Alternatives considered: All-in-SQLite (BLOBs), all-in-HDF5,
    per-project DuckDB.
  - Why this won: Arrays are too big to live efficiently in SQL rows;
    SQL is too good at relational queries to give up. Parquet is
    column-major, compressed, atomic-writable, and openable by every
    common data tool (pandas, DuckDB, Power Query). [\[apache_parquet\]](../references.md#apache_parquet)

- **Decision:** `ParsedData` is the universal contract.
  - Alternatives considered: One dataclass per technique.
  - Why this won: 90% of the post-parse infrastructure (storage,
    validation, caching) doesn't care which technique a result is —
    only that it has arrays, metadata, instrument, and a parser id.
    A single shape gives one code path for nine parsers and N future
    ones. Per-technique differences live in `arrays` and `metadata`.

- **Decision:** Confidence-pick dispatch via `BaseParser.can_parse(path) -> float`.
  - Alternatives considered: Extension-only routing, MIME-only routing,
    user-selects-format-per-file.
  - Why this won: Multiple instruments share extensions (the `.csv`
    from CasaXPS vs. a ledger spreadsheet, the two distinct `.xlsx`
    formats for UV-DRS vs. thermoelectric). A cheap-but-content-aware
    confidence score lets each parser read just enough header to
    decide. Threshold 0.5 separates "I'm pretty sure" from "wild
    guess"; tie-breaking by registration order.

- **Decision:** Parsers never raise; they return `ParsedData` with
  `issues`.
  - Alternatives considered: Exceptions for malformed input.
  - Why this won: A single bad file should not abort ingestion of 160
    good ones. The orchestrator-level outcome (`PARSED`,
    `PARSED_WITH_ISSUES`, `PARSE_FAILED`, ...) gives the UI enough
    information to triage without try/except scaffolding everywhere.

- **Decision:** Idempotent re-ingestion via `(sha256, parser_version)` cache key.
  - Alternatives considered: Re-parse always, file mtime as key.
  - Why this won: Re-opening a 161-file project takes 0.2 seconds
    instead of 2.6 seconds (~13× speedup). A `parser_version` bump
    automatically invalidates only the affected files. Content-hash
    based means moving / renaming a file doesn't trigger re-parse.

- **Decision:** Atomic writes via `.tmp` + `os.replace()` for every
  on-disk artifact.
  - Alternatives considered: Direct writes.
  - Why this won: Ctrl+C during a long ingest must never poison the
    parse cache with half-written Parquet. `os.replace` is atomic on
    both POSIX (rename) and Windows (MoveFileEx with REPLACE_EXISTING).

- **Decision:** UI runs ingestion on a `QThread` via `moveToThread`.
  - Alternatives considered: Synchronous ingestion in the GUI thread,
    `multiprocessing`.
  - Why this won: Qt docs pattern #2 (the worker-object pattern) keeps
    cancellation, signal-emission, and lifetime explicit. The GUI
    never freezes; cancellation via `threading.Event` polled from the
    orchestrator's progress callback.

- **Decision (Stage 1F):** Multi-sheet workbooks → one `Measurement` per
  sheet via `parse_all()`.
  - Alternatives considered: Keep "first sheet only" with a warning;
    one Measurement holding all sheets.
  - Why this won: Researchers put one sample per sheet in their
    UV-DRS / thermoelectric workbooks. The old "first-sheet-only"
    behaviour silently dropped 3 of 4 samples from `zT calculation.xlsx`.
    Multi-Measurement-per-file required relaxing the `UNIQUE(sha256)`
    constraint on the files table — same file can now belong to
    multiple measurements.

## 4. Methods / algorithms

- **SHA-256 content hashing** for file fingerprinting. Streamed in 1 MB
  chunks so multi-GB TIFF files don't load entirely into memory.
  Cached on `(path, mtime, size)` so re-walking a stable folder is
  effectively free.

- **Confidence-pick parser dispatch.** Each parser exposes
  `can_parse(path) → float ∈ [0, 1]`. Conventions:
  - 0.0: definitely not this format
  - 0.5: extension matches, structure unverified
  - 0.8: header magic / keywords confirmed
  - 1.0: unambiguous (e.g. unique XML namespace)

  Dispatch picks the highest scorer above 0.5; ties broken by
  registration order.

- **Sample-name heuristic (Stage 1 baseline).** A file's sample = the
  name of its immediate parent folder, *unless* that folder is generic
  ("XRD", "data", "raw", ...); then walk up to 3 levels for a non-
  generic ancestor, then fall back to the file's own stem with a
  `Severity.WARNING` issue. Deliberately dumb — Stage 2 replaces this
  with similarity-graph clustering.

- **Atomic file writes.** Pattern: write to `<target>.tmp`, then
  `os.replace(tmp, target)`. POSIX `rename` and Windows `MoveFileEx
  with REPLACE_EXISTING` are both atomic, so a crash mid-write leaves
  either the previous valid file or no file — never a corrupt one.

- **`(content_hash, producer_version)` cache key.** A Latos-wide
  pattern. Stage 1 uses `(sha256, parser_version)`; Stage 3 reuses the
  same shape with `(measurement_id, analyzer_name, analyzer_version,
  params_fingerprint)`. The producer bumps its own version when its
  output shape or values change; every prior cached output is
  automatically invalidated.

## 5. Implementation summary

| File | What it owns |
|---|---|
| `src/latos/core/enums.py` | `Technique`, `FileRole`, `Severity` enums |
| `src/latos/core/exceptions.py` | `LatosError` hierarchy (14 types) |
| `src/latos/core/models.py` | `Project`, `Sample`, `Measurement`, `FileRef`, `ValidationIssue` |
| `src/latos/persistence/schema.py` | SQLAlchemy ORM tables + `UtcDateTime` TypeDecorator |
| `src/latos/persistence/db.py` | Engine factory, SQLite PRAGMAs, path helpers |
| `src/latos/persistence/mappers.py` | Bidirectional domain ↔ ORM conversion |
| `src/latos/persistence/repository.py` | `ProjectRepository` (save / load / list / delete) |
| `migrations/versions/0001_initial_schema.py` | Initial Alembic migration |
| `migrations/versions/0002_drop_files_sha256_unique.py` | Stage 1F: enable multi-Measurement-per-file |
| `src/latos/ingestion/hashing.py` | SHA-256 streamer + `HashCache` |
| `src/latos/ingestion/parsed_data.py` | The universal parser-output contract |
| `src/latos/ingestion/base_parser.py` | `BaseParser` ABC + import-time validation |
| `src/latos/ingestion/array_store.py` | Atomic Parquet I/O |
| `src/latos/ingestion/parsers/*.py` | 9 instrument-specific parsers |
| `src/latos/ingestion/registry.py` | `ParserRegistry` + `default_registry()` |
| `src/latos/ingestion/crawler.py` | Folder walk + per-file classification |
| `src/latos/ingestion/orchestrator.py` | Integration layer wiring everything together |
| `src/latos/ui/app.py` + `main_window.py` | PySide6 `FluentWindow` shell |
| `src/latos/ui/pages/{welcome,project_picker,overview,sample_review}.py` | Sidebar pages |
| `src/latos/ui/services/{recent_projects,ingestion_worker}.py` | Qt-free background services |
| `src/latos/ui/dialogs/ingestion_progress.py` | Modal threaded-ingestion dialog |

Key invariants enforced:

- Every ID is a 32-char lowercase hex UUID.
- Every timestamp is timezone-aware (naive ones rejected at the type boundary).
- Every numeric array is 1-D; arrays within a Measurement are co-indexed.
- `parsed_at` and `parser_version` are recorded on every Measurement.
- The orchestrator's `IngestionResult` enumerates every file processed
  with one of six `Outcome` values.

## 6. Validation

- **Tests:** 700 passing at end of Stage 1E; 746 at end of Stage 2 (which
  Stage 1F preceded).
- **Coverage:** 89% on `ingestion/`, 95–100% on infra modules (hashing,
  ParsedData, BaseParser, ArrayStore, registry), 95% on `core/`.
- **Real-data behaviour (Dhivya, 161 files, ~590 MB):**
  - 76 files parsed correctly across 7 techniques
  - 84 correctly classified as non-data (PDFs, JPEGs, .docx, .spe)
  - 0 parser crashes
  - 0 hash failures
  - 12 samples inferred (Stage 1 heuristic; Stage 2 collapses to 11)
  - **First ingest: 2.6 s; cached re-ingest: 0.2 s (~13× speedup)**
- **Stage 1F regression (multi-sheet):** `zT calculation.xlsx` and the
  UV-DRS workbook now contribute one Measurement per sheet. Total
  parsed files rose from 65 to 76; total samples rose from 12 to 15.
- **Quality gates:** ruff + mypy strict clean continuously.

See `BENCHMARKS.json` entries 1A through 1E for the per-substage table.

## 7. Limitations

- **2-D image content deferred.** Microscopy TIFFs are stored as
  metadata-only (filename, size, technique). Image-pixel ingestion is
  Stage 5 work (vision-language model on micrographs).
- **Stage 1 sample heuristic is intentionally dumb.** It will
  over-split (`CS Pure` vs `CS (Pure)` become two samples). Stage 2
  fixes this. The dumb baseline is *deliberate* — it makes the Stage 2
  improvement visible and benchmarkable.
- **No file watcher.** Adding a new instrument file requires
  re-running ingestion. A live-watch background service is on the
  Stage 8 (polish & performance) roadmap.
- **No instrument auto-discovery.** New instruments require a new
  parser class. This is fine — instruments evolve slowly — but it
  means a researcher with an unsupported format must contribute or
  request a parser.

## 8. Thesis mapping

| Thesis section | What this stage feeds |
|---|---|
| 3.1 Problem statement: heterogeneous data | Motivation, the real-data outcome breakdown |
| 3.2 Domain model | `Project / Sample / Measurement / FileRef`, the immutability rationale |
| 3.3 Persistence architecture | SQLite + Parquet split, repository pattern, atomic writes, schema versioning |
| 3.4 Parser framework | `ParsedData` contract, `BaseParser` ABC, confidence-pick dispatch, the nine concrete parsers |
| 3.5 Ingestion orchestration | Crawler → parser → array store → repository pipeline; idempotent caching; cancellation |
| 3.6 Desktop application | PySide6 FluentWindow shell, off-thread ingestion, recent-projects service |
| 3.7 Results & evaluation | Dhivya 161-file benchmark, 2.6 s ingest / 0.2 s re-open, 0 crashes |

## See also

- [`RESULTS_LOG.md`](../../RESULTS_LOG.md) — chronological detail + bug log for 1A through 1F
- [`BENCHMARKS.json`](../../BENCHMARKS.json) — entries 1A through 1E plus the 1F notes appended to Stage 2's entry
- [`figures/architecture.md`](../figures/architecture.md) — layered diagram + ingestion sequence
- [`references.md`](../references.md) — `apache_parquet`, `sqlite`, `alembic`, `cullity2001` (XRD), `briggs2003` (XPS), `goldsmid2010` (thermoelectric)
