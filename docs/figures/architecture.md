# Architecture diagrams

All diagrams are Mermaid blocks — they render in GitHub, are version-
controlled, and can be exported to PNG with the Mermaid CLI when the
thesis needs raster figures (`mmdc -i architecture.md -o out.png`).

## Layered architecture

```mermaid
flowchart TB
    UI["UI Layer<br/>PySide6 + QFluentWidgets"]
    SVC["Service Layer<br/>RecentProjects, IngestionWorker, AnalysisService"]
    ING["Ingestion Layer<br/>Crawler, Parsers, Registry, Orchestrator"]
    LAB["Labeling Layer<br/>Hints, Normalize, Cluster, Decisions"]
    ANA["Analysis Layer<br/>BaseAnalyzer, AnalyzerRegistry, Tauc, ..."]
    PER["Persistence Layer<br/>SQLAlchemy, Alembic, ArrayStore, Repository"]
    DOM["Domain Layer<br/>Project, Sample, Measurement, FileRef,<br/>ValidationIssue, AnalysisResult"]

    UI --> SVC
    SVC --> ING
    SVC --> ANA
    ING --> LAB
    ING --> PER
    LAB --> PER
    ANA --> PER
    PER --> DOM
    ING --> DOM
    LAB --> DOM
    ANA --> DOM
    UI --> DOM
```

Arrows point in the direction of *dependency*. The domain layer
depends on nothing; the UI layer depends on everything beneath it.

## On-disk project layout

```mermaid
flowchart LR
    PROJECT["<project_root>/"]
    LATOS[".latos/"]
    DB["data.db<br/>(SQLite metadata)"]
    ARR["arrays/"]
    PARSED["measurement_id.parquet<br/>(parsed arrays)"]
    DERIVED["measurement_id.analyzer.short.parquet<br/>(derived arrays from Stage 3)"]
    EXP["exports/"]
    CLU["cluster_decisions.json<br/>(user merge/rename overrides)"]
    INSTRUMENT["instrument files<br/>(.txt .xrdml .csv .xlsx .tif ...)"]

    PROJECT --> LATOS
    PROJECT --> INSTRUMENT
    LATOS --> DB
    LATOS --> ARR
    LATOS --> EXP
    LATOS --> CLU
    ARR --> PARSED
    ARR --> DERIVED
```

## Ingestion pipeline (Stage 1)

```mermaid
sequenceDiagram
    participant UI as UI / CLI
    participant ORC as Orchestrator
    participant CRA as Crawler
    participant REG as ParserRegistry
    participant PAR as Parser
    participant AS as ArrayStore
    participant REPO as ProjectRepository

    UI->>ORC: ingest(folder)
    ORC->>CRA: walk(folder)
    CRA->>CRA: hash files (SHA-256)
    CRA->>REG: find_parser(path)
    REG-->>CRA: ParserMatch | None
    CRA-->>ORC: CrawlReport
    loop per file
        ORC->>PAR: parse_all(path)
        PAR-->>ORC: tuple[ParsedData]
        ORC->>AS: write(measurement_id, parsed)
        ORC->>REPO: save(project)
    end
    ORC-->>UI: IngestionResult
```

## Labeling pipeline (Stage 2)

```mermaid
flowchart LR
    P["Project<br/>(Stage 1 output)"] --> HINT["extract_hints<br/>(path / filename / metadata)"]
    HINT --> NORM["normalize<br/>(NFKC + lowercase +<br/>prefix scrub + separator strip)"]
    NORM --> CLU["cluster_samples<br/>(rapidfuzz + networkx<br/>connected components)"]
    CLU --> DEC["apply_decisions<br/>(splits → merges → renames)"]
    DEC --> CL["tuple[SampleCluster]"]
    JSON["cluster_decisions.json<br/>(user overrides)"] --> DEC
```

## Analysis pipeline (Stage 3)

```mermaid
sequenceDiagram
    participant UI as UI / CLI
    participant SVC as AnalysisService
    participant REPO as ProjectRepository
    participant AS as ArrayStore
    participant ANA as Analyzer

    UI->>SVC: run(analyzer, measurement, params)
    SVC->>SVC: check cache by (name, version, params_fp)
    alt cache hit
        SVC-->>UI: AnalysisRunOutcome(from_cache=True)
    else cache miss
        SVC->>AS: load(measurement_id)
        AS-->>SVC: arrays dict
        SVC->>ANA: analyze(AnalyzerInputs)
        ANA-->>SVC: AnalyzerOutput
        SVC->>AS: write derived arrays<br/>(measurement_id.analyzer.short.parquet)
        SVC->>REPO: save project with AnalysisResult appended
        SVC-->>UI: AnalysisRunOutcome(from_cache=False)
    end
```

## Cache-key strategy

```mermaid
flowchart LR
    subgraph "Parse cache (Stage 1)"
        SHA["file SHA-256"]
        PV["parser_version"]
        PC["(sha256, parser_version)"]
        SHA --> PC
        PV --> PC
    end
    subgraph "Analysis cache (Stage 3)"
        MID["measurement_id"]
        AN["analyzer_name"]
        AV["analyzer_version"]
        FP["params fingerprint<br/>(canonical JSON SHA-256)"]
        AC["(mid, name, version, fp)"]
        MID --> AC
        AN --> AC
        AV --> AC
        FP --> AC
    end
    PC -.same shape.-> AC
```

Both caches share the same invalidation philosophy: a version bump on
the producer (parser or analyzer) invalidates every entry it has
produced. The user never thinks about cache state.
