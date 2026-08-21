# Latos documentation

Thesis- and paper-ready notes that complement the day-to-day artifacts
in `RESULTS_LOG.md` and `BENCHMARKS.json`.

## Where to find what

| What you want | Where it lives |
|---|---|
| Chronological "what happened in each stage" + bug log | [`RESULTS_LOG.md`](../RESULTS_LOG.md) |
| Numerical metrics per stage (tests, coverage, runtimes) | [`BENCHMARKS.json`](../BENCHMARKS.json) |
| Per-stage thesis-ready writeup (methods, decisions, citations) | [`stages/`](stages/) |
| Methods + citations index | [`references.md`](references.md) |
| Materials-science + software terminology | [`glossary.md`](glossary.md) |
| Architecture diagrams | [`figures/architecture.md`](figures/architecture.md) |
| Map of stages → thesis chapters | [`THESIS_OUTLINE.md`](THESIS_OUTLINE.md) |
| Plan for the ionic-liquid campaign work | `DEVELOPMENT_PLAN.md` (local only) |

## What is documented, and what is not

The stage docs cover Stages 1, 2, 3 and 6. **Stages 4 and 5 have partial
implementations and no stage doc**, and `RESULTS_LOG.md` / `BENCHMARKS.json` skip
from Stage 2 straight to Stage 6. The gap is recorded in those files rather than
quietly closed: backfilling it now would be reconstruction, not a record.

Note also that most of `docs/` is gitignored. Only the files listed above are
tracked, along with everything under `stages/` and `figures/`. Working notes,
research write-ups and lab templates live here locally but are not part of the
repository.

## How to add a new stage doc

After a stage closes (after the commit lands, before moving to the
next stage), copy [`stages/_template.md`](stages/_template.md) to
`stages/stageN_<name>.md` and fill it in while context is fresh. The
template has eight sections; expected length is 500–1000 words.

The doc is *not* a duplicate of `RESULTS_LOG.md` — that log is
chronological / bug-focused, this doc is method-focused and
thesis-shaped.

## Workflow with these docs

```mermaid
flowchart LR
    A[Stage work] --> B[Commit lands]
    B --> C[Fill RESULTS_LOG.md entry]
    B --> D[Append BENCHMARKS.json entry]
    B --> E[Write docs/stages/stageN.md]
    E --> F[Add new citations to references.md]
    E --> G[Add new terms to glossary.md]
    F --> H[Future thesis chapter]
    G --> H
    E --> H
```
