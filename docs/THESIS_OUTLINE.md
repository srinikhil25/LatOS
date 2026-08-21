# Thesis outline → stage map

Working title:
**Development of a Materials Informatics Platform Integrating Multi-Modal
Characterization, Machine Learning, and Bayesian Optimization**

This document maps the planned thesis chapters to the Latos stages that
feed them. Update as the thesis structure firms up.

> **Note (2026-08-04).** "Stage N" here refers to the per-stage write-ups in
> `docs/stages/`, which is a different numbering from the one that used to be in
> `STAGES.md`. That file has been rewritten as a single workstream-based roadmap
> (`MV*`, `AX*`, `TE1*`, …) with no stage ladder and no degree-phase split, so
> do not try to reconcile the two schemes — this map owns `docs/stages/`, and
> `STAGES.md` owns what gets built next. New chapters should cite workstream IDs.

## Chapter map

| Chapter | Working title | Fed by | Status |
|---|---|---|---|
| 1 | Introduction & motivation | — (write last) | pending |
| 2 | Related work / literature review | — (continuous) | pending |
| 3 | Multi-modal ingestion architecture | Stage 1 (1A–1F) | ✅ stages complete; doc backfilled |
| 4 | Sample identity resolution | Stage 2 (2A–2D) | ✅ stages complete; doc backfilled |
| 5 | Derived analysis framework | Stage 3 (3A–3D) | ✅ stage doc written; the thermovoltage analyzer (Stage 6) extends it |
| 6 | Cross-modal correlation & feature extraction | Stage 4 | ⚠️ partly built (`reporting/correlation.py`), no stage doc |
| 7 | Vision-language inference on micrographs | Stage 5 | pending |
| 8 | Bayesian optimization of synthesis parameters | Stage 6 | ✅ [stage doc](stages/stage6_closed_loop.md); heteroscedastic surrogate, stopping verdict, pre-registration |
| 9 | Reliability-gated closed-loop discovery | Stage 6 | ✅ built, ❌ never run on a real sample |
| 10 | Case study: ionic-liquid mixture thermopower | Stage 6 applied | ⏳ experiment not yet started |
| 11 | Conclusion & future work | — (write last) | pending |

> **Two honest notes on this table.**
>
> Chapter 9 is built but unvalidated. Every number behind Stage 6 comes from
> synthetic objectives with known optima, which demonstrates the machinery and
> demonstrates nothing about any material. The chapter cannot be written from
> the code alone.
>
> The case study moved. It was Cs₃Bi₂I₉, whose data is four samples from another
> project and not cleared for publication. The prospective ionic-liquid mixture
> campaign replaces it, and is the better case anyway: it has not started, so
> every prediction in it can be pre-registered before the sample exists, which is
> the one thing a retrospective dataset can never support.

## How to use this map

When writing a thesis chapter:

1. Open the relevant `docs/stages/stageN_*.md` files.
2. Pull the **Methods / algorithms** sections into the chapter body —
   they're already cite-shaped.
3. Pull the **Design decisions** into the architecture subsection — the
   alternatives-considered bullets are usually the most reviewer-bait
   content.
4. Pull metrics from `BENCHMARKS.json` into a results table.
5. Use diagrams in `figures/architecture.md` directly (export to PNG if
   the publisher needs raster).

## Paper vs thesis

The paper is a single-publication-length subset. Likely candidate scope:

- **Paper option A** — "Latos: A multi-modal ingestion + analysis
  platform for materials characterization."
  Chapters 3 + 4 + 5 + a one-section case study from chapter 9.
  Target: a methods-track conference / open-source software journal
  (JOSS, SoftwareX) or a domain venue like *npj Computational Materials*.

- **Paper option B** — "Bayesian-optimization-driven thermoelectric
  composition tuning for Cs₃Bi₂I₉, enabled by an open-source
  characterization platform."
  Chapter 8 (the science result) leans on Chapters 3–7 as the platform.
  Target: *Advanced Materials*, *Journal of Materials Chemistry A*, or
  *Chemistry of Materials*.

Both options pull the same stage docs as evidence; the framing is what
differs. Decide closer to the result.

## Defense slide deck

The **Slide-Worthy Achievement** blurbs at the end of each
`RESULTS_LOG.md` stage entry are pre-written defense-slide content.
Each one is one slide:

- A claim ("Latos now does X")
- A one-liner of what made it possible
- Two or three "wow numbers"
