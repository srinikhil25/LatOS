# Thesis outline → stage map

Working title:
**Development of a Materials Informatics Platform Integrating Multi-Modal
Characterization, Machine Learning, and Bayesian Optimization**

This document maps the planned thesis chapters to the Latos stages that
feed them. Update as the thesis structure firms up.

## Chapter map

| Chapter | Working title | Fed by | Status |
|---|---|---|---|
| 1 | Introduction & motivation | — (write last) | pending |
| 2 | Related work / literature review | — (continuous) | pending |
| 3 | Multi-modal ingestion architecture | Stage 1 (1A–1F) | ✅ stages complete; doc backfilled |
| 4 | Sample identity resolution | Stage 2 (2A–2D) | ✅ stages complete; doc backfilled |
| 5 | Derived analysis framework | Stage 3 (3A–3D) | 🚧 3A+3B done, doc in progress |
| 6 | Cross-modal correlation & feature extraction | Stage 4 | pending |
| 7 | Vision-language inference on micrographs | Stage 5 | pending |
| 8 | Bayesian optimization of synthesis parameters | Stage 6 | pending |
| 9 | Case study: Cs₃Bi₂I₉ thermoelectric optimization | Stages 1–8 applied | pending |
| 10 | Conclusion & future work | — (write last) | pending |

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
