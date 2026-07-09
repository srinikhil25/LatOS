# Latos

> **An operating system for materials characterization.**
> *Lat* (lattice) + *OS* (operating system).

Latos is a local, open-source desktop app that turns a messy folder of raw
instrument data into traceable, analysed material properties — and then points to
the next experiment worth running.

Drop a folder of XRD, XPS, UV-DRS, Hall, thermoelectric, EDS, and TEM/SEM files.
Latos fingerprints each file, identifies its sample and technique, parses it,
flags data-quality issues for you to confirm, computes the properties (band gap,
composition, zT, fitted XRD peaks, …), and — once you confirm — runs Bayesian
optimization to recommend the next composition, with a **pre-registered
prediction** and a 95% interval.

Every number traces back to its raw file. Everything runs on your machine — **no
cloud, no LLM, fully reproducible.**

**Status:** pre-alpha. The core loop — **ingest → analyse → review → optimize** —
works end-to-end on real data; not yet packaged for distribution.

## Why Latos?

Researchers spend hours on work that should be automatic — and mistakes slip
through:

| Task | By hand | With Latos |
|------|---------|------------|
| Organizing raw instrument files | ~30 min | auto-detected |
| Band gap from a UV-DRS Tauc plot | ~10 min | instant |
| Deriving zT(T) from resistivity/Seebeck + LFA runs | spreadsheet juggling | instant, with a plausibility check |
| Catching a unit slip before it reaches a paper | easy to miss | flagged at ingest |
| Choosing the next composition to synthesize | guesswork | recommended, with a 95% interval |

## What works today

**Ingest & organise**
- Drop a folder → per-file fingerprinting, technique detection, sample identification
- Parsers: XRD (Rigaku, PANalytical), XPS (CasaXPS), UV-DRS, Hall, thermoelectric
  (LFA + resistivity/Seebeck), EDS (Bruker, EMSA), TEM/SEM/STEM images
- SQLite + Parquet store; instant project reload

**Data quality — provenance first**
- Flags implausible values, unreadable rows, missing timestamps,
  background-subtracted curves, and duplicate/mislabeled samples
- A **Review & Confirm** gate: nothing is analysed until a human confirms, and
  nothing is ever auto-merged

**Analysis (per technique)**
- XRD peak fitting · UV-DRS Tauc band gap · EDS composition · thermoelectric
  **zT(T)** derived from R&S + LFA with a physical-plausibility check
- Every derived value links back to its source file

**Bayesian optimization**
- Gaussian-process surrogate + Expected Improvement over your synthesis knob
  (e.g. doping %), maximizing a chosen property (e.g. derived zT)
- Reports the recommended next experiment with a 95% predictive interval and a
  plain-language verdict — no equations on screen
- **Pre-registration:** freeze the prediction (with a kernel-robustness check) to
  disk *before* you make the sample, so the recommendation is prospective, not
  hindsight

## Architecture

A monorepo: a headless Python core + a thin desktop shell.

- **`packages/core`** — Python. Parsing, analysis, persistence, and Bayesian
  optimization, exposed over a local FastAPI sidecar bound to `127.0.0.1` only.
- **`apps/desktop`** — a Tauri 2 + React 19 desktop app that spawns the sidecar
  and drives it over HTTP.

## Tech stack

- **Core:** Python 3.11+, FastAPI + Uvicorn, NumPy / SciPy, lmfit (fitting),
  scikit-learn (Gaussian-process BO), SQLAlchemy + SQLite + Parquet (pyarrow)
- **Desktop:** Tauri 2, React 19, TypeScript, Vite, Tailwind CSS, uPlot
- **Tests:** pytest (+ Hypothesis)
- **Local by design:** no cloud services, no LLM — every result is inspectable
  and reproducible. (GPU-scale GP via GPyTorch/BoTorch is an optional `ml` extra.)

## Getting started (developers)

Latos runs as two processes: the Python **sidecar** and the **desktop app**.

**1. Core / sidecar**

```bash
git clone https://github.com/srinikhil25/LatOS.git
cd LatOS
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -e "packages/core[server]"
python -m latos.server            # serves http://127.0.0.1:8765
```

**2. Desktop app** (in a second terminal)

```bash
cd apps/desktop
npm install
npm run tauri dev                 # opens the desktop window (waits for the sidecar)
```

**Tests**

```bash
cd packages/core
pytest                            # all tests
pytest -m "not slow"              # fast subset
```

## Project structure

```
packages/core/src/latos/
├── core/           Domain models (Project, Sample, Measurement)
├── ingestion/      Parsers, technique detection, sample labeling + review flags
├── analysis/       Per-technique analysis (XRD, UV-DRS, EDS, transport/zT)
├── optimization/   Bayesian optimization + pre-registration
├── persistence/    SQLite + Parquet
└── server/         FastAPI sidecar (the desktop app's API)

apps/desktop/           Tauri 2 + React 19 desktop shell
packages/core/tests/    Test suite
```

See [`CLAUDE.md`](./CLAUDE.md) and [`AGENTS.md`](./AGENTS.md) for development notes.

## Roadmap

- Multi-variable optimization (dopant type × amount × processing, together)
- zT at a target application temperature as a selectable BO objective
- Cross-technique correlation + one-click paper figures
- Experimental validation of recommendations at the bench
- Packaged installers

## Citation

If Latos contributes to your research, please cite:

```bibtex
@software{latos2026,
  author = {Srinikhil},
  title  = {Latos: An Operating System for Multi-Modal Materials Characterization},
  year   = {2026},
  url    = {https://github.com/srinikhil25/LatOS}
}
```

## License

[MIT](./LICENSE) © 2026 Srinikhil

## Acknowledgments

- The Ikeda–Hamasaki Lab for being the test bed
- [Materials Project](https://materialsproject.org/) for reference XRD data
- The maintainers of FastAPI, Tauri, React, lmfit, scikit-learn, and SQLAlchemy
- [Materials-Informatics](https://github.com/srinikhil25/Materials-Informatics) —
  the Streamlit predecessor that informed this rewrite
