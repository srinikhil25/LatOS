# Latos

> **An operating system for materials characterization.**
> *Lat* (lattice) + *OS* (operating system).

Latos is a local, open-source desktop app that turns a messy folder of raw
instrument data into traceable, analysed material properties — then recommends
the next experiment worth running, commits that prediction *before* you run it,
and scores the real result against it when you do.

Drop a folder of XRD, XPS, UV-DRS, Hall, thermoelectric, EDS, and TEM/SEM files.
Latos fingerprints each file, identifies its sample and technique, parses it,
flags data-quality issues for you to confirm, computes the properties (band gap,
composition, carrier type, fitted XRD peaks, zT, …), and — once you confirm —
runs Bayesian optimization to recommend the next composition, with a 95% interval
and an honest **reliability label**. Freeze that recommendation and it becomes a
timestamped **pre-registration**; synthesize the sample and Latos scores the
measured result against the frozen prediction, **closing the loop**.

Every number traces back to its raw file. Everything runs on your machine —
**no cloud, no LLM, fully reproducible.**

**Status:** pre-alpha. The full loop — **ingest → analyse → review → optimize →
pre-register → validate** — works end-to-end on real data. Not yet packaged for
distribution (runs as two dev processes today).

## Why Latos?

Researchers spend hours on work that should be automatic — and mistakes slip
through:

| Task | By hand | With Latos |
|------|---------|------------|
| Organizing raw instrument files | ~30 min | auto-detected |
| Band gap from a UV-DRS Tauc plot | ~10 min | instant |
| Deriving zT(T) from resistivity/Seebeck + LFA runs | spreadsheet juggling | instant, with a plausibility check |
| Catching a unit slip before it reaches a paper | easy to miss | flagged at ingest |
| Noticing a Hall measurement is noise-floor unreliable | rarely checked | flagged automatically |
| Choosing the next composition to synthesize | guesswork | recommended, with a 95% interval and a reliability label |
| Proving a prediction was made *before* the experiment | trust me | timestamped pre-registration on disk |

## What works today

**Ingest & organise**
- Drop a folder → per-file fingerprinting, technique detection, sample identification
- Parsers: XRD (Rigaku, PANalytical), XPS (CasaXPS), UV-DRS, Hall, thermoelectric
  (LFA + resistivity/Seebeck + zT), EDS (Bruker, EMSA), TEM/SEM/STEM images
- SQLite + Parquet store; instant project reload

**Data quality — provenance first**
- Flags implausible values, unreadable rows, missing timestamps,
  background-subtracted curves, and duplicate/mislabeled samples
- **Hall reliability checks:** flags when the two van der Pauw
  cross-configurations disagree (carrier type / n / µ sitting at the noise
  floor), and cross-checks the Hall carrier type against the independent Seebeck
  sign — disagreement is surfaced with the trustworthy determination named, not
  hidden
- A **Review & Confirm** gate: nothing is analysed until a human confirms, and
  nothing is ever auto-merged

**Analysis (per technique)**
- **XRD** — peak fitting + Bragg d-spacings + Scherrer crystallite size
- **UV-DRS** — Tauc band gap
- **EDS** — elemental composition (semi-quantitative)
- **Hall** — carrier type, concentration, mobility, and a σ = q·n·µ consistency
  check
- **XPS** — peak binding energies + the C 1s charge-reference offset
- **Transport** — Seebeck sign → carrier type, power-factor curve, LFA κ range
- **zT(T)** — derived from resistivity/Seebeck + LFA with a physical-plausibility
  check
- Every derived value links back to its source file

**Bayesian optimization**
- Gaussian-process surrogate + Expected Improvement over **any** synthesis knob
  (doping %, etching time, annealing temperature, …) — not locked to doping
- Objectives: **maximize, minimize, or reach a target value.** Targets include
  derived zT (optionally at a chosen operating temperature), power factor,
  thermal conductivity, band gap, and Hall metrics
- Optimize over a **measured axis** (e.g. Hall carrier concentration) when
  samples share no synthesis knob — the classic Ioffe analysis
- **Synthesis log:** drop a `synthesis.csv` (one row per sample, one column per
  variable) next to your raw files and the per-sample values fill in
  automatically at ingest
- Reports the recommended next experiment with a 95% predictive interval, a
  plain-language verdict, and a **reliability label** — *exploratory /
  indicative / calibrated* — computed from the data itself (observation-count
  tier + leave-one-out interval coverage)
- **Pre-registration:** freeze the prediction (with a kernel-robustness check and
  the reliability self-assessment) to disk *before* you make the sample, so the
  recommendation is prospective, not hindsight
- **Outcome validation:** enter the synthesized sample's measured value and Latos
  scores it against the frozen record — calibration (inside the interval?) and
  improvement (beat the prior best?) — writing the verdict beside the prediction
  so the closed loop stays auditable

## Architecture

A monorepo: a headless Python core + a thin desktop shell.

- **`packages/core`** — Python. Parsing, analysis, persistence, and Bayesian
  optimization, exposed over a local FastAPI sidecar bound to `127.0.0.1` only.
- **`apps/desktop`** — a Tauri 2 + React 19 desktop app that spawns the sidecar
  and drives it over HTTP.

The desktop app never computes science — it renders what the core returns. Any
result you see in the UI is reproducible by calling the same endpoint.

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

> Tip: if a change doesn't seem to take effect, make sure an old sidecar isn't
> still holding port `8765` — stop it and restart `python -m latos.server`.

**Tests**

```bash
cd packages/core
pytest                            # all tests
pytest -m "not slow"              # fast subset
ruff check . && mypy src/latos    # lint + types
```

## The workflow, end to end

1. **Open a folder.** Latos ingests, detects techniques, and identifies samples.
   Drop a `synthesis.csv` alongside the raw files to auto-fill synthesis values.
2. **Review & confirm.** Data-quality issues (including Hall reliability flags)
   are surfaced; you confirm sample identity. Analysis stays locked until you do.
3. **Analyse.** Each measurement gets its per-technique analysis; zT(T) is derived
   from the transport runs.
4. **Optimize.** Pick the variable and the objective; Latos recommends the next
   experiment with a 95% interval and a reliability label.
5. **Pre-register.** Freeze the prediction to a timestamped record before you make
   the sample.
6. **Validate.** Synthesize the sample, enter its measured value, and Latos scores
   it against the frozen prediction — closing the loop.

## Project structure

```
packages/core/src/latos/
├── core/           Domain models (Project, Sample, Measurement)
├── ingestion/      Parsers, technique detection, sample labeling + review flags
├── analysis/       Per-technique analysis (XRD, UV-DRS, EDS, XPS, Hall, transport)
├── optimization/   Bayesian optimization, reliability, pre-registration, validation
├── persistence/    SQLite + Parquet
└── server/         FastAPI sidecar (the desktop app's API)

apps/desktop/           Tauri 2 + React 19 desktop shell
packages/core/tests/    Test suite
```

See [`CLAUDE.md`](./CLAUDE.md) and [`AGENTS.md`](./AGENTS.md) for development notes.

## Roadmap

- Tabular dataset import — evaluate the optimizer on public datasets
  (Starrydata, sysTEm), not just in-house raw files
- Multi-variable / categorical optimization (dopant type × amount × processing,
  together)
- Cross-technique correlation + one-click paper figures
- Particle-size analysis from TEM/SEM images
- Packaged installers (self-spawning sidecar)

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
