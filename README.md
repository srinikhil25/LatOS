# Latos

> **An operating system for materials characterization.**
> *Lat* (lattice) + *OS* (operating system).

Latos is an open-source desktop application that automates multi-modal materials characterization. Drop a folder of raw instrument data — XRD, XPS, UV-DRS, Hall, TEM, EDS — and Latos detects techniques, identifies samples, parses files, fits peaks with publication-quality reports, and (eventually) suggests your next experiment via Bayesian Optimization.

**Status:** Stage 1 — Foundation (in progress)

## Why Latos?

Researchers spend hours on tasks that should be automatic:

| Task | Manual time | With Latos |
|------|-------------|------------|
| Organizing raw instrument files | 30 min | 0 (auto-detected) |
| Fitting an XPS spectrum in Origin | ~85 min | ~2 min |
| Bandgap from UV-DRS Tauc plot | 10 min | instant |
| Cross-checking XRD vs TEM particle size | 20 min | one click |
| Suggesting next composition to synthesize | guesswork | predicted ZT ± 95% CI |

Latos is built to **replace Origin** for daily fitting work and to **enable closed-loop autonomous discovery** for research labs.

## Features (by Stage)

- **Stage 1** — Desktop app, SQLite persistence, instant project reload
- **Stage 2** — Smart sample identification across inconsistent file naming
- **Stage 3** — Auto-validation: catches researcher errors before they reach papers
- **Stage 4** — Universal fit engine: XRD, XPS, Raman, EDS with publication-ready reports
- **Stage 5** — Local Vision-AI: reads scale bars and metadata directly from microscopy images
- **Stage 6** — Cross-technique correlation + one-click paper figures
- **Stage 7** — Bayesian Optimization: closed-loop experimental design
- **Stage 8** — Production polish, installer, paper submission

## Tech Stack

- Python 3.11+
- **UI:** PySide6 + QFluentWidgets
- **Plotting:** pyqtgraph + matplotlib
- **Database:** SQLite + Parquet
- **Fitting:** lmfit
- **ML/AI** *(all open-source, all local, all free):*
  - Qwen3-VL via Ollama (vision)
  - GPyTorch + BoTorch (Bayesian Optimization)
- **Tests:** pytest + pytest-qt + hypothesis

## Installation

> Note: Latos is in early development. Pre-built binaries will be available starting Stage 8.

### From source (developers)

```bash
git clone https://github.com/srinikhil25/LatOS.git
cd latos
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -e ".[all]"
pre-commit install
```

### Run the app

```bash
latos-app
```

### Run tests

```bash
pytest                              # all tests
pytest -m "not slow"                # fast subset
pytest --cov-report=html            # coverage HTML report
```

## Project Structure

```
src/latos/
├── core/           Domain models (Project, Sample, Measurement)
├── persistence/    SQLite + Parquet
├── ingestion/      File parsers + sample labeling
├── analysis/       Pure analysis functions per technique
├── fitting/        Universal fit engine (lmfit-based)
├── optimization/   Bayesian Optimization
├── visualization/  Plot builders + style templates
├── reporting/      Markdown/LaTeX export
└── ui/             PySide6 app (only Qt code lives here)

tests/              Comprehensive test suite
docs/               Algorithm documentation + monthly progress reports
```

See [`CLAUDE.md`](./CLAUDE.md) and [`AGENTS.md`](./AGENTS.md) for detailed development notes.

## Roadmap

This project is being built in **8 stages**, each delivering a working improvement. The current platform stays usable throughout. See [Releases](https://github.com/srinikhil25/LatOS/releases) for milestone tags.

## Citation

If Latos contributes to your research, please cite:

```bibtex
@software{latos2026,
  author = {Srinikhil},
  title = {Latos: An Operating System for Multi-Modal Materials Characterization},
  year = {2026},
  url = {https://github.com/srinikhil25/LatOS}
}
```

## License

[MIT](./LICENSE) © 2026 Srinikhil

## Acknowledgments

- The Ikeda-Hamasaki Lab for being the test bed
- [Materials Project](https://materialsproject.org/) for reference XRD data
- The maintainers of PySide6, lmfit, GPyTorch, BoTorch, and Ollama
- [Materials-Informatics](https://github.com/srinikhil25/Materials-Informatics) — the Streamlit predecessor that informed this rewrite
