# Contributing to Latos

Thanks for your interest in contributing! This document covers what you need to know.

## Getting Set Up

```bash
git clone https://github.com/srinikhil25/latos.git
cd latos
python -m venv venv
# Activate venv (Windows: venv\Scripts\activate, Unix: source venv/bin/activate)
pip install -e ".[all]"
pre-commit install
```

Verify your setup:
```bash
pytest          # tests should pass
ruff check .    # linting should pass
mypy src/       # type-checking should pass
```

## Development Workflow

1. **Find or open an issue** before starting non-trivial work
2. **Branch from `main`:** `git checkout -b stageN-feature-name`
3. **Write tests alongside code** — not after
4. **Keep PRs focused.** One feature/fix per PR.
5. **Update `RESULTS_LOG.md`** for milestones and benchmarks
6. **Run pre-commit hooks before pushing** (they run automatically on commit)

## Code Style

- **Formatter & linter:** ruff (configured in `pyproject.toml`)
- **Type checker:** mypy strict mode
- **Docstrings:** Google style on every public function/class
- **Line length:** 100 chars

Run before committing:
```bash
ruff check . --fix
ruff format .
mypy src/
```

## Testing

- **Coverage gate:** 70% minimum (increases over time)
- Every new feature needs tests in the same PR
- Bug fixes need a regression test

Test categories:
| Marker | What | Run with |
|--------|------|----------|
| (default) | Fast unit tests | `pytest` |
| `slow` | Long-running | `pytest -m slow` |
| `integration` | Full pipelines | `pytest -m integration` |
| `ui` | Qt UI tests | `pytest -m ui` |
| `gpu` | Need CUDA | `pytest -m gpu` |
| `ollama` | Need local Ollama | `pytest -m ollama` |

## Architecture Rules (Don't Break These)

1. **No PySide6 outside `src/latos/ui/`** — analysis core must be importable headlessly
2. **No SQL outside `src/latos/persistence/`** — use repository pattern
3. **No `print()`** — use `loguru`
4. **No raw `os.environ.get()`** — use `pydantic-settings`
5. **No bare `except:`** — always specify exception types

See `AGENTS.md` for the full list.

## Stage Discipline

Latos is built in 8 stages. Each PR must declare its stage in the title:

```
stage1: feat: add ProjectRepository.save()
stage4: fix: lmfit convergence on degenerate XPS spectra
```

If your work crosses stages, split into multiple PRs.

## Commit Message Format

```
<stage>: <type>: <short summary under 50 chars>

[optional body explaining WHY, not WHAT]

Refs: #issue (if applicable)
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

## Pull Request Checklist

- [ ] PR title follows `stage: type: description` format
- [ ] All tests pass (`pytest`)
- [ ] Coverage hasn't dropped (`pytest --cov`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type-checking passes (`mypy src/`)
- [ ] New features have docstrings
- [ ] New features have tests
- [ ] `RESULTS_LOG.md` updated if a milestone was hit
- [ ] No new dependencies added without justification

## Reporting Bugs

Use [GitHub Issues](https://github.com/srinikhil25/latos/issues). Include:

1. Latos version (`latos --version`)
2. OS + Python version
3. Steps to reproduce
4. Expected vs. actual behavior
5. Screenshots if UI-related
6. Sample data file if data-related (if shareable)

## Suggesting Features

Open a GitHub issue tagged `enhancement`. Describe:
1. The use case (what are you trying to do?)
2. Why current Latos can't do it
3. Your proposed approach (optional)

Note that Latos is being built in 8 stages — new features may be deferred to a relevant later stage.

## License

By contributing, you agree your contributions are licensed under MIT.
