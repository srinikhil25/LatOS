# AGENTS.md

This file provides instructions for AI coding agents working on the Latos codebase. It complements `CLAUDE.md` (which is more conversational) with concrete operational rules.

## Universal Rules

1. **Read `CLAUDE.md` first.** It has architecture, layout, and tech stack details.
2. **Run tests before claiming a task is done.** `pytest` must pass.
3. **Don't break the headless capability.** Analysis code must work without PySide6.
4. **Update `RESULTS_LOG.md`** whenever you complete a stage milestone.
5. **Stage scope discipline.** Don't bleed work from later stages into earlier ones.

## Pre-Flight Checklist (Before Writing Code)

- [ ] Have you read `CLAUDE.md` and the relevant section of `STAGES.md`?
- [ ] Do you know which **stage** the task belongs to?
- [ ] Does the task require new dependencies? (If yes, update `pyproject.toml`)
- [ ] Will this change touch the database schema? (If yes, write an Alembic migration)
- [ ] Do you have test data available for this task?

## Per-Task Workflow

```
1. Read STAGES.md section for current stage
2. Plan: list files to create/modify, tests to add
3. Implement: write code + tests together
4. Lint: `ruff check --fix && ruff format .`
5. Type-check: `mypy src/`
6. Test: `pytest tests/<relevant>/`
7. Update RESULTS_LOG.md if a milestone was hit
8. Commit with descriptive message
```

## Commit Message Format

```
<stage>: <type>: <short description>

[optional body explaining WHY, not WHAT]

Refs: #<issue> (if applicable)
```

Examples:
```
stage1: feat: add ProjectRepository.save() with file hashing
stage1: test: golden-file tests for XRD ASC parser
stage1: fix: SQLite WAL not cleaned on app close
stage2: refactor: extract SampleHints from project_builder
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

## Testing Discipline

### Required for every new feature:
- [ ] At least one **unit test** for the core function
- [ ] Edge cases (empty input, malformed input, boundary conditions)
- [ ] If touching parsers: at least one **golden-file test**
- [ ] If touching plots: a **snapshot test**
- [ ] If touching UI: a **pytest-qt test** clicking through the flow

### Test naming
```
test_<function>_<scenario>_<expected>
```
Examples:
- `test_parse_xrd_asc_valid_file_returns_pattern`
- `test_parse_xrd_asc_truncated_file_raises_validation_issue`
- `test_kubelka_munk_zero_reflectance_returns_inf`

## When You Find a Bug

1. **First:** Write a failing test that reproduces it
2. **Second:** Fix the code so the test passes
3. **Third:** Add the bug + fix to `RESULTS_LOG.md` under "Bugs Found & Fixed"

This guarantees regression coverage.

## When the User Asks for "Just a Quick Fix"

Don't skip:
- Tests (write at least one)
- Type hints (mypy strict mode is non-negotiable)
- Error handling (no bare `except:` clauses)

A "quick fix" is still going through code review and CI. Make it correct the first time.

## When You're Stuck

1. **Search the predecessor:** The Streamlit version at `D:/Materials-Informatics/` likely has a working implementation — adapt it.
2. **Check `tests/fixtures/`:** Real data examples are often clearer than docs.
3. **Read `docs/algorithms/`:** Technique-specific math is documented.
4. **Ask the user.** Don't guess at materials science domain knowledge — verify.

## What Triggers a User Confirmation

The agent should pause and ask the user before:

- Deleting files in `data/projects/` (user data)
- Schema migrations that drop or rename columns
- Adding dependencies to `pyproject.toml`
- Changing `pyproject.toml` lint/type rules
- Renaming public APIs
- Modifying `STAGES.md` (it's the source of truth for direction)

## Stage Boundary Rules

Each stage has a defined deliverable. **Do not implement Stage N+1 features while working on Stage N.** Examples:

| Working on… | Don't add… |
|-------------|-----------|
| Stage 1 (Persistence) | Sample clustering UI (Stage 2) |
| Stage 2 (Mech labeling) | VLM workers (Stage 5) |
| Stage 3 (Validation) | Fit engine (Stage 4) |
| Stage 4 (Fit Engine) | Bayesian Optimization (Stage 7) |

If you find yourself wanting to, **note it in `STAGES.md` under the relevant stage** and move on.

## Performance Discipline

Latos targets these performance budgets:

| Operation | Target | Stage |
|-----------|--------|-------|
| Project reload (Dhivya's 161 files) | <1 sec | 1 |
| First scan of 1000 files | <30 sec | 1 |
| Memory after typical project load | <500 MB | 1 |
| XRD pattern fit with 5 peaks | <2 sec | 4 |
| GP surrogate fit (50 points) | <5 sec | 7 |

If your change makes any of these regress, **don't merge** — investigate first.

## Documentation Discipline

Every public function/class needs:
- Google-style docstring with `Args`, `Returns`, `Raises`
- Type hints on every parameter and return
- At least one usage example in tests

For internal/private code:
- Single-line docstring is OK
- Type hints still required

## What's Forbidden

- `print()` for logging — use `loguru`
- `os.environ.get()` directly — use `pydantic-settings`
- Raw SQL outside `persistence/` — use repositories
- PySide6 imports outside `ui/` — breaks headless mode
- Skipping tests "for speed" — there is no speed without tests
- Touching `STAGES.md` without user permission
- Committing files >5 MB
- Committing `*.db`, `*.parquet`, or anything in `data/projects/`
- Force pushes to `main`
- Auto-merging without CI green
