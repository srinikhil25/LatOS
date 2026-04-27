"""One-shot quality gate — runs every check CI runs, fails fast.

Usage:
    python scripts/check.py            # Verify only (read-only). Mirrors CI.
    python scripts/check.py --fix      # Auto-fix lint + format, then re-verify.
    python scripts/check.py --fast     # Skip pytest (lint/format/mypy only).
    python scripts/check.py --no-cov   # Run pytest without coverage.

Exit code: 0 if every step passes, 1 on the first failure.

Why this exists:
    CI runs ruff, ruff format --check, mypy strict, and pytest. Forgetting
    any one of those locally means a red push and a wasted CI minute. This
    script collapses all four into one command, in the same order as CI,
    so "green here" implies "green there".

Design:
    - No third-party deps. Pure stdlib. Works on Windows, macOS, Linux.
    - Each step prints a banner so failures are easy to spot in scrollback.
    - Subprocess calls inherit stdout/stderr — full tool output is shown,
      not buffered or filtered.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ANSI colors. Fall back to no-color when stdout is not a TTY.
# Windows 10+ terminals support ANSI by default; older shells will see escape
# codes — acceptable trade-off vs. pulling in colorama.
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap `text` with ANSI color `code` if the terminal supports it."""
    if not _USE_COLOR:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def banner(label: str) -> None:
    """Print a visible section header for a step.

    Uses plain ASCII dashes — Windows cmd defaults to cp1252 and chokes on
    box-drawing characters when output is piped or redirected.
    """
    bar = "-" * 70
    print(f"\n{_c('36', bar)}")
    print(_c("36;1", f"  {label}"))
    print(f"{_c('36', bar)}")


def run(cmd: list[str], *, label: str) -> float:
    """Run `cmd` from the repo root, stream output, exit on failure.

    Returns the elapsed seconds. Exits the script (code 1) if the command
    returns non-zero, after printing a clear failure marker.
    """
    banner(label)
    print(_c("90", f"$ {' '.join(cmd)}\n"))
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        print(_c("31;1", f"\nFAILED: {label} (exit {result.returncode}, {elapsed:.1f}s)"))
        sys.exit(1)
    print(_c("32", f"\nOK: {label} ({elapsed:.1f}s)"))
    return elapsed


def python_exe() -> str:
    """Return the current interpreter — guarantees we use the active venv."""
    return sys.executable


def tool_exists(name: str) -> bool:
    """Return True if `name` is on PATH or installed as a console script."""
    return shutil.which(name) is not None


def main() -> int:
    """Parse args, run the gate steps in CI order, return exit code."""
    parser = argparse.ArgumentParser(
        description="Run all quality gates locally — lint, format, type-check, tests.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix ruff lint + format issues before verifying.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip pytest. Useful for quick iteration before a full run.",
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Run pytest without coverage reporting.",
    )
    args = parser.parse_args()

    py = python_exe()
    timings: dict[str, float] = {}
    targets = ["src", "tests", "migrations", "scripts"]

    # ─── 1. Ruff lint ─────────────────────────────────────────────────
    if args.fix:
        timings["ruff (autofix)"] = run(
            [py, "-m", "ruff", "check", "--fix", *targets],
            label="Ruff lint (autofix)",
        )
    else:
        timings["ruff"] = run(
            [py, "-m", "ruff", "check", *targets],
            label="Ruff lint",
        )

    # ─── 2. Ruff format ───────────────────────────────────────────────
    if args.fix:
        timings["ruff format (write)"] = run(
            [py, "-m", "ruff", "format", *targets],
            label="Ruff format (write)",
        )
    else:
        timings["ruff format --check"] = run(
            [py, "-m", "ruff", "format", "--check", *targets],
            label="Ruff format check",
        )

    # ─── 3. Mypy strict ───────────────────────────────────────────────
    timings["mypy"] = run(
        [py, "-m", "mypy", "src"],
        label="Mypy strict",
    )

    # ─── 4. Pytest ────────────────────────────────────────────────────
    if not args.fast:
        pytest_cmd = [
            py,
            "-m",
            "pytest",
            "-m",
            "not ui and not gpu and not ollama",
        ]
        if args.no_cov:
            pytest_cmd.append("--no-cov")
        timings["pytest"] = run(pytest_cmd, label="Pytest")
    else:
        print(_c("33", "\n(skipped pytest -- --fast)"))

    # ─── Summary ──────────────────────────────────────────────────────
    banner("All checks passed")
    total = sum(timings.values())
    for name, secs in timings.items():
        print(f"  {name:<28} {secs:>6.1f}s")
    print(_c("32;1", f"\n  total                        {total:>6.1f}s"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
