"""`python -m latos <command>` — the command line for a running campaign.

Deliberately small. The desktop app is where a project is browsed; this is for
the two things that happen at a bench between samples, where opening a GUI is
friction and writing a script is worse.

Commands:
    next        Read the recording workbook, recommend the next composition,
                and freeze the prediction before the sample exists.
    rehearse    Simulate the planned campaign against known objectives, to size
                the budget and audition a prior, before spending anything.
"""

from __future__ import annotations

import argparse
import sys

__all__ = ["main"]

_USAGE = "python -m latos {next,rehearse} ..."


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a subcommand. Returns the process exit code."""
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        print(f"usage: {_USAGE}")
        return 0 if args else 2

    command, rest = args[0], args[1:]

    # Imported inside the branch so that `python -m latos next` does not pay for
    # the optimizer's import when the user only wanted `--help`.
    if command == "next":
        from latos.campaign_cycle import main as run_next  # noqa: PLC0415

        return run_next(rest)
    if command == "rehearse":
        return _rehearse(rest)

    print(f"Unknown command {command!r}.")
    print(f"usage: {_USAGE}")
    return 2


def _rehearse(argv: list[str]) -> int:
    """Size a campaign, and audition a prior, before any sample is made."""
    from latos.optimization.rehearsal import rehearse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="latos rehearse",
        description="Simulate the planned campaign against known objectives.",
    )
    parser.add_argument("--low", type=float, default=0.0, help="low end of the search range")
    parser.add_argument("--high", type=float, default=1.0, help="high end of the search range")
    parser.add_argument(
        "--noise",
        type=float,
        required=True,
        help="measurement scatter as a fraction of the objective's range, e.g. 0.08",
    )
    parser.add_argument("--budget", type=int, required=True, help="total experiments available")
    parser.add_argument("--seeds", type=int, default=40, help="random repetitions per shape")
    args = parser.parse_args(argv)

    report = rehearse(
        bounds=(args.low, args.high),
        noise=args.noise,
        budget=args.budget,
        n_seeds=args.seeds,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
