"""Which revision of the source produced a result, and in what it ran.

A frozen recommendation can be reproduced only if the code that computed it can
be recovered. This module asks Git two questions about the working tree that
holds the running code: which commit is checked out, and whether anything in
that tree differs from it.

The answer has three states and is never guessed:

* ``clean``: no tracked or untracked file differs from the commit. Files Git
  ignores are not compared.
* ``dirty``: something differs from the commit, or is not in it. Modified,
  staged and untracked files all count, because any of them can be code that
  ran. Ignored files do not count.
* ``unknown``: the question could not be answered. Git is not installed, the
  code is not in a working tree, it sits in a path Git ignores (an installed
  copy inside someone's checkout is not that checkout), or Git has been told
  not to look at some tracked file.

That last case matters because it is silent. A tracked file marked
``assume-unchanged`` or ``skip-worktree`` is left out of ``git status`` whether
or not it has been edited, so a tree holding one can look exactly like its
commit and not be. Such a tree is reported ``unknown``, never ``clean``.

Nothing here decides what to do about a tree that is not clean. That belongs to
whoever is about to freeze something.

The questions are asked read-only. ``git status`` normally refreshes the index
as a side effect; it is told not to.

The commit says which code ran. `runtime_environment` says what it ran in: the
interpreter, the platform, and the versions of the numerical libraries. The
same commit on another platform or another library version can differ in the
last digits of a fitted model, so a record keeps both.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "WORKING_TREE_CLEAN",
    "WORKING_TREE_DIRTY",
    "WORKING_TREE_STATES",
    "WORKING_TREE_UNKNOWN",
    "SourceRevision",
    "commit_known",
    "runtime_environment",
    "source_revision",
]

WORKING_TREE_CLEAN = "clean"
WORKING_TREE_DIRTY = "dirty"
WORKING_TREE_UNKNOWN = "unknown"
WORKING_TREE_STATES = (WORKING_TREE_CLEAN, WORKING_TREE_DIRTY, WORKING_TREE_UNKNOWN)

_GIT_TIMEOUT_S = 30
# A record lists this many changed paths and counts the rest.
_MAX_LISTED_PATHS = 50
_IGNORED = 0  # the exit code of `git check-ignore` for an ignored path
# `git ls-files -v` tags a skip-worktree file "S" and writes the tag of an
# assume-unchanged file in lower case.
_SKIP_WORKTREE = "S"

# The distributions whose versions can change a fitted model's digits.
_NUMERICAL_LIBRARIES = ("numpy", "scipy", "scikit-learn")


@dataclass(frozen=True, slots=True)
class SourceRevision:
    """The state of the source tree at one moment.

    Attributes:
        git_commit: The full hash of the checked-out commit, or None when it
            could not be read.
        working_tree: ``clean``, ``dirty`` or ``unknown``.
        changed_paths: For a dirty tree, one ``git status --porcelain`` line
            per path that differs from the commit.
        note: Why the state is ``unknown``, in a sentence; otherwise empty.
    """

    git_commit: str | None
    working_tree: str
    changed_paths: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_clean(self) -> bool:
        """Whether there is a commit and no tracked or untracked file differs from it."""
        return self.working_tree == WORKING_TREE_CLEAN and self.git_commit is not None

    def describe(self) -> str:
        """One line for a report."""
        commit = self.git_commit or "no commit"
        if self.working_tree == WORKING_TREE_CLEAN:
            return f"{commit} (working tree clean)"
        if self.working_tree == WORKING_TREE_DIRTY:
            return f"{commit} (working tree DIRTY: {len(self.changed_paths)} path(s) differ)"
        return f"{commit} (working tree UNKNOWN: {self.note})"

    def to_dict(self) -> dict[str, Any]:
        """The state as JSON-safe data, as a frozen record stores it."""
        return {
            "git_commit": self.git_commit,
            "working_tree": self.working_tree,
            "changed_path_count": len(self.changed_paths),
            "changed_paths": list(self.changed_paths[:_MAX_LISTED_PATHS]),
            "note": self.note,
        }


class _UnansweredError(Exception):
    """Git could not answer. The message says why, for the record."""


def _git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one read-only Git query.

    Raises:
        _UnansweredError: If Git could not be run at all.
    """
    command = ["git", "--no-optional-locks", "-C", str(where), *args]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError:
        raise _UnansweredError("Git is not installed or not on the path") from None
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _UnansweredError(f"Git could not be run ({type(exc).__name__})") from exc


def _hidden_from_status(root: Path) -> tuple[str, ...]:
    """Tracked files Git has been told not to compare with the commit.

    Raises:
        _UnansweredError: If Git could not list the tracked files.
    """
    listing = _git(root, "ls-files", "-v")
    if listing.returncode != 0:
        raise _UnansweredError("Git could not list the tracked files")
    hidden = []
    for line in listing.stdout.splitlines():
        tag, _, path = line.partition(" ")
        if tag == _SKIP_WORKTREE or tag.islower():
            hidden.append(path)
    return tuple(hidden)


def source_revision(start: Path | None = None) -> SourceRevision:
    """The commit and working-tree state of the source that holds `start`.

    Args:
        start: A directory inside the source tree to ask about. Defaults to the
            directory of this module, which is the tree the running code was
            loaded from.

    Returns:
        The revision. It never raises: a question Git cannot answer gives the
        ``unknown`` state, with the reason.
    """
    where = Path(start) if start is not None else Path(__file__).resolve().parent
    if where.is_file():
        where = where.parent
    commit: str | None = None
    try:
        top = _git(where, "rev-parse", "--show-toplevel")
        if top.returncode != 0:
            raise _UnansweredError("the code is not inside a Git working tree")
        root = Path(top.stdout.strip())

        if _git(root, "check-ignore", "--quiet", "--", str(where)).returncode == _IGNORED:
            raise _UnansweredError("the code sits in a path Git ignores, so no commit describes it")

        head = _git(root, "rev-parse", "--verify", "HEAD")
        if head.returncode != 0:
            raise _UnansweredError("the working tree has no commit yet")
        commit = head.stdout.strip()

        status = _git(root, "status", "--porcelain", "--untracked-files=all")
        if status.returncode != 0:
            raise _UnansweredError("Git could not report the state of the working tree")
        changed = tuple(line for line in status.stdout.splitlines() if line.strip())

        # An empty status is only as good as what Git was allowed to look at.
        hidden = () if changed else _hidden_from_status(root)
        if hidden:
            raise _UnansweredError(
                f"{len(hidden)} tracked file(s) are marked assume-unchanged or skip-worktree, "
                f"so Git does not report whether they differ from the commit (first: {hidden[0]})"
            )
    except _UnansweredError as why:
        return SourceRevision(git_commit=commit, working_tree=WORKING_TREE_UNKNOWN, note=str(why))

    state = WORKING_TREE_DIRTY if changed else WORKING_TREE_CLEAN
    return SourceRevision(git_commit=commit, working_tree=state, changed_paths=changed)


def commit_known(commit: str, start: Path | None = None) -> bool | None:
    """Whether the repository that holds `start` has `commit`, read-only.

    Args:
        commit: A full commit hash, as a record states it.
        start: A directory inside the repository to ask. Defaults to the
            directory of this module.

    Returns:
        True or False, or None when Git cannot say: it is not installed, or
        `start` is not in a working tree.
    """
    where = Path(start) if start is not None else Path(__file__).resolve().parent
    if where.is_file():
        where = where.parent
    try:
        top = _git(where, "rev-parse", "--show-toplevel")
        if top.returncode != 0:
            return None
        found = _git(Path(top.stdout.strip()), "cat-file", "-e", f"{commit}^{{commit}}")
    except _UnansweredError:
        return None
    return found.returncode == 0


def runtime_environment() -> dict[str, str | None]:
    """The interpreter, the platform and the numerical libraries the code runs in.

    Returns:
        JSON-safe data, as a frozen record stores it. A library that is not
        installed is recorded as None rather than left out, so its absence is
        a statement and not a gap.
    """
    environment: dict[str, str | None] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in _NUMERICAL_LIBRARIES:
        try:
            environment[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            environment[name] = None
    return environment
