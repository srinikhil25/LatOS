"""Reading the commit and the state of a working tree, against real repositories.

Each repository here is made in a temporary directory with Git itself, so what
is tested is what Git says and not an imitation of it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from latos.experiments import SourceRevision, runtime_environment, source_revision
from latos.experiments import revision as revision_module
from latos.experiments.revision import (
    WORKING_TREE_CLEAN,
    WORKING_TREE_DIRTY,
    WORKING_TREE_STATES,
    WORKING_TREE_UNKNOWN,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs Git on the path")
COMMIT = "ab" * 20


def _git(repo: Path, *args: str) -> str:
    """Run Git in `repo`, ignoring whatever the machine's own Git settings are."""
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    identity = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid")
    done = subprocess.run(
        ["git", "-C", str(repo), *identity, "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one commit: a source file, and an ignore list."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    _git(root, "init", "--quiet")
    (root / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("data/\n*.log\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "first")
    return root


# ─── What Git says ──────────────────────────────────────────────────
@needs_git
def test_a_tree_that_is_exactly_its_commit_is_clean(repo: Path) -> None:
    revision = source_revision(repo / "src")
    assert revision.git_commit == _git(repo, "rev-parse", "HEAD")
    assert revision.git_commit is not None
    assert len(revision.git_commit) == 40
    assert revision.working_tree == WORKING_TREE_CLEAN
    assert revision.changed_paths == ()
    assert revision.note == ""
    assert revision.is_clean is True
    # A file inside the tree is as good a place to ask from as its directory.
    assert source_revision(repo / "src" / "code.py") == revision


@needs_git
def test_an_untracked_file_makes_the_tree_dirty(repo: Path) -> None:
    # A new module is code that can run, whether or not Git has been told of it.
    (repo / "src" / "new.py").write_text("y = 2\n", encoding="utf-8")
    revision = source_revision(repo / "src")
    assert revision.working_tree == WORKING_TREE_DIRTY
    assert revision.changed_paths == ("?? src/new.py",)
    assert revision.is_clean is False
    assert revision.git_commit == _git(repo, "rev-parse", "HEAD")  # the commit is still stated


@needs_git
def test_a_modified_file_makes_the_tree_dirty_staged_or_not(repo: Path) -> None:
    (repo / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")
    assert source_revision(repo).changed_paths == (" M src/code.py",)
    _git(repo, "add", "-A")
    staged = source_revision(repo)
    assert staged.working_tree == WORKING_TREE_DIRTY
    assert staged.changed_paths == ("M  src/code.py",)


@needs_git
def test_a_deleted_file_makes_the_tree_dirty(repo: Path) -> None:
    (repo / "src" / "code.py").unlink()
    assert source_revision(repo).changed_paths == (" D src/code.py",)


@needs_git
def test_every_untracked_file_is_listed_not_just_its_directory(repo: Path) -> None:
    (repo / "pkg").mkdir()
    for name in ("a.py", "b.py"):
        (repo / "pkg" / name).write_text("", encoding="utf-8")
    assert source_revision(repo).changed_paths == ("?? pkg/a.py", "?? pkg/b.py")


@needs_git
def test_ignored_files_do_not_count(repo: Path) -> None:
    (repo / "data").mkdir()
    (repo / "data" / "run.csv").write_text("1,2\n", encoding="utf-8")
    (repo / "run.log").write_text("ok\n", encoding="utf-8")
    assert source_revision(repo / "src").is_clean is True


@needs_git
def test_code_in_an_ignored_path_is_described_by_no_commit(repo: Path) -> None:
    # An installed copy inside someone's checkout is not that checkout.
    inside = repo / "data" / "site-packages"
    inside.mkdir(parents=True)
    revision = source_revision(inside)
    assert revision.working_tree == WORKING_TREE_UNKNOWN
    assert revision.git_commit is None
    assert "ignores" in revision.note
    assert revision.is_clean is False


@needs_git
def test_a_directory_outside_any_repository_is_unknown(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    revision = source_revision(plain)
    assert (revision.git_commit, revision.working_tree) == (None, WORKING_TREE_UNKNOWN)
    assert revision.note == "the code is not inside a Git working tree"


@needs_git
def test_a_repository_with_no_commit_is_unknown(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _git(empty, "init", "--quiet")
    revision = source_revision(empty)
    assert (revision.git_commit, revision.working_tree) == (None, WORKING_TREE_UNKNOWN)
    assert revision.note == "the working tree has no commit yet"


@needs_git
@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_a_file_git_was_told_not_to_look_at_is_never_reported_clean(repo: Path, flag: str) -> None:
    # Edited, then hidden: `git status` says nothing at all about this file.
    (repo / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "update-index", flag, "src/code.py")
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""

    revision = source_revision(repo / "src")
    assert revision.working_tree == WORKING_TREE_UNKNOWN
    assert revision.is_clean is False
    assert revision.git_commit == _git(repo, "rev-parse", "HEAD")
    assert "1 tracked file(s) are marked assume-unchanged or skip-worktree" in revision.note
    assert "src/code.py" in revision.note


@needs_git
def test_a_hidden_file_is_unknown_even_when_it_was_not_edited(repo: Path) -> None:
    # Whether it differs is exactly what Git will no longer say.
    _git(repo, "update-index", "--assume-unchanged", "src/code.py")
    assert source_revision(repo).working_tree == WORKING_TREE_UNKNOWN
    _git(repo, "update-index", "--no-assume-unchanged", "src/code.py")
    assert source_revision(repo).is_clean is True


@needs_git
def test_a_visible_change_is_dirty_whatever_else_is_hidden(repo: Path) -> None:
    (repo / "src" / "other.py").write_text("z = 3\n", encoding="utf-8")
    _git(repo, "update-index", "--assume-unchanged", "src/code.py")
    revision = source_revision(repo)
    assert revision.working_tree == WORKING_TREE_DIRTY
    assert revision.changed_paths == ("?? src/other.py",)


@needs_git
def test_asking_changes_nothing_in_the_repository(repo: Path) -> None:
    # Rewritten with the same content: Git would normally refresh its index
    # here as a side effect of `status`. The question must not write.
    code = repo / "src" / "code.py"
    code.write_text(code.read_text(encoding="utf-8"), encoding="utf-8")
    index = repo / ".git" / "index"
    before = (index.read_bytes(), index.stat().st_mtime_ns)
    listing = sorted(p.name for p in (repo / ".git").iterdir())

    assert source_revision(repo).is_clean is True

    assert (index.read_bytes(), index.stat().st_mtime_ns) == before
    assert sorted(p.name for p in (repo / ".git").iterdir()) == listing


def test_the_default_asks_about_the_tree_the_code_was_loaded_from() -> None:
    revision = source_revision()
    # Whatever the answer is on this machine, it is one of the three and is coherent.
    assert revision.working_tree in WORKING_TREE_STATES
    if revision.working_tree == WORKING_TREE_UNKNOWN:
        assert revision.note
    else:
        assert revision.git_commit is not None
    assert (revision.working_tree == WORKING_TREE_DIRTY) == bool(revision.changed_paths)


# ─── When Git cannot answer ─────────────────────────────────────────
@pytest.mark.parametrize(
    ("failure", "note"),
    [
        (FileNotFoundError("git"), "Git is not installed or not on the path"),
        (PermissionError("git"), "Git could not be run (PermissionError)"),
        (subprocess.TimeoutExpired("git", 30), "Git could not be run (TimeoutExpired)"),
    ],
)
def test_a_git_that_cannot_be_run_gives_unknown_and_never_raises(
    failure: Exception, note: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(revision_module.subprocess, "run", broken)
    revision = source_revision(tmp_path)
    assert revision == SourceRevision(git_commit=None, working_tree=WORKING_TREE_UNKNOWN, note=note)


def test_a_status_that_fails_after_the_commit_was_read_keeps_the_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def scripted(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--show-toplevel" in command:
            return subprocess.CompletedProcess(command, 0, stdout=str(tmp_path) + "\n", stderr="")
        if "check-ignore" in command:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        if "--verify" in command:
            return subprocess.CompletedProcess(command, 0, stdout=COMMIT + "\n", stderr="")
        return subprocess.CompletedProcess(command, 128, stdout="", stderr="fatal")

    monkeypatch.setattr(revision_module.subprocess, "run", scripted)
    revision = source_revision(tmp_path)
    assert revision.git_commit == COMMIT
    assert revision.working_tree == WORKING_TREE_UNKNOWN  # never guessed clean
    assert revision.note == "Git could not report the state of the working tree"
    assert revision.is_clean is False


def test_every_question_is_asked_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[list[str]] = []

    def record(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        asked.append(command)
        out = str(tmp_path) if "--show-toplevel" in command else COMMIT
        code = 1 if "check-ignore" in command else 0
        return subprocess.CompletedProcess(command, code, stdout=out + "\n", stderr="")

    monkeypatch.setattr(revision_module.subprocess, "run", record)
    source_revision(tmp_path)
    assert len(asked) == 4
    assert all(command[:2] == ["git", "--no-optional-locks"] for command in asked)
    verbs = {command[4] for command in asked}
    assert verbs == {"rev-parse", "check-ignore", "status"}  # nothing that writes


def test_a_clean_status_is_checked_against_what_git_was_allowed_to_see(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listings = {"clean": "H a.py\nH b.py\n", "hidden": "H a.py\nh b.py\nS c.py\ns d.py\n"}
    asked: list[str] = []

    def scripted(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        verb = command[4]
        asked.append(verb)
        if "--show-toplevel" in command:
            out = str(tmp_path)
        elif verb == "ls-files":
            out = listings[which]
        elif verb == "status":
            out = ""
        else:
            out = COMMIT
        code = 1 if verb == "check-ignore" else 0
        return subprocess.CompletedProcess(command, code, stdout=out + "\n", stderr="")

    monkeypatch.setattr(revision_module.subprocess, "run", scripted)
    which = "clean"
    assert source_revision(tmp_path) == SourceRevision(COMMIT, WORKING_TREE_CLEAN)
    assert asked[-1] == "ls-files"  # the last word before "clean" is said

    which = "hidden"
    revision = source_revision(tmp_path)
    assert revision.working_tree == WORKING_TREE_UNKNOWN
    assert revision.git_commit == COMMIT
    assert revision.note.startswith("3 tracked file(s) are marked assume-unchanged or skip-")
    assert revision.note.endswith("(first: b.py)")


def test_a_listing_git_cannot_give_is_unknown_not_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def scripted(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        verb = command[4]
        if verb == "ls-files":
            return subprocess.CompletedProcess(command, 128, stdout="", stderr="fatal")
        out = (
            str(tmp_path) if "--show-toplevel" in command else ("" if verb == "status" else COMMIT)
        )
        code = 1 if verb == "check-ignore" else 0
        return subprocess.CompletedProcess(command, code, stdout=out + "\n", stderr="")

    monkeypatch.setattr(revision_module.subprocess, "run", scripted)
    revision = source_revision(tmp_path)
    assert (revision.git_commit, revision.working_tree) == (COMMIT, WORKING_TREE_UNKNOWN)
    assert revision.note == "Git could not list the tracked files"


# ─── The value ──────────────────────────────────────────────────────
def test_only_a_clean_tree_with_a_commit_is_clean() -> None:
    assert SourceRevision(COMMIT, WORKING_TREE_CLEAN).is_clean is True
    assert SourceRevision(None, WORKING_TREE_CLEAN).is_clean is False
    assert SourceRevision(COMMIT, WORKING_TREE_DIRTY, ("?? a.py",)).is_clean is False
    assert SourceRevision(COMMIT, WORKING_TREE_UNKNOWN, note="why").is_clean is False


def test_a_revision_describes_itself_in_one_line() -> None:
    assert SourceRevision(COMMIT, WORKING_TREE_CLEAN).describe() == f"{COMMIT} (working tree clean)"
    dirty = SourceRevision(COMMIT, WORKING_TREE_DIRTY, (" M a.py", "?? b.py"))
    assert dirty.describe() == f"{COMMIT} (working tree DIRTY: 2 path(s) differ)"
    unknown = SourceRevision(None, WORKING_TREE_UNKNOWN, note="no Git here")
    assert unknown.describe() == "no commit (working tree UNKNOWN: no Git here)"


def test_a_record_lists_the_first_paths_and_counts_them_all() -> None:
    paths = tuple(f"?? module_{i}.py" for i in range(60))
    stored = SourceRevision(COMMIT, WORKING_TREE_DIRTY, paths).to_dict()
    assert stored["changed_path_count"] == 60
    assert stored["changed_paths"] == list(paths[:50])
    assert set(stored) == {
        "git_commit",
        "working_tree",
        "changed_path_count",
        "changed_paths",
        "note",
    }


# ─── What the code ran in ───────────────────────────────────────────
def test_the_environment_names_the_interpreter_the_platform_and_the_libraries() -> None:
    environment = runtime_environment()
    assert list(environment) == [
        "python",
        "implementation",
        "platform",
        "machine",
        "numpy",
        "scipy",
        "scikit-learn",
    ]
    assert all(isinstance(value, str) and value for value in environment.values())
    assert environment == runtime_environment()  # the same question, the same answer


def test_a_library_that_is_not_installed_is_recorded_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = revision_module.metadata.version

    def without_scipy(name: str) -> str:
        if name == "scipy":
            raise revision_module.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(revision_module.metadata, "version", without_scipy)
    environment = runtime_environment()
    assert environment["scipy"] is None  # stated, not left out
    assert environment["numpy"] is not None
