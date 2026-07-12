"""Move a path to the OS trash (Recycle Bin on Windows).

Used by ``POST /project/delete`` to remove a project's *derived* ``.latos/``
store while leaving the raw files untouched. Recoverable by design — a
mis-click on the app's Delete button should be undoable from the Recycle Bin.
Falls back to a permanent delete only when the shell operation is unavailable
(non-Windows host, or the API call fails), since ``.latos/`` is always
regenerable by re-ingesting the folder.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import stat
import sys
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

__all__ = ["trash_path"]


def _clear_readonly(func: Callable[[str], None], path: str, _exc: object) -> None:
    """Clear the read-only bit and retry — rmtree onerror hook (SQLite/WAL)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _windows_recycle(path: Path) -> bool:
    """Send `path` to the Recycle Bin via SHFileOperationW. True on success."""

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = (
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_uint16),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        )

    fo_delete = 0x0003
    fof_allowundo = 0x0040
    fof_noconfirmation = 0x0010
    fof_silent = 0x0004
    fof_noerrorui = 0x0400

    op = SHFILEOPSTRUCTW()
    op.wFunc = fo_delete
    # pFrom is a double-null-terminated list of paths.
    op.pFrom = str(path) + "\x00\x00"
    op.fFlags = fof_allowundo | fof_noconfirmation | fof_silent | fof_noerrorui

    # `windll` is Windows-only (this function is only reached on win32); the
    # `unused-ignore` keeps type-checking clean on non-Windows CI too, where
    # typeshed does define the attribute so the ignore would otherwise be unused.
    shell32 = ctypes.windll.shell32  # type: ignore[attr-defined, unused-ignore]
    result = shell32.SHFileOperationW(ctypes.byref(op))
    return result == 0 and not op.fAnyOperationsAborted


def trash_path(path: Path) -> bool:
    """Remove `path`, preferring the Recycle Bin.

    Returns True if it was recycled (recoverable), False if it had to be
    permanently removed. A missing path counts as recycled (nothing to do).
    Raises OSError only if the path could not be removed at all.
    """
    if not path.exists():
        return True
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            if _windows_recycle(path):
                return True
    shutil.rmtree(path, onerror=_clear_readonly)
    return False
