"""Content-addressable file hashing for change detection.

Latos uses SHA-256 hashes as the identity of a file's contents. Two files
with the same hash are guaranteed (modulo cosmic-ray collisions) to have
identical bytes — so the parser cache can be keyed on
`(file_hash, parser_version)` instead of paths.

This module is the **only** place SHA-256 is computed in Latos. Other
modules call `hash_file(path)`; nothing else.

Performance note: hashing a 1 GB .tif takes ~3s on an SSD. We stream in
1 MB chunks so we never load the full file into memory, and we expose a
mtime/size cache so re-scanning an unchanged folder costs ~0 extra time
beyond the `os.stat()` calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# 1 MB read buffer. Large enough that syscall overhead is negligible
# vs. the SHA-256 cost; small enough that even a tiny file doesn't waste
# heap. Tuned in benchmarks; do not change without re-benchmarking.
_CHUNK_SIZE = 1 << 20  # 1 MB


@dataclass
class FileFingerprint:
    """Identity stamp for a file at a specific moment.

    `mtime_ns` is integer nanoseconds (filesystem-precise on Linux/macOS,
    coarser on FAT/NTFS but still monotonic for our purposes). Combined
    with `size`, it's an extremely cheap proxy for "has this file
    changed?" — false positives (re-hash unchanged file) are acceptable;
    false negatives (skip a file that did change) would corrupt the cache.
    """

    path: Path
    mtime_ns: int
    size: int


@dataclass
class HashCache:
    """In-memory cache mapping `(path, mtime_ns, size) -> sha256 hex`.

    A single instance can be reused across many `hash_file` calls — the
    crawler creates one per scan and reuses it on subsequent refreshes,
    so unchanged files are never re-hashed.

    The cache is intentionally NOT persisted to disk: cache invalidation
    on a stale on-disk cache (e.g. after a power loss) is harder than
    re-hashing on the next launch.
    """

    _entries: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def get(self, fp: FileFingerprint) -> str | None:
        """Return the cached hash for this fingerprint, or None."""
        return self._entries.get((str(fp.path), fp.mtime_ns, fp.size))

    def put(self, fp: FileFingerprint, sha256: str) -> None:
        """Store the hash for this fingerprint."""
        self._entries[(str(fp.path), fp.mtime_ns, fp.size)] = sha256

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._entries.clear()


def fingerprint(path: Path) -> FileFingerprint:
    """Build a `FileFingerprint` from `path`'s current stat.

    Raises:
        FileNotFoundError: if `path` does not exist.
        IsADirectoryError: if `path` points at a directory.
    """
    if path.is_dir():
        raise IsADirectoryError(f"Cannot fingerprint a directory: {path}")
    stat = path.stat()
    return FileFingerprint(path=path, mtime_ns=stat.st_mtime_ns, size=stat.st_size)


def hash_file(path: Path, *, cache: HashCache | None = None) -> str:
    """Compute the SHA-256 of `path`'s contents.

    Args:
        path: File to hash. Must exist and be a regular file.
        cache: Optional `HashCache`. When provided, the file's
            (path, mtime_ns, size) is checked first; if present, the
            cached hex is returned without re-reading the file. After
            a cache miss, the freshly computed hash is stored.

    Returns:
        64-character lowercase hex SHA-256.

    Raises:
        FileNotFoundError: if `path` does not exist.
        IsADirectoryError: if `path` is a directory.
        OSError: on read failure.
    """
    fp = fingerprint(path)
    if cache is not None:
        cached = cache.get(fp)
        if cached is not None:
            return cached

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        # `iter(callable, sentinel)` is the canonical Python idiom for
        # streaming reads — terminates when read() returns b"".
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)

    digest = hasher.hexdigest()
    if cache is not None:
        cache.put(fp, digest)
    return digest


def hash_bytes(data: bytes) -> str:
    """Compute the SHA-256 of an in-memory byte string.

    Useful for tests and for hashing parser-generated content (e.g.
    canonicalized parsed data) without touching the filesystem.

    Returns:
        64-character lowercase hex SHA-256.
    """
    return hashlib.sha256(data).hexdigest()
