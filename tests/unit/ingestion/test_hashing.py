"""Tests for `latos.ingestion.hashing`."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from latos.ingestion.hashing import (
    FileFingerprint,
    HashCache,
    fingerprint,
    hash_bytes,
    hash_file,
)


# ─── hash_bytes ─────────────────────────────────────────────────────
class TestHashBytes:
    def test_known_hash_matches_hashlib(self):
        data = b"hello, latos"
        assert hash_bytes(data) == hashlib.sha256(data).hexdigest()

    def test_empty_bytes(self):
        # Empty input still has a defined SHA-256.
        assert hash_bytes(b"") == hashlib.sha256(b"").hexdigest()

    def test_returns_64_lowercase_hex(self):
        digest = hash_bytes(b"test")
        assert len(digest) == 64
        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)


# ─── hash_file basic correctness ────────────────────────────────────
class TestHashFile:
    def test_matches_hash_bytes(self, tmp_path: Path):
        data = b"some content"
        f = tmp_path / "f.bin"
        f.write_bytes(data)
        assert hash_file(f) == hash_bytes(data)

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert hash_file(f) == hashlib.sha256(b"").hexdigest()

    def test_large_file_streamed(self, tmp_path: Path):
        # Force >1 read chunk (chunk size is 1 MB). Use 2.5 MB random-ish data.
        # Deterministic content so we can verify.
        data = (b"x" * 1024 * 1024) + (b"y" * 1024 * 1024) + (b"z" * 512 * 1024)
        f = tmp_path / "big.bin"
        f.write_bytes(data)
        assert hash_file(f) == hashlib.sha256(data).hexdigest()

    def test_idempotent(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"abc")
        assert hash_file(f) == hash_file(f)

    def test_different_content_different_hash(self, tmp_path: Path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"alpha")
        b.write_bytes(b"beta")
        assert hash_file(a) != hash_file(b)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            hash_file(tmp_path / "does-not-exist.bin")

    def test_directory_raises(self, tmp_path: Path):
        with pytest.raises(IsADirectoryError):
            hash_file(tmp_path)


# ─── fingerprint ────────────────────────────────────────────────────
class TestFingerprint:
    def test_records_size_and_mtime(self, tmp_path: Path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"12345")
        fp = fingerprint(f)
        assert fp.path == f
        assert fp.size == 5
        assert fp.mtime_ns == f.stat().st_mtime_ns

    def test_directory_raises(self, tmp_path: Path):
        with pytest.raises(IsADirectoryError):
            fingerprint(tmp_path)

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            fingerprint(tmp_path / "nope")


# ─── HashCache ──────────────────────────────────────────────────────
class TestHashCache:
    def test_get_miss_returns_none(self, tmp_path: Path):
        cache = HashCache()
        f = tmp_path / "f.bin"
        f.write_bytes(b"x")
        assert cache.get(fingerprint(f)) is None

    def test_put_then_get_returns_value(self, tmp_path: Path):
        cache = HashCache()
        f = tmp_path / "f.bin"
        f.write_bytes(b"x")
        fp = fingerprint(f)
        cache.put(fp, "deadbeef" * 8)
        assert cache.get(fp) == "deadbeef" * 8

    def test_len_reflects_entries(self, tmp_path: Path):
        cache = HashCache()
        assert len(cache) == 0
        f = tmp_path / "f.bin"
        f.write_bytes(b"x")
        cache.put(fingerprint(f), "a" * 64)
        assert len(cache) == 1

    def test_clear_empties_cache(self, tmp_path: Path):
        cache = HashCache()
        f = tmp_path / "f.bin"
        f.write_bytes(b"x")
        cache.put(fingerprint(f), "a" * 64)
        cache.clear()
        assert len(cache) == 0

    def test_different_path_different_key(self, tmp_path: Path):
        cache = HashCache()
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        cache.put(fingerprint(a), "h_a")
        # Same content, different path → cache miss.
        assert cache.get(fingerprint(b)) is None

    def test_size_change_invalidates(self):
        cache = HashCache()
        fp1 = FileFingerprint(path=Path("/a"), mtime_ns=100, size=10)
        fp2 = FileFingerprint(path=Path("/a"), mtime_ns=100, size=11)
        cache.put(fp1, "h1")
        assert cache.get(fp2) is None

    def test_mtime_change_invalidates(self):
        cache = HashCache()
        fp1 = FileFingerprint(path=Path("/a"), mtime_ns=100, size=10)
        fp2 = FileFingerprint(path=Path("/a"), mtime_ns=200, size=10)
        cache.put(fp1, "h1")
        assert cache.get(fp2) is None


# ─── hash_file with cache ───────────────────────────────────────────
class TestHashFileWithCache:
    def test_cache_hit_returns_cached_value_without_reading(self, tmp_path: Path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"original")
        cache = HashCache()
        # Prime cache with a sentinel value that doesn't match the real hash.
        # If hash_file honors the cache, we get the sentinel back.
        sentinel = "z" * 64
        cache.put(fingerprint(f), sentinel)
        assert hash_file(f, cache=cache) == sentinel

    def test_cache_miss_computes_and_stores(self, tmp_path: Path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"hello")
        cache = HashCache()
        digest = hash_file(f, cache=cache)
        assert digest == hashlib.sha256(b"hello").hexdigest()
        # And the cache now contains it.
        assert cache.get(fingerprint(f)) == digest

    def test_cache_invalidates_on_content_change(self, tmp_path: Path):
        # Real-world scenario: file is hashed, then modified, then hashed again.
        # Cache must NOT return the stale hash.
        f = tmp_path / "f.bin"
        f.write_bytes(b"v1")
        cache = HashCache()
        h1 = hash_file(f, cache=cache)

        # Modify file. We need to ensure mtime_ns actually changes — on some
        # filesystems mtime resolution is coarse. Bump mtime explicitly.
        os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 1_000_000))
        f.write_bytes(b"v2-different-size")
        # mtime changes from write_bytes too, but size definitely differs.

        h2 = hash_file(f, cache=cache)
        assert h1 != h2

    def test_no_cache_arg_works(self, tmp_path: Path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"plain")
        # cache=None (default) means "don't use a cache" — should still hash correctly.
        assert hash_file(f) == hashlib.sha256(b"plain").hexdigest()
