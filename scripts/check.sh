#!/usr/bin/env bash
# Wrapper for scripts/check.py — POSIX shells (Linux, macOS, Git Bash).
# All args are forwarded: `./scripts/check.sh --fix` works.
set -e
cd "$(dirname "$0")/.."
exec python scripts/check.py "$@"
