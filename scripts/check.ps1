# Wrapper for scripts/check.py — PowerShell.
# All args are forwarded: `./scripts/check.ps1 --fix` works.
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    & python scripts/check.py @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
