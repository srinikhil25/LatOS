"""`python -m latos.server` — run the sidecar.

Binds to 127.0.0.1 ONLY. This is the privacy contract: the server is
an implementation detail of the desktop app, never a network service.
The Tauri shell passes `--port`; during development run it bare and
open http://127.0.0.1:8765/docs for the interactive API explorer.
"""

from __future__ import annotations

import argparse

import uvicorn

from latos.server.app import create_app

_DEFAULT_PORT = 8765
_LOCALHOST = "127.0.0.1"


def main() -> None:
    """Parse args and serve until the parent process kills us."""
    parser = argparse.ArgumentParser(prog="latos-server")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()

    uvicorn.run(
        create_app(),
        host=_LOCALHOST,  # never 0.0.0.0 — localhost is the contract
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
