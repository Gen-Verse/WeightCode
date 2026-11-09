"""Application MCP entrypoint, reusing the tested JSON-RPC transport."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import sys
from pathlib import Path

from weightcode.application.bundles import write
from weightcode.application.editor import EditorTools
from weightcode.application.runtime import Application
from weightcode.harness.mcp_server import McpSession, ServerStop, serve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    app = Application(args.run)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != str(app.config["threads"]):
            raise ValueError(f"{name} must match the recorded thread count")
    p = app.paths
    with p.server_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # No restart resets: logs and slots remain, and a second editor server is refused.
        write(p.logs / "application_server_started.json", {"pid": os.getpid()})
        protocol_out = os.dup(1)
        stop = {"pending": False}
        mcp = McpSession(EditorTools(app), p)

        def on_signal(signum, frame):
            stop["pending"] = True
            if not mcp.busy:
                raise ServerStop(str(signum))

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        # Keep numerical/library diagnostics out of the JSON-RPC stream.
        with p.server_stderr.open("a") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                serve(mcp, sys.stdin.buffer, protocol_out, stop)
            except ServerStop:
                pass
            finally:
                app.close()
                os.close(protocol_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
