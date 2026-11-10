"""Controlled editor integration for application tasks."""

from __future__ import annotations

import fcntl
import sys
from pathlib import Path

from weightcode.application.bundles import write
from weightcode.application.runtime import Application, prepare_run
from weightcode.harness.launch import ClaudeLauncher
from weightcode.sandbox.host import DEFAULT_BWRAP

TOOLS = {
    "python": ("Run NumPy-only Python in the isolated editor kernel. /work/trial is read-only; write programs "
               "under /work/patches. Only trial, patches and scratch are mounted.", {"code": {"type": "string"}}),
    "check": ("Execute a .py program from /work/patches and return selection accuracies and edit validation. "
              "Every submitted file consumes one check, including invalid programs.", {"program_file": {"type": "string"}}),
    "status": ("Remaining check and wall budgets and original selection accuracies.", {}),
    "finish": ("Freeze the best eligible checked program under the task's fixed selection rule. Ends editing; "
               "the final dataset is evaluated independently after the editor exits.", {}),
}

SYSTEM_PROMPT = """You write executable neural weight-update programs. Use only the provided MCP tools.
Read /work/trial/task.json, layout.json, forward.py, params.npz and experience.npz using the python tool.
The experience contains x_new, y_new, x_old, y_old. Inputs are uint8 RGB [N,3,224,224].
Write a self-contained NumPy program to /work/patches/<name>.py. It must define nonempty strings
HYPOTHESIS, PREDICTED_EFFECT and update(student, experience) -> {tensor_name: additive_delta}.
student has params, forward and layout. For this adapter, features is available as
student["forward"].__globals__["features"](params, images). Programs use Python -I; a bare import forward fails.
Each candidate starts from the original model, never a previous candidate. Programs are run in fresh sandboxes
with a deterministic virtual clock and resource caps. Selection labels are unavailable; check returns aggregate
new/old accuracies. The final validation dataset is unavailable and never used to choose a program.
The host chooses the eligible checked program with highest new+old accuracy sum, earliest on a tie;
otherwise it keeps the base. Eligibility requires a strictly positive new gain meeting min_new_gain and
old accuracy loss no greater than retention_drop. Call finish when ready. Do not claim a capability gain
based only on experience performance or on a proposed program that has not passed final validation.
"""


def synthesize(task: Path, out: Path, sandbox_python: Path, *, editor_model: str, checks: int,
               cells: int, wall_s: float, threads: int = 4, program_wall_s: float = 300,
               launcher=None, bwrap: Path = DEFAULT_BWRAP) -> dict:
    """Let the controlled editor write and check programs. Always sandboxed: a program a language model has just
    written never runs outside bubblewrap."""
    if sandbox_python is None:
        raise ValueError("synthesize always runs programs in the sandbox: --sandbox-python is required")
    launcher = launcher or ClaudeLauncher()
    prepare_run(task, out, sandbox_python, checks=checks, cells=cells, wall_s=wall_s,
                program_wall_s=program_wall_s, threads=threads, bwrap=bwrap)
    app = Application(out)
    p = app.paths
    session = {"schema": "weightcode.application_editor.v1", "paths": {"root": str(p.root)},
               "editor": {"model": editor_model, "effort": "high", "system_prompt_mode": "replace"},
               "budget": {"wall_timeout_s": wall_s}, "mcp": {"tool_timeout_ms": int((program_wall_s + 600) * 1000),
               "connect_timeout_ms": 120000}, "allowed_tools": [f"mcp__weightcode__{name}" for name in TOOLS]}
    write(p.session_json, session)
    (p.system_prompt).write_text(SYSTEM_PROMPT)
    prompt = "Install the capability described in /work/trial/task.json within its stated budgets and edit limits."
    p.task_prompt.write_text(prompt)
    environment = {name: str(threads) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    # Installed module entrypoint: no checkout path, private store or research instance identifier.
    write(p.mcp_json, {"mcpServers": {"weightcode": {"command": sys.executable,
        "args": ["-m", "weightcode.application.server", "--run", str(p.root)], "env": environment}}})
    try:
        outcome = launcher.launch(session, p, prompt)
        write(p.logs / "editor_outcome.json", outcome.as_dict())
        if outcome.exit_code != 0 or outcome.killed or outcome.error:
            app.freeze()
            raise RuntimeError("editor did not complete normally; outputs retained, no model finalized")
        # The server holds this same lock throughout all checks. Never finalize concurrently with it.
        with p.server_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return app.finalize()
    finally:
        app.close()


class EditorTools:
    """McpSession-compatible tools; deterministic host owns all scores and budget state."""

    def __init__(self, app: Application):
        self.app = app

    def tool_list(self) -> list[dict]:
        return [{"name": name, "description": text, "inputSchema": {"type": "object", "properties": fields,
                 "required": list(fields), "additionalProperties": False}} for name, (text, fields) in TOOLS.items()]

    def call(self, name: str, arguments: dict):
        from weightcode.harness.tools import ToolOutcome, UnknownTool

        if name not in TOOLS:
            raise UnknownTool(name)
        try:
            expected = TOOLS[name][1]
            if not isinstance(arguments, dict) or set(arguments) != set(expected) \
                    or any(not isinstance(v, str) for v in arguments.values()):
                raise ValueError("tool arguments do not match the schema")
            if name == "python":
                result = self.app.python(arguments["code"])
            elif name == "check":
                result = self.app.check(arguments["program_file"])
            elif name == "finish":
                self.app.require_open()
                result = self.app.freeze()
            else:
                result = self.app.status()
            return ToolOutcome(payload=result, is_error=False)
        except ValueError as exc:
            return ToolOutcome(payload={"error": str(exc)}, is_error=True)
