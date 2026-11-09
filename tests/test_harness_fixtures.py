"""Shared fixtures of the editor-harness tests: a toy evidence-v1 store and scripted sessions (no tests here).

The toy store is the evalkit one (``test_evalkit_instances_store``): dev seeds 100 and 101 (each the other's decoy),
main seeds 0..4, condition "deploy", 172 parameters, so the edit-size limit is 0.05 x 172 = 8.6 numbers and the
relative-Frobenius limit 0.5 applies to fc1.weight and fc2.weight. ``FIX_OPS`` lets a true-class-0 image (pixel 4
bright) excite hidden unit 0 and inhibit unit 1: it fixes the twelve planted (0 -> 1) errors of V_check and breaks
nothing, within every limit (edit size 2, relative norm about 0.42). Everything lives under pytest tmp dirs with
monkeypatched WEIGHTCODE_* variables; no test touches the real store, and no test starts Claude Code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_evalkit_instances_store import ToyStore, build_toy_store, build_v1

from weightcode.harness.runner import SANDBOX_PYTHON, SessionResult, run_session
from weightcode.harness.scripted import ScriptedLauncher, ToolReply, call_and_record

BWRAP = Path("/usr/bin/bwrap")
needs_sandbox = pytest.mark.skipif(
    not (BWRAP.exists() and SANDBOX_PYTHON.exists()),
    reason="scripted sessions need the real bubblewrap sandbox and the sandbox venv (scripts/setup_sandbox_venv.sh)",
)

FIX_OPS: list[dict[str, Any]] = [
    {"op": "add", "tensor": "fc1.weight", "index": [[0, 4], [1, 4]], "values": [6.0, -6.0]}
]
FIX_SCORE = 12  # the planted (0 -> 1) errors of V_check
BREAK_OPS: list[dict[str, Any]] = [{"op": "add", "tensor": "fc2.bias", "index": [[2]], "values": [100.0]}]
OVER_FROB_OPS: list[dict[str, Any]] = [{"op": "add", "tensor": "fc1.weight", "index": [[0, 0]], "values": [1000.0]}]
OVER_SIZE_OPS: list[dict[str, Any]] = [
    {"op": "add", "tensor": "fc1.weight", "index": [[5, k] for k in range(9)], "values": [0.01] * 9}
]


@dataclass
class HarnessStore:
    """A built toy store and its evidence-v1 instance ids by (split_role, seed)."""

    store: ToyStore
    ids: dict[tuple[str, int], str]

    def dev(self, seed: int = 100) -> str:
        return self.ids[("dev", seed)]

    def main(self, seed: int = 0) -> str:
        return self.ids[("main", seed)]

    @property
    def sessions(self) -> Path:
        return self.store.runs / "sessions"


@pytest.fixture
def harness_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> HarnessStore:
    """A toy store under a neutral directory name: the sandbox sees its session's host paths in
    /proc/self/mountinfo, so a path holding the test's name (for example an arm word) fails the self-test."""
    store = build_toy_store(tmp_path_factory.mktemp("hs") / "store")
    store.activate(monkeypatch)
    outcomes = build_v1(store)
    return HarnessStore(store, {(o.split_role, o.seed): o.instance_id for o in outcomes})


def patch_code(patches: Mapping[str, list[dict[str, Any]]], **extra: str) -> str:
    """Sandbox code writing each patch to /work/patches, signed with ``parent_sha256`` from /work/trial/task.json."""
    payload = json.dumps(json.dumps(dict(patches)))
    fields = json.dumps(json.dumps({"hypothesis": "unit 0 should read the class-0 pixel",
                                    "predicted_effect": "fixes (0 -> 1) errors, breaks none", **extra}))
    return (
        "import json\n"
        "parent = json.load(open('/work/trial/task.json'))['parent_sha256']\n"
        f"for name, ops in json.loads({payload}).items():\n"
        "    patch = {'schema': 'weightcode.patch.v0', 'parent_sha256': parent, 'ops': ops}\n"
        f"    patch.update(json.loads({fields}))\n"
        "    json.dump(patch, open('/work/patches/' + name, 'w'))\n"
        "print(sorted(json.loads(" + payload + ")))\n"
    )


Step = tuple[str, dict[str, Any] | None]


def scripted_session(instance_id: str, arm: str, steps: Iterable[Step], *, repeat: int = 0,
                     skip_audit: bool = True, **kwargs: Any) -> tuple[SessionResult, list[ToolReply]]:
    """Run one session whose editor is a fixed list of tool calls against the session's real MCP server."""
    calls = list(steps)

    def script(client: Any, replies: list[ToolReply]) -> None:
        for name, arguments in calls:
            call_and_record(client, replies, name, arguments)

    launcher = ScriptedLauncher(script)
    result = run_session(instance_id, arm, repeat, effort="low", wall_timeout_s=300.0, launcher=launcher,
                         skip_audit=skip_audit, **kwargs)
    return result, launcher.replies


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
