"""Rerun the kept program of a published session on its network and rebuild the edited network (no language model).

``replay_session`` stages the network and its experience exactly as the program transfer does
(``weightcode.harness.transfer.stage_student``), runs the session's final program once (``run_update``, in the
bubblewrap sandbox or with an ``UnisolatedRunner``), turns its deltas into the derived patch and applies it under the
limits the session recorded in ``session.json``. The rebuilt network's ``params_sha256`` is then compared with the
child hash of record: equal hashes mean the rerun reproduced the published network bit for bit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.evalkit.devreadout import run_limits
from weightcode.examples.fetch import file_sha256
from weightcode.harness.program import ProgramProblem, run_update, write_derived_patch
from weightcode.harness.transfer import stage_student
from weightcode.patch import apply_patch
from weightcode.sandbox.program import PROGRAM_WALL_S, ProgramRunner


class ReplayError(RuntimeError):
    """The session record, the program or the rerun does not match what was published."""


@dataclass(frozen=True)
class Replay:
    """The rebuilt network and how it compares with the record."""

    child: dict[str, np.ndarray]
    child_sha256: str
    exact: bool
    n_changed: int
    record: dict[str, Any]


def replay_session(instance: Instance, session_dir: Path, program_sha256: str, child_sha256: str,
                   runner: ProgramRunner, work: Path,
                   experience: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None,
                   wall_s: float = PROGRAM_WALL_S) -> Replay:
    """Rerun the final program of the session in ``session_dir`` on ``instance`` (module docstring).

    ``program_sha256`` and ``child_sha256`` are the published hashes; the program is refused unless its file has
    ``program_sha256``, which is what makes running it without the sandbox acceptable. ``experience`` replaces the
    default experience assembly (for example a scanskill instance's ``scan_experience``). ``work`` is created.
    """
    run = json.loads((session_dir / "run.json").read_text(encoding="utf-8"))
    final = run.get("final") or {}
    if run.get("instance", {}).get("params_sha256") != instance.params_sha256:
        raise ReplayError(f"{session_dir}: the session ran on {run.get('instance', {}).get('params_sha256')!r}, not on "
                          f"this network ({instance.params_sha256})")
    if final.get("child_sha256") != child_sha256:
        raise ReplayError(f"{session_dir}: the session's final child is {final.get('child_sha256')!r}, not the "
                          f"published {child_sha256}")
    program = session_dir / str(final.get("dir", "final")) / str(final.get("patch_id"))
    if not program.is_file():
        raise ReplayError(f"{program} is missing")
    if file_sha256(program) != program_sha256:
        raise ReplayError(f"{program}: sha256 {file_sha256(program)} is not the published {program_sha256}; it is "
                          "not run")
    limits, limits_source = run_limits(run, session_dir, n_tensors=len(instance.params))
    work.mkdir(parents=True)
    student, experience_file, experience_record = stage_student(instance, work / "student", experience=experience)
    try:
        update = run_update(student, experience_file, program, work / "run", runner, wall_s)
    except ProgramProblem as problem:
        raise ReplayError("the program failed: " + "\n".join(problem.messages)) from None
    patch_dir = work / "patch"
    patch_dir.mkdir()
    patch = write_derived_patch(update, instance.params, instance.params_sha256, patch_dir)
    result = apply_patch(instance.params, patch, patch_dir, limits, instance.params_sha256)
    record = {
        "session_id": run.get("session_id"), "program": program.name, "program_sha256": program_sha256,
        "runner": runner.as_dict(), "limits_source": limits_source,
        "experience_content_sha256": experience_record.get("content_sha256"),
        "deltas_sha256": update.record.get("deltas_sha256"), "tensors": update.tensors,
        "child_sha256": result.child_sha256, "child_sha256_of_record": child_sha256,
        "exact": result.child_sha256 == child_sha256, "n_changed": int(result.n_changed),
        "n_changed_of_record": final.get("n_changed"),
    }
    return Replay(child=result.child, child_sha256=result.child_sha256, exact=record["exact"],
                  n_changed=int(result.n_changed), record=record)


def recorded_experience_sha256(session_dir: Path) -> str | None:
    """The experience content hash the session recorded in ``session.json``, if any."""
    path = session_dir / "session.json"
    if not path.is_file():
        return None
    recorded: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8")).get("experience") or {}
    value = recorded.get("content_sha256")
    return str(value) if value is not None else None


__all__ = ["Replay", "ReplayError", "recorded_experience_sha256", "replay_session"]
