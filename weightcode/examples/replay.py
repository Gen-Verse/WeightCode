"""Rerun the kept program of a published session on its network and rebuild the edited network (no language model).

``replay_session`` stages the network and its experience as the program transfer does
(``weightcode.harness.transfer.stage_student``: the model spec without its model id, the weights, forward.py, the
spec's input files and experience.npz), runs the session's final program once (``run_update``, in the bubblewrap
sandbox or with an ``UnisolatedRunner``), turns its deltas into the derived patch and applies it under the limits the
session recorded in ``session.json`` (as ``weightcode.evalkit.devreadout.run_limits`` reads them). The rebuilt network's
``params_sha256`` is then compared with the child hash of record: equal hashes mean the rerun reproduced the published
network bit for bit.

This module imports numpy-only parts of the engine, so the examples run without PyTorch; the two functions named above
live in modules that import it, hence the local ``stage_student`` and ``session_limits``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256, save_params
from weightcode.examples.fetch import file_sha256
from weightcode.harness.experience import EXPERIENCE_FILE, build_experience
from weightcode.harness.jsonio import json_text
from weightcode.harness.program import ProgramProblem, Student, run_update, write_derived_patch
from weightcode.harness.trial import input_file_names
from weightcode.patch import Limits, apply_patch
from weightcode.sandbox.program import PROGRAM_WALL_S, ProgramRunner

SESSION_JSON = "session.json"


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


def session_limits(session_dir: Path) -> Limits:
    """The edit limits a session applied, every field of ``Limits`` as its ``session.json`` records them."""
    path = session_dir / SESSION_JSON
    recorded = json.loads(path.read_text(encoding="utf-8")).get("limits") if path.is_file() else None
    if not isinstance(recorded, Mapping):
        raise ReplayError(f"{path} records no limits; the session's final cannot be rebuilt as it was applied")
    return Limits(**recorded)


def stage_student(instance: Instance, dest: Path,
                  experience: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None
                  ) -> tuple[Student, Path, dict[str, Any]]:
    """The program's view of ``instance`` written into the new directory ``dest`` (module docstring), every file
    checked against the hash ``private.json`` records."""
    dest.mkdir(mode=0o700)
    private = instance.private
    spec_path = instance.instance_dir / "model_spec.json"
    if file_sha256(spec_path) != private["model_spec_sha256"]:
        raise ReplayError(f"{instance.instance_id}: model_spec.json differs from private.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    layout = {key: value for key, value in spec.items() if key != "model_id"}
    (dest / "model_spec.json").write_text(json_text(layout), encoding="utf-8")
    if save_params(dest / "model.npz", instance.params) != instance.params_sha256:
        raise ReplayError(f"{instance.instance_id}: the saved weights do not hash to the network")
    shutil.copyfile(instance.instance_dir / "forward.py", dest / "forward.py")
    if file_sha256(dest / "forward.py") != private["forward_sha256"]:
        raise ReplayError(f"{instance.instance_id}: forward.py differs from private.json forward_sha256")
    inputs = input_file_names(spec)
    for name in inputs:
        shutil.copyfile(instance.instance_dir / name, dest / name)
    arrays, record = build_experience(instance) if experience is None else experience
    np.savez(dest / EXPERIENCE_FILE, **arrays)
    with np.load(dest / EXPERIENCE_FILE, allow_pickle=False) as archive:
        if params_sha256({key: archive[key] for key in archive.files}) != record["content_sha256"]:
            raise ReplayError(f"{instance.instance_id}: the written experience differs from the assembled one")
    student = Student(params=instance.params, params_file=dest / "model.npz", forward_file=dest / "forward.py",
                      input_files={name: dest / name for name in inputs}, layout=layout)
    return student, dest / EXPERIENCE_FILE, record


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
    limits = session_limits(session_dir)
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
        "runner": runner.as_dict(), "limits_source": SESSION_JSON,
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
    path = session_dir / SESSION_JSON
    if not path.is_file():
        return None
    recorded: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8")).get("experience") or {}
    value = recorded.get("content_sha256")
    return str(value) if value is not None else None


__all__ = ["Replay", "ReplayError", "recorded_experience_sha256", "replay_session", "session_limits", "stage_student"]
