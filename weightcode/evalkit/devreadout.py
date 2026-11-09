"""Dev readout: held-out readouts of a dev instance's final child on V_select of its deployment condition.

Contract section 6 (dev readout, proposed in PREREG v0.2). Dev and main instances share the official test set, so a
dev instance never reads it. Its held-out readout is computed on V_select of the instance's deployment condition,
which no arm and no evidence draw reads; V_select served only base-model checks (a logged epoch monitor that never
selects, forward parity, recorded accuracy, the Colored-MNIST rho decision), none of which depends on an edit. The
fields are those of the sealed evaluation (``weightcode.evalkit.sealed.split_readouts``), so a dev number and a main number
mean the same thing. Dev readouts are development evidence: they guide protocol and prompt decisions and are not
reported as confirmation.

This module refuses main instances, superseded or stale instances (``check_task_current``) and never opens
``$WEIGHTCODE_SEALED``.

Usage::

    python -m weightcode.evalkit.devreadout PATH [PATH ...] [--out FILE]

Each PATH is a run.json or a directory searched for run.json files; a search that finds a run.json inside another
run's directory is refused (``run_files``), so for editor sessions name the run.json files. A run.json names its instance
(``instance.instance_id``) and its final pick (``final``). A base no-op final reads as zero deltas. Otherwise
``final.parent`` (default ``"instance"``) says how the child is recovered, and it must reproduce the recorded
``final.child_sha256`` either way:

- ``"instance"``: the final patch (``<run dir>/<final.dir>/patch.json``) is applied to the instance's base model with
  the executor, under the limits the run applied (``run_limits``): a harness session's own recorded limits (its
  ``session.json``, e.g. a program arm's 20% changed-entry cap); for a baseline run, the default limits, or the
  default limits without the size limits when its flags say it was not bound by them (a label-fit reference).
- ``"decoy_readdressed"`` (DECOY) and ``"zeros"`` (REWRITE, whose patch is addressed to the zero model): the saved
  child ``final.model``, a path relative to the run directory that must stay inside it, is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.companions import load_clean_rows
from weightcode.evalkit.e10 import e10_program_limits
from weightcode.patch.limits import dense_delta_bytes
from weightcode.evalkit.fewshot import FEWSHOT_RULE, fewshot_instance_dir, fewshot_program_limits, fewshot_readout
from weightcode.evalkit.fewshot import INSTANCE_ID_RE as FEWSHOT_ID_RE
from weightcode.evalkit.fscil import INSTANCE_ID_RE as FSCIL_ID_RE
from weightcode.evalkit.fscil import FSCIL_RULE, check_fscil_current, fscil_instance_dir, fscil_program_limits, fscil_readout
from weightcode.evalkit.graphnew import INSTANCE_ID_RE as GRAPHNEW_ID_RE
from weightcode.evalkit.graphnew import (
    GRAPHNEW_RULE,
    check_graphnew_current,
    graphnew_instance_dir,
    graphnew_program_limits,
    graphnew_readout,
)
from weightcode.evalkit.instances import check_task_current, instances_root
from weightcode.evalkit.scanskill import INSTANCE_ID_RE as SCAN_ID_RE
from weightcode.evalkit.scanskill import (
    SCAN_RULE,
    check_scan_current,
    load_recorded,
    scan_instance_dir,
    scan_program_limits,
    scan_readout,
    scan_readout_arrays,
)
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.sealed import split_readouts
from weightcode.evalkit.seqskill import INSTANCE_ID_RE as SEQSKILL_ID_RE
from weightcode.evalkit.seqskill import SEQSKILL_RULE, seqskill_instance_dir, seqskill_program_limits, seqskill_readout
from weightcode.evalkit.targets import deployment_splits, load_checked_split
from weightcode.patch import Limits, apply_patch_file, task_limits

DEFAULT_BATCH_SIZE = 1024
READOUT_SPLIT = "V_select"
SAVED_CHILD_PARENTS = ("decoy_readdressed", "zeros")


class DevReadoutError(RuntimeError):
    """A dev readout was refused: wrong instance role, stale instance, or a child that does not match its run."""


def _check_dev_instance(instance: Instance) -> None:
    role = instance.private.get("split_role")
    if role != "dev":
        raise DevReadoutError(f"instance {instance.instance_id} has split_role {role!r}; dev readouts are for dev only")
    check_task_current(instance.instance_dir)


def readout_split(instance: Instance) -> dict[str, np.ndarray]:
    """V_select of the instance's deployment condition, hash-checked against dataset.json."""
    splits = deployment_splits(str(instance.private["dataset"]), str(instance.private["condition"]))
    return load_checked_split(splits.v_select, splits.v_select_sha256)


def _check_child(instance: Instance, child: Mapping[str, np.ndarray]) -> None:
    base = instance.params
    if set(child) != set(base):
        raise DevReadoutError(f"child keys differ from the base of {instance.instance_id}: {sorted(set(child) ^ set(base))}")
    for key, value in base.items():
        if child[key].shape != value.shape or child[key].dtype != value.dtype:
            raise DevReadoutError(f"child {key} is {child[key].dtype}{child[key].shape}, base {value.dtype}{value.shape}")
        if not np.all(np.isfinite(child[key])):
            raise DevReadoutError(f"child {key} has non-finite values")


def dev_readout(
    instance: Instance,
    child: Mapping[str, np.ndarray],
    batch_size: int = DEFAULT_BATCH_SIZE,
    split: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Readouts of ``child`` against the instance's base model on V_select of its condition, paired per image."""
    _check_dev_instance(instance)
    _check_child(instance, child)
    data = readout_split(instance) if split is None else split
    base_pred = predict(instance.forward, instance.params, data["x"], batch_size)
    child_pred = predict(instance.forward, child, data["x"], batch_size)
    n_classes = len(instance.task["classes"])
    return {
        "instance_id": instance.instance_id,
        "dataset": instance.private["dataset"],
        "condition": instance.private["condition"],
        "split": READOUT_SPLIT,
        "n": int(len(data["y"])),
        "readouts": split_readouts(data, base_pred, child_pred, None, n_classes),
    }


def dev_readout_joint(
    instance: Instance, child: Mapping[str, np.ndarray], batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, Any]:
    """Readouts of one child on V_select of its condition (``readouts``) and on the clean companion rows
    (``readouts_clean``: the same rows with clean images, ``weightcode.evalkit.companions``), for the learning experiments.
    """
    record = dev_readout(instance, child, batch_size)
    clean = load_clean_rows(instance, READOUT_SPLIT)
    record["readouts_clean"] = dev_readout(instance, child, batch_size, split=clean.arrays())["readouts"]
    record["n_clean"] = int(len(clean.y))
    record["clean_split_sha256"] = clean.clean_sha256
    return record


SESSION_JSON = "session.json"


def run_limits(
    run: Mapping[str, Any],
    run_dir: Path,
    default: Limits | None = None,
    default_label: str = "task.json",
    n_tensors: int | None = None,
) -> tuple[Limits, str]:
    """The limits a run's final patch is rebuilt with, and where they come from.

    A harness session writes the limits it applied to its ``session.json`` next to run.json (every field of
    ``Limits``), and its final is rebuilt with exactly those: for a program arm such as LEARN that is a 20% cap on
    changed entries and no edit-size limit, which the default limits would refuse. A harness run.json (it names its
    ``session_result``) without that file is refused. A baseline run has no session.json; its
    ``flags.edit_limits_enforced`` says whether it ran under the default limits or without the size limits; without
    them, and given the model's ``n_tensors``, the op count may reach the tensor count, as the run's executor allowed
    (``Limits.without_edit_limits``).
    """
    session_path = run_dir / SESSION_JSON
    if session_path.is_file():
        recorded = json.loads(session_path.read_text()).get("limits")
        if not isinstance(recorded, Mapping):
            raise DevReadoutError(f"{session_path} records no limits; the final cannot be rebuilt as it was applied")
        return Limits(**recorded), SESSION_JSON
    if run.get("session_result") is not None:
        raise DevReadoutError(f"{run_dir}: a harness run without its {SESSION_JSON}; its applied limits are unknown")
    base, label = (Limits(), "default") if default is None else (default, default_label)
    if bool(run.get("flags", {}).get("edit_limits_enforced", True)):
        return base, label
    if n_tensors is None or n_tensors <= base.max_ops:
        return base.without_size_limits(), f"{label} without size limits"
    return base.without_edit_limits(n_tensors), f"{label} without size limits, max_ops {n_tensors}"


def final_child(
    run: Mapping[str, Any], run_dir: Path, instance: Instance, default_limits: Limits | None = None
) -> dict[str, np.ndarray]:
    """The final child a run selected, rebuilt from its patch and checked against the recorded child hash.

    ``default_limits`` are the limits a baseline run (no session.json) applied when it enforced them: ``Limits()``
    when None, the E2 program limits for a fewshot instance (``fewshot_program_limits``: head exempt from max_rel_frob,
    at most 20% of the entries changed, no edit-size limit).
    """
    final = run["final"]
    recorded = final.get("child_sha256")
    if final.get("is_base_noop"):
        if recorded is not None and recorded != instance.params_sha256:
            raise DevReadoutError(f"{run_dir}: base no-op final records child {recorded}, base is {instance.params_sha256}")
        return dict(instance.params)
    parent = final.get("parent", "instance")
    if parent == "instance":
        limits, _ = run_limits(run, run_dir, default_limits, n_tensors=len(instance.params))
        patch_path = run_dir / str(final["dir"]) / "patch.json"
        result = apply_patch_file(instance.params, patch_path, limits, instance.params_sha256)
        if result.child_sha256 != recorded:
            raise DevReadoutError(f"{run_dir}: rebuilt child {result.child_sha256} != recorded final child {recorded}")
        return result.child
    if parent not in SAVED_CHILD_PARENTS:
        raise DevReadoutError(f"{run_dir}: unknown final.parent {parent!r}; known: instance, {SAVED_CHILD_PARENTS}")
    relative = final.get("model")
    if not isinstance(relative, str) or not relative:
        raise DevReadoutError(f"{run_dir}: final.parent {parent!r} needs final.model, the saved child's relative path")
    root = run_dir.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise DevReadoutError(f"{run_dir}: final.model {relative!r} leaves the run directory")
    child = load_params(path)
    digest = params_sha256(child)
    if digest != recorded:
        raise DevReadoutError(f"{run_dir}: saved child {digest} != recorded final child {recorded}")
    return child


class NestedRunError(DevReadoutError):
    """A directory search found a run.json inside another run's directory."""


def run_files(paths: Sequence[str | Path]) -> list[Path]:
    """run.json files named directly or found under the given directories, in sorted order.

    Run records never nest. A run.json below a directory that itself holds a run.json is not a record the harness or a
    baseline runner wrote: an editor can write files under its session's ``patches/``. A directory search that finds
    one raises ``NestedRunError`` instead of reading it or silently skipping it. Naming run.json files directly avoids
    the search altogether and is the recommended use for editor sessions.
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            hits = sorted(path.rglob("run.json"))
            run_dirs = {hit.parent for hit in hits}
            nested = [hit for hit in hits if any(parent in run_dirs for parent in hit.parent.parents)]
            if nested:
                raise NestedRunError(
                    f"{path}: run.json files inside another run's directory {[str(n) for n in nested]}; "
                    "name the run.json files directly"
                )
            found.extend(hits)
        elif path.name == "run.json" and path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"{path} is neither a run.json nor a directory")
    return found


def readout_run(run_path: Path, batch_size: int = DEFAULT_BATCH_SIZE, clean: bool = False) -> dict[str, Any]:
    """Dev readout of the final child of one run.json, with the run's own V_check numbers next to it.

    With ``clean`` the record also carries the readouts on the clean companion of V_select (``dev_readout_joint``).
    """
    run = json.loads(run_path.read_text())
    if FEWSHOT_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        return _readout_fewshot_run(run, run_path, batch_size)
    if SEQSKILL_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        return _readout_seqskill_run(run, run_path, batch_size)
    if GRAPHNEW_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        return _readout_graphnew_run(run, run_path)
    if FSCIL_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        return _readout_fscil_run(run, run_path)
    if SCAN_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        return _readout_scan_run(run, run_path, batch_size)
    return _readout_repair_run(run, run_path, batch_size, clean)


def _readout_repair_run(run: Mapping[str, Any], run_path: Path, batch_size: int, clean: bool) -> dict[str, Any]:
    """readout_run of a repair-instance run: its final child on V_select, and with ``clean`` on its clean companion."""
    instance = load_instance(instances_root() / run["instance"]["instance_id"])
    program_limits, limits_label = None, "task.json"
    if str(run.get("limits_source", "")).startswith("e10_program_limits"):  # PREREG E10 reference runs
        expected = e10_program_limits(instance.task["limits"], len(instance.params),
                                      dense_delta_bytes(v.shape for v in instance.params.values()))
        program_limits = _program_baseline_limits(run, run_path.parent, expected, "e10", "E10")
        limits_label = E10_BASELINE_LIMITS
    child = final_child(run, run_path.parent, instance, program_limits)
    record = dev_readout_joint(instance, child, batch_size) if clean else dev_readout(instance, child, batch_size)
    final = run["final"]
    rebuilt_with_limits = not final.get("is_base_noop") and final.get("parent", "instance") == "instance"
    record.update(
        {
            "run": str(run_path),
            "method": run.get("method"),
            "final_limits": (run_limits(run, run_path.parent, program_limits, limits_label,
                                        n_tensors=len(instance.params))[1] if rebuilt_with_limits else None),
            "is_base_noop": bool(final.get("is_base_noop")),
            "edit_size": final.get("edit_size"),
            "vcheck": {"score": final.get("score"), "fixed": final.get("fixed"), "broken": final.get("broken")},
        }
    )
    return record


E10_BASELINE_LIMITS = "E10 program limits"
"""``final_limits`` label of an E10 baseline run, rebuilt with ``e10_program_limits``."""

FEWSHOT_BASELINE_LIMITS = "E2 program limits"
"""``final_limits`` label of a fewshot baseline run, rebuilt with ``fewshot_program_limits``."""


def _fewshot_baseline_limits(run: Mapping[str, Any], run_dir: Path, instance: Instance) -> Limits:
    """The E2 program limits of a fewshot instance; a baseline run on it must record them as the limits it applied.

    A harness run is rebuilt from its session.json (``run_limits``), so only a baseline run is checked here.
    """
    return _program_baseline_limits(run, run_dir, fewshot_program_limits(instance.task["limits"]), "fewshot", "E2")


def _program_baseline_limits(run: Mapping[str, Any], run_dir: Path, expected: Limits, kind: str, line: str) -> Limits:
    """``expected``, the program limits of a fewshot or seqskill instance, once a baseline run records them."""
    if (run_dir / SESSION_JSON).is_file() or run.get("session_result") is not None:
        return expected
    recorded = run.get("limits")
    try:
        matches = isinstance(recorded, Mapping) and Limits(**recorded) == expected
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise DevReadoutError(
            f"{run_dir}: a {kind} baseline run must record the {line} program limits {task_limits(expected)}; "
            f"it records {recorded!r}"
        )
    return expected


def _readout_fewshot_run(run: Mapping[str, Any], run_path: Path, batch_size: int) -> dict[str, Any]:
    """readout_run of a fewshot-v1 run (PREREG E2): ``fewshot_readout`` of its final child on the clean V_select."""
    instance = load_instance(fewshot_instance_dir(str(run["instance"]["instance_id"])))
    if instance.private.get("evidence_rule") != FEWSHOT_RULE or instance.private.get("split_role") != "dev":
        raise DevReadoutError(f"{instance.instance_id}: not a dev {FEWSHOT_RULE} instance")
    program_limits = _fewshot_baseline_limits(run, run_path.parent, instance)
    child = final_child(run, run_path.parent, instance, program_limits)
    final = run["final"]
    rebuilt_with_limits = not final.get("is_base_noop") and final.get("parent", "instance") == "instance"
    vcheck = {key: final.get(key) for key in ("score", "fixed", "broken", "score_new", "score_old")}
    return {
        "instance_id": instance.instance_id,
        "dataset": instance.private["dataset"],
        "arch": instance.private["arch"],
        "rule": FEWSHOT_RULE,
        "split": READOUT_SPLIT,
        "readouts_fewshot": fewshot_readout(instance, child, batch_size),
        "run": str(run_path),
        "method": run.get("method"),
        "final_limits": (
            run_limits(run, run_path.parent, program_limits, FEWSHOT_BASELINE_LIMITS)[1] if rebuilt_with_limits else None
        ),
        "is_base_noop": bool(final.get("is_base_noop")),
        "edit_size": final.get("edit_size"),
        "vcheck": vcheck,
    }


SEQSKILL_BASELINE_LIMITS = "E5 program limits"
"""``final_limits`` label of a seqskill baseline run, rebuilt with ``seqskill_program_limits``."""


def _readout_seqskill_run(run: Mapping[str, Any], run_path: Path, batch_size: int) -> dict[str, Any]:
    """readout_run of a seqskill-v1 run (PREREG E5): ``seqskill_readout`` of its final child on the condition's
    V_select, parts i to iv, child against base."""
    instance = load_instance(seqskill_instance_dir(str(run["instance"]["instance_id"])))
    if instance.private.get("evidence_rule") != SEQSKILL_RULE or instance.private.get("split_role") != "dev":
        raise DevReadoutError(f"{instance.instance_id}: not a dev {SEQSKILL_RULE} instance")
    program_limits = _program_baseline_limits(
        run, run_path.parent, seqskill_program_limits(instance.task["limits"], len(instance.params)), "seqskill", "E5")
    child = final_child(run, run_path.parent, instance, program_limits)
    final = run["final"]
    rebuilt_with_limits = not final.get("is_base_noop") and final.get("parent", "instance") == "instance"
    vcheck = {key: final.get(key) for key in ("score", "fixed", "broken", "score_new", "score_old")}
    return {
        "instance_id": instance.instance_id,
        "dataset": instance.private["dataset"],
        "condition": instance.private["condition"],
        "arch": instance.private["arch"],
        "seed": instance.private["seed"],
        "rule": SEQSKILL_RULE,
        "split": READOUT_SPLIT,
        "readouts_seqskill": seqskill_readout(instance, child, batch_size),
        "run": str(run_path),
        "method": run.get("method"),
        "final_limits": (
            run_limits(run, run_path.parent, program_limits, SEQSKILL_BASELINE_LIMITS,
                       n_tensors=len(instance.params))[1] if rebuilt_with_limits else None
        ),
        "respects_edit_limits": run.get("flags", {}).get("respects_edit_limits"),
        "is_base_noop": bool(final.get("is_base_noop")),
        "edit_size": final.get("edit_size"),
        "n_changed": final.get("n_changed"),
        "vcheck": vcheck,
    }


def json_safe(value: Any) -> Any:
    """``value`` with every non-finite float replaced by None, so a readout passes ``json.dumps(allow_nan=False)``.

    A graphnew readout holds an accuracy for every one of the 40 classes and hop bins; a class or bin with no rows in
    the readout split has accuracy NaN, which is recorded as null (its ``n`` is 0)."""
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


GRAPHNEW_BASELINE_LIMITS = "E7 program limits"
"""``final_limits`` label of a graphnew baseline run, rebuilt with ``graphnew_program_limits``."""


def _readout_graphnew_run(run: Mapping[str, Any], run_path: Path) -> dict[str, Any]:
    """readout_run of a graphnew-v1 run (PREREG E7): ``graphnew_readout`` of its final child on V_select (40-way,
    child against base; forward.py runs the whole graph, so there is no batch size)."""
    instance = load_instance(graphnew_instance_dir(str(run["instance"]["instance_id"])))
    if instance.private.get("evidence_rule") != GRAPHNEW_RULE or instance.private.get("split_role") != "dev":
        raise DevReadoutError(f"{instance.instance_id}: not a dev {GRAPHNEW_RULE} instance")
    check_graphnew_current(instance.instance_dir)
    program_limits = _program_baseline_limits(
        run, run_path.parent, graphnew_program_limits(instance.task["limits"], len(instance.params)), "graphnew", "E7")
    child = final_child(run, run_path.parent, instance, program_limits)
    final = run["final"]
    rebuilt_with_limits = not final.get("is_base_noop") and final.get("parent", "instance") == "instance"
    vcheck = {key: final.get(key) for key in ("score", "fixed", "broken", "score_new", "score_old")}
    return {
        "instance_id": instance.instance_id,
        "dataset": instance.private["dataset"],
        "arch": instance.private["arch"],
        "seed": instance.private["seed"],
        "rule": GRAPHNEW_RULE,
        "split": READOUT_SPLIT,
        "readouts_graphnew": json_safe(graphnew_readout(instance, child)),
        "run": str(run_path),
        "method": run.get("method"),
        "final_patch_id": final.get("patch_id"),
        "final_config": next((c.get("config") for c in run.get("candidates", [])
                              if c.get("patch_id") == final.get("patch_id")), None),
        "final_limits": (
            run_limits(run, run_path.parent, program_limits, GRAPHNEW_BASELINE_LIMITS,
                       n_tensors=len(instance.params))[1] if rebuilt_with_limits else None
        ),
        "respects_edit_limits": run.get("flags", {}).get("respects_edit_limits"),
        "is_base_noop": bool(final.get("is_base_noop")),
        "edit_size": final.get("edit_size"),
        "n_changed": final.get("n_changed"),
        "vcheck": vcheck,
    }


SCAN_BASELINE_LIMITS = "E13 program limits"
"""``final_limits`` label of a scanskill baseline run, rebuilt with ``scan_program_limits``."""


def _readout_scan_run(run: Mapping[str, Any], run_path: Path, batch_size: int) -> dict[str, Any]:
    """readout_run of a scanskill-v1 run (PREREG E13): ``scan_readout`` of its final child on the condition's new and
    old V_select, child against base. A rule-generated reference's run records its fit set's command ids
    (``config.fit_set.command_ids``), and its readout then carries the fit-set coverage beside the reading."""
    instance = load_instance(scan_instance_dir(str(run["instance"]["instance_id"])))
    if instance.private.get("evidence_rule") != SCAN_RULE or instance.private.get("split_role") != "dev":
        raise DevReadoutError(f"{instance.instance_id}: not a dev {SCAN_RULE} instance")
    check_scan_current(instance.instance_dir)
    program_limits = _program_baseline_limits(
        run, run_path.parent, scan_program_limits(instance.task["limits"], len(instance.params)), "scan", "E13")
    child = final_child(run, run_path.parent, instance, program_limits)
    fit_set = (run.get("config") or {}).get("fit_set")
    fit_ids = fit_set.get("command_ids") if isinstance(fit_set, Mapping) else None
    final = run["final"]
    rebuilt_with_limits = not final.get("is_base_noop") and final.get("parent", "instance") == "instance"
    vcheck = {key: final.get(key) for key in ("score", "fixed", "broken", "score_new", "score_old")}
    return {
        "instance_id": instance.instance_id,
        "dataset": instance.private["dataset"],
        "condition": instance.private["condition"],
        "arch": instance.private["arch"],
        "seed": instance.private["seed"],
        "rule": SCAN_RULE,
        "split": READOUT_SPLIT,
        "readouts_scan": json_safe(scan_readout(instance, child, batch_size, fit_ids)),
        "run": str(run_path),
        "method": run.get("method"),
        "final_patch_id": final.get("patch_id"),
        "final_config": next((c.get("config") for c in run.get("candidates", [])
                              if c.get("patch_id") == final.get("patch_id")), None),
        "final_limits": (
            run_limits(run, run_path.parent, program_limits, SCAN_BASELINE_LIMITS,
                       n_tensors=len(instance.params))[1] if rebuilt_with_limits else None
        ),
        "respects_edit_limits": run.get("flags", {}).get("respects_edit_limits"),
        "is_base_noop": bool(final.get("is_base_noop")),
        "edit_size": final.get("edit_size"),
        "n_changed": final.get("n_changed"),
        "vcheck": vcheck,
    }


FSCIL_BASELINE_LIMITS = "E9 program limits"
"""``final_limits`` label of an fscil baseline run, rebuilt with ``fscil_program_limits`` and max_ops the tensor count."""


def _readout_fscil_run(run: Mapping[str, Any], run_path: Path) -> dict[str, Any]:
    """readout_run of an fscil-v1 run (PREREG E9): ``fscil_readout`` of its final child against the instance's student
    under the session mask (batch 256, the rule's), on V_select and old_select."""
    import dataclasses

    instance = load_instance(fscil_instance_dir(str(run["instance"]["instance_id"])))
    if instance.private.get("evidence_rule") != FSCIL_RULE or instance.private.get("split_role") != "dev":
        raise DevReadoutError(f"{instance.instance_id}: not a dev {FSCIL_RULE} instance")
    check_fscil_current(instance.instance_dir)
    rule_limits = fscil_program_limits(instance.task["limits"])
    rule_limits = dataclasses.replace(rule_limits, max_ops=max(rule_limits.max_ops, len(instance.params)))
    program_limits = _program_baseline_limits(run, run_path.parent, rule_limits, "fscil", "E9")
    child = final_child(run, run_path.parent, instance, program_limits)
    final = run["final"]
    rebuilt_with_limits = not final.get("is_base_noop") and final.get("parent", "instance") == "instance"
    vcheck = {key: final.get(key) for key in ("score", "fixed", "broken", "score_new", "score_old")}
    return {
        "instance_id": instance.instance_id,
        "dataset": instance.private["dataset"],
        "arch": instance.private["arch"],
        "seed": instance.private["seed"],
        "session": instance.private["session"],
        "rule": FSCIL_RULE,
        "split": "V_select and old_select",
        "readouts_fscil": json_safe(fscil_readout(instance, child)),
        "run": str(run_path),
        "method": run.get("method"),
        "final_patch_id": final.get("patch_id"),
        "final_limits": (
            run_limits(run, run_path.parent, program_limits, FSCIL_BASELINE_LIMITS,
                       n_tensors=len(instance.params))[1] if rebuilt_with_limits else None
        ),
        "is_base_noop": bool(final.get("is_base_noop")),
        "edit_size": final.get("edit_size"),
        "n_changed": final.get("n_changed"),
        "vcheck": vcheck,
    }


RETENTION_FLOOR = -20
"""Default S_old floor of the equal-retention pick, the fewshot one: -20 of the 2000 old-class V_check rows, the -1.0
point floor (PREREG E2 supplement). An E1 joint run scores 4000 clean V_check rows, so its -1.0 point floor is -40."""

RETENTION_RULE = (
    "argmax S_joint over the checked candidates with S_old >= floor on V_check, plus the base no-op (choose_final: "
    "ties to fewer changed parameters, then the earlier query); not a run's own final rule"
)


def retention_choice(entries: Sequence[Mapping[str, Any]], s_old_floor: int = RETENTION_FLOOR) -> tuple[Candidate, int]:
    """The supplementary equal-retention pick among joint checks, and how many checks were eligible.

    ``entries`` are consumed joint checks (an oracle log: ``patch_id``, ``score`` = S_joint, ``score_new``,
    ``score_old``, ``n_changed``, ``query_index``). The pool is the checks with ``score_old >= s_old_floor`` plus the
    base no-op, which has S_joint = S_old = 0 and always qualifies, and ``choose_final`` picks from it, so a check
    with S_joint <= 0 never beats the base. This is the primary final rule on a filtered pool.
    """
    eligible = []
    for entry in entries:
        if int(entry["score"]) != int(entry["score_new"]) + int(entry["score_old"]):
            raise DevReadoutError(f"check {entry['patch_id']}: S_joint {entry['score']} != S_new + S_old")
        if int(entry["score_old"]) >= s_old_floor:
            eligible.append(Candidate(patch_id=str(entry["patch_id"]), score=int(entry["score"]),
                                      n_changed=int(entry["n_changed"]), query_index=int(entry["query_index"])))
    return choose_final(eligible), len(eligible)


def refusals(run: Mapping[str, Any]) -> dict[str, Any]:
    """Why a joint run has no check: its candidate count, how many were refused, and the refusals grouped by stage and
    reason (the run's ``invalid_stage`` and ``invalid`` per candidate). Raises unless every candidate was refused."""
    candidates = run.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DevReadoutError("a joint run without a check must list its refused candidates")
    if not all(isinstance(c, Mapping) and c.get("invalid") for c in candidates):
        valid = [c.get("patch_id") for c in candidates if not (isinstance(c, Mapping) and c.get("invalid"))]
        raise DevReadoutError(f"the oracle log is empty but candidates {valid} were not refused")
    groups: dict[tuple[str, str], int] = {}
    for c in candidates:
        key = (str(c.get("invalid_stage")), str(c["invalid"]))
        groups[key] = groups.get(key, 0) + 1
    return {"n_candidates": len(candidates), "n_refused": len(candidates),
            "refusals": [{"stage": stage, "reason": reason, "count": count}
                         for (stage, reason), count in sorted(groups.items())]}


def retention_final(run: Mapping[str, Any], s_old_floor: int = RETENTION_FLOOR) -> dict[str, Any]:
    """A joint baseline run's ``final`` replaced by its equal-retention pick (``retention_choice``) over its oracle log.

    Joint runs are the fewshot-v1 runs (PREREG E2) and the E1 joint runs. The run's own final must be ``choose_final``
    over the whole log, which checks that the log is read as the run read it. The returned final carries what
    ``final_child`` and the readouts read.
    """
    log = run.get("oracle_log")
    if not isinstance(log, list) or not all("score_old" in entry for entry in log):
        raise DevReadoutError("the equal-retention pick needs the run's joint oracle log (score_old per check)")
    refused = None
    if not log:
        # A joint run whose every candidate was refused has an empty log: its pool is the base no-op alone, as its own
        # final is. The selection states how many candidates were refused and why (PREREG E12 amendment, main ab0b935;
        # in R17 236542 the 64 MiB file cap refused every dense candidate on ResNet-18).
        if not isinstance(run.get("joint"), Mapping):
            raise DevReadoutError("the equal-retention pick needs the run's joint oracle log (score_old per check)")
        refused = refusals(run)
    primary = choose_final([Candidate(patch_id=str(e["patch_id"]), score=int(e["score"]),
                                      n_changed=int(e["n_changed"]), query_index=int(e["query_index"])) for e in log])
    if primary.patch_id != run["final"]["patch_id"] or primary.score != run["final"]["score"]:
        raise DevReadoutError(f"the run's final {run['final']['patch_id']} is not choose_final over its oracle log")
    chosen, n_eligible = retention_choice(log, s_old_floor)
    selection = {"rule": RETENTION_RULE, "s_old_floor": s_old_floor, "n_checks": len(log), "n_eligible": n_eligible,
                 "patch_id": chosen.patch_id, "query_index": None if chosen.patch_id is None else chosen.query_index,
                 "primary_patch_id": run["final"]["patch_id"]}
    if refused is not None:
        selection["no_valid_check"] = refused
    if chosen.patch_id is None:
        final = {"patch_id": None, "is_base_noop": True, "dir": None, "query_index": None, "child_sha256": None,
                 "edit_size": 0, **{key: 0 for key in ("score", "fixed", "broken", "score_new", "score_old")}}
        return {"final": final, "selection": selection}
    (entry,) = [e for e in log if str(e["patch_id"]) == chosen.patch_id]
    (candidate,) = [c for c in run["candidates"] if c.get("patch_id") == chosen.patch_id]
    final = {"patch_id": chosen.patch_id, "is_base_noop": False, "dir": chosen.patch_id,
             "query_index": chosen.query_index, "child_sha256": entry["child_sha256"],
             "edit_size": int(candidate["edit"]["edit_size"]),
             **{key: int(entry[key]) for key in ("score", "fixed", "broken", "score_new", "score_old")}}
    return {"final": final, "selection": selection}


def readout_retention(
    run_path: Path, s_old_floor: int = RETENTION_FLOOR, batch_size: int = DEFAULT_BATCH_SIZE, label: str | None = None
) -> dict[str, Any]:
    """Dev readout of a joint baseline run's equal-retention pick (``retention_final``) instead of its final.

    A fewshot-v1 run reads out as its final would (``_readout_fewshot_run``: V_select, new and old classes), an E1 joint
    run as ``readout_run`` with ``clean`` (V_select of the condition and its clean companion). Baseline runs only: a
    harness session keeps its checks in its own files, and the harness applies ``retention_choice`` to them. The
    record carries the pick as its final, its V_check S_new and S_old, a ``selection`` block naming the rule, and
    ``label`` when given (for example, that the selection is post hoc).
    """
    run = json.loads(run_path.read_text())
    if (run_path.parent / SESSION_JSON).is_file() or run.get("session_result") is not None:
        raise DevReadoutError(f"{run_path}: a harness run; apply retention_choice to the session's own checks")
    picked = retention_final(run, s_old_floor)
    swapped = {**run, "final": picked["final"]}
    if FEWSHOT_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        record = _readout_fewshot_run(swapped, run_path, batch_size)
    elif SEQSKILL_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        record = _readout_seqskill_run(swapped, run_path, batch_size)
    elif GRAPHNEW_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        record = _readout_graphnew_run(swapped, run_path)
    elif SCAN_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        record = _readout_scan_run(swapped, run_path, batch_size)
    else:
        record = _readout_repair_run(swapped, run_path, batch_size, clean=True)
        record["vcheck"].update(score_new=picked["final"]["score_new"], score_old=picked["final"]["score_old"])
    record["selection"] = picked["selection"]
    if label is not None:
        record["label"] = label
    return record


def readout_fewshot_retention(
    run_path: Path, s_old_floor: int = RETENTION_FLOOR, batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, Any]:
    """``readout_retention`` of a fewshot-v1 baseline run (the PREREG E2 supplement); other runs are refused."""
    run = json.loads(run_path.read_text())
    if not FEWSHOT_ID_RE.fullmatch(str(run["instance"]["instance_id"])):
        raise DevReadoutError(f"{run_path}: not a fewshot-v1 run; readout_retention reads E1 joint runs")
    return readout_retention(run_path, s_old_floor, batch_size)


FRONTIER_LABEL = ("descriptive frontier: every checked candidate read on V_select and its clean companion; selects "
                  "nothing (PREREG F1)")


def readout_candidates(run_path: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
    """Dev readouts of every checked candidate of one joint repair-instance baseline run, in check order.

    PREREG F1's descriptive frontier: each candidate the run's joint oracle checked is rebuilt from its patch with the
    executor under the limits the run applied (as ``final_child`` rebuilds a final), checked against the child hash its
    check recorded, and read on V_select of the condition and on its clean companion (``dev_readout_joint``'s fields),
    with the base predictions computed once. Nothing is selected; each record carries the candidate's V_check scores
    and configuration next to its readouts. Harness runs and non-joint runs are refused.
    """
    run = json.loads(run_path.read_text())
    run_dir = run_path.parent
    if (run_dir / SESSION_JSON).is_file() or run.get("session_result") is not None:
        raise DevReadoutError(f"{run_path}: a harness run; the frontier reads baseline runs only")
    instance_id = str(run["instance"]["instance_id"])
    if SCAN_ID_RE.fullmatch(instance_id):
        return _scan_candidates(run, run_path, batch_size)
    if any(pattern.fullmatch(instance_id) for pattern in (FEWSHOT_ID_RE, SEQSKILL_ID_RE, GRAPHNEW_ID_RE, FSCIL_ID_RE,
                                                          SCAN_ID_RE)):
        raise DevReadoutError(f"{run_path}: the frontier reads repair-instance joint runs only")
    log = run.get("oracle_log")
    if not isinstance(log, list) or not all("score_old" in entry for entry in log):
        raise DevReadoutError(f"{run_path}: the frontier needs a joint oracle log (score_old per check)")
    instance = load_instance(instances_root() / instance_id)
    _check_dev_instance(instance)
    program_limits, limits_label = None, "task.json"
    if str(run.get("limits_source", "")).startswith("e10_program_limits"):
        program_limits = _program_baseline_limits(
            run, run_dir, e10_program_limits(instance.task["limits"], len(instance.params),
                                                            dense_delta_bytes(v.shape for v in instance.params.values())),
            "e10", "E10")
        limits_label = E10_BASELINE_LIMITS
    limits, limits_name = run_limits(run, run_dir, program_limits, limits_label, n_tensors=len(instance.params))
    data = readout_split(instance)
    clean = load_clean_rows(instance, READOUT_SPLIT)
    clean_arrays = clean.arrays()
    base_pred = predict(instance.forward, instance.params, data["x"], batch_size)
    base_clean = predict(instance.forward, instance.params, clean_arrays["x"], batch_size)
    n_classes = len(instance.task["classes"])
    candidates = {entry.get("patch_id"): entry for entry in run.get("candidates", [])}
    records = []
    for entry in log:
        patch_id = str(entry["patch_id"])
        candidate = candidates[patch_id]
        result = apply_patch_file(instance.params, run_dir / patch_id / "patch.json", limits, instance.params_sha256)
        if result.child_sha256 != entry["child_sha256"]:
            raise DevReadoutError(f"{run_path} {patch_id}: rebuilt child {result.child_sha256} != checked "
                                  f"{entry['child_sha256']}")
        _check_child(instance, result.child)
        child_pred = predict(instance.forward, result.child, data["x"], batch_size)
        child_clean = predict(instance.forward, result.child, clean_arrays["x"], batch_size)
        records.append({
            "instance_id": instance.instance_id,
            "dataset": instance.private["dataset"],
            "condition": instance.private["condition"],
            "split": READOUT_SPLIT,
            "n": int(len(data["y"])),
            "readouts": split_readouts(data, base_pred, child_pred, None, n_classes),
            "readouts_clean": split_readouts(clean_arrays, base_clean, child_clean, None, n_classes),
            "n_clean": int(len(clean.y)),
            "clean_split_sha256": clean.clean_sha256,
            "run": str(run_path),
            "method": run.get("method"),
            "patch_id": patch_id,
            "query_index": int(entry["query_index"]),
            "config": candidate.get("config"),
            "child_sha256": result.child_sha256,
            "candidate_limits": limits_name,
            "within_program_limits": candidate.get("edit", {}).get("within_limits"),
            "vcheck": {key: entry.get(key) for key in ("score", "fixed", "broken", "score_new", "score_old")},
            "label": FRONTIER_LABEL,
        })
    return records


def _scan_candidates(run: Mapping[str, Any], run_path: Path, batch_size: int) -> list[dict[str, Any]]:
    """``readout_candidates`` of a scanskill-v1 baseline run (PREREG E13: every configuration for F1): each checked
    candidate rebuilt under the limits the run applied, checked against its recorded child hash, and read as
    ``scan_readout`` reads a final, with the base predictions computed once. Selects nothing."""
    run_dir = run_path.parent
    log = run.get("oracle_log")
    if not isinstance(log, list) or not all("score_old" in entry for entry in log):
        raise DevReadoutError(f"{run_path}: the frontier needs a joint oracle log (score_old per check)")
    instance = load_instance(scan_instance_dir(str(run["instance"]["instance_id"])))
    if instance.private.get("evidence_rule") != SCAN_RULE or instance.private.get("split_role") != "dev":
        raise DevReadoutError(f"{instance.instance_id}: not a dev {SCAN_RULE} instance")
    check_scan_current(instance.instance_dir)
    program_limits = _program_baseline_limits(
        run, run_dir, scan_program_limits(instance.task["limits"], len(instance.params)), "scan", "E13")
    limits, limits_name = run_limits(run, run_dir, program_limits, SCAN_BASELINE_LIMITS, n_tensors=len(instance.params))
    new, old = load_recorded(instance.private, "V_select"), load_recorded(instance.private, "old_select")
    base_pred = {name: predict(instance.forward, instance.params, arrays["x"], batch_size)
                 for name, arrays in (("new", new), ("old", old))}
    fit_set = (run.get("config") or {}).get("fit_set")
    fit_ids = fit_set.get("command_ids") if isinstance(fit_set, Mapping) else None
    candidates = {entry.get("patch_id"): entry for entry in run.get("candidates", [])}
    records = []
    for entry in log:
        patch_id = str(entry["patch_id"])
        candidate = candidates[patch_id]
        result = apply_patch_file(instance.params, run_dir / patch_id / "patch.json", limits, instance.params_sha256)
        if result.child_sha256 != entry["child_sha256"]:
            raise DevReadoutError(f"{run_path} {patch_id}: rebuilt child {result.child_sha256} != checked "
                                  f"{entry['child_sha256']}")
        records.append({
            "instance_id": instance.instance_id,
            "dataset": instance.private["dataset"],
            "condition": instance.private["condition"],
            "rule": SCAN_RULE,
            "split": READOUT_SPLIT,
            "readouts_scan": json_safe(scan_readout_arrays(new, old, instance.forward, instance.params, result.child,
                                                           batch_size, base_pred=base_pred, fit_ids=fit_ids)),
            "run": str(run_path),
            "method": run.get("method"),
            "patch_id": patch_id,
            "query_index": int(entry["query_index"]),
            "config": candidate.get("config"),
            "child_sha256": result.child_sha256,
            "candidate_limits": limits_name,
            "within_program_limits": candidate.get("edit", {}).get("within_limits"),
            "n_changed": entry.get("n_changed"),
            "vcheck": {key: entry.get(key) for key in ("score", "fixed", "broken", "score_new", "score_old")},
            "label": FRONTIER_LABEL,
        })
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m weightcode.evalkit.devreadout", description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="run.json files or directories to search for them")
    parser.add_argument("--out", type=Path, help="write one JSON record per run to this file")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--clean", action="store_true",
                        help="also read out on the clean companion of V_select (named conditions only)")
    parser.add_argument("--retention-floor", type=int, default=None, metavar="S_OLD",
                        help="joint baseline runs only: read out the equal-retention pick (argmax S_joint over checks "
                             "with S_old >= S_OLD, plus the base) instead of the final, E1 joint runs on both "
                             f"V_selects; the PREREG E2 supplement uses {RETENTION_FLOOR}, E1 joint runs -40")
    parser.add_argument("--label", help="a note stored in every record, e.g. that a selection is post hoc")
    parser.add_argument("--all-candidates", action="store_true",
                        help="joint repair-instance baseline runs only: read every checked candidate on V_select and "
                             "its clean companion, one record per candidate (PREREG F1's descriptive frontier)")
    args = parser.parse_args(argv)
    if args.all_candidates:
        if args.clean or args.retention_floor is not None or args.label is not None:
            parser.error("--all-candidates reads both V_selects and selects nothing; drop --clean, --retention-floor "
                         "and --label")
        frontier = [record for path in run_files(args.paths) for record in readout_candidates(path, args.batch_size)]
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in frontier))
        for r in frontier:
            if "readouts_scan" in r:
                rn = r["readouts_scan"]
                print(f"{r['instance_id']} {r['method']} {r['patch_id']} {r['config']}: V_check S {r['vcheck']['score']} "
                      f"(S_new {r['vcheck']['score_new']}, S_old {r['vcheck']['score_old']}); V_select gain "
                      f"{rn['gain_points']:+.2f} clean {rn['clean_points']:+.2f} joint {rn['joint_points']:+.2f}")
                continue
            ro, rc = r["readouts"], r["readouts_clean"]
            print(f"{r['instance_id']} {r['method']} {r['patch_id']} {r['config']}: V_check S {r['vcheck']['score']} "
                  f"(S_new {r['vcheck']['score_new']}, S_old {r['vcheck']['score_old']}); V_select delta_acc "
                  f"{ro['delta_acc']:+.4f}, clean {rc['delta_acc']:+.4f}, worst_class clean "
                  f"{rc['worst_class_delta']:+.4f}; within program limits {r['within_program_limits']}")
        return 0
    if args.retention_floor is not None:
        if args.clean:
            parser.error("--retention-floor reads E1 joint runs on the clean companion already; drop --clean")
        records = [readout_retention(path, args.retention_floor, args.batch_size, args.label)
                   for path in run_files(args.paths)]
    else:
        records = [readout_run(path, args.batch_size, args.clean) for path in run_files(args.paths)]
        if args.label is not None:
            for record in records:
                record["label"] = args.label
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records))
    for r in records:
        if "readouts_fscil" in r:
            rc = r["readouts_fscil"]
            print(f"{r['instance_id']} session {r['session']} {r['run']}: V_check S {r['vcheck']['score']} (S_new "
                  f"{r['vcheck']['score_new']}, S_old {r['vcheck']['score_old']}); V_select bal_seen "
                  f"{rc['bal_seen_child']} (delta {rc['delta_bal_seen']}), bal_new {rc['bal_new_child']}, bal_old "
                  f"{rc['bal_old_child']} (delta {rc['delta_bal_old']})")
            continue
        if "readouts_graphnew" in r:
            rg = r["readouts_graphnew"]
            picked = ""
            if "selection" in r:
                picked = (f" [kept pick {r['selection']['patch_id'] or 'base'} of {r['selection']['n_eligible']}/"
                          f"{r['selection']['n_checks']} eligible checks]")
            print(
                f"{r['instance_id']} {r['method']} {r['run']}:{picked} V_check S {r['vcheck']['score']} "
                f"(S_new {r['vcheck']['score_new']}, S_old {r['vcheck']['score_old']}); V_select all-class "
                f"{100 * rg['acc_all_child']:.2f}% ({rg['all_points']:+.2f}), new {100 * rg['acc_new_child']:.2f}%, "
                f"old {100 * rg['delta_old']:+.2f} points (n {rg['n_all']} = {rg['n_new']} new + {rg['n_old']} old)"
            )
            continue
        if "readouts_scan" in r:
            rn = r["readouts_scan"]
            picked = ""
            if "selection" in r:
                picked = (f" [kept pick {r['selection']['patch_id'] or 'base'} of {r['selection']['n_eligible']}/"
                          f"{r['selection']['n_checks']} eligible checks]")
            coverage = ""
            if "fit_coverage" in rn:
                fc = rn["fit_coverage"]
                outside = fc["not_in_fit_set"]
                coverage = (f"; fit set holds {fc['n_in_fit_set']}/{fc['n_new']} new V_select commands, EM on the "
                            f"{fc['n_not_in_fit_set']} it does not hold "
                            + (f"{outside['em_child']:.4f}" if outside["n"] else "n/a"))
            print(
                f"{r['instance_id']} {r['method']} {r['run']}:{picked} V_check S {r['vcheck']['score']} "
                f"(S_new {r['vcheck']['score_new']}, S_old {r['vcheck']['score_old']}); V_select gain "
                f"{rn['gain_points']:+.2f} clean {rn['clean_points']:+.2f} joint {rn['joint_points']:+.2f} (new EM "
                f"{rn['new']['em_child']:.4f} vs base {rn['new']['em_base']:.4f}){coverage}"
            )
            continue
        if "readouts_seqskill" in r:
            rs = r["readouts_seqskill"]
            picked = ""
            if "selection" in r:
                picked = (f" [kept pick {r['selection']['patch_id'] or 'base'} of {r['selection']['n_eligible']}/"
                          f"{r['selection']['n_checks']} eligible checks]")
            print(
                f"{r['instance_id']} {r['method']} {r['run']}:{picked} V_check S {r['vcheck']['score']} "
                f"(S_new {r['vcheck']['score_new']}, S_old {r['vcheck']['score_old']}); V_select points i "
                f"{rs['i_points']:+.2f} ii {rs['ii_points']:+.2f} iii {rs['iii_points']:+.2f} iv {rs['iv_points']:+.2f} "
                f"(child EM i {rs['i']['em_child']:.4f}, iv {rs['iv']['em_child']:.4f} vs base {rs['iv']['em_base']:.4f})"
            )
            continue
        if "readouts_fewshot" in r:
            rf = r["readouts_fewshot"]
            picked = ""
            if "selection" in r:
                picked = (f" [equal-retention pick {'base' if r['is_base_noop'] else r['vcheck'].get('score')} of "
                          f"{r['selection']['n_eligible']}/{r['selection']['n_checks']} eligible checks, S_old "
                          f"{r['vcheck']['score_old']}]")
            print(
                f"{r['instance_id']} {r['run']}:{picked} V_check S {r['vcheck']['score']}; V_select all {rf['all_points']:+.2f} "
                f"new {100 * rf['delta_new']:+.2f} old {100 * rf['delta_old']:+.2f} points (n {rf['n_all']} = "
                f"{rf['n_new']} new + {rf['n_old']} old), worst old class {100 * rf['worst_old_class_delta']:+.2f}"
            )
            continue
        ro = r["readouts"]
        extra = f" dCFR {ro['delta_cfr']:+.4f}" if "delta_cfr" in ro else ""
        if "selection" in r:
            extra += (f"; equal-retention pick {r['selection']['patch_id'] or 'base'} of {r['selection']['n_eligible']}/"
                      f"{r['selection']['n_checks']} eligible checks (S_old {r['vcheck']['score_old']})")
        if "readouts_clean" in r:
            rc = r["readouts_clean"]
            extra += (f"; clean V_select delta_acc {rc['delta_acc']:+.4f} (fixed {rc['fixed']}, broken {rc['broken']}) "
                      f"worst_class {rc['worst_class_delta']:+.4f}")
        print(
            f"{r['instance_id']} {r['run']}: V_check S {r['vcheck']['score']}; V_select n {r['n']} "
            f"delta_acc {ro['delta_acc']:+.4f} (fixed {ro['fixed']}, broken {ro['broken']}) "
            f"worst_class {ro['worst_class_delta']:+.4f}{extra}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
