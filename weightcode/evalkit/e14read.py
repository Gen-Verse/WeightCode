"""Readouts of PREREG E14 reference runs on V_select, and the E14 bar.

An E14 reference run (``weightcode.baselines.fit.e14refs``, a run.json under ``$WEIGHTCODE_RUNS/baselines/<instance>/<method>/
e14/<seed>/``) checked 8 candidates with the rollout joint oracle on V_check. ``read_run`` rebuilds every checked
candidate from its patch with the executor, under the limits the run applied (``devreadout.run_limits``: the E14
program limits for a constrained method, those limits without the size limits for one that ignores them), checks each
rebuilt child against the hash its check recorded, and reads it with ``weightcode.evalkit.e14.e14_readout`` (greedy rollouts
on the 1,000 new and 1,000 old V_select episodes, child against base). The three readouts of the PREREG follow from the
candidates without a second rollout:

- ``own``: the run's own final (argmax S_joint over its checks plus the base no-op, ``choose_final``);
- ``kept``: the kept pick under joint-retain-v1 at S_old >= -3 of the 500 old V_check episodes
  (``devreadout.retention_final``: the same rule on the checks with S_old >= -3, plus the base no-op);
- ``candidates``: every configuration (PREREG F1's descriptive frontier), selecting nothing.

A pick of the base no-op reads as the base: gain, clean and joint 0.

``e14_bar`` takes the records of the 13 references at equal information (behaviour cloning in E5's five shapes and
F1's eight) and returns the bar of B (the best kept joint plus 1), the best kept references, the PREREG E14 addition's
supplementary test (a kept behaviour-cloning reference at >= 90% success on new V_select), and the label-equivalent
curve beside the bar. Nothing here reads the sealed store.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weightcode.evalkit import devreadout as R
from weightcode.evalkit import e14 as E
from weightcode.patch import Limits, apply_patch_file, task_limits
from weightcode.patch.limits import dense_delta_bytes

S_OLD_FLOOR = -3
"""PREREG E14 kept rule joint-retain-v1: S_old >= -3 of the 500 old V_check episodes."""
SUPPLEMENTARY_SUCCESS = 0.90
"""PREREG E14 addition (main d744142): a kept behaviour-cloning reference at this new V_select success or more makes
E14 supplementary, and its session is not run."""
BAR_MARGIN = 1.0
E14_LIMITS_LABEL = "E14 program limits"
READ_SCHEMA = "weightcode.e14.reference_readout.v1"


class E14ReadError(RuntimeError):
    """An E14 run cannot be read as it was checked."""


def program_limits(instance: Any) -> Limits:
    """The E14 program limits of ``instance`` (``e14_program_limits`` with its tensor count and dense file bytes)."""
    return E.e14_program_limits(instance.task["limits"], len(instance.params),
                                dense_delta_bytes(v.shape for v in instance.params.values()))


SPLIT_KEYS = ("demonstrated", "not_demonstrated")
"""New V_select episodes by whether their (color, type) occurs among the 8 demonstrations (main d744142)."""


def demonstrated_split(new_readout: Mapping[str, Any], which: str = "success_child") -> dict[str, Any]:
    """New V_select success split by demonstrated (color, type), from a readout's ``new`` part (``which``:
    ``success_child`` or ``success_base``)."""
    split = new_readout["by_demonstrated_descriptor"]
    return {key: {"n": int(split[key]["n"]), "success": split[key][which]} for key in SPLIT_KEYS}


def _base_pick(readout_base: Mapping[str, Any], base_split: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"patch_id": None, "is_base_noop": True, "config": None, "gain": 0.0, "clean": 0.0, "joint": 0.0,
            "success_new": readout_base["new"], "success_old": readout_base["old"],
            "success_new_split": None if base_split is None else dict(base_split),
            "vcheck": {"score": 0, "score_new": 0, "score_old": 0}}


def _pick(candidates: Sequence[Mapping[str, Any]], patch_id: str | None, base: Mapping[str, Any]) -> dict[str, Any]:
    if patch_id is None:  # the base's split from any candidate's readout (each holds the base's episodes)
        base_split = demonstrated_split(candidates[0]["readout"]["new"], "success_base") if candidates else None
        return _base_pick(base, base_split)
    hits = [c for c in candidates if c["patch_id"] == patch_id]
    if len(hits) != 1:
        raise E14ReadError(f"pick {patch_id} is not one checked candidate")
    c = hits[0]
    ro = c["readout"]
    return {"patch_id": patch_id, "is_base_noop": False, "config": c["config"], "gain": ro["gain"],
            "clean": ro["clean"], "joint": ro["joint"], "success_new": ro["new"]["success_child"],
            "success_old": ro["old"]["success_child"], "success_new_split": demonstrated_split(ro["new"]),
            "vcheck": dict(c["vcheck"])}


def read_run(run_path: str | Path, s_old_floor: int = S_OLD_FLOOR) -> dict[str, Any]:
    """Every checked candidate of one E14 reference run read on V_select, with its own final and its kept pick."""
    run_path = Path(run_path)
    run = json.loads(run_path.read_text())
    run_dir = run_path.parent
    if (run_dir / R.SESSION_JSON).is_file() or run.get("session_result") is not None:
        raise E14ReadError(f"{run_path}: a harness run; read a session from its own records")
    instance_id = str(run["instance"]["instance_id"])
    if not E.INSTANCE_ID_RE.fullmatch(instance_id):
        raise E14ReadError(f"{run_path}: {instance_id} is not an E14 instance id")
    instance = E.load_e14_instance(instance_id)
    if instance.private.get("split_role") != "dev":
        raise E14ReadError(f"{instance_id}: not a dev instance")
    E.check_e14_current(instance.instance_dir)
    if run.get("e14", {}).get("rule") != E.POLICYSKILL_RULE:
        raise E14ReadError(f"{run_path}: not an E14 run (record e14.rule)")
    expected = program_limits(instance)
    if run.get("limits") != task_limits(expected):
        raise E14ReadError(f"{run_path}: records limits {run.get('limits')!r}, not the E14 program limits "
                           f"{task_limits(expected)}")
    limits, limits_name = R.run_limits(run, run_dir, expected, E14_LIMITS_LABEL, n_tensors=len(instance.params))
    log = run.get("oracle_log")
    if not isinstance(log, list) or not all("score_old" in entry for entry in log):
        raise E14ReadError(f"{run_path}: no joint oracle log")
    by_id = {c.get("patch_id"): c for c in run.get("candidates", [])}
    candidates = []
    for entry in log:
        patch_id = str(entry["patch_id"])
        result = apply_patch_file(instance.params, run_dir / patch_id / "patch.json", limits, instance.params_sha256)
        if result.child_sha256 != entry["child_sha256"]:
            raise E14ReadError(f"{run_path} {patch_id}: rebuilt child {result.child_sha256} != checked "
                               f"{entry['child_sha256']}")
        candidates.append({
            "patch_id": patch_id,
            "query_index": int(entry["query_index"]),
            "config": by_id[patch_id].get("config"),
            "child_sha256": result.child_sha256,
            "n_changed": int(entry["n_changed"]),
            "within_program_limits": by_id[patch_id].get("edit", {}).get("within_limits"),
            "vcheck": {key: int(entry[key]) for key in ("score", "score_new", "score_old", "fixed_new", "broken_new",
                                                        "fixed_old", "broken_old")},
            "readout": E.e14_readout(instance, result.child),
        })
    base_vselect = {"new": float(instance.private["base_vselect_success_new"]),
                    "old": float(instance.private["base_vselect_success_old"])}
    kept = R.retention_final(run, s_old_floor)
    own_id = run["final"]["patch_id"]
    return {
        "schema": READ_SCHEMA,
        "instance_id": instance_id,
        "model_id": instance.private["model_id"],
        "method": run.get("method"),
        "run": str(run_path),
        "split": "V_select",
        "candidate_limits": limits_name,
        "base_success": base_vselect,
        "own": _pick(candidates, own_id, base_vselect),
        "kept": {**_pick(candidates, kept["final"]["patch_id"], base_vselect), "selection": kept["selection"]},
        "s_old_floor": s_old_floor,
        "candidates": candidates,
    }


PICK_KEYS = ("gain", "clean", "joint", "success_new", "success_old", "success_new_split", "patch_id", "config")


def wrapper_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """The wrapper record (``run_e14_refs.py wrapper``: PREREG E14's hand-written wrapper, beside the bar) in the bar's
    terms: gain, clean, joint, new and old V_select success, and new success split by demonstrated (color, type)."""
    ro = record["readout"]
    return {"label": record.get("label"), "instance_id": record.get("instance_id"),
            "gain": ro["gain"], "clean": ro["clean"], "joint": ro["joint"],
            "success_new": ro["new"]["success_child"], "success_old": ro["old"]["success_child"],
            "success_new_split": demonstrated_split(ro["new"]),
            "base_success_new_split": demonstrated_split(ro["new"], "success_base")}


def e14_bar(records: Mapping[str, Mapping[str, Any]], bc_methods: Sequence[str], f1_methods: Sequence[str],
            labeleq_methods: Sequence[str] = (), wrapper: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The bar of PREREG E14 B from ``read_run`` records by method: the best kept joint over the behaviour-cloning
    shapes and F1's eight, plus 1; the supplementary test on the kept references; the label-equivalent curve
    (``labeleq_methods``, beside the bar, with ``bc_full`` as k = 8); and, given the wrapper record of the same
    instance, the wrapper beside the bar (``wrapper_summary``). Every pick carries its demonstrated split."""
    in_bar = [*bc_methods, *f1_methods]
    missing = [m for m in in_bar if m not in records]
    if missing:
        raise E14ReadError(f"the bar needs every reference at equal information; missing {missing}")
    instances = {str(record["instance_id"]) for record in records.values()}
    if len(instances) != 1:
        raise E14ReadError(f"the reads are of several instances {sorted(instances)}")
    (instance_id,) = instances
    beside = None
    if wrapper is not None:
        if str(wrapper.get("instance_id")) != instance_id:
            raise E14ReadError(f"the wrapper record is of {wrapper.get('instance_id')!r}, not {instance_id}")
        beside = wrapper_summary(wrapper)
    kept = {m: records[m]["kept"] for m in in_bar}
    best = max(in_bar, key=lambda m: (kept[m]["joint"], -in_bar.index(m)))
    best_bc = max(bc_methods, key=lambda m: (kept[m]["success_new"], -in_bar.index(m)))
    best_all = max(in_bar, key=lambda m: (kept[m]["success_new"], -in_bar.index(m)))
    curve = []
    for m in ("bc_full", *labeleq_methods):
        if m not in records:
            continue
        k = 8 if m == "bc_full" else int(m.removeprefix("labeleq_k"))
        curve.append({"k": k, "method": m, "kept": {key: records[m]["kept"][key] for key in PICK_KEYS},
                      "own": {key: records[m]["own"][key] for key in PICK_KEYS}})
    return {
        "instance_id": instance_id,
        "rule": f"bar = best kept joint over {len(in_bar)} references + {BAR_MARGIN} (kept: S_old >= "
                f"{records[best]['s_old_floor']} of 500 old V_check episodes)",
        "best_kept_reference": best,
        "best_kept_joint": kept[best]["joint"],
        "bar_joint": kept[best]["joint"] + BAR_MARGIN,
        "kept": {m: {key: kept[m][key] for key in PICK_KEYS} for m in in_bar},
        "supplementary": {
            "threshold_success_new": SUPPLEMENTARY_SUCCESS,
            "best_kept_bc_shape": best_bc, "best_kept_bc_shape_success_new": kept[best_bc]["success_new"],
            "best_kept_any": best_all, "best_kept_any_success_new": kept[best_all]["success_new"],
            "met_by_bc_shapes": kept[best_bc]["success_new"] >= SUPPLEMENTARY_SUCCESS,
            "met_by_any_reference": kept[best_all]["success_new"] >= SUPPLEMENTARY_SUCCESS,
        },
        "labeleq_curve": curve,
        "wrapper": beside,
    }


__all__ = ["BAR_MARGIN", "E14ReadError", "PICK_KEYS", "READ_SCHEMA", "SPLIT_KEYS", "S_OLD_FLOOR",
           "SUPPLEMENTARY_SUCCESS", "demonstrated_split", "e14_bar", "program_limits", "read_run", "wrapper_summary"]
