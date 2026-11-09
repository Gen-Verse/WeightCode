"""PREREG E13 "T11 SCAN controls and M04" (docs/PREREG_v0_DRAFT.md): diagnostics beside the bar on the E13 instances
ncc5c260f (addprim_jump) and n8826c49e (around_right), their LEARN finals and their references of record. No LLM call,
no NSCC job, no sealed read; V_select and old V_select of each condition only.

Subcommands, run in this order:

    programs [--conditions C ...]
        Rerun each final program, unmodified, in the LEARN program sandbox at the session's own sandbox settings (from
        its final/program_run.json): first on the instance's own student and experience, which must reproduce
        final/deltas.npz and the final child bit for bit (otherwise nothing else runs for that condition); then on the
        relabelled student (``scan_t11.default_relabel``: word ids and action ids permuted, the layout and the
        experience rewritten), with its deltas moved back to the original ids and compared with the final's. Records
        $WEIGHTCODE_RUNS/t11scan/reruns/<condition>_<arm>.json, children <condition>_<arm>/child.npz.
    arms --condition C
        One readout per arm of the condition (``ARMS``) on new and old V_select: exact match, the command-structure
        groups, per-row correctness, and the arm's own checks. $WEIGHTCODE_RUNS/devreadouts/t11_e13_<arm>_<instance>.jsonl
        (one line each, never overwritten).
    mech
        M04 on addprim_jump: the four networks (base, E only, H only, E+H), traced forward parity, the identity checks,
        slot-level outputs and margins, and the hidden states at the decision slots against the base's states for the
        same commands with jump renamed to walk, look and run. $WEIGHTCODE_RUNS/t11scan/m04_<instance>.json.
    summary
        $WEIGHTCODE_RUNS/t11scan/e13_t11_summary.json and e13_t11_tables.md, from the records above, with the readings fixed
        in the PREREG entry.

    bash -c 'source scripts/env.sh; OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python -B scripts/e13/t11_scan.py arms \\
        --condition addprim_jump'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_RERUN = "weightcode.t11.e13_program_rerun.v1"
SCHEMA_ARM = "weightcode.t11.e13_arm_readout.v1"
SCHEMA_M04 = "weightcode.t11.e13_m04.v1"
SCHEMA_SUMMARY = "weightcode.t11.e13_summary.v1"
BATCH = 1024
JOINT_TOLERANCE = 1.0
"""Points: an arm 'matches' the final when its joint is at least the final's minus this (PREREG entry)."""
SHARE_WLR_AT_JUMP = 0.9
"""M04: E only is read as 'the structure runs, the action is missing' when at least this share of its I_JUMP target
slots is predicted as I_WALK, I_LOOK or I_RUN."""

CONDITIONS: Mapping[str, Mapping[str, Any]] = {
    "addprim_jump": {
        "instance": "ncc5c260f", "session": "s9be29c2c302e4daa", "final_program": "centroid_meanrow_floor.py",
        "final_query": 1,
        "refs": {"rulegen_mixed_grad_full": "q2", "rulegen_mixed_sparse19_sal": "q3",
                 "rulegen_mixed_sparse19_traj": "q2", "ft_full_noreplay": "q7"},
        "refs_record": "e13_refs_ncc5c260f_addprim_jump_floor5_dev_nscc_20261106.jsonl",
        "walk_record": "e13_walk_ncc5c260f_addprim_jump_dev_nscc_20261106.json",
    },
    "around_right": {
        "instance": "n8826c49e", "session": "s41876e82b0843c53", "final_program": "around_right_v3.py",
        "final_query": 2,
        "refs": {"rulegen_sparse19_traj": "q1", "rulegen_mixed_grad_full": "q2", "rulegen_mixed_sparse19_sal": "q2",
                 "rulegen_mixed_sparse19_traj": "q1", "seq_grad_emb_head": "q0"},
        "refs_record": "e13_refs_n8826c49e_around_right_floor5_dev_nscc_20261106.jsonl",
        "walk_record": None,
    },
}
"""The finals (kept query of each session) and the kept picks of record (joint-retain-v1, S_old >= -5, the floor5
records) that are not the base; the last reference of each list is the best kept one at equal information."""
RULEGEN_PREFIX = "rulegen"
S_OLD_FLOOR = -5
"""joint-retain-v1 of PREREG E13: the kept pick of a reference is its best check with S_old >= -5 of 1000."""
ARMS: Mapping[str, tuple[str, ...]] = {
    "addprim_jump": ("base", "final", "E_only", "H_only", *(f"ref_{m}" for m in CONDITIONS["addprim_jump"]["refs"]),
                     "walk_row", "rename_auto", "rename_walk", "rename_look", "rename_run", "relabelled_program"),
    "around_right": ("base", "final", *(f"ref_{m}" for m in CONDITIONS["around_right"]["refs"]), "swap_lr",
                     "relabelled_program"),
}
ARM_DEFINITIONS: Mapping[str, str] = {
    "base": "the instance's base student",
    "final": "the session's kept final (final/model.npz)",
    "E_only": "the base with the final's tok_emb.weight (the jump row is its only change)",
    "H_only": "the base with the final's head.weight and head.bias (the I_JUMP row and bias are their only change)",
    "walk_row": "PREREG E13's fixed walk diagnostic: jump's embedding row set to walk's, new part read with I_JUMP as "
                "I_WALK (not a system: it cannot emit I_JUMP)",
    "rename_auto": "external renaming on the base: jump becomes the first of walk, look, run the command does not hold, "
                   "and that primitive's action becomes I_JUMP in the output (consistent on every new row)",
    "rename_walk": "external renaming with p = walk; consistent on the rows without walk, which are the reading",
    "rename_look": "external renaming with p = look; consistent on the rows without look, which are the reading",
    "rename_run": "external renaming with p = run; consistent on the rows without run, which are the reading",
    "swap_lr": "external renaming on the base: on rows holding 'around right', left and right swap in the command and "
               "I_TURN_LEFT and I_TURN_RIGHT swap in the output (consistent on every row; in the base's training "
               "distribution only without 'around left')",
    "relabelled_program": "the unmodified final program rerun on the relabelled student (word and action ids permuted), "
                          "read on the relabelled rows against the relabelled base",
}


class T11ScanError(RuntimeError):
    """A check of the T11 SCAN entry failed; nothing is overwritten."""


# --------------------------------------------------------------------------------------------------------------------
# Paths and records


def runs() -> Path:
    from weightcode.common.paths import runs_dir

    return runs_dir()


def t11_dir() -> Path:
    return runs() / "t11scan"


def session_dir(session: str) -> Path:
    return runs() / "sessions" / session


def arm_record_path(arm: str, instance: str) -> Path:
    return runs() / "devreadouts" / f"t11_e13_{arm}_{instance}.jsonl"


def rerun_record_path(condition: str, arm: str) -> Path:
    return t11_dir() / "reruns" / f"{condition}_{arm}.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    from weightcode.evalkit.targets import jsonable

    return jsonable(value)


def write_new_json(path: Path, payload: Any) -> str:
    """Write ``payload`` once (refusing an existing file); returns the file's sha256."""
    if path.exists():
        raise T11ScanError(f"{path} exists; records are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def write_new_line(path: Path, payload: Any) -> str:
    if path.exists():
        raise T11ScanError(f"{path} exists; records are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def read_line(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise T11ScanError(f"{path}: expected one record, found {len(lines)} lines")
    return json.loads(lines[0])


def code_record() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True, check=True)
    return {"git_head": head.stdout.strip(), "git_dirty": bool(status.stdout.strip()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "scan_t11_py_sha256": sha256_file(root / "weightcode" / "evalkit" / "scan_t11.py"),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "blas_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                            "MKL_NUM_THREADS")}}


def utc() -> str:
    from weightcode.evalkit.targets import utc_now

    return utc_now()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def deltas_equal(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Keys, dtypes, shapes and bytes of two delta sets, with the largest absolute difference per tensor."""
    from weightcode.evalkit.scan_t11 import bytes_equal, max_abs_difference

    per: dict[str, Any] = {}
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            per[key] = {"equal": False, "missing": True}
            continue
        same_shape = np.asarray(a[key]).shape == np.asarray(b[key]).shape
        per[key] = {"equal": bytes_equal(a[key], b[key]),
                    "max_abs_difference": max_abs_difference(a[key], b[key]) if same_shape else None}
    return {"equal": sorted(a) == sorted(b) and all(v["equal"] for v in per.values()), "per_tensor": per}


# --------------------------------------------------------------------------------------------------------------------
# Instances, finals, references


def instance_of(condition: str) -> Any:
    from weightcode.evalkit.scanskill import check_scan_current, load_scan_instance

    instance = load_scan_instance(str(CONDITIONS[condition]["instance"]))
    if instance.private.get("condition") != condition or instance.private.get("split_role") != "dev":
        raise T11ScanError(f"{instance.instance_id} is not the dev instance of {condition}")
    check_scan_current(instance.instance_dir)
    return instance


def session_info(session: str) -> dict[str, Any]:
    return json.loads((runs() / "devreadouts" / "learn" / session / "info.json").read_text(encoding="utf-8"))


def final_child(condition: str, instance: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The session's kept final, checked against info.json (child hash, kept query, program)."""
    from weightcode.common.identity import load_params, params_sha256

    spec = CONDITIONS[condition]
    info = session_info(str(spec["session"]))
    final = info["final"]
    if final["query_index"] != spec["final_query"] or final["patch_id"] != spec["final_program"]:
        raise T11ScanError(f"{spec['session']}: the kept final is {final['patch_id']} (query {final['query_index']})")
    child = load_params(session_dir(str(spec["session"])) / "final" / "model.npz")
    digest = params_sha256(child)
    if digest != final["child_sha256"]:
        raise T11ScanError(f"{spec['session']}: final/model.npz hashes to {digest}, info.json {final['child_sha256']}")
    if sorted(child) != sorted(instance.params):
        raise T11ScanError(f"{spec['session']}: the final's tensors differ from the base's")
    return child, {"child_sha256": digest, "trajectory_row": info["trajectory"][spec["final_query"]]}


def refs_records(condition: str) -> dict[str, dict[str, Any]]:
    path = runs() / "devreadouts" / str(CONDITIONS[condition]["refs_record"])
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        out[record["method"]] = record
    return out


def reference_child(condition: str, method: str, instance: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The kept pick of record of one reference, rebuilt from its run as its floor5 record was (``retention_final``
    at S_old >= -5, ``final_child`` under the E13 program limits), checked against the record."""
    from weightcode.evalkit import devreadout as DR
    from weightcode.evalkit.scanskill import scan_program_limits

    record = refs_records(condition)[method]
    expected = CONDITIONS[condition]["refs"][method]
    if record.get("final_patch_id") != expected or record.get("is_base_noop"):
        raise T11ScanError(f"{condition} {method}: the floor5 record keeps {record.get('final_patch_id')}, "
                           f"PREREG names {expected}")
    run_path = Path(record["run"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    picked = DR.retention_final(run, S_OLD_FLOOR)
    if picked["selection"]["s_old_floor"] != S_OLD_FLOOR or picked["final"]["patch_id"] != expected:
        raise T11ScanError(f"{condition} {method}: retention_final picks {picked['final']['patch_id']}")
    swapped = {**run, "final": picked["final"]}
    limits = DR._program_baseline_limits(run, run_path.parent,
                                         scan_program_limits(instance.task["limits"], len(instance.params)), "scan",
                                         "E13")
    child = DR.final_child(swapped, run_path.parent, instance, limits)
    fit_set = (run.get("config") or {}).get("fit_set")
    fit_ids = fit_set.get("command_ids") if isinstance(fit_set, Mapping) else None
    return child, {"run": str(run_path), "patch_id": expected, "child_sha256": picked["final"]["child_sha256"],
                   "config": record.get("final_config"), "fit_ids": fit_ids, "record": record}


def vselect(instance: Any) -> dict[str, dict[str, np.ndarray]]:
    from weightcode.evalkit.scanskill import load_recorded

    return {"new": load_recorded(instance.private, "V_select"), "old": load_recorded(instance.private, "old_select")}


def predictor(instance: Any, params: Mapping[str, np.ndarray]) -> Any:
    from weightcode.evalkit.patch_check import predict

    return lambda x: predict(instance.forward, params, np.ascontiguousarray(x), BATCH).astype(np.int64)


# --------------------------------------------------------------------------------------------------------------------
# Program reruns


def sandbox_of(session: str) -> tuple[Any, dict[str, Any]]:
    """The session's program sandbox, from its final/program_run.json."""
    from weightcode.sandbox.program import ProgramSandbox, driver_sha256

    recorded = json.loads((session_dir(session) / "final" / "program_run.json").read_text(encoding="utf-8"))["sandbox"]
    sandbox = ProgramSandbox(python=Path(recorded["python"]), memory_bytes=int(recorded["memory_bytes"]),
                             threads=int(recorded["threads"]), tmpfs_bytes=int(recorded["tmpfs_bytes"]),
                             bwrap=Path(recorded["bwrap"]), max_output_chars=int(recorded["max_output_chars"]))
    return sandbox, {"recorded": recorded, "driver_sha256_now": driver_sha256(),
                     "driver_equal_session": driver_sha256() == recorded["driver_sha256"]}


def relabelled_student(instance: Any, dest: Path, relabel: Any) -> tuple[Any, Path, dict[str, Any]]:
    """The instance's student and experience under ``relabel``, staged as a transfer run stages a student."""
    import shutil

    from weightcode.common.identity import params_sha256, save_params
    from weightcode.evalkit import scan_t11 as T
    from weightcode.harness import seqskill
    from weightcode.harness.program import Student

    dest.mkdir(mode=0o700, parents=True)
    spec = json.loads((instance.instance_dir / "model_spec.json").read_text(encoding="utf-8"))
    layout = T.relabel_layout({k: v for k, v in spec.items() if k != "model_id"}, relabel)
    params = T.relabel_params(instance.params, relabel)
    digest = save_params(dest / "model.npz", params)
    if digest != params_sha256(params):
        raise T11ScanError("the relabelled weights do not hash to themselves")
    shutil.copyfile(instance.instance_dir / "forward.py", dest / "forward.py")
    (dest / "model_spec.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
    arrays, _ = seqskill.experience(instance)
    moved = {}
    for key, value in arrays.items():
        if key.startswith("x_"):
            moved[key] = T.relabel_tokens(value, relabel)
        elif key.startswith(("y_", "pred_")):
            moved[key] = T.relabel_classes(value, relabel)
        else:
            moved[key] = value
    np.savez(dest / "experience.npz", **moved)
    student = Student(params=params, params_file=dest / "model.npz", forward_file=dest / "forward.py",
                      input_files={}, layout=layout)
    record = {"params_sha256": digest, "layout_sha256": sha256_file(dest / "model_spec.json"),
              "experience_sha256": sha256_file(dest / "experience.npz"), "relabel": relabel.as_dict()}
    return student, dest / "experience.npz", record


def isomorphism_check(instance: Any, relabel: Any) -> dict[str, Any]:
    """The relabelled base on the relabelled evidence rows against the base on the evidence rows: logits column for
    column under the class map, and argmax predictions (the evidence rows only; no V_select row is read here)."""
    from weightcode.evalkit import scan_t11 as T

    rows = np.concatenate([instance.evidence[k] for k in ("x_prim", "x_old") if k in instance.evidence])
    base = np.asarray(instance.forward(instance.params, rows))
    moved = np.asarray(instance.forward(T.relabel_params(instance.params, relabel), T.relabel_tokens(rows, relabel)))
    back = moved[..., np.asarray(relabel.class_map)]
    return {"n_rows": int(len(rows)), "logits_bytes_equal": T.bytes_equal(back, base),
            "max_abs_logit_difference": T.max_abs_difference(back, base),
            "argmax_equal": bool(np.array_equal(back.argmax(-1), base.argmax(-1)))}


def apply_deltas(params: Mapping[str, np.ndarray], deltas_update: Any, work: Path, limits: Any
                 ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Write the derived patch of a program run and apply it to ``params`` under ``limits``."""
    from weightcode.common.identity import params_sha256
    from weightcode.harness.program import write_derived_patch
    from weightcode.patch import apply_patch

    patch_dir = work / "patch"
    patch_dir.mkdir()
    parent = params_sha256(params)
    patch = write_derived_patch(deltas_update, params, parent, patch_dir)
    result = apply_patch(params, patch, patch_dir, limits, parent)
    return result.child, {"patch_sha256": sha256_file(patch_dir / "derived_patch.json"),
                          "n_changed": int(result.n_changed), "changed_frac": float(result.changed_frac),
                          "child_sha256": result.child_sha256}


def cmd_programs(args: argparse.Namespace) -> int:
    from weightcode.common.identity import save_params
    from weightcode.evalkit import scan_t11 as T
    from weightcode.harness.program import run_update
    from weightcode.harness.transfer import stage_student
    from weightcode.patch import Limits

    for condition in args.conditions:
        spec = CONDITIONS[condition]
        session = str(spec["session"])
        instance = instance_of(condition)
        final_dir = session_dir(session) / "final"
        final_run = json.loads((final_dir / "program_run.json").read_text(encoding="utf-8"))
        final_deltas = load_npz(final_dir / "deltas.npz")
        _, final_meta = final_child(condition, instance)
        limits = Limits(**json.loads((session_dir(session) / "session.json").read_text(encoding="utf-8"))["limits"])
        program = final_dir / str(spec["final_program"])
        if sha256_file(program) != final_run["program_sha256"]:
            raise T11ScanError(f"{program} does not hash to the session's final program")
        sandbox, sandbox_record = sandbox_of(session)
        relabel = T.default_relabel()
        for arm in ("program", "program_relabelled"):
            path = rerun_record_path(condition, arm)
            work = t11_dir() / "reruns" / f"{condition}_{arm}"
            if path.exists() or work.exists():
                raise T11ScanError(f"{path} or {work} exists; a rerun is never repeated")
            work.mkdir(parents=True, mode=0o700)
            if arm == "program":
                student, experience_file, experience = stage_student(instance, work / "student")
                staged = {"experience": experience}
            else:
                student, experience_file, staged = relabelled_student(instance, work / "student", relabel)
                staged["isomorphism_on_evidence_rows"] = isomorphism_check(instance, relabel)
            started = time.perf_counter()
            update = run_update(student, experience_file, program, work / "run", sandbox)
            wall = time.perf_counter() - started
            child, applied = apply_deltas(student.params, update, work, limits)
            child_sha = save_params(work / "child.npz", child)
            checks: dict[str, Any] = {}
            stdout = update.record["run"]["stdout"]
            checks["stdout_equal_session"] = stdout == final_run["run"]["stdout"]
            if arm == "program":
                checks["deltas_vs_final"] = deltas_equal(update.deltas, final_deltas)
                checks["deltas_file_sha256_equal"] = update.record["deltas_sha256"] == final_run["deltas_sha256"]
                checks["child_equals_final"] = child_sha == final_meta["child_sha256"]
                checks["inputs_equal_session"] = {name: update.record["inputs_sha256"].get(name) == digest
                                                  for name, digest in final_run["inputs_sha256"].items()}
            else:
                back = T.unrelabel_deltas(update.deltas, relabel)
                checks["deltas_moved_back_vs_final"] = deltas_equal(back, final_deltas)
                final_params, _ = final_child(condition, instance)
                relabelled_final = T.relabel_params(final_params, relabel)
                checks["child_equals_relabelled_final"] = all(
                    T.bytes_equal(child[k], relabelled_final[k]) for k in child)
                checks["child_max_abs_difference_to_relabelled_final"] = max(
                    T.max_abs_difference(child[k], relabelled_final[k]) for k in child)
            record = {"schema": SCHEMA_RERUN, "condition": condition, "arm": arm, "instance_id": instance.instance_id,
                      "session": session, "created_at": utc(),
                      "program": {"path": str(program), "sha256": sha256_file(program)},
                      "staged": staged, "sandbox": sandbox_record, "limits": dataclass_dict(limits),
                      "inputs_sha256": update.record["inputs_sha256"], "deltas_sha256": update.record["deltas_sha256"],
                      "deltas_path": str(update.deltas_path), "tensors": list(update.tensors),
                      "stdout": stdout, "stderr": update.record["run"]["stderr"],
                      "duration_s": update.record["run"]["duration_s"], "run_wall_s": wall, **applied,
                      "child_path": str(work / "child.npz"), "child_sha256_saved": child_sha, "checks": checks,
                      "code": code_record()}
            digest = write_new_json(path, record)
            print(f"{condition} {arm}: n_changed {applied['n_changed']} child {child_sha[:12]} run "
                  f"{record['duration_s']:.1f} s -> {path} ({digest[:12]})", flush=True)
            if arm == "program" and not (checks["deltas_vs_final"]["equal"] and checks["child_equals_final"]):
                raise T11ScanError(f"{condition}: the rerun of the unmodified final does not reproduce it bit for "
                                   "bit; T11 stops for this condition")
    return 0


def dataclass_dict(value: Any) -> dict[str, Any]:
    import dataclasses

    return dataclasses.asdict(value)


# --------------------------------------------------------------------------------------------------------------------
# Arm readouts


def rows_right_list(y: np.ndarray, pred: np.ndarray) -> list[int]:
    from weightcode.evalkit.metrics import rows_right

    return rows_right(y, pred).astype(int).tolist()


def part_block(commands: Sequence[str], condition: str, part: str, y: np.ndarray, base: np.ndarray,
               child: np.ndarray) -> dict[str, Any]:
    from weightcode.evalkit import scan_t11 as T

    return {"all": T.part_readout(y, base, child), "groups": T.grouped_readout(commands, condition, part, y, base, child),
            "right": rows_right_list(y, child), "pred_sha256": T.predictions_sha256(child)}


def points(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, float]:
    gain, clean = 100.0 * new["delta"], 100.0 * old["delta"]
    return {"gain_points": gain, "clean_points": clean, "joint_points": gain + clean}


def same_summary(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    keys = ("n", "em_base", "em_child", "delta", "fixed", "broken", "token_acc_base", "token_acc_child")
    return all(a.get(k) == b.get(k) for k in keys)


def arm_readout(condition: str, arm: str, instance: Any, data: Mapping[str, Mapping[str, np.ndarray]],
                commands: Mapping[str, list[str]], base_pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    from weightcode.common.identity import load_params, params_sha256
    from weightcode.data import scan as S
    from weightcode.evalkit import scan_t11 as T

    y = {part: np.asarray(data[part]["y"]) for part in ("new", "old")}
    record: dict[str, Any] = {"schema": SCHEMA_ARM, "condition": condition, "arm": arm,
                              "definition": ARM_DEFINITIONS.get(arm, ARM_DEFINITIONS.get(arm.split("_", 1)[0], arm)),
                              "instance_id": instance.instance_id, "created_at": utc(), "checks": {}}
    child: dict[str, np.ndarray] | None = None
    preds: dict[str, np.ndarray] = {}
    targets = dict(y)
    if arm == "base":
        child = dict(instance.params)
    elif arm == "final" or arm in ("E_only", "H_only"):
        final, meta = final_child(condition, instance)
        if arm == "final":
            child = final
            record["checks"]["child_sha256_equals_info"] = True
        else:
            children, comp = T.component_children(instance.params, final)
            child = children[arm]
            record["components"] = comp
            record["checks"]["EH_equals_final_bit_for_bit"] = params_sha256(children["EH"]) == meta["child_sha256"]
    elif arm.startswith("ref_"):
        method = arm[len("ref_"):]
        child, meta = reference_child(condition, method, instance)
        record["definition"] = (f"kept pick of record of {method} ({meta['patch_id']}, config {meta['config']}), "
                                "rebuilt from its run")
        record["reference"] = {k: v for k, v in meta.items() if k not in ("record", "fit_ids")}
        if meta["fit_ids"] is not None:
            in_fit = np.isin(np.asarray(data["new"]["idx"]), np.asarray(meta["fit_ids"], dtype=np.int64))
            record["fit_set_rows"] = {"n_in_fit_set": int(in_fit.sum()), "n_not_in_fit_set": int((~in_fit).sum()),
                                      "in_fit_set_mask_sha256": hashlib.sha256(in_fit.tobytes()).hexdigest()}
            record["_in_fit"] = in_fit
        record["_record_readouts"] = meta["record"]["readouts_scan"]
    elif arm == "walk_row":
        child = {k: np.array(v, copy=True) for k, v in instance.params.items()}
        child[T.EMBEDDING][S.TOKEN_ID["jump"]] = instance.params[T.EMBEDDING][S.TOKEN_ID["walk"]]
        targets["new"] = np.where(y["new"] == S.CLASS_ID["I_JUMP"], S.CLASS_ID["I_WALK"], y["new"])
    elif arm.startswith("rename_"):
        choice = arm[len("rename_"):]
        pred_new, consistent = T.rename_jump_predictions(predictor(instance, instance.params), data["new"]["x"], choice)
        preds = {"new": pred_new, "old": base_pred["old"]}
        record["consistent_rows"] = {"n": int(consistent.sum()), "n_inconsistent": int((~consistent).sum()),
                                     "mask": consistent.astype(int).tolist()}
        record["_consistent"] = consistent
    elif arm == "swap_lr":
        pred_new, applied = T.swap_lr_predictions(predictor(instance, instance.params), data["new"]["x"])
        pred_old, applied_old = T.swap_lr_predictions(predictor(instance, instance.params), data["old"]["x"])
        if not applied.all() or applied_old.any():
            raise T11ScanError("swap_lr: every new row holds around right and no old row does")
        preds = {"new": pred_new, "old": pred_old}
        record["_around_left"] = T.holds_around_left(commands["new"])
    elif arm == "relabelled_program":
        rerun = json.loads(rerun_record_path(condition, "program_relabelled").read_text(encoding="utf-8"))
        moved_child = load_params(Path(rerun["child_path"]))
        if params_sha256(moved_child) != rerun["child_sha256"]:
            raise T11ScanError("the relabelled child does not hash to its rerun record")
        relabel = T.default_relabel()
        moved_base = T.relabel_params(instance.params, relabel)
        from weightcode.evalkit.patch_check import predict

        for part in ("new", "old"):
            xr = T.relabel_tokens(data[part]["x"], relabel)
            base_r = predict(instance.forward, moved_base, xr, BATCH).astype(np.int64)
            child_r = predict(instance.forward, moved_child, xr, BATCH).astype(np.int64)
            base_back = T.unrelabel_classes(base_r, relabel)
            child_back = T.unrelabel_classes(child_r, relabel)
            labelled = y[part] != -1
            record["checks"][f"relabelled_base_equals_base_{part}"] = bool(
                np.array_equal(base_back[labelled], base_pred[part][labelled]))
            preds[part] = child_back
        record["rerun"] = {"path": str(rerun_record_path(condition, "program_relabelled")),
                           "child_sha256": rerun["child_sha256"]}
    else:
        raise T11ScanError(f"unknown arm {arm!r} for {condition}")
    if child is not None:
        record["child_sha256"] = params_sha256(child)
        record["n_changed"] = int(sum(int(np.count_nonzero(np.asarray(child[k]) != np.asarray(instance.params[k])))
                                      for k in child))
        preds = {part: predictor(instance, child)(data[part]["x"]) for part in ("new", "old")}
    blocks = {part: part_block(commands[part], condition, part, targets[part], base_pred[part], preds[part])
              for part in ("new", "old")}
    record["new"], record["old"] = blocks["new"], blocks["old"]
    record.update(points(blocks["new"]["all"], blocks["old"]["all"]))
    in_fit = record.pop("_in_fit", None)
    if in_fit is not None:
        record["fit_set_rows"]["in_fit_set"] = T.part_readout(targets["new"], base_pred["new"], preds["new"], in_fit)
        record["fit_set_rows"]["not_in_fit_set"] = T.part_readout(targets["new"], base_pred["new"], preds["new"],
                                                                  ~in_fit)
    consistent = record.pop("_consistent", None)
    if consistent is not None:
        record["consistent_rows"]["consistent"] = T.part_readout(targets["new"], base_pred["new"], preds["new"],
                                                                 consistent)
        record["consistent_rows"]["inconsistent_not_mechanism_evidence"] = T.part_readout(
            targets["new"], base_pred["new"], preds["new"], ~consistent)
    around_left = record.pop("_around_left", None)
    if around_left is not None:
        record["around_left_rows"] = {
            "without_around_left": T.part_readout(targets["new"], base_pred["new"], preds["new"], ~around_left),
            "with_around_left": T.part_readout(targets["new"], base_pred["new"], preds["new"], around_left)}
    recorded = record.pop("_record_readouts", None)
    if recorded is not None:
        record["checks"]["new_equals_floor5_record"] = same_summary(blocks["new"]["all"], recorded["new"])
        record["checks"]["old_equals_floor5_record"] = same_summary(blocks["old"]["all"], recorded["old"])
    if arm == "base":
        expected = instance.private["base_vselect_em"]
        record["checks"]["em_equals_instance_record"] = (
            round(blocks["new"]["all"]["em_child"], 6) == round(float(expected["new"]), 6)
            and round(blocks["old"]["all"]["em_child"], 6) == round(float(expected["old"]), 6))
    if arm == "final":
        row = final_child(condition, instance)[1]["trajectory_row"]["readouts_scan"]
        record["checks"]["new_equals_session_readout"] = same_summary(blocks["new"]["all"], row["new"])
        record["checks"]["old_equals_session_readout"] = same_summary(blocks["old"]["all"], row["old"])
    if arm == "walk_row":
        walk = json.loads((runs() / "devreadouts" / str(CONDITIONS[condition]["walk_record"])).read_text())
        record["checks"]["new_equals_walk_record"] = same_summary(blocks["new"]["all"], walk["new_relabelled"])
        record["checks"]["old_equals_walk_record"] = same_summary(blocks["old"]["all"], walk["old"])
    record["code"] = code_record()
    return record


def cmd_arms(args: argparse.Namespace) -> int:
    from weightcode.data import scan as S
    from weightcode.evalkit.patch_check import predict

    condition = args.condition
    instance = instance_of(condition)
    data = vselect(instance)
    commands = {part: S.decode(np.asarray(data[part]["x"])) for part in ("new", "old")}
    base_pred = {part: predict(instance.forward, instance.params, np.asarray(data[part]["x"]), BATCH).astype(np.int64)
                 for part in ("new", "old")}
    arms = args.arms or list(ARMS[condition])
    for arm in arms:
        if arm not in ARMS[condition]:
            raise T11ScanError(f"{arm} is not an arm of {condition}")
        path = arm_record_path(arm, instance.instance_id)
        if path.exists():
            raise T11ScanError(f"{path} exists")
        started = time.perf_counter()
        record = arm_readout(condition, arm, instance, data, commands, base_pred)
        record["wall_s"] = time.perf_counter() - started
        digest = write_new_line(path, record)
        failed = [k for k, v in record["checks"].items() if v is False]
        print(f"{condition} {arm}: -> {path.name} ({digest[:12]}); checks failed: {failed or 'none'}", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# M04


def cmd_mech(args: argparse.Namespace) -> int:
    from weightcode.common.identity import params_sha256
    from weightcode.data import scan as S
    from weightcode.evalkit import scan_t11 as T

    condition = "addprim_jump"
    instance = instance_of(condition)
    path = t11_dir() / f"m04_{instance.instance_id}.json"
    if path.exists():
        raise T11ScanError(f"{path} exists")
    started = time.perf_counter()
    final, meta = final_child(condition, instance)
    children, comp = T.component_children(instance.params, final)
    nets = {"base": dict(instance.params), "E_only": children["E_only"], "H_only": children["H_only"],
            "EH": children["EH"]}
    record: dict[str, Any] = {"schema": SCHEMA_M04, "instance_id": instance.instance_id, "created_at": utc(),
                              "components": comp, "checks": {}, "slots": {}, "states": {}}
    record["checks"]["EH_equals_final_bit_for_bit"] = params_sha256(nets["EH"]) == meta["child_sha256"]
    data = vselect(instance)
    jump_cls = S.CLASS_ID["I_JUMP"]
    traces: dict[str, dict[str, Any]] = {}
    for part in ("new", "old"):
        x, y = np.asarray(data[part]["x"]), np.asarray(data[part]["y"])
        every = np.ones(x.shape, dtype=bool)
        for name, params in nets.items():
            trace = T.traced_forward(instance.forward, params, x, every, with_blocks=False)
            reference = np.asarray(instance.forward(params, x))
            record["checks"][f"traced_logits_equal_forward_{name}_{part}"] = T.bytes_equal(trace["logits"], reference)
            record["slots"].setdefault(name, {})[part] = T.slot_readout(y, trace["logits"])
            traces[f"{name}_{part}"] = {"head_in": trace["head_in"], "logits": trace["logits"]}
            del trace
        record["checks"][f"H_only_head_input_equals_base_{part}"] = T.bytes_equal(
            traces[f"H_only_{part}"]["head_in"], traces[f"base_{part}"]["head_in"])
        record["checks"][f"H_only_head_input_max_abs_difference_{part}"] = T.max_abs_difference(
            traces[f"H_only_{part}"]["head_in"], traces[f"base_{part}"]["head_in"])
        record["checks"][f"E_only_head_input_equals_EH_{part}"] = T.bytes_equal(
            traces[f"E_only_{part}"]["head_in"], traces[f"EH_{part}"]["head_in"])
        others = [c for c in range(S.N_CLASSES) if c != jump_cls]
        record["checks"][f"H_only_non_jump_logits_max_abs_difference_to_base_{part}"] = T.max_abs_difference(
            traces[f"H_only_{part}"]["logits"][..., others], traces[f"base_{part}"]["logits"][..., others])
        if part == "old":
            record["checks"]["E_only_old_logits_equal_base"] = T.bytes_equal(traces["E_only_old"]["logits"],
                                                                              traces["base_old"]["logits"])
        for key in [k for k in traces if k.endswith(part)]:
            del traces[key]
    # hidden states at the decision slots of the new part: action slots (target not END), split into I_JUMP slots and
    # the other action slots; the base on the commands with jump renamed to each trained primitive is the reference
    x, y = np.asarray(data["new"]["x"]), np.asarray(data["new"]["y"])
    action = (y != -1) & (y != S.END)
    is_jump = y[action] == jump_cls
    layers: dict[str, dict[str, np.ndarray]] = {}
    for name, params, rows in [("base", nets["base"], x), ("E_only", nets["E_only"], x),
                               *[(f"ref_{p}", nets["base"], T.rename_rows(x, p)) for p in T.PRIMITIVES_TRAINED]]:
        trace = T.traced_forward(instance.forward, params, rows, action)
        layers[name] = {**{block: states for block, states in zip(trace["block_names"], trace["blocks"])},
                        "head_in": trace["head_in"]}
        del trace
    for layer in layers["base"]:
        refs_j = [layers[f"ref_{p}"][layer][is_jump] for p in T.PRIMITIVES_TRAINED]
        refs_o = [layers[f"ref_{p}"][layer][~is_jump] for p in T.PRIMITIVES_TRAINED]
        record["states"][layer] = {
            net: {"jump_slots": T.state_comparison(layers[net][layer][is_jump], refs_j),
                  "other_action_slots": T.state_comparison(layers[net][layer][~is_jump], refs_o)}
            for net in ("base", "E_only")}
    record["states_definition"] = (
        "per new V_select action slot (target not END): the network's residual after each block and its head input, "
        "against the base's at the same slot of the command with every jump renamed to walk, look and run (the "
        "references); H_only's states are the base's and EH's are E_only's (checked bit for bit on the head input)")
    record["n_action_slots"] = {"jump": int(is_jump.sum()), "other": int((~is_jump).sum())}
    record["wall_s"] = time.perf_counter() - started
    record["code"] = code_record()
    digest = write_new_json(path, record)
    failed = [k for k, v in record["checks"].items() if v is False]
    print(f"m04 -> {path} ({digest[:12]}); checks failed: {failed or 'none'}", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# Summary


def paired(a: Sequence[int], b: Sequence[int], mask: np.ndarray | None = None) -> dict[str, int]:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if mask is not None:
        a, b = a[mask], b[mask]
    return {"both": int(np.sum(a & b)), "first_only": int(np.sum(a & ~b)), "second_only": int(np.sum(~a & b)),
            "neither": int(np.sum(~a & ~b))}


def cmd_summary(args: argparse.Namespace) -> int:
    summary: dict[str, Any] = {"schema": SCHEMA_SUMMARY, "created_at": utc(), "conditions": {}, "readings": {},
                               "code": code_record()}
    lines: list[str] = []
    for condition, spec in CONDITIONS.items():
        instance = str(spec["instance"])
        records = {arm: read_line(arm_record_path(arm, instance)) for arm in ARMS[condition]
                   if arm_record_path(arm, instance).exists()}
        reruns = {arm: json.loads(rerun_record_path(condition, arm).read_text()) for arm in ("program",
                  "program_relabelled") if rerun_record_path(condition, arm).exists()}
        final = records.get("final")
        table = []
        for arm, rec in records.items():
            row = {"arm": arm, "new_em": rec["new"]["all"]["em_child"], "old_em": rec["old"]["all"]["em_child"],
                   "gain": rec["gain_points"], "clean": rec["clean_points"], "joint": rec["joint_points"],
                   "n_changed": rec.get("n_changed"),
                   "checks_failed": [k for k, v in rec["checks"].items() if v is False]}
            if final is not None and arm != "final":
                row["paired_new_final_vs_arm"] = paired(final["new"]["right"], rec["new"]["right"])
                if "consistent_rows" in rec:
                    mask = np.asarray(rec["consistent_rows"]["mask"], dtype=bool)
                    row["consistent_rows"] = {
                        "n": int(mask.sum()), "arm_em": float(np.mean(np.asarray(rec["new"]["right"])[mask])),
                        "final_em": float(np.mean(np.asarray(final["new"]["right"])[mask])),
                        "paired_final_vs_arm": paired(final["new"]["right"], rec["new"]["right"], mask)}
                row["paired_old_final_vs_arm"] = paired(final["old"]["right"], rec["old"]["right"])
            table.append(row)
        refs = [r for r in table if r["arm"].startswith(f"ref_{RULEGEN_PREFIX}")]
        best_rule = max(refs, key=lambda r: r["joint"]) if refs else None
        wrappers = [r for r in table if r["arm"] in ("rename_auto", "swap_lr")]
        readings: dict[str, Any] = {}
        if final is not None and best_rule is not None:
            readings["rulegen_matches_final"] = {
                "best_rulegen": best_rule["arm"], "joint": best_rule["joint"], "final_joint": final["joint_points"],
                "matches": best_rule["joint"] >= final["joint_points"] - JOINT_TOLERANCE}
        for w in wrappers:
            if final is not None:
                readings[f"{w['arm']}_matches_final"] = {"joint": w["joint"], "final_joint": final["joint_points"],
                                                         "matches": w["joint"] >= final["joint_points"] - JOINT_TOLERANCE}
        if "relabelled_program" in records and final is not None:
            rec = records["relabelled_program"]
            readings["relabelled_program_equals_final_row_for_row"] = (
                rec["new"]["right"] == final["new"]["right"] and rec["old"]["right"] == final["old"]["right"])
        if condition == "addprim_jump" and final is not None:
            m04_path = t11_dir() / f"m04_{instance}.json"
            m04 = json.loads(m04_path.read_text()) if m04_path.exists() else None
            stop: dict[str, Any] = {}
            for comp in ("E_only", "H_only"):
                if comp in records:
                    rec = records[comp]
                    stop[comp] = {"new_em": rec["new"]["all"]["em_child"], "clean": rec["clean_points"],
                                  "suffices": (100 * rec["new"]["all"]["em_child"]
                                               >= 100 * final["new"]["all"]["em_child"] - JOINT_TOLERANCE
                                               and rec["clean_points"] >= -1.0)}
            if m04 is not None:
                at_jump = m04["slots"]["E_only"]["new"]["by_target"].get("I_JUMP", {})
                predicted = at_jump.get("predicted", {})
                share = sum(predicted.get(name, 0) for name in ("I_WALK", "I_LOOK", "I_RUN")) / max(at_jump.get("n", 1), 1)
                stop["E_only_share_jump_slots_predicted_wlr"] = share
                stop["E_only_structure_runs_action_missing"] = share >= SHARE_WLR_AT_JUMP
                head = m04["states"]["head_in"]
                stop["head_input_median_rel_dist_jump_slots"] = {
                    net: head[net]["jump_slots"]["rel_dist_to_mean"]["median"] for net in ("base", "E_only")}
                stop["m04_checks_failed"] = [k for k, v in m04["checks"].items() if v is False]
            if stop:
                both_needed = not any(stop.get(c, {}).get("suffices", False) for c in ("E_only", "H_only"))
                stop["reading"] = ("both components needed" if both_needed else
                                   "one component suffices: " + ", ".join(c for c in ("E_only", "H_only")
                                                                           if stop.get(c, {}).get("suffices")))
                readings["m04"] = stop
        summary["conditions"][condition] = {"instance": instance, "table": table,
                                            "reruns": {arm: {k: rec.get(k) for k in ("checks", "n_changed",
                                                                                     "child_sha256", "duration_s")}
                                                       for arm, rec in reruns.items()}}
        summary["readings"][condition] = readings
        lines += [f"## {condition} ({instance})", "",
                  "| arm | new EM | old EM | gain | clean | joint | n_changed | final only / arm only (new) |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in table:
            pn = row.get("paired_new_final_vs_arm")
            lines.append(f"| {row['arm']} | {100 * row['new_em']:.2f} | {100 * row['old_em']:.2f} | {row['gain']:+.2f} | "
                         f"{row['clean']:+.2f} | {row['joint']:+.2f} | {row['n_changed'] if row['n_changed'] is not None else ''} | "
                         f"{'' if pn is None else str(pn['first_only']) + ' / ' + str(pn['second_only'])} |")
        lines.append("")
        if final is not None:
            group_arms = [a for a in records if a in ("base", "final", "E_only", "H_only", "rename_auto", "swap_lr")
                          or a == (best_rule or {}).get("arm")]
            lines += ["| part | dimension | value | n | " + " | ".join(group_arms) + " |",
                      "|---|---|---|---|" + "---|" * len(group_arms)]
            for part in ("new", "old"):
                for dim, values in final[part]["groups"].items():
                    for value, block in values.items():
                        cells = [f"{100 * records[a][part]['groups'][dim][value]['em_child']:.2f}" for a in group_arms]
                        lines.append(f"| {part} | {dim} | {value} | {block['n']} | " + " | ".join(cells) + " |")
            lines.append("")
    summary["tables_markdown"] = "\n".join(lines)
    out = t11_dir() / "e13_t11_summary.json"
    digest = write_new_json(out, summary)
    md = t11_dir() / "e13_t11_tables.md"
    if md.exists():
        raise T11ScanError(f"{md} exists")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary -> {out} ({digest[:12]}), {md} ({sha256_file(md)[:12]})")
    print(json.dumps(summary["readings"], indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/e13/t11_scan.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    programs = sub.add_parser("programs")
    programs.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    arms = sub.add_parser("arms")
    arms.add_argument("--condition", required=True, choices=list(CONDITIONS))
    arms.add_argument("--arms", nargs="+", default=None)
    sub.add_parser("mech")
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    return {"programs": cmd_programs, "arms": cmd_arms, "mech": cmd_mech, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
