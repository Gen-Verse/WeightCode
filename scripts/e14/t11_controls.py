"""PREREG E14 T11 BabyAI controls (docs/PREREG_v0_DRAFT.md, E14 entry "T11 BabyAI controls"): diagnostics beside the
bar on instance p6ce8ad67 and the final of session s2502912e3f1b2c04. No LLM call, no NSCC job, no sealed read.

Subcommands, run in this order:

    programs --sandbox-python PATH [--threads 4]
        Write the two ablation copies of the final program (exact-once replacements of its text, diff recorded) under
        $WEIGHTCODE_RUNS/t11/programs/ beside a byte copy of the final, run each once in the LEARN program sandbox on the
        instance's student and experience as a transfer run stages them (``stage_student`` with the policyskill
        experience, ``run_update``, ``write_derived_patch``), and apply its deltas within the LEARN limits (a refused
        child is recorded and kept without the size limits, as transfer does). The unmodified final runs first; if its
        deltas are not final/deltas.npz bit for bit, nothing else runs. One record per program,
        $WEIGHTCODE_RUNS/t11/reruns/<arm>.json, with the child in $WEIGHTCODE_RUNS/t11/reruns/<arm>/child.npz.
    read --arms ARM [ARM ...]
        Traced V_select readouts (``e14_fn_readout`` on a ``RolloutTrace``) of each arm against the base, new and old
        mission, with every check of the entry: base success against the instance record, wrapper_v1 against the
        record of record, program arms against ``e14_readout`` and the program against the session's readout. One
        record per arm, $WEIGHTCODE_RUNS/devreadouts/t11_e14_<arm>_<instance>.jsonl (one line, never overwritten).
    independence
        The final against the base: tensor names, shapes and dtypes, patch ops, n_changed, forward.py, and forward time
        of base, final and the external rules on the same rows. $WEIGHTCODE_RUNS/t11/independence.json.
    summary
        $WEIGHTCODE_RUNS/t11/e14_t11_summary.json and $WEIGHTCODE_RUNS/t11/e14_t11_table.md from the records above.

    bash -c 'source scripts/env.sh; OMP_NUM_THREADS=2 nice -n 10 <venv>/bin/python -B scripts/e14/t11_controls.py read \\
        --arms base wrapper_v1'

The program sandbox runs at the session's own sandbox settings (4 threads, 8 GiB, 300 s), so a rerun is the session's
run; every other step runs at the caller's BLAS threads.
"""

from __future__ import annotations

import argparse
import difflib
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

INSTANCE = "p6ce8ad67"
SESSION = "s2502912e3f1b2c04"
FINAL_PROGRAM = "pickup_blockfwd_v2.py"
FINAL_QUERY = 1  # the session's kept check (info.json trajectory); its q0 is the program without step 4b
Q0_QUERY = 0
RULE_ARMS = ("wrapper_v2", "deadend_only", "wrapper_v1_goto", "wrapper_v2_goto")
PROGRAM_ARMS = {"program": FINAL_PROGRAM, "program_no_deadend": "t11_no_deadend.py",
                "program_no_trigger": "t11_no_trigger.py"}
ARMS = ("base", "wrapper_v1", *RULE_ARMS, *PROGRAM_ARMS)
ARM_DEFINITIONS = {
    "base": "the instance's base policy, greedy",
    "wrapper_v1": "the wrapper of record: base, pickup when the faced cell has the instruction's color and type",
    "wrapper_v2": "base, then the dead-end rule (forward below the row's minimum when the faced cell is not empty), "
                  "then wrapper_v1's trigger",
    "deadend_only": "base and the dead-end rule, no trigger",
    "wrapper_v1_goto": "wrapper_v1 with the base reading go/to in place of pick/up",
    "wrapper_v2_goto": "wrapper_v2 with the base reading go/to in place of pick/up",
    "program": "the session's final program (verb copy, trigger, dead-end), rerun in the sandbox",
    "program_no_deadend": "the final program with step 4b (dead-end) skipped",
    "program_no_trigger": "the final program with the writes of steps 2 to 5 (trigger) skipped; steps 1 and 4b kept",
}
PAIRS = (("program", "wrapper_v2_goto"), ("program", "wrapper_v1"), ("program", "wrapper_v2"),
         ("program", "program_no_deadend"), ("wrapper_v2", "wrapper_v1"), ("wrapper_v2_goto", "wrapper_v1_goto"),
         ("wrapper_v1_goto", "wrapper_v1"), ("program_no_deadend", "wrapper_v1_goto"),
         ("deadend_only", "base"), ("program_no_trigger", "base"))
"""Paired comparisons on the same episodes: discordant counts per mission."""
SCHEMA_RERUN = "weightcode.t11.e14_program_rerun.v1"
SCHEMA_ARM = "weightcode.t11.e14_arm_readout.v1"
SCHEMA_SUMMARY = "weightcode.t11.e14_summary.v1"
LEARN_ARM = "LEARN"


class T11Error(RuntimeError):
    """A check of the T11 entry failed; nothing is overwritten."""


# --------------------------------------------------------------------------------------------------------------------
# Paths, records


def runs() -> Path:
    from weightcode.common.paths import runs_dir

    return runs_dir()


def t11_dir() -> Path:
    return runs() / "t11"


def session_dir(session: str) -> Path:
    return runs() / "sessions" / session


def arm_record_path(arm: str, instance: str) -> Path:
    return runs() / "devreadouts" / f"t11_e14_{arm}_{instance}.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_json(path: Path, payload: Any) -> str:
    """Write ``payload`` once (refusing an existing file); returns the file's sha256."""
    if path.exists():
        raise T11Error(f"{path} exists; records are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def write_new_line(path: Path, payload: Any) -> str:
    if path.exists():
        raise T11Error(f"{path} exists; records are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def read_line(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise T11Error(f"{path}: expected one record, found {len(lines)} lines")
    return json.loads(lines[0])


def code_record() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                            check=True)
    return {"git_head": head.stdout.strip(), "git_dirty": bool(status.stdout.strip()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "e14_py_sha256": sha256_file(root / "weightcode" / "evalkit" / "e14.py"),
            "numpy": np.__version__, "blas_threads": {k: os.environ.get(k) for k in
                                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def session_info(session: str) -> dict[str, Any]:
    return json.loads((runs() / "devreadouts" / "learn" / session / "info.json").read_text(encoding="utf-8"))


def session_selection(session: str) -> dict[str, Any]:
    return json.loads((session_dir(session) / "final" / "selection.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------------------------------
# Program copies


def _line_index(lines: list[str], line: str) -> int:
    hits = [i for i, text in enumerate(lines) if text == line]
    if len(hits) != 1:
        raise T11Error(f"expected exactly one line {line!r}, found {len(hits)}")
    return hits[0]


def guard_block(text: str, first: str, last: str, flag: str) -> str:
    """Put the lines ``first`` to ``last`` (each exactly one full line of ``text``, inclusive) under ``if <flag>:``;
    blank lines stay blank."""
    lines = text.split("\n")
    start, stop = _line_index(lines, first), _line_index(lines, last)
    if stop < start:
        raise T11Error(f"{last!r} comes before {first!r}")
    indent = first[: len(first) - len(first.lstrip())]
    body = [("    " + line) if line.strip() else line for line in lines[start:stop + 1]]
    return "\n".join([*lines[:start], f"{indent}if {flag}:", *body, *lines[stop + 1:]])


def add_flag(text: str, flag: str, note: str) -> str:
    anchor = "EVAL_EXPERIENCE = True\n"
    if text.count(anchor) != 1:
        raise T11Error(f"expected exactly one {anchor!r}")
    return text.replace(anchor, f"{anchor}{flag} = False  # PREREG E14 T11 ablation copy: {note}\n")


def no_deadend_text(final: str) -> str:
    """The final with step 4b skipped and nothing else changed."""
    text = add_flag(final, "T11_DEADEND", "step 4b (dead-end) skipped")
    return guard_block(text,
                       "    # ---- step 4b: forward suppression when the faced cell is blocked (wall or object).",
                       "          % (kB, kR, jb, float(unp[:, jb].max()), float(uorig[:, jb].max()), "
                       "float(np.abs(Wh[:, jb]).max()), theta, Delta))",
                       "T11_DEADEND")


def no_trigger_text(final: str) -> str:
    """The final with the writes of steps 2 to 5 skipped (the instruction code unit, the conv1 and conv2 detector
    channels, the fc route of the trigger unit, the pickup bias); every quantity step 4b reads is still computed."""
    text = add_flag(final, "T11_TRIGGER", "the trigger writes of steps 2 to 5 skipped")
    text = guard_block(text, '    new["instr.weight"][js] = 0.0', '    new["film2.weight"][:, js] = 0.0', "T11_TRIGGER")
    text = guard_block(text, "    for k, sgn in ((kP, 1.0), (kN, -1.0)):",
                       '    new["film2.weight"][C + k2] = 0.0; new["film2.bias"][C + k2] = 3.0', "T11_TRIGGER")
    text = guard_block(text, '    new["fc.weight"][jstar, blk] = B', '    new["fc.weight"][jstar, blk] = B',
                       "T11_TRIGGER")
    line = '    new["head.bias"][a_pick] = params["head.bias"][a_pick] + kappa'
    return guard_block(text, line, line, "T11_TRIGGER")


def write_programs(session: str) -> dict[str, dict[str, Any]]:
    """The three program files under runs/t11/programs/ (written once; an existing file must be identical)."""
    source = session_dir(session) / "final" / FINAL_PROGRAM
    final = source.read_text(encoding="utf-8")
    texts = {"program": final, "program_no_deadend": no_deadend_text(final), "program_no_trigger": no_trigger_text(final)}
    folder = t11_dir() / "programs"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for arm, text in texts.items():
        compile(text, PROGRAM_ARMS[arm], "exec")
        path = folder / PROGRAM_ARMS[arm]
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise T11Error(f"{path} exists with other content")
        else:
            path.write_text(text, encoding="utf-8")
        diff = "".join(difflib.unified_diff(final.splitlines(True), text.splitlines(True), fromfile=FINAL_PROGRAM,
                                            tofile=PROGRAM_ARMS[arm], n=1))
        out[arm] = {"path": str(path), "sha256": sha256_file(path), "diff_vs_final": diff}
    if out["program"]["sha256"] != sha256_file(source):
        raise T11Error("the copy of the final differs from the session's final program")
    return out


# --------------------------------------------------------------------------------------------------------------------
# Program runs


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def deltas_equal(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Bit-for-bit comparison of two delta sets: keys, dtypes, shapes and bytes."""
    keys_equal = sorted(a) == sorted(b)
    per = {}
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            per[key] = "missing"
            continue
        x, y = np.asarray(a[key]), np.asarray(b[key])
        per[key] = bool(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes())
    return {"equal": keys_equal and all(v is True for v in per.values()), "keys_equal": keys_equal, "per_tensor": per}


def run_program_arm(instance: Any, arm: str, program: Path, sandbox: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from weightcode.common.identity import save_params
    from weightcode.evalkit.e14 import e14_program_limits
    from weightcode.harness import policyskill
    from weightcode.harness.arms import applied_limits, limits_dict
    from weightcode.harness.program import write_derived_patch, run_update
    from weightcode.harness.transfer import _child, stage_student
    from weightcode.patch.limits import dense_delta_bytes

    work = t11_dir() / "reruns" / arm
    if work.exists():
        raise T11Error(f"{work} exists; a rerun is never repeated into the same directory")
    work.mkdir(parents=True, mode=0o700)
    student, experience_file, experience = stage_student(instance, work / "student",
                                                         experience=policyskill.experience(instance))
    started = time.perf_counter()
    update = run_update(student, experience_file, program, work / "run", sandbox)
    run_s = time.perf_counter() - started
    patch_dir = work / "patch"
    patch_dir.mkdir()
    patch = write_derived_patch(update, instance.params, instance.params_sha256, patch_dir)
    n_tensors = len(student.params)
    dense = dense_delta_bytes(v.shape for v in student.params.values())
    limits = applied_limits(instance.task["limits"], LEARN_ARM, n_tensors, dense)
    if limits != e14_program_limits(instance.task["limits"], n_tensors, dense):
        raise T11Error("the LEARN limits differ from e14_program_limits")
    applied = _child(instance, patch, patch_dir, limits)
    result = applied.pop("result")
    child_sha = save_params(work / "child.npz", result.child)
    record = {
        "schema": SCHEMA_RERUN, "arm": arm, "instance_id": instance.instance_id, "created_at": utc(),
        "program": {"path": str(program), "name": program.name, "sha256": sha256_file(program)},
        "experience": experience, "sandbox": sandbox.as_dict(), "limits": limits_dict(limits),
        "inputs_sha256": update.record["inputs_sha256"], "deltas_sha256": update.record["deltas_sha256"],
        "deltas_path": str(update.deltas_path), "tensors": list(update.tensors),
        "stdout": update.record["run"]["stdout"], "stderr": update.record["run"]["stderr"],
        "duration_s": update.record["run"]["duration_s"], "run_wall_s": run_s,
        "patch_sha256": sha256_file(patch_dir / "derived_patch.json"), **applied,
        "child_sha256": child_sha, "child_path": str(work / "child.npz"), "n_changed": result.n_changed,
        "changed_frac": result.changed_frac, "per_tensor_changed": {k: v["changed"] for k, v in result.per_tensor.items()},
        "code": code_record(),
    }
    return record, update.deltas


def utc() -> str:
    from weightcode.evalkit.targets import utc_now

    return utc_now()


def cmd_programs(args: argparse.Namespace) -> int:
    from weightcode.evalkit.e14 import load_e14_instance
    from weightcode.sandbox.program import ProgramSandbox

    instance = load_e14_instance(args.instance)
    sandbox = ProgramSandbox(python=args.sandbox_python, memory_bytes=args.memory_bytes, threads=args.threads)
    final_dir = session_dir(args.session) / "final"
    final_run = json.loads((final_dir / "program_run.json").read_text(encoding="utf-8"))
    selection = session_selection(args.session)
    candidates = {c["query_index"]: c for c in selection["candidates"]}
    programs = write_programs(args.session)
    final_deltas = load_npz(final_dir / "deltas.npz")
    for arm in PROGRAM_ARMS:
        path = t11_dir() / "reruns" / f"{arm}.json"
        if path.exists():
            raise T11Error(f"{path} exists")
        record, deltas = run_program_arm(instance, arm, Path(programs[arm]["path"]), sandbox)
        record["program"]["diff_vs_final"] = programs[arm]["diff_vs_final"]
        checks: dict[str, Any] = {}
        if arm == "program":
            checks["deltas_vs_final"] = deltas_equal(deltas, final_deltas)
            checks["deltas_file_sha256_equal"] = record["deltas_sha256"] == final_run["deltas_sha256"]
            checks["child_equals_final"] = record["child_sha256"] == candidates[FINAL_QUERY]["child_sha256"]
            checks["n_changed_equals_final"] = record["n_changed"] == candidates[FINAL_QUERY]["n_changed"]
            checks["inputs_equal_session"] = {name: record["inputs_sha256"].get(name) == digest
                                              for name, digest in final_run["inputs_sha256"].items()}
            checks["sandbox_equal_session"] = {key: record["sandbox"].get(key) == value
                                               for key, value in final_run["sandbox"].items()}
        if arm == "program_no_deadend":
            q0_dir = session_dir(args.session) / "attempts" / candidates[Q0_QUERY]["attempt"]
            checks["deltas_vs_q0"] = deltas_equal(deltas, load_npz(q0_dir / "deltas.npz"))
            checks["child_equals_q0"] = record["child_sha256"] == candidates[Q0_QUERY]["child_sha256"]
            checks["n_changed_equals_q0"] = record["n_changed"] == candidates[Q0_QUERY]["n_changed"]
        record["checks"] = checks
        digest = write_new_json(path, record)
        print(f"{arm}: n_changed {record['n_changed']} within_limits {record['within_limits']} child "
              f"{record['child_sha256'][:12]} run {record['duration_s']:.1f} s; checks "
              f"{json.dumps({k: (v['equal'] if isinstance(v, dict) and 'equal' in v else v) for k, v in checks.items()})}"
              f" -> {path} ({digest[:12]})", flush=True)
        if arm == "program" and not (checks["deltas_vs_final"]["equal"] and checks["child_equals_final"]):
            raise T11Error("the rerun of the unmodified final does not reproduce its deltas bit for bit; T11 stops")
    return 0


# --------------------------------------------------------------------------------------------------------------------
# Readouts


def _drop_descriptors(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "demonstrated_descriptors"}


def _per_episode(children: Mapping[str, Any], steps: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    from weightcode.evalkit.e14 import seeds_sha256, trajectory_features

    out = {}
    for (part, child), part_steps in zip(children.items(), steps):
        feats = trajectory_features(part_steps, child.success)
        out[part] = {"seeds_sha256": seeds_sha256(child.seeds), "success": child.success.astype(int).tolist(),
                     "length": child.length.tolist(), "n_pickups": child.n_pickups.tolist(),
                     **{key: np.asarray(value).astype(int).tolist() for key, value in feats.items()}}
    return out


def arm_fn(instance: Any, arm: str) -> Any:
    from weightcode.common.identity import load_params, params_sha256
    from weightcode.evalkit.e14 import T11_RULES, control_logits_fn, logits_fn, wrapper_logits_fn

    if arm == "base":
        return logits_fn(instance.forward, instance.params), None
    if arm == "wrapper_v1":
        return wrapper_logits_fn(instance.forward, instance.params), None
    if arm in RULE_ARMS:
        return control_logits_fn(instance.forward, instance.params, **T11_RULES[arm]), None
    rerun = json.loads((t11_dir() / "reruns" / f"{arm}.json").read_text(encoding="utf-8"))
    child = load_params(Path(rerun["child_path"]))
    if params_sha256(child) != rerun["child_sha256"]:
        raise T11Error(f"{arm}: child.npz does not hash to its rerun record")
    return logits_fn(instance.forward, child), child


def read_arm(instance: Any, arm: str, session: str) -> dict[str, Any]:
    from weightcode.evalkit.e14 import RolloutTrace, e14_fn_readout, e14_readout, e14_wrapper_readout, success_sha256

    fn, child = arm_fn(instance, arm)
    trace = RolloutTrace(fn)
    started = time.perf_counter()
    readout, children = e14_fn_readout(instance, trace)
    traced_s = time.perf_counter() - started
    steps = trace.steps([children["new"].length, children["old"].length])
    checks: dict[str, Any] = {}
    timing = {"traced_readout_s": traced_s}
    if arm == "base":
        got = {part: success_sha256(result.success) for part, result in children.items()}
        checks["success_sha256_equals_instance_record"] = got == dict(instance.private["base_vselect_success_sha256"])
    if arm == "wrapper_v1":
        started = time.perf_counter()
        existing = e14_wrapper_readout(instance)
        timing["existing_readout_s"] = time.perf_counter() - started
        record = read_line(runs() / "devreadouts" / f"e14_wrapper_{instance.instance_id}.jsonl")
        checks["traced_equals_e14_wrapper_readout"] = readout == existing
        checks["equals_record_field_for_field"] = existing == record["readout"]
        checks["record_differences"] = sorted(diff_paths(existing, record["readout"]))
    if child is not None:
        started = time.perf_counter()
        existing = e14_readout(instance, child)
        timing["existing_readout_s"] = time.perf_counter() - started
        checks["traced_equals_e14_readout"] = readout == _drop_descriptors(existing)
        query = {"program": FINAL_QUERY, "program_no_deadend": Q0_QUERY}.get(arm)
        if query is not None:
            row = session_info(session)["trajectory"][query]
            checks[f"equals_session_query_{query}_readout"] = existing == row["readouts_policyskill"]
            checks[f"session_query_{query}_differences"] = sorted(diff_paths(existing, row["readouts_policyskill"]))
    return {"schema": SCHEMA_ARM, "arm": arm, "definition": ARM_DEFINITIONS[arm], "instance_id": instance.instance_id,
            "session": session, "created_at": utc(), "readout": readout, "checks": checks, "timing_s": timing,
            "per_episode": _per_episode(children, steps), "code": code_record()}


def diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    """The paths at which two JSON values differ."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{prefix}/{key} (missing)")
            else:
                out.extend(diff_paths(a[key], b[key], f"{prefix}/{key}"))
        return out
    return [] if a == b else [prefix or "/"]


def cmd_read(args: argparse.Namespace) -> int:
    from weightcode.evalkit.e14 import load_e14_instance

    unknown = [arm for arm in args.arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known {list(ARMS)}")
    instance = load_e14_instance(args.instance)
    for arm in args.arms:
        path = arm_record_path(arm, args.instance)
        if path.exists():
            raise T11Error(f"{path} exists")
        record = read_arm(instance, arm, args.session)
        digest = write_new_line(path, record)
        ro = record["readout"]
        print(f"{arm}: new {ro['new']['success_child']:.3f} old {ro['old']['success_child']:.3f} gain "
              f"{ro['gain']:+.1f} clean {ro['clean']:+.1f} joint {ro['joint']:+.1f}; checks "
              f"{json.dumps(record['checks'], sort_keys=True)}; {json.dumps(record['timing_s'])} -> {path.name} "
              f"({digest[:12]})", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# Independence


def cmd_independence(args: argparse.Namespace) -> int:
    from weightcode.common.identity import count_changed, load_params, params_sha256
    from weightcode.evalkit.e14 import (control_logits_fn, load_e14_instance, logits_fn, T11_RULES, wrapper_logits_fn)

    path = t11_dir() / "independence.json"
    instance = load_e14_instance(args.instance)
    final_dir = session_dir(args.session) / "final"
    child = load_params(final_dir / "model.npz")
    base = instance.params
    patch = json.loads((final_dir / "patch.json").read_text(encoding="utf-8"))
    selection = session_selection(args.session)
    final_row = {c["query_index"]: c for c in selection["candidates"]}[FINAL_QUERY]
    run = json.loads((final_dir / "program_run.json").read_text(encoding="utf-8"))
    ops = [{"op": op.get("op"), "tensor": op.get("tensor")} for op in patch["ops"]]
    rows = np.concatenate([instance.evidence["x_new"], instance.evidence["x_old"]])
    rows = np.tile(rows, (10, 1))  # 1,010 rows
    fns = {"base forward.py": logits_fn(instance.forward, base), "final forward.py": logits_fn(instance.forward, child),
           "wrapper_v1 (rule outside)": wrapper_logits_fn(instance.forward, base),
           **{f"{arm} (rule outside)": control_logits_fn(instance.forward, base, **T11_RULES[arm]) for arm in RULE_ARMS}}
    timing = {}
    for name, fn in fns.items():
        fn(rows[:16])
        laps = []
        for _ in range(5):
            started = time.perf_counter()
            fn(rows)
            laps.append(time.perf_counter() - started)
        timing[name] = {"rows": int(len(rows)), "median_s": float(np.median(laps)), "laps_s": laps}
    record = {
        "schema": "weightcode.t11.e14_independence.v1", "instance_id": args.instance, "session": args.session,
        "created_at": utc(),
        "final_model_sha256": params_sha256(child), "final_child_sha256_of_record": final_row["child_sha256"],
        "same_tensor_names_in_order": list(child) == list(base),
        "same_shapes_and_dtypes": all(child[k].shape == base[k].shape and child[k].dtype == base[k].dtype for k in base),
        "n_tensors": len(child), "n_parameters": int(sum(v.size for v in child.values())),
        "n_changed": count_changed(base, child), "n_changed_of_record": final_row["n_changed"],
        "patch_ops": ops, "patch_ops_all_add_dense_on_existing": all(o["op"] == "add_dense" and o["tensor"] in base
                                                                     for o in ops),
        "forward_py_sha256": sha256_file(Path(instance.instance_dir) / "forward.py"),
        "forward_py_sha256_of_record": instance.private["forward_sha256"],
        "model_spec_sha256": sha256_file(Path(instance.instance_dir) / "model_spec.json"),
        "model_spec_sha256_of_record": instance.private["model_spec_sha256"],
        "readout_path": "weightcode.evalkit.e14.e14_readout(instance, child): logits_fn(instance forward.py, child), argmax "
                        "per step, no rule outside the weights",
        "program_run_duration_s": run["run"]["duration_s"], "forward_timing": timing, "code": code_record(),
    }
    digest = write_new_json(path, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("patch_ops", "forward_timing", "code")}, indent=1))
    print(json.dumps({k: round(v["median_s"], 4) for k, v in timing.items()}, indent=1), f"-> {path} ({digest[:12]})")
    return 0


# --------------------------------------------------------------------------------------------------------------------
# Summary


def _category_table(records: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    from weightcode.evalkit.e14 import T11_CATEGORIES, episode_categories

    out: dict[str, Any] = {}
    for part in ("new", "old"):
        feats = {k: np.asarray(v, dtype=bool) for k, v in base["per_episode"][part].items()
                 if k in ("before_blocked", "before_wrong", "before_stuck", "no_target")}
        masks = episode_categories(feats)
        out[part] = {"n": {name: int(masks[name].sum()) for name in T11_CATEGORIES}, "success": {}}
        for arm, record in records.items():
            success = np.asarray(record["per_episode"][part]["success"], dtype=bool)
            out[part]["success"][arm] = {name: {"k": int(success[masks[name]].sum()), "n": int(masks[name].sum())}
                                         for name in T11_CATEGORIES}
    return out


def _arm_row(record: Mapping[str, Any], rerun: Mapping[str, Any] | None) -> dict[str, Any]:
    ro = record["readout"]
    row = {"arm": record["arm"], "definition": record["definition"],
           "success_new": ro["new"]["success_child"], "success_old": ro["old"]["success_child"],
           "gain": ro["gain"], "clean": ro["clean"], "joint": ro["joint"],
           "fixed_new": ro["new"]["fixed"], "broken_new": ro["new"]["broken"],
           "fixed_old": ro["old"]["fixed"], "broken_old": ro["old"]["broken"],
           "success_new_demonstrated": ro["new"]["by_demonstrated_descriptor"]["demonstrated"]["success_child"],
           "success_new_not_demonstrated": ro["new"]["by_demonstrated_descriptor"]["not_demonstrated"]["success_child"]}
    for part in ("new", "old"):
        pe = record["per_episode"][part]
        row[f"forward_blocked_episodes_{part}"] = int(np.count_nonzero(pe["forward_blocked_steps"]))
        row[f"forward_blocked_steps_{part}"] = int(np.sum(pe["forward_blocked_steps"]))
        row[f"carried_nontarget_{part}"] = int(np.sum(pe["carried_nontarget"]))
        row[f"drops_{part}"] = int(np.sum(pe["drops"]))
        row[f"episodes_with_pickup_{part}"] = int(np.count_nonzero(pe["n_pickups"]))
        row[f"mean_length_{part}"] = ro[part]["mean_length_child"]
    if rerun is not None:
        row.update(n_changed=rerun["n_changed"], within_limits=rerun["within_limits"], child_sha256=rerun["child_sha256"])
    return row


def _pairs(records: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for a, b in PAIRS:
        if a not in records or b not in records:
            continue
        row: dict[str, Any] = {"a": a, "b": b}
        for part in ("new", "old"):
            sa = np.asarray(records[a]["per_episode"][part]["success"], dtype=bool)
            sb = np.asarray(records[b]["per_episode"][part]["success"], dtype=bool)
            row[part] = {"a_only": int(np.count_nonzero(sa & ~sb)), "b_only": int(np.count_nonzero(~sa & sb)),
                         "both": int(np.count_nonzero(sa & sb)), "neither": int(np.count_nonzero(~sa & ~sb))}
        row["joint_a_minus_b"] = records[a]["readout"]["joint"] - records[b]["readout"]["joint"]
        out.append(row)
    return out


def _fmt(value: Any, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def markdown(summary: Mapping[str, Any]) -> str:
    from weightcode.evalkit.e14 import T11_CATEGORIES

    lines = [f"# E14 T11 controls ({summary['instance_id']}, session {summary['session']}; dev, seed 100)", "",
             "V_select, 1,000 new and 1,000 old episodes, child against the base. Points are 100 x delta success; "
             "joint = gain + clean. Diagnostics beside the bar; the B reading is unchanged.", "",
             "| arm | new success | old success | gain | clean | joint | fixed new / old | broken new / old | "
             "n_changed |", "|---|---|---|---|---|---|---|---|---|"]
    for row in summary["arms"]:
        lines.append(f"| {row['arm']} | {row['success_new']:.3f} | {row['success_old']:.3f} | {row['gain']:+.1f} | "
                     f"{row['clean']:+.1f} | {row['joint']:+.1f} | {row['fixed_new']} / {row['fixed_old']} | "
                     f"{row['broken_new']} / {row['broken_old']} | {_fmt(row.get('n_changed'), ',')} |")
    for part, title in (("new", "new mission (pick up)"), ("old", "old mission (go to)")):
        cats = summary["categories"][part]
        lines += ["", f"Success per episode category, {title}; categories from the base's own trajectory before it "
                      "first faces the target (n per category in the header).", "",
                  "| arm | " + " | ".join(f"{name} (n {cats['n'][name]})" for name in T11_CATEGORIES) + " |",
                  "|---|" + "---|" * len(T11_CATEGORIES)]
        for arm, by in cats["success"].items():
            lines.append(f"| {arm} | " + " | ".join(
                f"{by[name]['k']}/{by[name]['n']}" + (f" ({by[name]['k'] / by[name]['n']:.3f})" if by[name]["n"] else "")
                for name in T11_CATEGORIES) + " |")
    lines += ["", "Per-arm trajectory facts: episodes with a forward step into a cell that is not empty (steps), "
                  "episodes that pick up a non-target object, drop actions, episodes with a pickup.", "",
              "| arm | forward-blocked new | forward-blocked old | carried non-target new / old | drops new / old | "
              "pickup episodes new / old |", "|---|---|---|---|---|---|"]
    for row in summary["arms"]:
        lines.append(f"| {row['arm']} | {row['forward_blocked_episodes_new']} ({row['forward_blocked_steps_new']}) | "
                     f"{row['forward_blocked_episodes_old']} ({row['forward_blocked_steps_old']}) | "
                     f"{row['carried_nontarget_new']} / {row['carried_nontarget_old']} | {row['drops_new']} / "
                     f"{row['drops_old']} | {row['episodes_with_pickup_new']} / {row['episodes_with_pickup_old']} |")
    lines += ["", "Paired comparisons on the same episodes (a only / b only successes).", "",
              "| a | b | new a only / b only | old a only / b only | joint a - b |", "|---|---|---|---|---|"]
    for row in summary["pairs"]:
        lines.append(f"| {row['a']} | {row['b']} | {row['new']['a_only']} / {row['new']['b_only']} | "
                     f"{row['old']['a_only']} / {row['old']['b_only']} | {row['joint_a_minus_b']:+.1f} |")
    return "\n".join(lines) + "\n"


def cmd_summary(args: argparse.Namespace) -> int:
    records, sources = {}, {}
    for arm in ARMS:
        path = arm_record_path(arm, args.instance)
        if path.exists():
            records[arm] = read_line(path)
            sources[arm] = {"path": str(path), "sha256": sha256_file(path)}
    if "base" not in records:
        raise T11Error("the base record is needed for the categories")
    reruns = {}
    for arm in PROGRAM_ARMS:
        path = t11_dir() / "reruns" / f"{arm}.json"
        if path.exists():
            reruns[arm] = json.loads(path.read_text(encoding="utf-8"))
            sources[f"rerun:{arm}"] = {"path": str(path), "sha256": sha256_file(path)}
    independence_path = t11_dir() / "independence.json"
    independence = json.loads(independence_path.read_text(encoding="utf-8")) if independence_path.exists() else None
    if independence is not None:
        sources["independence"] = {"path": str(independence_path), "sha256": sha256_file(independence_path)}
    summary = {
        "schema": SCHEMA_SUMMARY, "instance_id": args.instance, "session": args.session, "created_at": utc(),
        "prereg": "docs/PREREG_v0_DRAFT.md, E14 entry 'T11 BabyAI controls'",
        "arms_missing": [arm for arm in ARMS if arm not in records],
        "arms": [_arm_row(records[arm], reruns.get(arm)) for arm in ARMS if arm in records],
        "categories": _category_table(records, records["base"]),
        "pairs": _pairs(records),
        "checks": {arm: record["checks"] for arm, record in records.items()},
        "rerun_checks": {arm: rerun["checks"] for arm, rerun in reruns.items()},
        "timing_s": {arm: record["timing_s"] for arm, record in records.items()},
        "independence": independence, "sources": sources, "code": code_record(),
    }
    summary_path = t11_dir() / "e14_t11_summary.json"
    table_path = t11_dir() / "e14_t11_table.md"
    for path in (summary_path, table_path):
        if path.exists() and not args.replace:
            raise T11Error(f"{path} exists (--replace rebuilds the summary and table from the records)")
    text = json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    table_path.write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    print(f"-> {summary_path} ({sha256_file(summary_path)[:12]}), {table_path} ({sha256_file(table_path)[:12]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, DEFAULT_THREADS

    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", default=INSTANCE)
    parser.add_argument("--session", default=SESSION)
    sub = parser.add_subparsers(dest="command", required=True)
    programs = sub.add_parser("programs")
    programs.add_argument("--sandbox-python", type=Path, required=True,
                          help="the LEARN sandbox interpreter (the session's: its worktree's .venv-sandbox)")
    programs.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    programs.add_argument("--memory-bytes", type=int, default=DEFAULT_MEMORY_BYTES)
    read = sub.add_parser("read")
    read.add_argument("--arms", nargs="+", required=True)
    sub.add_parser("independence")
    summary = sub.add_parser("summary")
    summary.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    return {"programs": cmd_programs, "read": cmd_read, "independence": cmd_independence,
            "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
