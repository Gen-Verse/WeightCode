"""PREREG E12 C: the two E12 B finals of record, rerun unchanged on five more pretrained students, and F6's frozen delta.

Spec: docs/PREREG_v0_DRAFT.md, entry "E12 C, reuse on five more pretrained students". No LLM call, no sealed read; dev
instances only (V_check and the sealed store are never read; main instances are refused by the harness).

Stages:

* ``programs`` (this workstation: the LEARN program sandbox exists only here). For each target instance, E3's
  ``stage_student`` and ``run_update`` (``weightcode.harness.transfer``): the program runs once in the sandbox with the same wall
  limit and virtual clock as the session, and its deltas are kept. One record per run in ``$WEIGHTCODE_RUNS/e12/c/rerun/``
  (``rerun_<program sha256[:12]>_<instance>.json``, never overwritten). The source instance is run too as a gate: its
  deltas must hash to the session's final deltas. Needs at least ``MIN_FREE_GB`` of free memory before each run.
* ``read`` (NSCC CPU, or any machine). ``--kind rerun``: the program-run deltas of a record become the child exactly as E3
  makes it (``write_derived_patch``, the target's LEARN limits, ``transfer._child``) and are read on V_select of the
  condition and on its clean companion (``clean.dev_readout_joint``). ``--kind frozen``: the session's final deltas
  (``weightcode.harness.frozen_edit``, F6) added unchanged to the target, read the same way. Both reuse the code after the
  program run that E3 and F6 share; F6's seed-100 gate showed that this gives E3's child and readouts field for field.
* ``summary``: per condition the table (rerun and frozen, gain and clean, within limits), the reuse ratio against the
  own gain of E3's rerun on the session's own student, and the readings fixed in the entry.

Readings (PREREG E12 C, fixed before any number; E10's C as fixed for E11): C is supported when the reuse ratio (mean
rerun gain over the five targets / the own gain) is at least 0.7, the rerun's clean change is at least -1.0 on at least 4
of 5 targets and every rerun child is within the LEARN limits. Frozen against rerun as PREREG F6 reads it: a frozen mean
gain at most 50% of the rerun mean gain reads "the program, not its values", at least 90% "the values transfer", in
between both are reported.

    source scripts/env.sh
    python scripts/e12/e12c_run.py programs --condition gaussian_noise_s5
    python scripts/e12/e12c_run.py read --kind rerun --condition gaussian_noise_s5 --instance i... --out DIR
    python scripts/e12/e12c_run.py summary --out FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

THREADS = 4
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, str(THREADS))  # before numpy is imported, or BLAS ignores it

import numpy as np  # noqa: E402

DATASET = "cifar10_r18"
SEEDS = (101, 102, 103, 104, 105)
SOURCE_SEED = 100
MIN_FREE_GB = 6.0
RATIO_BAR = 0.7
CLEAN_FLOOR = -1.0
CLEAN_OK_NEEDED = 4
RERUN_SCHEMA = "weightcode.e12c.rerun.v0"
FINALS: dict[str, dict[str, str]] = {
    "gaussian_noise_s5": {"session": "s2abaefab30d20b42", "source_instance": "i26d23d3e",
                          "own_record": "transfer_2ac2033a2936_i26d23d3e.json"},
    "contrast_s5": {"session": "sccd12677dcdd125b", "source_instance": "ie3840b28",
                    "own_record": "transfer_985ba1fd5ee0_ie3840b28.json"},
}


class E12CError(RuntimeError):
    """A final, a target or a record does not match what the entry fixes."""


def runs_root() -> Path:
    from weightcode.common.paths import runs_dir

    return Path(runs_dir())


def out_root() -> Path:
    return runs_root() / "e12" / "c"


def session_dir(condition: str) -> Path:
    return runs_root() / "sessions" / FINALS[condition]["session"]


def targets(condition: str) -> dict[int, str]:
    """Seed -> instance id of the E12 dev instances of ``condition`` for the students of seeds 101 to 105 (one each)."""
    from weightcode.evalkit.instances import instances_root

    index = json.loads((instances_root() / "index.json").read_text(encoding="utf-8"))
    rows = [r for r in index["instances"] if r.get("dataset") == DATASET and r.get("condition") == condition
            and r.get("split_role") == "dev" and r.get("evidence_rule") == "evidence-v1"]
    found: dict[int, str] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in found:
            raise E12CError(f"two active {condition} instances for seed {seed}: {found[seed]} and {row['instance_id']}")
        found[seed] = str(row["instance_id"])
    missing = [s for s in (SOURCE_SEED, *SEEDS) if s not in found]
    if missing:
        raise E12CError(f"no {DATASET} {condition} dev instance for seeds {missing}")
    if found[SOURCE_SEED] != FINALS[condition]["source_instance"]:
        raise E12CError(f"the seed-{SOURCE_SEED} instance {found[SOURCE_SEED]} is not the final's "
                        f"{FINALS[condition]['source_instance']}")
    return {s: found[s] for s in (SOURCE_SEED, *SEEDS)}


def free_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    raise E12CError("MemAvailable is not in /proc/meminfo")


def wait_for_memory(minimum: float = MIN_FREE_GB, timeout_s: float = 1800.0) -> None:
    started = time.monotonic()
    while free_gb() < minimum:
        if time.monotonic() - started > timeout_s:
            raise E12CError(f"less than {minimum} GB free for {timeout_s:.0f} s; not starting the run")
        time.sleep(30)


def final_program(condition: str) -> tuple[Path, str]:
    from weightcode.harness.jsonio import file_sha256, read_json

    root = session_dir(condition)
    run = read_json(root / "run.json")
    program = run["final"]["program"]
    path = root / program["file"]
    digest = file_sha256(path)
    if digest != program["sha256"]:
        raise E12CError(f"{path} hashes to {digest}, run.json says {program['sha256']}")
    return path, str(program["deltas_sha256"])


# --------------------------------------------------------------------------------------------------------------------
# programs (the LEARN program sandbox, local)


def rerun_one(program: Path, instance: Any, work: Path, sandbox: Any, final_deltas_sha: str | None) -> dict[str, Any]:
    """E3's program run on one target (``transfer_one`` up to the deltas), without the child and the readout."""
    from weightcode.harness.jsonio import file_sha256, utc_now
    from weightcode.harness.program import ProgramProblem, run_update
    from weightcode.harness.transfer import stage_student
    from weightcode.sandbox.program import PROGRAM_WALL_S

    work.mkdir(mode=0o700)
    student, experience_file, experience = stage_student(instance, work / "student")
    record: dict[str, Any] = {
        "schema": RERUN_SCHEMA, "created_at": utc_now(),
        "program": {"path": str(program), "name": program.name, "sha256": file_sha256(program)},
        "target": {"instance_id": instance.instance_id, "dataset": instance.private.get("dataset"),
                   "condition": instance.private.get("condition"), "split_role": instance.private.get("split_role"),
                   "model_id": instance.private.get("model_id"), "seed": instance.private.get("seed"),
                   "params_sha256": instance.params_sha256},
        "experience": experience, "sandbox": sandbox.as_dict(), "wall_s": PROGRAM_WALL_S, "work_dir": str(work),
    }
    try:
        update = run_update(student, experience_file, program, work / "run", sandbox, PROGRAM_WALL_S)
    except ProgramProblem as problem:
        record.update(status="program_failed", messages=problem.messages, run=problem.run)
        return record
    record.update(status="program_ok", run=update.record, deltas_path=str(update.deltas_path),
                  deltas_sha256=update.record["deltas_sha256"], tensors=update.tensors,
                  hypothesis=update.hypothesis, predicted_effect=update.predicted_effect)
    if final_deltas_sha is not None:  # the gate: the source instance gives the session's final deltas back
        record["gate_final_deltas_sha256"] = final_deltas_sha
        record["gate_equal"] = update.record["deltas_sha256"] == final_deltas_sha
    return record


def cmd_targets(args: argparse.Namespace) -> int:
    """Print the five target instance ids (seeds 101 to 105) of a condition, one line, space separated."""
    by_seed = targets(args.condition)
    print(" ".join(by_seed[s] for s in SEEDS))
    return 0


def cmd_programs(args: argparse.Namespace) -> int:
    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import instances_root
    from weightcode.harness.jsonio import file_sha256, write_json
    from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, SANDBOX_PYTHON
    from weightcode.harness.transfer import TransferError
    from weightcode.sandbox.program import ProgramSandbox

    condition = args.condition
    program, final_deltas_sha = final_program(condition)
    by_seed = targets(condition)
    wanted = args.instance or [by_seed[s] for s in (SOURCE_SEED, *SEEDS)]
    sandbox = ProgramSandbox(python=Path(args.sandbox_python or SANDBOX_PYTHON), memory_bytes=DEFAULT_MEMORY_BYTES,
                             threads=THREADS)
    folder = out_root() / "rerun"
    folder.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(program)
    for instance_id in wanted:
        if instance_id not in by_seed.values():
            raise E12CError(f"{instance_id} is not one of {condition}'s targets {sorted(by_seed.values())}")
        stem = f"rerun_{digest[:12]}_{instance_id}"
        path = folder / f"{stem}.json"
        if path.exists() or (folder / stem).exists():
            raise TransferError(f"{path} or its work directory exists already; records are never overwritten")
        instance = load_instance(instances_root() / instance_id)
        is_source = instance_id == FINALS[condition]["source_instance"]
        wait_for_memory()
        started = time.monotonic()
        record = rerun_one(program, instance, folder / stem, sandbox, final_deltas_sha if is_source else None)
        write_json(path, record)
        print(json.dumps({"record": str(path), "target": instance_id, "status": record["status"],
                          "gate_equal": record.get("gate_equal"), "seconds": round(time.monotonic() - started, 1)}),
              flush=True)
        if record["status"] != "program_ok":
            return 2
        if is_source and not record["gate_equal"]:
            print("GATE FAILED: the source instance's rerun deltas differ from the session's final deltas", flush=True)
            return 3
    return 0


# --------------------------------------------------------------------------------------------------------------------
# read (NSCC CPU)


def cmd_read(args: argparse.Namespace) -> int:
    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
    from weightcode.evalkit.instances import instances_root
    from weightcode.harness import frozen_edit as F
    from weightcode.harness.jsonio import file_sha256, read_json, write_json

    condition = args.condition
    by_seed = targets(condition)
    if args.instance not in by_seed.values():
        raise E12CError(f"{args.instance} is not one of {condition}'s targets {sorted(by_seed.values())}")
    batch = args.batch_size or DEFAULT_BATCH_SIZE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "frozen":
        for path in F.run_frozen(session_dir(condition), [args.instance], out, batch):
            record = read_json(path)
            print(json.dumps({"record": str(path), "target": args.instance, "status": record["status"],
                              "within_limits": record.get("within_limits"),
                              "gain": (record.get("readouts") or {}).get("delta_acc"),
                              "clean": (record.get("readouts_clean") or {}).get("delta_acc")}), flush=True)
        return 0
    program, _ = final_program(condition)
    digest = file_sha256(program)
    rerun_path = out_root() / "rerun" / f"rerun_{digest[:12]}_{args.instance}.json"
    rerun = read_json(rerun_path)
    if rerun.get("status") != "program_ok" or rerun["program"]["sha256"] != digest:
        raise E12CError(f"{rerun_path}: status {rerun.get('status')!r} or another program")
    deltas_path = Path(rerun["deltas_path"])
    if file_sha256(deltas_path) != rerun["deltas_sha256"]:
        raise E12CError(f"{deltas_path} does not hash to the record's deltas_sha256")
    stem = f"rerun_{digest[:12]}_{args.instance}"
    path = out / f"{stem}.json"
    if path.exists() or (out / stem).exists():
        raise E12CError(f"{path} or its work directory exists already; records are never overwritten")
    source = {"kind": "rerun", "program": rerun["program"], "run_record": rerun["run"], "experience": rerun["experience"],
              "rerun_record": str(rerun_path), "deltas": str(deltas_path), "deltas_sha256": rerun["deltas_sha256"]}
    edit = F.FrozenEdit(session_id=f"rerun_{digest[:12]}", source_instance_id=args.instance, deltas_path=deltas_path,
                        deltas_sha256=rerun["deltas_sha256"], hypothesis=str(rerun.get("hypothesis", "")),
                        predicted_effect=str(rerun.get("predicted_effect", "")), record=source)
    instance = load_instance(instances_root() / args.instance)
    record = F.frozen_one(edit, instance, out / stem, batch)
    record["schema"] = "weightcode.e12c.rerun_read.v0"
    write_json(path, record)
    print(json.dumps({"record": str(path), "target": args.instance, "status": record["status"],
                      "within_limits": record.get("within_limits"),
                      "gain": (record.get("readouts") or {}).get("delta_acc"),
                      "clean": (record.get("readouts_clean") or {}).get("delta_acc")}), flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# summary


def points(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 3)


def condition_table(rerun: Sequence[Mapping[str, Any]], frozen: Sequence[Mapping[str, Any]],
                    own: Mapping[str, Any]) -> dict[str, Any]:
    """The reading of one condition from its rerun and frozen records on the five targets and E3's own-student record."""
    from weightcode.harness import frozen_edit as F

    def row(r: Mapping[str, Any]) -> dict[str, Any]:
        return {"instance_id": str(r["target"]["instance_id"]), "seed": int(r["target"]["seed"]), "status": r["status"],
                "within_limits": r.get("within_limits"), "gain": points(r["readouts"]["delta_acc"]),
                "clean": points(r["readouts_clean"]["delta_acc"]), "n_changed": r["n_changed"],
                "child_sha256": r["child_sha256"]}

    rr = sorted((row(r) for r in rerun if int(r["target"]["seed"]) in SEEDS), key=lambda x: x["seed"])
    ff = sorted((row(r) for r in frozen if int(r["target"]["seed"]) in SEEDS), key=lambda x: x["seed"])
    if [r["seed"] for r in rr] != list(SEEDS) or [r["seed"] for r in ff] != list(SEEDS):
        raise E12CError(f"seeds read: rerun {[r['seed'] for r in rr]}, frozen {[r['seed'] for r in ff]}; need {SEEDS}")
    if any(r["status"] != "ok" for r in rr + ff):
        raise E12CError("a read has status other than ok")
    own_gain, own_clean = points(own["readouts"]["delta_acc"]), points(own["readouts_clean"]["delta_acc"])
    rerun_mean = float(np.mean([r["gain"] for r in rr]))
    frozen_mean = float(np.mean([r["gain"] for r in ff]))
    ratio = rerun_mean / own_gain if own_gain and own_gain > 0 else None
    clean_ok = sum(r["clean"] >= CLEAN_FLOOR for r in rr)
    within = all(r["within_limits"] for r in rr)
    supported = bool(ratio is not None and ratio >= RATIO_BAR and clean_ok >= CLEAN_OK_NEEDED and within)
    share = frozen_mean / rerun_mean if rerun_mean > 0 else None
    if share is None:
        share_reading = "undefined: the rerun mean gain is not positive"
    elif share <= F.PROGRAM_SHARE:
        share_reading = F.READ_PROGRAM
    elif share >= F.VALUES_SHARE:
        share_reading = F.READ_VALUES
    else:
        share_reading = F.READ_BOTH
    return {"own": {"gain": own_gain, "clean": own_clean, "within_limits": own["within_limits"],
                    "child_sha256": own["child_sha256"]},
            "rerun": rr, "frozen": ff, "rerun_mean_gain": round(rerun_mean, 3), "frozen_mean_gain": round(frozen_mean, 3),
            "rerun_mean_clean": round(float(np.mean([r["clean"] for r in rr])), 3),
            "frozen_mean_clean": round(float(np.mean([r["clean"] for r in ff])), 3),
            "reuse_ratio": None if ratio is None else round(ratio, 4), "ratio_bar": RATIO_BAR,
            "rerun_clean_ok": clean_ok, "clean_ok_needed": CLEAN_OK_NEEDED, "rerun_all_within_limits": within,
            "frozen_clean_ok": sum(r["clean"] >= CLEAN_FLOOR for r in ff),
            "frozen_all_within_limits": all(r["within_limits"] for r in ff),
            "C_supported": supported, "frozen_share": None if share is None else round(share, 4),
            "frozen_reading": share_reading}


def cmd_summary(args: argparse.Namespace) -> int:
    from weightcode.harness.jsonio import file_sha256, read_json, write_json

    base = Path(args.reads) if args.reads else out_root() / "read"
    result: dict[str, Any] = {"schema": "weightcode.e12c.summary.v0", "conditions": {}, "files": {}}
    for condition, final in FINALS.items():
        program, _ = final_program(condition)
        digest = file_sha256(program)
        rerun = [read_json(p) for p in sorted((base / condition).glob(f"rerun_{digest[:12]}_*.json"))]
        frozen = [read_json(p) for p in sorted((base / condition).glob("frozen_*.json"))]
        own = read_json(runs_root() / "devreadouts" / "learn" / final["session"] / "transfer" / final["own_record"])
        result["conditions"][condition] = condition_table(rerun, frozen, own)
        for p in sorted((base / condition).glob("*.json")):
            result["files"][f"{condition}/{p.name}"] = file_sha256(p)
    for condition, t in result["conditions"].items():
        print(f"== {condition}: own {t['own']['gain']:+.3f} / {t['own']['clean']:+.3f}")
        print("   seed instance   rerun gain  clean  within | frozen gain  clean  within")
        for r, f in zip(t["rerun"], t["frozen"]):
            print(f"   {r['seed']}  {r['instance_id']}   {r['gain']:+9.3f} {r['clean']:+7.3f}  {str(r['within_limits']):5s}  |"
                  f" {f['gain']:+9.3f} {f['clean']:+7.3f}  {f['within_limits']}")
        print(f"   rerun mean gain {t['rerun_mean_gain']:+.3f} (clean {t['rerun_mean_clean']:+.3f}), ratio "
              f"{t['reuse_ratio']} (bar {t['ratio_bar']}), clean >= {CLEAN_FLOOR} on {t['rerun_clean_ok']}/5 "
              f"(need {t['clean_ok_needed']}), all within limits {t['rerun_all_within_limits']} -> C supported: "
              f"{t['C_supported']}")
        print(f"   frozen mean gain {t['frozen_mean_gain']:+.3f} (clean {t['frozen_mean_clean']:+.3f}), share "
              f"{t['frozen_share']} -> {t['frozen_reading']}")
    if args.out:
        write_json(Path(args.out), result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    programs = sub.add_parser("programs", help="run the final on the targets in the local program sandbox")
    programs.add_argument("--condition", choices=sorted(FINALS), required=True)
    programs.add_argument("--instance", nargs="+", help="only these target instances (default: source and five targets)")
    programs.add_argument("--sandbox-python", help="the sandbox interpreter (default: the harness's .venv-sandbox of this "
                                                   "checkout; a worktree has none, so name the main tree's)")
    listing = sub.add_parser("targets", help="the five target instance ids of a condition")
    listing.add_argument("--condition", choices=sorted(FINALS), required=True)
    read = sub.add_parser("read", help="child and readout of one rerun or frozen edit")
    read.add_argument("--kind", choices=("rerun", "frozen"), required=True)
    read.add_argument("--condition", choices=sorted(FINALS), required=True)
    read.add_argument("--instance", required=True)
    read.add_argument("--out", required=True, help="directory for the read record (one per condition)")
    read.add_argument("--batch-size", type=int)
    summary = sub.add_parser("summary", help="the tables and the readings fixed in the entry")
    summary.add_argument("--reads", help="directory holding <condition>/ read records (default runs/e12/c/read)")
    summary.add_argument("--out")
    args = parser.parse_args(argv)
    return {"programs": cmd_programs, "read": cmd_read, "summary": cmd_summary, "targets": cmd_targets}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
