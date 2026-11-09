"""Run a stream of frozen LEARN update programs on one student, round after round (PREREG E4, HARNESS_SPEC section 5;
``scripts/run_program_stream.py``).

A stream is a list of rounds, each one program and one dev instance. All instances of a stream share one base model,
the student before round 1, and no instance appears twice. Round r works on the current student, the one left by
round r - 1:

1. Experience (``stream_experience``): rule evidence-v1 drawn for the current student on round r's condition
   (``weightcode.evalkit.student_evidence.draw_evidence_for``: 16 rows from this student's own errors on that V_evidence and
   48 random rows), the dataset's replay rows (rule replay-v1), and the current student's predictions on both. On the
   base model this is exactly the experience a LEARN trial and a transfer run get (``build_experience``); the round
   records whether it is.
2. The program runs once, as in a transfer run (``weightcode.harness.transfer.stage_student`` with the current weights, then
   ``weightcode.harness.program.run_update``, same sandbox and wall limit).
3. Its deltas become the derived patch on the current student, applied under the LEARN limits of round r's instance
   and measured against the current student. A refused draw, a failed program and a refused patch each leave the
   student unchanged; the round records which happened and the messages.
4. Every instance of the stream is read out on the new student (``weightcode.harness.clean.dev_readout_joint``: V_select of
   its condition and its clean companion, each against the base model), whether its round has come yet or not.

The summary gives, for each round's condition, the gain right after its own round, the gain after the last round and
the forgetting between the two (the first minus the second), the mean gain after the last round over the conditions
not in ``mean_excludes``, and the clean change after the last round. Gains and changes are fractions (delta_acc).
Nothing reads V_check, the sealed store or an editor session; main instances are refused. One JSON record per stream
is written, never overwritten; each round's student is saved in the stream's work directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import count_changed, params_sha256, save_params
from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.student_evidence import draw_evidence_for
from weightcode.harness import clean
from weightcode.harness.arms import applied_limits, limits_dict
from weightcode.harness.experience import EXPERIENCE_FILE, PREDICT_BATCH, build_experience
from weightcode.harness.jsonio import file_sha256, utc_now, write_json
from weightcode.harness.program import DERIVED_PATCH, ProgramProblem, program_name, run_update, write_derived_patch
from weightcode.harness.transfer import LEARN_ARM, stage_student
from weightcode.patch import PatchError, apply_patch
from weightcode.patch.limits import dense_delta_bytes
from weightcode.sandbox.program import PROGRAM_WALL_S, ProgramSandbox

STREAM_SCHEMA = "weightcode.harness.program_stream.v0"
STREAM_EXPERIENCE_SOURCE = "student_evidence_v1_and_replay_v1"
STATUS_OK = "ok"
STATUS_EVIDENCE_REFUSED = "evidence_refused"
STATUS_PROGRAM_FAILED = "program_failed"
STATUS_PATCH_REFUSED = "patch_refused"
# PREREG E4: the stream's mean gain is taken over the four conditions other than colormix_d1, on which the base model
# loses only about 3.5 points (its bar is retention).
E4_MEAN_EXCLUDES = ("colormix_d1",)
READOUT_FIELDS = ("base_acc", "child_acc", "delta_acc", "fixed", "broken", "worst_class_delta")


class StreamError(RuntimeError):
    """A stream cannot run as specified (its instances, programs or experience are inconsistent)."""


def stream_experience(instance: Instance, params: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray],
                                                                                   dict[str, Any]]:
    """The experience of the student ``params`` on the condition of ``instance`` (module docstring, step 1).

    Raises ``ValueError`` (``StudentEvidenceError`` among them) when evidence-v1 has no draw for this student.
    """
    drawn = draw_evidence_for(instance, dict(params))
    new = {
        "x_new": np.ascontiguousarray(drawn.x),
        "y_new": np.ascontiguousarray(drawn.y.astype(np.int64)),
        "pred_new": np.ascontiguousarray(drawn.pred.astype(np.int64)),
        "split_ids_new": np.ascontiguousarray(np.asarray(drawn.arrays["split_ids"])),
    }
    replay, record = clean.load_replay(str(instance.private["dataset"]))
    x_old = replay["x"]
    if x_old.shape[1:] != new["x_new"].shape[1:] or x_old.dtype != new["x_new"].dtype:
        raise StreamError(f"replay inputs are {x_old.dtype}{x_old.shape[1:]}, the new-condition inputs "
                          f"{new['x_new'].dtype}{new['x_new'].shape[1:]}")
    shared = sorted(set(int(i) for i in replay["idx"]) & set(int(i) for i in drawn.source_idx))
    if shared:
        raise StreamError(f"{instance.instance_id}: replay rows share dataset idx {shared[:10]} with the drawn "
                          "evidence rows; a clean view of an evidence image must never reach a program")
    if set(replay["split_ids"].tolist()) & set(new["split_ids_new"].tolist()):
        raise StreamError("a replay id equals an evidence id")
    predicted = predict(instance.forward, params, new["x_new"], PREDICT_BATCH)
    if not np.array_equal(predicted, new["pred_new"]):
        raise StreamError(f"pred_new disagrees with the student on {int(np.sum(predicted != new['pred_new']))} rows")
    experience = {**new, "x_old": x_old, "y_old": replay["y"].astype(np.int64),
                  "pred_old": predict(instance.forward, params, x_old, PREDICT_BATCH),
                  "split_ids_old": replay["split_ids"]}
    summary = {
        "file": EXPERIENCE_FILE, "source": STREAM_EXPERIENCE_SOURCE, "content_sha256": params_sha256(experience),
        "n_new": int(len(new["y_new"])), "n_new_fail": drawn.n_fail, "n_old": int(len(x_old)),
        "student_params_sha256": drawn.params_sha256, "evidence_seed": int(drawn.seed),
        "evidence_content_sha256": drawn.content_sha256, "n_errors_vevidence": drawn.n_errors_vevidence,
        "replay": {**record, "arrays_sha256": params_sha256(replay), "source": clean.source_name()},
        "replay_idx_disjoint_from_evidence_idx": True,
    }
    return experience, summary


def _readout(instance: Instance, child: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
    record = clean.dev_readout_joint(instance, child, batch_size)
    return {
        "condition": instance.private.get("condition"),
        "select": {name: record["readouts"].get(name) for name in READOUT_FIELDS},
        "clean": {name: record["readouts_clean"].get(name) for name in READOUT_FIELDS},
        "n_select": record["n"], "n_clean": record["n_clean"], "clean_split_sha256": record["clean_split_sha256"],
    }


def _load_stream(instance_ids: Sequence[str]) -> list[Instance]:
    if not instance_ids:
        raise StreamError("a stream needs at least one round")
    if len(set(instance_ids)) != len(instance_ids):
        raise StreamError(f"an instance appears twice in the stream: {list(instance_ids)}")
    instances = [load_instance(instances_root() / instance_id) for instance_id in instance_ids]
    for instance in instances:
        if instance.private.get("split_role") != "dev":
            raise StreamError(f"{instance.instance_id} has split_role {instance.private.get('split_role')!r}; streams "
                              "run on dev instances only")
    bases = {instance.params_sha256 for instance in instances}
    if len(bases) != 1:
        raise StreamError(f"the stream's instances have {len(bases)} different base models; a stream has one student")
    return instances


def _program(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    program_name(resolved.name)
    if not resolved.is_file() or resolved.is_symlink():
        raise StreamError(f"{path} is not a regular program file")
    return {"path": str(resolved), "name": resolved.name, "sha256": file_sha256(resolved)}


def _round(index: int, program: Path, instance: Instance, params: dict[str, np.ndarray], work: Path,
           sandbox: ProgramSandbox, wall_s: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One round on the student ``params``; returns the next student and the round record (steps 1 to 3)."""
    work.mkdir(mode=0o700)
    parent_sha = params_sha256(params)
    limits = applied_limits(instance.task["limits"], LEARN_ARM, len(params),
                            dense_delta_bytes(v.shape for v in params.values()))
    record: dict[str, Any] = {"round": index, "instance_id": instance.instance_id,
                              "condition": instance.private.get("condition"), "parent_sha256": parent_sha,
                              "program": {"name": program.name, "sha256": file_sha256(program)},
                              "limits": limits_dict(limits)}
    try:
        experience = stream_experience(instance, params)
    except ValueError as refusal:  # evidence-v1 has no draw for this student (StudentEvidenceError is a ValueError)
        record.update(status=STATUS_EVIDENCE_REFUSED, messages=[str(refusal)], child_sha256=parent_sha)
        return params, record
    if parent_sha == instance.params_sha256:
        experience[1]["same_as_instance_experience"] = (
            experience[1]["content_sha256"] == build_experience(instance)[1]["content_sha256"])
    student, experience_file, summary = stage_student(instance, work / "student", params=params,
                                                      experience=experience)
    record["experience"] = summary
    try:
        update = run_update(student, experience_file, program, work / "run", sandbox, wall_s)
    except ProgramProblem as problem:
        record.update(status=STATUS_PROGRAM_FAILED, messages=problem.messages, run=problem.run, child_sha256=parent_sha)
        return params, record
    record.update(run=update.record, deltas_sha256=update.record["deltas_sha256"], tensors=update.tensors,
                  hypothesis=update.hypothesis, predicted_effect=update.predicted_effect)
    patch_dir = work / "patch"
    patch_dir.mkdir()
    patch = write_derived_patch(update, params, parent_sha, patch_dir)
    record["patch_sha256"] = file_sha256(patch_dir / DERIVED_PATCH)
    try:
        result = apply_patch(params, patch, patch_dir, limits, parent_sha)
    except PatchError as refusal:
        record.update(status=STATUS_PATCH_REFUSED, messages=list(refusal.messages), child_sha256=parent_sha)
        return params, record
    child = {name: np.array(value) for name, value in result.child.items()}
    if save_params(work / "student_after.npz", child) != result.child_sha256:
        raise StreamError(f"round {index}: the saved student does not hash to the child")
    record.update(status=STATUS_OK, child_sha256=result.child_sha256, n_changed=result.n_changed,
                  changed_frac=result.changed_frac)
    return child, record


def stream_summary(instances: Sequence[Instance], rounds: Sequence[Mapping[str, Any]],
                   mean_excludes: Sequence[str]) -> dict[str, Any]:
    """Per-condition gains and forgetting, the mean gain and the clean change after the last round (docstring)."""
    last = rounds[-1]["readouts"]
    conditions: dict[str, Any] = {}
    for index, instance in enumerate(instances):
        own = rounds[index]["readouts"][instance.instance_id]["select"]["delta_acc"]
        final = last[instance.instance_id]["select"]["delta_acc"]
        conditions[str(instance.private.get("condition"))] = {
            "round": index + 1, "instance_id": instance.instance_id, "round_status": rounds[index]["status"],
            "base_acc": last[instance.instance_id]["select"]["base_acc"],
            "gain_after_own_round": own, "gain_after_last_round": final, "forgetting": own - final,
        }
    included = [name for name in conditions if name not in set(mean_excludes)]
    earlier = [name for name, row in conditions.items() if row["round"] < len(instances)]
    clean_rows = {instance.instance_id: last[instance.instance_id]["clean"]["delta_acc"] for instance in instances}
    clean_splits = {last[instance.instance_id]["clean_split_sha256"] for instance in instances}
    return {
        "conditions": conditions,
        "mean_excludes": list(mean_excludes),
        "mean_gain_after_last_round": (float(np.mean([conditions[name]["gain_after_last_round"] for name in included]))
                                       if included else None),
        "max_forgetting": max((conditions[name]["forgetting"] for name in earlier), default=None),
        "clean_delta_acc_after_last_round": clean_rows,
        "clean_split_shared": len(clean_splits) == 1,
        "clean_delta_acc_mean_after_last_round": float(np.mean(list(clean_rows.values()))),
        "n_rounds_ok": sum(1 for row in rounds if row["status"] == STATUS_OK),
    }


def run_stream(programs: Sequence[Path], instance_ids: Sequence[str], out_dir: Path, sandbox: ProgramSandbox,
               batch_size: int = DEFAULT_BATCH_SIZE, wall_s: float = PROGRAM_WALL_S,
               mean_excludes: Sequence[str] = E4_MEAN_EXCLUDES) -> Path:
    """Run the stream (module docstring) and write its record into ``out_dir``; an existing record is never replaced."""
    if len(programs) != len(instance_ids):
        raise StreamError(f"{len(programs)} programs for {len(instance_ids)} rounds; a stream has one program per round")
    shown = [_program(Path(path)) for path in programs]
    instances = _load_stream(instance_ids)
    base = {name: np.array(value) for name, value in instances[0].params.items()}
    base_sha = instances[0].params_sha256
    digest = hashlib.sha256(json.dumps([[item["sha256"] for item in shown], list(instance_ids)]).encode()).hexdigest()
    stem = f"stream_{digest[:12]}_{base_sha[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path, work = out_dir / f"{stem}.json", out_dir / stem
    if path.exists() or work.exists():
        raise StreamError(f"{path} or its work directory exists already; records are never overwritten")
    work.mkdir(mode=0o700)
    n_params = sum(value.size for value in base.values())
    params, rounds = base, []
    for index, (item, instance) in enumerate(zip(shown, instances), start=1):
        params, record = _round(index, Path(item["path"]), instance, params, work / f"round_{index}", sandbox, wall_s)
        changed = count_changed(base, params)
        record.update(n_changed_vs_base=changed, changed_frac_vs_base=changed / n_params,
                      readouts={other.instance_id: _readout(other, params, batch_size) for other in instances})
        rounds.append(record)
    for item in shown:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise StreamError(f"{item['path']} changed while the stream ran")
    stream = {
        "schema": STREAM_SCHEMA, "created_at": utc_now(), "programs": shown,
        "instances": [{"round": index, "instance_id": instance.instance_id,
                       "dataset": instance.private.get("dataset"), "condition": instance.private.get("condition"),
                       "model_id": instance.private.get("model_id"), "split_role": instance.private.get("split_role")}
                      for index, instance in enumerate(instances, start=1)],
        "base_params_sha256": base_sha, "sandbox": sandbox.as_dict(), "wall_s": wall_s, "batch_size": batch_size,
        "readout_source": clean.source_name(), "work_dir": str(work), "rounds": rounds,
        "summary": stream_summary(instances, rounds, mean_excludes),
    }
    write_json(path, stream)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, DEFAULT_THREADS, SANDBOX_PYTHON

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--programs", type=Path, nargs="+", required=True, help="the frozen programs, in round order")
    parser.add_argument("--instances", nargs="+", required=True,
                        help="one dev instance per round, in round order, all with the same base model")
    parser.add_argument("--out", type=Path, required=True, help="directory for the stream record")
    parser.add_argument("--mean-excludes", nargs="*", default=list(E4_MEAN_EXCLUDES),
                        help="conditions left out of the mean gain (PREREG E4: colormix_d1)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--memory-bytes", type=int, default=DEFAULT_MEMORY_BYTES)
    parser.add_argument("--sandbox-python", type=Path, default=SANDBOX_PYTHON)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    sandbox = ProgramSandbox(python=args.sandbox_python, memory_bytes=args.memory_bytes, threads=args.threads)
    path = run_stream(args.programs, args.instances, args.out, sandbox, args.batch_size,
                      mean_excludes=args.mean_excludes)
    record = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"record": str(path), "rounds": [row["status"] for row in record["rounds"]],
                      "summary": record["summary"]}, indent=1))
    return 0


__all__ = ["E4_MEAN_EXCLUDES", "STREAM_SCHEMA", "StreamError", "main", "run_stream", "stream_experience",
           "stream_summary"]
