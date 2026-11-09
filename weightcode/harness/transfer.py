"""Run one frozen LEARN update program on other students (HARNESS_SPEC section 5; ``scripts/run_program_transfer.py``).

For each target dev instance the student and its experience are assembled exactly as LEARN trial assembly does
(``weightcode.harness.experience.build_experience``; the visible ``model_spec.json``, ``forward.py``, the input files and
``model.npz`` of the real weights), the program runs once in the same sandbox driver with the same wall limit
(``weightcode.harness.program.run_update``), its deltas become the derived patch, and the patch is applied to the target
under the LEARN limits of the target's task (``applied_limits(..., "LEARN")``). The record says whether the update
was within the limits. If the limits refuse it, the refusal (the executor's full messages) is recorded and the
unbounded child (the same deltas without ``max_rel_frob`` and ``max_changed_frac``) is read out instead, marked
``readout_child: "unbounded"``. Readouts come from ``weightcode.harness.clean.dev_readout_joint``: V_select of the target's
condition and its clean companion. Nothing reads V_check, the sealed store or an editor session; main instances are
refused. One JSON file per (program, target) run is written into the output directory, never overwritten.

A fewshot-v1 target (PREREG E2, an ``f`` + 8 hex id) gets its own evidence file as experience
(``weightcode.harness.fewshot.experience``) and the fewshot readout on its V_select (``weightcode.harness.fewshot.readout``):
``readouts`` over all rows, ``readouts_new`` and ``readouts_old`` over the new-class and old-class rows, and
``readouts_fewshot``, the full readout. It must be current (``check_fewshot_current``). A seqskill-v1 target
(PREREG E5, a ``q`` + 8 hex id) likewise gets its own evidence file (``weightcode.harness.seqskill.experience``) and the
seqskill readout on its V_select: ``readouts`` is part i (the new operator alone), ``readouts_old`` part iv (old
operators) and ``readouts_seqskill`` the full readout. It must be current (``check_seqskill_current``). A graphnew-v1
target (PREREG E7, a ``g`` + 8 hex id) likewise gets its own evidence file (``weightcode.harness.graphnew.experience``),
graph.npz beside its forward.py, the graphnew limits (``weightcode.harness.graphnew.program_limits``, equal to the LEARN
limits of its task) and the graphnew readout on its V_select (``weightcode.harness.graphnew.readout``): ``readouts``,
``readouts_new`` and ``readouts_old`` as for fewshot-v1, ``readouts_graphnew`` the full readout, and
``student_norms``, the Frobenius norm of each graph_conv weight of the target's student. It must be current
(``check_graphnew_current``). An fscil-v1 target (PREREG E9, a ``c`` + 8 hex id: one session on one student) gets
its own evidence file in three parts (``weightcode.harness.fscil.experience``), the fscil limits
(``weightcode.harness.fscil.program_limits``) and the fscil readout under its session mask (``weightcode.harness.fscil.readout``):
``readouts`` over all seen classes, ``readouts_new`` over the session's classes, ``readouts_prev`` over the earlier
incremental classes, ``readouts_old`` over the base classes and ``readouts_fscil`` the full readout. It must be current
(``check_fscil_current``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import params_sha256, save_params
from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
from weightcode.harness import clean, fewshot, fscil, graphnew, seqskill
from weightcode.harness.arms import applied_limits, limits_dict
from weightcode.harness.experience import EXPERIENCE_FILE, build_experience
from weightcode.harness.jsonio import file_sha256, json_text, utc_now, write_json
from weightcode.harness.program import DERIVED_PATCH, ProgramProblem, Student, program_name, run_update, write_derived_patch
from weightcode.harness.trial import input_file_names
from weightcode.patch import Limits, PatchError, apply_patch
from weightcode.patch.limits import dense_delta_bytes
from weightcode.sandbox.program import PROGRAM_WALL_S, ProgramSandbox

TRANSFER_SCHEMA = "weightcode.harness.program_transfer.v0"
LEARN_ARM = "LEARN"
CHILD_BOUNDED = "within_limits"
CHILD_UNBOUNDED = "unbounded"
STATUS_OK = "ok"
STATUS_PROGRAM_FAILED = "program_failed"
STATUS_PATCH_FAILED = "patch_failed"


class TransferError(RuntimeError):
    """A target cannot be used (not a dev instance, or its files are inconsistent)."""


def stage_student(instance: Instance, dest: Path, params: Mapping[str, np.ndarray] | None = None,
                  experience: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None,
                  ) -> tuple[Student, Path, dict[str, Any]]:
    """Write the target's student files and experience into the new directory ``dest``, as trial assembly does.

    ``params`` replaces the instance's weights (a stream round's current student, ``weightcode.harness.stream``) and
    ``experience`` the experience assembled from the instance's evidence (``build_experience``, the default).
    """
    if instance.private.get("split_role") != "dev":
        raise TransferError(f"{instance.instance_id} has split_role {instance.private.get('split_role')!r}; "
                            "program transfer runs on dev instances only")
    dest.mkdir(mode=0o700)
    spec = json.loads((instance.instance_dir / "model_spec.json").read_text(encoding="utf-8"))
    if file_sha256(instance.instance_dir / "model_spec.json") != instance.private["model_spec_sha256"]:
        raise TransferError(f"{instance.instance_id}: model_spec.json differs from private.json")
    layout = {key: value for key, value in spec.items() if key != "model_id"}
    (dest / "model_spec.json").write_text(json_text(layout), encoding="utf-8")
    weights = instance.params if params is None else params
    expected = instance.params_sha256 if params is None else params_sha256(weights)
    if save_params(dest / "model.npz", weights) != expected:
        raise TransferError(f"{instance.instance_id}: the saved weights do not hash to the student")
    shutil.copyfile(instance.instance_dir / "forward.py", dest / "forward.py")
    if file_sha256(dest / "forward.py") != instance.private["forward_sha256"]:
        raise TransferError(f"{instance.instance_id}: forward.py differs from private.json forward_sha256")
    inputs = input_file_names(spec)
    for name in inputs:
        shutil.copyfile(instance.instance_dir / name, dest / name)
    if graphnew.is_graphnew(instance.private):  # forward.py reads this copy beside itself
        if inputs != (graphnew.GRAPH_FILE,) or file_sha256(dest / graphnew.GRAPH_FILE) != instance.private[
                "graph_file_sha256"]:
            raise TransferError(f"{instance.instance_id}: the {graphnew.GRAPH_FILE} copy differs from private.json "
                                "graph_file_sha256")
    if experience is None:
        if fewshot.is_fewshot(instance.private):
            experience = fewshot.experience(instance)
        elif seqskill.is_seqskill(instance.private):
            experience = seqskill.experience(instance)
        elif graphnew.is_graphnew(instance.private):
            experience = graphnew.experience(instance)
        elif fscil.is_fscil(instance.private):
            experience = fscil.experience(instance)
        else:
            experience = build_experience(instance)
    experience, record = experience
    np.savez(dest / EXPERIENCE_FILE, **experience)
    with np.load(dest / EXPERIENCE_FILE, allow_pickle=False) as archive:
        if params_sha256({key: archive[key] for key in archive.files}) != record["content_sha256"]:
            raise TransferError(f"{instance.instance_id}: the written experience differs from the assembled one")
    student = Student(params=weights, params_file=dest / "model.npz", forward_file=dest / "forward.py",
                      input_files={name: dest / name for name in inputs}, layout=layout)
    return student, dest / EXPERIENCE_FILE, record


def _child(instance: Instance, patch: dict[str, Any], patch_dir: Path, limits: Limits) -> dict[str, Any]:
    """Apply the derived patch within ``limits``; on a refusal, record it and apply it without the size limits."""
    parent_sha = instance.params_sha256
    try:
        result = apply_patch(instance.params, patch, patch_dir, limits, parent_sha)
        return {"within_limits": True, "limit_messages": [], "readout_child": CHILD_BOUNDED, "result": result}
    except PatchError as refusal:
        messages = list(refusal.messages)
    unbounded = dataclasses.replace(limits, max_rel_frob=None, max_changed_frac=None)
    result = apply_patch(instance.params, patch, patch_dir, unbounded, parent_sha)
    return {"within_limits": False, "limit_messages": messages, "readout_child": CHILD_UNBOUNDED, "result": result,
            "unbounded_limits": limits_dict(unbounded)}


def transfer_one(program: Path, instance: Instance, work: Path, sandbox: ProgramSandbox,
                 batch_size: int = DEFAULT_BATCH_SIZE, wall_s: float = PROGRAM_WALL_S) -> dict[str, Any]:
    """Run ``program`` on one target and return its record (module docstring); ``work`` is created."""
    work.mkdir(mode=0o700)
    student, experience_file, experience = stage_student(instance, work / "student")
    gnew = graphnew.is_graphnew(instance.private)
    fsc = fscil.is_fscil(instance.private)
    try:
        limits = (graphnew.program_limits(instance.task["limits"], len(student.params)) if gnew
                  else fscil.program_limits(instance.task["limits"], len(student.params)) if fsc
                  else applied_limits(instance.task["limits"], LEARN_ARM, len(student.params),
                                      dense_delta_bytes(v.shape for v in student.params.values())))
    except (graphnew.GraphnewHarnessError, fscil.FscilHarnessError) as exc:
        raise TransferError(f"{instance.instance_id}: {exc}") from None
    record: dict[str, Any] = {
        "schema": TRANSFER_SCHEMA,
        "created_at": utc_now(),
        "program": {"path": str(program), "name": program.name, "sha256": file_sha256(program)},
        "target": {"instance_id": instance.instance_id, "dataset": instance.private.get("dataset"),
                   "condition": instance.private.get("condition"), "split_role": instance.private.get("split_role"),
                   "model_id": instance.private.get("model_id"), "params_sha256": instance.params_sha256},
        "experience": experience,
        "sandbox": sandbox.as_dict(),
        "wall_s": wall_s,
        "limits": limits_dict(limits),
        "work_dir": str(work),
    }
    try:
        update = run_update(student, experience_file, program, work / "run", sandbox, wall_s)
    except ProgramProblem as problem:
        record.update(status=STATUS_PROGRAM_FAILED, messages=problem.messages, run=problem.run)
        return record
    record.update(run=update.record, deltas_sha256=update.record["deltas_sha256"], tensors=update.tensors,
                  hypothesis=update.hypothesis, predicted_effect=update.predicted_effect)
    patch_dir = work / "patch"
    patch_dir.mkdir()
    patch = write_derived_patch(update, instance.params, instance.params_sha256, patch_dir)
    try:
        applied = _child(instance, patch, patch_dir, limits)
    except PatchError as problem:  # a format limit (op count, file bytes) refuses even the unbounded child
        record.update(status=STATUS_PATCH_FAILED, messages=list(problem.messages),
                      patch_sha256=file_sha256(patch_dir / DERIVED_PATCH))
        return record
    result = applied.pop("result")
    record.update(status=STATUS_OK, **applied, patch_sha256=file_sha256(patch_dir / DERIVED_PATCH),
                  child_sha256=result.child_sha256, n_changed=result.n_changed, edit_frac=result.edit_frac,
                  changed_frac=result.changed_frac)
    if fewshot.is_fewshot(instance.private):
        parts = fewshot.readout(instance, result.child, batch_size)
        record.update(n_select=parts["parts"]["all"]["n"], readouts=parts["parts"]["all"],
                      readouts_new=parts["parts"]["new"], readouts_old=parts["parts"]["old"],
                      readouts_fewshot=parts["readouts_fewshot"], readout_source=fewshot.FEWSHOT_RULE)
        return record
    if gnew:
        parts = graphnew.readout(instance, result.child)
        record.update(n_select=parts["parts"]["all"]["n"], readouts=parts["parts"]["all"],
                      readouts_new=parts["parts"]["new"], readouts_old=parts["parts"]["old"],
                      readouts_graphnew=parts["readouts_graphnew"], readout_source=graphnew.GRAPHNEW_RULE,
                      student_norms=graphnew.student_norms(instance.params, student.layout))
        return record
    if fsc:
        parts = fscil.readout(instance, result.child)
        record.update(n_select=parts["parts"]["seen"]["n"], readouts=parts["parts"]["seen"],
                      readouts_new=parts["parts"]["new"], readouts_prev=parts["parts"]["prev"],
                      readouts_old=parts["parts"]["old"], readouts_fscil=parts["readouts_fscil"],
                      readout_source=fscil.FSCIL_RULE)
        return record
    if seqskill.is_scan(instance.private):
        raise TransferError(f"{instance.instance_id}: PREREG E13 defines no transfer of a program to a scanskill-v1 "
                            "instance")
    if seqskill.is_seqskill(instance.private):
        parts = seqskill.readout(instance, result.child, batch_size)
        record.update(n_select=parts["parts"]["i"]["n"], readouts=parts["parts"]["i"],
                      readouts_old=parts["parts"]["iv"], readouts_seqskill=parts["readouts_seqskill"],
                      readout_source=seqskill.SEQSKILL_RULE)
        return record
    readout = clean.dev_readout_joint(instance, result.child, batch_size)
    record.update(n_select=readout["n"], readouts=readout["readouts"], n_clean=readout["n_clean"],
                  readouts_clean=readout["readouts_clean"], clean_split_sha256=readout["clean_split_sha256"],
                  readout_source=clean.source_name())
    return record


def run_transfer(program: Path, instance_ids: Sequence[str], out_dir: Path, sandbox: ProgramSandbox,
                 batch_size: int = DEFAULT_BATCH_SIZE, wall_s: float = PROGRAM_WALL_S) -> list[Path]:
    """One JSON record per target in ``out_dir``; an existing record is never overwritten."""
    program = Path(program).resolve()
    program_name(program.name)
    if not program.is_file() or program.is_symlink():
        raise TransferError(f"{program} is not a regular program file")
    digest = file_sha256(program)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for instance_id in instance_ids:
        stem = f"transfer_{digest[:12]}_{instance_id}"
        path = out_dir / f"{stem}.json"
        if path.exists() or (out_dir / stem).exists():
            raise TransferError(f"{path} or its work directory exists already; records are never overwritten")
        if fewshot.is_fewshot_id(instance_id):
            try:
                fewshot.check_current(fewshot.instance_dir(instance_id))
            except fewshot.stale_error() as exc:
                raise TransferError(f"{instance_id}: {exc}") from None
        if seqskill.is_seqskill_id(instance_id):
            try:
                seqskill.check_current(seqskill.instance_dir(instance_id))
            except seqskill.stale_error() as exc:
                raise TransferError(f"{instance_id}: {exc}") from None
            instance = load_instance(seqskill.instance_dir(instance_id))
        elif graphnew.is_graphnew_id(instance_id):
            try:
                graphnew.check_current(graphnew.instance_dir(instance_id))
            except (graphnew.stale_error(), graphnew.GraphnewHarnessError) as exc:
                raise TransferError(f"{instance_id}: {exc}") from None
            instance = load_instance(graphnew.instance_dir(instance_id))
        elif fscil.is_fscil_id(instance_id):
            try:
                fscil.check_current(fscil.instance_dir(instance_id))
            except (fscil.stale_error(), fscil.FscilHarnessError) as exc:
                raise TransferError(f"{instance_id}: {exc}") from None
            instance = load_instance(fscil.instance_dir(instance_id))
        else:
            instance = fewshot.load_any(instance_id)
        record = transfer_one(program, instance, out_dir / stem, sandbox, batch_size, wall_s)
        if record["program"]["sha256"] != digest:
            raise TransferError(f"{program} changed during the transfer runs")
        write_json(path, record)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, DEFAULT_THREADS, SANDBOX_PYTHON

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--program", type=Path, required=True, help="the final program file of a LEARN session")
    parser.add_argument("--instances", nargs="+", required=True, help="target dev instance ids")
    parser.add_argument("--out", type=Path, required=True, help="directory for one JSON record per run")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--memory-bytes", type=int, default=DEFAULT_MEMORY_BYTES)
    parser.add_argument("--sandbox-python", type=Path, default=SANDBOX_PYTHON)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    sandbox = ProgramSandbox(python=args.sandbox_python, memory_bytes=args.memory_bytes, threads=args.threads)
    for path in run_transfer(args.program, args.instances, args.out, sandbox, args.batch_size):
        record = json.loads(path.read_text(encoding="utf-8"))
        line = {"record": str(path), "target": record["target"]["instance_id"], "status": record["status"],
                "within_limits": record.get("within_limits"),
                "delta_acc_select": (record.get("readouts") or {}).get("delta_acc")}
        if "readouts_seqskill" in record:
            line.update(delta_em_old=record["readouts_old"]["delta_acc"],
                        delta_em_ii=record["readouts_seqskill"]["ii"]["delta"],
                        delta_em_iii=record["readouts_seqskill"]["iii"]["delta"])
        elif "readouts_graphnew" in record:
            line.update(delta_acc_new=record["readouts_new"]["delta_acc"], delta_acc_old=record["readouts_old"]["delta_acc"],
                        delta_new_balanced=record["readouts_graphnew"]["delta_new_balanced"])
        elif "readouts_fscil" in record:
            line.update(delta_acc_new=record["readouts_new"]["delta_acc"], delta_acc_old=record["readouts_old"]["delta_acc"],
                        delta_bal_seen=record["readouts_fscil"]["delta_bal_seen"])
        elif "readouts_fewshot" in record:
            line.update(delta_acc_new=record["readouts_new"]["delta_acc"], delta_acc_old=record["readouts_old"]["delta_acc"])
        else:
            line["delta_acc_clean"] = (record.get("readouts_clean") or {}).get("delta_acc")
        print(json.dumps(line))
    return 0


__all__ = ["TRANSFER_SCHEMA", "TransferError", "main", "run_transfer", "stage_student", "transfer_one"]
