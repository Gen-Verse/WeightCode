"""PREREG F6: a LEARN session's final edit, frozen, added unchanged to other students and read as E3 reads a rerun.

E3 (``weightcode.harness.transfer``) reruns a session's final program on each target student, so the program recomputes its
edit on that network. F6 separates that from the edit's values: the session's final deltas (``final/deltas.npz``, the
program's output on its own student, W'_100 - W_100) are added unchanged to each target and read the same way.

The source (``session_edit``). A complete standard LEARN session (session.json arm LEARN, no ``e11_variant``, the
variant views E3 refuses) whose run.json final is a checked program on its instance. Checked before any target:

* ``final/deltas.npz`` hashes to run.json ``final.program.deltas_sha256`` and to final/program_run.json's record;
* the source instance's base hashes to final/patch.json's ``parent_sha256``;
* the deltas pass ``weightcode.harness.program.load_deltas`` against the base (names, shapes, dtypes, finite; float64);
* model.npz equals the deltas added to the base: float32(float64(base) + delta) on every delta tensor, the base on
  every other tensor, entry for entry, so ``delta = model.npz - base`` at the stored precision. The largest
  |float64(model) - float64(base) - delta| (the float32 rounding) is recorded;
* the executor applying the derived patch under session.json's limits gives a child of run.json's ``child_sha256``.

Per target (``frozen_one``), each step is E3's (``transfer_one``) with the program run left out: the target must be a
dev instance whose E3 readout is the joint one (not a fewshot, seqskill, graphnew or fscil instance); the deltas file
must still hash to the source's digest and passes ``load_deltas`` against the target's weights; ``write_derived_patch``
writes the same ops (the model's tensor order) and float64 files; the target's LEARN limits
(``applied_limits(task limits, "LEARN", n_tensors, dense_delta_bytes)``); E3's ``_child`` applies the patch within
them, or records the refusal and reads the unbounded child; ``weightcode.harness.clean.dev_readout_joint`` reads V_select of
the condition and its clean companion with the instance's forward. The record has E3's readout fields under the same
names. No V_check row is predicted or scored (the store check of ``deployment_splits`` opens the V_check and
V_evidence files for their row indices, as in E3), the sealed store is never opened, and ``devreadout`` refuses main
instances.

``compare`` joins F6 records with E3's records per target and applies PREREG F6's reading, fixed before any number:
the frozen mean gain over the seeds 101 to 105 students as a share of E3's rerun mean gain; <= 0.5 reads "the
program, not its values", >= 0.9 "the values transfer", and in between both are reported. Its seed-100 gate: the
frozen edit on its own student must give E3's seed-100 child and readouts exactly.

Usage::

    python -m weightcode.harness.frozen_edit run --session $WEIGHTCODE_RUNS/sessions/<id> --instances <id> ... --out DIR
    python -m weightcode.harness.frozen_edit compare --f6 DIR --e3 $WEIGHTCODE_RUNS/devreadouts/learn/<id>/transfer
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
from weightcode.harness import clean, fewshot, fscil, graphnew, seqskill
from weightcode.harness.arms import applied_limits, limits_dict
from weightcode.harness.jsonio import file_sha256, read_json, utc_now, write_json
from weightcode.harness.program import DERIVED_PATCH, ProgramProblem, Update, load_deltas, write_derived_patch
from weightcode.harness.transfer import LEARN_ARM, STATUS_OK, STATUS_PATCH_FAILED, TransferError, _child
from weightcode.patch import Limits, PatchError, apply_patch
from weightcode.patch.limits import dense_delta_bytes

FROZEN_SCHEMA = "weightcode.harness.frozen_edit.v0"
PROGRAM_SHARE = 0.5
"""PREREG F6: a frozen mean gain at most this share of the rerun mean gain reads "the program, not its values"."""
VALUES_SHARE = 0.9
"""PREREG F6: a frozen mean gain at least this share of the rerun mean gain reads "the values transfer"."""
READ_PROGRAM = "the program, not its values"
READ_VALUES = "the values transfer"
READ_BOTH = "both reported, neither wording alone"
SOURCE_SEED = 100


class FrozenEditError(TransferError):
    """A source session or a target cannot be used, or the deltas file does not match its record."""


@dataclass(frozen=True)
class FrozenEdit:
    """A session's final deltas, checked against the session's records (module docstring)."""

    session_id: str
    source_instance_id: str
    deltas_path: Path
    deltas_sha256: str
    hypothesis: str
    predicted_effect: str
    record: dict[str, Any]


def _special(private: Mapping[str, Any]) -> str | None:
    for name, test in (("fewshot", fewshot.is_fewshot), ("seqskill", seqskill.is_seqskill),
                       ("graphnew", graphnew.is_graphnew), ("fscil", fscil.is_fscil)):
        if test(private):
            return name
    return None


def _derived_patch(deltas: dict[str, np.ndarray], hypothesis: str, predicted_effect: str, deltas_path: Path,
                   instance: Instance, dest: Path) -> dict[str, Any]:
    update = Update(deltas=deltas, hypothesis=hypothesis, predicted_effect=predicted_effect, record={},
                    deltas_path=deltas_path)
    return write_derived_patch(update, instance.params, instance.params_sha256, dest)


def session_edit(session_dir: Path, work: Path) -> FrozenEdit:
    """The final deltas of the session in ``session_dir``, checked (module docstring); ``work`` is created."""
    root = Path(session_dir)
    session = read_json(root / "session.json")
    run = read_json(root / "run.json")
    if session.get("arm") != LEARN_ARM or session.get("e11_variant"):
        raise FrozenEditError(f"{root.name}: arm {session.get('arm')!r}, variant {session.get('e11_variant')!r}; F6 "
                              "reads the final of a standard LEARN session, as E3 does")
    final = run.get("final") or {}
    program = final.get("program") or {}
    if run.get("status") != "complete" or final.get("is_base_noop") or final.get("parent") != "instance" \
            or not program.get("deltas"):
        raise FrozenEditError(f"{root.name}: the final is not a checked program on the instance")
    deltas_path = root / program["deltas"]
    digest = file_sha256(deltas_path)
    run_record = read_json(root / program["run_record"])
    if not (digest == program.get("deltas_sha256") == run_record.get("deltas_sha256")):
        raise FrozenEditError(f"{deltas_path}: sha256 {digest} differs from run.json final.program.deltas_sha256 "
                              f"{program.get('deltas_sha256')} or program_run.json {run_record.get('deltas_sha256')}")
    patch = read_json(root / program["patch"])
    source = fewshot.load_any(str(session["instance_id"]))
    if source.params_sha256 != patch.get("parent_sha256"):
        raise FrozenEditError(f"{root.name}: the base of {source.instance_id} is not the final patch's parent")
    try:
        deltas = load_deltas(deltas_path, source.params)
    except ProgramProblem as problem:
        raise FrozenEditError(f"{deltas_path}: {problem.messages}") from None
    model = load_params(root / final["model"])
    if set(model) != set(source.params):
        raise FrozenEditError(f"{root / final['model']}: tensor names differ from the base")
    residual = 0.0
    for name, base in source.params.items():
        if name in deltas:
            expected = (base.astype(np.float64) + deltas[name]).astype(np.float32)
            residual = max(residual, float(np.max(np.abs(model[name].astype(np.float64) - base.astype(np.float64)
                                                         - deltas[name]))))
        else:
            expected = base
        if model[name].dtype != np.float32 or not np.array_equal(model[name], expected):
            raise FrozenEditError(f"{root / final['model']}: tensor {name} is not float32(base + delta)")
    work.mkdir(mode=0o700)
    hypothesis, predicted = str(patch.get("hypothesis", "")), str(patch.get("predicted_effect", ""))
    derived = _derived_patch(deltas, hypothesis, predicted, deltas_path, source, work)
    result = apply_patch(source.params, derived, work, Limits(**session["limits"]), source.params_sha256)
    if not (result.child_sha256 == final.get("child_sha256") == params_sha256(model)):
        raise FrozenEditError(f"{root.name}: the executor's child {result.child_sha256} is not run.json's "
                              f"{final.get('child_sha256')}")
    record = {"session_id": root.name, "session_dir": str(root), "source_instance_id": source.instance_id,
              "source_params_sha256": source.params_sha256, "condition": session.get("condition"),
              "deltas": str(deltas_path), "deltas_sha256": digest, "tensors": list(deltas),
              "program": {key: program.get(key) for key in ("file", "sha256")},
              "model_sha256": params_sha256(model), "child_sha256": result.child_sha256,
              "n_changed": result.n_changed, "max_abs_float32_residual": residual,
              "session_limits": dict(session["limits"]), "derived_patch_sha256": file_sha256(work / DERIVED_PATCH)}
    return FrozenEdit(session_id=root.name, source_instance_id=source.instance_id, deltas_path=deltas_path,
                      deltas_sha256=digest, hypothesis=hypothesis, predicted_effect=predicted, record=record)


def frozen_one(edit: FrozenEdit, instance: Instance, work: Path, batch_size: int = DEFAULT_BATCH_SIZE
               ) -> dict[str, Any]:
    """Add the frozen deltas to one target and read it out as E3 does (module docstring); ``work`` is created."""
    if instance.private.get("split_role") != "dev":
        raise FrozenEditError(f"{instance.instance_id} has split_role {instance.private.get('split_role')!r}; F6 runs "
                              "on dev instances only")
    kind = _special(instance.private)
    if kind is not None:
        raise FrozenEditError(f"{instance.instance_id} is a {kind} instance; F6 reads the joint readout only")
    if file_sha256(edit.deltas_path) != edit.deltas_sha256:
        raise FrozenEditError(f"{edit.deltas_path} changed after the source check")
    work.mkdir(mode=0o700)
    params = instance.params
    limits = applied_limits(instance.task["limits"], LEARN_ARM, len(params),
                            dense_delta_bytes(v.shape for v in params.values()))
    record: dict[str, Any] = {
        "schema": FROZEN_SCHEMA,
        "created_at": utc_now(),
        "source": edit.record,
        "target": {"instance_id": instance.instance_id, "dataset": instance.private.get("dataset"),
                   "condition": instance.private.get("condition"), "split_role": instance.private.get("split_role"),
                   "model_id": instance.private.get("model_id"), "seed": instance.private.get("seed"),
                   "params_sha256": instance.params_sha256},
        "limits": limits_dict(limits),
        "work_dir": str(work),
        "deltas_sha256": edit.deltas_sha256,
    }
    try:
        deltas = load_deltas(edit.deltas_path, params)
    except ProgramProblem as problem:
        raise FrozenEditError(f"{instance.instance_id}: the frozen deltas do not fit its weights: "
                              f"{problem.messages}") from None
    record["tensors"] = list(deltas)
    patch_dir = work / "patch"
    patch_dir.mkdir()
    patch = _derived_patch(deltas, edit.hypothesis, edit.predicted_effect, edit.deltas_path, instance, patch_dir)
    try:
        applied = _child(instance, patch, patch_dir, limits)
    except PatchError as problem:
        record.update(status=STATUS_PATCH_FAILED, messages=list(problem.messages),
                      patch_sha256=file_sha256(patch_dir / DERIVED_PATCH))
        return record
    result = applied.pop("result")
    record.update(status=STATUS_OK, **applied, patch_sha256=file_sha256(patch_dir / DERIVED_PATCH),
                  child_sha256=result.child_sha256, n_changed=result.n_changed, edit_frac=result.edit_frac,
                  changed_frac=result.changed_frac)
    readout = clean.dev_readout_joint(instance, result.child, batch_size)
    record.update(n_select=readout["n"], readouts=readout["readouts"], n_clean=readout["n_clean"],
                  readouts_clean=readout["readouts_clean"], clean_split_sha256=readout["clean_split_sha256"],
                  readout_source=clean.source_name())
    return record


def run_frozen(session_dir: Path, instance_ids: Sequence[str], out_dir: Path,
               batch_size: int = DEFAULT_BATCH_SIZE) -> list[Path]:
    """One JSON record per target in ``out_dir`` (``frozen_<deltas sha256[:12]>_<id>.json``); never overwritten."""
    for instance_id in instance_ids:
        if not isinstance(instance_id, str) or not instance_id.startswith("i") or fewshot.is_fewshot_id(instance_id):
            raise FrozenEditError(f"{instance_id!r} is not an evidence-v1 instance id")
    out_dir.mkdir(parents=True, exist_ok=True)
    # One source check per process; the name holds the targets, so parallel processes on one session never collide.
    source_work = out_dir / f"source_{Path(session_dir).name}_{'_'.join(instance_ids)}"
    if source_work.exists():
        raise FrozenEditError(f"{source_work} exists already; records are never overwritten")
    edit = session_edit(Path(session_dir).resolve(), source_work)
    written = []
    for instance_id in instance_ids:
        stem = f"frozen_{edit.deltas_sha256[:12]}_{instance_id}"
        path = out_dir / f"{stem}.json"
        if path.exists() or (out_dir / stem).exists():
            raise FrozenEditError(f"{path} or its work directory exists already; records are never overwritten")
        record = frozen_one(edit, fewshot.load_any(instance_id), out_dir / stem, batch_size)
        write_json(path, record)
        written.append(path)
    return written


def _points(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 3)


def compare(f6_records: Sequence[Mapping[str, Any]], e3_records: Sequence[Mapping[str, Any]],
            seeds: Mapping[str, int]) -> dict[str, Any]:
    """Per target: frozen gain and clean beside E3's rerun gain and clean (points); the means over the non-source
    seeds, the share and PREREG F6's reading; and the seed-100 gate (module docstring). ``seeds``: id -> student seed.
    """
    e3 = {str(r["target"]["instance_id"]): r for r in e3_records}
    rows, gate = [], None
    for record in sorted(f6_records, key=lambda r: seeds[str(r["target"]["instance_id"])]):
        target = str(record["target"]["instance_id"])
        rerun = e3.get(target)
        if rerun is None:
            raise FrozenEditError(f"no E3 record for {target}")
        if record.get("status") != STATUS_OK or rerun.get("status") != STATUS_OK:
            raise FrozenEditError(f"{target}: status F6 {record.get('status')!r}, E3 {rerun.get('status')!r}")
        row = {"instance_id": target, "seed": seeds[target], "frozen_within_limits": record["within_limits"],
               "frozen_gain": _points(record["readouts"]["delta_acc"]),
               "frozen_clean": _points(record["readouts_clean"]["delta_acc"]),
               "rerun_within_limits": rerun["within_limits"],
               "rerun_gain": _points(rerun["readouts"]["delta_acc"]),
               "rerun_clean": _points(rerun["readouts_clean"]["delta_acc"]),
               "frozen_n_changed": record["n_changed"], "rerun_n_changed": rerun["n_changed"],
               "same_child": record["child_sha256"] == rerun["child_sha256"]}
        rows.append(row)
        if seeds[target] == SOURCE_SEED:
            gate = {"instance_id": target, "same_child": row["same_child"],
                    "same_readouts": record["readouts"] == rerun["readouts"],
                    "same_readouts_clean": record["readouts_clean"] == rerun["readouts_clean"]}
            gate["passed"] = bool(gate["same_child"] and gate["same_readouts"] and gate["same_readouts_clean"])
    others = [row for row in rows if row["seed"] != SOURCE_SEED]
    if not others:
        raise FrozenEditError("no target outside the source seed")
    frozen_mean = float(np.mean([row["frozen_gain"] for row in others]))
    rerun_mean = float(np.mean([row["rerun_gain"] for row in others]))
    share = frozen_mean / rerun_mean if rerun_mean > 0 else None
    if share is None:
        reading = "undefined: the rerun mean gain is not positive"
    elif share <= PROGRAM_SHARE:
        reading = READ_PROGRAM
    elif share >= VALUES_SHARE:
        reading = READ_VALUES
    else:
        reading = READ_BOTH
    return {"rows": rows, "n_unseen": len(others), "frozen_mean_gain": round(frozen_mean, 4),
            "rerun_mean_gain": round(rerun_mean, 4), "share": None if share is None else round(share, 4),
            "frozen_mean_clean": round(float(np.mean([row["frozen_clean"] for row in others])), 4),
            "rerun_mean_clean": round(float(np.mean([row["rerun_clean"] for row in others])), 4),
            "frozen_clean_ok": sum(row["frozen_clean"] >= -1.0 for row in others),
            "rerun_clean_ok": sum(row["rerun_clean"] >= -1.0 for row in others),
            "reading": reading, "seed100_gate": gate}


def _load_records(directory: Path, pattern: str) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted(Path(directory).glob(pattern))]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="add a session's final deltas to targets and read them out")
    run.add_argument("--session", type=Path, required=True, help="the session directory ($WEIGHTCODE_RUNS/sessions/<id>)")
    run.add_argument("--instances", nargs="+", required=True, help="target dev instance ids")
    run.add_argument("--out", type=Path, required=True, help="directory for one JSON record per target")
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    table = sub.add_parser("compare", help="F6 records beside E3's records, and PREREG F6's reading")
    table.add_argument("--f6", type=Path, required=True, help="directory of frozen_*.json records")
    table.add_argument("--e3", type=Path, required=True, help="directory of E3's transfer_*.json records")
    table.add_argument("--out", type=Path, help="write the comparison as JSON")
    args = parser.parse_args(argv)
    if args.command == "run":
        for path in run_frozen(args.session, args.instances, args.out, args.batch_size):
            record = read_json(path)
            print(json.dumps({"record": str(path), "target": record["target"]["instance_id"],
                              "status": record["status"], "within_limits": record.get("within_limits"),
                              "delta_acc_select": (record.get("readouts") or {}).get("delta_acc"),
                              "delta_acc_clean": (record.get("readouts_clean") or {}).get("delta_acc")}))
        return 0
    f6 = _load_records(args.f6, "frozen_*.json")
    e3 = _load_records(args.e3, "transfer_*.json")
    seeds = {str(r["target"]["instance_id"]): int(r["target"]["seed"]) for r in f6}
    result = compare(f6, [r for r in e3 if str(r["target"]["instance_id"]) in seeds], seeds)
    result["f6_files"] = {path.name: file_sha256(path) for path in sorted(args.f6.glob("frozen_*.json"))}
    if args.out is not None:
        write_json(args.out, result)
    print(json.dumps(result, indent=1))
    return 0


__all__ = ["FROZEN_SCHEMA", "PROGRAM_SHARE", "READ_BOTH", "READ_PROGRAM", "READ_VALUES", "VALUES_SHARE", "FrozenEdit",
           "FrozenEditError", "compare", "frozen_one", "main", "run_frozen", "session_edit"]

if __name__ == "__main__":
    raise SystemExit(main())
