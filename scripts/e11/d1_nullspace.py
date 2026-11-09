"""PREREG D1 (main 3bc7838): closed-form null-space projection of the two E11 finals' edits, then the C reading.

The spec is the D1 bullet of docs/PREREG_v0_DRAFT.md ("D1 and D2, fine-tuning strategies as tools for a program").
No LLM call, no program run (the finals' deltas are the saved C transfer runs), no label read by the projection. Dev
only; nothing reads the sealed store.

* The finals: contrast_s5 s760cb52ce82fb937 (k145_gated_b2_r6_orig_targets, program 207b312d...) and gaussian_noise_s5
  sc918e11dceef637a (gate3_lower, program 3c1cebc3...). Each final's edit on a student is the child of its saved
  transfer run there (``runs/devreadouts/learn/<session>/transfer/transfer_<program12>_<instance>``), rebuilt from the
  run's derived patch under the limits the run recorded (the E10 program limits) and checked against the run's child
  hash, student hash and experience content hash.
* The wrapper (``weightcode.analysis.nullspace``): per edited weight tensor, the uncentred second moment of its layer's
  inputs over x_old (the run's 64 clean experience rows, all tokens) on the unedited student; the top eigenvectors
  holding a fraction e of the energy; the delta's input side projected onto their complement; biases and other 1-D
  deltas unchanged. Scope s1: every edited weight tensor; s2: the edited weight tensors outside the freed gate units
  (the gate blocks' MLP tensors), which is the patch projection for the contrast final and nothing for the gaussian
  final (s2 is its no-op and is not checked). Grid e in {0.90, 0.99, 0.999} x scope, scope outer: 6 configurations
  for the contrast final, 3 for the gaussian final.
* ``grid FINAL``: on student 100 (the final's own instance), every configuration is one check
  (``weightcode.baselines.fit.runner.execute_run`` with ``joint=True, e10=True``: the joint oracle on V_check and its clean
  companion, the E10 program limits, at most 8 checks). The pick is joint-retain-v1 (S_old >= -20 of 4000,
  ``devreadout.readout_retention``), read on V_select; every configuration is also read on V_select
  (``devreadout.readout_candidates``, descriptive).
* ``transfer FINAL``: the picked configuration, fixed, applied to the final's transfer runs on students 101 to 105,
  each with its own x_old on its own unedited student, applied under the E10 program limits (on a refusal, recorded
  and applied without the size limits, as the transfer runs are), read on V_select (``dev_readout_joint``). C as E3
  reads it: reuse ratio = mean unseen gain / the wrapped final's gain on student 100; clean >= -1.0 on k of 5.
* ``summary``: the tables and D1's reading, fixed in PREREG before any run: D1 fixes retention if, for the contrast
  final, C is met with the wrapper (ratio >= 0.7, clean >= -1.0 on at least 4 of 5) while the student-100 joint stays
  >= 75% of the final's (+13.49).

Records go to ``$WEIGHTCODE_RUNS/devreadouts/d1/`` (never overwritten, each with a .sha256 file); the check runs to
``$WEIGHTCODE_RUNS/devreadouts/d1/runs/``.

    source scripts/env.sh
    .venv/bin/python -B scripts/e11/d1_nullspace.py --threads 8 grid contrast
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
SCHEMA = "weightcode.d1.nullspace.v1"
SPEC = "docs/PREREG_v0_DRAFT.md main 3bc7838, D1 (null-space projection of a final's edit)"
RETENTION_FLOOR = -20
MAX_CHECKS = 8
CLEAN_FLOOR = -1.0
RATIO_BAR = 0.7
K_OF_5 = 4
JOINT_BAR_100 = 13.49
"""PREREG D1: the student-100 joint must stay >= 75% of the contrast final's (+13.49 points, as written there)."""
SCOPES = ("s1", "s2")


@dataclass(frozen=True)
class Final:
    """One E11 final: its session, program, student-100 instance and the blocks of its freed gate units."""

    name: str
    session: str
    program12: str
    instance_id: str
    condition: str
    gate_blocks: tuple[int, ...]  # on student 100, as the program printed them
    s2_expected: tuple[str, ...]
    final_readout: tuple[float, float]  # V_select (gain, clean) of record, points (DEV_RESULTS E11 B)


FINALS = {
    "contrast": Final("contrast", "s760cb52ce82fb937", "207b312d4295", "i3eb3fc78", "contrast_s5", (2,),
                      ("tokenizer.conv_layers.0.0.weight",), (17.925, 0.05)),
    "gaussian": Final("gaussian", "sc918e11dceef637a", "3c1cebc35ef0", "i00802425", "gaussian_noise_s5", (2, 4, 6),
                      (), (31.625, 0.125)),
}


class D1Error(RuntimeError):
    """An input does not match its record, or a step cannot run as PREREG D1 specifies."""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for numpy and BLAS")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("grid", "transfer"):
        p = sub.add_parser(name)
        p.add_argument("final", choices=sorted(FINALS))
    sub.add_parser("summary")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------------------------------------
# Paths and records


def runs_root() -> Path:
    from weightcode.common.paths import runs_dir

    return Path(runs_dir())


def out_dir() -> Path:
    return runs_root() / "devreadouts" / "d1"


def transfer_dir(final: Final) -> Path:
    return runs_root() / "devreadouts" / "learn" / final.session / "transfer"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_records(name: str, records: Sequence[Mapping[str, Any]]) -> Path:
    from weightcode.baselines.fit.runner import jsonable

    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.jsonl"
    if path.exists():
        raise D1Error(f"{path} exists; records are never overwritten")
    path.write_text("".join(json.dumps(jsonable(r), sort_keys=True, allow_nan=False) + "\n" for r in records),
                    encoding="utf-8")
    (folder / f"{name}.jsonl.sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="utf-8")
    print(f"wrote {path} sha256 {file_sha256(path)}", flush=True)
    return path


def read_records(name: str) -> list[dict[str, Any]]:
    path = out_dir() / f"{name}.jsonl"
    if not path.is_file():
        raise D1Error(f"{path} does not exist; run its subcommand first")
    sha_file = path.with_name(path.name + ".sha256")
    if sha_file.read_text().split()[0] != file_sha256(path):
        raise D1Error(f"{path} does not hash as its .sha256 file says")
    return [json.loads(line) for line in path.read_text().splitlines()]


def provenance(threads: int) -> dict[str, Any]:
    import numpy as np

    from weightcode.baselines.fit.runner import git_state
    from weightcode.analysis import nullspace

    return {"script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
            "nullspace_sha256": file_sha256(Path(nullspace.__file__)), "git": git_state(), "threads": threads,
            "numpy": np.__version__, "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "spec": SPEC}


def points(value: float) -> float:
    return 100.0 * float(value)


def summarize(record: Mapping[str, Any]) -> dict[str, Any]:
    """V_select fields in points: gain, clean, joint = gain + clean, the worst classes.

    ``record`` is a ``dev_readout_joint``-style record (the devreadout picks and frontier, the C readouts: ``n``) or
    a transfer run record (``weightcode.harness.program_transfer.v0``: ``n_select``)."""
    ro, rc = record["readouts"], record["readouts_clean"]
    n = record["n"] if "n" in record else record["n_select"]
    gain, clean = points(ro["delta_acc"]), points(rc["delta_acc"])
    return {"gain": gain, "clean": clean, "joint": gain + clean, "worst_class_clean": points(rc["worst_class_delta"]),
            "worst_class_gain": points(ro["worst_class_delta"]), "fixed": ro["fixed"], "broken": ro["broken"],
            "fixed_clean": rc["fixed"], "broken_clean": rc["broken"], "n": n, "n_clean": record["n_clean"],
            "clean_split_sha256": record.get("clean_split_sha256")}


# --------------------------------------------------------------------------------------------------------------------
# A final's edit on one student, from its saved transfer run


@dataclass
class Edit:
    instance: Any
    record: dict[str, Any]
    record_path: Path
    child: dict[str, Any]
    x_new: Any
    x_old: Any
    limits: Any


def unbounded_limits(limits: Any) -> Any:
    """The limits without the size limits, as ``weightcode.harness.transfer._child`` applies a refused derived patch."""
    import dataclasses

    return dataclasses.replace(limits, max_rel_frob=None, max_changed_frac=None)


def load_edit(final: Final, instance_id: str) -> Edit:
    """The final's child on ``instance_id``, rebuilt from the transfer run's derived patch and checked."""
    import numpy as np

    from weightcode.baselines.fit.cli import load_checked_instances
    from weightcode.common.identity import params_sha256
    from weightcode.evalkit.e10 import e10_program_limits
    from weightcode.patch import Limits, apply_patch

    path = transfer_dir(final) / f"transfer_{final.program12}_{instance_id}.json"
    record = json.loads(path.read_text())
    if record.get("status") != "ok" or record["target"]["instance_id"] != instance_id:
        raise D1Error(f"{path}: not an ok transfer run on {instance_id}")
    (inst,) = load_checked_instances([instance_id], allow_main=False)
    if inst.private.get("condition") != final.condition:
        raise D1Error(f"{instance_id} is not a {final.condition} instance")
    if record["target"]["params_sha256"] != inst.params_sha256:
        raise D1Error(f"{path}: the run's student is not {instance_id}'s")
    limits = Limits(**record["limits"])
    if limits != e10_program_limits(inst.task["limits"], len(inst.params)):
        raise D1Error(f"{path}: the run's limits are not the E10 program limits")
    work = Path(record["work_dir"])
    if params_sha256(dict(np.load(work / "student" / "model.npz", allow_pickle=False))) != inst.params_sha256:
        raise D1Error(f"{work}/student/model.npz is not the unedited student")
    patch_file = work / "patch" / "derived_patch.json"
    if hashlib.sha256(patch_file.read_bytes()).hexdigest() != record["patch_sha256"]:
        raise D1Error(f"{patch_file} does not hash as the run recorded")
    patch = json.loads(patch_file.read_text())
    if record["readout_child"] == "within_limits":
        applied = apply_patch(inst.params, patch, patch_file.parent, limits, inst.params_sha256)
    else:
        applied = apply_patch(inst.params, patch, patch_file.parent, unbounded_limits(limits), inst.params_sha256)
    if applied.child_sha256 != record["child_sha256"]:
        raise D1Error(f"{path}: the rebuilt child does not hash as the run recorded")
    with np.load(work / "student" / "experience.npz", allow_pickle=False) as archive:
        experience = {key: archive[key] for key in archive.files}
    if params_sha256(experience) != record["experience"]["content_sha256"]:
        raise D1Error(f"{work}/student/experience.npz does not hash as the run recorded")
    child = {key: np.asarray(value) for key, value in applied.child.items()}
    return Edit(inst, record, path, child, experience["x_new"], experience["x_old"], limits)


MAX_GATE_UNITS = 8
"""At most this many freed units per block (the finals free 2 per gate, or 6); more is not a freed-unit edit."""


def gate_units(base: Mapping[str, Any], child: Mapping[str, Any], edited: Sequence[str]) -> dict[int, list[int]]:
    """The freed gate units per block: the hidden units whose linear1 row or linear2 column the edit changes.

    Every edited MLP weight of the two finals is a freed-unit edit (the program frees a few little-used units and
    rewrites their linear1 rows and linear2 columns); this is checked: per block, the units whose linear1 row or
    linear2 column changes are at most ``MAX_GATE_UNITS``, and ``scope_tensors`` checks that both are edited. The
    program picks its gate blocks per student (a label-free oracle), so the blocks are read from the edit.
    """
    import numpy as np

    from weightcode.analysis.nullspace import layer_of

    blocks: dict[int, set[int]] = {}
    for tensor in edited:
        kind, block = layer_of(tensor) if tensor.startswith("classifier.blocks.") else ("other", None)
        if kind not in ("linear1", "linear2"):
            continue
        changed = np.asarray(base[tensor]) != np.asarray(child[tensor])  # linear1 [hidden, dim], linear2 [dim, hidden]
        units = np.flatnonzero(changed.any(axis=1 if kind == "linear1" else 0))
        blocks.setdefault(int(block), set()).update(int(u) for u in units)
    for block, units in blocks.items():
        if len(units) > MAX_GATE_UNITS:
            raise D1Error(f"block {block}: the MLP edit changes {len(units)} units, not a freed-unit edit")
    return {block: sorted(units) for block, units in sorted(blocks.items())}


def scope_tensors(final: Final, edited: Sequence[str], gates: Mapping[int, Sequence[int]]) -> dict[str, list[str]]:
    """s1: every edited weight tensor; s2: those outside the freed gate units (checked against PREREG D1's list)."""
    gate = {f"classifier.blocks.{b}.{part}.weight" for b in gates for part in ("linear1", "linear2")}
    s2 = [t for t in edited if t not in gate]
    if tuple(s2) != final.s2_expected:
        raise D1Error(f"{final.name}: s2 would be {s2}, PREREG D1 names {list(final.s2_expected)}")
    if not gate <= set(edited):
        raise D1Error(f"{final.name}: the gate blocks' MLP weights {sorted(gate - set(edited))} are not all edited")
    return {"s1": list(edited), "s2": s2}


def wrapper_configs(final: Final, scopes: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    """The grid in check order (scope outer, e inner); an empty scope is a no-op and is not a configuration."""
    from weightcode.analysis.nullspace import ENERGY_FRACTIONS

    return [{"scope": s, "e": e} for s in SCOPES if scopes[s] for e in ENERGY_FRACTIONS]


def wrapped(edit: Edit, config: Mapping[str, Any], tensors: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The edit's child with the projection of ``config`` on ``tensors``, and the projection record."""
    from weightcode.analysis import nullspace as N

    module = N.forward_module(edit.instance.instance_dir / "forward.py", edit.instance.forward_sha256)
    moments, rows, parity = N.moments(module, edit.instance.params, edit.x_old, tensors)
    bases = {t: N.old_basis(t, moments[t], float(config["e"]), rows[t]) for t in tensors}
    child, per_tensor = N.projected_child(edit.instance.params, edit.child, bases)
    return child, {"config": dict(config), "tensors": list(tensors), "per_tensor": per_tensor,
                   "replay_parity_max_abs": parity, "x_old_rows": int(len(edit.x_old)),
                   "moments_on": "the unedited student", "labels_read": False}


# --------------------------------------------------------------------------------------------------------------------
# grid: the checks on student 100


def cmd_grid(final: Final, threads: int) -> None:
    import numpy as np

    from weightcode.analysis import nullspace as N
    from weightcode.baselines.fit.runner import CandidatePatch, MethodSpec, execute_run
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE, readout_candidates, readout_retention

    edit = load_edit(final, final.instance_id)
    edited = N.edited_weight_tensors(edit.instance.params, edit.child)
    gates = gate_units(edit.instance.params, edit.child, edited)
    if tuple(gates) != final.gate_blocks:
        raise D1Error(f"{final.name} on student 100: gate blocks {list(gates)}, the program printed {final.gate_blocks}")
    scopes = scope_tensors(final, edited, gates)
    configs = wrapper_configs(final, scopes)
    if len(configs) > MAX_CHECKS:
        raise D1Error(f"{len(configs)} configurations exceed {MAX_CHECKS} checks")
    run_dir = out_dir() / "runs" / f"d1_{final.name}_s100"
    projections: dict[int, dict[str, Any]] = {}

    def generate(evidence: Any) -> tuple[list[Any], dict[str, Any]]:
        k = evidence.n_deployment
        if not (np.array_equal(evidence.x[:k], edit.x_new) and np.array_equal(evidence.x[k:], edit.x_old)):
            raise D1Error("the run's evidence rows are not the transfer run's x_new and x_old")
        candidates = []
        for slot, config in enumerate(configs):
            child, projection = wrapped(edit, config, scopes[config["scope"]])
            projection["gate_units"] = gates
            projections[slot] = projection
            candidates.append(CandidatePatch(
                slot=slot, child=child, config=dict(config),
                hypothesis=(f"the {final.name} final's edit with the input side of {config['scope']}'s weight deltas "
                            f"projected off the top eigenvectors holding {config['e']} of the x_old input energy"),
                predicted_effect="less change on clean inputs, at some cost in the condition gain",
                notes={"projection": projection}))
            print(f"{final.name} generated {config}: kept-norm ratios "
                  f"{ {t: round(v['kept_frob_ratio'], 4) for t, v in projection['per_tensor'].items()} }", flush=True)
        return candidates, {"configs": configs, "scopes": scopes, "labels_read": False,
                            "inputs": "x_old of the transfer run's experience (no label), the final's saved edit"}

    spec = MethodSpec(method=f"d1_nullspace_{final.name}",
                      label=(f"PREREG D1: null-space projection of the E11 {final.name} final's edit "
                             "(AlphaEdit / GPM idea, closed form, label-free)"),
                      uses_labels_for_fit=False, edit_limits_enforced=True)
    started = time.perf_counter()
    record = execute_run(instance=edit.instance, spec=spec, run_dir=run_dir, run_seed=0,
                         config={"configs": configs, "scopes": scopes, "spec": SPEC,
                                 "final": {"session": final.session, "transfer_run": str(edit.record_path),
                                           "transfer_run_sha256": file_sha256(edit.record_path),
                                           "child_sha256": edit.record["child_sha256"]},
                                 "kept_rule": f"joint-retain-v1, S_old >= {RETENTION_FLOOR} of 4000"},
                         generate=generate, joint=True, replay_in_fit=True, e10=True)
    run_path = run_dir / "run.json"
    frontier = {r["patch_id"]: r for r in readout_candidates(run_path, DEFAULT_BATCH_SIZE)} if record["oracle_log"] \
        else {}
    rows = []
    for entry in record["candidates"]:
        item: dict[str, Any] = {"schema": SCHEMA, "final": final.name, "student": 100,
                                "instance_id": final.instance_id, "patch_id": entry.get("patch_id"),
                                "config": entry["config"], "invalid": entry.get("invalid"),
                                "projection": projections[int(entry["slot"])] if "slot" in entry else None}
        if "check" in entry:
            c = entry["check"]
            item["vcheck"] = {key: c.get(key) for key in ("query_index", "score", "score_new", "score_old", "fixed_new",
                                                         "broken_new", "fixed_old", "broken_old", "n_changed")}
            item["eligible"] = int(c["score_old"]) >= RETENTION_FLOOR
        if "edit" in entry:
            item["edit"] = {key: entry["edit"].get(key) for key in ("within_limits", "violations", "n_changed",
                                                                    "changed_frac", "edit_size")}
        if entry.get("patch_id") in frontier:
            item["vselect"] = summarize(frontier[entry["patch_id"]])
        rows.append(item)
    if record["oracle_log"]:
        pick = readout_retention(run_path, RETENTION_FLOOR, DEFAULT_BATCH_SIZE,
                                 label="PREREG D1: joint-retain-v1 pick (floor -20) read on V_select")
        selection, vselect, vcheck = pick["selection"], summarize(pick), pick["vcheck"]
        picked = next((r["config"] for r in rows if r["patch_id"] == selection["patch_id"]), None)
    else:
        selection = {"patch_id": None, "n_checks": 0, "n_eligible": 0, "rule": "base (no valid candidate)"}
        vselect, vcheck, picked = None, None, None
    gain, clean = final.final_readout
    pick_record = {"schema": SCHEMA, "final": final.name, "student": 100, "instance_id": final.instance_id,
                   "selection": selection, "picked_config": picked, "pick_vcheck": vcheck, "pick_vselect": vselect,
                   "final_vselect_of_record": {"gain": gain, "clean": clean, "joint": gain + clean},
                   "final_transfer_vselect": summarize(edit.record),
                   "queries_used": record["queries_used"], "n_invalid": record["n_invalid"],
                   "run_json": str(run_path), "run_json_sha256": file_sha256(run_path),
                   "wall_s": time.perf_counter() - started, "provenance": provenance(threads)}
    write_records(f"d1_{final.name}_s100_grid", rows)
    write_records(f"d1_{final.name}_s100_pick", [pick_record])


# --------------------------------------------------------------------------------------------------------------------
# transfer: the picked wrapper, fixed, on students 101 to 105


def unseen_ids(final: Final) -> list[str]:
    prefix = f"transfer_{final.program12}_"
    ids = sorted(p.name[len(prefix):-5] for p in transfer_dir(final).glob(f"{prefix}*.json"))
    if final.instance_id not in ids or len(ids) != 6:
        raise D1Error(f"{transfer_dir(final)}: expected the student-100 run and five more, found {ids}")
    return [i for i in ids if i != final.instance_id]


def student_seed(instance: Any) -> int:
    seed = instance.private.get("seed")
    if not isinstance(seed, int) or not 101 <= seed <= 105:
        raise D1Error(f"{instance.instance_id}: student seed {seed!r} is not one of the C students 101 to 105")
    return seed


def apply_wrapped(edit: Edit, child: Mapping[str, Any], work: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write ``child - base`` as a delta patch and apply it under the run's limits; on a refusal, record it and apply
    it without the size limits (the transfer runs' rule)."""
    import numpy as np

    from weightcode.baselines.instance_io import write_delta_patch
    from weightcode.common.identity import params_sha256
    from weightcode.patch import PatchError, apply_patch

    inst = edit.instance
    patch_file = write_delta_patch(work, inst.params_sha256, inst.params, child,
                                   "PREREG D1 wrapped final (null-space projection of the transfer run's edit)",
                                   "D1 transfer of the fixed wrapper; not a teacher program")
    patch = json.loads(patch_file.read_text())
    try:
        result = apply_patch(inst.params, patch, patch_file.parent, edit.limits, inst.params_sha256)
        within, messages = True, []
    except PatchError as refusal:
        within, messages = False, list(refusal.messages)
        result = apply_patch(inst.params, patch, patch_file.parent, unbounded_limits(edit.limits), inst.params_sha256)
    applied = {key: np.asarray(value) for key, value in result.child.items()}
    return applied, {"within_limits": within, "limit_messages": messages, "child_sha256": result.child_sha256,
                     "roundtrip_exact": result.child_sha256 == params_sha256(child),
                     "patch_sha256": file_sha256(patch_file)}


def cmd_transfer(final: Final, threads: int) -> None:
    from weightcode.analysis import nullspace as N
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
    from weightcode.harness import clean as clean_source

    (pick,) = read_records(f"d1_{final.name}_s100_pick")
    config = pick["picked_config"]
    rows: list[dict[str, Any]] = []
    if config is None:
        rows.append({"schema": SCHEMA, "final": final.name, "picked_config": None,
                     "note": "the student-100 pick is the base: no wrapper to transfer, C not read",
                     "provenance": provenance(threads)})
        write_records(f"d1_{final.name}_c", rows)
        return
    for instance_id in unseen_ids(final):
        started = time.perf_counter()
        edit = load_edit(final, instance_id)
        edited = N.edited_weight_tensors(edit.instance.params, edit.child)
        gates = gate_units(edit.instance.params, edit.child, edited)
        scopes = scope_tensors(final, edited, gates)
        child, projection = wrapped(edit, config, scopes[config["scope"]])
        projection["gate_units"] = gates
        work = out_dir() / "runs" / f"d1_{final.name}_c_{instance_id}"
        if work.exists():
            raise D1Error(f"{work} exists; never overwritten")
        applied, application = apply_wrapped(edit, child, work)
        readout = clean_source.dev_readout_joint(edit.instance, applied, DEFAULT_BATCH_SIZE)
        rows.append({"schema": SCHEMA, "final": final.name, "instance_id": instance_id,
                     "student_seed": student_seed(edit.instance), "picked_config": config,
                     "projection": projection, "application": application, "vselect": summarize(readout),
                     "unwrapped_vselect": summarize(edit.record), "transfer_run": str(edit.record_path),
                     "transfer_run_sha256": file_sha256(edit.record_path), "readout_source": clean_source.source_name(),
                     "wall_s": time.perf_counter() - started})
        v = rows[-1]["vselect"]
        print(f"{final.name} {instance_id}: wrapped gain {v['gain']:+.2f} clean {v['clean']:+.2f} "
              f"(unwrapped {rows[-1]['unwrapped_vselect']['gain']:+.2f} / {rows[-1]['unwrapped_vselect']['clean']:+.2f})",
              flush=True)
    own = pick["pick_vselect"]
    gains = [r["vselect"]["gain"] for r in rows]
    cleans = [r["vselect"]["clean"] for r in rows]
    unwrapped = [r["unwrapped_vselect"]["gain"] for r in rows]
    c = {"schema": SCHEMA, "final": final.name, "picked_config": config, "own_gain": own["gain"],
         "own_clean": own["clean"], "own_joint": own["joint"], "unseen_gains": gains, "unseen_cleans": cleans,
         "mean_unseen_gain": sum(gains) / len(gains),
         "reuse_ratio": (sum(gains) / len(gains)) / own["gain"] if own["gain"] else None,
         "n_clean_ok": sum(1 for v in cleans if v >= CLEAN_FLOOR),
         "unwrapped_reuse_ratio": (sum(unwrapped) / len(unwrapped)) / pick["final_transfer_vselect"]["gain"],
         "unwrapped_n_clean_ok": sum(1 for r in rows if r["unwrapped_vselect"]["clean"] >= CLEAN_FLOOR),
         "provenance": provenance(threads)}
    c["c_met"] = bool(c["reuse_ratio"] is not None and c["reuse_ratio"] >= RATIO_BAR and c["n_clean_ok"] >= K_OF_5)
    rows.append({"summary": c})
    write_records(f"d1_{final.name}_c", rows)


# --------------------------------------------------------------------------------------------------------------------
# summary


def cmd_summary() -> None:
    for name in ("contrast", "gaussian"):
        try:
            grid = read_records(f"d1_{name}_s100_grid")
            (pick,) = read_records(f"d1_{name}_s100_pick")
        except D1Error as missing:
            print(f"== {name}: {missing}")
            continue
        f = pick["final_vselect_of_record"]
        print(f"== {name} final (student 100): of record gain {f['gain']:+.2f} clean {f['clean']:+.2f} "
              f"joint {f['joint']:+.2f}")
        print("   config              S_joint S_new S_old elig | V_select gain  clean  joint worst-cls | within lim")
        for r in grid:
            v, s = r.get("vselect") or {}, r.get("vcheck") or {}
            mark = "K" if r["patch_id"] is not None and r["patch_id"] == pick["selection"]["patch_id"] else " "
            print(f" {mark} {r['config']['scope']} e={r['config']['e']:<6} {str(s.get('score')):>7} "
                  f"{str(s.get('score_new')):>5} {str(s.get('score_old')):>5} {str(r.get('eligible')):>5} | "
                  f"{v.get('gain', float('nan')):+7.2f} {v.get('clean', float('nan')):+6.2f} "
                  f"{v.get('joint', float('nan')):+6.2f} {v.get('worst_class_clean', float('nan')):+7.2f} | "
                  f"{(r.get('edit') or {}).get('within_limits')}")
        p = pick["pick_vselect"]
        print(f"   pick {pick['selection']['patch_id']} {pick['picked_config']}: "
              + (f"gain {p['gain']:+.2f} clean {p['clean']:+.2f} joint {p['joint']:+.2f}" if p else "the base"))
        try:
            rows = read_records(f"d1_{name}_c")
        except D1Error as missing:
            print(f"   C: {missing}")
            continue
        summary = rows[-1].get("summary")
        if summary is None:
            print(f"   C: {rows[-1].get('note')}")
            continue
        for r in sorted(rows[:-1], key=lambda item: item["student_seed"]):
            v, u = r["vselect"], r["unwrapped_vselect"]
            print(f"   {r['instance_id']} seed {r['student_seed']}: wrapped gain {v['gain']:+.2f} clean "
                  f"{v['clean']:+.2f} | unwrapped gain {u['gain']:+.2f} clean {u['clean']:+.2f} | "
                  f"within limits {r['application']['within_limits']}")
        print(f"   C with the wrapper: ratio {summary['reuse_ratio']:.3f} (mean {summary['mean_unseen_gain']:+.2f} "
              f"against {summary['own_gain']:+.2f}), clean >= -1.0 on {summary['n_clean_ok']} of 5: "
              f"{'met' if summary['c_met'] else 'not met'}; unwrapped ratio {summary['unwrapped_reuse_ratio']:.3f}, "
              f"{summary['unwrapped_n_clean_ok']} of 5")
        if name == "contrast":
            fixes = summary["c_met"] and p is not None and p["joint"] >= JOINT_BAR_100
            print(f"   D1 reading (contrast): C {'met' if summary['c_met'] else 'not met'}, student-100 joint "
                  f"{p['joint']:+.2f} against {JOINT_BAR_100:+.2f}: projection "
                  f"{'fixes retention (a candidate library tool)' if fixes else 'does not fix retention here'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE
    if args.command == "summary":
        cmd_summary()
    elif args.command == "grid":
        cmd_grid(FINALS[args.final], args.threads)
    else:
        cmd_transfer(FINALS[args.final], args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
