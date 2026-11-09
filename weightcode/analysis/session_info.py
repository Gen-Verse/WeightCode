"""Information-use readout of one editor session on a dev instance (dev only; never reads the sealed test).

It answers "what did this session use, and where did its gain come from":

- ``trajectory``: every checked patch in query order, with its ops (op name and scalar fields), the V_check numbers the
  editor was shown (S, fixed, broken, accuracies) and the dev readout of the same child on V_select, which the editor
  never saw. Each child is rebuilt from ``attempts/<attempt>/`` with the arm's own applier
  (``weightcode.harness.applying.ArmApplier``) and must reproduce the logged child hash. Query 0 was written before any
  V_check feedback, so its V_select readout is the gain from the editor's own analysis of the weights and evidence;
  later queries show what the feedback added.
- ``access``: for each evidence array, trial file and weight tensor, the executed cells whose code names it. This is
  a lexical count, not a data-flow analysis (a tensor reached through a loop over the parameters is not counted); the
  audit's taint analysis (verdict, label-access cells, label-scored candidates) is reported next to it. ``audits``
  lists every audit report on record, ``audit.json`` first and then ``audit.rerun*.json`` in name order, so a verdict
  replaced by a later audit version stays visible; ``audit`` is the last of them.
- ``final``: the final pick with its query index, ops, hypothesis, predicted effect and dev readout, and the declared
  final. A declared final (``kind: declared``, a session with a patch-check budget of zero such as M2NL) was never
  checked: it is rebuilt from its own attempt snapshot like a trajectory row, must reproduce the recorded child hash,
  and has no query index and no V_check numbers; the trajectory of such a session is empty.
- LEARN (a program arm): each trajectory row is a checked program, rebuilt from its derived patch, with the joint
  check numbers (score_new, score_old, S_joint) and two dev readouts from ``weightcode.harness.clean.dev_readout_joint``:
  ``vselect`` on V_select of the condition and ``vselect_clean`` on its clean companion. The final repeats both and
  names the program file and its hashes; ``access`` counts the experience arrays.
- LEARN on a fewshot-v1 instance (PREREG E2): the readout is ``weightcode.harness.fewshot.readout`` on V_select:
  ``vselect`` over all rows (10-way accuracy, the quantity the E2 gate reads), ``vselect_new`` over the new-class rows,
  ``vselect_old`` over the old-class rows, and ``readouts_fewshot``, the full readout with per-class changes and the
  worst old class.
- LEARN on a seqskill-v1 instance (PREREG E5): the readout is ``weightcode.harness.seqskill.readout`` on V_select by exact
  match: ``vselect`` over part i (the new operator alone, the gated part), ``vselect_old`` over part iv (old
  operators), and ``readouts_seqskill``, the full readout with parts ii and iii and the breakdowns. On a
  scanskill-v1 instance (PREREG E13, the seqskill variant scan_rule) the readout is ``weightcode.evalkit.scanskill.
  scan_readout``: ``vselect`` and ``vselect_new`` the new V_select commands, ``vselect_old`` the old V_select commands,
  and ``readouts_scan`` the full readout (by construction, and the gain, clean and joint points).
- LEARN on a graphnew-v1 instance (PREREG E7): the readout is ``weightcode.harness.graphnew.readout`` on V_select (40-way):
  ``vselect``, ``vselect_new`` and ``vselect_old`` as for fewshot-v1, and ``readouts_graphnew``, the full readout with
  per-class changes, class-balanced new-class accuracy, the new-class rows by hop distance to the evidence and
  old_select cut by class (an accuracy over zero rows is null).
- LEARN on an fscil-v1 instance (PREREG E9): the readout is ``weightcode.harness.fscil.readout`` under the session mask,
  child against the session's student: ``vselect`` over all seen classes, ``vselect_new`` (the session's classes),
  ``vselect_prev`` (the earlier incremental classes), ``vselect_old`` (the base classes, old_select) and
  ``readouts_fscil``, the full readout with pooled and class-balanced accuracies, per class and per session.
- LEARN on a policyskill-v1 instance (PREREG E14): the readout is ``weightcode.harness.policyskill.readout``, greedy
  rollouts on V_select, child against base: ``vselect`` over the new-mission episodes (success per episode in the
  accuracy fields), ``vselect_old`` over the trained-mission episodes, and ``readouts_policyskill``, the full readout
  with gain, clean and joint in points, lengths, pickups and the breakdowns.
- ``compare`` (with ``--reference``): ``weightcode.analysis.compare_patches.compare_runs`` of the session's run.json against
  a reference run.json, for example tie_inputs on the same instance.
- ``usage``: status, wall time, cells and queries used, token counts and the first and last rate-limit events.

Usage::

    python -m weightcode.analysis.session_info SESSION_DIR [--reference REF_RUN_JSON] [--out FILE]

Without ``--out`` the JSON goes to stdout and the table to stderr; with ``--out`` the table goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE, dev_readout, readout_split
from weightcode.evalkit.instances import instances_root
from weightcode.harness import clean, fewshot, fscil, graphnew, policyskill, seqskill
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import arm_spec
from weightcode.harness.experience import EXPERIENCE_FILE, EXPERIENCE_KEYS
from weightcode.harness.finalize import KIND_DECLARED

SCHEMA = "weightcode.analysis.session_info.v0"
EVIDENCE_KEYS = ("x_fail", "y_fail", "pred_fail", "x_rand", "y_rand", "pred_rand", "split_ids")
TRIAL_FILES = ("model.npz", "evidence.npz", "model_spec.json", "forward.py", "task.json", "model_shapes.json")
OP_SCALAR_FIELDS = ("tensor", "axis", "side", "strength", "factor", "value")
VCHECK_FIELDS = ("score", "fixed", "broken", "n", "n_base_errors", "base_acc", "child_acc")
JOINT_VCHECK_FIELDS = ("score_new", "score_old", "fixed_new", "broken_new", "fixed_old", "broken_old", "n_new", "n_old")
AUDIT_FIELDS = ("verdict", "candidate_count", "candidate_count_is_lower_bound", "candidate_limit", "candidate_settings",
                "label_access_cells", "flags", "rule_counts", "notes", "n_cells", "n_analysed", "unparsed_cells")


class SessionInfoError(RuntimeError):
    """The session's records disagree with each other or with the rebuilt children."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def op_summary(patch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The op name and the scalar fields of each op; index lists and file payloads are left out."""
    summary = []
    for op in patch.get("ops", []):
        entry: dict[str, Any] = {"op": op.get("op")}
        entry.update({key: op[key] for key in OP_SCALAR_FIELDS if key in op and not isinstance(op[key], (list, dict))})
        summary.append(entry)
    return summary


def trajectory(root: Path, session: Mapping[str, Any], instance: Instance, split: Mapping[str, np.ndarray] | None,
               batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
    """Every checked patch in query order with its V_check numbers and its V_select dev readout (LEARN: the checked
    programs, with the joint numbers and the readouts on V_select and on its clean companion)."""
    applier = ArmApplier(session)
    joint = arm_spec(str(session["arm"])).program
    few = fewshot.is_fewshot(instance.private)
    seq = seqskill.is_seqskill(instance.private)
    scan = seqskill.is_scan(instance.private)
    gnew = graphnew.is_graphnew(instance.private)
    fsc = fscil.is_fscil(instance.private)
    pol = policyskill.is_policyskill(instance.private)
    checks = sorted(_read_jsonl(root / "logs" / "checks.jsonl"), key=lambda check: int(check["query_index"]))
    rows = []
    for check in checks:
        attempt_dir = root / "attempts" / str(check["attempt"])
        patch = _read_json(attempt_dir / str(check["patch_file"]))
        result = applier.apply(patch, attempt_dir)
        if result.child_sha256 != check["child_sha256"]:
            raise SessionInfoError(f"{root.name}: query {check['query_index']} rebuilds child {result.child_sha256}, "
                                   f"logged {check['child_sha256']}")
        if few:
            readout = fewshot.readout(instance, result.child, batch_size)
            readout["readouts"] = readout["parts"]["all"]
        elif seq:
            readout = seqskill.readout(instance, result.child, batch_size)
            readout["readouts"] = readout["parts"]["new" if scan else "i"]
        elif gnew:  # one full-graph forward for the base and one for the child
            readout = graphnew.readout(instance, result.child)
            readout["readouts"] = readout["parts"]["all"]
        elif fsc:  # under the session mask, against the session's student
            readout = fscil.readout(instance, result.child)
            readout["readouts"] = readout["parts"]["seen"]
        elif pol:  # greedy rollouts on the V_select episodes, new and old
            readout = policyskill.readout(instance, result.child)
            readout["readouts"] = readout["parts"]["new"]
        elif joint:
            readout = clean.dev_readout_joint(instance, result.child, batch_size)
        else:
            readout = dev_readout(instance, result.child, batch_size, split=split)
        rows.append({
            "query_index": int(check["query_index"]),
            "patch_id": check["patch_id"],
            "attempt": check["attempt"],
            "ops": op_summary(patch),
            "hypothesis": patch.get("hypothesis"),
            "predicted_effect": patch.get("predicted_effect"),
            "edit_size": check["executor"]["edit_size"],
            "edit_frac": check["executor"]["edit_frac"],
            "child_sha256": check["child_sha256"],
            "vcheck": {key: check.get(key) for key in VCHECK_FIELDS + (JOINT_VCHECK_FIELDS if joint else ())},
            "vselect": readout["readouts"],
        })
        if joint:
            rows[-1].update(program_file=check["program_file"], program_sha256=check["program_sha256"],
                            deltas_sha256=check["deltas_sha256"], tensors=[op.get("tensor") for op in patch["ops"]])
        if few:
            rows[-1].update(vselect_new=readout["parts"]["new"], vselect_old=readout["parts"]["old"],
                            readouts_fewshot=readout["readouts_fewshot"])
        elif scan:
            rows[-1].update(vselect_new=readout["parts"]["new"], vselect_old=readout["parts"]["old"],
                            readouts_scan=readout["readouts_scan"])
        elif seq:
            rows[-1].update(vselect_old=readout["parts"]["iv"], readouts_seqskill=readout["readouts_seqskill"])
        elif gnew:
            rows[-1].update(vselect_new=readout["parts"]["new"], vselect_old=readout["parts"]["old"],
                            readouts_graphnew=readout["readouts_graphnew"])
        elif fsc:
            rows[-1].update(vselect_new=readout["parts"]["new"], vselect_prev=readout["parts"]["prev"],
                            vselect_old=readout["parts"]["old"], readouts_fscil=readout["readouts_fscil"])
        elif pol:
            rows[-1].update(vselect_old=readout["parts"]["old"],
                            readouts_policyskill=readout["readouts_policyskill"])
        elif joint:
            rows[-1].update(vselect_clean=readout["readouts_clean"], n_clean=readout["n_clean"],
                            clean_split_sha256=readout["clean_split_sha256"])
    return rows


def _cells_naming(cells: Sequence[Mapping[str, Any]], token: str) -> list[int]:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])")
    return [int(cell["index"]) for cell in cells if pattern.search(str(cell.get("code") or ""))]


def access(root: Path, tensor_names: Sequence[str], program: bool = False,
           experience_keys: Sequence[str] = EXPERIENCE_KEYS) -> dict[str, Any]:
    """Executed cells whose code names each evidence array (LEARN: experience array, ``experience_keys``), trial file
    and tensor (lexical, see the module doc)."""
    cells = [cell for cell in _read_jsonl(root / "logs" / "cells.jsonl") if cell.get("executed")]
    keys, files = (experience_keys, (*TRIAL_FILES, EXPERIENCE_FILE)) if program else (EVIDENCE_KEYS, TRIAL_FILES)
    return {
        "method": "lexical: executed cells whose code contains the name as a whole token",
        "n_executed_cells": len(cells),
        "evidence_arrays": {key: _cells_naming(cells, key) for key in keys},
        "trial_files": {name: _cells_naming(cells, name) for name in files},
        "tensors": {name: _cells_naming(cells, name) for name in tensor_names},
    }


def _declared_final(root: Path, final: Mapping[str, Any], session: Mapping[str, Any], instance: Instance,
                    split: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
    """The declared final rebuilt from ``attempts/<attempt>/<patch_id>`` with the arm's applier, and its dev readout."""
    attempt_dir = root / "attempts" / str(final["attempt"])
    patch = _read_json(attempt_dir / str(final["patch_id"]))
    result = ArmApplier(session).apply(patch, attempt_dir)
    if result.child_sha256 != final["child_sha256"]:
        raise SessionInfoError(f"{root.name}: the declared final rebuilds child {result.child_sha256}, recorded "
                               f"{final['child_sha256']}")
    readout = dev_readout(instance, result.child, batch_size, split=split)
    return {"query_index": None, "patch_id": final["patch_id"], "attempt": final["attempt"], "ops": op_summary(patch),
            "hypothesis": patch.get("hypothesis"), "predicted_effect": patch.get("predicted_effect"), "vcheck": None,
            "vselect": readout["readouts"]}


def _final(root: Path, result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], session: Mapping[str, Any],
           instance: Instance, split: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
    final = result["final"]
    record: dict[str, Any] = {"is_base_noop": bool(final["is_base_noop"]), "kind": final.get("kind"),
                              "reason": final.get("reason"), "child_sha256": final.get("child_sha256"),
                              "declared_final": result.get("declared_final")}
    if record["is_base_noop"]:
        record.update(query_index=None, ops=[], vselect=None)
        if "program" in final and fewshot.is_fewshot(instance.private):
            record.update(vselect_new=None, vselect_old=None, readouts_fewshot=None, program=None)
        elif "program" in final and seqskill.is_scan(instance.private):
            record.update(vselect_new=None, vselect_old=None, readouts_scan=None, program=None)
        elif "program" in final and seqskill.is_seqskill(instance.private):
            record.update(vselect_old=None, readouts_seqskill=None, program=None)
        elif "program" in final and graphnew.is_graphnew(instance.private):
            record.update(vselect_new=None, vselect_old=None, readouts_graphnew=None, program=None)
        elif "program" in final and fscil.is_fscil(instance.private):
            record.update(vselect_new=None, vselect_prev=None, vselect_old=None, readouts_fscil=None, program=None)
        elif "program" in final and policyskill.is_policyskill(instance.private):
            record.update(vselect_old=None, readouts_policyskill=None, program=None)
        elif "program" in final:
            record.update(vselect_clean=None, program=None)
        return record
    if final.get("kind") == KIND_DECLARED:
        record.update(_declared_final(root, final, session, instance, split, batch_size))
        return record
    matches = [row for row in rows if row["child_sha256"] == final["child_sha256"]]
    if not matches:
        raise SessionInfoError(f"{root.name}: the final child {final['child_sha256']} is not among the checked children")
    first = min(matches, key=lambda row: row["query_index"])
    patch = _read_json(root / str(final["dir"]) / "patch.json")
    record.update(query_index=first["query_index"], patch_id=first["patch_id"], ops=op_summary(patch),
                  hypothesis=patch.get("hypothesis"), predicted_effect=patch.get("predicted_effect"),
                  vcheck=first["vcheck"], vselect=first["vselect"])
    if "readouts_fewshot" in first:
        record.update(vselect_new=first["vselect_new"], vselect_old=first["vselect_old"],
                      readouts_fewshot=first["readouts_fewshot"], program=final.get("program"))
    elif "readouts_scan" in first:
        record.update(vselect_new=first["vselect_new"], vselect_old=first["vselect_old"],
                      readouts_scan=first["readouts_scan"], program=final.get("program"))
    elif "readouts_seqskill" in first:
        record.update(vselect_old=first["vselect_old"], readouts_seqskill=first["readouts_seqskill"],
                      program=final.get("program"))
    elif "readouts_graphnew" in first:
        record.update(vselect_new=first["vselect_new"], vselect_old=first["vselect_old"],
                      readouts_graphnew=first["readouts_graphnew"], program=final.get("program"))
    elif "readouts_fscil" in first:
        record.update(vselect_new=first["vselect_new"], vselect_prev=first["vselect_prev"],
                      vselect_old=first["vselect_old"], readouts_fscil=first["readouts_fscil"],
                      program=final.get("program"))
    elif "readouts_policyskill" in first:
        record.update(vselect_old=first["vselect_old"], readouts_policyskill=first["readouts_policyskill"],
                      program=final.get("program"))
    elif "vselect_clean" in first:
        record.update(vselect_clean=first["vselect_clean"], program=final.get("program"))
    return record


def _usage(result: Mapping[str, Any]) -> dict[str, Any]:
    stream = result.get("stream") or {}
    return {
        "status": result.get("status"),
        "infra": result.get("infra"),
        "wall_s": result.get("wall_s"),
        "editor": result.get("editor"),
        "cells": result.get("cells"),
        "queries": result.get("queries"),
        "tokens": result.get("tokens"),
        "rate_limit_first": stream.get("rate_limit_first"),
        "rate_limit_last": stream.get("rate_limit_last"),
    }


def session_info(root: Path, *, reference: Path | None = None, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    """The full information-use record of the session at ``root`` (see the module docstring)."""
    root = Path(root)
    result = _read_json(root / "session_result.json")
    session = _read_json(root / "session.json")
    run = _read_json(root / "run.json")
    instance_id = str(run["instance"]["instance_id"])
    if seqskill.is_seqskill_id(instance_id):
        instance = load_instance(seqskill.instance_dir(instance_id))
    elif graphnew.is_graphnew_id(instance_id):
        instance = load_instance(graphnew.instance_dir(instance_id))
    elif fscil.is_fscil_id(instance_id):
        instance = load_instance(fscil.instance_dir(instance_id))
    elif policyskill.is_policyskill_id(instance_id):
        instance = load_instance(policyskill.instance_dir(instance_id))
    else:
        instance = fewshot.load_any(instance_id)
    own = (fewshot.is_fewshot(instance.private) or seqskill.is_seqskill(instance.private)
           or graphnew.is_graphnew(instance.private) or fscil.is_fscil(instance.private)
           or policyskill.is_policyskill(instance.private))
    split = None if own else readout_split(instance)
    rows = trajectory(root, session, instance, split, batch_size)
    audits = [{"file": path.name, **{key: report.get(key) for key in AUDIT_FIELDS}, "auditor_sha256": report.get("auditor_sha256")}
              for path in [root / "audit.json", *sorted(root.glob("audit.rerun*.json"))] if path.exists()
              for report in [_read_json(path)]]
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "session": root.name,
        "arm": result.get("arm"),
        "repeat": result.get("repeat"),
        "instance_id": instance.instance_id,
        "dataset": instance.private.get("dataset"),
        "condition": instance.private.get("condition"),
        "usage": _usage(result),
        "kept_rule": session.get("kept_rule"),  # None: the v1 rule (weightcode.harness.retention)
        "library": None if (lib := session.get("library")) is None else [  # LEARN v3 (weightcode.harness.library)
            {key: entry[key] for key in ("label", "score_new", "score_old", "score", "S", "B_c") if key in entry}
            | {"source": entry["source"]} for entry in lib["entries"]],  # S and B_c: an E9 v2 library
        "trajectory": rows,
        "final": _final(root, result, rows, session, instance, split, batch_size),
        "access": access(root, sorted(instance.params), arm_spec(str(session["arm"])).program,
                         fscil.EXPERIENCE_KEYS if "fscil" in session else
                         policyskill.EXPERIENCE_KEYS if "policyskill" in session else EXPERIENCE_KEYS),
        "audits": audits,
        "audit": audits[-1] if audits else None,
    }
    if reference is not None:
        from weightcode.analysis.compare_patches import compare_runs

        record["compare"] = {"reference": str(reference), **compare_runs(root / "run.json", Path(reference), batch_size)}
    return record


def _delta_acc(readout: Mapping[str, Any] | None) -> str:
    if not readout or readout.get("delta_acc") is None:
        return "-"
    return f"{float(readout['delta_acc']):+.4f}"


def _points(part: Mapping[str, Any]) -> str:
    """A seqskill readout part's exact-match change in points."""
    return f"{100 * float(part['delta']):+.2f}"


def _ops_text(ops: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for op in ops:
        fields = " ".join(f"{key}={op[key]}" for key in OP_SCALAR_FIELDS if key in op and key != "tensor")
        parts.append(" ".join(str(item) for item in (op.get("op"), op.get("tensor", ""), fields) if item))
    return "; ".join(parts) or "-"


def format_table(record: Mapping[str, Any]) -> str:
    """A short human-readable summary of ``record``."""
    usage = record["usage"]
    cells, queries = usage.get("cells") or {}, usage.get("queries") or {}
    audit = record.get("audit") or {}
    lines = [
        f"session {record['session']}  arm {record['arm']}  instance {record['instance_id']} ({record['dataset']}, "
        f"{record['condition']})  status {usage['status']}  wall {usage['wall_s']} s  cells {cells.get('used')}/"
        f"{cells.get('budget')}  queries {queries.get('used')}/{queries.get('budget')}  audit {audit.get('verdict')}"
        + ((f"  kept {kept['name']}" + (f" (S_old >= {kept['s_old_floor']})" if "s_old_floor" in kept else ""))
           if (kept := record.get("kept_rule")) else "")
        + (f"  library {len(lib)} programs" if (lib := record.get("library")) else ""),
        "q  patch                     S_vcheck  dacc_vselect  ops",
    ]
    few = any("readouts_fewshot" in row for row in record["trajectory"]) or "readouts_fewshot" in record["final"]
    seq = any("readouts_seqskill" in row for row in record["trajectory"]) or "readouts_seqskill" in record["final"]
    scan = any("readouts_scan" in row for row in record["trajectory"]) or "readouts_scan" in record["final"]
    gnew = any("readouts_graphnew" in row for row in record["trajectory"]) or "readouts_graphnew" in record["final"]
    fsc = any("readouts_fscil" in row for row in record["trajectory"]) or "readouts_fscil" in record["final"]
    pol = (any("readouts_policyskill" in row for row in record["trajectory"])
           or "readouts_policyskill" in record["final"])
    joint = (few or seq or scan or gnew or fsc or pol
             or any("vselect_clean" in row for row in record["trajectory"])
             or "vselect_clean" in record["final"])
    if scan:
        lines[-1] = "q  program                   S_joint  S_new  S_old  gain  clean  joint  tensors"
    elif seq:
        lines[-1] = "q  program                   S_joint  S_new  S_old  dem_i  dem_ii  dem_iii  dem_iv  tensors"
    elif pol:
        lines[-1] = "q  program                   S_joint  S_new  S_old    gain   clean   joint  tensors"
    elif few:
        lines[-1] = "q  program                   S_joint  S_new  S_old  dacc_new  dacc_old  dacc_all  tensors"
    elif gnew:
        lines[-1] = "q  program                   S_joint  S_new  S_old  dacc_new  dacc_old  dacc_all  dbal_new  tensors"
    elif fsc:
        lines[-1] = "q  program                   S_joint  S_new  S_old  dacc_new  dacc_prev  dacc_old  dbal_seen  tensors"
    elif joint:
        lines[-1] = "q  program                   S_joint  S_new  S_old  dacc_vselect  dacc_clean  tensors"
    for row in record["trajectory"]:
        if scan:
            full = row["readouts_scan"]
            lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>7}  "
                         f"{row['vcheck']['score_new']!s:>5}  {row['vcheck']['score_old']!s:>5}  "
                         f"{full['gain_points']:>+5.2f}  {full['clean_points']:>+5.2f}  {full['joint_points']:>+5.2f}  "
                         f"{', '.join(map(str, row['tensors'])) or '-'}")
            continue
        if pol:
            full = row["readouts_policyskill"]
            lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>7}  "
                         f"{row['vcheck']['score_new']!s:>5}  {row['vcheck']['score_old']!s:>5}  "
                         + "  ".join(f"{float(full[key]):>+6.1f}" for key in ("gain", "clean", "joint"))
                         + f"  {', '.join(map(str, row['tensors'])) or '-'}")
            continue
        if seq:
            full = row["readouts_seqskill"]
            lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>7}  "
                         f"{row['vcheck']['score_new']!s:>5}  {row['vcheck']['score_old']!s:>5}  "
                         + "  ".join(f"{_points(full[part]):>{width}}"
                                     for part, width in (("i", 5), ("ii", 6), ("iii", 7), ("iv", 6)))
                         + f"  {', '.join(map(str, row['tensors'])) or '-'}")
            continue
        if gnew:
            balanced = row["readouts_graphnew"].get("delta_new_balanced")
            lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>7}  "
                         f"{row['vcheck']['score_new']!s:>5}  {row['vcheck']['score_old']!s:>5}  "
                         f"{_delta_acc(row['vselect_new']):>8}  {_delta_acc(row['vselect_old']):>8}  "
                         f"{_delta_acc(row['vselect']):>8}  {_delta_acc({'delta_acc': balanced}):>8}  "
                         f"{', '.join(map(str, row['tensors'])) or '-'}")
            continue
        if fsc:
            balanced = row["readouts_fscil"].get("delta_bal_seen")
            lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>7}  "
                         f"{row['vcheck']['score_new']!s:>5}  {row['vcheck']['score_old']!s:>5}  "
                         f"{_delta_acc(row['vselect_new']):>8}  {_delta_acc(row['vselect_prev']):>9}  "
                         f"{_delta_acc(row['vselect_old']):>8}  {_delta_acc({'delta_acc': balanced}):>9}  "
                         f"{', '.join(map(str, row['tensors'])) or '-'}")
            continue
        if few:
            lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>7}  "
                         f"{row['vcheck']['score_new']!s:>5}  {row['vcheck']['score_old']!s:>5}  "
                         f"{_delta_acc(row['vselect_new']):>8}  {_delta_acc(row['vselect_old']):>8}  "
                         f"{_delta_acc(row['vselect']):>8}  {', '.join(map(str, row['tensors'])) or '-'}")
            continue
        if joint:
            lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>7}  "
                         f"{row['vcheck']['score_new']!s:>5}  {row['vcheck']['score_old']!s:>5}  "
                         f"{_delta_acc(row['vselect']):>12}  {_delta_acc(row['vselect_clean']):>10}  "
                         f"{', '.join(map(str, row['tensors'])) or '-'}")
            continue
        lines.append(f"{row['query_index']:<2} {str(row['patch_id'])[:25]:<25} {row['vcheck']['score']!s:>8}  "
                     f"{_delta_acc(row['vselect']):>12}  {_ops_text(row['ops'])}")
    final = record["final"]
    if final["is_base_noop"] and queries.get("budget") == 0:
        lines.append(f"final: base model ({final.get('reason')})")
    elif final["is_base_noop"]:
        lines.append("final: base model (no checked patch with S > 0)")
    elif final.get("kind") == KIND_DECLARED:
        lines.append(f"final: declared {final['patch_id']} (unchecked)  dacc_vselect {_delta_acc(final['vselect'])}  "
                     f"hypothesis: {final.get('hypothesis')}")
    elif scan:
        full = final["readouts_scan"]
        lines.append(f"final: query {final['query_index']} program {final['patch_id']}  "
                     f"em_new {100 * full['new']['em_child']:.2f} (gain {full['gain_points']:+.2f})  "
                     f"em_old {100 * full['old']['em_child']:.2f} (clean {full['clean_points']:+.2f})  "
                     f"joint {full['joint_points']:+.2f}  hypothesis: {final.get('hypothesis')}")
    elif pol:
        full = final["readouts_policyskill"]
        lines.append(f"final: query {final['query_index']} program {final['patch_id']}  success new "
                     f"{full['new']['success_child']:.3f} (gain {float(full['gain']):+.1f})  success old "
                     f"{full['old']['success_child']:.3f} (clean {float(full['clean']):+.1f})  joint "
                     f"{float(full['joint']):+.1f}  hypothesis: {final.get('hypothesis')}")
    elif seq:
        full = final["readouts_seqskill"]
        lines.append(f"final: query {final['query_index']} program {final['patch_id']}  "
                     + "  ".join(f"em_{part} {100 * full[part]['em_child']:.2f} ({_points(full[part])})"
                                 for part in ("i", "ii", "iii", "iv"))
                     + f"  hypothesis: {final.get('hypothesis')}")
    elif gnew:
        full = final["readouts_graphnew"]
        lines.append(f"final: query {final['query_index']} program {final['patch_id']}  dacc_new "
                     f"{_delta_acc(final['vselect_new'])}  dacc_old {_delta_acc(final['vselect_old'])}  dacc_all "
                     f"{_delta_acc(final['vselect'])}  dbal_new {_delta_acc({'delta_acc': full.get('delta_new_balanced')})}"
                     f"  worst old class {full.get('worst_old_class')}  hypothesis: {final.get('hypothesis')}")
    elif fsc:
        full = final["readouts_fscil"]
        lines.append(f"final: query {final['query_index']} program {final['patch_id']}  session {full['session']}  "
                     f"dacc_new {_delta_acc(final['vselect_new'])}  dacc_prev {_delta_acc(final['vselect_prev'])}  "
                     f"dacc_old {_delta_acc(final['vselect_old'])}  bal_seen {full.get('bal_seen_child')} "
                     f"({_delta_acc({'delta_acc': full.get('delta_bal_seen')})})  hypothesis: {final.get('hypothesis')}")
    elif few:
        worst = final["readouts_fewshot"]
        lines.append(f"final: query {final['query_index']} program {final['patch_id']}  dacc_new "
                     f"{_delta_acc(final['vselect_new'])}  dacc_old {_delta_acc(final['vselect_old'])}  dacc_all "
                     f"{_delta_acc(final['vselect'])}  worst old class {worst['worst_old_class']} "
                     f"({100 * worst['worst_old_class_delta']:+.2f})  hypothesis: {final.get('hypothesis')}")
    elif joint:
        lines.append(f"final: query {final['query_index']} program {final['patch_id']}  dacc_vselect "
                     f"{_delta_acc(final['vselect'])}  dacc_clean {_delta_acc(final['vselect_clean'])}  "
                     f"hypothesis: {final.get('hypothesis')}")
    else:
        lines.append(f"final: query {final['query_index']} {final['patch_id']}  dacc_vselect "
                     f"{_delta_acc(final['vselect'])}  hypothesis: {final.get('hypothesis')}")
    named = {key: cells for key, cells in record["access"]["evidence_arrays"].items() if cells}
    lines.append(f"evidence arrays named (cell indices): {named or 'none'}")
    if audit:
        lines.append(f"audit: label-access cells {audit.get('label_access_cells')}, label-scored candidates "
                     f"{audit.get('candidate_count')}")
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="session directory under $WEIGHTCODE_RUNS/sessions")
    parser.add_argument("--reference", type=Path, help="run.json of a reference run on the same instance")
    parser.add_argument("--out", type=Path, help="write the JSON record here")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    record = session_info(args.session, reference=args.reference, batch_size=args.batch_size)
    text = json.dumps(record, indent=1, allow_nan=False, default=_json_default)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(format_table(record))
    else:
        print(text)
        print(format_table(record), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
