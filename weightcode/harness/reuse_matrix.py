"""PREREG T08: the reuse matrix of the ten E11 B source finals over the six students of their condition (no LLM).

Sources and targets are fixed by PREREG T08 (44ce46a) and NSCC R31, never chosen here: per condition the B final of
record, the two T02 sessions (seed 100, repeats 1 and 2) and the two T01 sessions (seeds 101 and 102), and the six
instances of the condition on the students of seeds 100 to 105. ``check_targets`` confirms in the store's index that
each (condition, seed) still has exactly the fixed instance, and ``source_final`` reads each source's final from its
session records.

Two arms per cell, both existing code with no change to what they compute:

* program arm: the source's final program rerun unchanged on the target (``weightcode.harness.transfer.run_transfer``,
  E3's rerun: the target's own experience, the LEARN program sandbox with the driver's virtual clock virtual-v1, the
  program wall limit, the target's LEARN limits, V_select of the condition and its clean companion). Six targets per
  source, the source's own instance included;
* copy arm: the source's final/deltas.npz added unchanged to the target (``weightcode.harness.frozen_edit.run_frozen``,
  F6, with F6's source checks). The five non-own targets.

The diagonal gate (``diagonal``, PREREG T08): the program arm on the source's own instance must give the session's
final child exactly (params sha256 equal), from the same program file. Whether the rerun's float64 deltas.npz is also
bit-identical is recorded beside it and does not gate: the child is float32(base + delta), and on another CPU
(AVX2 workstation against AVX-512 ASPIRE2B nodes) the deltas can differ in their last bits while every float32 entry
of the child stays the same. A source whose gate fails is an implementation defect: its row is reported with the cells
but not read (no reading is computed from it) until the mismatch is explained and fixed.

Cell status (``program_cell``, ``copy_cell``; fixed before any T08 number):

* ``ok``: the update was within the target's LEARN limits and changed at least one entry;
* ``base fallback``: the update was valid and within the limits but changed no entry (the child is the base);
* ``limit``: the program hit the wall limit, or the update exceeded the LEARN limits (E3 then reads the unbounded
  child, which the cell keeps as ``unbounded_*`` fields), or a format limit refused even the unbounded patch;
* ``program error``: the program raised, died or returned invalid deltas.

Readings (``readings``), descriptive only, nothing pooled, no test across cells. A cell that is not ``ok`` or ``base
fallback`` enters every reading as the base (gain 0, clean 0), as a deployment that refuses the update would leave
the student, and is listed. Points are 100 x the accuracy change on 4,000 rows; joint = gain + clean.

* per source (gate passed): its own gain (the diagonal program cell); the reuse share, the mean program gain over its
  five non-own targets divided by its own gain (undefined when the own gain is not positive); C in E3's form (share
  >= 0.7, clean >= -1.0 on at least 4 of the 5 non-own targets, every non-own cell within the limits) and C per
  target (gain_t / own gain >= 0.7 and clean_t >= -1.0 on at least 4 of the 5 non-own targets); the copy arm's means
  over the same five targets; and program minus copy (gain, clean and joint, difference of the two means);
* per target: the mean over the condition's sources of the program cells (all five sources, and the non-own sources
  alone with their count) and of the copy cells (non-own sources).

``cross_check`` compares the program cells with E3's records and the copy cells with F6's records (child sha256,
deltas sha256, n_changed, within-limits flag and every readout field) for the sources E3 and F6 ran.

Usage::

    python -m weightcode.harness.reuse_matrix cells                 # the cell list for the job script (checks first)
    python -m weightcode.harness.reuse_matrix table --root $WEIGHTCODE_RUNS/t08 --out FILE [--e3-root DIR --f6-root DIR]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.harness.jsonio import file_sha256, read_json, utc_now, write_json

SCHEMA = "weightcode.harness.reuse_matrix.v0"
GAUSSIAN = "gaussian_noise_s5"
CONTRAST = "contrast_s5"
CONDITIONS = (GAUSSIAN, CONTRAST)
SOURCES: dict[str, tuple[str, ...]] = {
    # B final of record, T02 repeat 1, T02 repeat 2 (seed 100), T01 seed 101, T01 seed 102 (PREREG T08, R31)
    GAUSSIAN: ("sc918e11dceef637a", "s4253319a365b126a", "scb92c3eb5d5226cc", "scf16f7fb35855c03",
               "sf45990dee41ac233"),
    CONTRAST: ("s760cb52ce82fb937", "s7ae7d105d7655ed5", "sfeb1cbc77d6a6042", "s3db14093bc6c3538",
               "s037ead942a4cc8aa"),
}
TARGET_SEEDS = (100, 101, 102, 103, 104, 105)
TARGETS: dict[str, tuple[str, ...]] = {  # seeds 100 to 105, in that order (PREREG T08)
    GAUSSIAN: ("i00802425", "i905213a3", "iab18c0aa", "ib1b195f6", "ie1be7cd9", "i36df1f8f"),
    CONTRAST: ("i3eb3fc78", "i4a486d70", "i5d76f25c", "id83bf155", "ib1e4e794", "iecf07b07"),
}
PROGRAM_ARM = "program"
COPY_ARM = "copy"
RATIO_MIN = 0.7
"""C: the reuse ratio (E3's form: the share) or each target's ratio (per-target form) at least this."""
CLEAN_MIN = -1.0
"""C: a clean change of at least this many points on a target."""
N_TARGETS_MIN = 4
"""C: the clean (and, per target, the ratio) condition on at least this many of the five non-own targets."""
STATUS_OK = "ok"
STATUS_BASE = "base fallback"
STATUS_LIMIT = "limit"
STATUS_ERROR = "program error"
READ_STATUSES = (STATUS_OK, STATUS_BASE)


class ReuseMatrixError(RuntimeError):
    """A fixed source or target does not match the store, or the records are incomplete or inconsistent."""


@dataclass(frozen=True)
class Source:
    """One source final as its session records it."""

    session_id: str
    condition: str
    instance_id: str
    seed: int
    has_program: bool
    program_path: str | None
    program_sha256: str | None
    deltas_sha256: str | None
    child_sha256: str | None
    n_changed: int | None


def check_targets(find: Any = None) -> dict[str, list[str]]:
    """Condition -> the store's instances of seeds 100 to 105 (``weightcode.harness.e11.find``); must equal ``TARGETS``."""
    if find is None:
        from weightcode.harness.e11 import find
    found = {condition: [find(condition, seed) for seed in TARGET_SEEDS] for condition in CONDITIONS}
    for condition in CONDITIONS:
        if tuple(found[condition]) != TARGETS[condition]:
            raise ReuseMatrixError(f"{condition}: the store's seed 100 to 105 instances are {found[condition]}, not "
                                   f"PREREG T08's {list(TARGETS[condition])}")
    return found


def source_final(session_dir: Path) -> Source:
    """The final of the session in ``session_dir`` (run.json final, session.json instance and seed)."""
    root = Path(session_dir)
    session, run = read_json(root / "session.json"), read_json(root / "run.json")
    final = run.get("final") or {}
    program = final.get("program") or {}
    has_program = bool(run.get("status") == "complete" and not final.get("is_base_noop")
                       and final.get("parent") == "instance" and program.get("file") and program.get("deltas"))
    if has_program:
        path = root / program["file"]
        if file_sha256(path) != program.get("sha256"):
            raise ReuseMatrixError(f"{path}: sha256 differs from run.json final.program.sha256")
    return Source(session_id=root.name, condition=str(session["condition"]), instance_id=str(session["instance_id"]),
                  seed=int(session["seed"]), has_program=has_program,
                  program_path=str(root / program["file"]) if has_program else None,
                  program_sha256=program.get("sha256") if has_program else None,
                  deltas_sha256=program.get("deltas_sha256") if has_program else None,
                  child_sha256=final.get("child_sha256") if has_program else None,
                  n_changed=final.get("n_changed") if has_program else None)


def load_sources(sessions_root: Path) -> dict[str, Source]:
    """Every fixed source, checked: its condition, and its instance among the condition's fixed targets."""
    sources = {}
    for condition in CONDITIONS:
        for session_id in SOURCES[condition]:
            source = source_final(Path(sessions_root) / session_id)
            if source.condition != condition or source.instance_id not in TARGETS[condition]:
                raise ReuseMatrixError(f"{session_id}: condition {source.condition}, instance {source.instance_id}; "
                                       f"PREREG T08 lists it under {condition}")
            if TARGET_SEEDS[TARGETS[condition].index(source.instance_id)] != source.seed:
                raise ReuseMatrixError(f"{session_id}: seed {source.seed} is not its instance's seed")
            sources[session_id] = source
    return sources


def cell_list(sources: Mapping[str, Source]) -> list[dict[str, str]]:
    """The cells to run: the program arm on all six targets, the copy arm on the five non-own targets."""
    cells = []
    for condition in CONDITIONS:
        for session_id in SOURCES[condition]:
            source = sources[session_id]
            if not source.has_program:
                continue
            for target in TARGETS[condition]:
                cells.append({"arm": PROGRAM_ARM, "session_id": session_id, "target": target,
                              "program": str(source.program_path)})
            for target in TARGETS[condition]:
                if target != source.instance_id:
                    cells.append({"arm": COPY_ARM, "session_id": session_id, "target": target, "program": ""})
    return cells


def _points(value: float | None) -> float | None:
    return None if value is None else round(100.0 * float(value), 3)


def _readout(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    gain = _points((record.get("readouts") or {}).get("delta_acc"))
    clean = _points((record.get("readouts_clean") or {}).get("delta_acc"))
    joint = None if gain is None or clean is None else round(gain + clean, 3)
    return {f"{prefix}gain": gain, f"{prefix}clean": clean, f"{prefix}joint": joint}


def _status_of_applied(record: Mapping[str, Any]) -> str:
    if record.get("status") == "patch_failed" or record.get("within_limits") is False:
        return STATUS_LIMIT
    return STATUS_BASE if int(record["n_changed"]) == 0 else STATUS_OK


def program_cell(record: Mapping[str, Any]) -> dict[str, Any]:
    """One program-arm cell from E3's transfer record (status rules in the module docstring)."""
    run = record.get("run") or {}
    sandbox_run = run.get("run") or {}
    cell: dict[str, Any] = {"arm": PROGRAM_ARM, "instance_id": record["target"]["instance_id"],
                            "target_params_sha256": record["target"]["params_sha256"],
                            "program_sha256": record["program"]["sha256"],
                            "execution_s": sandbox_run.get("duration_s"), "clock": run.get("clock"),
                            "deltas_sha256": record.get("deltas_sha256"), "child_sha256": record.get("child_sha256"),
                            "n_changed": record.get("n_changed"), "within_limits": record.get("within_limits"),
                            "record_status": record["status"]}
    if record["status"] == "program_failed":
        cell["status"] = STATUS_LIMIT if sandbox_run.get("status") == "timeout" else STATUS_ERROR
        cell["messages"] = list(record.get("messages") or [])[:5]
        cell.update(gain=None, clean=None, joint=None)
        return cell
    cell["status"] = _status_of_applied(record)
    if record["status"] == "patch_failed":
        cell["messages"] = list(record.get("messages") or [])[:5]
        cell.update(gain=None, clean=None, joint=None)
        return cell
    if record.get("within_limits") is False:  # E3 read the unbounded child; kept apart from the cell's value
        cell.update(_readout(record, "unbounded_"), gain=None, clean=None, joint=None)
        return cell
    cell.update(_readout(record))
    return cell


def copy_cell(record: Mapping[str, Any]) -> dict[str, Any]:
    """One copy-arm cell from F6's frozen record (status rules in the module docstring)."""
    source = record.get("source") or {}
    cell: dict[str, Any] = {"arm": COPY_ARM, "instance_id": record["target"]["instance_id"],
                            "target_params_sha256": record["target"]["params_sha256"],
                            "program_sha256": (source.get("program") or {}).get("sha256"), "execution_s": None,
                            "deltas_sha256": record.get("deltas_sha256"), "child_sha256": record.get("child_sha256"),
                            "n_changed": record.get("n_changed"), "within_limits": record.get("within_limits"),
                            "record_status": record["status"], "status": _status_of_applied(record)}
    if record["status"] == "patch_failed":
        cell["messages"] = list(record.get("messages") or [])[:5]
        cell.update(gain=None, clean=None, joint=None)
    elif record.get("within_limits") is False:
        cell.update(_readout(record, "unbounded_"), gain=None, clean=None, joint=None)
    else:
        cell.update(_readout(record))
    return cell


def diagonal(source: Source, record: Mapping[str, Any]) -> dict[str, Any]:
    """The diagonal gate: the program arm on the source's own instance gives the session's final child (module
    docstring; ``same_deltas`` is recorded, not gated)."""
    if record["target"]["instance_id"] != source.instance_id:
        raise ReuseMatrixError(f"{source.session_id}: the diagonal record is on {record['target']['instance_id']}")
    gate = {"instance_id": source.instance_id, "status": record["status"],
            "session_deltas_sha256": source.deltas_sha256, "rerun_deltas_sha256": record.get("deltas_sha256"),
            "session_child_sha256": source.child_sha256, "rerun_child_sha256": record.get("child_sha256"),
            "same_program": record["program"]["sha256"] == source.program_sha256}
    gate["same_deltas"] = gate["rerun_deltas_sha256"] == gate["session_deltas_sha256"]
    gate["same_child"] = gate["rerun_child_sha256"] == gate["session_child_sha256"]
    gate["passed"] = bool(record["status"] == "ok" and gate["same_program"] and gate["same_child"])
    return gate


def _value(cell: Mapping[str, Any] | None) -> tuple[float, float, bool]:
    """(gain, clean, counted as the base) of a cell for the readings (module docstring)."""
    if cell is None:
        raise ReuseMatrixError("a cell of a read row is missing")
    if cell["status"] in READ_STATUSES:
        return float(cell["gain"]), float(cell["clean"]), False
    return 0.0, 0.0, True


def _mean(values: Sequence[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def source_reading(source: Source, gate: Mapping[str, Any], program: Mapping[str, Mapping[str, Any]],
                   copy: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The readings of one source's row (module docstring); ``program`` and ``copy``: target id -> cell."""
    others = [target for target in TARGETS[source.condition] if target != source.instance_id]
    reading: dict[str, Any] = {"session_id": source.session_id, "own_instance": source.instance_id,
                               "own_seed": source.seed, "gate_passed": bool(gate["passed"])}
    if not gate["passed"]:
        reading["read"] = False
        reading["reason"] = "diagonal gate failed: not read until the mismatch is explained and fixed"
        return reading
    reading["read"] = True
    own_gain, own_clean, own_base = _value(program.get(source.instance_id))
    p = {t: _value(program.get(t)) for t in others}
    c = {t: _value(copy.get(t)) for t in others}
    pg, pc = [p[t][0] for t in others], [p[t][1] for t in others]
    cg, cc = [c[t][0] for t in others], [c[t][1] for t in others]
    share = float(np.mean(pg)) / own_gain if own_gain > 0 else None
    clean_ok = sum(value >= CLEAN_MIN for value in pc)
    within = all(program[t]["status"] in READ_STATUSES for t in others)
    per_target = ([t for t in others if p[t][0] / own_gain >= RATIO_MIN and p[t][1] >= CLEAN_MIN]
                  if own_gain > 0 else [])
    reading.update(
        own_gain=own_gain, own_clean=own_clean, own_joint=round(own_gain + own_clean, 3),
        own_counted_as_base=own_base,
        program_mean_gain=_mean(pg), program_mean_clean=_mean(pc),
        program_mean_joint=_mean([g + k for g, k in zip(pg, pc)]),
        reuse_share=None if share is None else round(share, 4),
        program_clean_ok=clean_ok, program_all_within_limits=within,
        c_e3_form=bool(share is not None and share >= RATIO_MIN and clean_ok >= N_TARGETS_MIN and within),
        c_per_target_n=len(per_target), c_per_target_targets=per_target,
        c_per_target=bool(len(per_target) >= N_TARGETS_MIN),
        copy_mean_gain=_mean(cg), copy_mean_clean=_mean(cc), copy_mean_joint=_mean([g + k for g, k in zip(cg, cc)]),
        program_minus_copy_gain=round(float(np.mean(pg)) - float(np.mean(cg)), 4),
        program_minus_copy_clean=round(float(np.mean(pc)) - float(np.mean(cc)), 4),
        program_minus_copy_joint=round(float(np.mean(pg) + np.mean(pc)) - float(np.mean(cg) + np.mean(cc)), 4),
        counted_as_base={"program": [t for t in others if p[t][2]], "copy": [t for t in others if c[t][2]]},
        reuse_share_undefined_reason=None if share is not None else "own gain is not positive")
    return reading


def target_reading(condition: str, target: str, sources: Mapping[str, Source], read: Sequence[str],
                   program: Mapping[str, Mapping[str, Mapping[str, Any]]],
                   copy: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Per target: the mean over the condition's read sources of the program and copy cells (module docstring)."""
    ids = [sid for sid in SOURCES[condition] if sid in read]
    all_p = [_value(program[sid].get(target)) for sid in ids]
    non_own = [sid for sid in ids if sources[sid].instance_id != target]
    nown_p = [_value(program[sid].get(target)) for sid in non_own]
    nown_c = [_value(copy[sid].get(target)) for sid in non_own]

    def means(values: Sequence[tuple[float, float, bool]]) -> dict[str, Any]:
        return {"n": len(values), "gain": _mean([v[0] for v in values]), "clean": _mean([v[1] for v in values]),
                "joint": _mean([v[0] + v[1] for v in values])}

    return {"instance_id": target, "seed": TARGET_SEEDS[TARGETS[condition].index(target)],
            "program_all_sources": means(all_p), "program_non_own": means(nown_p), "copy_non_own": means(nown_c)}


def _records(directory: Path, pattern: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(directory).glob(pattern)):
        record = read_json(path)
        target = str(record["target"]["instance_id"])
        if target in out:
            raise ReuseMatrixError(f"two records for {target} in {directory}")
        record["_file"] = {"name": path.name, "sha256": file_sha256(path)}
        out[target] = record
    return out


def build(sources: Mapping[str, Source], program_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
          copy_records: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """The matrices, diagonal gates and readings. ``*_records``: session id -> target id -> record."""
    result: dict[str, Any] = {"schema": SCHEMA, "created_at": utc_now(), "sources": {}, "conditions": {}}
    for condition in CONDITIONS:
        rows, gates, readings, read = {}, {}, [], []
        program_cells: dict[str, dict[str, Any]] = {}
        copy_cells: dict[str, dict[str, Any]] = {}
        for sid in SOURCES[condition]:
            source = sources[sid]
            result["sources"][sid] = asdict(source)
            if not source.has_program:
                rows[sid] = {"status": "no program"}
                continue
            precs, crecs = program_records.get(sid, {}), copy_records.get(sid, {})
            missing = [t for t in TARGETS[condition] if t not in precs]
            missing += [f"copy:{t}" for t in TARGETS[condition] if t != source.instance_id and t not in crecs]
            if missing:
                raise ReuseMatrixError(f"{sid}: no record for {missing}")
            extra = [t for t in list(precs) + list(crecs) if t not in TARGETS[condition]] + (
                [source.instance_id] if source.instance_id in crecs else [])
            if extra:
                raise ReuseMatrixError(f"{sid}: records outside its cells: {extra}")
            for target, record in precs.items():
                if record["program"]["sha256"] != source.program_sha256:
                    raise ReuseMatrixError(f"{sid} on {target}: the record ran another program")
            for target, record in crecs.items():
                if record.get("deltas_sha256") != source.deltas_sha256:
                    raise ReuseMatrixError(f"{sid} on {target}: the copied deltas are not the session's")
            program_cells[sid] = {t: program_cell(precs[t]) for t in TARGETS[condition]}
            copy_cells[sid] = {t: copy_cell(crecs[t]) for t in TARGETS[condition] if t != source.instance_id}
            gates[sid] = diagonal(source, precs[source.instance_id])
            reading = source_reading(source, gates[sid], program_cells[sid], copy_cells[sid])
            readings.append(reading)
            if reading["read"]:
                read.append(sid)
            rows[sid] = {"program": program_cells[sid], "copy": copy_cells[sid],
                         "records": {"program": {t: r["_file"] for t, r in precs.items() if "_file" in r},
                                     "copy": {t: r["_file"] for t, r in crecs.items() if "_file" in r}}}
        result["conditions"][condition] = {
            "targets": list(TARGETS[condition]), "rows": rows, "diagonal_gates": gates, "source_readings": readings,
            "target_readings": [target_reading(condition, t, sources, read, program_cells, copy_cells)
                                for t in TARGETS[condition]],
            "read_sources": read}
    return result


_COMPARED = ("child_sha256", "deltas_sha256", "n_changed", "within_limits", "readouts", "readouts_clean")


def cross_check(records: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]
                ) -> dict[str, Any]:
    """Field-by-field comparison of T08 records with E3's (program arm) or F6's (copy arm), per shared target."""
    rows = {}
    for target in sorted(set(records) & set(reference)):
        ours, theirs = records[target], reference[target]
        differ = [key for key in _COMPARED if ours.get(key) != theirs.get(key)]
        rows[target] = {"equal": not differ, "differ": differ,
                        "ours": _readout(ours), "reference": _readout(theirs),
                        "reference_file": theirs.get("_file")}
    return {"n": len(rows), "all_equal": all(row["equal"] for row in rows.values()), "targets": rows}


def table(root: Path, sessions_root: Path, e3_root: Path | None = None, f6_root: Path | None = None
          ) -> dict[str, Any]:
    """Read every T08 record under ``root`` (``program/<sid>/transfer_*.json``, ``copy/<sid>/frozen_*.json``)."""
    sources = load_sources(sessions_root)
    program = {sid: _records(root / PROGRAM_ARM / sid, "transfer_*.json") for sid in sources}
    copy = {sid: _records(root / COPY_ARM / sid, "frozen_*.json") for sid in sources}
    result = build(sources, program, copy)
    checks: dict[str, Any] = {}
    for sid in sources:
        if e3_root is not None and (Path(e3_root) / sid / "transfer").is_dir():
            checks.setdefault(sid, {})["program_vs_e3"] = cross_check(
                program[sid], _records(Path(e3_root) / sid / "transfer", "transfer_*.json"))
        if f6_root is not None and (Path(f6_root) / sid).is_dir():
            checks.setdefault(sid, {})["copy_vs_f6"] = cross_check(
                copy[sid], _records(Path(f6_root) / sid, "frozen_*.json"))
    result["cross_check"] = checks
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cells", help="check the fixed targets and sources, then print one JSON line per cell")
    tab = sub.add_parser("table", help="the matrices, diagonal gates, readings and cross-checks as JSON")
    tab.add_argument("--root", type=Path, required=True, help="the T08 records ($WEIGHTCODE_RUNS/t08)")
    tab.add_argument("--e3-root", type=Path, help="E3's records ($WEIGHTCODE_RUNS/devreadouts/learn)")
    tab.add_argument("--f6-root", type=Path, help="F6's records ($WEIGHTCODE_RUNS/f6)")
    tab.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sessions_root = Path(os.environ["WEIGHTCODE_RUNS"]) / "sessions"
    if args.command == "cells":
        check_targets()
        for cell in cell_list(load_sources(sessions_root)):
            print(json.dumps(cell))
        return 0
    if args.out.exists():
        raise ReuseMatrixError(f"{args.out} exists; tables are never overwritten")
    write_json(args.out, table(args.root, sessions_root, args.e3_root, args.f6_root))
    print(args.out)
    return 0


__all__ = ["CLEAN_MIN", "CONDITIONS", "COPY_ARM", "N_TARGETS_MIN", "PROGRAM_ARM", "RATIO_MIN", "SCHEMA", "SOURCES",
           "STATUS_BASE", "STATUS_ERROR", "STATUS_LIMIT", "STATUS_OK", "TARGETS", "TARGET_SEEDS", "ReuseMatrixError",
           "Source", "build", "cell_list", "check_targets", "copy_cell", "cross_check", "diagonal", "load_sources",
           "main", "program_cell", "source_final", "source_reading", "table", "target_reading"]

if __name__ == "__main__":
    raise SystemExit(main())
