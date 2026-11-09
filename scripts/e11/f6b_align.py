"""PREREG F6b, the aligned frozen edit: align each seed 101 to 105 E11 student to the seed-100 student, map the seed-100
final's delta through the alignment, add it to the target and read it as F6 and E3 read.

Spec: docs/PREREG_v0_DRAFT.md, the bullet "F6b, the aligned frozen edit". No LLM call, no sealed read, no V_check read for
a decision; dev instances only. Library: ``weightcode.analysis.align``. The reading (F6's ``frozen_one``) and the source checks
(``session_edit``) are F6's (``weightcode.harness.frozen_edit``).

Stages:

* ``rows`` (once, on the workstation): the first 2,048 rows of cifar10_vl7's clean train split in stored order, written
  to ``$WEIGHTCODE_PRIVATE/f6b/align_rows_2048.npz`` with the sha256 of the split file (the only input of the alignment besides
  the two students; no V_select, V_check, V_evidence, test or sealed row).
* ``run --condition C --instance ID --out DIR [--gate-only]``: one target. Alignment (residual permutation by the Hungarian
  method on the summed correlations of the eight residual places, one hidden permutation per edited block), the
  validity gate (exact symmetry on the target's condition V_select and clean V_select), then, if the gate passes, the
  seed-100 delta mapped into the target's coordinates, added to the target and read. A failed gate stops the target with
  exit status 3 and a record that says so; nothing is read. ``--gate-only`` stops after the gate (a smoke).
* ``summary``: the table per condition (aligned, F6 frozen and E3 rerun, side by side), the readings fixed in the entry,
  the matched correlations against the identity, and, for gaussian, the descriptive unit overlap and gate thresholds.

    source scripts/env.sh
    python scripts/e11/f6b_align.py rows
    python scripts/e11/f6b_align.py run --condition gaussian_noise_s5 --instance i905213a3 --out DIR
    python scripts/e11/f6b_align.py summary --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

THREADS = int(os.environ.get("F6B_THREADS", "4"))
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, str(THREADS))  # before numpy is imported, or BLAS ignores it

import numpy as np  # noqa: E402

DATASET = "cifar10_vl7"
N_ROWS = 2048
SOURCE_SEED = 100
SHARE_PROGRAM = 0.5
SHARE_VALUES = 0.9
FINALS: dict[str, dict[str, Any]] = {
    "gaussian_noise_s5": {"session": "sc918e11dceef637a", "source": "i00802425", "blocks": (2, 4, 6),
                          "program12": "3c1cebc35ef0"},
    "contrast_s5": {"session": "s760cb52ce82fb937", "source": "i3eb3fc78", "blocks": (2,), "program12": "207b312d4295"},
}


class F6bError(RuntimeError):
    """An input does not match the record it must reproduce."""


def runs_root() -> Path:
    from weightcode.common.paths import runs_dir

    return Path(runs_dir())


def private_root() -> Path:
    from weightcode.common.paths import private_dir

    return Path(private_dir())


def rows_path() -> Path:
    return private_root() / "f6b" / "align_rows_2048.npz"


def session_dir(condition: str) -> Path:
    return runs_root() / "sessions" / FINALS[condition]["session"]


def targets(condition: str) -> list[str]:
    from weightcode.harness import e11

    found = e11.transfer_targets(FINALS[condition]["source"])
    if len(found) != 5:
        raise F6bError(f"{condition}: {len(found)} targets, not 5")
    return found


# --------------------------------------------------------------------------------------------------------------------
# rows


def cmd_targets(args: argparse.Namespace) -> int:
    """Print the five target instance ids (seeds 101 to 105) of a condition, one line, space separated."""
    print(" ".join(targets(args.condition)))
    return 0


def cmd_rows(args: argparse.Namespace) -> int:
    from weightcode.data.store import load_split, split_sha256
    from weightcode.harness.jsonio import file_sha256, write_json

    path = rows_path()
    if path.exists():
        raise F6bError(f"{path} exists; records are never overwritten")
    train = load_split(DATASET, "train")
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.ascontiguousarray(train["x"][:N_ROWS])
    np.savez(path, x=x, idx=np.asarray(train["idx"][:N_ROWS]))
    write_json(path.with_suffix(".json"), {
        "rows": N_ROWS, "source": f"{DATASET}/splits/train.npz, the first {N_ROWS} rows in stored order",
        "train_split_sha256": split_sha256(train), "file_sha256": file_sha256(path), "x_shape": list(x.shape)})
    print(json.dumps({"written": str(path), "sha256": file_sha256(path)}))
    return 0


def load_rows() -> tuple[np.ndarray, str]:
    from weightcode.harness.jsonio import file_sha256, read_json

    path = rows_path()
    record = read_json(path.with_suffix(".json"))
    digest = file_sha256(path)
    if digest != record["file_sha256"]:
        raise F6bError(f"{path} hashes to {digest}, its record says {record['file_sha256']}")
    with np.load(path, allow_pickle=False) as archive:
        x = np.array(archive["x"])
    if x.shape != (N_ROWS, 3, 32, 32) or x.dtype != np.uint8:
        raise F6bError(f"{path}: {x.shape} {x.dtype}")
    return x, digest


# --------------------------------------------------------------------------------------------------------------------
# run


def describe_gaussian(alignment: Any, source_delta: dict[str, np.ndarray], source: Any, target: Any,
                      condition: str, instance_id: str) -> dict[str, Any] | None:
    """Descriptive, not read against a bar (gaussian only): the hidden units the rerun program changes on this target,
    mapped through the alignment into seed-100 coordinates, their overlap with the seed-100 units, and the gate
    thresholds (the changed linear1 biases) of the target beside seed 100's."""
    if condition != "gaussian_noise_s5":
        return None
    final = FINALS[condition]
    rerun_path = (runs_root() / "devreadouts" / "learn" / final["session"] / "transfer"
                  / f"transfer_{final['program12']}_{instance_id}" / "run" / "out" / "deltas.npz")
    if not rerun_path.is_file():
        return {"unavailable": f"{rerun_path} is not in this store"}
    with np.load(rerun_path, allow_pickle=False) as archive:
        rerun = {k: np.asarray(archive[k], dtype=np.float64) for k in archive.files}
    edited = sorted(int(m.group(1)) for k in rerun if (m := re.match(r"^classifier\.blocks\.(\d+)\.linear1\.weight$", k))
                    and np.any(rerun[k] != 0))
    out: dict[str, Any] = {"rerun_deltas": str(rerun_path), "blocks": {}, "target_edited_blocks": edited,
                           "seed100_edited_blocks": list(final["blocks"]),
                           "target_blocks_without_alignment": [b for b in edited if b not in final["blocks"]]}
    for block in final["blocks"]:
        w = f"classifier.blocks.{block}.linear1.weight"
        b = f"classifier.blocks.{block}.linear1.bias"
        src_units = sorted(int(i) for i in np.nonzero(np.any(source_delta[w] != 0, axis=1))[0])
        tgt_units = (sorted(int(i) for i in np.nonzero(np.any(rerun[w] != 0, axis=1))[0]) if w in rerun else [])
        inverse = alignment.inverse_hid(block)  # target unit -> the source unit it was matched to
        mapped = sorted(int(inverse[t]) for t in tgt_units)
        overlap = sorted(set(mapped) & set(src_units))
        out["blocks"][str(block)] = {
            "seed100_units": src_units, "target_units": tgt_units, "target_units_in_seed100_coordinates": mapped,
            "overlap": overlap, "n_overlap": len(overlap),
            "target_block_edited_by_the_program": bool(tgt_units),
            "matched_correlation_of_target_units": [float(alignment.hid_corr[block][s]) for s in mapped],
            "seed100_bias_after_edit": [float(np.asarray(source.params[b])[u] + source_delta[b][u]) for u in src_units],
            "target_bias_after_rerun": [float(np.asarray(target.params[b])[u] + rerun[b][u]) for u in tgt_units],
            "seed100_bias_before_edit": [float(np.asarray(source.params[b])[u]) for u in src_units],
            "target_bias_before_rerun": [float(np.asarray(target.params[b])[u]) for u in tgt_units]}
    return out


def cmd_run(args: argparse.Namespace) -> int:
    from weightcode.analysis import align as A
    from weightcode.analysis import nullspace as N
    from weightcode.evalkit.companions import load_clean_rows
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE, readout_split
    from weightcode.harness import frozen_edit as F
    from weightcode.harness import fewshot
    from weightcode.harness.jsonio import file_sha256, utc_now, write_json
    from weightcode.harness.program import load_deltas

    condition, instance_id = args.condition, args.instance
    final = FINALS[condition]
    if instance_id not in targets(condition):
        raise F6bError(f"{instance_id} is not one of {condition}'s seed 101 to 105 targets")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"f6b_{condition}_{instance_id}"
    path = out / f"{stem}.json"
    if path.exists() or (out / stem).exists():
        raise F6bError(f"{path} or its work directory exists already; records are never overwritten")
    work = out / stem
    work.mkdir(mode=0o700)
    edit = F.session_edit(session_dir(condition), work / "source")
    source = fewshot.load_any(edit.source_instance_id)
    target = fewshot.load_any(instance_id)
    if target.private.get("split_role") != "dev" or int(source.private["seed"]) != SOURCE_SEED:
        raise F6bError("F6b runs on dev instances, the source being the seed-100 student")
    delta = load_deltas(edit.deltas_path, source.params)
    x, rows_sha = load_rows()
    module = N.forward_module(target.instance_dir / "forward.py", target.private["forward_sha256"])
    alignment = A.fit_alignment(module, source.params, target.params, x, final["blocks"])
    images = {"condition_V_select": readout_split(target)["x"], "clean_V_select": load_clean_rows(target, "V_select").x}
    gate = A.symmetry_gate(module, target.params, alignment, images)
    record: dict[str, Any] = {
        "schema": "weightcode.harness.f6b.v0", "created_at": utc_now(), "condition": condition,
        "target": {"instance_id": instance_id, "seed": target.private.get("seed"),
                   "model_id": target.private.get("model_id"), "params_sha256": target.params_sha256},
        "source": {**edit.record, "alignment_rows_sha256": rows_sha, "rows": N_ROWS},
        "alignment": {"stats": alignment.stats, "blocks": list(final["blocks"]),
                      "res_fixed_points": int(np.sum(alignment.res == np.arange(len(alignment.res)))),
                      "res_sha256": hashlib.sha256(alignment.res.tobytes()).hexdigest(),
                      "hid_sha256": {str(b): hashlib.sha256(p.tobytes()).hexdigest()
                                     for b, p in alignment.hid.items()}},
        "gate": gate}
    if not gate["passed"]:
        record["status"] = "gate_failed"
        write_json(path, record)
        print(json.dumps({"record": str(path), "target": instance_id, "status": "gate_failed"}), flush=True)
        return 3
    record["descriptive"] = describe_gaussian(alignment, delta, source, target, condition, instance_id)
    if args.gate_only:
        record["status"] = "gate_only"
        write_json(path, record)
        print(json.dumps({"record": str(path), "target": instance_id, "status": "gate_only", "gate_passed": True,
                          "residual_sum_matched": alignment.stats["residual_sum_matched"],
                          "residual_sum_identity": alignment.stats["residual_sum_identity"]}), flush=True)
        return 0
    mapped = A.map_delta(delta, alignment)
    norms = {k: [float(np.linalg.norm(delta[k])), float(np.linalg.norm(mapped[k]))] for k in delta}
    if any(abs(a - b) > 1e-9 * max(1.0, a) for a, b in norms.values()):
        raise F6bError("the mapped delta has another norm than the source delta; the mapping is not a permutation")
    mapped_path = work / "deltas.npz"
    np.savez(mapped_path, **mapped)  # stored .npy members, float64, as load_deltas requires
    aligned = F.FrozenEdit(session_id=f"f6b_{edit.session_id}", source_instance_id=edit.source_instance_id,
                           deltas_path=mapped_path, deltas_sha256=file_sha256(mapped_path),
                           hypothesis=edit.hypothesis, predicted_effect=edit.predicted_effect,
                           record={"kind": "f6b_aligned", "source_edit": edit.record,
                                   "alignment_res_sha256": record["alignment"]["res_sha256"]})
    read = F.frozen_one(aligned, target, work / "frozen", DEFAULT_BATCH_SIZE)
    record.update(status=read["status"], within_limits=read.get("within_limits"), readouts=read.get("readouts"),
                  readouts_clean=read.get("readouts_clean"), n_changed=read.get("n_changed"),
                  child_sha256=read.get("child_sha256"), limits=read.get("limits"),
                  limit_messages=read.get("limit_messages"), mapped_deltas_sha256=aligned.deltas_sha256,
                  delta_norms_source_mapped=norms, readout_source=read.get("readout_source"))
    write_json(path, record)
    print(json.dumps({"record": str(path), "target": instance_id, "status": record["status"],
                      "within_limits": record["within_limits"],
                      "gain": (record["readouts"] or {}).get("delta_acc"),
                      "clean": (record["readouts_clean"] or {}).get("delta_acc")}), flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# summary


def points(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 3)


def share_reading(share: float | None) -> str:
    if share is None:
        return "undefined: the rerun mean gain is not positive"
    if share <= SHARE_PROGRAM:
        return "per-network recomputation carries reuse beyond a coordinate alignment (the program, not its values)"
    if share >= SHARE_VALUES:
        return "the values transfer once the networks are aligned"
    return "both reported, neither wording alone"


def cmd_summary(args: argparse.Namespace) -> int:
    from weightcode.harness.jsonio import file_sha256, read_json, write_json

    base = Path(args.reads) if args.reads else runs_root() / "f6b"
    result: dict[str, Any] = {"schema": "weightcode.harness.f6b_summary.v0", "conditions": {}, "files": {}}
    for condition, final in FINALS.items():
        sid, prog = final["session"], final["program12"]
        ids = targets(condition)
        aligned = {r["target"]["instance_id"]: r for r in
                   (read_json(base / condition / f"f6b_{condition}_{i}.json") for i in ids)}
        frozen = {i: read_json(p) for i in ids for p in runs_root().glob(f"f6/{sid}/frozen_*_{i}.json")}
        rerun = {i: read_json(runs_root() / "devreadouts" / "learn" / sid / "transfer" / f"transfer_{prog}_{i}.json")
                 for i in ids}
        rows = []
        for i in ids:
            a, f, r = aligned[i], frozen[i], rerun[i]
            if a["status"] != "ok":
                raise F6bError(f"{i}: aligned status {a['status']}")
            rows.append({"instance_id": i, "seed": int(a["target"]["seed"]),
                         "aligned_gain": points(a["readouts"]["delta_acc"]), "aligned_clean": points(a["readouts_clean"]["delta_acc"]),
                         "aligned_within_limits": a["within_limits"],
                         "frozen_gain": points(f["readouts"]["delta_acc"]), "frozen_clean": points(f["readouts_clean"]["delta_acc"]),
                         "rerun_gain": points(r["readouts"]["delta_acc"]), "rerun_clean": points(r["readouts_clean"]["delta_acc"]),
                         "gate_passed": a["gate"]["passed"]})
        mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
        rerun_mean = mean("rerun_gain")
        share = mean("aligned_gain") / rerun_mean if rerun_mean > 0 else None
        corr = {i: aligned[i]["alignment"]["stats"] for i in ids}
        result["conditions"][condition] = {
            "rows": rows, "aligned_mean_gain": round(mean("aligned_gain"), 3), "aligned_mean_clean": round(mean("aligned_clean"), 3),
            "frozen_mean_gain": round(mean("frozen_gain"), 3), "rerun_mean_gain": round(rerun_mean, 3),
            "rerun_mean_clean": round(mean("rerun_clean"), 3),
            "aligned_share_of_rerun": None if share is None else round(share, 4), "reading": share_reading(share),
            "frozen_share_of_rerun": round(mean("frozen_gain") / rerun_mean, 4) if rerun_mean > 0 else None,
            "gates_passed": all(row["gate_passed"] for row in rows),
            "alignment_stats": corr,
            "descriptive": {i: aligned[i].get("descriptive") for i in ids}}
        for i in ids:
            result["files"][f"{condition}/f6b_{condition}_{i}.json"] = file_sha256(base / condition / f"f6b_{condition}_{i}.json")
    for condition, t in result["conditions"].items():
        print(f"== {condition}: gates passed {t['gates_passed']}")
        print("   seed instance   aligned gain/clean   F6 frozen gain   E3 rerun gain/clean")
        for r in t["rows"]:
            print(f"   {r['seed']}  {r['instance_id']}   {r['aligned_gain']:+8.3f} {r['aligned_clean']:+7.3f}   "
                  f"{r['frozen_gain']:+8.3f}   {r['rerun_gain']:+8.3f} {r['rerun_clean']:+7.3f}")
        print(f"   means: aligned {t['aligned_mean_gain']:+.3f} (clean {t['aligned_mean_clean']:+.3f}), frozen "
              f"{t['frozen_mean_gain']:+.3f}, rerun {t['rerun_mean_gain']:+.3f} (clean {t['rerun_mean_clean']:+.3f})")
        print(f"   aligned / rerun = {t['aligned_share_of_rerun']} -> {t['reading']}")
    if args.out:
        write_json(Path(args.out), result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("rows", help="write the 2,048 clean train rows of the alignment")
    listing = sub.add_parser("targets", help="the five target instance ids of a condition")
    listing.add_argument("--condition", choices=sorted(FINALS), required=True)
    run = sub.add_parser("run", help="align, gate and (if the gate passes) read one target")
    run.add_argument("--condition", choices=sorted(FINALS), required=True)
    run.add_argument("--instance", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--gate-only", action="store_true", help="stop after the validity gate (a smoke; reads no gain)")
    summary = sub.add_parser("summary", help="the tables and the readings fixed in the entry")
    summary.add_argument("--reads", help="directory holding <condition>/f6b_*.json (default runs/f6b)")
    summary.add_argument("--out")
    args = parser.parse_args(argv)
    return {"rows": cmd_rows, "run": cmd_run, "summary": cmd_summary, "targets": cmd_targets}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
