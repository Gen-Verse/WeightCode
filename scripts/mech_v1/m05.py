"""PREREG M05 phase 1 (mechanism plan v1.0 section 10): the two seed-100 representative E11 programs, as T08 rebuilt
them on each of the six students of their condition, read inside the network: does each rebuilt child carry the same
functional relation (a gate that opens on the condition and stays shut on clean inputs, and a correction that removes
the condition's offset), and how do the target's gain and clean change go with it?

    python scripts/mech_v1/m05.py cell --program gaussian_noise --target i905213a3 --out DIR
    python scripts/mech_v1/m05.py summary --out DIR

Cells are T08's program-arm cells (``runs/t08/program/<session>/transfer_<program sha256[:12]>_<target>.json``), all six
targets of each program (the diagonal included), none picked: the gaussian final sc918e11dceef637a (gate3_lower.py)
and the contrast final s760cb52ce82fb937 (k145_gated_b2_r6_orig_targets.py). Each child is rebuilt from the cell's
derived patch under the cell's limits and must hash to the cell's child_sha256; the gates are read from the cell's own
stdout and checked against the child's weights (``weightcode.analysis.vit_internal``). The target's gain and clean change are
the cell's V_select readouts of record (T08), not recomputed.

Per cell, on the target's V_select condition rows and their clean companions (paired by idx, 4000 each), with the
target's own forward.py arithmetic (``vit_internal.trace``, checked against forward.py on the first chunk):

* gaussian: per printed gate, A2's selectivity in the child (above threshold, fully open, mean opening, gate output norm,
  on patch and class tokens; images with any patch token above; per-class gated rate);
* contrast: M03's gate statistics in the child (gate unit open on patch and class tokens, images with any or a majority
  of patch tokens open, the 8 units' output norm, per-class open rate) and the tokenizer token spread, condition over
  clean;
* both: at each gate block's output, the condition-minus-clean offset in the base, within the child and of the child's
  condition image against the base's clean image, and the edit's move of each split (``vit_internal.OFFSET_PAIRS``).

The diagonal cells must reproduce the records of the same quantities: A2 (runs/devreadouts/e11_mech/a2.jsonl) for the
gaussian final and M03 (runs/mech_v1/r30_.../m03_internal.json, network N_FBR) for the contrast final; rates within two
tokens or images, continuous values within 1e-6 relative. ``summary`` applies the pre-registered reading per target and
the descriptive Spearman correlations over the five non-own targets of each program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "weightcode.mech.m05.v1"
SPEC = "docs/PREREG_v0_DRAFT.md section 0.1, 'T07 and M05 phase 1'; NSCC R38-t07-m05"
N_CLASSES = 10
OPEN_ON_CONDITION_MIN = 0.90
SHUT_ON_CLEAN_MAX = 0.05
RESIDUAL_SHARE_MAX = 0.50
"""Same functional relation rebuilt, per target: the main gate opens on >= 90% of the condition's gated tokens and on
<= 5% of the clean ones, and within the child the condition offset at the main gate's block output (patch tokens, the
norm of the mean offset) is at most half the base's."""
RATE_TOKENS_TOL = 2
REL_TOL = 1e-6

PROGRAMS: dict[str, dict[str, Any]] = {
    "gaussian_noise": {
        "session": "sc918e11dceef637a", "program": "gate3_lower.py",
        "program_sha256": "3c1cebc35ef05b3bc4552a556319509dfac240a4cce5bdc07acbda6c2f0d5946", "own": "i00802425",
        "cells": {"i00802425": "741e42ec20a887990182d91accf61d426f13c9a320876adb41ffe30db31f9735",
                  "i905213a3": "ee7141e762dd3fdf418ddee1e98441738d235be6f104b10580b0852255bb8385",
                  "iab18c0aa": "c0c1f53186827ce92675f4d150ab195b2b8adebb1b170219715f3206e4afbfca",
                  "ib1b195f6": "d99a4a269bf3da6a7a5a52df8712ab7276fe952f87ec120a1dabdc5bfb9dee7c",
                  "ie1be7cd9": "cce5f1964767c4d711d72201df6f10b92276e518836d47f2557f296c74441a1a",
                  "i36df1f8f": "c95e344cda6b346f8dbbbe77f992bd2b1a350d113e05e4b1afdb8440ce35fbd8"}},
    "contrast": {
        "session": "s760cb52ce82fb937", "program": "k145_gated_b2_r6_orig_targets.py",
        "program_sha256": "207b312d4295895e5fed63699f908c9560015e1a8b9ad362935181148dd8495b", "own": "i3eb3fc78",
        "cells": {"i3eb3fc78": "cb230e17a7c6e8b89e7f6c666ee126e605e769246789d65acb6f58072ccdd2d8",
                  "i4a486d70": "46c52c0c018d3bbfdf1c643b405b0b1908f848f9ad80ec05645eface332e4ef7",
                  "i5d76f25c": "bd376bca9fca351fd6dafc86b3cbf95cb9e5b5410c98973a618153f5afc89170",
                  "id83bf155": "7a6195c71bd518d819ea5e9c8e36ce126a53df19dabafb53305d5bd41fac9bf4",
                  "ib1e4e794": "dc9b3d2342594c29a05b87e519e77d2f09040af698911c21824bd480b983303e",
                  "iecf07b07": "053a14c61f59497d06d066827e4f3df1836e69b5da85e35a8996f8b33dd788c0"}},
}
A2_REL = "devreadouts/e11_mech/a2.jsonl"
A2_SHA256 = "10698e042c8c5471d3ac0ce1b01583216b14be48efb87a2ab083579e6dfa7587"
M03_REL = "mech_v1/r30_20261106_205510_85bbf4d14aaf/m03_internal.json"
M03_SHA256 = "372cc755aa892ab948fd6120e240c4021aa9d4def73e9630801733ee5bb595c7"
SPLITS = ("cond", "clean")


class M05Error(RuntimeError):
    """A cell, a gate or a reference does not match its record."""


# ----------------------------------------------------------------------------------------------------------------
# Pure parts (tests/test_mech_m05.py)


def close_rate(ours: float, ref: float, n: int, tokens: int = RATE_TOKENS_TOL) -> bool:
    """Rates over ``n`` tokens or images agree within ``tokens`` of them."""
    return abs(float(ours) - float(ref)) <= tokens / float(n) + 1e-12


def close_value(ours: float, ref: float, rel: float = REL_TOL) -> bool:
    return math.isclose(float(ours), float(ref), rel_tol=rel, abs_tol=1e-12)


def relation_rebuilt(cond_rate: float, clean_rate: float, residual_share: float | None) -> dict[str, Any]:
    """The pre-registered per-target reading (``OPEN_ON_CONDITION_MIN``, ``SHUT_ON_CLEAN_MAX``,
    ``RESIDUAL_SHARE_MAX``)."""
    parts = {"opens_on_condition": cond_rate >= OPEN_ON_CONDITION_MIN,
             "shut_on_clean": clean_rate <= SHUT_ON_CLEAN_MAX,
             "removes_offset": residual_share is not None and residual_share <= RESIDUAL_SHARE_MAX}
    return {**parts, "same_relation": all(parts.values())}


def main_gate(program: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    """The main gate's cond and clean rates, block and offset numbers of one cell result (gaussian: the patch gate,
    above threshold on patch tokens; contrast: the gate unit open on patch tokens)."""
    if program == "gaussian_noise":
        (gate,) = [g for g in cell["gates"] if g["kind"] == "patch"]
        block = int(gate["block"])
        sel = cell["selectivity"][f"patch_block{block}"]
        cond_rate = sel["cond"]["patch"]["frac_above_threshold"]
        clean_rate = sel["clean"]["patch"]["frac_above_threshold"]
        magnitude = sel["cond"]["patch"]["mean_gate_output_norm"]
    else:
        block = int(cell["gate"]["block"])
        cond_rate = cell["gate_stats"]["cond"]["rate_patch_tokens"]
        clean_rate = cell["gate_stats"]["clean"]["rate_patch_tokens"]
        magnitude = cell["gate_stats"]["cond"]["units_output_norm_patch_token_mean"]
    off = cell["offsets"][f"block{block}"]
    base = off["base"]["patch"]["mean_offset_norm"]
    within = off["edited_within"]["patch"]["mean_offset_norm"]
    return {"block": block, "cond_rate": float(cond_rate), "clean_rate": float(clean_rate),
            "correction_norm_cond_patch": float(magnitude), "base_offset_patch": float(base),
            "edited_offset_patch": float(within), "residual_share": (within / base) if base else None,
            "offset_reduction": (1.0 - within / base) if base else None,
            "edited_vs_base_clean_offset_patch": float(off["edited_vs_base_clean"]["patch"]["mean_offset_norm"]),
            "edit_move_clean_patch": float(off["edit_move_clean"]["patch"]["mean_paired_norm"]),
            "edit_move_cond_patch": float(off["edit_move_cond"]["patch"]["mean_paired_norm"])}


# ----------------------------------------------------------------------------------------------------------------
# Files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise M05Error(f"{path} hashes to {actual}, the record says {expected}")


def runs_root() -> Path:
    from weightcode.common.paths import runs_dir

    return Path(runs_dir())


def write_json(path: Path, data: Any) -> str:
    from weightcode.baselines.fit.runner import jsonable

    if path.exists():
        raise M05Error(f"{path} exists; records are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(data), indent=1, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(path)
    return hashlib.sha256(text.encode()).hexdigest()


def provenance() -> dict[str, Any]:
    import weightcode.analysis.vit_internal as V
    from weightcode.baselines.fit.runner import git_state

    return {"script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
            "vit_internal_sha256": file_sha256(Path(V.__file__)), "git": git_state(), "numpy": np.__version__,
            "python": sys.version.split()[0], "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "created_at": datetime.now(UTC).isoformat(), "host": os.uname().nodename,
            "pbs_jobid": os.environ.get("PBS_JOBID"), "spec": SPEC}


def cell_dir(program: str, target: str) -> Path:
    spec = PROGRAMS[program]
    return runs_root() / "t08" / "program" / spec["session"] / f"transfer_{spec['program_sha256'][:12]}_{target}"


def load_cell(program: str, target: str) -> tuple[dict[str, Any], str, Any, dict[str, np.ndarray]]:
    """The T08 cell record, its program stdout, the target instance and the child rebuilt from the cell's patch."""
    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import instances_root
    from weightcode.patch import Limits, apply_patch_file

    spec = PROGRAMS[program]
    if target not in spec["cells"]:
        raise M05Error(f"{target} is not a T08 target of the {program} program")
    folder = cell_dir(program, target)
    require_sha(folder.with_suffix(".json"), spec["cells"][target])
    record = json.loads(folder.with_suffix(".json").read_text(encoding="utf-8"))
    if (record["status"] != "ok" or not record["within_limits"] or record["program"]["sha256"] != spec["program_sha256"]
            or record["target"]["instance_id"] != target or record["target"]["condition"] != f"{program}_s5"):
        raise M05Error(f"{folder}.json is not an ok, within-limits cell of {spec['program']} on {target}")
    stdout = (folder / "run" / "stdout.txt").read_text(encoding="utf-8")
    if stdout != record["run"]["run"]["stdout"]:
        raise M05Error(f"{folder}/run/stdout.txt differs from the stdout the record holds")
    inst = load_instance(instances_root() / target)
    if inst.params_sha256 != record["target"]["params_sha256"] or inst.private.get("split_role") != "dev":
        raise M05Error(f"{target}: base {inst.params_sha256}, the cell's target {record['target']['params_sha256']}")
    result = apply_patch_file(inst.params, folder / "patch" / "derived_patch.json", Limits(**record["limits"]),
                              inst.params_sha256)
    if result.child_sha256 != record["child_sha256"]:
        raise M05Error(f"{target}: rebuilt child {result.child_sha256}, the cell records {record['child_sha256']}")
    return record, stdout, inst, {k: np.asarray(v) for k, v in result.child.items()}


def readout_of_record(record: Mapping[str, Any]) -> dict[str, Any]:
    def points(split: Mapping[str, Any]) -> dict[str, Any]:
        return {"delta_points": 100.0 * float(split["delta_acc"]), "fixed": int(split["fixed"]),
                "broken": int(split["broken"]), "worst_class_points": 100.0 * float(split["worst_class_delta"]),
                "base_acc": float(split["base_acc"]), "child_acc": float(split["child_acc"]), "n": int(split["n"])}

    return {"gain": points(record["readouts"]), "clean": points(record["readouts_clean"]),
            "n_changed": int(record["n_changed"]), "source": "T08 cell record (V_select and clean companion)"}


# ----------------------------------------------------------------------------------------------------------------
# One cell


def cell_internals(program: str, inst: Any, child: Mapping[str, np.ndarray], stdout: str) -> dict[str, Any]:
    import weightcode.analysis.vit_internal as V
    import weightcode.evalkit.devreadout as D
    from weightcode.evalkit.companions import load_clean_rows

    cond = D.readout_split(inst)
    clean = load_clean_rows(inst, D.READOUT_SPLIT)
    if not (np.array_equal(cond["idx"], clean.idx) and np.array_equal(cond["y"], clean.y)):
        raise M05Error(f"{inst.instance_id}: V_select rows and their clean companion are not paired by idx")
    base = {k: np.asarray(v) for k, v in inst.params.items()}
    ns = V.forward_namespace(inst.forward)
    gelu = ns["_gelu"]
    if program == "gaussian_noise":
        gates = V.parse_gaussian_gates(stdout)
        gate_info: Any = V.check_ramp_gates(base, child, gates)
        blocks = sorted({g.block for g in gates})
    else:
        gate_info = V.contrast_gate(base, child, stdout)
        blocks = [int(gate_info["block"])]
    xs = {"cond": cond["x"], "clean": clean.x}
    sums = V.Sums()
    spreads: dict[str, list[np.ndarray]] = {f"{m}_{s}": [] for m in ("base", "edit") for s in SPLITS}
    parity: dict[str, float] = {}
    chunk = int(ns["CHUNK"])
    for start in range(0, len(cond["y"]), chunk):
        y = cond["y"][start:start + chunk]
        traces = {}
        for split, x in xs.items():
            for model, params in (("base", base), ("edit", child)):
                tr = V.trace(ns, params, x[start:start + chunk], blocks)
                if start == 0 and split == "cond":
                    parity[model] = V.parity(ns, params, x[start:start + chunk], tr.logits)
                traces[f"{model}_{split}"] = tr
                spreads[f"{model}_{split}"].append(V.token_spread(tr.tokens))
            mlp = traces[f"edit_{split}"].mlp_in
            if program == "gaussian_noise":
                for g in gates:
                    V.ramp_gate_sums(sums, f"{split}_g{g.block}_", mlp[g.block], child, g, gelu, y, N_CLASSES)
            else:
                V.contrast_gate_sums(sums, f"{split}_gate_", mlp[gate_info["block"]], child, gate_info["units"],
                                     gate_info["gate_unit"], gelu, y, N_CLASSES)
        for b in blocks:
            V.offset_sums(sums, f"b{b}_", {key: tr.block_out[b] for key, tr in traces.items()})
    out: dict[str, Any] = {"n_images": int(len(cond["y"])), "blocks": blocks, "parity_max_abs_logit_diff": parity,
                           "offsets": {f"block{b}": V.offset_finish(sums, f"b{b}_") for b in blocks},
                           "x_sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                        for k, v in xs.items()}}
    spread = {k: np.concatenate(v) for k, v in spreads.items()}
    out["token_spread_cond_over_clean"] = {m: V.ratio_summary(spread[f"{m}_cond"], spread[f"{m}_clean"])
                                           for m in ("base", "edit")}
    if program == "gaussian_noise":
        out["gates"] = gate_info
        out["selectivity"] = {f"{g.kind}_block{g.block}": {s: V.ramp_gate_finish(sums, f"{s}_g{g.block}_", True)
                                                           for s in SPLITS} for g in gates}
    else:
        out["gate"] = gate_info
        out["gate_stats"] = {s: V.contrast_gate_finish(sums, f"{s}_gate_", True) for s in SPLITS}
    return out


def a2_checks(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The gaussian diagonal cell against A2 (same rows, same final, same definitions)."""
    path = runs_root() / A2_REL
    require_sha(path, A2_SHA256)
    records = {r["name"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}
    checks = []
    for gate in cell["gates"]:
        b = int(gate["block"])
        rec = records[f"gate_b{b}"]
        if [rec["gate"]["j1"], rec["gate"]["j2"]] != [gate["j1"], gate["j2"]]:
            raise M05Error(f"block {b}: A2 units {rec['gate']['j1']},{rec['gate']['j2']}, ours {gate['j1']},{gate['j2']}")
        sel = cell["selectivity"][f"{gate['kind']}_block{b}"]
        for split in SPLITS:
            for tok in ("cls", "patch"):
                ours, ref = sel[split][tok], rec["selectivity"][f"{split}_{tok}"]
                n = ref["n_tokens"]
                for key in ("frac_above_threshold", "frac_fully_open"):
                    checks.append({"check": f"A2 b{b} {split} {tok} {key}", "ours": ours[key], "record": ref[key],
                                   "equal": close_rate(ours[key], ref[key], n) and ours["n_tokens"] == n})
                for key in ("mean_opening", "mean_gate_output_norm"):
                    checks.append({"check": f"A2 b{b} {split} {tok} {key}", "ours": ours[key], "record": ref[key],
                                   "equal": close_value(ours[key], ref[key])})
            key = "images_any_patch_above"
            ref = rec["selectivity"][f"{split}_{key}"]
            checks.append({"check": f"A2 b{b} {split} {key}", "ours": sel[split][key], "record": ref,
                           "equal": close_rate(sel[split][key], ref, cell["n_images"])})
        off = cell["offsets"][f"block{b}"]
        for ours_name, ref_name in (("base", "base"), ("edited_within", "final")):
            for tok in ("cls", "patch"):
                for ours_key, ref_key in (("mean_offset_norm", f"{tok}_mean_offset_norm"),
                                          ("mean_paired_norm", f"{tok}_mean_paired_norm")):
                    ours, ref = off[ours_name][tok][ours_key], rec["offsets"][ref_name][ref_key]
                    checks.append({"check": f"A2 b{b} offset {ref_name} {ref_key}", "ours": ours, "record": ref,
                                   "equal": close_value(ours, ref)})
        for split in SPLITS:
            for tok in ("cls", "patch"):
                ours = off[f"edit_move_{split}"][tok]["mean_paired_norm"]
                ref = rec["edit_moves_block_output"][split][f"{tok}_mean_norm"]
                checks.append({"check": f"A2 b{b} edit move {split} {tok}", "ours": ours, "record": ref,
                               "equal": close_value(ours, ref)})
    return checks


def m03_checks(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The contrast diagonal cell against M03's N_FBR (the final), contrast_s5 and clean V_select."""
    path = runs_root() / M03_REL
    require_sha(path, M03_SHA256)
    m03 = json.loads(path.read_text(encoding="utf-8"))
    net = m03["networks"]["N_FBR"]
    checks = [{"check": "M03 gate unit", "ours": cell["gate"]["gate_unit"], "record": m03["gate_unit"],
               "equal": cell["gate"]["gate_unit"] == m03["gate_unit"]},
              {"check": "M03 units", "ours": cell["gate"]["units"], "record": sorted(m03["units"]),
               "equal": cell["gate"]["units"] == sorted(m03["units"])}]
    for split, ref_set in (("cond", "contrast_s5"), ("clean", "clean")):
        ours, ref = cell["gate_stats"][split], net["sets"][ref_set]
        n_img, n_tok = ref["n_images"], ref["n_patch_tokens"]
        for key, n in (("rate_patch_tokens", n_tok), ("rate_cls_token", n_img),
                       ("share_images_any_patch_token_open", n_img), ("share_images_majority_patch_tokens_open", n_img)):
            checks.append({"check": f"M03 {split} {key}", "ours": ours[key], "record": ref["gate"][key],
                           "equal": close_rate(ours[key], ref["gate"][key], n)})
        checks.append({"check": f"M03 {split} open_patch_tokens", "ours": ours["open_patch_tokens"],
                       "record": ref["gate"]["open_patch_tokens"],
                       "equal": abs(ours["open_patch_tokens"] - ref["gate"]["open_patch_tokens"]) <= RATE_TOKENS_TOL})
        for ours_key, ref_key in (("units_output_norm_patch_token_mean", "patch_token_mean"),
                                  ("units_output_norm_cls_token_mean", "cls_token_mean")):
            checks.append({"check": f"M03 {split} {ref_key}", "ours": ours[ours_key],
                           "record": ref["units_output_norm"][ref_key],
                           "equal": close_value(ours[ours_key], ref["units_output_norm"][ref_key])})
    ours = cell["token_spread_cond_over_clean"]["edit"]["ratio_of_means"]
    ref = net["tokenizer_condition_over_clean"]["contrast_s5"]["token_spread"]["ratio_of_means"]
    checks.append({"check": "M03 token spread ratio of means", "ours": ours, "record": ref,
                   "equal": close_value(ours, ref)})
    return checks


def cmd_cell(args: argparse.Namespace) -> int:
    started = time.time()
    spec = PROGRAMS[args.program]
    record, stdout, inst, child = load_cell(args.program, args.target)
    cell = cell_internals(args.program, inst, child, stdout)
    checks: list[dict[str, Any]] = []
    if args.target == spec["own"]:
        checks = a2_checks(cell) if args.program == "gaussian_noise" else m03_checks(cell)
    reproduced = all(c["equal"] for c in checks)
    mg = main_gate(args.program, cell)
    out = {"schema": SCHEMA, "kind": "cell", "program": args.program, "session": spec["session"],
           "program_sha256": spec["program_sha256"], "target": args.target, "own": args.target == spec["own"],
           "cell_record": str(cell_dir(args.program, args.target)) + ".json",
           "cell_record_sha256": spec["cells"][args.target], "child_sha256": record["child_sha256"],
           "base_sha256": inst.params_sha256, "readouts_of_record": readout_of_record(record), **cell,
           "main_gate": mg, "reading": relation_rebuilt(mg["cond_rate"], mg["clean_rate"], mg["residual_share"]),
           "reference_checks": checks, "reference_reproduced": reproduced if checks else None,
           "seconds": round(time.time() - started, 1), "provenance": provenance()}
    sha = write_json(Path(args.out) / f"m05_{args.program}_{args.target}.json", out)
    print(f"M05 {args.program} {args.target}: cond rate {mg['cond_rate']:.4f} clean rate {mg['clean_rate']:.4f} "
          f"residual {mg['residual_share']:.3f} same relation {out['reading']['same_relation']} reference "
          f"{out['reference_reproduced']} ({out['seconds']} s) sha256 {sha}", flush=True)
    return 0 if reproduced else 1


def cmd_summary(args: argparse.Namespace) -> int:
    from weightcode.analysis.vit_internal import spearman

    out_dir = Path(args.out)
    result: dict[str, Any] = {"schema": SCHEMA, "kind": "summary", "programs": {}}
    ok = True
    for program, spec in PROGRAMS.items():
        rows = []
        for target in spec["cells"]:
            cell = json.loads((out_dir / f"m05_{program}_{target}.json").read_text(encoding="utf-8"))
            if cell["cell_record_sha256"] != spec["cells"][target]:
                raise M05Error(f"{program} {target}: cell record hash differs")
            if cell["own"]:
                ok = ok and bool(cell["reference_reproduced"])
            ro = cell["readouts_of_record"]
            rows.append({"target": target, "own": cell["own"], "gain_points": ro["gain"]["delta_points"],
                         "clean_points": ro["clean"]["delta_points"],
                         "worst_class_clean_points": ro["clean"]["worst_class_points"], **cell["main_gate"],
                         **cell["reading"], "reference_reproduced": cell["reference_reproduced"]})
        others = [r for r in rows if not r["own"]]
        corr = {
            "gain_vs_cond_rate": spearman([r["gain_points"] for r in others], [r["cond_rate"] for r in others]),
            "clean_vs_clean_rate": spearman([r["clean_points"] for r in others], [r["clean_rate"] for r in others]),
            "gain_vs_offset_reduction": spearman([r["gain_points"] for r in others],
                                                 [r["offset_reduction"] for r in others]),
            "clean_vs_clean_edit_move": spearman([r["clean_points"] for r in others],
                                                 [r["edit_move_clean_patch"] for r in others]),
            "n_targets": len(others), "reading": "descriptive only (five targets)"}
        result["programs"][program] = {"rows": rows, "spearman_non_own": corr,
                                       "same_relation_count": sum(r["same_relation"] for r in rows),
                                       "n_targets": len(rows)}
        print(f"\n{program} ({spec['session']}): target, gain, clean, cond rate, clean rate, residual share, "
              f"clean edit move, same relation")
        for r in rows:
            print(f"  {r['target']}{' (own)' if r['own'] else ''}: {r['gain_points']:+.3f} {r['clean_points']:+.3f} "
                  f"{r['cond_rate']:.4f} {r['clean_rate']:.4f} {r['residual_share']:.3f} "
                  f"{r['edit_move_clean_patch']:.4f} {r['same_relation']}")
        print(f"  Spearman over the five non-own targets: {corr}")
    result["references_reproduced"] = ok
    result["provenance"] = provenance()
    sha = write_json(out_dir / "m05_summary.json", result)
    print(f"\nreferences reproduced: {ok}; m05_summary.json sha256 {sha}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cell = sub.add_parser("cell")
    cell.add_argument("--program", choices=sorted(PROGRAMS), required=True)
    cell.add_argument("--target", required=True)
    cell.add_argument("--out", required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return cmd_cell(args) if args.cmd == "cell" else cmd_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
