"""PREREG E7 validity gate (task validity; it reads no arm): the newclass base against the all-class reference.

Both students are dev seed 100 of the inductive protocol (``ogbn_arxiv_ind``) with E6's recipe and graph: the base
is the newclass student of the class draw (``weightcode.data.arxiv_classes``; fewshot store), the all-class reference is
E6's inductive base (role ``base``, every label of the train split). Read on V_select with 40-way argmax:

- V1: the fraction of V_select rows the base predicts in a new class; passes at <= ``V1_MAX_FRACTION``.
- V2, headroom: H = all-class accuracy of the reference minus that of the base, in points, with its paired SE and
  discordant counts; passes at >= ``V2_MIN_POINTS``.
- V3, the retention criterion R: D_old = old-part accuracy (rows of the train classes) of the reference minus that
  of the base. R = ``RETENTION_DEFAULT`` if D_old >= it, else R = D_old.

A miss of V1 or V2 pauses the condition. Reported with them, not gating: the reference's new-class accuracy pooled,
class-balanced and per class, and both students on old_select cut by class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from weightcode.analysis.arxiv_headroom import paired_difference
from weightcode.common.identity import load_params
from weightcode.data import arxiv as A
from weightcode.data.arxiv_classes import class_draw_block
from weightcode.data.store import models_dir, read_dataset_json
from weightcode.models.export import fewshot_models_dir, read_fewshot_index, read_index
from weightcode.models.gcn_export import checked_split
from weightcode.models.importer import checked_forward

V1_MAX_FRACTION = 0.01
V2_MIN_POINTS = 3.0
RETENTION_DEFAULT = -1.0
DATASET = A.INDUCTIVE_DATASET


def _row(rows: list[dict[str, Any]], what: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError(f"expected one {what}, found {len(rows)}")
    return rows[0]


def retention_criterion(d_old_points: float) -> float:
    """R of PREREG E7 V3: -1.0 if D_old >= -1.0, else D_old (points)."""
    return RETENTION_DEFAULT if d_old_points >= RETENTION_DEFAULT else float(d_old_points)


def validity_report(seed: int = 100) -> dict[str, Any]:
    """V1 to V3 of PREREG E7 for one seed (module docstring)."""
    manifest = read_dataset_json(DATASET)
    draw = class_draw_block(DATASET)
    new_classes, train_classes = draw["new_classes"], draw["train_classes"]
    base_row = _row([r for r in read_fewshot_index(DATASET) if r["role"] == "newclass" and r["seed"] == seed
                     and r["train_classes"] == train_classes], f"newclass student of seed {seed}")
    ref_row = _row([r for r in read_index(DATASET) if r["role"] == "base" and r["seed"] == seed],
                   f"all-class base of seed {seed}")
    forward, forward_sha = checked_forward(DATASET, manifest)
    params = {"base": load_params(fewshot_models_dir(DATASET) / base_row["model_id"] / "model.npz"),
              "reference": load_params(models_dir(DATASET) / ref_row["model_id"] / "model.npz")}
    splits = {name: checked_split(DATASET, name, manifest) for name in ("V_select", "old_select")}
    nodes = np.concatenate([splits["V_select"]["x"], splits["old_select"]["x"]])
    n_sel = splits["V_select"]["x"].size
    pred = {}
    for role, values in params.items():
        p = np.argmax(forward(dict(values), nodes), axis=1)
        pred[role] = {"V_select": p[:n_sel], "old_select": p[n_sel:]}
    y = splits["V_select"]["y"]
    in_new = np.isin(y, new_classes)
    correct = {role: pred[role]["V_select"] == y for role in pred}
    v1_fraction = float(np.mean(np.isin(pred["base"]["V_select"], new_classes)))
    h = paired_difference(correct["base"], correct["reference"])
    d_old = paired_difference(correct["base"][~in_new], correct["reference"][~in_new])
    r = retention_criterion(d_old["diff_points"])
    per_class = {str(c): {"n": int(np.count_nonzero(y == c)),
                          "acc_reference": float(np.mean(correct["reference"][y == c]))} for c in new_classes}
    y_o = splits["old_select"]["y"]
    new_o = np.isin(y_o, new_classes)
    old_select = {part: {role: float(np.mean(pred[role]["old_select"][mask] == y_o[mask])) for role in pred}
                  | {"n": int(mask.sum())} for part, mask in (("old", ~new_o), ("new", new_o))}
    return {
        "prereg": "E7 validity gate (V1 to V3), reads no arm",
        "seed": seed,
        "dataset": DATASET,
        "forward_sha256": forward_sha,
        "class_draw": draw,
        "base": {"model_id": base_row["model_id"], "params_sha256": base_row["params_sha256"]},
        "reference": {"model_id": ref_row["model_id"], "params_sha256": ref_row["params_sha256"]},
        "n_vselect": int(y.size), "n_vselect_new": int(in_new.sum()), "n_vselect_old": int((~in_new).sum()),
        "V1": {"base_pred_in_new_fraction": v1_fraction,
               "base_pred_in_new_count": int(np.count_nonzero(np.isin(pred["base"]["V_select"], new_classes))),
               "max_fraction": V1_MAX_FRACTION, "pass": bool(v1_fraction <= V1_MAX_FRACTION)},
        "V2": {**h, "min_points": V2_MIN_POINTS, "pass": bool(h["diff_points"] >= V2_MIN_POINTS)},
        "V3": {**d_old, "retention_default": RETENTION_DEFAULT, "R": r},
        "reference_new_class": {
            "acc_pooled": float(np.mean(correct["reference"][in_new])),
            "acc_balanced": float(np.mean([v["acc_reference"] for v in per_class.values()])),
            "per_class": per_class,
            "acc_base_pooled": float(np.mean(correct["base"][in_new])),
        },
        "old_select": old_select,
        "gate": "pass" if v1_fraction <= V1_MAX_FRACTION and h["diff_points"] >= V2_MIN_POINTS else "fail",
    }
