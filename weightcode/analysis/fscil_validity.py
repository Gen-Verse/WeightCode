"""PREREG E9 validity gate (task validity; it reads no arm): the session-1 student against the all-class reference.

Both students are dev seed 100 ResNet-20s with the cosine head in the fewshot store of ``cifar100_fscil``: the base
(role fscil_base, the base classes only) and the all-class reference (role fscil_allclass, the same recipe on the base
train set plus ref_pool and the shots of every new class). Read under the session-2 mask (argmax over outputs 0-64) on
the session-2 readout rows of ``weightcode.evalkit.fscil.readout_rows``: old_select for the base classes (1800) and the
V_select rows of the session-2 classes (1000).

- V1: the fraction of the readout rows the base predicts in a session-2 class; passes at <= ``V1_MAX_FRACTION``.
  The same fraction over all 8000 V_select rows is reported with it (not gating).
- V2, headroom: H = all-seen class-balanced accuracy of the reference minus that of the base, in points, with a
  per-class paired SE; passes at >= ``V2_MIN_POINTS``.
- V3, the retention criterion R: D_old = base-class accuracy (old_select, 30 rows per class, so pooled equals
  class-balanced) of the reference minus that of the base, with its paired SE and discordant counts.
  R = ``RETENTION_DEFAULT`` if D_old >= it, else R = D_old.

A miss of V1 or V2 pauses E9. Context, never gating: both students' masked readouts at every session 1 to 9 (the
all-class reference's joint accuracy per session is the E9 context line), from one forward over old_select and
V_select per student.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from weightcode.analysis.arxiv_headroom import paired_difference
from weightcode.data.store import dataset_json_path
from weightcode.evalkit import fscil as FS
from weightcode.evalkit.targets import file_sha256, load_checked_split, load_dataset_forward, read_dataset_meta, split_path
from weightcode.models.export import read_fewshot_index

V1_MAX_FRACTION = 0.01
V2_MIN_POINTS = 3.0
RETENTION_DEFAULT = -1.0
GATE_SESSION = 2


def retention_criterion(d_old_points: float) -> float:
    """R of PREREG E9 V3: -1.0 if D_old >= -1.0, else D_old (points)."""
    return RETENTION_DEFAULT if d_old_points >= RETENTION_DEFAULT else float(d_old_points)


def balanced_difference(y: np.ndarray, correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """Class-balanced accuracy of b minus a on the same rows, in points, with the per-class paired SE.

    bal = mean over classes of the class accuracy; SE^2 = sum_c var_c / n_c / C^2 with var_c the sample variance of
    the per-row differences b - a in class c.
    """
    y = np.asarray(y)
    a, b = np.asarray(correct_a, dtype=np.float64), np.asarray(correct_b, dtype=np.float64)
    classes = np.unique(y)
    means, variances = [], []
    for c in classes:
        d = b[y == c] - a[y == c]
        means.append(d.mean())
        variances.append(d.var(ddof=1) / d.size if d.size > 1 else 0.0)
    bal_a = float(np.mean([a[y == c].mean() for c in classes]))
    bal_b = float(np.mean([b[y == c].mean() for c in classes]))
    return {"n": int(y.size), "n_classes": int(classes.size), "bal_a": bal_a, "bal_b": bal_b,
            "diff_points": float(100.0 * np.mean(means)),
            "se_points": float(100.0 * math.sqrt(float(np.sum(variances))) / classes.size)}


def _student(role: str, seed: int) -> dict[str, Any]:
    rows = [row for row in read_fewshot_index(FS.DATASET)
            if row.get("role") == role and row.get("arch") == FS.ARCH and int(row["seed"]) == int(seed)]
    if len(rows) != 1:
        raise ValueError(f"expected one {role} student of seed {seed}, found {len(rows)}")
    return rows[0]


def _logits(forward: Any, params: Mapping[str, np.ndarray], x: np.ndarray, batch: int) -> np.ndarray:
    parts = [np.asarray(forward(dict(params), x[start : start + batch])) for start in range(0, len(x), batch)]
    logits = np.concatenate(parts)
    if logits.shape != (len(x), FS.N_CLASSES) or not np.all(np.isfinite(logits)):
        raise FloatingPointError(f"forward returned logits of shape {logits.shape} or non-finite values")
    return logits


def validity_report(seed: int = FS.DEV_SEED, batch_size: int = FS.EVALUATION_BATCH) -> dict[str, Any]:
    """V1 to V3 of PREREG E9 for one seed, with both students' readouts at every session (module docstring)."""
    meta = read_dataset_meta(FS.DATASET)
    forward, forward_sha, _ = load_dataset_forward(FS.DATASET, meta)
    rows = {"base": _student(FS.BASE_ROLE, seed), "reference": _student(FS.ALLCLASS_ROLE, seed)}
    params = {}
    for role, row in rows.items():
        _, values, private, _ = FS.model_record(row, FS.BASE_ROLE if role == "base" else FS.ALLCLASS_ROLE)
        expected = list(range(FS.N_BASE_CLASSES if role == "base" else FS.N_CLASSES))
        if private.get("train_classes") != expected:
            raise ValueError(f"{role} student {row['model_id']} has train_classes {private.get('train_classes')}")
        params[role] = values
    v_sel = load_checked_split(split_path(FS.DATASET, "V_select"), meta["splits"]["V_select"]["sha256"])
    o_sel = load_checked_split(split_path(FS.DATASET, "old_select"), meta["splits"]["old_select"]["sha256"])
    x_all = np.ascontiguousarray(np.concatenate([o_sel["x"], v_sel["x"]]))
    y_all = np.concatenate([o_sel["y"], v_sel["y"]]).astype(np.int64)
    logits = {role: _logits(forward, values, x_all, batch_size) for role, values in params.items()}

    def session_rows(t: int) -> tuple[np.ndarray, np.ndarray]:
        """Positions in x_all of the session's readout rows (readout_rows order) and their parts."""
        in_old = np.arange(len(o_sel["y"]))
        y_v = np.asarray(v_sel["y"])
        prev = len(o_sel["y"]) + np.flatnonzero(np.isin(y_v, FS.prev_classes(t)))
        new = (len(o_sel["y"]) + np.flatnonzero(np.isin(y_v, FS.session_classes(t)))
               if t >= FS.FIRST_SESSION else np.zeros(0, dtype=np.int64))
        part = np.array(["old"] * in_old.size + ["prev"] * prev.size + ["new"] * new.size)
        return np.concatenate([in_old, prev, new]).astype(np.int64), part

    per_session = {}
    for t in range(1, FS.LAST_SESSION + 1):
        position, part = session_rows(t)
        per_session[str(t)] = {role: FS.accuracy_summary(logits[role][position, : FS.n_seen(t)].argmax(axis=1),
                                                         y_all[position], part, t) for role in logits}
        for role in logits:
            per_session[str(t)][role].pop("per_class")

    position, part = session_rows(GATE_SESSION)
    y = y_all[position]
    k = FS.n_seen(GATE_SESSION)
    pred = {role: logits[role][position, :k].argmax(axis=1) for role in logits}
    correct = {role: pred[role] == y for role in pred}
    new_classes = FS.session_classes(GATE_SESSION)
    in_base_classes = part == "old"
    v1_count = int(np.count_nonzero(np.isin(pred["base"], new_classes)))
    v1_fraction = v1_count / y.size
    base_all_vselect = logits["base"][len(o_sel["y"]) :, :k].argmax(axis=1)
    h = balanced_difference(y, correct["base"], correct["reference"])
    h_pooled = paired_difference(correct["base"], correct["reference"])
    d_old = paired_difference(correct["base"][in_base_classes], correct["reference"][in_base_classes])
    r = retention_criterion(d_old["diff_points"])
    in_new = part == "new"
    reference_new = {str(c): {"n": int(np.count_nonzero(y == c)),
                              "acc_reference": float(np.mean(correct["reference"][y == c])),
                              "acc_base": float(np.mean(correct["base"][y == c]))} for c in new_classes}
    gate = "pass" if v1_fraction <= V1_MAX_FRACTION and h["diff_points"] >= V2_MIN_POINTS else "fail"
    return {
        "prereg": "E9 validity gate (V1 to V3), reads no arm",
        "seed": int(seed),
        "dataset": FS.DATASET,
        "dataset_json_sha256": file_sha256(dataset_json_path(FS.DATASET)),
        "forward_sha256": forward_sha,
        "splits": {name: meta["splits"][name]["sha256"] for name in ("V_select", "old_select")},
        "base": {"model_id": rows["base"]["model_id"], "params_sha256": rows["base"]["params_sha256"]},
        "reference": {"model_id": rows["reference"]["model_id"], "params_sha256": rows["reference"]["params_sha256"]},
        "gate_session": GATE_SESSION,
        "n_seen": k,
        "n_rows": int(y.size), "n_rows_new": int(in_new.sum()), "n_rows_base_classes": int(in_base_classes.sum()),
        "V1": {"base_pred_in_new_fraction": v1_fraction, "base_pred_in_new_count": v1_count,
               "rows": "session-2 readout rows (old_select + V_select of classes 60-64)",
               "all_vselect_fraction": float(np.mean(np.isin(base_all_vselect, new_classes))),
               "all_vselect_count": int(np.count_nonzero(np.isin(base_all_vselect, new_classes))),
               "max_fraction": V1_MAX_FRACTION, "pass": bool(v1_fraction <= V1_MAX_FRACTION)},
        "V2": {**h, "pooled": h_pooled, "min_points": V2_MIN_POINTS, "pass": bool(h["diff_points"] >= V2_MIN_POINTS)},
        "V3": {**d_old, "retention_default": RETENTION_DEFAULT, "R": r},
        "session2_new_classes": {
            "bal_reference": float(np.mean([v["acc_reference"] for v in reference_new.values()])),
            "bal_base": float(np.mean([v["acc_base"] for v in reference_new.values()])),
            "per_class": reference_new,
        },
        "per_session": per_session,
        "gate": gate,
    }


__all__ = ["RETENTION_DEFAULT", "V1_MAX_FRACTION", "V2_MIN_POINTS", "balanced_difference", "retention_criterion",
           "validity_report"]
