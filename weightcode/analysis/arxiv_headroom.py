"""PREREG E6 headroom report of one ogbn-arxiv protocol: the gate and its diagnostics, before any instance exists.

Gate (task validity; it reads no arm), dev seed 100. G_new = ceiling minus base accuracy on V_select (2018), in
points; G_old = the same on old_select (up to 2017); G_shift = G_new - G_old. A protocol passes if G_new >= 3 and
G_shift >= 2. Each difference is paired over the same nodes: its SE is sd(c_i - b_i) / sqrt(n) with c_i, b_i the 0/1
correctness of ceiling and base on node i (ddof 1), and its discordance is the count of nodes only one of them gets
right, each way. G_shift's SE combines the two independent node sets: sqrt(SE_new^2 + SE_old^2).

Diagnostics (not gating; the base only; fitted on the ceiling's data, valid_unused and old_replay, read on V_select
and old_select as gain and old change in points against the base):

* ``prior_oracle``: conv3.bias moved by the E1 label-shift direction (``label_shift_direction``: add-one label
  counts against the mean base softmax, centred) estimated on all valid_unused nodes, at fraction 1.
  ``recovered_fraction`` = its gain / G_new.
* ``last_many``: the E1 ``ridge_last`` refit of conv3 on the aggregated penultimate rows Z = (A H2)[ids] of
  valid_unused plus old_replay, with E1's Gauss-Newton targets and lam = s * mean ||[z, 1]||^2 for the 8 scales of
  ``RIDGE_SCALES``; each solution is read unbounded (``free``) and with its conv3.weight delta scaled into the
  rel-Frob limit (``bounded``; the bias is 1-D and exempt). All 8 are reported; ``best_joint`` picks the scale with
  the highest gain + old change on V_select itself, an optimistic read stated as such.
* ``separability``: AUC of 2018 (V_select) against up to 2017 (old_select) nodes for (i) degree alone, score =
  -degree in the deployment graph (newer papers have fewer citations; direction fixed before any reading), and (ii) an
  L2 logistic regression (penalty ``LOGISTIC_L2`` on the standardized non-intercept weights, Newton steps) on
  [log(1 + degree), the 128 features], fitted on one draw of 64 valid_unused and 64 old_replay nodes,
  ``numpy.random.default_rng([split_seed, SEPARABILITY_STREAM])``, standardized with the draw's mean and sd.

Activations come from the dataset's forward.py internals (``_load_graph``, ``_aggregate``, ``_batchnorm``) in
float64; the logits rebuilt from them must match forward.py's output to within its float32 cast.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from weightcode.baselines.fit.edits import MAX_REL_FROB, scale_to_norm_limit
from weightcode.baselines.fit.ridge import RIDGE_SCALES, ridge_solve, ridge_targets, softmax
from weightcode.baselines.fit.rules import label_shift_direction
from weightcode.common.identity import load_params
from weightcode.data import arxiv as A
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.store import models_dir, read_dataset_json
from weightcode.evalkit.patch_check import ForwardFn
from weightcode.models.export import read_index
from weightcode.models.gcn_export import checked_graph, checked_split
from weightcode.models.importer import checked_forward
from weightcode.models.parity import FLOAT32_STEP

GATE_G_NEW = 3.0
GATE_G_SHIFT = 2.0
SEPARABILITY_STREAM = 704
SEPARABILITY_DRAW = 64
LOGISTIC_L2 = 1.0
READ_SPLITS: tuple[str, ...] = ("V_select", "old_select")
FIT_SPLITS: tuple[str, ...] = ("valid_unused", "old_replay")
LAST_WEIGHT, LAST_BIAS = "conv3.weight", "conv3.bias"


def paired_difference(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """b minus a on the same nodes, in points, with its paired SE and the discordant counts."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("need two 0/1 vectors over the same n >= 2 nodes")
    d = b.astype(np.float64) - a.astype(np.float64)
    return {
        "n": int(a.size),
        "acc_a": float(a.mean()),
        "acc_b": float(b.mean()),
        "diff_points": float(100.0 * d.mean()),
        "paired_se_points": float(100.0 * d.std(ddof=1) / math.sqrt(a.size)),
        "only_b_right": int(np.count_nonzero(b & ~a)),
        "only_a_right": int(np.count_nonzero(a & ~b)),
        "discordant_fraction": float(np.mean(a != b)),
    }


def auc(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted 1/2 (Mann-Whitney with mid-ranks)."""
    pos, neg = np.asarray(score_pos, np.float64), np.asarray(score_neg, np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need positives and negatives")
    scores = np.concatenate([pos, neg])
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    midranks = ends - (counts - 1) / 2.0
    rank_sum = float(midranks[inverse[: pos.size]].sum())
    return (rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = LOGISTIC_L2, max_iter: int = 100,
                 tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """L2-penalized logistic regression by Newton steps; the intercept is not penalized. Returns (w, b)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    n, d = x.shape
    design = np.hstack([x, np.ones((n, 1))])
    penalty = np.full(d + 1, l2)
    penalty[-1] = 0.0
    theta = np.zeros(d + 1)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(design @ theta)))
        grad = design.T @ (p - y) + penalty * theta
        hess = (design * (p * (1.0 - p))[:, None]).T @ design + np.diag(penalty) + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if float(np.max(np.abs(step))) < tol:
            return theta[:-1], float(theta[-1])
    raise RuntimeError(f"logistic regression did not converge in {max_iter} Newton steps")


def activations(forward: ForwardFn, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Float64 Z = A H2 (the input rows of conv3) and logits = Z W3^T + b3 at every node, checked against forward.py."""
    module = forward.__globals__
    graph = module["_load_graph"]()
    n_layers = module["_n_layers"](params)
    if n_layers != 3:
        raise ValueError(f"the headroom report expects 3 layers, got {n_layers}")
    tensor = module["_tensor"]
    h = graph["aggregated_features"] @ tensor(params, "conv1.weight").T + tensor(params, "conv1.bias")
    h = np.maximum(module["_batchnorm"](h, params, "bn1"), 0.0)
    h = module["_aggregate"](h @ tensor(params, "conv2.weight").T, graph) + tensor(params, "conv2.bias")
    h = np.maximum(module["_batchnorm"](h, params, "bn2"), 0.0)
    z = module["_aggregate"](h, graph)
    del h
    logits = z @ tensor(params, LAST_WEIGHT).T + tensor(params, LAST_BIAS)
    reference = forward(dict(params), np.arange(z.shape[0], dtype=np.int64)).astype(np.float64)
    bound = FLOAT32_STEP * np.maximum(1.0, np.abs(logits))
    if not np.all(np.abs(reference - logits) <= bound):
        raise RuntimeError(f"rebuilt logits differ from forward.py by {float(np.max(np.abs(reference - logits)))}")
    return {"z": z, "logits": logits}


def _read(logits_rows: Mapping[str, np.ndarray], base_correct: Mapping[str, np.ndarray],
          splits: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    """Gain on V_select and old change on old_select of a child, in points against the base."""
    out: dict[str, Any] = {}
    for name in READ_SPLITS:
        correct = np.argmax(logits_rows[name], axis=1) == splits[name]["y"]
        out[name] = paired_difference(base_correct[name], correct)
    out["gain_points"] = out["V_select"]["diff_points"]
    out["old_change_points"] = out["old_select"]["diff_points"]
    out["joint_points"] = out["gain_points"] + out["old_change_points"]
    return out


def diagnostics(forward: ForwardFn, params: Mapping[str, np.ndarray], graph: Mapping[str, np.ndarray],
                splits: Mapping[str, Mapping[str, np.ndarray]], g_new: float) -> dict[str, Any]:
    """prior_oracle, last_many and separability of the base (module docstring)."""
    acts = activations(forward, params)
    z, logits = acts["z"], acts["logits"]
    base_correct = {name: np.argmax(logits[splits[name]["x"]], axis=1) == splits[name]["y"] for name in READ_SPLITS}
    read_rows = {name: splits[name]["x"] for name in READ_SPLITS}

    unused = splits["valid_unused"]
    stats = label_shift_direction(logits[unused["x"]], unused["y"])
    direction = stats["direction"]
    prior = _read({name: logits[rows] + direction for name, rows in read_rows.items()}, base_correct, splits)
    prior["recovered_fraction"] = prior["gain_points"] / g_new if g_new > 0 else None
    prior["fit"] = {"split": "valid_unused", "n": int(unused["y"].size), "rule": "E1 label_shift_direction, fraction 1",
                    "marginal_gap_l1": float(np.sum(np.abs(stats["p_dep"] - stats["p_pred"])))}

    fit_ids = np.concatenate([splits[name]["x"] for name in FIT_SPLITS])
    fit_y = np.concatenate([splits[name]["y"] for name in FIT_SPLITS])
    h1 = np.hstack([z[fit_ids], np.ones((fit_ids.size, 1))])
    targets, scale = ridge_targets(logits[fit_ids], fit_y)
    w0 = np.hstack([np.asarray(params[LAST_WEIGHT], np.float64), np.asarray(params[LAST_BIAS], np.float64)[:, None]])
    unit = float(np.mean(np.sum(h1 * h1, axis=1)))
    curve = []
    for s in RIDGE_SCALES:
        solution = ridge_solve(h1, targets, w0, s * unit)
        child = {**dict(params), LAST_WEIGHT: solution[:, :-1].astype(np.float32),
                 LAST_BIAS: solution[:, -1].astype(np.float32)}
        bounded, records = scale_to_norm_limit(params, child, (LAST_WEIGHT, LAST_BIAS), limit=MAX_REL_FROB)
        entry: dict[str, Any] = {"scale": s, "lam": s * unit}
        for label, weights in (("free", child), ("bounded", bounded)):
            w = np.asarray(weights[LAST_WEIGHT], np.float64)
            b = np.asarray(weights[LAST_BIAS], np.float64)
            entry[label] = _read({name: z[rows] @ w.T + b for name, rows in read_rows.items()}, base_correct, splits)
        entry["bounded"]["norm_records"] = {key: {k: v for k, v in record.items()} for key, record in records.items()}
        curve.append(entry)
    last_many = {
        "fit": {"splits": list(FIT_SPLITS), "n": int(fit_ids.size), "targets": "E1 gauss_newton", "c": scale,
                "lam_unit": unit},
        "curve": curve,
        "best_joint": {label: max(({"scale": e["scale"], **{k: e[label][k] for k in ("gain_points", "old_change_points",
                                                                                      "joint_points")}}
                                   for e in curve), key=lambda row: row["joint_points"])
                       for label in ("free", "bounded")},
        "best_joint_note": "selected on V_select itself: an optimistic read, not a reference",
    }

    degree = np.diff(np.asarray(graph["indptr"], np.int64)) - 1
    pos, neg = splits["V_select"]["x"], splits["old_select"]["x"]
    rng = np.random.default_rng([SPLIT_SEED, SEPARABILITY_STREAM])
    draw_new = rng.choice(unused["x"], SEPARABILITY_DRAW, replace=False)
    draw_old = rng.choice(splits["old_replay"]["x"], SEPARABILITY_DRAW, replace=False)

    def design(ids: np.ndarray) -> np.ndarray:
        return np.hstack([np.log1p(degree[ids])[:, None], np.asarray(graph["features"], np.float64)[ids]])

    fit_x = design(np.concatenate([draw_new, draw_old]))
    mean, sd = fit_x.mean(axis=0), fit_x.std(axis=0)
    sd[sd == 0] = 1.0
    w, b = fit_logistic((fit_x - mean) / sd, np.r_[np.ones(SEPARABILITY_DRAW), np.zeros(SEPARABILITY_DRAW)])

    def score(ids: np.ndarray) -> np.ndarray:
        return ((design(ids) - mean) / sd) @ w + b

    separability = {
        "degree_auc": auc(-degree[pos], -degree[neg]),
        "degree_median": {"V_select": float(np.median(degree[pos])), "old_select": float(np.median(degree[neg]))},
        "logistic_auc": auc(score(pos), score(neg)),
        "logistic": {"features": "log(1 + degree) and the 128 node features, standardized on the draw",
                     "l2": LOGISTIC_L2, "draw": f"{SEPARABILITY_DRAW} valid_unused + {SEPARABILITY_DRAW} old_replay, "
                     f"default_rng([split_seed, {SEPARABILITY_STREAM}])"},
        "read": "V_select (2018) positive against old_select (up to 2017)",
    }
    base_marginal = softmax(logits[unused["x"]]).mean(axis=0)
    return {"prior_oracle": prior, "last_many": last_many, "separability": separability,
            "base_mean_softmax_valid_unused_max": float(base_marginal.max())}


def _student(dataset: str, role: str, seed: int) -> dict[str, Any]:
    rows = [row for row in read_index(dataset) if row["role"] == role and row["seed"] == seed]
    if len(rows) != 1:
        raise ValueError(f"{dataset}: expected one {role} seed {seed} in the index, found {len(rows)}")
    return rows[0]


def headroom_report(protocol: str, seed: int = 100) -> dict[str, Any]:
    """The E6 headroom gate and diagnostics of one protocol (module docstring)."""
    dataset = A.PROTOCOLS[protocol]
    manifest = read_dataset_json(dataset)
    forward, forward_sha = checked_forward(dataset, manifest)
    graph = checked_graph(dataset, manifest)
    splits = {name: checked_split(dataset, name, manifest) for name in READ_SPLITS + FIT_SPLITS}
    rows = {role: _student(dataset, role, seed) for role in ("base", "ceiling")}
    params = {role: load_params(models_dir(dataset) / row["model_id"] / "model.npz") for role, row in rows.items()}
    correct = {}
    for role, values in params.items():
        logits = forward(dict(values), np.arange(manifest["graph"]["n_nodes"], dtype=np.int64))
        correct[role] = {name: np.argmax(logits[splits[name]["x"]], axis=1) == splits[name]["y"] for name in READ_SPLITS}
        del logits
    new = paired_difference(correct["base"]["V_select"], correct["ceiling"]["V_select"])
    old = paired_difference(correct["base"]["old_select"], correct["ceiling"]["old_select"])
    g_new, g_old = new["diff_points"], old["diff_points"]
    g_shift = g_new - g_old
    shift_se = math.hypot(new["paired_se_points"], old["paired_se_points"])
    report: dict[str, Any] = {
        "experiment": "E6 headroom gate (PREREG, e6-gnn 9da2f39)",
        "protocol": protocol,
        "dataset": dataset,
        "seed": seed,
        "models": {role: {"model_id": row["model_id"], "params_sha256": row["params_sha256"]} for role, row in rows.items()},
        "forward_sha256": forward_sha,
        "graph_sha256": manifest["graph"]["sha256"],
        "split_sha256": {name: manifest["splits"][name]["sha256"] for name in READ_SPLITS + FIT_SPLITS},
        "accuracy": {role: {name: float(np.mean(values[name])) for name in READ_SPLITS} for role, values in correct.items()},
        "G_new": new,
        "G_old": old,
        "G_shift": {"points": g_shift, "se_points": shift_se},
        "gate": {"rule": f"G_new >= {GATE_G_NEW} and G_shift >= {GATE_G_SHIFT}",
                 "G_new_pass": g_new >= GATE_G_NEW, "G_shift_pass": g_shift >= GATE_G_SHIFT,
                 "pass": g_new >= GATE_G_NEW and g_shift >= GATE_G_SHIFT},
    }
    report["diagnostics"] = diagnostics(forward, params["base"], graph, splits, g_new)
    return report
