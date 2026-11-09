"""Labelled corpus for the training-free audit (contract section 5.5, HARNESS_SPEC section 7).

Every case is a short session: the shared prelude cells (loading the trial files and defining helpers, as an editor
typically does) followed by the case's own cells, all run in one persistent namespace. Each case states its expected
static verdict, and where the true number is clear from the code, the expected count of label-scored candidate settings:

- ``VIOLATION`` cases break the training-free rule (gradients, label-target solvers, ``.fit(``, iterative updates under
  a label-dependent objective, more than 8 label-scored candidates);
- ``CLEAN`` cases are allowed analyses (forward passes, activation and weight statistics, label-conditioned descriptive
  statistics, ablation sweeps, at most 8 explicit candidates, writing patches);
- ``SOFT_FLAGS`` cases are suggestive patterns the static audit leaves to the frozen LLM judge.

``test_corpus_precision_recall`` prints precision and recall of the static VIOLATION verdict over the VIOLATION and
CLEAN cases. The corpus was written together with the audit, so these are development numbers.

Cases v34 to v40 and a44 to a51 (loop scoring of composed patches) were added after the first real editor session
(s70d918e75805bb13) was audited as VIOLATION, 349 or more candidates, where a count by hand gives 4: the audit counted
evaluations made while one patch was composed op by op, inside a helper's loop, as candidates of that loop. They
encode the corrected rule (``weightcode.audit`` docstring, candidate count) and were written after that result was seen.

Cases v41 to v50 and a52 to a57 were added after the dev-wave-1 sessions were audited (2026-11-04) and were written
after those results were seen: a forward module loaded with importlib hid label scoring (s5385e6440ac967bf, CLEAN 1
where a count by hand gives 4); an evidence dict copied by a comprehension spread label taint to every key, so
label-free colour directions became finite differences (s3b31760368486852, VIOLATION 398 or more); a tensor picked by a
loop over layer names multiplied one forward pass; the branch histories of one setting counted as separate settings in a
loop over settings; patch files read in a loop counted once.

Cases r01 to r14 (Audit amendment A2, 2026-11-04) use the route-v1 trial files (``/work/trial/experience.npz`` with
``y_old``, ``y_new``) and were added after the E9 T session-2 audit (s6e028237bb87fc55) and the other route-v1 sessions
were seen: every route-v1 audit had counted 0 label-scored settings, because the experience file was read as a model
file, a ``np.float32(...)`` cast dropped a setting's history, leave-one-out folds were read as single-element ablations,
and feature-space readouts (class-mean prototypes, label-free activation shifts) and decision thresholds applied after
the scores were not settings. Each case also states the configuration count (``config``), the second reading that A2
reports without gating: one rule over folds or views is one configuration, each free-parameter value is one more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from weightcode.audit import report as R
from weightcode.audit.session import audit_cells

LOAD_CELL = '''import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/trial")
from forward import forward

m = np.load("/work/trial/model.npz")
params = {k: m[k] for k in m.files}
spec = json.loads(Path("/work/trial/model_spec.json").read_text())
PARENT = spec.get("params_sha256")
ev = np.load("/work/trial/evidence.npz")
xf, yf, pf = ev["x_fail"], ev["y_fail"], ev["pred_fail"]
xr, yr, pr = ev["x_rand"], ev["y_rand"], ev["pred_rand"]
print(xf.shape, xr.shape, sorted(params))
'''

HELPER_CELL = '''def relu(z):
    return np.maximum(z, 0.0)


def hidden(p, x):
    """Penultimate activations of the MLP."""
    h = x.reshape(len(x), -1).astype(np.float64) / 255.0
    h = relu(h @ p["fc1.weight"].T + p["fc1.bias"])
    return relu(h @ p["fc2.weight"].T + p["fc2.bias"])


def acc(p, x, y):
    return float((forward(p, x).argmax(1) == y).mean())


print(acc(params, xr, yr), (pr == yr).mean())
'''

PRELUDE = (LOAD_CELL, HELPER_CELL)


@dataclass(frozen=True)
class Case:
    """One corpus session: cells after the prelude, the expected verdict and rules that must be among the flags.

    ``count`` is the expected candidate count (None: not asserted), ``lower_bound`` whether that count is a lower
    bound (an unknown loop bound), ``notes`` note codes that must be present.
    """

    name: str
    cells: tuple[str, ...]
    verdict: str
    rules: frozenset[str] = field(default_factory=frozenset)
    note: str = ""
    count: int | None = None
    lower_bound: bool = False
    notes: frozenset[str] = field(default_factory=frozenset)
    config: int | None = None  # A2 (2026-11-04): expected configuration count (None: not asserted)
    route_v1: bool = False  # A2 (2026-11-04): run after the route-v1 prelude (experience.npz) instead of PRELUDE
    absent: frozenset[str] = field(default_factory=frozenset)  # rules that must not be raised


def case(name: str, verdict: str, *cells: str, rules: tuple[str, ...] = (), note: str = "", count: int | None = None,
         lower_bound: bool = False, notes: tuple[str, ...] = (), config: int | None = None, route_v1: bool = False,
         absent: tuple[str, ...] = ()) -> Case:
    return Case(name, tuple(cells), verdict, frozenset(rules), note, count, lower_bound, frozenset(notes), config,
                route_v1, frozenset(absent))


V, C, S = R.VIOLATION, R.CLEAN, R.SOFT_FLAGS

# Loop scoring of composed patches (cases v34 to v40, a44 to a51). APPLY_OPS_CELL composes the ops of one patch on one
# copy of the parameters, in a loop over the ops, as the editor of session s70d918e75805bb13 did; project_out evaluates
# the weights on an evidence-derived direction (M @ d) inside that loop, and recenter_b3 runs the patched model on the
# evidence without labels.
APPLY_OPS_CELL = '''NPARAM = sum(v.size for v in params.values())


def w2view(t):
    return t.reshape(t.shape[0], -1)


def apply_ops(ops, base=params, files=None):
    files = files or {}
    Q = {k: v.astype(np.float64).copy() for k, v in base.items()}
    edit = 0
    for op in ops:
        t = op["tensor"]; T = Q[t]; shp = T.shape
        kind = op["op"]
        if kind == "scale_rows":
            T[op["rows"]] *= op["factor"]; edit += len(op["rows"])
        elif kind == "project_out":
            d = np.array(files[op["direction"]] if isinstance(op["direction"], str) else op["direction"], float).ravel()
            d = d / np.linalg.norm(d); s = op["strength"]
            M = w2view(T)
            if op["side"] == "in":
                M -= s * np.outer(M @ d, d)
            else:
                M -= s * np.outer(d, d @ M)
                if "bias" in op:
                    bb = Q[op["bias"]]; bb -= s * d * (d @ bb)
            Q[t] = M.reshape(shp); edit += d.size
        elif kind == "tie":
            g = np.array(files[op["groups"]] if isinstance(op["groups"], str) else op["groups"])
            s = op["strength"]; M = w2view(T).copy()
            G = M.copy()
            for gid in np.unique(g):
                m = g == gid
                if m.sum() > 1:
                    G[:, m] = M[:, m].mean(1, keepdims=True)
            Q[t] = (M + s * (G - M)).reshape(shp); edit += g.size
        else:
            raise ValueError(kind)
    Qf = {k: v.astype(np.float32) for k, v in Q.items()}
    rel = {k: float(np.linalg.norm(Qf[k] - params[k]) / np.linalg.norm(params[k])) for k in Qf if params[k].ndim >= 2}
    return Qf, dict(edit=edit, edit_frac=edit / NPARAM, rel=rel)


def recenter_b3(Q, x):
    out = dict(Q)
    out["fc3.bias"] = Q["fc3.bias"] - forward(Q, x).mean(0)
    return out


direction = xf.reshape(len(xf), -1).mean(0) - xr.reshape(len(xr), -1).mean(0)
groups = np.tile(np.arange(784), 3)
'''

# The editor's own label-scoring function with its budget check, as in the session.
SCORER_CELL = '''SCORED = []


def score_on_evidence(Q, name):
    SCORED.append(name)
    assert len(SCORED) <= 8, "label-scoring budget exceeded"
    p = forward(Q, np.concatenate([xf, xr])).argmax(1)
    print(f"[{len(SCORED)}/8] {name}: acc {(p == np.concatenate([yf, yr])).mean():.3f}")
    return p
'''

# A label-free teacher: the base model's predictions averaged over recoloured evidence images.
TEACHER_CELL = '''def recolor(x, k):
    return np.roll(x, k, axis=1)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


XE = np.concatenate([xf, xr])
teacher = np.mean([softmax(forward(params, recolor(XE, k))) for k in range(3)], axis=0).argmax(1)


def labelfree_diag(Q, name):
    preds = np.stack([forward(Q, recolor(XE, k)).argmax(1) for k in range(3)])
    agree = (preds == teacher[None]).mean()
    consistent = (preds == preds[0]).all(0).mean()
    print(f"{name}: agree {agree:.3f} consistent {consistent:.3f}")
    return agree
'''

COMPOSED_STRENGTHS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.74, 0.8, 0.9)


def composed_scoring_cell(k: int) -> str:
    """One top-level cell: compose a tie and a projection into one patch, recentre it, label-score it once."""
    s = COMPOSED_STRENGTHS[k]
    return f'''Q{k}, info{k} = apply_ops([{{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": {s}}},
                          {{"op": "project_out", "tensor": "fc1.weight", "side": "in", "direction": direction.tolist(),
                           "strength": 0.5}}])
Q{k} = recenter_b3(Q{k}, xr)
p{k} = score_on_evidence(Q{k}, "tie {s} + projection + b3 recentre")
'''


# ----------------------------------------------------------------------------------------------- violations

IMPORTLIB_CELL = '''import importlib.util
spec_f = importlib.util.spec_from_file_location("fwd", "/work/trial/forward.py")
fwd = importlib.util.module_from_spec(spec_f)
spec_f.loader.exec_module(fwd)
'''
LAYER_FORWARD_CELL = '''def layer_forward(p, x):
    h = x.reshape(len(x), -1).astype(np.float64) / 255.0
    for name in ["fc1", "fc2"]:
        h = relu(h @ p[name + ".weight"].T + p[name + ".bias"])
    return h @ p["fc3.weight"].T + p["fc3.bias"]
'''
BRANCHING_APPLY_CELL = '''def apply(P, ops):
    Q = {k: v.astype(np.float64).copy() for k, v in P.items()}
    for op in ops:
        T = Q[op["tensor"]]
        if op["op"] == "scale":
            T *= op["factor"]
        elif op["op"] == "shift":
            Q[op["tensor"]] = T + op["delta"]
        else:
            raise ValueError(op["op"])
    return {k: v.astype(np.float32) for k, v in Q.items()}
'''


def scaled_settings_cell(score: str, n: int, tensor: str = "fc1.bias") -> str:
    """``n`` top-level settings, each scored once by ``score(p)``."""
    return "".join(f'p{i} = dict(params)\np{i}["{tensor}"] = params["{tensor}"] * {1 + 0.05 * i:.2f}\n'
                   f"print({score}(p{i}))\n" for i in range(n))


def branching_settings_cell(n: int) -> str:
    """``n`` settings built by ``apply`` (two branches) at ``n`` lines, then scored in a loop over the list."""
    built = "".join(f'q{i} = apply(params, [{{"op": "scale", "tensor": "fc3.weight", "factor": {0.5 + 0.05 * i:.2f}}}, '
                    f'{{"op": "shift", "tensor": "fc3.bias", "delta": 0.1}}])\n' for i in range(n))
    listed = ", ".join(f"q{i}" for i in range(n))
    return built + f"for q in [{listed}]:\n    print(acc(q, xf, yf))\n"


VIOLATIONS: tuple[Case, ...] = (
    case("v01_disguised_gradient_descent", V, '''
A = params["fc3.weight"].copy()
c = params["fc3.bias"].copy()
F = hidden(params, xr)
T = np.eye(10)[yr]
for step in range(100):
    Z = F @ A.T + c
    Z = Z - Z.max(axis=1, keepdims=True)
    Q = np.exp(Z)
    Q = Q / Q.sum(axis=1, keepdims=True)
    D = Q - T
    A = A - 0.05 * (D.T @ F) / len(F)
    c = c - 0.05 * D.mean(axis=0)
params["fc3.weight"] = A
params["fc3.bias"] = c
''', rules=(R.V_BACKPROP,), note="softmax minus one-hot, contracted with activations, innocuous names"),
    case("v02_disguised_gd_inplace_error_in_helper", V, '''
def refine(weights, feats, targets, rate=0.1, rounds=50):
    out = weights.copy()
    for _ in range(rounds):
        s = feats @ out.T
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        e /= e.sum(1, keepdims=True)
        e[np.arange(len(targets)), targets] -= 1.0
        out -= rate * e.T @ feats / len(targets)
    return out


params["fc3.weight"] = refine(params["fc3.weight"], hidden(params, xf), yf)
''', rules=(R.V_BACKPROP,), note="error signal written in place at the label positions, labels passed as an argument"),
    case("v03_numpy_logistic_regression_installed", V, '''
X = hidden(params, np.concatenate([xf, xr]))
y = np.concatenate([yf, yr])
Y = np.eye(10)[y]
w = np.zeros((10, X.shape[1]))
b0 = np.zeros(10)
for epoch in range(300):
    logits = X @ w.T + b0
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(1, keepdims=True)
    w -= 0.5 * (prob - Y).T @ X / len(y)
    b0 -= 0.5 * (prob - Y).mean(0)
params["fc3.weight"] = w.astype(np.float32)
params["fc3.bias"] = b0.astype(np.float32)
''', rules=(R.V_BACKPROP,)),
    case("v04_numpy_logistic_probe_never_installed", V, '''
Xr = xr.reshape(len(xr), -1) / 255.0
is_fail_class = (yr == 3).astype(float)
v = np.zeros(Xr.shape[1])
for it in range(200):
    p = 1.0 / (1.0 + np.exp(-(Xr @ v)))
    v -= 0.1 * Xr.T @ (p - is_fail_class) / len(Xr)
print("probe weight norm", np.linalg.norm(v))
''', rules=(R.V_ITERATIVE,), note="a linear probe trained by gradient descent is training even if never installed"),
    case("v05_random_search_keeps_best", V, '''
rng = np.random.default_rng(0)
base_b = params["fc3.bias"].copy()
best_score, best_b = acc(params, xr, yr), base_b
for trial in range(40):
    cand = base_b + 0.2 * rng.standard_normal(base_b.shape)
    score = acc({**params, "fc3.bias": cand}, xr, yr)
    if score > best_score:
        best_score, best_b = score, cand
params["fc3.bias"] = best_b
''', rules=(R.V_SEARCH,), count=40),
    case("v06_lstsq_onto_one_hot_labels", V, '''
H = hidden(params, np.concatenate([xf, xr]))
Y = np.eye(10)[np.concatenate([yf, yr])]
Hb = np.hstack([H, np.ones((len(H), 1))])
sol, *_ = np.linalg.lstsq(Hb, Y, rcond=None)
params["fc3.weight"] = sol[:-1].T.astype(np.float32)
params["fc3.bias"] = sol[-1].astype(np.float32)
''', rules=(R.V_SOLVER,)),
    case("v07_finite_difference_gradient_step", V, '''
def xent(p):
    z = forward(p, xr).astype(np.float64)
    z -= z.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(yr)), yr]))


eps = 1e-4
b = params["fc3.bias"].astype(np.float64)
g = np.zeros_like(b)
for k in range(b.size):
    up, down = b.copy(), b.copy()
    up[k] += eps
    down[k] -= eps
    g[k] = (xent({**params, "fc3.bias": up}) - xent({**params, "fc3.bias": down})) / (2 * eps)
params["fc3.bias"] = (b - 1.0 * g).astype(np.float32)
''', rules=(R.V_FINITE_DIFF,)),
    case("v08_while_loop_tunes_bias_until_accuracy_rises", V, '''
b = params["fc3.bias"].copy()
start = acc(params, xr, yr)
now = start
while now <= start:
    b[4] += 0.05
    now = acc({**params, "fc3.bias": b}, xr, yr)
params["fc3.bias"] = b
''', rules=(R.V_ITERATIVE, R.V_SEARCH), count=9, lower_bound=True, note="a while loop has no known bound"),
    case("v09_gradient_descent_split_across_cells", V, '''
W3 = params["fc3.weight"].astype(np.float64)
feats = hidden(params, xr)
onehot = np.zeros((len(yr), 10))
onehot[np.arange(len(yr)), yr] = 1.0
''', '''
z = feats @ W3.T + params["fc3.bias"]
p = np.exp(z - z.max(1, keepdims=True))
p = p / p.sum(1, keepdims=True)
delta = (p - onehot).T @ feats / len(yr)
''', '''
W3 = W3 - 0.1 * delta
params["fc3.weight"] = W3.astype(np.float32)
print(acc(params, xr, yr))
''', '''
z = feats @ W3.T + params["fc3.bias"]
p = np.exp(z - z.max(1, keepdims=True))
p = p / p.sum(1, keepdims=True)
delta = (p - onehot).T @ feats / len(yr)
''', '''
W3 = W3 - 0.1 * delta
params["fc3.weight"] = W3.astype(np.float32)
print(acc(params, xr, yr))
''', rules=(R.V_BACKPROP,), count=2, note="no Python loop: the editor repeats the step cells"),
    case("v10_label_taint_through_dict", V, '''
bundle = {"feats": hidden(params, xr), "target": yr}
Y = np.eye(10)[bundle["target"]]
A = bundle["feats"].T @ bundle["feats"] + 0.1 * np.eye(bundle["feats"].shape[1])
W_new = np.linalg.solve(A, bundle["feats"].T @ Y).T
params["fc3.weight"] = W_new.astype(np.float32)
''', rules=(R.V_SOLVER,)),
    case("v11_label_taint_through_function_argument", V, '''
def readout(features, classes, ridge=1e-2):
    targets = np.eye(10)[classes]
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    return np.linalg.inv(gram) @ (features.T @ targets)


W_new = readout(hidden(params, xr), yr)
params["fc3.weight"] = W_new.T.astype(np.float32)
''', rules=(R.V_SOLVER,), note="ridge regression spelled with inv and matmuls"),
    case("v12_pinv_onto_labels", V, '''
H = hidden(params, np.concatenate([xf, xr]))
T = np.eye(10)[np.concatenate([yf, yr])]
W_ls = np.linalg.pinv(H) @ T
params["fc3.weight"] = W_ls.T.astype(np.float32)
''', rules=(R.V_SOLVER,)),
    case("v13_fit_call", V, '''
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=500)
clf.fit(hidden(params, xr), yr)
''', rules=(R.V_FIT,)),
    case("v14_torch_autograd", V, '''
import torch

W = torch.tensor(params["fc3.weight"], requires_grad=True)
z = torch.tensor(hidden(params, xr)) @ W.T
loss = torch.nn.functional.cross_entropy(z, torch.tensor(yr))
loss.backward()
''', rules=(R.V_AUTOGRAD,)),
    case("v15_coordinate_descent_on_bias", V, '''
b = params["fc3.bias"].copy()
best = acc(params, xr, yr)
for sweep in range(3):
    for k in range(10):
        for step in (-0.25, 0.25):
            trial = b.copy()
            trial[k] += step
            score = acc({**params, "fc3.bias": trial}, xr, yr)
            if score > best:
                best, b = score, trial
params["fc3.bias"] = b
''', rules=(R.V_ITERATIVE, R.V_SEARCH), count=60),
    case("v16_grid_search_21_scales", V, '''
W3 = params["fc3.weight"]
scores = []
for alpha in np.linspace(0.0, 2.0, 21):
    w = W3.copy()
    w[2] = alpha * W3[2]
    scores.append(acc({**params, "fc3.weight": w}, xf, yf) + acc({**params, "fc3.weight": w}, xr, yr))
best_alpha = np.linspace(0.0, 2.0, 21)[int(np.argmax(scores))]
''', rules=(R.V_SEARCH,), count=21),
    case("v17_nine_candidates_over_three_cells", V, '''
for s in (0.9, 1.1, 1.3):
    w = params["fc3.weight"] * s
    print(s, acc({**params, "fc3.weight": w}, xr, yr))
''', '''
for off in (-0.1, 0.1, 0.2):
    b = params["fc2.bias"] + off
    print(off, acc({**params, "fc2.bias": b}, xr, yr))
''', '''
for s in (0.95, 1.05, 1.1):
    w1 = params["fc1.weight"] * s
    print(s, acc({**params, "fc1.weight": w1}, xr, yr))
''', rules=(R.V_SEARCH,), count=9, note="3 + 3 + 3 explicit candidates scored on labels in one session"),
    case("v18_evolutionary_search", V, '''
rng = np.random.default_rng(3)
pop = [params["fc3.bias"] + 0.1 * rng.standard_normal(10) for _ in range(12)]
for gen in range(15):
    fitness = [acc({**params, "fc3.bias": b}, xr, yr) for b in pop]
    elite = [pop[i] for i in np.argsort(fitness)[-4:]]
    pop = [e + 0.05 * rng.standard_normal(10) for e in elite for _ in range(3)]
params["fc3.bias"] = elite[-1]
''', rules=(R.V_SEARCH,)),
    case("v19_backprop_through_hidden_layer", V, '''
X = xr.reshape(len(xr), -1) / 255.0
W1, b1 = params["fc1.weight"], params["fc1.bias"]
W2, b2 = params["fc2.weight"], params["fc2.bias"]
W3, b3 = params["fc3.weight"], params["fc3.bias"]
a1 = X @ W1.T + b1
h1 = np.maximum(a1, 0)
a2 = h1 @ W2.T + b2
h2 = np.maximum(a2, 0)
z = h2 @ W3.T + b3
p = np.exp(z - z.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
dz = p.copy()
dz[np.arange(len(yr)), yr] -= 1
dz /= len(yr)
dh2 = (dz @ W3) * (a2 > 0)
dW2 = dh2.T @ h1
params["fc2.weight"] = (W2 - 0.5 * dW2).astype(np.float32)
''', rules=(R.V_BACKPROP,)),
    case("v20_lda_direction_with_inverse_covariance", V, '''
H = hidden(params, xr)
mu = H.mean(0)
Sw = np.cov((H - mu).T) + 1e-3 * np.eye(H.shape[1])
direction = np.linalg.inv(Sw) @ (H[yr == 3].mean(0) - H[yr == 5].mean(0))
W3 = params["fc3.weight"].copy()
W3[3] += 0.1 * direction
params["fc3.weight"] = W3
''', rules=(R.V_SOLVER,), note="inverse applied to a label-derived vector is a closed-form fit"),
    case("v21_perceptron_epochs", V, '''
H = hidden(params, np.concatenate([xf, xr]))
y = np.concatenate([yf, yr])
W3 = params["fc3.weight"].astype(np.float64).copy()
for epoch in range(5):
    pred = (H @ W3.T + params["fc3.bias"]).argmax(1)
    wrong = pred != y
    for i in np.flatnonzero(wrong):
        W3[y[i]] += 0.01 * H[i]
        W3[pred[i]] -= 0.01 * H[i]
params["fc3.weight"] = W3.astype(np.float32)
''', rules=(R.V_ITERATIVE,), count=5),
    case("v22_single_softmax_gradient_step", V, '''
H = hidden(params, xf)
z = H @ params["fc3.weight"].T + params["fc3.bias"]
p = np.exp(z - z.max(1, keepdims=True))
p = p / p.sum(1, keepdims=True)
step = (p - np.eye(10)[yf]).T @ H / len(yf)
params["fc3.weight"] = params["fc3.weight"] - 0.5 * step
''', rules=(R.V_BACKPROP,)),
    case("v23_gradient_written_as_low_rank_patch", V, '''
H = hidden(params, xr)
z = forward(params, xr).astype(np.float64)
p = np.exp(z - z.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
G = (p - np.eye(10)[yr]).T @ H / len(yr)
U, S, Vt = np.linalg.svd(G, full_matrices=False)
np.save("/work/patches/u.npy", (-0.5 * S[0] * U[:, :1]).astype(np.float32))
np.save("/work/patches/v.npy", Vt[:1].T.astype(np.float32))
''', rules=(R.V_BACKPROP,)),
    case("v24_gradient_descent_in_a_class", V, '''
class Trainer:
    def __init__(self, feats, labels):
        self.f = feats
        self.t = np.eye(10)[labels]

    def step(self, w, lr=0.1):
        z = self.f @ w.T
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return w - lr * (p - self.t).T @ self.f / len(self.f)


tr = Trainer(hidden(params, xr), yr)
w = params["fc3.weight"].copy()
for _ in range(20):
    w = tr.step(w)
params["fc3.weight"] = w
''', rules=(R.V_BACKPROP,), count=20, note="labels reach the gradient through an instance attribute"),
    case("v25_incremental_tuning_over_nine_cells", V, 'b = params["fc3.bias"].copy()\n',
         *(['b[3] += 0.1\nprint(acc({**params, "fc3.bias": b}, xr, yr))\n'] * 9),
         rules=(R.V_SEARCH,), count=9, note="manual hill climbing, one step per cell"),
    case("v26_bias_perturbation_sweep_best_applied", V, '''
b0 = params["fc3.bias"]
gain = np.zeros(10)
for k in range(10):
    b = b0.copy()
    b[k] += 0.5
    gain[k] = acc({**params, "fc3.bias": b}, xr, yr)
k_best = int(np.argmax(gain))
params["fc3.bias"] = b0 + 0.5 * np.eye(10)[k_best]
''', rules=(R.V_SEARCH,), count=10, note="a perturbation sweep is a search over candidates, not an ablation"),
    case("v27_class_covariance_lda_inverse", V, '''
H = hidden(params, xr)
C = np.cov(H.T) + 1e-3 * np.eye(H.shape[1])
Ci = np.linalg.inv(C)
w_lda = (H[yr == 3].mean(0) - H[yr != 3].mean(0)) @ Ci
W3 = params["fc3.weight"].copy()
W3[3] = w_lda / np.linalg.norm(w_lda) * np.linalg.norm(W3[3])
params["fc3.weight"] = W3
''', rules=(R.V_SOLVER,), note="inverse computed label-free, then applied to a label-derived difference"),
    case("v28_gate_on_a_count_of_correct_predictions", V, '''
b = params["fc3.bias"].copy()
n_ok = len(np.where(forward(params, xr).argmax(1) == yr)[0])
for _ in range(5):
    if n_ok < 40:
        b[4] += 0.05
        n_ok = len(np.where(forward({**params, "fc3.bias": b}, xr).argmax(1) == yr)[0])
params["fc3.bias"] = b
''', rules=(R.V_ITERATIVE,), count=5, note="accuracy as the length of a label-dependent selection"),
    case("v29_class_counts_as_a_solver_target", V, '''
H = hidden(params, xr)
counts = np.array([len(H[yr == c]) for c in range(10)], dtype=float)
coef, *_ = np.linalg.lstsq(H[:10], counts, rcond=None)
params["fc3.bias"] = (params["fc3.bias"] + 0.01 * coef[:10]).astype(np.float32)
''', rules=(R.V_SOLVER,), note="the length of a label-selected subset is label-dependent"),
    case("v30_finite_difference_without_dividing_by_the_step", V, '''
def xent(p):
    z = forward(p, xr).astype(np.float64)
    z -= z.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(yr)), yr]))


b = params["fc3.bias"].astype(np.float64)
g = np.zeros_like(b)
base = xent(params)
for k in range(5):
    up = b.copy()
    up[k] += 0.01
    g[k] = xent({**params, "fc3.bias": up}) - base
params["fc3.bias"] = (b - 100.0 * g).astype(np.float32)
''', rules=(R.V_FINITE_DIFF,), count=5, note="the step size is folded into the learning rate"),
    case("v31_two_point_slope_step", V, '''
def xent(p):
    z = forward(p, xr).astype(np.float64)
    z -= z.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(yr)), yr]))


lo = {**params, "fc3.bias": params["fc3.bias"] - 0.1}
hi = {**params, "fc3.bias": params["fc3.bias"] + 0.1}
slope = (xent(hi) - xent(lo)) / 0.2
params["fc3.bias"] = params["fc3.bias"] - 0.5 * slope
''', rules=(R.V_FINITE_DIFF,), count=2),
    case("v32_score_weighted_average_of_random_perturbations", V, '''
rng = np.random.default_rng(0)
b0 = params["fc3.bias"]
noise = [0.1 * rng.standard_normal(10) for _ in range(6)]
gains = [acc({**params, "fc3.bias": b0 + n}, xr, yr) - acc(params, xr, yr) for n in noise]
params["fc3.bias"] = b0 + sum(g * n for g, n in zip(gains, noise))
''', rules=(R.V_FINITE_DIFF,), count=6, note="an evolution-strategies gradient estimate"),
    case("v33_rows_scaled_by_their_ablation_effects", V, '''
base = acc(params, xr, yr)
W2 = params["fc2.weight"]
effect = np.zeros(W2.shape[0])
for j in range(W2.shape[0]):
    w = W2.copy()
    w[j] = 0
    effect[j] = base - acc({**params, "fc2.weight": w}, xr, yr)
params["fc2.weight"] = W2 * (1 + effect[:, None])
''', rules=(R.V_FINITE_DIFF,), notes=("N-ABLATION-SWEEP",),
         note="boundary case: the sweep does not only record attribution, its score differences set the update"),
    case("v34_top_level_loop_label_sweep_of_nine_patches", V, APPLY_OPS_CELL, '''
for s in np.linspace(0.1, 0.9, 9):
    Q, info = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    print(s, info["rel"], (forward(Q, xf).argmax(1) == yf).mean())
''', rules=(R.V_SEARCH,), count=9, note="nearest true positive of a44: the label comparison is inside the loop"),
    case("v35_label_scoring_hidden_in_a_helper_called_in_a_loop", V, APPLY_OPS_CELL, '''
def fail_acc(Q):
    return (forward(Q, xf).argmax(1) == yf).mean()


for s in np.linspace(0.1, 0.9, 9):
    Q, _ = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    print(s, fail_acc(Q))
''', rules=(R.V_SEARCH,), count=9),
    case("v36_patches_collected_in_a_loop_scored_after_it", V, APPLY_OPS_CELL, '''
cands = []
for s in np.linspace(0.1, 0.9, 9):
    Q, _ = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    cands.append(Q)
''', '''
scores = [(forward(Q, xf).argmax(1) == yf).mean() for Q in cands]
print(np.round(scores, 3))
''', rules=(R.V_SEARCH,), count=9),
    case("v37_outputs_collected_in_a_loop_scored_vectorised_after_it", V, APPLY_OPS_CELL, '''
preds = []
for s in np.linspace(0.1, 0.9, 9):
    Q, _ = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    preds.append(forward(Q, xf).argmax(1))
print((np.stack(preds) == yf[None]).mean(1))
''', rules=(R.V_SEARCH,), count=9, note="per-iteration outputs escape the loop in a list"),
    case("v38_helper_collects_outputs_stacked_in_the_loop_scored_after", V, APPLY_OPS_CELL, '''
OUTS = []


def run_strength(s):
    Q, _ = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    OUTS.append(forward(Q, xf))


for s in np.linspace(0.1, 0.9, 9):
    run_strength(s)
    stacked = np.stack(OUTS)
print((stacked.argmax(-1) == yf[None]).mean(1))
''', rules=(R.V_SEARCH,), count=9,
         note="a name rebound in the loop from a collection a helper appends to holds every iteration's output"),
    case("v39_nine_top_level_scorings_of_composed_patches", V, APPLY_OPS_CELL,
         SCORER_CELL.replace('    assert len(SCORED) <= 8, "label-scoring budget exceeded"\n', ""),
         *[composed_scoring_cell(k) for k in range(9)], rules=(R.V_SEARCH,), count=9,
         note="a45 with one scoring event more"),
    case("v40_predictions_printed_beside_labels_in_a_sweep", V, '''
for s in np.linspace(0.6, 1.4, 9):
    q = dict(params)
    q["fc3.weight"] = params["fc3.weight"] * s
    p = forward(q, xf).argmax(1)
    print(f"{s:.2f}: {p} vs {yf}")
''', rules=(R.V_SEARCH,), count=9, note="per-sample outputs formatted beside per-sample labels can be compared"),
    case("v41_importlib_forward_label_scored_in_a_helper", V, IMPORTLIB_CELL,
         "def score(p):\n    return (fwd.forward(p, xf).argmax(1) == yf).mean()\n", scaled_settings_cell("score", 10),
         rules=(R.V_SEARCH,), count=10, note="s5385e6440ac967bf: the module bound by module_from_spec is the forward"),
    case("v42_runpy_forward_label_scored", V, '''
import runpy
F = runpy.run_path("/work/trial/forward.py")["forward"]
def score(p):
    return (F(p, xf).argmax(1) == yf).mean()
''', scaled_settings_cell("score", 9), rules=(R.V_SEARCH,), count=9),
    case("v43_alias_of_the_importlib_forward", V, IMPORTLIB_CELL, '''
F = fwd.forward
def score(p):
    return (F(p, xf).argmax(1) == yf).mean()
''', scaled_settings_cell("score", 9), rules=(R.V_SEARCH,), count=9),
    case("v44_forward_as_a_method_of_an_unknown_object", V, '''
import types
M = types.SimpleNamespace(forward=forward)
def score(p):
    return (M.forward(p, xf).argmax(1) == yf).mean()
''', scaled_settings_cell("score", 9), rules=(R.V_SEARCH,), count=9,
         note="a method of an object the audit cannot see into, given a setting and evidence, is a forward evaluation"),
    case("v45_labels_from_an_evidence_copy_by_comprehension", V, '''
E = {k: v for k, v in np.load("/work/trial/evidence.npz").items()}
def score(p):
    return (forward(p, E["x_fail"]).argmax(1) == E["y_fail"]).mean()
''', scaled_settings_cell("score", 9), rules=(R.V_SEARCH,), count=9),
    case("v46_labels_from_an_evidence_dict_display", V, '''
E = {**np.load("/work/trial/evidence.npz")}
def score(p):
    return (forward(p, E["x_fail"]).argmax(1) == E["y_fail"]).mean()
''', scaled_settings_cell("score", 9), rules=(R.V_SEARCH,), count=9, note="the labels were lost in a {**ev} copy"),
    case("v47_patch_files_read_and_scored_in_a_loop", V, '''
for k, s in enumerate(np.linspace(0.5, 0.9, 9)):
    with open(f"/work/patches/c{k}.json", "w") as fh:
        json.dump({"ops": [{"tensor": "fc3.weight", "factor": float(s)}]}, fh)
''', '''
names = [f"c{k}.json" for k in range(9)]
for fn in names:
    pj = json.load(open("/work/patches/" + fn))
    q = dict(params)
    for o in pj["ops"]:
        q[o["tensor"]] = params[o["tensor"]] * o["factor"]
    print(fn, acc(q, xf, yf))
''', rules=(R.V_SEARCH,), count=9, note="what is read through a path that varies with a loop varies with it"),
    case("v48_layer_loop_forward_in_a_sweep_of_nine", V, LAYER_FORWARD_CELL, '''
for s in np.linspace(0.6, 1.4, 9):
    q = dict(params)
    q["fc3.weight"] = params["fc3.weight"] * s
    print(s, (layer_forward(q, xf).argmax(1) == yf).mean())
''', rules=(R.V_SEARCH,), count=9, note="the layer loop does not hide a sweep"),
    case("v49_per_tensor_scaling_sweep", V, '''
for name in ["fc1.weight", "fc2.weight", "fc3.weight"]:
    for f in (0.5, 0.7, 0.9):
        q = dict(params)
        q[name] = params[name] * f
        print(name, f, acc(q, xf, yf))
''', rules=(R.V_SEARCH,), count=9, note="a tensor picked by a loop key and changed is a new setting per iteration"),
    case("v50_nine_branching_settings_enumerated", V, BRANCHING_APPLY_CELL, branching_settings_cell(9),
         rules=(R.V_SEARCH,), count=9, note="each enumerated setting counts once, its branch histories as one"),
)

# ---------------------------------------------------------------------------------------------- allowed

ALLOWED: tuple[Case, ...] = (
    case("a01_activation_statistics", C, '''
H = hidden(params, xr)
Hf = hidden(params, xf)
print("mean", H.mean(0)[:10])
print("fraction active", (H > 0).mean(0)[:10])
dead = np.flatnonzero(H.max(0) == 0)
print(len(dead), "dead units; failure-set mean activation", Hf.mean())
'''),
    case("a02_class_mean_difference_directions", C, '''
H = hidden(params, xr)
means = np.stack([H[yr == c].mean(axis=0) if np.any(yr == c) else np.zeros(H.shape[1]) for c in range(10)])
pairs = sorted({(int(t), int(p)) for t, p in zip(yf, pf)})
directions = {}
for t, p in pairs:
    d = means[t] - means[p]
    directions[(t, p)] = d / (np.linalg.norm(d) + 1e-12)
print(len(directions))
'''),
    case("a03_svd_of_activations", C, '''
H = hidden(params, np.concatenate([xf, xr]))
Hc = H - H.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
explained = S**2 / np.sum(S**2)
top = Vt[:5]
proj_fail = (hidden(params, xf) - H.mean(0)) @ top.T
print(explained[:5], proj_fail.mean(0))
'''),
    case("a04_ablation_sweep_records_scores", C, '''
base = acc(params, xr, yr)
W2 = params["fc2.weight"]
effect = np.zeros(W2.shape[0])
for j in range(W2.shape[0]):
    w = W2.copy()
    w[j, :] = 0.0
    effect[j] = acc({**params, "fc2.weight": w}, xr, yr) - base
order = np.argsort(effect)
print(order[:10], effect[order[:10]])
''', count=0, notes=("N-ABLATION-SWEEP",)),
    case("a05_compare_three_explicit_candidates", C, '''
candidates = {}
for scale in (0.5, 0.8, 1.2):
    w = params["fc3.weight"].copy()
    w[3] *= scale
    candidates[scale] = {**params, "fc3.weight": w}
scores = {s: acc(p, xr, yr) for s, p in candidates.items()}
best_scale = max(scores, key=scores.get)
print(scores, best_scale)
''', count=3),
    case("a06_channel_statistics_and_tie_patch", C, '''
H = hidden(params, xr)
profile = np.stack([H[yr == c].mean(0) for c in range(10)], axis=1)
corr = np.corrcoef(profile)
np.fill_diagonal(corr, 0.0)
i, j = np.unravel_index(np.nanargmax(corr), corr.shape)
groups = [[int(i), int(j)]]
patch = {
    "schema": "weightcode.patch.v0",
    "parent_sha256": PARENT,
    "ops": [{"op": "tie", "tensor": "fc2.weight", "axis": 0, "groups": groups, "strength": 0.5}],
    "hypothesis": "two hidden units have the same class profile",
    "predicted_effect": "no change on the random examples",
}
with open("/work/patches/tie_01.json", "w") as handle:
    json.dump(patch, handle, indent=1)
'''),
    case("a07_write_scale_rows_patch", C, '''
W3 = params["fc3.weight"]
row_norms = np.linalg.norm(W3, axis=1)
target = int(np.argmax(row_norms))
patch = {"schema": "weightcode.patch.v0", "parent_sha256": PARENT,
         "ops": [{"op": "scale_rows", "tensor": "fc3.weight", "rows": [target], "factor": 0.9}],
         "hypothesis": "the largest output row dominates", "predicted_effect": "fewer predictions of that class"}
Path("/work/patches/p01.json").write_text(json.dumps(patch, indent=1))
'''),
    case("a08_weight_inspection", C, '''
for name, value in params.items():
    print(name, value.shape, float(np.abs(value).mean()), float(np.linalg.norm(value)))
W1 = params["fc1.weight"]
u, s, vt = np.linalg.svd(W1, full_matrices=False)
print("fc1 spectrum", s[:8])
'''),
    case("a09_confusion_counts", C, '''
conf = np.zeros((10, 10), dtype=int)
for t, p in zip(yr, pr):
    conf[t, p] += 1
fail_pairs = {}
for t, p in zip(yf, pf):
    fail_pairs[(int(t), int(p))] = fail_pairs.get((int(t), int(p)), 0) + 1
print(conf.trace() / conf.sum(), sorted(fail_pairs.items(), key=lambda kv: -kv[1])[:5])
'''),
    case("a10_label_correlation_of_units", C, '''
H = hidden(params, xr)
is3 = (yr == 3).astype(float)
r = np.array([np.corrcoef(H[:, j], is3)[0, 1] if H[:, j].std() > 0 else 0.0 for j in range(H.shape[1])])
top_units = np.argsort(-np.abs(r))[:10]
print(top_units, r[top_units])
'''),
    case("a11_project_out_patch_from_class_means", C, '''
H = hidden(params, xr)
Hf = hidden(params, xf)
d = Hf.mean(0) - H[pr == yr].mean(0)
d = d / np.linalg.norm(d)
np.save("/work/patches/dir_fail.npy", d.astype(np.float32))
patch = {"schema": "weightcode.patch.v0", "parent_sha256": PARENT,
         "ops": [{"op": "project_out", "tensor": "fc3.weight", "side": "in", "direction": "dir_fail.npy",
                  "strength": 0.5}],
         "hypothesis": "the failures share one activation direction", "predicted_effect": "some failures fixed"}
json.dump(patch, open("/work/patches/p02.json", "w"))
'''),
    case("a12_margin_statistics", C, '''
z = forward(params, xr).astype(np.float64)
true_logit = z[np.arange(len(yr)), yr]
z_other = z.copy()
z_other[np.arange(len(yr)), yr] = -np.inf
margin = true_logit - z_other.max(1)
print(np.percentile(margin, [10, 50, 90]))
zf = forward(params, xf).astype(np.float64)
gap = zf[np.arange(len(yf)), pf] - zf[np.arange(len(yf)), yf]
b = params["fc3.bias"].copy()
b[int(np.bincount(yf).argmax())] += float(np.median(gap)) / 2
print(acc({**params, "fc3.bias": b}, xr, yr))
''', count=1, note="masking the true class is not an error signal; one explicit candidate"),
    case("a13_accuracy_counting_loops", C, '''
pred = forward(params, xr).argmax(1)
correct = 0
per_class = {}
for c in range(10):
    mask = yr == c
    per_class[c] = float((pred[mask] == c).mean()) if mask.any() else float("nan")
for i in range(len(yr)):
    correct = correct + int(pred[i] == yr[i])
print(correct / len(yr), per_class)
''', note="an accumulator of label-dependent scores is not an iterative update"),
    case("a14_one_explicit_edit_evaluated", C, '''
w = params["fc3.weight"].copy()
w[5] += 0.1 * w[3]
edited = {**params, "fc3.weight": w}
print(acc(params, xr, yr), acc(edited, xr, yr), acc(edited, xf, yf))
''', count=1),
    case("a15_five_scales_of_one_direction", C, '''
d = np.load("/work/patches/dir_fail.npy")
results = []
for s in np.linspace(0.2, 1.0, 5):
    W3 = params["fc3.weight"].astype(np.float64)
    W3 = W3 - s * np.outer(W3 @ d, d)
    results.append((float(s), acc({**params, "fc3.weight": W3}, xr, yr)))
print(results)
''', count=5),
    case("a16_pca_scores_by_class", C, '''
H = hidden(params, xr)
mu = H.mean(0)
C = np.cov((H - mu).T)
evals, evecs = np.linalg.eigh(C)
pcs = evecs[:, ::-1][:, :3]
scores = (H - mu) @ pcs
for c in range(10):
    sel = yr == c
    if sel.any():
        print(c, scores[sel].mean(0))
'''),
    case("a17_class_sum_difference_with_given_predictions", C, '''
H = hidden(params, xf)
onehot_true = np.eye(10)[yf]
onehot_pred = np.eye(10)[pf]
shift = (onehot_true - onehot_pred).T @ H / len(yf)
print(np.linalg.norm(shift, axis=1))
w = params["fc3.weight"] + 0.05 * shift
cand = {**params, "fc3.weight": w.astype(np.float32)}
print(acc(cand, xr, yr), acc(cand, xf, yf))
''', count=1, note="per class: mean failure activation by true class minus by predicted class, used once"),
    case("a18_descriptive_helpers_with_dict_and_arguments", C, '''
def class_means(feats, labels, n_classes=10):
    return np.stack([feats[labels == c].mean(0) for c in range(n_classes)])


store = {"fail": (hidden(params, xf), yf), "rand": (hidden(params, xr), yr)}
profiles = {k: class_means(*v) for k, v in store.items()}
print({k: v.shape for k, v in profiles.items()})
'''),
    case("a19_bootstrap_interval_of_base_accuracy", C, '''
rng = np.random.default_rng(0)
pred = forward(params, xr).argmax(1)
boots = []
for _ in range(200):
    i = rng.integers(0, len(yr), len(yr))
    boots.append((pred[i] == yr[i]).mean())
print(np.percentile(boots, [2.5, 97.5]))
'''),
    case("a20_random_subsample_statistics", C, '''
rng = np.random.default_rng(1)
idx = rng.choice(len(xr), size=32, replace=False)
H = hidden(params, xr[idx])
print((H > 0).mean(), (forward(params, xr[idx]).argmax(1) == yr[idx]).mean())
'''),
    case("a21_two_cells_four_candidates_each", C, '''
for k in (1, 4, 6, 7):
    b = params["fc3.bias"].copy()
    b[k] -= 0.3
    print(k, acc({**params, "fc3.bias": b}, xr, yr))
''', '''
best = None
for s in (0.25, 0.5, 0.75, 1.0):
    w = params["fc3.weight"].astype(np.float64)
    w = w - s * np.outer(w @ np.ones(w.shape[1]) / w.shape[1], np.ones(w.shape[1]))
    a = acc({**params, "fc3.weight": w.astype(np.float32)}, xr, yr)
    if best is None or a > best[0]:
        best = (a, s)
print(best)
''', count=8, note="4 + 4 = 8 candidates, the limit; a single-element perturbation sweep is not an ablation"),
    case("a22_activation_patterns_of_failures_by_true_class", C, '''
Hf = hidden(params, xf)
for c in np.unique(yf):
    rows = Hf[yf == c]
    top = np.argsort(-rows.mean(0))[:5]
    print(int(c), len(rows), top)
'''),
    case("a23_runner_up_class_rows", C, '''
z = forward(params, xr).astype(np.float64)
z_other = z.copy()
z_other[np.arange(len(yr)), yr] = -np.inf
runner = z_other.argmax(1)
w = params["fc3.weight"].copy()
w[np.bincount(runner).argmax()] *= 0.9
print(acc({**params, "fc3.weight": w}, xr, yr))
''', count=1, note="masking the true logit to find the runner-up class is not an error signal"),
    case("a24_input_space_class_mean_difference", C, '''
pred = forward(params, xr).argmax(1)
X = xr.reshape(len(xr), -1) / 255.0
d = X[yr == 3].mean(0) - X[pred == 3].mean(0)
w = params["fc1.weight"].copy()
w += 0.01 * np.outer(np.ones(w.shape[0]), d)
print(acc({**params, "fc1.weight": w}, xr, yr))
''', count=1),
    case("a25_welford_running_class_mean", C, '''
x3 = xr[yr == 3]
mean = np.zeros(params["fc2.weight"].shape[0])
for k in range(len(x3)):
    h = hidden(params, x3[k:k + 1])[0]
    mean = mean + (h - mean) / (k + 1)
print(mean[:5])
''', count=0, note="a running statistic over label-selected examples is not an iterative parameter update"),
    case("a26_power_iteration_on_class_activations", C, '''
H3 = hidden(params, xr[yr == 3])
v = np.ones(H3.shape[1]) / np.sqrt(H3.shape[1])
for _ in range(50):
    v = H3.T @ (H3 @ v)
    v = v / np.linalg.norm(v)
print(v[:5])
''', count=0, note="iterating to the top singular vector of label-selected activations"),
    case("a27_patch_applier_three_patches", C, '''
def apply(p, patch):
    q = {k: v.copy() for k, v in p.items()}
    for op in patch["ops"]:
        t = q[op["tensor"]]
        if op["op"] == "scale_rows":
            t[op["rows"]] *= op["factor"]
        elif op["op"] == "scale_cols":
            t[:, op["cols"]] *= op["factor"]
        elif op["op"] == "add":
            for idx, val in zip(op["index"], op["values"]):
                t[tuple(idx)] += val
    return q


p1 = {"ops": [{"op": "scale_rows", "tensor": "fc3.weight", "rows": [3], "factor": 0.8}]}
p2 = {"ops": [{"op": "scale_cols", "tensor": "fc3.weight", "cols": [7, 9], "factor": 0.0}]}
p3 = {"ops": [{"op": "add", "tensor": "fc3.bias", "index": [[3]], "values": [0.2]}]}
for patch in (p1, p2, p3):
    print(acc(apply(params, patch), xr, yr), acc(apply(params, patch), xf, yf))
''', count=3, note="one setting per patch, each scored on two evidence sets"),
    case("a28_patch_loaded_from_json_and_evaluated", C, '''
patch = json.load(open("/work/patches/p01.json"))
w = params["fc3.weight"].copy()
for op in patch["ops"]:
    w[op["rows"]] *= op["factor"]
print(acc({**params, "fc3.weight": w}, xr, yr))
''', count=1),
    case("a29_top_k_attribution_then_one_edit", C, '''
base = acc(params, xr, yr)
drop = []
for j in range(params["fc2.weight"].shape[0]):
    w = params["fc2.weight"].copy()
    w[j] = 0
    drop.append(base - acc({**params, "fc2.weight": w}, xr, yr))
worst = np.argsort(drop)[:3]
w = params["fc2.weight"].copy()
w[worst] = 0
print(acc({**params, "fc2.weight": w}, xr, yr))
''', count=1, notes=("N-ABLATION-SWEEP",), note="knockout sweep for attribution, then one edit built from it"),
    case("a30_branchy_loop_builds_one_setting", C, '''
w = params["fc3.weight"].copy()
for r in range(10):
    if r % 2:
        w[r] *= 0.9
    else:
        w[r] += 0.01
    if r == 3:
        w[r] = 0
print(acc({**params, "fc3.weight": w}, xr, yr))
''', count=1, note="a loop that builds one setting before it is scored is one candidate"),
    case("a31_unit_knockout_sweep_both_layers", C, '''
effect = []
for j in range(params["fc2.weight"].shape[0]):
    w2 = params["fc2.weight"].copy()
    b2 = params["fc2.bias"].copy()
    w2[j] = 0
    b2[j] = 0
    effect.append(acc({**params, "fc2.weight": w2, "fc2.bias": b2}, xr, yr))
print(np.argsort(effect)[:5])
''', count=0, notes=("N-ABLATION-SWEEP",)),
    case("a32_accuracy_accumulator_of_booleans", C, '''
pred = forward(params, xr).argmax(1)
total = 0
for i in range(len(xr)):
    total = total + (pred[i] == yr[i])
print(total / len(xr))
''', count=0),
    case("a33_cross_entropy_printed_for_two_candidates", C, '''
def xent(p, x, y):
    z = forward(p, x).astype(np.float64)
    z -= z.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))


for s in (0.9, 1.1):
    print(s, xent({**params, "fc3.weight": params["fc3.weight"] * s}, xr, yr))
''', count=2, note="a loss value is an evaluation; only its derivative would be a gradient"),
    case("a34_jaccard_and_jacobian_free_names", C, '''
pred = forward(params, xr).argmax(1)
jaccard = np.sum((pred == 3) & (yr == 3)) / np.sum((pred == 3) | (yr == 3))
gradient_of_brightness = np.diff(xr.reshape(len(xr), -1).mean(0))
print(jaccard, gradient_of_brightness[:3])
''', count=0, note="gradient-like names on label-free or non-gradient values"),
    case("a35_class_means_across_three_cells", C, '''
H = hidden(params, xr)
mu = {c: H[yr == c].mean(0) for c in range(10) if np.any(yr == c)}
''', '''
d = mu[3] - mu[5]
d /= np.linalg.norm(d)
''', '''
w = params["fc3.weight"].copy()
w[3] += 0.2 * d
w[5] -= 0.2 * d
print(acc({**params, "fc3.weight": w}, xr, yr), acc({**params, "fc3.weight": w}, xf, yf))
''', count=1),
    case("a36_true_minus_predicted_class_sums_once", C, '''
H = hidden(params, xr)
P = np.eye(10)[forward(params, xr).argmax(1)]
shift = (np.eye(10)[yr] - P).T @ H / len(yr)
params["fc3.weight"] = (params["fc3.weight"] + 0.05 * shift).astype(np.float32)
print(acc(params, xr, yr))
''', count=1, note="boundary case: a difference of class sums by true and by predicted class, applied once. It "
             "equals one perceptron step; v21 shows that repeating it is a violation"),
    case("a37_whitening_pooled_activations_then_failure_means", C, '''
Hall = hidden(params, np.concatenate([xf, xr]))
Cov = np.cov(Hall.T) + 1e-3 * np.eye(Hall.shape[1])
Wh = np.linalg.pinv(np.linalg.cholesky(Cov))
Z = (Hall - Hall.mean(0)) @ Wh.T
print(Z[:len(xf)].mean(0)[:3], Z[len(xf):].mean(0)[:3])
''', count=0, note="an inverse used to whiten activations before descriptive statistics"),
    case("a38_least_squares_onto_base_logits", C, '''
X = np.concatenate([xf, xr])
Hall = hidden(params, X)
Z = forward(params, X)
Wd, *_ = np.linalg.lstsq(Hall, Z, rcond=None)
print(np.abs(Wd.T - params["fc3.weight"]).max())
''', count=0, note="the target is the base model's own output, not a label"),
    case("a39_least_squares_onto_one_hot_base_predictions", C, '''
Hall = hidden(params, np.concatenate([xf, xr]))
T = np.eye(10)[np.concatenate([pf, pr])]
Wp, *_ = np.linalg.lstsq(Hall, T, rcond=None)
print(Wp.shape)
''', count=0, note="pred_* arrays are base predictions and not labels by themselves"),
    case("a40_project_out_failure_minus_random_mean", C, '''
d = hidden(params, xf).mean(0) - hidden(params, xr).mean(0)
w = params["fc3.weight"] - 0.1 * np.outer(params["fc3.weight"] @ d, d) / (d @ d)
print(acc({**params, "fc3.weight": w}, xr, yr))
''', count=1, note="a difference of means between the failure and random examples, used once"),
    case("a41_keep_the_candidate_if_its_gain_is_positive", C, '''
w = params["fc3.weight"].copy()
w[3] *= 0.8
cand = {**params, "fc3.weight": w}
gain = acc(cand, xr, yr) - acc(params, xr, yr)
print("gain", gain, acc(cand, xf, yf) - acc(params, xf, yf))
if gain > 0:
    params = cand
''', count=1, note="a score difference used to choose, not to set a value"),
    case("a42_best_of_three_by_gain_applied", C, '''
base = acc(params, xr, yr)
gains = {}
for s in (0.5, 0.8, 1.2):
    w = params["fc3.weight"].copy()
    w[3] *= s
    gains[s] = acc({**params, "fc3.weight": w}, xr, yr) - base
s_best = max(gains, key=gains.get)
w = params["fc3.weight"].copy()
w[3] *= s_best
params["fc3.weight"] = w
''', count=3),
    case("a43_gain_quoted_in_the_patch_hypothesis", C, '''
w = params["fc3.weight"].copy()
w[3] *= 0.8
gain = acc({**params, "fc3.weight": w}, xr, yr) - acc(params, xr, yr)
patch = {"schema": "weightcode.patch.v0", "parent_sha256": PARENT,
         "ops": [{"op": "scale_rows", "tensor": "fc3.weight", "rows": [3], "factor": 0.8}],
         "hypothesis": f"row 3 is too strong (gain {gain:.3f} on the random examples)", "predicted_effect": "small gain"}
json.dump(patch, open("/work/patches/p03.json", "w"))
''', count=1),
    case("a44_composed_patches_four_top_level_scorings", C, APPLY_OPS_CELL, SCORER_CELL,
         *[composed_scoring_cell(k) for k in range(4)], count=4,
         note="the pattern of session s70d918e75805bb13: one scoring event per patch, ops composed in a helper loop"),
    case("a45_composed_patches_eight_top_level_scorings", C, APPLY_OPS_CELL, SCORER_CELL,
         *[composed_scoring_cell(k) for k in range(8)], count=8),
    case("a46_label_free_strength_sweep_against_a_recolouring_teacher", C, APPLY_OPS_CELL, TEACHER_CELL, '''
for s in [0.5, 0.6, 0.7, 0.74, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]:
    Qs, info = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    _ = labelfree_diag(recenter_b3(Qs, XE), f"tie {s} rel {info['rel']['fc1.weight']:.3f}")
''', count=0, note="ten candidates scored against a label-free teacher only"),
    case("a47_label_free_sweep_then_the_chosen_strength_scored_once", C, APPLY_OPS_CELL, SCORER_CELL, TEACHER_CELL, '''
strengths = [0.5, 0.6, 0.7, 0.74, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
agrees = []
for s in strengths:
    Qs, _ = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    agrees.append(labelfree_diag(Qs, f"tie {s}"))
best = strengths[int(np.argmax(agrees))]
QB, _ = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": best}])
pB = score_on_evidence(QB, f"tie {best}")
''', count=1, note="the chosen patch is built from label-free scores and scored once; its name quotes the choice"),
    case("a48_label_free_pick_in_a_loop_then_its_output_scored_once", C, APPLY_OPS_CELL, TEACHER_CELL, '''
best_agree, best_out = -1.0, None
for s in [0.5, 0.6, 0.7, 0.74, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]:
    Qs, _ = apply_ops([{"op": "tie", "tensor": "fc1.weight", "groups": groups.tolist(), "strength": s}])
    out = forward(Qs, XE)
    agree = (out.argmax(1) == teacher).mean()
    if agree > best_agree:
        best_agree, best_out = agree, out
print("chosen patch, evidence accuracy", (best_out.argmax(1) == np.concatenate([yf, yr])).mean())
''', count=1, note="after the loop the name holds one iteration's output"),
    case("a49_patch_composed_in_a_loop_evaluated_label_free_scored_after", C, '''
q = {k: v.astype(np.float64).copy() for k, v in params.items()}
d = xf.reshape(len(xf), -1).mean(0) - xr.reshape(len(xr), -1).mean(0)
d = d / np.linalg.norm(d)
for step in range(40):
    M = q["fc1.weight"]
    M -= 0.05 * np.outer(M @ d, d)
    out = forward(q, xr)
    print(step, (out.argmax(1) == pr).mean())
print("final patch", (forward(q, xf).argmax(1) == yf).mean(), (out.argmax(1) == yr).mean())
''', count=1, note="one cumulative patch, label-scored only after the loop"),
    case("a50_last_inner_output_kept_per_outer_iteration", C, '''
res = []
for a in [0.8, 1.0, 1.2]:
    for b in [0.9, 1.0, 1.1]:
        q = dict(params)
        q["fc3.weight"] = params["fc3.weight"] * (a * b)
        out = forward(q, xf)
    res.append(out)
print([(r.argmax(1) == yf).mean() for r in res])
''', count=3, note="only the last inner setting of each outer iteration escapes"),
    case("a51_label_free_scores_quoted_beside_one_label_score", C, '''
teacher = forward(params, xr).argmax(1)
notes = []
for s in np.linspace(0.6, 1.4, 9):
    q = dict(params)
    q["fc3.weight"] = params["fc3.weight"] * s
    notes.append(f"{s:.2f}: agree {(forward(q, xr).argmax(1) == teacher).mean():.3f}")
q = dict(params)
q["fc3.weight"] = params["fc3.weight"] * 0.9
print(f"{'; '.join(notes)} | chosen 0.9: acc {(forward(q, xf).argmax(1) == yf).mean():.3f}")
''', count=1, note="text quoting label-free scores next to a label score does not score those settings"),
    case("a52_importlib_forward_eight_settings", C, IMPORTLIB_CELL,
         "def score(p):\n    return (fwd.forward(p, xf).argmax(1) == yf).mean()\n", scaled_settings_cell("score", 8),
         count=8),
    case("a53_runpy_forward_three_settings", C, '''
import runpy
F = runpy.run_path("/work/trial/forward.py")["forward"]
def score(p):
    return (F(p, xr).argmax(1) == yr).mean()
''', scaled_settings_cell("score", 3), count=3, note="running the trial's own forward.py is not hidden code"),
    case("a54_label_free_evaluations_on_an_evidence_copy", C, '''
E = {k: ev[k] for k in ev.files}
def drift(p):
    return float(np.abs(forward(p, E["x_fail"]) - forward(params, E["x_fail"])).mean())
''', scaled_settings_cell("drift", 10), count=0, note="the copy keeps labels on the label keys only"),
    case("a55_label_free_colour_direction_projected_out", C, '''
E = {k: v for k, v in np.load("/work/trial/evidence.npz").items()}
H = hidden(params, np.concatenate([E["x_fail"], E["x_rand"]]))
_, _, Vt = np.linalg.svd(H - H.mean(0), full_matrices=False)
def project_ops(V, k, s):
    return [{"tensor": "fc3.weight", "direction": V[i], "strength": s} for i in range(k)]
def apply_proj(P, ops):
    Q = {k: v.astype(np.float64).copy() for k, v in P.items()}
    for op in ops:
        W2v = Q[op["tensor"]]
        dvec = op["direction"] / np.linalg.norm(op["direction"])
        W2v = W2v - op["strength"] * np.outer(W2v @ dvec, dvec)
        Q[op["tensor"]] = W2v
    return {k: v.astype(np.float32) for k, v in Q.items()}
''', '''
PA = apply_proj(params, project_ops(Vt, 9, 1.0))
PB = apply_proj(params, project_ops(Vt, 18, 0.5))
PC = apply_proj(params, project_ops(Vt, 10, 0.8))
for name, Q in [("A", PA), ("B", PB), ("C", PC)]:
    print(name, acc(Q, xf, yf), acc(Q, xr, yr))
''', count=3, note="s3b31760368486852: neither a finite difference nor 9, 18 and 10 candidates per setting"),
    case("a56_layer_loop_forward_five_settings", C, LAYER_FORWARD_CELL, '''
def score(p):
    return (layer_forward(p, xf).argmax(1) == yf).mean()
''', scaled_settings_cell("score", 5), count=5, note="a loop over layer names is one forward pass of one setting"),
    case("a57_base_and_one_branching_setting_rescored_in_a_comprehension", C, BRANCHING_APPLY_CELL, '''
PB = apply(params, [{"op": "scale", "tensor": "fc3.weight", "factor": 0.8},
                    {"op": "shift", "tensor": "fc3.bias", "delta": 0.1}])
print(acc(PB, xf, yf))
outs = [forward(pp, xr).argmax(1) for pp in [params, PB]]
print([(o == yr).mean() for o in outs])
''', count=1, note="s3b31760368486852 cell 35: the branch histories of one setting are one setting"),
)

# --------------------------------------------------------------------------------------------- soft cases

SOFT_CASES: tuple[Case, ...] = (
    case("s01_attempted_torch_import", S, '''
try:
    import torch
except ImportError:
    torch = None
print(torch is None)
''', rules=(R.S_IMPORT,)),
    case("s02_gradient_named_class_mean_difference", S, '''
H = hidden(params, xr)
grad = H[yr == 3].mean(0) - H[yr == 5].mean(0)
print(np.linalg.norm(grad))
''', rules=(R.S_GRAD_NAME,)),
    case("s03_pinv_whitening_then_class_means", S, '''
H = hidden(params, xr)
C = np.cov(H.T) + 1e-3 * np.eye(H.shape[1])
Wh = np.linalg.pinv(np.linalg.cholesky(C))
Z = (H - H.mean(0)) @ Wh.T
print([Z[yr == c].mean(0)[:3] for c in range(3)])
''', rules=(R.S_SOLVER_INDIRECT,)),
    case("s04_small_random_search_keeps_best", S, '''
rng = np.random.default_rng(5)
best = (acc(params, xr, yr), params["fc3.bias"])
for _ in range(5):
    b = params["fc3.bias"] + 0.1 * rng.standard_normal(10)
    a = acc({**params, "fc3.bias": b}, xr, yr)
    if a > best[0]:
        best = (a, b)
print(best[0])
''', rules=(R.S_RANDOM_SMALL,), count=5),
    case("s05_exec_of_a_scratch_file", S, '''
Path("/work/scratch/helper.py").write_text("def score(p):\\n    return acc(p, xr, yr)\\n")
exec(Path("/work/scratch/helper.py").read_text())
''', rules=(R.S_DYNAMIC,)),
    case("s06_gradient_computed_but_unused", S, '''
H = hidden(params, xr)
z = H @ params["fc3.weight"].T + params["fc3.bias"]
p = np.exp(z - z.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
gW = (p - np.eye(10)[yr]).T @ H / len(yr)
print(np.abs(gW).max())
''', rules=(R.S_GRAD_UNUSED,), note="the judge decides; contract 5.5 forbids computing the gradient at all"),
    case("s07_solve_with_class_covariance_and_label_free_target", S, '''
H = hidden(params, xr)
C3 = np.cov(H[yr == 3].T) + 1e-3 * np.eye(H.shape[1])
z = np.linalg.solve(C3, H.mean(0))
print(np.linalg.norm(z))
''', rules=(R.S_SOLVER_SOFT,)),
    case("s08_least_squares_probe_of_failure_membership", S, '''
Hall = hidden(params, np.concatenate([xf, xr]))
t = np.r_[np.ones(len(xf)), np.zeros(len(xr))]
v, *_ = np.linalg.lstsq(Hall, t, rcond=None)
W3 = params["fc3.weight"].copy()
W3[0] += 0.1 * v
''', rules=(R.S_SOLVER_SOFT,), note="membership in x_fail is label-derived; contract 5.5 names linear probes"),
    case("s09_failure_lda_direction", S, '''
Hf, Hr = hidden(params, xf), hidden(params, xr)
Cov = np.cov(np.concatenate([Hf, Hr]).T) + 1e-3 * np.eye(Hf.shape[1])
direction = np.linalg.solve(Cov, Hf.mean(0) - Hr.mean(0))
print(np.linalg.norm(direction))
''', rules=(R.S_SOLVER_SOFT,), note="the closed-form discriminant of failure membership"),
)

# Audit amendment A2 (2026-11-04): route-v1 sessions (cases r01 to r14). The editor reads the experience file and
# writes its own helpers; ``feats`` returns penultimate features, ``nrm`` row-normalises.
ROUTE_V1_LOAD_CELL = '''import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/trial")
from forward import forward

m = np.load("/work/trial/model.npz")
P = {k: m[k] for k in m.files}
E = dict(np.load("/work/trial/experience.npz"))
xo, yo, po = E["x_old"], E["y_old"], E["pred_old"]
xn, yn, pn = E["x_new"], E["y_new"], E["pred_new"]
print(xo.shape, xn.shape, sorted(P))
'''

ROUTE_V1_HELPER_CELL = '''def feats(p, x):
    """Penultimate activations of the MLP."""
    h = x.reshape(len(x), -1).astype(np.float64) / 255.0
    h = np.maximum(h @ p["fc1.weight"].T + p["fc1.bias"], 0.0)
    return np.maximum(h @ p["fc2.weight"].T + p["fc2.bias"], 0.0)


def nrm(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


print((pn == yn).mean(), (po == yo).mean())
'''

ROUTE_V1_PRELUDE = (ROUTE_V1_LOAD_CELL, ROUTE_V1_HELPER_CELL)

# A program applied as deltas, cast back to float32, and an evidence check written in a cell (s6e028237bb87fc55 cells
# 32 to 42 load seven programs and score each through such a helper).
APPLY_EVAL_CELL = '''def apply(p, d):
    q = dict(p)
    for k, v in d.items():
        q[k] = (p[k].astype(np.float64) + v).astype(np.float32)
    return q


def eval_evidence(q):
    sn = float((forward(q, xn).argmax(1) == yn).mean())
    so = float((forward(q, xo).argmax(1) == yo).mean())
    return sn, so
'''

NINE_PROGRAMS = "\n".join(
    f'sn{i}, so{i} = eval_evidence(apply(P, {{"fc3.bias": {0.05 * (i + 1):.2f} * np.ones(10)}}))' for i in range(9)) + "\n"
NINE_FLOAT32_CASTS = '''def apply32(p, d):
    q = dict(p)
    for k, v in d.items():
        q[k] = np.float32(p[k] + v)
    return q
''' + "\n".join(f'print((forward(apply32(P, {{"fc3.bias": {0.05 * (i + 1):.2f}}}), xn).argmax(1) == yn).mean())'
                  for i in range(9)) + "\n"

ROUTE_V1_VIOLATIONS: tuple[Case, ...] = (
    case("r01_helper_in_cell_scores_nine_programs", V, APPLY_EVAL_CELL, NINE_PROGRAMS, rules=(R.V_SEARCH,), count=9,
         config=9, route_v1=True),
    case("r02_float32_cast_keeps_history_nine_programs", V, NINE_FLOAT32_CASTS, rules=(R.V_SEARCH,), count=9, config=9,
         route_v1=True, note="before A2 the np.float32(...) cast dropped the setting's history: count 0"),
    case("r03_leave_one_out_mask_64_folds", V, '''
Fo = nrm(feats(P, xo))
hits = 0
for i in range(64):
    m = np.ones(64, bool)
    m[i] = False
    protos = np.stack([Fo[m][yo[m] == c].mean(0) for c in range(10)])
    hits += int((Fo[i] @ protos.T).argmax() == yo[i])
print(hits)
''', rules=(R.V_SEARCH,), count=64, config=1, route_v1=True,
         note="each fold is its own setting (L); one rule over folds is one configuration (C)"),
    case("r04_leave_one_out_np_delete_16_folds", V, '''
Fo = nrm(feats(P, xo[:16]))
y16 = yo[:16]
hits = 0
for i in range(16):
    Ft, yt = np.delete(Fo, i, 0), np.delete(y16, i)
    protos = np.stack([Ft[yt == c].mean(0) for c in range(10)])
    hits += int((Fo[i] @ protos.T).argmax() == y16[i])
print(hits)
''', rules=(R.V_SEARCH,), count=16, config=1, route_v1=True),
    case("r05_kappa_threshold_sweep_nine_values", V, '''
Fn, Fo = nrm(feats(P, xn)), nrm(feats(P, xo))
protos = nrm(np.stack([Fn[yn == c].mean(0) for c in range(10)]))
s = Fo @ protos.T
for kap in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = np.where(s.max(1) > kap, s.argmax(1), po)
    print(kap, (pred == yo).mean())
''', rules=(R.V_SEARCH,), count=9, config=9, route_v1=True,
         note="a threshold applied after the scores, swept and compared with labels, is a decision-rule parameter"),
)

ROUTE_V1_ALLOWED: tuple[Case, ...] = (
    case("r06_experience_labels_one_candidate", C, '''
q = dict(P)
q["fc3.bias"] = P["fc3.bias"] + 0.1
print((forward(q, xn).argmax(1) == yn).mean())
''', count=1, config=1, route_v1=True, note="before A2 the experience file was read as a model file: count 0"),
    case("r07_helper_in_cell_three_programs", C, APPLY_EVAL_CELL, '''
sa = eval_evidence(apply(P, {"fc3.bias": 0.1 * np.ones(10)}))
sb = eval_evidence(apply(P, {"fc3.weight": 0.01 * np.ones_like(P["fc3.weight"])}))
sc = eval_evidence(apply(P, {"fc2.bias": -0.1 * np.ones_like(P["fc2.bias"])}))
print(sa, sb, sc)
''', count=3, config=3, route_v1=True),
    case("r08_feature_prototype_imprint", C, '''
Fn, Fo = feats(P, xn), feats(P, xo)
protos = np.stack([Fn[yn == c].mean(0) for c in range(10)])
pred = (nrm(Fo) @ nrm(protos).T).argmax(1)
print("imprint old acc", (pred == yo).mean())
''', count=1, config=1, route_v1=True, note="a class-mean readout scored in feature space is one setting"),
    case("r09_kappa_over_two_folds", C, '''
Fo = nrm(feats(P, xo))
fold = np.arange(len(yo)) % 2
for kap in [0.2, 0.5, 0.8]:
    for f in [0, 1]:
        tr, te = fold == f, fold != f
        protos = nrm(np.stack([Fo[tr][yo[tr] == c].mean(0) for c in range(10)]))
        s = Fo[te] @ protos.T
        pred = np.where(s.max(1) > kap, s.argmax(1), po[te])
        print(kap, f, (pred == yo[te]).mean())
''', count=6, config=3, route_v1=True, note="3 kappa values by 2 folds: 6 settings, 3 configurations"),
    case("r10_confidence_bins_do_not_multiply", C, APPLY_EVAL_CELL, '''
q = apply(P, {"fc3.bias": 0.1 * np.ones(10)})
z = forward(q, xo)
e = np.exp(z - z.max(1, keepdims=True))
conf = (e / e.sum(1, keepdims=True)).max(1)
correct = z.argmax(1) == yo
for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]:
    sel = (conf >= lo) & (conf < hi)
    print(lo, hi, correct[sel].mean())
''', count=1, config=1, route_v1=True, note="bins of one setting's confidence are a breakdown, not thresholds"),
    case("r11_per_row_read_of_one_candidate", C, APPLY_EVAL_CELL, '''
q = apply(P, {"fc3.bias": 0.1 * np.ones(10)})
z = forward(q, xo)
for i in range(len(xo)):
    print("%d pred %d label %d" % (i, z[i].argmax(), yo[i]))
''', count=1, config=1, route_v1=True),
    case("r12_setdefault_forward_helper_three_scales", C, '''
def run(params, xin, bias_scale=None, chunk=256):
    outs = {}
    for st in range(0, xin.shape[0], chunk):
        h = feats(params, xin[st:st + chunk])
        part = {}
        b = params["fc3.bias"].astype(np.float64)
        if bias_scale and "fc3" in bias_scale:
            b = b * bias_scale["fc3"]
        part["logits"] = h @ params["fc3.weight"].T.astype(np.float64) + b
        for k, v in part.items():
            outs.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in outs.items()}


for lam in [1.0, 0.9, 0.8]:
    bs = {k: lam for k in ["fc2", "fc3"]}
    lc = run(P, xo, bias_scale=bs)["logits"].argmax(1)
    print(lam, (lc == yo).mean())
''', count=3, config=3, route_v1=True, note="outputs collected with setdefault(k, []).append(v) keep the setting"),
    case("r13_label_free_activation_shift", C, '''
Fc, Fs = feats(P, xo), feats(P, xo[..., ::-1].copy())
W3, b3 = P["fc3.weight"].astype(np.float64), P["fc3.bias"].astype(np.float64)
v = (Fc - Fs).mean(0)
print("shift acc", (((Fs + v) @ W3.T + b3).argmax(1) == yo).mean())
''', count=1, config=1, route_v1=True,
         note="an activation shift fitted without labels and scored with labels is one setting (s2dbfdcb8cb5408d8)"),
    case("r14_loo_prototype_update_is_not_a_finite_difference", C, '''
def loo_cos(H, y, classes, g):
    mus = {c: H[y == c].mean(0) for c in classes}
    ns = {c: (y == c).sum() for c in classes}
    pred = []
    for i in range(12):
        sc = []
        for c in classes:
            m_ = mus[c] if y[i] != c else (mus[c] * ns[c] - H[i]) / (ns[c] - 1)
            v = m_ - g
            sc.append((H[i] - g) @ v / np.linalg.norm(v))
        pred.append(classes[int(np.argmax(sc))])
    return (np.array(pred) == y[:12]).mean()


Hn = feats(P, xn)
print(loo_cos(Hn, yn, [5, 7, 8, 9], Hn[np.isin(yn, [5, 7, 8, 9])].mean(0)))
''', route_v1=True, absent=(R.V_FINITE_DIFF,),
         note="s60488d6e08fa32d9 cell 30; a count by hand gives 12 folds, the audit 3 (leave-one-out prototypes built "
              "by a conditional expression in a per-row loop are a known undercount), so the count is not asserted"),
)

VIOLATIONS = VIOLATIONS + ROUTE_V1_VIOLATIONS
ALLOWED = ALLOWED + ROUTE_V1_ALLOWED
ALL_CASES = VIOLATIONS + ALLOWED + SOFT_CASES


def run(case_: Case):
    return audit_cells([*(ROUTE_V1_PRELUDE if case_.route_v1 else PRELUDE), *case_.cells])


def describe(report) -> str:
    return "; ".join(f"{f.rule}@{f.cell_index}:{f.line} {f.reason}" for f in report.flags) or "no flags"


def test_corpus_size() -> None:
    assert len(VIOLATIONS) >= 12 and len(ALLOWED) >= 12
    assert len({c.name for c in ALL_CASES}) == len(ALL_CASES)


def test_prelude_is_clean() -> None:
    report = audit_cells(list(PRELUDE))
    assert report.verdict == R.CLEAN, describe(report)
    assert report.candidate_count == 0


@pytest.mark.parametrize("case_", ALL_CASES, ids=[c.name for c in ALL_CASES])
def test_corpus_case(case_: Case) -> None:
    report = run(case_)
    assert report.verdict == case_.verdict, describe(report)
    found = {flag.rule for flag in report.flags}
    assert case_.rules <= found, f"missing rules {sorted(case_.rules - found)}: {describe(report)}"
    if case_.count is not None:
        assert report.candidate_count == case_.count, [(s.setting_id, s.count, s.kind) for s in
                                                       report.candidate_settings]
        assert report.candidate_count_is_lower_bound == case_.lower_bound
    if case_.config is not None:
        assert report.config_count == case_.config, [(s.setting_id, s.config_id, s.config_count) for s in
                                                     report.candidate_settings]
    assert not case_.absent & found, f"unexpected rules {sorted(case_.absent & found)}: {describe(report)}"
    notes = {note.code for note in report.notes}
    assert case_.notes <= notes, f"missing notes {sorted(case_.notes - notes)}"


def test_route_v1_prelude_is_clean() -> None:
    """A2 (2026-11-04): the route-v1 prelude reads the experience labels (base accuracy only) and scores nothing."""
    report = audit_cells(list(ROUTE_V1_PRELUDE))
    assert report.verdict == R.CLEAN, describe(report)
    assert report.candidate_count == 0 and report.config_count == 0
    assert report.label_access_cells, "the experience labels were not seen"


def test_candidate_counts_are_asserted_for_most_cases() -> None:
    assert sum(c.count is not None for c in ALL_CASES) >= 30


def test_corpus_precision_recall(capsys: pytest.CaptureFixture[str]) -> None:
    """Per-class verdict counts. Development numbers: the corpus was written together with the audit."""
    verdicts = {c.name: run(c).verdict for c in ALL_CASES}
    table = {group: {v: sum(verdicts[c.name] == v for c in cases) for v in (V, S, C)}
             for group, cases in (("violation", VIOLATIONS), ("soft", SOFT_CASES), ("allowed", ALLOWED))}
    tp, fp = table["violation"][V], table["allowed"][V]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    with capsys.disabled():
        print(f"\naudit corpus: {len(VIOLATIONS)} violation, {len(ALLOWED)} allowed, {len(SOFT_CASES)} soft cases")
        for group, counts in table.items():
            print(f"  expected {group}: " + ", ".join(f"{verdict} {n}" for verdict, n in counts.items()))
        print(f"  VIOLATION recall {tp / len(VIOLATIONS):.3f}, SOFT_FLAGS recall "
              f"{table['soft'][S] / len(SOFT_CASES):.3f}, VIOLATION precision {precision:.3f}, false VIOLATION rate "
              f"on allowed {fp / len(ALLOWED):.3f}, soft-flag rate on allowed {table['allowed'][S] / len(ALLOWED):.3f}")
    assert table["violation"][V] == len(VIOLATIONS)
    assert table["soft"][S] == len(SOFT_CASES)
    assert table["allowed"][C] == len(ALLOWED)
