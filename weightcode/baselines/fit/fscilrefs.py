"""Reference streams of PREREG E9 (rule fscil-v1): label fits of the ResNet-20 cosine student, session after session.

A stream starts at session 2 from the fscil_base student of seed 100 and, at every session t = 2 .. 9, runs its method
on its own current student (its kept model of session t - 1) and keeps one model. The streams run in lockstep
(``run_streams``), because the kept rule of session t needs the flip rate d_t of every stream's checks at t:

1. every stream fits its candidates on the rows of session t and checks them (8 checks at most);
2. d_t = the median over every checked candidate of every stream of (fixed_old + broken_old) / n_c, and
   floor_t = ``weightcode.evalkit.fscil.joint_retain_floor(n_c, n_s, R, d_t)`` with (n_c, n_s) = ``old_part_sizes(t)``;
3. every stream keeps the argmax of S = S_new + S_old over its checked candidates with S_old >= floor_t and its
   current student (S = 0); ties go to fewer changed entries, then to the earlier check (kept rule joint-retain-v1);
4. the kept models are read out and become the current students of session t + 1.

Rows. Every stream gets the rows a teacher session t gets (``weightcode.evalkit.fscil``'s instance plan): the 25 shots of
session t ("new"), the 64 replay rows of the base classes ("old", one draw for every session) and the shots of
sessions 2 .. t - 1 ("prev"). The fits use all of them, unweighted, under the session-t mask (argmax and the
cross-entropy over the n_seen(t) outputs of the classes seen so far). The check is ``JointPatchCheckOracle`` on
``fscil_check_parts`` (new: the V_check rows of session t's classes; old: old_check plus the V_check rows of the
classes of sessions 2 .. t - 1) with forward.py under the session-t mask, against the stream's current student, in
batches of ``EVALUATION_BATCH``: the oracle a teacher session gets. Run seed 0.

The student is rebuilt as a functional torch forward over the parameter tensors (``resnet_features``), in eval mode:
BatchNorm uses its running statistics, which are fixed inputs, never trained. Before each fit the float64 and float32
rebuilds are checked against forward.py on the fit rows (``weightcode.models.parity.check_logit_parity``). f denotes the
pooled 64-dimensional encoder features, f^ = f / ||f||, and the head is logits = 16 cos(f, W_c), W = fc.weight.

Methods (``STREAM_METHODS``; the names are PREREG E9's):

* ``imprint``: Decoupled-Cosine prototypes, 2 candidates. "mean": the row of each class of session t is the mean of f
  over its 5 shots (the prototype of CEC's ``update_fc_avg``); "centered": normalize(mean over the shots of
  normalize(f - m)), m the mean f over the 64 replay rows (SimpleShot's CL2N, centred by the base-class mean the
  experience holds). Only the 5 rows of session t's classes change. The cosine head is invariant to a row's scale and
  has no bias, so E2's scales and offset have no analog here.
* ``imprint_retain`` (supplement only, never in the gate): per imprint direction u_c (unit), the row
  normalize(u_c - beta mu^), mu^ = normalize(m), with beta the smallest value of ``RETAIN_BETAS`` at which no
  protected row breaks; the protected rows are the replay and prev rows the current student classifies correctly
  under the mask, predicted from the cached features. When every beta breaks one, the beta that breaks the fewest
  (the smallest among ties), recorded. It is the analog of E2's retain offset: moving the new rows away from the mean
  base feature lowers their cosine with base features, which share one orthant (ReLU features), so fewer protected
  rows move to a new class. 2 candidates.
* ``ridge_margin``: the seen rows of the head refitted toward the current ones on normalized features: h1 = 16 f^,
  w0 = the normalized current rows 0 .. n_seen - 1, z0 = h1 w0^T (the current masked logits), T = E2's margin targets
  of z0, and each row w_j = argmin over unit vectors of ||h1 w - T_j||^2 + lam ||w - w0_j||^2
  (``sphere_ridge_rows``), lam = s x mean ||h1||^2 = 256 s for s in ``RIDGE_SCALES`` (1e-3 .. 1e4). 8 candidates.
  The cosine head only sees row directions, so the ridge objective is solved over unit rows: the logits after
  forward.py's renormalization are the fitted ones. (Amended after session 2 of fscil 9691ad7, before any later
  session or teacher session: that run used the unconstrained ``ridge_solve`` and renormalized afterwards, which
  undid the fit, since the margin targets of the new classes need cosines above 1 and the rows of unseen classes
  have norms near 0.001; S_new was 0 at every s.)
* ``grad_last``: full-batch Adam on fc.weight with the encoder fixed (the features cached), cross-entropy over the seen
  outputs; the rows of unseen classes get no gradient and stay. E7's grid, lr {1e-3, 1e-2} x steps {10, 30, 100, 300}.
* ``grad_full``: the same on every convolution weight, BatchNorm weight and bias and fc.weight (``trainable_names``),
  edit limits ignored (recorded as out of limits); the unconstrained reference.
* ``grad_sparse19_sal`` and ``grad_sparse19_traj``: E5's masks as in E7, a fixed top 19% of each trainable tensor's
  entries by |d loss / d w| at the current student, or by |theta_full - current|; gradients masked before each step.
* ``interp``: current + alpha (theta_full - current) on the trainable tensors, alpha in {1/8, ..., 8/8}, edit limits
  ignored. theta_full is unconstrained Adam at lr 1e-3 for 300 steps from the stream's current student.

Instrument, never gating (``INSTRUMENT_METHODS``): ``imprint_plain``, imprint's "mean" candidate written every session
with no check and always kept: Decoupled-Cosine as the literature defines it.

Limits: ``fscil_program_limits`` of the session's task (rel-Frob 0.5 on every tensor of rank 2 or more except
fc.weight, at most 20% of all entries changed, BatchNorm running statistics counted). The methods with enforced limits
scale over-limit tensors down to the norm limit (``scale_to_norm_limit``, the head exempt) and record it; a 19% mask
of the trainable entries stays under the cap by construction, as do the head-only methods (2.3% of the entries).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.common.identity import Params, params_sha256
from weightcode.data.store import dataset_json_path
from weightcode.evalkit import fscil as FS
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.evalkit.patch_check import ForwardFn, JointPatchCheckOracle, validate_child
from weightcode.evalkit.targets import file_sha256, load_checked_split, load_dataset_forward, read_dataset_meta, split_path
from weightcode.models import resnet20 as R
from weightcode.models.fscil_export import normalization
from weightcode.models.parity import check_logit_parity
from weightcode.patch import Limits

from .edits import edit_stats_under, scale_to_norm_limit
from .grad import candidate_grid as grad_grid
from .graphrefs import INTERP_ALPHAS, MASK_FRACTION, TRAJ_LR, TRAJ_STEPS, adam_checkpoints, top_fraction_masks
from .imprint import class_directions
from .ridge import RIDGE_SCALES, margin_targets

STREAMS_SCHEMA = "weightcode.fscil_reference_streams.v1"
STREAM_METHODS: tuple[str, ...] = ("imprint", "imprint_retain", "ridge_margin", "grad_last", "grad_full",
                                   "grad_sparse19_sal", "grad_sparse19_traj", "interp")
INSTRUMENT_METHODS: tuple[str, ...] = ("imprint_plain",)
STREAMS: tuple[str, ...] = STREAM_METHODS + INSTRUMENT_METHODS
GATE_EXCLUDED: tuple[str, ...] = ("imprint_retain",)
LIMITS_IGNORED: tuple[str, ...] = ("grad_full", "interp")
CHECKS_PER_SESSION = 8
RUN_SEED = 0
IMPRINT_DIRECTIONS: tuple[str, ...] = ("mean", "centered")
RETAIN_BETAS: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(41))
"""The beta ladder of imprint_retain: 0, 0.05, ..., 2.0 (beta = 0 is imprint's row)."""
SPLITS: tuple[str, ...] = ("V_evidence", "old_replay", "V_check", "old_check", "V_select", "old_select")
"""The store splits a stream run reads; never the sealed test."""


class FscilRefError(RuntimeError):
    """A stream input, rebuild or record that is not what PREREG E9 defines."""


# ---------------------------------------------------------------------------------------------------------------
# Data of a run and rows of a session


@dataclass(frozen=True)
class StreamData:
    """Everything a stream run reads: the dataset's splits, forward.py, the base and all-class students."""

    meta: Mapping[str, Any]
    forward: ForwardFn
    forward_sha256: str
    splits: Mapping[str, Mapping[str, np.ndarray]]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    spec: Mapping[str, Any]
    class_names: tuple[str, ...]
    base_row: Mapping[str, Any]
    base_params: Params
    allclass_row: Mapping[str, Any] | None
    allclass_params: Params | None

    def provenance(self) -> dict[str, Any]:
        """Hashes of every input, for the run record."""
        return {
            "dataset": FS.DATASET,
            "dataset_json_sha256": file_sha256(dataset_json_path(FS.DATASET)),
            "forward_sha256": self.forward_sha256,
            "splits": {name: self.meta["splits"][name]["sha256"] for name in SPLITS},
            "base": {"model_id": self.base_row["model_id"], "params_sha256": self.base_row["params_sha256"]},
            "allclass": (None if self.allclass_row is None else
                         {"model_id": self.allclass_row["model_id"],
                          "params_sha256": self.allclass_row["params_sha256"]}),
        }


def _allclass_row(seed: int) -> Mapping[str, Any] | None:
    from weightcode.models.export import read_fewshot_index

    rows = [row for row in read_fewshot_index(FS.DATASET)
            if row.get("role") == FS.ALLCLASS_ROLE and row.get("arch") == FS.ARCH and int(row["seed"]) == int(seed)]
    if len(rows) > 1:
        raise FscilRefError(f"{len(rows)} all-class students of seed {seed}")
    return rows[0] if rows else None


def load_stream_data(seed: int = FS.DEV_SEED, *, need_allclass: bool = True) -> StreamData:
    """Load and hash-check the splits, forward.py and the students of ``seed``."""
    import json

    meta = read_dataset_meta(FS.DATASET)
    forward, forward_sha, _ = load_dataset_forward(FS.DATASET, meta)
    splits = {name: load_checked_split(split_path(FS.DATASET, name), meta["splits"][name]["sha256"])
              for name in SPLITS}
    mean, std = normalization(meta)
    base_row = FS.base_student(seed)
    _, base_params, base_private, spec_text = FS.model_record(base_row, FS.BASE_ROLE)
    if base_private.get("train_classes") != list(range(FS.N_BASE_CLASSES)):
        raise FscilRefError(f"base student {base_row['model_id']} is not trained on the base classes only")
    allclass_row = _allclass_row(seed)
    if allclass_row is None and need_allclass:
        raise FscilRefError(f"no all-class student of seed {seed}")
    allclass_params = None if allclass_row is None else FS.model_record(allclass_row, FS.ALLCLASS_ROLE)[1]
    return StreamData(meta=meta, forward=forward, forward_sha256=forward_sha, splits=splits, mean=tuple(mean),
                      std=tuple(std), spec=json.loads(spec_text), class_names=tuple(meta["class_names"]),
                      base_row=base_row, base_params=base_params, allclass_row=allclass_row,
                      allclass_params=allclass_params)


@dataclass(frozen=True)
class SessionRows:
    """The rows of session t, the same for every stream: fit rows (new, old, prev), check parts, readout positions."""

    session: int
    x_fit: np.ndarray
    y_fit: np.ndarray
    part_fit: np.ndarray
    check: Mapping[str, np.ndarray]
    limits: Limits
    record: Mapping[str, Any]

    @property
    def n_seen(self) -> int:
        return FS.n_seen(self.session)


def session_rows(data: StreamData, session: int) -> SessionRows:
    """The rows of session ``session`` (module docstring), as ``weightcode.evalkit.fscil``'s instance plan draws them."""
    t = int(session)
    if t not in FS.SESSIONS:
        raise FscilRefError(f"session {session} is not one of {list(FS.SESSIONS)}")
    v_ev, replay = data.splits["V_evidence"], data.splits["old_replay"]
    new_rows = FS.shot_rows(v_ev["y"], [t])
    prev_rows = FS.shot_rows(v_ev["y"], list(range(FS.FIRST_SESSION, t)))
    replay_rows = FS.draw_replay_rows(len(replay["y"]))
    x = np.concatenate([v_ev["x"][new_rows], replay["x"][replay_rows], v_ev["x"][prev_rows]])
    y = np.concatenate([v_ev["y"][new_rows], replay["y"][replay_rows], v_ev["y"][prev_rows]]).astype(np.int64)
    part = np.array(["new"] * len(new_rows) + ["old"] * len(replay_rows) + ["prev"] * len(prev_rows))
    if np.any(y >= FS.n_seen(t)) or np.any(y[part == "old"] >= FS.N_BASE_CLASSES):
        raise FscilRefError(f"session {t} fit rows hold a class outside the seen or base classes")
    check = FS.fscil_check_parts(data.splits["V_check"], data.splits["old_check"], t)
    n_c, _ = FS.old_part_sizes(t, data.meta)
    if len(check["y_old"]) != n_c:
        raise FscilRefError(f"session {t}: the check's old part has {len(check['y_old'])} rows, not {n_c}")
    task = FS.build_fscil_task(data.class_names, t, FS.fscil_limits(data.spec))
    limits = FS.fscil_program_limits(task["limits"])
    record = {"session": t, "n_seen": FS.n_seen(t), "n_new": int(len(new_rows)), "n_old": int(len(replay_rows)),
              "n_prev": int(len(prev_rows)), "evidence_rows_new": new_rows.tolist(),
              "evidence_rows_prev": prev_rows.tolist(), "replay_rows": replay_rows.tolist(),
              "n_check_new": int(len(check["y_new"])), "n_check_old": int(len(check["y_old"]))}
    return SessionRows(session=t, x_fit=np.ascontiguousarray(x), y_fit=y, part_fit=part, check=check, limits=limits,
                       record=record)


# ---------------------------------------------------------------------------------------------------------------
# The functional ResNet-20


def trainable_names(params: Mapping[str, np.ndarray]) -> list[str]:
    """Every tensor but the BatchNorm running statistics, in the model's order."""
    return [name for name in params if not name.endswith(("running_mean", "running_var"))]


def to_tensors(params: Mapping[str, np.ndarray], dtype: torch.dtype,
               trainable: Sequence[str] = ()) -> dict[str, torch.Tensor]:
    """Fresh torch copies of ``params`` in ``dtype``; the ``trainable`` ones require gradients."""
    unknown = sorted(set(trainable) - set(params))
    if unknown:
        raise FscilRefError(f"trainable tensors {unknown} are not parameters of the student")
    return {key: torch.tensor(np.asarray(value), dtype=dtype).requires_grad_(key in trainable)
            for key, value in params.items()}


def _bn(h: torch.Tensor, t: Mapping[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return F.batch_norm(h, t[f"{prefix}.running_mean"], t[f"{prefix}.running_var"], t[f"{prefix}.weight"],
                        t[f"{prefix}.bias"], training=False, eps=R.BN_EPS)


def resnet_features(t: Mapping[str, torch.Tensor], x: torch.Tensor, mean: Sequence[float],
                    std: Sequence[float]) -> torch.Tensor:
    """Pooled encoder features [N, 64] of raw uint8 images, in the dtype of the tensors (BatchNorm in eval form)."""
    dtype = t["conv1.weight"].dtype
    m = torch.tensor(list(mean), dtype=torch.float64).view(1, -1, 1, 1).to(dtype)
    s = torch.tensor(list(std), dtype=torch.float64).view(1, -1, 1, 1).to(dtype)
    h = (x.to(dtype) / 255.0 - m) / s
    h = F.relu(_bn(F.conv2d(h, t["conv1.weight"], stride=1, padding=1), t, "bn1"))
    for name, c_in, c_out, stride in R.block_names():
        out = F.relu(_bn(F.conv2d(h, t[f"{name}.conv1.weight"], stride=stride, padding=1), t, f"{name}.bn1"))
        out = _bn(F.conv2d(out, t[f"{name}.conv2.weight"], stride=1, padding=1), t, f"{name}.bn2")
        if R.has_downsample(c_in, c_out, stride):
            residual = _bn(F.conv2d(h, t[f"{name}.downsample.0.weight"], stride=stride), t, f"{name}.downsample.1")
        else:
            residual = h
        h = F.relu(out + residual)
    return h.mean(dim=(2, 3))


def cosine_logits(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The cosine head: 16 x cos(f, W_c) for every class."""
    return R.COSINE_SCALE * F.linear(F.normalize(features, p=2, dim=-1, eps=R.NORM_EPS),
                                     F.normalize(weight, p=2, dim=-1, eps=R.NORM_EPS))


def resnet_logits(t: Mapping[str, torch.Tensor], x: torch.Tensor, mean: Sequence[float],
                  std: Sequence[float]) -> torch.Tensor:
    return cosine_logits(resnet_features(t, x, mean, std), t[FS.HEAD_TENSOR])


@dataclass
class FitContext:
    """One (stream, session): the current student, the session's rows and their cached features and logits."""

    method: str
    rows: SessionRows
    params: Params
    mean: tuple[float, ...]
    std: tuple[float, ...]
    x: torch.Tensor = field(init=False)
    y: torch.Tensor = field(init=False)
    f64: np.ndarray = field(init=False)
    f32: torch.Tensor = field(init=False)
    pred: np.ndarray = field(init=False)
    parity: dict[str, float] = field(init=False)

    def bind(self, forward: ForwardFn) -> FitContext:
        """Cache the features and check both rebuilds against forward.py on the fit rows."""
        self.x = torch.from_numpy(self.rows.x_fit)
        self.y = torch.from_numpy(self.rows.y_fit)
        with torch.no_grad():
            t64 = to_tensors(self.params, torch.float64)
            t32 = to_tensors(self.params, torch.float32)
            f64 = resnet_features(t64, self.x, self.mean, self.std)
            self.f32 = resnet_features(t32, self.x, self.mean, self.std)
            exact = cosine_logits(f64, t64[FS.HEAD_TENSOR]).numpy()
            single = cosine_logits(self.f32, t32[FS.HEAD_TENSOR]).numpy()
        self.f64 = f64.numpy()
        mine = np.concatenate([np.asarray(forward(dict(self.params), self.rows.x_fit[i : i + FS.EVALUATION_BATCH]),
                                          dtype=np.float64)
                               for i in range(0, len(self.rows.x_fit), FS.EVALUATION_BATCH)])
        self.parity = check_logit_parity(mine, exact, single)
        self.pred = mine[:, : self.rows.n_seen].argmax(axis=1)
        return self

    def loss_full(self) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
        k = self.rows.n_seen

        def loss(t: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return F.cross_entropy(resnet_logits(t, self.x, self.mean, self.std)[:, :k], self.y)

        return loss

    def loss_head(self) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
        k = self.rows.n_seen

        def loss(t: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return F.cross_entropy(cosine_logits(self.f32, t[FS.HEAD_TENSOR])[:, :k], self.y)

        return loss


@dataclass
class StreamCandidate:
    """One candidate of a stream at a session: the child, its configuration and notes."""

    patch_id: str
    child: Params
    config: dict[str, Any]
    notes: dict[str, Any]


# ---------------------------------------------------------------------------------------------------------------
# Methods


def _with_head_rows(params: Params, rows: Sequence[int], values: np.ndarray) -> Params:
    child = dict(params)
    head = np.array(params[FS.HEAD_TENSOR], copy=True)
    head[np.asarray(rows)] = np.asarray(values, dtype=np.float64).astype(head.dtype)
    child[FS.HEAD_TENSOR] = head
    return child


def _unit(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.all(norms > 0.0):
        raise FscilRefError("a zero vector has no direction")
    return v / norms


def imprint_directions(ctx: FitContext) -> dict[str, np.ndarray]:
    """The rows of session t's classes for each imprint direction (module docstring), float64 [5, 64]."""
    classes = FS.session_classes(ctx.rows.session)
    new = ctx.rows.part_fit == "new"
    f, y = ctx.f64, ctx.rows.y_fit
    mean = np.stack([f[new & (y == c)].mean(axis=0) for c in classes])
    center = f[ctx.rows.part_fit == "old"].mean(axis=0)
    centered, n_without = class_directions(f[new], y[new], classes, center=center)
    if n_without:
        raise FscilRefError(f"{n_without} shot features equal the replay mean")
    return {"mean": mean, "centered": centered}


def _seen_pred(f64: np.ndarray, head: np.ndarray, k: int) -> np.ndarray:
    """Masked argmax of the cosine head on cached features (forward.py's rule; the scale does not move the argmax)."""
    return (_unit(f64) @ _unit(np.asarray(head, dtype=np.float64)[:k]).T).argmax(axis=1)


def retain_beta(ctx: FitContext, unit_rows: np.ndarray) -> dict[str, Any]:
    """imprint_retain's beta for one direction: the smallest ladder value that breaks no protected row."""
    classes = FS.session_classes(ctx.rows.session)
    k = ctx.rows.n_seen
    part, y = ctx.rows.part_fit, ctx.rows.y_fit
    protected = (part != "new") & (ctx.pred == y)
    mu = _unit(ctx.f64[part == "old"].mean(axis=0))
    head = np.asarray(ctx.params[FS.HEAD_TENSOR], dtype=np.float64)
    ladder = []
    for beta in RETAIN_BETAS:
        rows = _unit(unit_rows - beta * mu[None, :])
        trial = head.copy()
        trial[classes] = rows
        pred = _seen_pred(ctx.f64, trial, k)
        ladder.append({"beta": beta, "broken": int(np.count_nonzero(protected & (pred != y))),
                       "new_right": int(np.count_nonzero((part == "new") & (pred == y)))})
    best = min(ladder, key=lambda entry: (entry["broken"], entry["beta"]))
    return {"beta": best["beta"], "broken_protected": best["broken"], "new_right_fit": best["new_right"],
            "n_protected": int(np.count_nonzero(protected)), "n_new_fit": int(np.count_nonzero(part == "new")),
            "ladder": ladder, "rows": _unit(unit_rows - best["beta"] * mu[None, :])}


def _imprint_candidates(ctx: FitContext, retain: bool) -> list[StreamCandidate]:
    classes = FS.session_classes(ctx.rows.session)
    out = []
    for direction, rows in imprint_directions(ctx).items():
        notes: dict[str, Any] = {"classes": classes}
        if retain:
            beta = retain_beta(ctx, _unit(rows))
            rows = beta.pop("rows")
            notes["retain"] = beta
        out.append(StreamCandidate(patch_id="", child=_with_head_rows(ctx.params, classes, rows),
                                   config={"direction": direction, **({"beta": notes["retain"]["beta"]}
                                                                      if retain else {})}, notes=notes))
    return out


def sphere_ridge_rows(h1: np.ndarray, targets: np.ndarray, w0: np.ndarray, lam: float) -> np.ndarray:
    """Per row j, the unit w minimizing ||h1 w - targets[:, j]||^2 + lam ||w - w0_j||^2 (``ridge_margin``).

    The ridge objective restricted to unit rows, the only rows a cosine head distinguishes. With G = h1^T h1 + lam I
    and b = h1^T t + lam w0_j, the minimizer is w = (G - nu I)^-1 b with nu < min eig(G) such that ||w|| = 1 (the
    trust-region secular equation; ||w(nu)|| increases in nu there), found by bisection to float64 precision. The
    hard case (b orthogonal to the bottom eigenvector with ||w(e_min)|| < 1) adds that eigenvector to reach norm 1.
    """
    if not lam > 0.0:
        raise FscilRefError(f"lam must be positive, got {lam}")
    gram = h1.T @ h1 + lam * np.eye(h1.shape[1])
    e, q = np.linalg.eigh(gram)
    rows = []
    for j in range(w0.shape[0]):
        b = q.T @ (h1.T @ targets[:, j] + lam * w0[j])

        def norm2(nu: float, b: np.ndarray = b) -> float:
            return float(np.sum((b / (e - nu)) ** 2))

        hi = float(e[0])
        lo = hi - float(np.linalg.norm(b)) - 1.0
        if norm2(np.nextafter(hi, -np.inf)) < 1.0:  # hard case
            if not np.all(e[1:] > hi):
                raise FscilRefError("hard case with a repeated bottom eigenvalue; not handled")
            w = b.copy()
            w[1:] = b[1:] / (e[1:] - hi)
            w[0] = np.sqrt(max(0.0, 1.0 - float(np.sum(w[1:] ** 2))))
        else:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if mid in (lo, hi):
                    break
                lo, hi = (mid, hi) if norm2(mid) < 1.0 else (lo, mid)
            w = b / (e - lo)
        rows.append(q @ (w / np.linalg.norm(w)))
    return np.stack(rows)


def _ridge_candidates(ctx: FitContext) -> list[StreamCandidate]:
    k = ctx.rows.n_seen
    h1 = R.COSINE_SCALE * _unit(ctx.f64)
    w0 = _unit(np.asarray(ctx.params[FS.HEAD_TENSOR], dtype=np.float64)[:k])
    z0 = h1 @ w0.T
    targets, c = margin_targets(z0, ctx.rows.y_fit)
    mean_sq = float(np.mean(np.sum(h1 * h1, axis=1)))
    out = []
    for s in RIDGE_SCALES:
        w = sphere_ridge_rows(h1, targets, w0, s * mean_sq)
        head = np.concatenate([w, np.zeros((FS.N_CLASSES - k, w.shape[1]))])
        out.append(StreamCandidate(patch_id="", child=_with_head_rows(ctx.params, list(range(k)), w),
                                   config={"s": s}, notes={"lam": s * mean_sq, "margin_c": c,
                                                           "fit_right": int(np.count_nonzero(
                                                               _seen_pred(ctx.f64, head, k) == ctx.rows.y_fit)),
                                                           "fit_objective": float(
                                                               np.sum((h1 @ w.T - targets) ** 2)
                                                               + s * mean_sq * np.sum((w - w0) ** 2))}))
    return out


def _scale(ctx: FitContext, child: Params, trainable: Sequence[str], notes: dict[str, Any]) -> Params:
    limits = ctx.rows.limits
    child, scaling = scale_to_norm_limit(ctx.params, child, trainable, limit=limits.max_rel_frob,
                                         min_ndim=limits.rel_frob_min_ndim, exempt=limits.rel_frob_exempt)
    notes["norm_scaled_tensors"] = [key for key, entry in scaling.items() if entry["scaled"]]
    notes["norm_scaling_factors"] = {key: scaling[key]["factor"] for key in notes["norm_scaled_tensors"]}
    return child


def _check_frozen(parent: Params, child: Mapping[str, np.ndarray], trainable: Sequence[str],
                  masks: Mapping[str, np.ndarray] | None) -> None:
    for key, value in parent.items():
        if key not in trainable and not np.array_equal(child[key], value):
            raise FscilRefError(f"{key} is not trainable but changed")
        if masks is not None and key in masks and not np.array_equal(child[key][~masks[key]], value[~masks[key]]):
            raise FscilRefError(f"{key} changed outside its mask")


def _grad_candidates(ctx: FitContext, trainable: Sequence[str], loss: Callable, enforced: bool,
                     masks: Mapping[str, np.ndarray] | None = None) -> list[StreamCandidate]:
    out: list[StreamCandidate] = []
    by_lr: dict[float, list[int]] = {}
    for lr, steps in grad_grid():
        by_lr.setdefault(lr, []).append(steps)
    for lr, checkpoints in by_lr.items():
        started = time.perf_counter()
        for steps, child, value in adam_checkpoints(ctx.params, loss, trainable=trainable, lr=lr,
                                                    checkpoints=checkpoints, seed=RUN_SEED, masks=masks):
            _check_frozen(ctx.params, child, trainable, masks)
            notes: dict[str, Any] = {"fit_loss_at_checkpoint": value,
                                     "trajectory_s_to_checkpoint": time.perf_counter() - started}
            if enforced:
                child = _scale(ctx, child, trainable, notes)
            out.append(StreamCandidate(patch_id="", child=child, config={"lr": lr, "steps": steps}, notes=notes))
    return out


def theta_full(ctx: FitContext) -> dict[str, np.ndarray]:
    """Unconstrained Adam at TRAJ_LR for TRAJ_STEPS steps on every trainable tensor of the current student."""
    ((_, theta, _),) = adam_checkpoints(ctx.params, ctx.loss_full(), trainable=trainable_names(ctx.params),
                                        lr=TRAJ_LR, checkpoints=(TRAJ_STEPS,), seed=RUN_SEED)
    return theta


def saliency(ctx: FitContext) -> dict[str, np.ndarray]:
    """|d loss / d w| at the current student for every trainable tensor."""
    names = trainable_names(ctx.params)
    t = to_tensors(ctx.params, torch.float32, names)
    ctx.loss_full()(t).backward()
    return {key: np.abs(t[key].grad.numpy()) for key in names}


def generate(ctx: FitContext) -> list[StreamCandidate]:
    """The candidates of ``ctx.method`` at its session, in check order, with patch ids set."""
    method = ctx.method
    names = trainable_names(ctx.params)
    if method == "imprint":
        out = _imprint_candidates(ctx, retain=False)
    elif method == "imprint_plain":
        out = _imprint_candidates(ctx, retain=False)[:1]
    elif method == "imprint_retain":
        out = _imprint_candidates(ctx, retain=True)
    elif method == "ridge_margin":
        out = _ridge_candidates(ctx)
    elif method == "grad_last":
        out = _grad_candidates(ctx, [FS.HEAD_TENSOR], ctx.loss_head(), enforced=True)
    elif method == "grad_full":
        out = _grad_candidates(ctx, names, ctx.loss_full(), enforced=False)
    elif method == "grad_sparse19_sal":
        masks = top_fraction_masks(saliency(ctx), names, MASK_FRACTION)
        out = _grad_candidates(ctx, names, ctx.loss_full(), enforced=True, masks=masks)
    elif method == "grad_sparse19_traj":
        theta = theta_full(ctx)
        score = {key: np.asarray(theta[key], dtype=np.float64) - np.asarray(ctx.params[key], dtype=np.float64)
                 for key in names}
        masks = top_fraction_masks(score, names, MASK_FRACTION)
        out = _grad_candidates(ctx, names, ctx.loss_full(), enforced=True, masks=masks)
    elif method == "interp":
        theta = theta_full(ctx)
        out = []
        for alpha in INTERP_ALPHAS:
            child = dict(ctx.params)
            for key in names:
                parent = np.asarray(ctx.params[key], dtype=np.float64)
                child[key] = (parent + alpha * (np.asarray(theta[key], dtype=np.float64) - parent)).astype(
                    ctx.params[key].dtype)
            out.append(StreamCandidate(patch_id="", child=child, config={"alpha": alpha}, notes={}))
    else:
        raise FscilRefError(f"unknown stream method {method!r}")
    if len(out) > CHECKS_PER_SESSION:
        raise FscilRefError(f"{method} made {len(out)} candidates, more than {CHECKS_PER_SESSION} checks")
    for index, candidate in enumerate(out):
        candidate.patch_id = f"{method}_s{ctx.rows.session}_q{index}"
        candidate.child = validate_child(ctx.params, candidate.child)
        edit = edit_stats_under(ctx.params, candidate.child, ctx.rows.limits)
        edit.pop("per_tensor")
        candidate.notes["edit"] = edit
        if method not in LIMITS_IGNORED and not edit["within_limits"]:
            raise FscilRefError(f"{candidate.patch_id} breaks the limits: {edit['violations']}")
    return out


def fit_stream(method: str, params: Params, rows: SessionRows, data: StreamData) -> dict[str, Any]:
    """Generate one stream's candidates at one session; returns them with the parity record and timings."""
    started = time.perf_counter()
    ctx = FitContext(method=method, rows=rows, params=dict(params), mean=data.mean, std=data.std).bind(data.forward)
    candidates = generate(ctx)
    return {"method": method, "session": rows.session, "parent_params_sha256": params_sha256(params),
            "parity": ctx.parity, "parent_fit_right": int(np.count_nonzero(ctx.pred == rows.y_fit)),
            "candidates": candidates, "fit_s": time.perf_counter() - started}


# ---------------------------------------------------------------------------------------------------------------
# Checks, the floor and the pick


class ChunkedForward:
    """forward.py over rows in chunks of ``EVALUATION_BATCH`` from row 0, optionally mapped over a process pool.

    The chunks are the batches ``predict(forward, params, x, EVALUATION_BATCH)`` would pass, so the logits are the
    oracle's, whatever the pool.
    """

    def __init__(self, forward: ForwardFn, pool_map: Callable[[Callable, list], list] | None = None,
                 chunk_fn: Callable[[Params, np.ndarray], np.ndarray] | None = None) -> None:
        self._forward = forward
        self._map = pool_map
        self._chunk_fn = chunk_fn

    def __call__(self, params: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        chunks = [x[i : i + FS.EVALUATION_BATCH] for i in range(0, len(x), FS.EVALUATION_BATCH)]
        if self._map is None or len(chunks) < 2:
            parts = [np.asarray(self._forward(dict(params), chunk)) for chunk in chunks]
        else:
            parts = self._map(self._chunk_fn, [(dict(params), chunk) for chunk in chunks])
        return np.concatenate(parts) if parts else np.zeros((0, FS.N_CLASSES))


def check_stream(fit: Mapping[str, Any], parent: Params, rows: SessionRows, forward: ForwardFn) -> list[dict[str, Any]]:
    """Check every candidate of one stream once, in order, on the session's joint oracle; returns the oracle log."""
    parts = rows.check
    oracle = JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"],
                                   FS.masked_forward(forward, rows.session), parent, CHECKS_PER_SESSION,
                                   batch_size=10**9)
    for candidate in fit["candidates"]:
        oracle.query(candidate.child, candidate.patch_id)
    return oracle.log


def flip_rate(logs: Mapping[str, Sequence[Mapping[str, Any]]], n_c: int) -> dict[str, Any]:
    """d = median over every checked candidate of (fixed_old + broken_old) / n_c, with its count and range."""
    flips = []
    for method, log in logs.items():
        for entry in log:
            if int(entry["n_old"]) != n_c:
                raise FscilRefError(f"{method}: a check's old part has {entry['n_old']} rows, not {n_c}")
            flips.append((int(entry["fixed_old"]) + int(entry["broken_old"])) / n_c)
    if not flips:
        raise FscilRefError("no checked candidate to take the flip rate from")
    return {"d": float(statistics.median(flips)), "n_checks": len(flips), "min": min(flips), "max": max(flips),
            "streams": sorted(logs)}


def keep(log: Sequence[Mapping[str, Any]], floor: int) -> Candidate:
    """Kept rule joint-retain-v1: argmax S over the checks with S_old >= floor and the current student (S = 0)."""
    eligible = [Candidate(patch_id=str(entry["patch_id"]), score=int(entry["score"]),
                          n_changed=int(entry["n_changed"]), query_index=int(entry["query_index"]))
                for entry in log if int(entry["score_old"]) >= floor]
    return choose_final(eligible)


def candidate_floors(log: Sequence[Mapping[str, Any]], n_c: int, n_s: int, r_points: float) -> list[int]:
    """Per check, floor_c = joint_retain_floor(n_c, n_s, R, d_c) with d_c its own (fixed_old + broken_old) / n_c."""
    return [FS.joint_retain_floor(n_c, n_s, r_points, (int(e["fixed_old"]) + int(e["broken_old"])) / n_c)
            for e in log]


def keep_per_candidate(log: Sequence[Mapping[str, Any]], n_c: int, n_s: int, r_points: float) -> Candidate:
    """Supplement, never gating (WEIGHTCODE-main, 2026-11-04): argmax S over the checks with S_old >= their own floor_c,
    and the current student (S = 0); ties as in ``keep``."""
    floors = candidate_floors(log, n_c, n_s, r_points)
    eligible = [Candidate(patch_id=str(e["patch_id"]), score=int(e["score"]), n_changed=int(e["n_changed"]),
                          query_index=int(e["query_index"]))
                for e, floor in zip(log, floors) if int(e["score_old"]) >= floor]
    return choose_final(eligible)


# ---------------------------------------------------------------------------------------------------------------
# Readouts


@dataclass(frozen=True)
class ReadoutRows:
    """old_select then all of V_select: one forward per model serves every session's readout (slices)."""

    x: np.ndarray
    y: np.ndarray
    n_old: int

    def session(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Positions of session t's readout rows (``readout_rows`` order: old, prev, new) and their parts."""
        y_v = self.y[self.n_old :]
        old = np.arange(self.n_old)
        prev = self.n_old + np.flatnonzero(np.isin(y_v, FS.prev_classes(t)))
        new = (self.n_old + np.flatnonzero(np.isin(y_v, FS.session_classes(t))) if t >= FS.FIRST_SESSION
               else np.zeros(0, dtype=np.int64))
        part = np.array(["old"] * old.size + ["prev"] * prev.size + ["new"] * new.size)
        return np.concatenate([old, prev, new]).astype(np.int64), part


def readout_rows(data: StreamData) -> ReadoutRows:
    o_sel, v_sel = data.splits["old_select"], data.splits["V_select"]
    if np.any(np.asarray(o_sel["y"]) >= FS.N_BASE_CLASSES):
        raise FscilRefError("old_select holds a row of an incremental class")
    return ReadoutRows(x=np.ascontiguousarray(np.concatenate([o_sel["x"], v_sel["x"]])),
                       y=np.concatenate([o_sel["y"], v_sel["y"]]).astype(np.int64), n_old=int(len(o_sel["y"])))


def session_readout(rows: ReadoutRows, parent_logits: np.ndarray, child_logits: np.ndarray, base_logits: np.ndarray,
                    t: int) -> dict[str, Any]:
    """Absolute masked readout of the kept model at session t, against its parent and against the session-1 base."""
    position, part = rows.session(t)
    k = FS.n_seen(t)
    y = rows.y[position]
    child = child_logits[position, :k].argmax(axis=1)
    parent = parent_logits[position, :k].argmax(axis=1)
    base = base_logits[position, :k].argmax(axis=1)
    absolute = FS.accuracy_summary(child, y, part, t)
    against_parent = FS.readout_summary(parent, child, y, part, t)
    against_base = FS.readout_summary(base, child, y, part, t)
    for summary in (against_parent, against_base):
        summary.pop("per_class")
    return {"absolute": absolute, "vs_parent": against_parent, "vs_base": against_base}


# ---------------------------------------------------------------------------------------------------------------
# The lockstep run


FitMany = Callable[[list[tuple[str, Params, int]]], list[dict[str, Any]]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Any, payload: Any) -> str:
    import hashlib
    import json
    from pathlib import Path

    text = json.dumps(_jsonable(payload), indent=1, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def serial_fit_many(data: StreamData) -> FitMany:
    def run(tasks: list[tuple[str, Params, int]]) -> list[dict[str, Any]]:
        return [fit_stream(method, params, session_rows(data, t), data) for method, params, t in tasks]
    return run


def run_streams(
    data: StreamData,
    *,
    r_points: float,
    out_dir: Any,
    sessions: Sequence[int],
    methods: Sequence[str] = STREAMS,
    fit_many: FitMany | None = None,
    forward: ForwardFn | None = None,
    provenance: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the streams in lockstep over ``sessions`` (consecutive, from 2 or resuming after the last written one).

    Writes ``run.json`` (once), ``session_<t>.json`` per session and the kept models ``models/<method>/s<t>.npz``
    under ``out_dir``; refuses to overwrite. Resuming at session t needs every earlier session's record there.
    Returns the per-session floors.
    """
    from pathlib import Path

    from weightcode.common.identity import load_params, save_params

    out = Path(out_dir)
    sessions = [int(t) for t in sessions]
    if sessions != list(range(sessions[0], sessions[0] + len(sessions))) or not set(sessions) <= set(FS.SESSIONS):
        raise FscilRefError(f"sessions {sessions} must be consecutive within {list(FS.SESSIONS)}")
    unknown = sorted(set(methods) - set(STREAMS))
    if unknown or not set(methods) & set(STREAM_METHODS):
        raise FscilRefError(f"methods {list(methods)}: unknown {unknown} or no checked stream")
    fit_many = serial_fit_many(data) if fit_many is None else fit_many
    forward = data.forward if forward is None else forward
    run_record = {"schema": STREAMS_SCHEMA, "methods": list(methods), "r_points": float(r_points),
                  "run_seed": RUN_SEED, "checks_per_session": CHECKS_PER_SESSION, "retain_betas": list(RETAIN_BETAS),
                  "inputs": data.provenance(), **dict(provenance or {})}
    if (out / "run.json").exists():
        import json

        written = json.loads((out / "run.json").read_text(encoding="utf-8"))
        for key in ("schema", "methods", "r_points", "run_seed", "inputs", "validity", "fit_threads", "code_commit"):
            if written.get(key) != _jsonable(run_record.get(key)):
                raise FscilRefError(f"{out}/run.json differs in {key}; resume refused")
    else:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "run.json", run_record)
    first = sessions[0]
    for t in range(FS.FIRST_SESSION, first):
        if not (out / f"session_{t}.json").exists():
            raise FscilRefError(f"resume at session {first} needs {out}/session_{t}.json")
    current: dict[str, Params] = {}
    for method in methods:
        current[method] = (dict(data.base_params) if first == FS.FIRST_SESSION
                           else load_params(out / "models" / method / f"s{first - 1}.npz"))
    readout = readout_rows(data)
    started = time.perf_counter()
    base_logits = np.asarray(forward(dict(data.base_params), readout.x), dtype=np.float64)
    parent_logits = {}
    by_sha: dict[str, np.ndarray] = {params_sha256(data.base_params): base_logits}
    for method in methods:
        sha = params_sha256(current[method])
        if sha not in by_sha:
            by_sha[sha] = np.asarray(forward(dict(current[method]), readout.x), dtype=np.float64)
        parent_logits[method] = by_sha[sha]
    log(f"readout logits of the base and the current students: {time.perf_counter() - started:.0f} s")
    floors = {}
    checked = [m for m in methods if m in STREAM_METHODS]
    for t in sessions:
        rows = session_rows(data, t)
        t0 = time.perf_counter()
        fits = {fit["method"]: fit for fit in fit_many([(m, current[m], t) for m in methods])}
        t1 = time.perf_counter()
        logs = {m: check_stream(fits[m], current[m], rows, forward) for m in checked}
        t2 = time.perf_counter()
        n_c, n_s = FS.old_part_sizes(t, data.meta)
        flips = flip_rate(logs, n_c)
        floor = FS.joint_retain_floor(n_c, n_s, r_points, flips["d"])
        floors[t] = {"n_c": n_c, "n_s": n_s, **flips, "floor": floor}
        log(f"session {t}: fits {t1 - t0:.0f} s, checks {t2 - t1:.0f} s; d {flips['d']:.5f} over "
            f"{flips['n_checks']} checks, floor {floor}")
        records = {}
        (out / "models").mkdir(exist_ok=True)
        for m in methods:
            candidates = {c.patch_id: c for c in fits[m]["candidates"]}
            if m in INSTRUMENT_METHODS:
                (only,) = candidates.values()
                kept_id, child, kept, free = only.patch_id, only.child, None, None
            else:
                kept = keep(logs[m], floor)
                free = keep(logs[m], floor=-(10**9))  # no floor: argmax S, as context only
                kept_id = kept.patch_id
                child = current[m] if kept_id is None else candidates[kept_id].child
            sha = params_sha256(child)
            if sha not in by_sha:
                by_sha[sha] = np.asarray(forward(dict(child), readout.x), dtype=np.float64)
            child_logits = by_sha[sha]
            (out / "models" / m).mkdir(exist_ok=True)
            model_path = out / "models" / m / f"s{t}.npz"
            if model_path.exists():
                raise FscilRefError(f"{model_path} exists; nothing overwritten")
            save_params(model_path, child)
            entry = next((e for e in logs.get(m, []) if e["patch_id"] == kept_id), None)
            supplement = None
            if m not in INSTRUMENT_METHODS:
                own = keep_per_candidate(logs[m], n_c, n_s, r_points)
                own_child = current[m] if own.patch_id is None else candidates[own.patch_id].child
                own_sha = params_sha256(own_child)
                if own_sha not in by_sha:
                    by_sha[own_sha] = np.asarray(forward(dict(own_child), readout.x), dtype=np.float64)
                supplement = {
                    "rule": "per-candidate floor: argmax S over checks with S_old >= joint_retain_floor(n_c, n_s, R, "
                            "d_c), d_c the check's own (fixed_old + broken_old) / n_c, and the current student; "
                            "never gating, never the stream's next student",
                    "floors": candidate_floors(logs[m], n_c, n_s, r_points),
                    "pick": own.patch_id, "score": own.score, "params_sha256": own_sha,
                    "readout": session_readout(readout, parent_logits[m], by_sha[own_sha], base_logits, t),
                }
            records[m] = {
                "method": m,
                "fit": {key: fits[m][key] for key in ("parent_params_sha256", "parity", "parent_fit_right", "fit_s")},
                "candidates": [{"patch_id": c.patch_id, "config": c.config, "notes": c.notes}
                               for c in fits[m]["candidates"]],
                "checks": logs.get(m),
                "kept": {"patch_id": kept_id, "rule": "always kept, never checked" if m in INSTRUMENT_METHODS
                         else "joint-retain-v1", "check": entry, "params_sha256": sha, "model": str(model_path),
                         "is_parent": kept_id is None},
                "unconstrained_pick": None if free is None else {"patch_id": free.patch_id, "score": free.score},
                "supplement_per_candidate_floor": supplement,
                "readout": session_readout(readout, parent_logits[m], child_logits, base_logits, t),
            }
            parent_logits[m] = child_logits
            current[m] = child
        digest = _write_json(out / f"session_{t}.json", {"schema": STREAMS_SCHEMA, "session": t,
                                                          "rows": rows.record, "floor": floors[t],
                                                          "streams": records})
        floors[t]["record_sha256"] = digest
        log(f"session {t} written ({time.perf_counter() - t0:.0f} s)")
        by_sha = {params_sha256(data.base_params): base_logits,
                  **{params_sha256(current[m]): parent_logits[m] for m in methods}}
    return floors


__all__ = [
    "CHECKS_PER_SESSION", "ChunkedForward", "FitContext", "FscilRefError", "GATE_EXCLUDED", "IMPRINT_DIRECTIONS",
    "INSTRUMENT_METHODS", "LIMITS_IGNORED", "RETAIN_BETAS", "RUN_SEED", "ReadoutRows", "SPLITS", "STREAMS",
    "STREAMS_SCHEMA", "STREAM_METHODS", "SessionRows", "StreamCandidate", "StreamData", "candidate_floors",
    "check_stream", "cosine_logits", "fit_stream", "flip_rate", "generate", "imprint_directions", "keep",
    "keep_per_candidate", "load_stream_data", "readout_rows", "resnet_features", "resnet_logits", "retain_beta",
    "run_streams", "saliency", "serial_fit_many", "session_readout", "session_rows", "sphere_ridge_rows",
    "theta_full", "to_tensors", "trainable_names",
]
