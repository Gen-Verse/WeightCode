"""PREREG F1s, a stronger fine-tuning control at equal information (WEIGHTCODE-main 2026-11-05 at 16:10 Beijing, main
9a65b20; no LLM call). Does F1's result survive a better-tuned fine-tuning recipe?

Every arm fits F1's rows: the 64 condition rows, then the 64 clean replay rows, with the references' labels. Each arm
has 8 configurations and makes one joint check per configuration, under F1's oracle, kept rule and readouts. All
arms use F1's eval-form forward (no dropout), fp32, Adam (0.9, 0.999, 1e-8) with no weight decay, mean cross-entropy
and run seed 0. One trajectory per (arm part, lr) is checkpointed at the grid's steps. Nothing in a trajectory
depends on its checkpoints, so each checkpoint is the run of that many steps. None is held to the E10 program
limits, like F1's methods.

* ``f1s_s1_gentle``, gentle full fine-tuning: full batch, a 10-step linear warmup (step t uses lr * min(1, t/10)),
  lr {3e-6, 1e-5} x steps {30, 100, 300, 1000}.
* ``f1s_s2_aug``, full fine-tuning with augmentation. Each step takes a minibatch of 32 condition rows and 32 replay
  rows; each half walks its 64 rows in passes, with a fresh permutation per pass (``labeleq.BalancedBatchSampler``).
  Every row is randomly cropped (zero padding 4 on the uint8 image) and flipped horizontally with probability 1/2,
  with draws from a generator seeded with run seed + 1. lr {1e-5, 3e-5} x steps {100, 300, 1000, 3000}.
* ``f1s_s3_lora_aug``: LoRA rank 4 on F1's 28 targets (``ftcompare.lora_targets``, alpha = r, A from seed 0 and B
  = 0), merged for the check. s2's minibatches and augmentation. lr {3e-4, 1e-3} x steps {30, 100, 300, 1000}.
* ``f1s_s4_est_prior``, a label-free corruption estimate used as augmentation (equal data plus a prior: a menu of two
  corruption families). From x_new (the condition rows) against x_old (the replay rows) only
  (``estimate_corruption``), it fits two families. One is additive gaussian noise, with sigma from the high-pass
  residual. The other is contrast reduction, with the factor from the per-image pixel spread. It keeps the family
  whose synthetic copies of x_old match x_new better on two pixel statistics. It then fine-tunes, full batch and
  without crop or flip, on x_new with y_new, the synthetic copies with y_old, and x_old with y_old (192 rows).
  Grid {full fine-tuning lr 1e-5, LoRA r4 lr 1e-3} x steps {30, 100, 300, 1000}.
* ``f1s_s5_oracle_prior``: s4 with the true corruption, the pinned official make_cifar_c.py at severity 5 applied to
  x_old (``scripts/f1s/f1s_s5_rows.py``, run in E10's pinned environment). This gives more information than the
  program had. It is reported beside s4 and decides nothing about equal information.
* ``k1_ft_rule`` (PREREG K1's control, "rule to data to fine-tuning"; WEIGHTCODE-main main f155d9b): fine-tuning on
  synthetic copies of the 64 replay rows x_old made by the stated rule (the pinned official gaussian_noise at
  severity 5), 8 draws per image (seeds 1005 + k, k = 0 to 7), each with y_old, then the 64 x_old with y_old: 576
  rows, full batch, no crop or flip, and no x_new row anywhere in the fit (``generate`` reads only evidence rows
  after the condition rows). Grid and rule as s4's. The two trajectories run in two processes
  (``K1_WORKERS``), each a pure function of its inputs, so the result does not depend on the pool.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance

from .ftcompare import F1_DATASET, LORA_ALPHA_OVER_RANK, lora_init, lora_targets
from .grad import ADAM_BETAS, ADAM_EPS, _check_frozen_entries
from .labeleq import BalancedBatchSampler
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import (
    VITLITE,
    Architecture,
    architecture_from,
    assert_parity,
    images_tensor,
    logits,
    to_tensors,
    trainable_tensors,
)

PREREG_F1S = ("PREREG F1s (a stronger fine-tuning control at equal information, WEIGHTCODE-main 2026-11-05 at 16:10 "
              "Beijing, main 9a65b20)")
F1S_LABEL = "f1s"
"""Run-directory label: ``baselines/<instance_id>/<method>/f1s/<seed>/``."""
WARMUP_STEPS = 10
HALF = 32
CROP_PAD = 4
LORA_RANK = 4
HP_NOISE_GAIN = 8.0 / 9.0
"""Variance of the 3x3 box high-pass residual of i.i.d. noise per unit noise variance: (1 - 1/9)^2 + 8 (1/9)^2."""
FAMILIES = ("gaussian_noise", "contrast")
S5_ROWS_SCHEMA = "weightcode.f1s.s5_rows.v1"

GRIDS: dict[str, dict[str, Any]] = {
    "f1s_s1_gentle": {"parts": [("full", lr) for lr in (3e-6, 1e-5)], "steps": (30, 100, 300, 1000)},
    "f1s_s2_aug": {"parts": [("full", lr) for lr in (1e-5, 3e-5)], "steps": (100, 300, 1000, 3000)},
    "f1s_s3_lora_aug": {"parts": [("lora", lr) for lr in (3e-4, 1e-3)], "steps": (30, 100, 300, 1000)},
    "f1s_s4_est_prior": {"parts": [("full", 1e-5), ("lora", 1e-3)], "steps": (30, 100, 300, 1000)},
    "f1s_s5_oracle_prior": {"parts": [("full", 1e-5), ("lora", 1e-3)], "steps": (30, 100, 300, 1000)},
    "k1_ft_rule": {"parts": [("full", 1e-5), ("lora", 1e-3)], "steps": (30, 100, 300, 1000)},
}
RECIPES: dict[str, dict[str, Any]] = {
    "f1s_s1_gentle": {"warmup": WARMUP_STEPS, "minibatch": False, "augment": False, "prior": None},
    "f1s_s2_aug": {"warmup": 0, "minibatch": True, "augment": True, "prior": None},
    "f1s_s3_lora_aug": {"warmup": 0, "minibatch": True, "augment": True, "prior": None},
    "f1s_s4_est_prior": {"warmup": 0, "minibatch": False, "augment": False, "prior": "estimated"},
    "f1s_s5_oracle_prior": {"warmup": 0, "minibatch": False, "augment": False, "prior": "oracle"},
    "k1_ft_rule": {"warmup": 0, "minibatch": False, "augment": False, "prior": "rule"},
}

_TAIL = f"gradient-trained (not training-free), edit limits ignored; {PREREG_F1S}"
F1S_SPECS: dict[str, MethodSpec] = {
    "f1s_s1_gentle": MethodSpec(method="f1s_s1_gentle", uses_labels_for_fit=True, edit_limits_enforced=False,
                                label=f"F1s s1: gentle full fine-tuning (10-step warmup), condition and replay rows; {_TAIL}"),
    "f1s_s2_aug": MethodSpec(method="f1s_s2_aug", uses_labels_for_fit=True, edit_limits_enforced=False,
                             label=("F1s s2: full fine-tuning, crop and flip, minibatches of 32 condition + 32 replay "
                                    f"rows; {_TAIL}")),
    "f1s_s3_lora_aug": MethodSpec(method="f1s_s3_lora_aug", uses_labels_for_fit=True, edit_limits_enforced=False,
                                  label=f"F1s s3: LoRA r4 with s2's augmentation and minibatches; {_TAIL}"),
    "f1s_s4_est_prior": MethodSpec(method="f1s_s4_est_prior", uses_labels_for_fit=True, edit_limits_enforced=False,
                                   label=("F1s s4: fine-tuning with a label-free estimated corruption (two-family "
                                          f"prior) applied to the replay rows; {_TAIL}")),
    "f1s_s5_oracle_prior": MethodSpec(method="f1s_s5_oracle_prior", uses_labels_for_fit=True,
                                      edit_limits_enforced=False,
                                      label=("F1s s5: fine-tuning with the true corruption (official make_cifar_c.py, "
                                             f"severity 5) applied to the replay rows; more information; {_TAIL}")),
}
F1S_SPECS["k1_ft_rule"] = MethodSpec(
    method="k1_ft_rule", uses_labels_for_fit=True, edit_limits_enforced=False,
    label=("K1 control k1_ft_rule: fine-tuning on 8 synthetic copies (the stated rule, the pinned official "
           f"gaussian_noise at severity 5) of each replay row plus the replay rows, no condition row; {_TAIL}"))
F1S_METHODS: tuple[str, ...] = tuple(F1S_SPECS)
K1_METHOD = "k1_ft_rule"
K1_DRAWS = 8
K1_SEED0 = 1005
K1_ROWS_SCHEMA = "weightcode.k1.rule_rows.v1"
K1_WORKERS = 2


class F1sError(ValueError):
    """An F1s run was refused: unknown method, a non-E11 instance, or s5 rows that do not match."""


def f1s_configs(method: str) -> list[dict[str, Any]]:
    """The 8 configurations of ``method`` in check order: part (mode, lr) outer, steps inner."""
    if method not in GRIDS:
        raise F1sError(f"unknown F1s method {method!r}; expected one of {F1S_METHODS}")
    grid = GRIDS[method]
    with_mode = method in ("f1s_s4_est_prior", "f1s_s5_oracle_prior", K1_METHOD)
    return [({"mode": mode} if with_mode else {}) | {"lr": lr, "steps": s}
            for mode, lr in grid["parts"] for s in grid["steps"]]


# --------------------------------------------------------------------------------------------- augmentation


def crop_flip(x: torch.Tensor, shape: tuple[int, int, int], generator: torch.Generator) -> torch.Tensor:
    """Random crop with zero padding ``CROP_PAD`` and a horizontal flip with probability 1/2, per uint8 image.

    The draws per call are fixed: one (i, j) offset pair per image, then one flip draw per image."""
    c, h, w = shape
    images = x.reshape(x.shape[0], c, h, w)
    padded = F.pad(images, (CROP_PAD, CROP_PAD, CROP_PAD, CROP_PAD))
    offsets = torch.randint(0, 2 * CROP_PAD + 1, (images.shape[0], 2), generator=generator)
    flips = torch.rand(images.shape[0], generator=generator) < 0.5
    out = torch.empty_like(images)
    for k in range(images.shape[0]):
        i, j = int(offsets[k, 0]), int(offsets[k, 1])
        crop = padded[k, :, i:i + h, j:j + w]
        out[k] = crop.flip(-1) if bool(flips[k]) else crop
    return out.reshape(x.shape)


# --------------------------------------------------------------------------------------------- training


def train_checkpoints(arch: Architecture, params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor, *,
                      mode: str, lr: float, checkpoints: Sequence[int], seed: int, warmup: int = 0,
                      minibatch_split: int | None = None, augment: bool = False
                      ) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Adam from ``params`` on mean cross-entropy; (steps, params, full-set eval loss) at each checkpoint.

    ``mode`` "full" trains ``trainable_tensors``; "lora" trains rank-``LORA_RANK`` adapters on F1's targets and
    returns the merged weights. ``minibatch_split``: rows [0, split) and [split, N) are two halves; each step takes
    ``HALF`` rows from each (``BalancedBatchSampler``, generator seeded ``seed``). Otherwise every step is full
    batch. ``augment``: ``crop_flip`` on every row of every step (generator seeded ``seed + 1``). ``warmup``: step t
    uses lr * min(1, t / warmup)."""
    if mode not in ("full", "lora"):
        raise F1sError(f"unknown mode {mode!r}")
    wanted = sorted(set(int(s) for s in checkpoints))
    total = wanted[-1]
    torch.manual_seed(seed)
    if mode == "full":
        trainable = trainable_tensors(arch, params)
        tensors = to_tensors(params, trainable=trainable)
        leaves = [tensors[k] for k in trainable]

        def current() -> dict[str, torch.Tensor]:
            return tensors
    else:
        targets = lora_targets(arch, params)
        base = to_tensors(params)
        a, b = lora_init(params, targets, LORA_RANK, seed)
        leaves = [t for name in targets for t in (a[name], b[name])]

        def current() -> dict[str, torch.Tensor]:
            merged = dict(base)
            for name in targets:
                merged[name] = base[name] + LORA_ALPHA_OVER_RANK * (b[name] @ a[name])
            return merged
    optimizer = torch.optim.Adam(leaves, lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0, foreach=False)
    batches = None
    if minibatch_split is not None:
        n = int(len(y))
        batches = iter(BalancedBatchSampler(minibatch_split, n - minibatch_split, HALF, total,
                                            torch.Generator().manual_seed(seed)))
    aug_generator = torch.Generator().manual_seed(seed + 1)
    snapshots: list[tuple[int, dict[str, np.ndarray], float]] = []
    for step in range(1, total + 1):
        if batches is not None:
            index = torch.tensor(next(batches), dtype=torch.long)
            xb, yb = x[index], y[index]
        else:
            xb, yb = x, y
        if augment:
            xb = crop_flip(xb, arch.input_shape, aug_generator)
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, step / warmup) if warmup else lr
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(logits(arch, current(), xb), yb).backward()
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                merged = current()
                loss = float(F.cross_entropy(logits(arch, merged, x), y))
                snapshots.append((step, {k: v.detach().numpy().copy() for k, v in merged.items()}, loss))
    return snapshots


# --------------------------------------------------------------------------------------------- s4: estimate


def _unit(x: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return x.reshape(x.shape[0], *shape).astype(np.float64) / 255.0


def highpass_energy(a: np.ndarray) -> float:
    """Mean squared residual of each interior pixel against its 3x3 box mean, per channel (a: N, C, H, W in [0, 1])."""
    h, w = a.shape[2], a.shape[3]
    box = sum(a[:, :, 1 + di:h - 1 + di, 1 + dj:w - 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9.0
    return float(np.mean((a[:, :, 1:h - 1, 1:w - 1] - box) ** 2))


def pixel_spread(a: np.ndarray) -> float:
    """Mean over images of the root-mean-square deviation from the image's per-channel means."""
    centred = a - a.mean(axis=(2, 3), keepdims=True)
    return float(np.mean(np.sqrt(np.mean(centred ** 2, axis=(1, 2, 3)))))


def _store_uint8(a: np.ndarray) -> np.ndarray:
    """make_cifar_c.py's storage: clip to [0, 1], scale by 255, np.uint8 (truncation toward zero)."""
    return np.uint8(np.clip(a, 0.0, 1.0) * 255.0)


def synthesize(family: str, value: float, x_old: np.ndarray, shape: tuple[int, int, int], seed: int) -> np.ndarray:
    """Synthetic copies of ``x_old`` (uint8, as stored) under one family: additive gaussian noise of std ``value``
    (one ``default_rng(seed)`` draw in row order), or contrast reduction by the factor ``value`` around each image's
    per-channel means (no draw)."""
    a = _unit(x_old, shape)
    if family == "gaussian_noise":
        out = a + np.random.default_rng(seed).normal(0.0, value, size=a.shape)
    elif family == "contrast":
        means = a.mean(axis=(2, 3), keepdims=True)
        out = (a - means) * value + means
    else:
        raise F1sError(f"unknown family {family!r}")
    return _store_uint8(out).reshape(x_old.shape)


def estimate_corruption(x_new: np.ndarray, x_old: np.ndarray, shape: tuple[int, int, int], seed: int
                        ) -> tuple[np.ndarray, dict[str, Any]]:
    """The s4 estimate, label-free (x_new against x_old only): the chosen family's synthetic copies of x_old, and a
    record of both fits and their distances.

    gaussian noise: sigma^2 = max(0, hp(x_new) - hp(x_old)) / HP_NOISE_GAIN, from the high-pass energy hp. contrast:
    c = min(1, spread(x_new) / spread(x_old)), from the pixel spread. Each family's copies are scored by
    |log(spread ratio)| + |log(hp ratio)| against x_new, and the lower score is kept. A tie goes to the family listed
    first."""
    new, old = _unit(x_new, shape), _unit(x_old, shape)
    stats_new = {"spread": pixel_spread(new), "highpass": highpass_energy(new)}
    stats_old = {"spread": pixel_spread(old), "highpass": highpass_energy(old)}
    fits = {"gaussian_noise": float(np.sqrt(max(0.0, stats_new["highpass"] - stats_old["highpass"]) / HP_NOISE_GAIN)),
            "contrast": float(min(1.0, stats_new["spread"] / stats_old["spread"]))}
    record: dict[str, Any] = {"x_new_stats": stats_new, "x_old_stats": stats_old, "families": {}, "seed": int(seed),
                              "rule": "argmin |log spread ratio| + |log highpass ratio| against x_new"}
    copies = {}
    for family in FAMILIES:
        synth = synthesize(family, fits[family], x_old, shape, seed)
        s = _unit(synth, shape)
        stats = {"spread": pixel_spread(s), "highpass": highpass_energy(s)}
        distance = sum(abs(np.log(max(stats[k], 1e-12) / max(stats_new[k], 1e-12))) for k in ("spread", "highpass"))
        copies[family] = synth
        record["families"][family] = {"parameter": "sigma" if family == "gaussian_noise" else "factor",
                                      "value": fits[family], "synthetic_stats": stats, "distance": float(distance)}
    chosen = min(FAMILIES, key=lambda f: (record["families"][f]["distance"], FAMILIES.index(f)))
    record["chosen"] = chosen
    return copies[chosen], record


# --------------------------------------------------------------------------------------------- s5: oracle rows


def load_s5_rows(path: Path, condition: str, replay_sha256: str | None, n_old: int) -> tuple[np.ndarray, dict[str, Any]]:
    """The oracle copies of the replay rows for ``condition`` from the s5 record (``scripts/f1s/f1s_s5_rows.py``),
    checked: the record's replay content hash equals the evidence's, and the file's hash equals the record's."""
    record = json.loads(Path(path).read_text())
    if record.get("schema") != S5_ROWS_SCHEMA:
        raise F1sError(f"{path}: not a {S5_ROWS_SCHEMA} record")
    family = condition.rsplit("_s", 1)[0]
    entry = record["rows"].get(family)
    if entry is None:
        raise F1sError(f"{path}: no rows for {family}")
    if record["replay"]["content_sha256"] != replay_sha256:
        raise F1sError(f"{path}: made from replay content {record['replay']['content_sha256'][:12]}, the evidence has "
                       f"{str(replay_sha256)[:12]}")
    file = Path(path).parent / entry["file"]
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if digest != entry["file_sha256"]:
        raise F1sError(f"{file}: sha256 {digest[:12]} differs from the record's {entry['file_sha256'][:12]}")
    with np.load(file, allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["x"])
    if len(x) != n_old:
        raise F1sError(f"{file}: {len(x)} rows, the evidence has {n_old} replay rows")
    return x, {"record": str(path), "file": str(file), "file_sha256": digest, "seed": entry["seed"],
               "official_sha256": record["official"]["sha256"], "severity": record["severity"]}


def load_k1_rows(path: Path, condition: str, replay_sha256: str | None, n_old: int) -> tuple[np.ndarray, dict[str, Any]]:
    """The K1 rule draws of the replay rows (``scripts/k1/k1_rule_rows.py``): an array [K1_DRAWS, n_old, ...] of uint8
    rows, draw k made with seed ``K1_SEED0 + k`` in replay row order. Checked: the record's schema, draws, seeds,
    replay content hash and the file's hash."""
    record = json.loads(Path(path).read_text())
    if record.get("schema") != K1_ROWS_SCHEMA:
        raise F1sError(f"{path}: not a {K1_ROWS_SCHEMA} record")
    function = condition.rsplit("_s", 1)[0]
    entry = record["rows"].get(function)
    if entry is None:
        raise F1sError(f"{path}: no rows for {function}")
    if entry["seeds"] != [K1_SEED0 + k for k in range(K1_DRAWS)]:
        raise F1sError(f"{path}: seeds {entry['seeds']} are not {K1_SEED0} + k for k < {K1_DRAWS}")
    if record["replay"]["content_sha256"] != replay_sha256:
        raise F1sError(f"{path}: made from replay content {record['replay']['content_sha256'][:12]}, the evidence has "
                       f"{str(replay_sha256)[:12]}")
    file = Path(path).parent / entry["file"]
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if digest != entry["file_sha256"]:
        raise F1sError(f"{file}: sha256 {digest[:12]} differs from the record's {entry['file_sha256'][:12]}")
    with np.load(file, allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["x"])
    if x.dtype != np.uint8 or x.shape[:2] != (K1_DRAWS, n_old):
        raise F1sError(f"{file}: rows {x.dtype}{x.shape}, expected uint8 [{K1_DRAWS}, {n_old}, ...]")
    return x, {"record": str(path), "file": str(file), "file_sha256": digest, "seeds": entry["seeds"],
               "official_sha256": record["official"]["sha256"], "severity": record["severity"]}


def _part_worker(payload: tuple[Any, ...]) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """One trajectory in its own process: (arch, params, x, y, mode, lr, steps, seed, threads)."""
    arch, params, x_np, y_np, mode, lr, steps, seed, threads = payload
    torch.set_num_threads(int(threads))
    return train_checkpoints(arch, params, images_tensor(x_np), torch.from_numpy(y_np), mode=mode, lr=lr,
                             checkpoints=steps, seed=seed)


# --------------------------------------------------------------------------------------------- generator


def f1s_generator(instance: Instance, method: str, run_seed: int, s5_rows: Path | None = None) -> Generator:
    """Candidate generator of one F1s arm on one E11 instance (module docstring)."""
    configs = f1s_configs(method)
    recipe = RECIPES[method]
    if (recipe["prior"] in ("oracle", "rule")) != (s5_rows is not None):
        raise F1sError(f"{method}: --s5-rows is required by f1s_s5_oracle_prior and k1_ft_rule (the rows record) "
                       "and used by no other arm")

    def generate_k1(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        """K1's control: fit rows are the rule draws of x_old with y_old, then x_old with y_old. No x_new row is read
        by the fit, the parity check or the base loss."""
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        if arch.family != VITLITE:
            raise F1sError(f"{method} runs on {VITLITE} students, not {arch.family}")
        k = evidence.n_deployment
        x_old, y_old = np.ascontiguousarray(evidence.x[k:]), evidence.y[k:].astype(np.int64)
        if len(x_old) == 0:
            raise F1sError(f"{method} fits the replay rows; the evidence has none")
        draws, rows_info = load_k1_rows(Path(s5_rows), str(instance.private.get("condition")),
                                        evidence.replay_sha256, len(x_old))
        draws = draws.reshape(K1_DRAWS, *x_old.shape)
        x_np = np.ascontiguousarray(np.concatenate([draws.reshape(-1, *x_old.shape[1:]), x_old]))
        y_np = np.concatenate([np.tile(y_old, K1_DRAWS), y_old])
        parity = assert_parity(arch, params, instance.forward, x_old)
        info: dict[str, Any] = {
            "architecture": arch.describe(), "parity_max_abs_logit_diff": parity, "prereg": PREREG_F1S,
            "recipe": dict(recipe), "rule_rows": rows_info, "x_new_rows_used": 0,
            "fit_rows": (f"{len(y_np)} rows: {K1_DRAWS} rule draws of each of the {len(x_old)} replay rows (seeds "
                         f"{K1_SEED0} + k, labels y_old), then the {len(x_old)} replay rows; no condition row"),
            "rows_sha256": hashlib.sha256(x_np.tobytes() + y_np.tobytes()).hexdigest()}
        with torch.no_grad():
            info["base_fit_loss"] = float(F.cross_entropy(logits(arch, to_tensors(params), images_tensor(x_np)),
                                                          torch.from_numpy(y_np)))
        info["trainable"] = {"full": trainable_tensors(arch, params), "lora": lora_targets(arch, params)}
        steps = GRIDS[method]["steps"]
        threads = max(1, torch.get_num_threads())
        payloads = [(arch, dict(params), x_np, y_np, mode, lr, tuple(steps), run_seed, threads)
                    for mode, lr in GRIDS[method]["parts"]]
        if K1_WORKERS > 1 and len(payloads) > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=min(K1_WORKERS, len(payloads)),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                results = list(pool.map(_part_worker, payloads))
        else:
            results = [_part_worker(payload) for payload in payloads]
        candidates: list[CandidatePatch] = []
        for (mode, lr), runs in zip(GRIDS[method]["parts"], results, strict=True):
            trainable = info["trainable"][mode]
            for step, child, loss in runs:
                _check_frozen_entries(params, child, trainable, None)
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config={"mode": mode, "lr": lr, "steps": step},
                    hypothesis=f"{method}: {mode} Adam lr={lr:g} for {step} steps on {info['fit_rows']}; "
                               f"{F1S_SPECS[method].label}",
                    predicted_effect="fewer errors on held-out condition images if the stated rule's synthetic copies "
                                     "carry the condition; retention is fitted on the clean replay rows",
                    notes={"objective_at_checkpoint": {"ce_fit_rows_eval": loss}}))
        if [c.config for c in candidates] != configs:
            raise RuntimeError(f"{method}: candidates {[c.config for c in candidates]} != configurations {configs}")
        return candidates, info

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        if recipe["prior"] == "rule":
            return generate_k1(evidence)
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        if arch.family != VITLITE:
            raise F1sError(f"F1s runs on {VITLITE} students, not {arch.family}")
        if evidence.n_replay == 0:
            raise F1sError("F1s fits the condition and replay rows; the evidence has no replay rows")
        k = evidence.n_deployment
        parity = assert_parity(arch, params, instance.forward, evidence.x)
        info: dict[str, Any] = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity,
                                "prereg": PREREG_F1S, "recipe": dict(recipe),
                                "fit_rows": f"{len(evidence.y)} rows ({evidence.composition()})"}
        x_np, y_np = evidence.x, evidence.y.astype(np.int64)
        if recipe["prior"] is not None:
            x_new, x_old, y_old = evidence.x[:k], evidence.x[k:], evidence.y[k:].astype(np.int64)
            if recipe["prior"] == "estimated":
                synth, info["estimate"] = estimate_corruption(x_new, x_old, arch.input_shape, run_seed)
            else:
                synth, info["oracle_rows"] = load_s5_rows(Path(s5_rows), str(instance.private.get("condition")),
                                                          evidence.replay_sha256, len(x_old))
            if synth.shape != x_old.shape or synth.dtype != np.uint8:
                raise F1sError(f"synthetic rows {synth.dtype}{synth.shape} differ from the replay rows {x_old.shape}")
            x_np = np.ascontiguousarray(np.concatenate([x_new, synth, x_old]))
            y_np = np.concatenate([evidence.y[:k].astype(np.int64), y_old, y_old])
            info["fit_rows"] = (f"{len(y_np)} rows: the {k} condition rows, {len(synth)} synthetic copies of the replay "
                                f"rows ({recipe['prior']} corruption, labels y_old), the {len(x_old)} replay rows")
            info["synthetic_sha256"] = hashlib.sha256(synth.tobytes()).hexdigest()
        x = images_tensor(x_np)
        y = torch.from_numpy(y_np)
        with torch.no_grad():
            info["base_fit_loss"] = float(F.cross_entropy(logits(arch, to_tensors(params), x), y))
        info["trainable"] = {"full": trainable_tensors(arch, params), "lora": lora_targets(arch, params)}
        candidates: list[CandidatePatch] = []
        steps = GRIDS[method]["steps"]
        for mode, lr in GRIDS[method]["parts"]:
            runs = train_checkpoints(arch, params, x, y, mode=mode, lr=lr, checkpoints=steps, seed=run_seed,
                                     warmup=recipe["warmup"], minibatch_split=k if recipe["minibatch"] else None,
                                     augment=recipe["augment"])
            trainable = info["trainable"][mode]
            for step, child, loss in runs:
                _check_frozen_entries(params, child, trainable, None)
                config = ({"mode": mode} if "mode" in configs[0] else {}) | {"lr": lr, "steps": step}
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config=config,
                    hypothesis=(f"{method}: {mode} Adam lr={lr:g} for {step} steps on {info['fit_rows']}; "
                                f"{F1S_SPECS[method].label}"),
                    predicted_effect=("fewer errors on held-out condition images if the fit generalizes; retention is "
                                      f"fitted on the {evidence.n_replay} clean replay rows"),
                    notes={"objective_at_checkpoint": {"ce_fit_rows_eval": loss}}))
        if [c.config for c in candidates] != configs:
            raise RuntimeError(f"{method}: candidates {[c.config for c in candidates]} != configurations {configs}")
        return candidates, info

    return generate


def f1s_config(method: str, run_seed: int, s5_rows: Path | None) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {
        "prereg": PREREG_F1S, "candidate_order": f1s_configs(method), "recipe": dict(RECIPES[method]),
        "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
        "warmup_steps": RECIPES[method]["warmup"], "minibatch": ({"condition": HALF, "replay": HALF}
                                                                  if RECIPES[method]["minibatch"] else "full"),
        "augment": (f"crop (zero pad {CROP_PAD}) and horizontal flip, generator seed run_seed + 1"
                    if RECIPES[method]["augment"] else None),
        "lora": {"rank": LORA_RANK, "alpha_over_rank": LORA_ALPHA_OVER_RANK}, "s5_rows": None if s5_rows is None
        else str(s5_rows), "dtype": "float32", "seed": run_seed, "norm_limit": None,
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": "cpu",
    }


def run_f1s(instance: Instance, method: str, run_dir: Path, *, run_seed: int = 0, s5_rows: Path | None = None
            ) -> dict[str, Any]:
    """Run one F1s arm on one E11 dev instance under F1's joint oracle and executor (without the size limits)."""
    if method not in F1S_SPECS:
        raise F1sError(f"unknown F1s method {method!r}; expected one of {F1S_METHODS}")
    if instance.private.get("dataset") != F1_DATASET:
        raise F1sError(f"F1s runs on {F1_DATASET} instances, got dataset {instance.private.get('dataset')!r}")
    return execute_run(
        instance=instance, spec=F1S_SPECS[method], run_dir=run_dir, run_seed=run_seed,
        config=f1s_config(method, run_seed, s5_rows), generate=f1s_generator(instance, method, run_seed, s5_rows),
        allow_main=False, joint=True, replay_in_fit=True, e10=True,
    )


__all__ = ["F1S_LABEL", "F1S_METHODS", "F1S_SPECS", "F1sError", "crop_flip", "estimate_corruption", "f1s_config",
           "f1s_configs", "f1s_generator", "highpass_energy", "load_s5_rows", "pixel_spread", "run_f1s", "synthesize",
           "train_checkpoints"]
