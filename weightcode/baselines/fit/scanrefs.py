"""PREREG E13 references on scanskill-v1 instances (SCAN; R22-e13 decisions of 2026-11-06, main d1c3cec).

Every reference runs through ``execute_run(scan=True)``: run seed 0, 8 candidates, each checked once by the scan joint
oracle (S_new on new V_check, S_old on the 1000 old V_check rows), under ``scan_program_limits`` (at most 20% of the
entries changed, the rel-Frob limit lifted) when the method is held to the limits, and with the edit limits ignored
otherwise. The student is rebuilt in torch (``weightcode.baselines.fit.seqgrad.rebuild``: weightcode.models.transformer with 4 heads
and pad 0, which SCAN's forward shares with seqops) and checked against forward.py on the fit rows before any step.
Full-batch Adam (betas 0.9/0.999, eps 1e-8, no weight decay) from the base weights; one trajectory per learning rate
(and weight), checkpointed at the listed steps, so a checkpoint equals a separate run of that length. The loss is the
mean token cross-entropy over the OUT positions of the fit rows.

The experience rows (``load_scan_evidence``) are the rows a LEARN session gets: on addprim_jump the "jump" row, then
the 64 replay rows of old commands; on around_right the 64 replay rows.

* E5's seqgrad set, on the experience rows, lr {1e-3, 1e-2} x steps {10, 30, 100, 300}:
  ``seq_grad_full`` (every tensor, edit limits ignored), ``seq_grad_emb_head`` (tok_emb and the head), ``seq_grad_new_row``
  (the primitive's token-embedding row; addprim_jump only, PREREG E13 decision 1), ``seq_grad_sparse19_sal`` and
  ``seq_grad_sparse19_traj`` (every tensor, a fixed top 19% of each tensor's entries by |d loss / d w| at the base, or by
  |w - w_base| after Adam lr 1e-3 for 300 steps; E5's diagnostic masks). The last four are held to the limits; with the
  rel-Frob limit lifted no checkpoint is rescaled.
* F1's eight configurations on tfm (decision 2), on the experience rows, edit limits ignored:
  ``ft_ln`` (every LayerNorm weight and bias), ``lora_r4`` and ``lora_r16`` (W = W0 + B A, alpha = r, on every block's
  q, k, v, o, fc1 and fc2 weights; A as nn.Linear's initialisation from a generator seeded by the run seed, B = 0;
  the merged weights are the candidate), ``ft_full_noreplay`` (every tensor, the primitive row alone; addprim_jump
  only), ``ft_full_lwf`` (every tensor, plus w KL(p_base || p_theta) over the OUT positions of the replay rows),
  ``ft_full_l2sp`` (every tensor, plus (w / 2) ||theta - theta_base||^2), ``f1_full`` (every tensor) and ``f1_last``
  (the head). F1's grids: lr {1e-5, 1e-4} x steps {10, 30, 100, 300} (LoRA lr {1e-4, 1e-3}); LwF w {1, 10} and
  L2-SP w {1e-2, 1e-1} x lr {1e-5, 1e-4} x steps {30, 300}.
* Rule-generated fine-tuning (decision 3), E8's two fit sets on the seqgrad grid: ``rulegen_*`` fits 2000 commands
  drawn uniformly without replacement from the grammar's commands that hold the primitive, labelled by the
  interpretation function, plus the experience rows; ``rulegen_mixed_*`` adds 2000 commands without the primitive.
  Recipes ``grad_full`` (unconstrained), ``sparse19_sal`` and ``sparse19_traj`` (inside the cap). The generator reads no
  file of the new part (``rulegen_fit_set``: the grammar and fixed streams); its commands may coincide with held-out
  ones. Each run records the command ids of its fit set, so its readouts report how many new V_select commands the fit
  set holds and the exact match on those it does not hold (coordinator, 2026-11-06).

Device (R22-e13, 2026-11-06): the fits run on ``device``, the CPU by default or one CUDA GPU (``configure_device``:
deterministic algorithms, deterministic cuBLAS workspace, IEEE float32 matmuls, so GPU logits are true float32). On a
GPU the rebuild is checked against forward.py on the device too, at the same bound. The parameters leave the device as
float32 numpy arrays, and every check and readout runs forward.py on the CPU, as on a CPU run. The rule-generated fits
(2065 or 4065 rows) cost about 5 to 15 s per full-batch step on 1 to 4 CPU threads, and torch's CPU Adam crashed on
NSCC at 8 or more threads (E14), hence the GPU for R22-e13.

Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance>/<method>/<label>/0/``, label ``f1`` for F1's eight (decision 2) and
``e13`` for the rest. ``walk_diagnostic`` is the fixed diagnostic (not a
reference) of PREREG E13.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from weightcode.baselines.instance_io import Instance
from weightcode.data import scan as S
from weightcode.data import seqops
from weightcode.data.scan_build import FORWARD_N_HEADS
from weightcode.evalkit.metrics import IGNORE
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.scanskill import (
    SCAN_RULE,
    construction_masks,
    load_recorded,
    part_summary,
    scan_program_limits,
)
from weightcode.models import transformer as T

from . import seqgrad as SG
from .ftcompare import (
    FT_LRS,
    L2SP_WEIGHTS,
    LORA_ALPHA_OVER_RANK,
    LORA_LRS,
    LORA_RANKS,
    LWF_TEMPERATURE,
    F1_LABEL,
    LWF_WEIGHTS,
    PREREG_F1,
    REG_STEPS,
    lora_init,
)
from .grad import ADAM_BETAS, ADAM_EPS, LEARNING_RATES, STEP_CHECKPOINTS
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, baseline_run_dir, execute_run

if SG.N_HEADS != FORWARD_N_HEADS or seqops.PAD != S.PAD or seqops.IGNORE != S.IGNORE:  # pragma: no cover
    raise RuntimeError("seqgrad's rebuild assumes 4 heads, pad 0 and IGNORE -1, which SCAN's forward must share")

PREREG_E13 = "PREREG E13 (main 9df26e9), references as built and decided under R22-e13 (main d1c3cec)"
E13_LABEL = "e13"
"""Run-directory label of the E13 references other than F1's: ``baselines/<instance_id>/<method>/e13/<seed>/``."""
MASK_FRACTION = 0.19
TRAJ_LR = 1e-3
TRAJ_STEPS = 300
EMBEDDING = "tok_emb.weight"
HEAD_TENSORS: tuple[str, str] = ("head.weight", "head.bias")
LORA_MODULES: tuple[str, ...] = ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.fc1", "mlp.fc2")
N_RULEGEN = 2000
N_RULEGEN_OLD = 2000
RULEGEN_STREAMS: Mapping[str, Mapping[str, tuple[int, int, int]]] = {
    "addprim_jump": {"prim": (20261106, 1320, 1), "old": (20261106, 1320, 2)},
    "around_right": {"prim": (20261106, 1321, 1), "old": (20261106, 1321, 2)},
}
"""Seed lists of the rule-generated draws (fixed by this module before any draw)."""
RULEGEN_FITS: Mapping[str, str] = {"rulegen": "full", "rulegen_mixed": "mixed"}
RECIPES: tuple[str, ...] = ("grad_full", "sparse19_sal", "sparse19_traj")

SEQ_METHODS: tuple[str, ...] = ("seq_grad_full", "seq_grad_emb_head", "seq_grad_new_row", "seq_grad_sparse19_sal",
                                "seq_grad_sparse19_traj")
F1_METHODS: tuple[str, ...] = ("ft_ln", "lora_r4", "lora_r16", "ft_full_noreplay", "ft_full_lwf", "ft_full_l2sp",
                               "f1_full", "f1_last")
RULEGEN_METHODS: tuple[str, ...] = tuple(f"{prefix}_{recipe}" for prefix in RULEGEN_FITS for recipe in RECIPES)
METHODS: tuple[str, ...] = (*SEQ_METHODS, *F1_METHODS, *RULEGEN_METHODS)
PRIMITIVE_ROW_ONLY: frozenset[str] = frozenset({"seq_grad_new_row", "ft_full_noreplay"})
"""Methods defined only where the condition has a primitive row (addprim_jump; PREREG E13 decisions 1 and 2)."""
LIMITED: frozenset[str] = frozenset({"seq_grad_emb_head", "seq_grad_new_row", "seq_grad_sparse19_sal",
                                     "seq_grad_sparse19_traj", "rulegen_sparse19_sal", "rulegen_sparse19_traj",
                                     "rulegen_mixed_sparse19_sal", "rulegen_mixed_sparse19_traj"})
"""Methods held to ``scan_program_limits``; every other one runs with the edit limits ignored."""


class ScanRefError(ValueError):
    """An E13 reference was refused: unknown method, a method not defined on the condition, or a bad fit set."""


DEVICES: tuple[str, ...] = ("cpu", "cuda")
CUBLAS_WORKSPACE = ":4096:8"
"""The deterministic cuBLAS workspace a GPU run needs (set before torch initialises CUDA)."""


def configure_device(device: str) -> dict[str, Any]:
    """Prepare torch for fits on ``device`` and return the settings for run.json.

    The CPU changes nothing. CUDA needs ``CUBLAS_WORKSPACE_CONFIG`` set before CUDA initialises; it switches on
    deterministic algorithms and cuDNN, and IEEE float32 matmuls (no TF32), as ``weightcode.models.train`` does for students.
    """
    if device not in DEVICES:
        raise ScanRefError(f"device {device!r} is not one of {DEVICES}")
    if device == "cpu":
        return device_record(device)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE:
        raise ScanRefError(f"set CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE} before torch initialises CUDA")
    if not torch.cuda.is_available():
        raise ScanRefError("device cuda asked for, but torch sees no CUDA device")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    return device_record(device)


def device_record(device: str) -> dict[str, Any]:
    """The fit device's settings as run.json records them; a CUDA device must have been set up by
    ``configure_device``."""
    if device == "cpu":
        return {"device": "cpu"}
    record = {
        "device": device,
        "name": torch.cuda.get_device_name(0),
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if not (record["use_deterministic_algorithms"] and record["cudnn_deterministic"]
            and record["cuda_matmul_fp32_precision"] == "ieee" and record["cublas_workspace_config"] == CUBLAS_WORKSPACE):
        raise ScanRefError(f"device {device} is not set up for deterministic float32 fits: {record}")
    return record


def device_parity(model: T.SeqTransformer, params: Mapping[str, np.ndarray], forward: Any, x: np.ndarray,
                  device: str) -> dict[str, float]:
    """The rebuild's parity with forward.py on ``x``, on the CPU and, for a GPU device, on the device too (one
    forward.py pass; the same bound, ``seqgrad.PARITY_RTOL`` x max(1, max |logit|)); a larger difference raises."""
    if device == "cpu":
        diff, bound = SG.seq_parity(model, params, forward, x)
        return {"parity_max_abs_logit_diff": diff, "parity_bound": bound}
    theirs = np.asarray(forward(dict(params), x), dtype=np.float64)
    bound = float(SG.float32_tolerance(theirs))
    out: dict[str, float] = {"parity_bound": bound}
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64))
    for name, on in (("parity_max_abs_logit_diff", "cpu"), ("device_parity_max_abs_logit_diff", device)):
        with torch.no_grad():
            mine = model.to(on)(x_t.to(on)).to("cpu").numpy().astype(np.float64)
        if mine.shape != theirs.shape:
            raise RuntimeError(f"torch logits {mine.shape} != forward.py logits {theirs.shape}")
        diff = float(np.max(np.abs(mine - theirs)))
        if not diff <= bound:
            raise RuntimeError(f"the torch rebuild on {on} differs from forward.py by {diff:.3g} > {bound:.3g}")
        out[name] = diff
    return out


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to("cpu").numpy().copy()


def condition_methods(condition: str) -> tuple[str, ...]:
    """The E13 references of a condition, in run order."""
    if condition not in S.CONDITIONS:
        raise ScanRefError(f"unknown SCAN condition {condition!r}")
    prim = S.PRIMITIVE_ROW[condition] is not None
    return tuple(method for method in METHODS if prim or method not in PRIMITIVE_ROW_ONLY)


def method_spec(method: str) -> MethodSpec:
    """The run.json description of one E13 reference."""
    if method not in METHODS:
        raise ScanRefError(f"unknown E13 reference {method!r}; one of {METHODS}")
    limited = method in LIMITED
    tail = ("within the E13 program limits (20% of the entries, rel-Frob lifted)" if limited
            else "edit limits ignored") + f"; not training-free; {PREREG_E13}"
    words = {
        "seq_grad_full": "E5 gradient reference: Adam on all parameters, experience rows",
        "seq_grad_emb_head": "E5 gradient reference: Adam on the token embedding and the head, experience rows",
        "seq_grad_new_row": "E5 gradient reference: Adam on the primitive's token-embedding row, experience rows",
        "seq_grad_sparse19_sal": "E5 sparse reference: every tensor, top 19% of its entries by |gradient| at the base",
        "seq_grad_sparse19_traj": "E5 sparse reference: every tensor, top 19% of its entries by the Adam trajectory",
        "ft_ln": "F1 fine-tuning: Adam on the LayerNorm weights and biases only, experience rows",
        "lora_r4": "F1 fine-tuning: LoRA rank 4 (alpha 4) on q, k, v, o, fc1 and fc2, merged, experience rows",
        "lora_r16": "F1 fine-tuning: LoRA rank 16 (alpha 16) on q, k, v, o, fc1 and fc2, merged, experience rows",
        "ft_full_noreplay": "F1 fine-tuning: Adam on all parameters, the primitive row only (no replay)",
        "ft_full_lwf": "F1 fine-tuning: Adam on all parameters plus LwF on the replay rows, experience rows",
        "ft_full_l2sp": "F1 fine-tuning: Adam on all parameters plus L2-SP toward the base, experience rows",
        "f1_full": "F1 fine-tuning: Adam on all parameters, experience rows (F1's full, F1's grid)",
        "f1_last": "F1 fine-tuning: Adam on the head only, experience rows (F1's last layer, F1's grid)",
    }
    if method in RULEGEN_METHODS:
        prefix, recipe = rulegen_parts(method)
        fit = "commands with the primitive" if RULEGEN_FITS[prefix] == "full" else (
            "commands with and without the primitive")
        label = f"rule-generated fine-tuning ({recipe}): rows of {fit} written from the stated grammar, plus the experience rows"
    else:
        label = words[method]
    return MethodSpec(method=method, label=f"{label}; {tail}", uses_labels_for_fit=True, edit_limits_enforced=limited)


def run_label(method: str) -> str:
    """The run-directory label of a reference: ``f1`` for F1's eight (PREREG E13 decision 2), ``e13`` otherwise."""
    if method not in METHODS:
        raise ScanRefError(f"unknown E13 reference {method!r}")
    return F1_LABEL if method in F1_METHODS else E13_LABEL


def rulegen_parts(method: str) -> tuple[str, str]:
    """(fit prefix, recipe) of a rule-generated method."""
    for prefix in sorted(RULEGEN_FITS, key=len, reverse=True):
        if method.startswith(prefix + "_") and method[len(prefix) + 1:] in RECIPES:
            return prefix, method[len(prefix) + 1:]
    raise ScanRefError(f"{method!r} is not a rule-generated method")


# ---------------------------------------------------------------------------------------------------------------
# Rule-generated fit sets


def command_ids() -> dict[str, int]:
    """Command id = index in the sorted grammar (equal to the sorted commands of tasks.txt, checked at the store build)."""
    return {command: index for index, command in enumerate(S.grammar_commands())}


def _sha256_lines(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def construction_counts(commands: Sequence[str]) -> dict[str, int]:
    """Commands per construction of ``S.CONSTRUCTIONS`` (a command can count in several), and ``none``."""
    x, _ = S.encode(commands)
    return {name: int(mask.sum()) for name, mask in construction_masks(x).items()}


def rulegen_fit_set(condition: str, fit: str) -> dict[str, Any]:
    """The generated commands of one fit set (``full`` or ``mixed``), drawn from the grammar alone.

    ``full``: N_RULEGEN commands uniformly without replacement from the grammar's commands that hold the primitive
    (sorted), with ``default_rng(RULEGEN_STREAMS[condition]["prim"])``. ``mixed``: the same, then N_RULEGEN_OLD commands
    from those without it, with the ``old`` stream. Returns the commands, their rows (``S.encode``: labels by the
    interpretation function), their command ids and a record."""
    if fit not in ("full", "mixed"):
        raise ScanRefError(f"fit set {fit!r} is not full or mixed")
    grammar = S.grammar_commands()
    with_prim = [c for c in grammar if S.has_primitive(c, condition)]
    without = [c for c in grammar if not S.has_primitive(c, condition)]
    streams = RULEGEN_STREAMS[condition]
    prim_pick = np.random.default_rng(list(streams["prim"])).choice(len(with_prim), size=N_RULEGEN, replace=False)
    commands = [with_prim[int(i)] for i in prim_pick]
    record: dict[str, Any] = {
        "fit": fit, "condition": condition, "pool_with_primitive": len(with_prim), "pool_without": len(without),
        "n_with_primitive": N_RULEGEN, "streams": {"prim": list(streams["prim"])},
        "with_primitive_sha256": _sha256_lines(commands),
        "with_primitive_by_construction": construction_counts(commands),
    }
    if fit == "mixed":
        old_pick = np.random.default_rng(list(streams["old"])).choice(len(without), size=N_RULEGEN_OLD, replace=False)
        old = [without[int(i)] for i in old_pick]
        commands = commands + old
        record.update(n_without=N_RULEGEN_OLD, without_sha256=_sha256_lines(old))
        record["streams"]["old"] = list(streams["old"])
    x, y = S.encode(commands)
    ids = command_ids()
    record.update(n_generated=len(commands), commands_sha256=_sha256_lines(commands))
    return {"commands": commands, "x": x, "y": y, "ids": np.array([ids[c] for c in commands], dtype=np.int64),
            "record": record}


def check_rows_against_official(raw_tasks: Path, commands: Sequence[str], y: np.ndarray) -> dict[str, Any]:
    """Every generated row's targets against the official tasks.txt answer of its command (an independent source of
    the labels); raises on any mismatch. Local only: raw/ never goes to NSCC."""
    official = dict(S.parse_tasks(raw_tasks.read_text(encoding="utf-8")))
    expected = [list(official[c]) for c in commands]
    found = S.actions_of(y)
    mismatches = sum(1 for a, b in zip(expected, found) if a != b)
    if mismatches:
        raise ScanRefError(f"{mismatches} generated rows differ from tasks.txt")
    return {"n_rows": len(commands), "mismatches": 0, "tasks_txt_sha256": S.file_sha256(raw_tasks)}


# ---------------------------------------------------------------------------------------------------------------
# Training pieces


def trainable_for(method: str, params: Mapping[str, np.ndarray]) -> list[str]:
    """The tensors a method trains (LoRA: the adapted weights)."""
    names = sorted(params)
    if method in ("seq_grad_full", "ft_full_noreplay", "ft_full_lwf", "ft_full_l2sp", "f1_full") or (
            method.startswith("seq_grad_sparse19") or method in RULEGEN_METHODS):
        return names
    if method == "seq_grad_emb_head":
        return sorted([EMBEDDING, *HEAD_TENSORS])
    if method == "seq_grad_new_row":
        return [EMBEDDING]
    if method == "ft_ln":
        found = sorted(n for n in names if n.split(".")[-2].startswith("ln"))
        if not found:
            raise ScanRefError("the student has no LayerNorm tensor")
        return found
    if method == "f1_last":
        return sorted(HEAD_TENSORS)
    if method in LORA_RANKS:
        return lora_targets(params)
    raise ScanRefError(f"unknown E13 reference {method!r}")


def lora_targets(params: Mapping[str, np.ndarray]) -> list[str]:
    """The weights LoRA adapts, in forward order: every block's q, k, v, o, fc1 and fc2 weights."""
    blocks = sorted({int(k.split(".")[0][len(T.BLOCK_PREFIX):]) for k in params
                     if k.startswith(T.BLOCK_PREFIX) and k.split(".")[0][len(T.BLOCK_PREFIX):].isdigit()})
    names = [f"{T.BLOCK_PREFIX}{i}.{module}.weight" for i in blocks for module in LORA_MODULES]
    for name in names:
        if name not in params or np.ndim(params[name]) != 2:
            raise ScanRefError(f"LoRA target {name} is not a 2-D tensor of the student")
    return names


def top_fraction(score: Mapping[str, np.ndarray], names: Sequence[str]) -> dict[str, np.ndarray]:
    """Per tensor, the top MASK_FRACTION of its entries by |score| (stable argsort, ties by flat index)."""
    masks = {}
    for name in names:
        flat = np.abs(np.asarray(score[name])).ravel()
        k = int(np.floor(MASK_FRACTION * flat.size))
        if k < 1:
            raise ScanRefError(f"{name}: {flat.size} entries give an empty {MASK_FRACTION:g} mask")
        order = np.argsort(-flat, kind="stable")
        mask = np.zeros(flat.size, dtype=bool)
        mask[order[:k]] = True
        masks[name] = mask.reshape(np.shape(score[name]))
    return masks


def _rows(x: np.ndarray, y: np.ndarray, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(device),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)).to(device))


def _tensors(params: Mapping[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.array(v, dtype=np.float32, copy=True)).to(device) for k, v in params.items()}


def _ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE)


def _adam(leaves: list[torch.Tensor], lr: float) -> torch.optim.Adam:
    return torch.optim.Adam(leaves, lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0, foreach=False)


def lora_checkpoints(params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor, *, targets: Sequence[str],
                     rank: int, lr: float, checkpoints: tuple[int, ...], seed: int,
                     device: str = "cpu") -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Full-batch Adam on LoRA's A and B (``ftcompare.lora_init``, drawn on the CPU and then moved to ``device``);
    (steps, merged params, loss) at each checkpoint."""
    model = SG.rebuild(params).to(device)
    x, y = x.to(device), y.to(device)
    base = _tensors(params, device)
    a, b = lora_init(params, targets, rank, seed)
    if device != "cpu":
        a = {k: v.detach().to(device).requires_grad_(True) for k, v in a.items()}
        b = {k: v.detach().to(device).requires_grad_(True) for k, v in b.items()}
    optimizer = _adam([t for name in targets for t in (a[name], b[name])], lr)
    scale = LORA_ALPHA_OVER_RANK

    def merged() -> dict[str, torch.Tensor]:
        tensors = dict(base)
        for name in targets:
            tensors[name] = base[name] + scale * (b[name] @ a[name])
        return tensors

    wanted, snapshots = set(checkpoints), []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        _ce(functional_call(model, merged(), (x,)), y).backward()
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                tensors = merged()
                loss = float(_ce(functional_call(model, tensors, (x,)), y))
            snapshots.append((step, {k: _numpy(v) for k, v in tensors.items()}, loss))
    return snapshots


Penalty = Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor]


def regularized_checkpoints(params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor, *, lr: float,
                            checkpoints: tuple[int, ...], seed: int, penalty: Penalty,
                            device: str = "cpu") -> list[tuple[int, dict[str, np.ndarray], dict[str, float]]]:
    """Full-batch Adam on every tensor: mean token cross-entropy plus ``penalty(tensors, logits)``; at each checkpoint
    the params and the objective's parts after the step. The penalty's own tensors must be on ``device``."""
    torch.manual_seed(seed)
    model = SG.rebuild(params).to(device)
    x, y = x.to(device), y.to(device)
    tensors = {k: v.requires_grad_(True) for k, v in _tensors(params, device).items()}
    optimizer = _adam([tensors[k] for k in sorted(tensors)], lr)

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = functional_call(model, tensors, (x,))
        ce = _ce(out, y)
        reg = penalty(tensors, out)
        return ce + reg, ce, reg

    wanted, snapshots = set(checkpoints), []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        objective()[0].backward()
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                total, ce, reg = objective()
            snapshots.append((step, {k: _numpy(v) for k, v in tensors.items()},
                              {"ce": float(ce), "penalty": float(reg), "total": float(total)}))
    return snapshots


def lwf_penalty(base_logits: torch.Tensor, replay_out: torch.Tensor, k: int, weight: float) -> Penalty:
    """w KL(p_base || p_theta) at temperature 1, averaged over the OUT positions of the replay rows (the rows after the
    first ``k``); ``replay_out`` marks those positions [n_replay, L]."""
    target = F.log_softmax(base_logits[replay_out] / LWF_TEMPERATURE, dim=-1)

    def penalty(_: Mapping[str, torch.Tensor], out: torch.Tensor) -> torch.Tensor:
        student = F.log_softmax(out[k:][replay_out] / LWF_TEMPERATURE, dim=-1)
        return weight * F.kl_div(student, target, log_target=True, reduction="batchmean")

    return penalty


def l2sp_penalty(params: Mapping[str, np.ndarray], weight: float, device: str = "cpu") -> Penalty:
    """(w / 2) sum over every entry of (theta - theta_base)^2, the anchor on ``device``."""
    anchor = _tensors(params, device)

    def penalty(tensors: Mapping[str, torch.Tensor], _: torch.Tensor) -> torch.Tensor:
        return 0.5 * weight * sum(((tensors[k] - anchor[k]) ** 2).sum() for k in sorted(anchor))

    return penalty


def configs(method: str) -> list[dict[str, Any]]:
    """The 8 configurations of ``method`` in check order: weight (if any) outer, then lr, then steps."""
    if method in SEQ_METHODS or method in RULEGEN_METHODS:
        return [{"lr": lr, "steps": steps} for lr in LEARNING_RATES for steps in STEP_CHECKPOINTS]
    if method in ("ft_ln", "ft_full_noreplay", "f1_full", "f1_last"):
        return [{"lr": lr, "steps": steps} for lr in FT_LRS for steps in STEP_CHECKPOINTS]
    if method in LORA_RANKS:
        return [{"lr": lr, "steps": steps} for lr in LORA_LRS for steps in STEP_CHECKPOINTS]
    if method == "ft_full_lwf":
        return [{"weight": w, "lr": lr, "steps": s} for w in LWF_WEIGHTS for lr in FT_LRS for s in REG_STEPS]
    if method == "ft_full_l2sp":
        return [{"weight": w, "lr": lr, "steps": s} for w in L2SP_WEIGHTS for lr in FT_LRS for s in REG_STEPS]
    raise ScanRefError(f"unknown E13 reference {method!r}")


def _check_frozen(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], trainable: Sequence[str],
                  masks: Mapping[str, np.ndarray] | None) -> None:
    for key, parent in base.items():
        if key not in trainable:
            if not np.array_equal(child[key], parent):
                raise RuntimeError(f"{key} is not trainable but changed during training")
        elif masks is not None and key in masks and not np.array_equal(child[key][~masks[key]], parent[~masks[key]]):
            raise RuntimeError(f"{key} changed outside its mask")


# ---------------------------------------------------------------------------------------------------------------
# Generators


def fit_rows(method: str, instance: Instance, evidence: Evidence,
             fit_set: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The rows a method fits and their record: the experience rows; the primitive row alone (ft_full_noreplay); or a
    rule-generated fit set followed by the experience rows."""
    k = evidence.n_deployment
    if method == "ft_full_noreplay":
        if k == 0:
            raise ScanRefError("ft_full_noreplay needs a primitive row; the condition has none")
        return evidence.x[:k], evidence.y[:k], {"rows": f"the {k} primitive row(s) of the experience, no replay"}
    if method in RULEGEN_METHODS:
        if fit_set is None:
            raise ScanRefError(f"{method} needs its rule-generated fit set")
        prefix, _ = rulegen_parts(method)
        if fit_set["record"]["fit"] != RULEGEN_FITS[prefix] or fit_set["record"]["condition"] != instance.private[
                "condition"]:
            raise ScanRefError(f"{method}: the fit set is {fit_set['record']['fit']} of "
                               f"{fit_set['record']['condition']}")
        x = np.concatenate([fit_set["x"], evidence.x])
        y = np.concatenate([fit_set["y"], evidence.y])
        record = {**fit_set["record"], "n_experience": int(len(evidence.y)), "n_fit": int(len(y)),
                  "command_ids": [int(i) for i in fit_set["ids"]],
                  "rows": f"{len(fit_set['y'])} rule-generated rows, then the {len(evidence.y)} experience rows "
                  f"({evidence.composition()})"}
        return x, y, record
    return evidence.x, evidence.y, {"rows": f"the {len(evidence.y)} experience rows ({evidence.composition()})"}


def scan_generator(instance: Instance, method: str, run_seed: int,
                   fit_set: Mapping[str, Any] | None = None, device: str = "cpu") -> Generator:
    """Candidate generator of one E13 reference on one scanskill-v1 instance (module docstring), fitting on
    ``device``."""
    condition = str(instance.private["condition"])
    if method not in condition_methods(condition):
        raise ScanRefError(f"{method} is not an E13 reference of {condition}")
    params = instance.params
    limits = scan_program_limits(instance.task["limits"], len(params))
    if limits.max_rel_frob is not None:  # pragma: no cover - scan_program_limits lifts it
        raise RuntimeError("the E13 limits must lift the rel-Frob limit")

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        x_fit, y_fit, rows = fit_rows(method, instance, evidence, fit_set)
        model = SG.rebuild(params)
        parity = device_parity(model, params, instance.forward, x_fit, device)
        model = model.to(device)
        x_t, y_t = _rows(x_fit, y_fit, device)
        trainable = trainable_for(method, params)
        masks: dict[str, np.ndarray] | None = None
        info: dict[str, Any] = {"model_config": model.config.as_dict(), **parity, "parity_rows": "the fit rows",
                                "fit_rows": rows, "trainable": trainable, "device": device}
        if method == "seq_grad_new_row":
            token = str(instance.private["primitive_row"])
            masks = {EMBEDDING: np.zeros(params[EMBEDDING].shape, dtype=bool)}
            masks[EMBEDDING][S.TOKEN_ID[token]] = True
            info["mask_rule"] = f"{EMBEDDING} row {S.TOKEN_ID[token]} ({token})"
        sparse = method.endswith("sparse19_sal") or method.endswith("sparse19_traj")
        if sparse:
            if method.endswith("sal"):
                for tensor in model.parameters():
                    tensor.requires_grad_(True)
                SG.token_loss(model, x_t, y_t).backward()
                score = {name: _numpy(p.grad) for name, p in model.named_parameters()}
            else:
                (_, traj, _), = SG.adam_checkpoints(params, x_t, y_t, trainable=trainable, lr=TRAJ_LR,
                                                    checkpoints=(TRAJ_STEPS,), seed=run_seed, device=device)
                score = {name: traj[name].astype(np.float64) - params[name].astype(np.float64) for name in trainable}
            masks = top_fraction(score, trainable)
            info.update(mask_rule=method, mask_fraction_per_tensor=MASK_FRACTION, mask_rows="the fit rows")
        n_params = int(sum(v.size for v in params.values()))
        n_trainable = int(sum(params[k].size if masks is None or k not in masks else masks[k].sum() for k in trainable))
        info.update(n_trainable_entries=n_trainable, n_params=n_params, trainable_frac=n_trainable / n_params)
        with torch.no_grad():
            info["base_fit_loss"] = float(SG.token_loss(model, x_t, y_t))
        grid = configs(method)
        snapshots: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]] = []
        if method in LORA_RANKS:
            for lr in LORA_LRS:
                for steps, child, loss in lora_checkpoints(params, x_t, y_t, targets=trainable,
                                                           rank=LORA_RANKS[method], lr=lr,
                                                           checkpoints=STEP_CHECKPOINTS, seed=run_seed,
                                                           device=device):
                    snapshots.append(({"lr": lr, "steps": steps}, child, {"fit_loss_at_checkpoint": loss}))
            info.update(lora_rank=LORA_RANKS[method], lora_alpha=LORA_RANKS[method], lora_init_seed=run_seed)
        elif method in ("ft_full_lwf", "ft_full_l2sp"):
            weights = LWF_WEIGHTS if method == "ft_full_lwf" else L2SP_WEIGHTS
            k = evidence.n_deployment
            if method == "ft_full_lwf":
                with torch.no_grad():
                    base_logits = model(x_t[k:])
                replay_out = y_t[k:] != IGNORE
                info["lwf"] = {"rows": f"the {len(y_t) - k} replay rows' OUT positions", "temperature": LWF_TEMPERATURE}
            for w in weights:
                penalty = (lwf_penalty(base_logits, replay_out, k, w) if method == "ft_full_lwf"
                           else l2sp_penalty(params, w, device))
                for lr in FT_LRS:
                    for steps, child, parts in regularized_checkpoints(params, x_t, y_t, lr=lr, checkpoints=REG_STEPS,
                                                                       seed=run_seed, penalty=penalty, device=device):
                        snapshots.append(({"weight": w, "lr": lr, "steps": steps}, child, {"objective": parts}))
        else:
            lrs = LEARNING_RATES if (method in SEQ_METHODS or method in RULEGEN_METHODS) else FT_LRS
            for lr in lrs:
                for steps, child, loss in SG.adam_checkpoints(params, x_t, y_t, trainable=trainable, lr=lr,
                                                              checkpoints=STEP_CHECKPOINTS, seed=run_seed, masks=masks,
                                                              device=device):
                    snapshots.append(({"lr": lr, "steps": steps}, child, {"fit_loss_at_checkpoint": loss}))
        if [config for config, _, _ in snapshots] != grid:
            raise RuntimeError(f"{method}: the checkpoints {[c for c, _, _ in snapshots]} are not its grid {grid}")
        candidates: list[CandidatePatch] = []
        for config, child, notes in snapshots:
            if method not in LORA_RANKS:
                _check_frozen(params, child, trainable, masks)
            candidates.append(CandidatePatch(
                slot=len(candidates), child=child, config=config,
                hypothesis=(f"{method}: Adam {json.dumps(config, sort_keys=True)} full-batch on the mean token "
                            f"cross-entropy over {rows['rows']}; trained: {_scope(method, trainable)}"),
                predicted_effect=("the new commands answered right if the fit generalizes to the primitive inside the "
                                  "old compositions; retention is carried by the replay rows in the fit"),
                notes=notes))
        return candidates, info

    return generate


def _scope(method: str, trainable: Sequence[str]) -> str:
    if method in LORA_RANKS:
        return f"LoRA rank {LORA_RANKS[method]} on {len(trainable)} weights, merged"
    if method.endswith("sparse19_sal") or method.endswith("sparse19_traj"):
        return f"every tensor, top {MASK_FRACTION:g} of each tensor's entries by the {method} mask"
    if method == "seq_grad_new_row":
        return "the primitive's token-embedding row"
    return ", ".join(trainable) if len(trainable) <= 6 else f"{len(trainable)} tensors (all)"


def run_config(method: str, run_seed: int, instance: Instance, fit_record: Mapping[str, Any] | None,
               device: str = "cpu") -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {
        "prereg": PREREG_E13,
        "f1": PREREG_F1 if method in F1_METHODS else None,
        "candidate_order": configs(method),
        "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
        "loss": "mean token cross-entropy over the OUT positions of the fit rows" + (
            " plus the method's penalty" if method in ("ft_full_lwf", "ft_full_l2sp") else ""),
        "batch": "full",
        "dtype": "float32",
        "seed": run_seed,
        "limits_enforced": method in LIMITED,
        "mask_fraction_per_tensor": MASK_FRACTION if "sparse19" in method else None,
        "traj_mask": {"lr": TRAJ_LR, "steps": TRAJ_STEPS} if method.endswith("traj") else None,
        "fit_set": None if fit_record is None else dict(fit_record),
        "condition": instance.private["condition"],
        "rule": SCAN_RULE,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "device_settings": device_record(device),
    }


def run_reference(instance: Instance, method: str, *, run_seed: int = 0,
                  fit_set: Mapping[str, Any] | None = None, run_dir: Path | None = None,
                  device: str = "cpu") -> dict[str, Any]:
    """Run one E13 reference on one scanskill-v1 instance, fitting on ``device`` (``configure_device`` first for a
    GPU); returns its run record (also run.json)."""
    record = None if fit_set is None else fit_set["record"]
    if (method in RULEGEN_METHODS) != (fit_set is not None):
        raise ScanRefError(f"{method}: a fit set is given exactly to the rule-generated methods")
    target = run_dir if run_dir is not None else baseline_run_dir(instance.instance_id, method, run_seed, run_label(method))
    config = run_config(method, run_seed, instance, record, device)
    if fit_set is not None:
        config["fit_set"] = {**dict(record), "command_ids": [int(i) for i in fit_set["ids"]]}
    return execute_run(instance=instance, spec=method_spec(method), run_dir=target, run_seed=run_seed, config=config,
                       generate=scan_generator(instance, method, run_seed, fit_set, device),
                       require_valid_candidate=True,
                       scan=True)


# ---------------------------------------------------------------------------------------------------------------
# Readout diagnostics


def walk_diagnostic(instance: Instance, batch_size: int = 1024) -> dict[str, Any]:
    """PREREG E13's fixed diagnostic (addprim_jump only, not a reference): jump's token-embedding row set to walk's,
    read on new V_select with I_JUMP read as I_WALK (a position is right when the prediction equals the target with
    every I_JUMP replaced by I_WALK), and on old V_select against the base, plus the new rows without walk."""
    condition = str(instance.private["condition"])
    if S.PRIMITIVE_ROW[condition] != "jump":
        raise ScanRefError(f"the walk diagnostic is defined for addprim_jump only, not {condition}")
    params = instance.params
    child = {k: np.array(v, copy=True) for k, v in params.items()}
    child[EMBEDDING][S.TOKEN_ID["jump"]] = params[EMBEDDING][S.TOKEN_ID["walk"]]
    new, old = load_recorded(instance.private, "V_select"), load_recorded(instance.private, "old_select")
    y_new = np.asarray(new["y"]).copy()
    relabelled = np.where(y_new == S.CLASS_ID["I_JUMP"], S.CLASS_ID["I_WALK"], y_new)
    base_new = predict(instance.forward, params, new["x"], batch_size)
    child_new = predict(instance.forward, child, new["x"], batch_size)
    walk_free = np.array(["walk" not in c.split() for c in S.decode(new["x"])], dtype=bool)
    by_construction = {name: part_summary(relabelled[rows], base_new[rows], child_new[rows]) if rows.any() else {"n": 0}
                       for name, rows in construction_masks(np.asarray(new["x"])).items()}
    base_old = predict(instance.forward, params, old["x"], batch_size)
    child_old = predict(instance.forward, child, old["x"], batch_size)
    return {
        "diagnostic": "walk_row: tok_emb row of jump set to walk's; new V_select read with I_JUMP as I_WALK",
        "instance_id": instance.instance_id,
        "condition": condition,
        "child_changes": f"{EMBEDDING} row {S.TOKEN_ID['jump']} := row {S.TOKEN_ID['walk']}",
        "new_relabelled": part_summary(relabelled, base_new, child_new),
        "new_relabelled_walk_free": part_summary(relabelled[walk_free], base_new[walk_free], child_new[walk_free]),
        "new_relabelled_by_construction": by_construction,
        "new_unrelabelled": part_summary(y_new, base_new, child_new),
        "old": part_summary(np.asarray(old["y"]), base_old, child_old),
        "note": "base columns read the unedited student against the same (relabelled) targets",
    }


__all__ = [
    "E13_LABEL", "F1_METHODS", "LIMITED", "METHODS", "PRIMITIVE_ROW_ONLY", "RECIPES", "RULEGEN_FITS",
    "RULEGEN_METHODS", "RULEGEN_STREAMS", "SEQ_METHODS", "ScanRefError", "check_rows_against_official", "command_ids",
    "condition_methods", "configs", "lora_targets", "method_spec", "rulegen_fit_set", "rulegen_parts",
    "run_label", "run_reference", "scan_generator", "top_fraction", "trainable_for", "walk_diagnostic",
]
