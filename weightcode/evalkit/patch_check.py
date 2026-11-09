"""Counted patch-check oracle shared by the editor harness and every baseline (contract sections 4 and 6).

The oracle holds V_check inputs and labels in the calling process. Callers receive aggregate summaries only. The
score of a query is S = fixed - broken over all V_check images; a target pair, when given, adds secondary fields.
A query is consumed only after the child parameters pass validation, so an invalid patch reveals nothing and
costs nothing. Every consumed query is logged with the child's canonical hash.

``JointPatchCheckOracle`` is the learning experiments' variant: one budget, two image sets (a condition's V_check
and its clean companion), score S_joint = S_new + S_old.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import numpy as np

from weightcode.common.identity import count_changed, params_sha256
from weightcode.evalkit.metrics import Candidate, choose_final, summarize

ForwardFn = Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]


class BudgetExceeded(RuntimeError):
    """Raised when a caller asks for more patch-check queries than its budget allows."""


def load_forward(path: str | Path) -> tuple[ForwardFn, str]:
    """Import ``forward`` from a trial forward.py by file path and return it with the file's sha256.

    ``forward(params, x)`` takes an integer input array: uint8 pixels [N, C, H, W], int64 token ids [N, L], or
    int64 node ids [N] for a graph model whose forward.py reads its graph from its own directory.
    """
    path = Path(path).resolve()
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    # Execute exactly the bytes that were hashed (no second read of the file), and write no __pycache__ next to
    # forward.py, which may sit in a private data directory or a read-only trial view.
    module = ModuleType(f"weightcode_trial_forward_{digest[:16]}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    forward = getattr(module, "forward", None)
    if not callable(forward):
        raise ImportError(f"{path} does not define a callable forward(params, x)")
    return forward, digest


def predict(forward: ForwardFn, params: Mapping[str, np.ndarray], x: np.ndarray, batch_size: int) -> np.ndarray:
    """Argmax predictions in batches, with a finiteness check on the logits.

    Logits [B, C] give one prediction per row ([N]); per-position logits [B, L, C] of a sequence model give one per
    position ([N, L]), which ``weightcode.evalkit.metrics.rows_right`` scores by exact match.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs = []
    for start in range(0, len(x), batch_size):
        logits = np.asarray(forward(dict(params), x[start : start + batch_size]))
        rows = min(batch_size, len(x) - start)
        sequence = logits.ndim == 3 and x.ndim == 2 and logits.shape[1] == x.shape[1]
        if not (logits.ndim == 2 or sequence) or logits.shape[0] != rows:
            raise ValueError(f"forward returned logits of shape {logits.shape}")
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("forward produced non-finite logits")
        outputs.append(np.argmax(logits, axis=-1))
    return np.concatenate(outputs).astype(np.int64)


def validate_child(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Child parameters with the base's keys, shapes and dtypes and finite values; raises ValueError otherwise."""
    if set(child) != set(base):
        raise ValueError(f"child keys differ from base: {sorted(set(child) ^ set(base))}")
    cleaned = {}
    for key, base_value in base.items():
        value = np.ascontiguousarray(child[key])
        if value.shape != base_value.shape or value.dtype != base_value.dtype:
            raise ValueError(f"{key}: expected {base_value.dtype}{base_value.shape}, got {value.dtype}{value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains non-finite values")
        cleaned[key] = value
    return cleaned


class PatchCheckOracle:
    """Answers at most ``budget`` patch-check queries for one trial with aggregates only."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        forward: ForwardFn,
        base_params: Mapping[str, np.ndarray],
        target: tuple[int, int] | None,
        budget: int,
        batch_size: int = 1024,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(f"V_check inputs must be an integer array (uint8 pixels, token ids or node ids), got {x.dtype}")
        self._x = x
        self._y = np.asarray(y, dtype=np.int64)
        self._forward = forward
        self._base = {key: np.ascontiguousarray(value) for key, value in base_params.items()}
        self._base_sha = params_sha256(self._base)
        self._a: int | None = None if target is None else int(target[0])
        self._b: int | None = None if target is None else int(target[1])
        self._budget = int(budget)
        self._batch_size = batch_size
        self._base_pred = predict(forward, self._base, self._x, batch_size)
        self._used = 0
        self._log: list[dict[str, object]] = []

    @property
    def base_sha256(self) -> str:
        return self._base_sha

    @property
    def queries_remaining(self) -> int:
        return self._budget - self._used

    @property
    def log(self) -> list[dict[str, object]]:
        """Consumed queries in order, each with its child hash, change count and summary."""
        return [dict(entry) for entry in self._log]

    def _validate_child(self, child: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return validate_child(self._base, child)

    def query(self, child_params: Mapping[str, np.ndarray], patch_id: str) -> dict[str, object]:
        """Evaluate one child on V_check. Validation failures raise before any budget is consumed."""
        child = self._validate_child(child_params)
        if any(entry["patch_id"] == patch_id for entry in self._log):
            raise ValueError(f"patch id {patch_id!r} was already checked")
        if self._used >= self._budget:
            raise BudgetExceeded(f"patch-check budget of {self._budget} queries is spent")
        child_pred = predict(self._forward, child, self._x, self._batch_size)
        summary = summarize(self._y, self._base_pred, child_pred, self._a, self._b)
        query_index = self._used
        self._used += 1
        entry: dict[str, object] = {
            "query_index": query_index,
            "patch_id": patch_id,
            "child_sha256": params_sha256(child),
            "n_changed": count_changed(self._base, child),
            **summary.as_dict(),
            "queries_remaining": self.queries_remaining,
        }
        self._log.append(entry)
        return dict(entry)

    def final(self) -> Candidate:
        """Final patch under contract 5.1: argmax S over consumed queries plus the base no-op."""
        return choose_final(self.candidates())

    def candidates(self) -> list[Candidate]:
        """Consumed queries as candidates for the final-patch rule."""
        return [
            Candidate(
                patch_id=str(entry["patch_id"]),
                score=int(entry["score"]),  # type: ignore[arg-type]
                n_changed=int(entry["n_changed"]),  # type: ignore[arg-type]
                query_index=int(entry["query_index"]),  # type: ignore[arg-type]
            )
            for entry in self._log
        ]


JOINT_PARTS: tuple[str, ...] = ("new", "old")


class JointPatchCheckOracle:
    """Counted patch-check on two image sets under one budget, for the learning experiments (E1).

    ``new`` is the condition's V_check and ``old`` its clean companion (``weightcode.evalkit.companions``: the same rows
    with clean images), so one query scores the new capability and the old one together. Each part is
    S = fixed - broken over its own images, as in ``PatchCheckOracle``. The query's ``score`` is
    S_joint = S_new + S_old, and the final rule is argmax S_joint over consumed queries plus the base no-op
    (``choose_final``). The unsuffixed ``fixed``, ``broken``, ``n``, ``n_base_errors``, ``base_acc`` and
    ``child_acc`` are pooled over both sets, so child_acc - base_acc = S_joint / n. Per-part fields carry the
    suffixes ``_new`` and ``_old``. Validation failures raise before any budget is consumed, and callers receive
    aggregates only.
    """

    def __init__(
        self,
        x_new: np.ndarray,
        y_new: np.ndarray,
        x_old: np.ndarray,
        y_old: np.ndarray,
        forward: ForwardFn,
        base_params: Mapping[str, np.ndarray],
        budget: int,
        batch_size: int = 1024,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self._x = {"new": x_new, "old": x_old}
        self._y = {"new": np.asarray(y_new, dtype=np.int64), "old": np.asarray(y_old, dtype=np.int64)}
        for part in JOINT_PARTS:
            if not np.issubdtype(self._x[part].dtype, np.integer):
                raise TypeError(f"{part} V_check inputs must be an integer array, got {self._x[part].dtype}")
            if len(self._x[part]) != len(self._y[part]):
                raise ValueError(f"{part} V_check has {len(self._x[part])} inputs and {len(self._y[part])} labels")
        self._forward = forward
        self._base = {key: np.ascontiguousarray(value) for key, value in base_params.items()}
        self._base_sha = params_sha256(self._base)
        self._budget = int(budget)
        self._batch_size = batch_size
        self._base_pred = {part: self._predict_part(part, self._base) for part in JOINT_PARTS}
        self._used = 0
        self._log: list[dict[str, object]] = []

    def _predict_part(self, part: str, params: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per-example outputs of ``params`` on one part, compared with that part's targets by ``summarize``: the
        forward's argmax here; a subclass may score by other means (PREREG E14: greedy rollouts, 1 on success)."""
        return predict(self._forward, params, self._x[part], self._batch_size)

    @property
    def base_sha256(self) -> str:
        return self._base_sha

    @property
    def queries_remaining(self) -> int:
        return self._budget - self._used

    @property
    def log(self) -> list[dict[str, object]]:
        """Consumed queries in order, each with its child hash, change count, joint score and per-part fields."""
        return [dict(entry) for entry in self._log]

    def query(self, child_params: Mapping[str, np.ndarray], patch_id: str) -> dict[str, object]:
        """Evaluate one child on both sets. Validation failures raise before any budget is consumed."""
        child = validate_child(self._base, child_params)
        if any(entry["patch_id"] == patch_id for entry in self._log):
            raise ValueError(f"patch id {patch_id!r} was already checked")
        if self._used >= self._budget:
            raise BudgetExceeded(f"patch-check budget of {self._budget} queries is spent")
        parts = {
            part: summarize(self._y[part], self._base_pred[part], self._predict_part(part, child)).as_dict()
            for part in JOINT_PARTS
        }
        query_index = self._used
        self._used += 1
        pooled = {
            key: sum(int(parts[part][key]) for part in JOINT_PARTS)  # type: ignore[call-overload]
            for key in ("score", "fixed", "broken", "n", "n_base_errors")
        }
        base_right = pooled["n"] - pooled["n_base_errors"]
        entry: dict[str, object] = {
            "query_index": query_index,
            "patch_id": patch_id,
            "child_sha256": params_sha256(child),
            "n_changed": count_changed(self._base, child),
            **pooled,
            "base_acc": base_right / pooled["n"],
            "child_acc": (base_right + pooled["score"]) / pooled["n"],
            **{f"{key}_{part}": value for part in JOINT_PARTS for key, value in parts[part].items() if value is not None},
            "queries_remaining": self.queries_remaining,
        }
        self._log.append(entry)
        return dict(entry)

    def final(self) -> Candidate:
        """Final patch: argmax S_joint over consumed queries plus the base no-op (contract 5.2 rule on S_joint)."""
        return choose_final(self.candidates())

    def candidates(self) -> list[Candidate]:
        """Consumed queries as candidates for the final-patch rule, scored by S_joint."""
        return [
            Candidate(
                patch_id=str(entry["patch_id"]),
                score=int(entry["score"]),  # type: ignore[arg-type]
                n_changed=int(entry["n_changed"]),  # type: ignore[arg-type]
                query_index=int(entry["query_index"]),  # type: ignore[arg-type]
            )
            for entry in self._log
        ]
