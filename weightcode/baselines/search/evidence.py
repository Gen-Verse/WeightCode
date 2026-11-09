"""Evidence sets and the counted instance forward used by both search baselines (contract sections 3 and 4).

Negative inputs are the evidence failures ``x_fail``: in evidence-v1 instances, random base-model errors of any
class (each must be misclassified by the base model); in the superseded evidence-v0 instances, true class a
predicted b. Positive inputs are the random evidence images ``x_rand`` that the base model classifies correctly.
Both searches score candidates against the true labels. Every call of the instance forward on evidence goes through
``CountedForward`` so the run record reports the number of calls and images, as contract section 4 requires
(compute on evidence is reported, not capped).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

ForwardFn = Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]


class CountedForward:
    """The instance's ``forward.py`` with call and image counters; returns float64 logits."""

    def __init__(self, forward_fn: ForwardFn) -> None:
        self._forward = forward_fn
        self.calls = 0
        self.images = 0

    def __call__(self, params: Mapping[str, np.ndarray], x_uint8: np.ndarray) -> np.ndarray:
        logits = np.asarray(self._forward(dict(params), x_uint8), dtype=np.float64)
        self.calls += 1
        self.images += int(len(x_uint8))
        if logits.ndim != 2 or logits.shape[0] != len(x_uint8):
            raise ValueError(f"forward returned logits of shape {logits.shape} for {len(x_uint8)} images")
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("forward produced non-finite logits on evidence")
        return logits


@dataclass(frozen=True)
class EvidenceSets:
    """Negative and positive evidence inputs with labels; ``x_all`` stacks negatives then positives.

    ``target`` is the (a, b) pair of an evidence-v0 instance and None for evidence-v1; the searches do not use it.
    """

    target: tuple[int, int] | None
    x_neg: np.ndarray
    y_neg: np.ndarray
    x_pos: np.ndarray
    y_pos: np.ndarray
    pos_rand_indices: np.ndarray
    n_rand: int
    n_rand_wrong: int

    @property
    def n_neg(self) -> int:
        """Number of negative inputs (evidence failures)."""
        return int(len(self.y_neg))

    @property
    def n_pos(self) -> int:
        """Number of positive inputs (correctly classified random evidence images)."""
        return int(len(self.y_pos))

    @property
    def x_all(self) -> np.ndarray:
        """Negatives then positives."""
        return np.concatenate([self.x_neg, self.x_pos], axis=0)

    @property
    def y_all(self) -> np.ndarray:
        """Labels of ``x_all``."""
        return np.concatenate([self.y_neg, self.y_pos], axis=0)

    def summary(self) -> dict[str, object]:
        """Set sizes and the true-label counts of the negatives, for the run record."""
        labels, counts = np.unique(self.y_neg, return_counts=True)
        return {
            "n_neg": self.n_neg,
            "n_pos": self.n_pos,
            "n_rand": self.n_rand,
            "n_rand_wrong": self.n_rand_wrong,
            "neg_true_label_counts": {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)},
            "target": None if self.target is None else list(self.target),
        }


def build_evidence_sets(
    evidence: Mapping[str, np.ndarray],
    target: tuple[int, int] | None,
    base_params: Mapping[str, np.ndarray],
    counted: CountedForward,
) -> EvidenceSets:
    """Split the editor-visible evidence into negatives and positives, checking the failure definition.

    Raises if a ``x_fail`` image is classified correctly by the base model (with a pair: if it is not a true-a image
    predicted b), or if the evidence file's ``pred_fail`` / ``pred_rand`` disagree with the base model's predictions
    through the instance forward, since the baselines would then optimise against a wrong failure set.
    """
    for key in ("x_fail", "y_fail", "x_rand", "y_rand"):
        if key not in evidence:
            raise KeyError(f"evidence lacks {key}")
    x_fail = np.asarray(evidence["x_fail"])
    y_fail = np.asarray(evidence["y_fail"]).astype(np.int64)
    x_rand = np.asarray(evidence["x_rand"])
    y_rand = np.asarray(evidence["y_rand"]).astype(np.int64)
    if x_fail.dtype != np.uint8 or x_rand.dtype != np.uint8:
        raise TypeError("evidence images must be uint8")
    if len(x_fail) != len(y_fail) or len(x_rand) != len(y_rand) or len(x_fail) == 0:
        raise ValueError("evidence images and labels differ in length or x_fail is empty")
    fail_pred = np.argmax(counted(base_params, x_fail), axis=1)
    rand_pred = np.argmax(counted(base_params, x_rand), axis=1)
    for key, computed in (("pred_fail", fail_pred), ("pred_rand", rand_pred)):
        if key in evidence and not np.array_equal(np.asarray(evidence[key]).astype(np.int64), computed):
            raise ValueError(f"evidence {key} differs from the base model's predictions through forward.py")
    if np.any(fail_pred == y_fail):
        raise ValueError(f"{int(np.count_nonzero(fail_pred == y_fail))} x_fail images are classified correctly")
    pair: tuple[int, int] | None = None
    if target is not None:
        pair = (int(target[0]), int(target[1]))
        if np.any(y_fail != pair[0]) or np.any(fail_pred != pair[1]):
            raise ValueError(f"x_fail must be true class {pair[0]} predicted {pair[1]} by the base model")
    correct = np.flatnonzero(rand_pred == y_rand)
    if correct.size == 0:
        raise ValueError("no correctly classified x_rand image; positive set would be empty")
    return EvidenceSets(
        target=pair,
        x_neg=x_fail,
        y_neg=y_fail,
        x_pos=x_rand[correct],
        y_pos=y_rand[correct],
        pos_rand_indices=correct,
        n_rand=int(len(y_rand)),
        n_rand_wrong=int(len(y_rand) - correct.size),
    )


def evidence_readout(logits: np.ndarray, sets: EvidenceSets) -> dict[str, int]:
    """Counts on evidence for logits computed on ``sets.x_all`` (negatives first), against the true labels."""
    pred = np.argmax(logits, axis=1)
    neg, pos = pred[: sets.n_neg], pred[sets.n_neg :]
    fixed = int(np.count_nonzero(neg == sets.y_neg))
    return {
        "neg_fixed": fixed,
        "neg_still_wrong": sets.n_neg - fixed,
        "pos_broken": int(np.count_nonzero(pos != sets.y_pos)),
        "n_neg": sets.n_neg,
        "n_pos": sets.n_pos,
    }
