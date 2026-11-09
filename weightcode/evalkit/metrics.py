"""Paired readouts of a child model against its base model, and the mechanical final-patch rule (contract 5.1, 6).

All readouts compare a child model with its base model on the same images, so every count is paired per image.
The primary quantity (user decision 2026-11-03) is the edited model's own capability: S = fixed - broken over all
images, where ``fixed`` counts base-wrong images the child gets right and ``broken`` counts base-right images the
child gets wrong. On the sealed test S / n is the paired accuracy change. A target confusion pair (a, b), images of
true class ``a`` that the base model predicts as ``b``, is optional and gives secondary readouts only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

IGNORE = -1
"""Per-position target of a sequence row where no answer is scored (sequence datasets, ``y`` [N, L])."""


def rows_right(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-row correctness: ``pred == y`` for labels [N]; exact match over labelled positions for targets [N, L].

    A sequence row is right when every position whose target is not IGNORE is predicted right; a row without any
    labelled position is refused, so no row is right vacuously.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    if pred.shape != y.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {y.shape}")
    if y.ndim == 1:
        return pred == y
    if y.ndim != 2:
        raise ValueError(f"targets must be [N] or [N, L], got {y.shape}")
    labelled = y != IGNORE
    if np.any(~labelled.any(axis=1)):
        raise ValueError("a sequence row has no labelled position")
    return np.all((pred == y) | ~labelled, axis=1)


def token_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of labelled positions predicted right (sequence targets [N, L])."""
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 2 or pred.shape != y.shape:
        raise ValueError(f"token accuracy needs [N, L] targets and predictions, got {y.shape} and {pred.shape}")
    labelled = y != IGNORE
    return float(np.mean(pred[labelled] == y[labelled]))


@dataclass(frozen=True)
class CheckSummary:
    """Aggregate patch-check readout. Contains no per-image information.

    ``score`` is S = fixed - broken over all images, the quantity the final-patch rule maximizes. The target
    fields are None unless a pair (a, b) was given; ``target_score`` = fixed_target - broken is the earlier
    pair-repair score, kept as a secondary readout.
    """

    n: int
    n_base_errors: int
    fixed: int
    broken: int
    base_acc: float
    child_acc: float
    score: int
    target_a: int | None = None
    target_b: int | None = None
    n_target_class: int | None = None
    n_base_target_errors: int | None = None
    base_target_conf: float | None = None
    child_target_conf: float | None = None
    fixed_target: int | None = None
    target_score: int | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _validate(y: np.ndarray, base_pred: np.ndarray, child_pred: np.ndarray, a: int | None, b: int | None) -> None:
    if y.ndim not in (1, 2) or base_pred.shape != y.shape or child_pred.shape != y.shape:
        raise ValueError(f"shape mismatch: y {y.shape}, base {base_pred.shape}, child {child_pred.shape}")
    if y.shape[0] == 0:
        raise ValueError("no images")
    if y.ndim == 2 and a is not None:
        raise ValueError("a target pair needs one label per row")
    for name, array in (("y", y), ("base_pred", base_pred), ("child_pred", child_pred)):
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must be an integer array, got {array.dtype}")
    if (a is None) != (b is None):
        raise ValueError("give both a and b of the target pair, or neither")
    if a is not None:
        if a == b:
            raise ValueError("target pair must have a != b")
        if not np.any(y == a):
            raise ValueError(f"no images of target class {a}")


def summarize(
    y: np.ndarray, base_pred: np.ndarray, child_pred: np.ndarray, a: int | None = None, b: int | None = None
) -> CheckSummary:
    """Patch-check summary: S = fixed - broken over all images, plus target-pair readouts if (a, b) is given.

    ``fixed`` counts base-wrong images the child gets right and ``broken`` base-right images the child gets
    wrong, over all images. For sequence rows (targets [N, L]) right means exact match (``rows_right``). With a pair, ``fixed_target`` counts images with y = a, base prediction b and child
    prediction a.
    """
    y = np.asarray(y)
    base_pred = np.asarray(base_pred)
    child_pred = np.asarray(child_pred)
    _validate(y, base_pred, child_pred, a, b)
    base_right = rows_right(y, base_pred)
    child_right = rows_right(y, child_pred)
    fixed = int(np.count_nonzero(~base_right & child_right))
    broken = int(np.count_nonzero(base_right & ~child_right))
    target: dict[str, int | float] = {}
    if a is not None and b is not None:
        in_a = y == a
        base_target_error = in_a & (base_pred == b)
        fixed_target = int(np.count_nonzero(base_target_error & (child_pred == a)))
        target = dict(
            target_a=int(a),
            target_b=int(b),
            n_target_class=int(np.count_nonzero(in_a)),
            n_base_target_errors=int(np.count_nonzero(base_target_error)),
            base_target_conf=float(np.mean(base_pred[in_a] == b)),
            child_target_conf=float(np.mean(child_pred[in_a] == b)),
            fixed_target=fixed_target,
            target_score=fixed_target - broken,
        )
    return CheckSummary(
        n=int(y.shape[0]),
        n_base_errors=int(np.count_nonzero(~base_right)),
        fixed=fixed,
        broken=broken,
        base_acc=float(np.mean(base_right)),
        child_acc=float(np.mean(child_right)),
        score=fixed - broken,
        **target,  # type: ignore[arg-type]
    )


def sealed_readout(
    y: np.ndarray,
    base_pred: np.ndarray,
    child_pred: np.ndarray,
    a: int | None = None,
    b: int | None = None,
    n_classes: int | None = None,
) -> dict[str, float | int | list[float] | None]:
    """Hidden-evaluation readouts, paired per image against the base model (contract section 6).

    Primary endpoint (PREREG section 3, user decision 2026-11-03): delta_acc = (fixed - broken) / n, the edited
    model's paired accuracy change. Secondary: per-class accuracy deltas and the worst one. With a target pair,
    R = (fixed_target - broken) / n_base_target_errors as well, None when the base makes no target errors.
    ``n_classes`` is required; it stays a keyword after (a, b) so that older positional calls keep working.
    """
    if n_classes is None:
        raise TypeError("sealed_readout needs n_classes")
    summary = summarize(y, base_pred, child_pred, a, b)
    y = np.asarray(y)
    base_pred = np.asarray(base_pred)
    child_pred = np.asarray(child_pred)
    per_class_delta = []
    for cls in range(n_classes):
        mask = y == cls
        if not np.any(mask):
            raise ValueError(f"class {cls} absent from the evaluation set")
        per_class_delta.append(float(np.mean(child_pred[mask] == cls) - np.mean(base_pred[mask] == cls)))
    readout: dict[str, float | int | list[float] | None] = dict(summary.as_dict())
    readout.update(
        delta_acc=summary.score / summary.n,
        per_class_acc_delta=per_class_delta,
        worst_class_delta=min(per_class_delta),
    )
    if a is not None and b is not None:
        n_err = summary.n_base_target_errors
        readout.update(
            R=(summary.target_score / n_err) if n_err else None,  # type: ignore[operator]
            delta_target=summary.child_target_conf - summary.base_target_conf,  # type: ignore[operator]
            delta_recall_a=per_class_delta[a],
        )
    return readout


def color_following_rate(pred: np.ndarray, y: np.ndarray, color: np.ndarray) -> float:
    """Colored-MNIST mechanism readout: P(prediction = color class | color class != label).

    Class c owns color c, so a model that follows the spurious color predicts the color's class whenever color
    and label disagree. A repair that removes the cause lowers this rate; one that only patches the target
    pair leaves it high.
    """
    pred = np.asarray(pred)
    y = np.asarray(y)
    color = np.asarray(color)
    if not (pred.shape == y.shape == color.shape) or y.ndim != 1:
        raise ValueError(f"shape mismatch: pred {pred.shape}, y {y.shape}, color {color.shape}")
    off_color = color != y
    if not np.any(off_color):
        raise ValueError("no images whose color disagrees with the label")
    return float(np.mean(pred[off_color] == color[off_color]))


def color_split_target_fixes(
    y: np.ndarray, base_pred: np.ndarray, child_pred: np.ndarray, color: np.ndarray, a: int, b: int
) -> dict[str, int]:
    """Colored-MNIST decomposition of the target pair (a, b) by cause.

    Base target errors (y = a, base prediction b) split into those whose image has color b (the spurious cue
    points at b) and the rest (shape-driven). A repair that removes the color cause should fix the first group
    preferentially; a pair-specific patch need not.
    """
    y = np.asarray(y)
    base_pred = np.asarray(base_pred)
    child_pred = np.asarray(child_pred)
    color = np.asarray(color)
    _validate(y, base_pred, child_pred, a, b)
    if color.shape != y.shape:
        raise ValueError(f"color shape {color.shape} != y shape {y.shape}")
    base_error = (y == a) & (base_pred == b)
    fixed = base_error & (child_pred == a)
    color_b = color == b
    return {
        "n_base_target_errors_color_b": int(np.count_nonzero(base_error & color_b)),
        "n_base_target_errors_color_other": int(np.count_nonzero(base_error & ~color_b)),
        "fixed_target_color_b": int(np.count_nonzero(fixed & color_b)),
        "fixed_target_color_other": int(np.count_nonzero(fixed & ~color_b)),
    }


def color_split_fixes(
    y: np.ndarray, base_pred: np.ndarray, child_pred: np.ndarray, color: np.ndarray
) -> dict[str, int]:
    """Colored-MNIST decomposition of all base errors by cause, with no target pair.

    A base error is color-following when the base predicts the color's class (pred = color != y). A repair that
    removes the color cause should fix these preferentially; ``fixed_*`` counts child-correct images per group.
    """
    y = np.asarray(y)
    base_pred = np.asarray(base_pred)
    child_pred = np.asarray(child_pred)
    color = np.asarray(color)
    _validate(y, base_pred, child_pred, None, None)
    if color.shape != y.shape:
        raise ValueError(f"color shape {color.shape} != y shape {y.shape}")
    base_error = base_pred != y
    following = base_error & (base_pred == color)
    fixed = base_error & (child_pred == y)
    return {
        "n_base_errors_color_following": int(np.count_nonzero(following)),
        "n_base_errors_other": int(np.count_nonzero(base_error & ~following)),
        "fixed_color_following": int(np.count_nonzero(fixed & following)),
        "fixed_other": int(np.count_nonzero(fixed & ~following)),
    }


@dataclass(frozen=True)
class Candidate:
    """One patch-checked candidate. ``patch_id`` None is the base model, the S = 0 no-op."""

    patch_id: str | None
    score: int
    n_changed: int
    query_index: int


BASE_CANDIDATE = Candidate(patch_id=None, score=0, n_changed=0, query_index=-1)


def choose_final(candidates: Sequence[Candidate]) -> Candidate:
    """Final-patch rule (contract 5.1): argmax S = fixed - broken over patch-checked candidates plus the base no-op.

    Ties go to fewer changed parameters, then to the earlier query. The base model always competes, so a
    patch with S <= 0 is never chosen over it.
    """
    pool = [BASE_CANDIDATE, *candidates]
    for candidate in candidates:
        if candidate.patch_id is None or candidate.query_index < 0:
            raise ValueError(f"checked candidates need a patch id and a query index >= 0: {candidate}")
    return min(pool, key=lambda c: (-c.score, c.n_changed, c.query_index))


def sequence_readout(
    y: np.ndarray, base_pred: np.ndarray, child_pred: np.ndarray, part: np.ndarray | None = None
) -> dict[str, object]:
    """Paired readout of sequence rows: the row summary, delta_acc = S / n, token accuracies, and the same per part.

    ``part`` (int [N]) labels the rows of a multi-part set; each part gets its own summary and delta_acc.
    """
    summary = summarize(y, base_pred, child_pred)
    readout: dict[str, object] = dict(summary.as_dict())
    readout.update(
        delta_acc=summary.score / summary.n,
        base_token_acc=token_accuracy(y, base_pred),
        child_token_acc=token_accuracy(y, child_pred),
    )
    if part is not None:
        part = np.asarray(part)
        if part.shape != (np.asarray(y).shape[0],):
            raise ValueError("part must label every row")
        per_part = {}
        for value in np.unique(part):
            rows = part == value
            sub = summarize(np.asarray(y)[rows], np.asarray(base_pred)[rows], np.asarray(child_pred)[rows])
            per_part[str(int(value))] = {
                **sub.as_dict(),
                "delta_acc": sub.score / sub.n,
                "base_token_acc": token_accuracy(np.asarray(y)[rows], np.asarray(base_pred)[rows]),
                "child_token_acc": token_accuracy(np.asarray(y)[rows], np.asarray(child_pred)[rows]),
            }
        readout["parts"] = per_part
    return readout
