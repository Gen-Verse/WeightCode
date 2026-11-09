"""PREREG T03's checks of the edit interface on the native ResNet-18 (``weightcode.models.resnet18native``).

The pass rule (PREREG "T03", fixed before any ImageNet image was decoded):

(1) on D, forward.py's logits meet ``PARITY_RULE`` (``weightcode.models.parity.check_logit_parity``) against the float64
    and float32 torch rebuilds; the float32 argmax disagreements are recorded (``parity_record``);
(2) saving and reloading the params gives the same params_sha256 and identical logits (``save_reload``); the fixed
    test edit ``FIXED_EDIT_OPS``, written as a patch file through the patch interface (``weightcode.patch``), applied, saved
    and reloaded (``apply_fixed_edit``), gives logits that meet the same rule against torch with the same edited
    weights;
(3) two runs give bitwise identical logits (``bitwise_equal``);
(4) the base's top-1 accuracy on D lies within ``ACCURACY_REFERENCE`` +- ``ACCURACY_BAND`` points (``accuracy_record``).
    A value outside the band is a preprocessing or label-order defect; nothing is tuned to pass it.

The test edit is an interface check, not an improvement: four ops of fixed small size on the head, a head bias, an
early convolution and a late convolution, so that the edited logits test that an edit anywhere in the network reaches
forward.py and torch alike.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.models.export import ParityError
from weightcode.models.parity import FLOAT32_STEP, PARITY_RULE, check_logit_parity, float32_tolerance, logit_scale
from weightcode.patch import Limits, apply_patch_file
from weightcode.patch.ops import SCHEMA as PATCH_SCHEMA

ACCURACY_REFERENCE = 69.758
"""torchvision's published top-1 accuracy (%) of ResNet18_Weights.IMAGENET1K_V1 on the full validation split."""
ACCURACY_BAND = 5.0
FIXED_EDIT_OPS: tuple[dict[str, Any], ...] = (
    {"op": "add", "tensor": "fc.weight", "index": [[0, 0], [1, 7], [65, 100], [207, 511], [281, 3], [409, 256],
                                                   [530, 42], [999, 511]],
     "values": [0.05, -0.05, 0.02, -0.03, 0.04, 0.01, -0.02, 0.03]},
    {"op": "add", "tensor": "fc.bias", "index": [[1], [999]], "values": [0.1, -0.1]},
    {"op": "add", "tensor": "layer1.0.conv1.weight", "index": [[0, 0, 1, 1], [5, 3, 0, 2]], "values": [0.01, -0.01]},
    {"op": "scale_rows", "tensor": "layer4.1.conv2.weight", "rows": [3, 17], "factor": 0.9},
)
FIXED_EDIT_LIMITS = Limits()
PATCH_FILE = "patch.json"


def fixed_edit_patch(parent_sha256: str) -> dict[str, Any]:
    """The fixed test edit as a ``weightcode.patch.v0`` patch on the model with ``parent_sha256``."""
    return {
        "schema": PATCH_SCHEMA,
        "parent_sha256": parent_sha256,
        "ops": [dict(op) for op in FIXED_EDIT_OPS],
        "hypothesis": "T03 interface check: a fixed small edit of the head, a head bias and two convolutions",
        "predicted_effect": "forward.py and torch give the same logits on the edited weights; accuracy is not the aim",
    }


def save_reload(params: Mapping[str, np.ndarray], path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Save ``params`` with ``save_params``, reload with ``load_params``; return the reloaded params and a record
    (both hashes, whether they agree, the file size and whether every reloaded array equals the original bitwise)."""
    path = Path(path)
    before = params_sha256(params)
    saved = save_params(path, params)
    reloaded = load_params(path)
    after = params_sha256(reloaded)
    same_arrays = set(reloaded) == set(params) and all(bitwise_equal(reloaded[k], np.asarray(params[k]))
                                                         for k in params)
    record = {"params_sha256": before, "saved_sha256": saved, "reloaded_sha256": after,
              "same_sha256": before == saved == after, "arrays_bitwise_equal": bool(same_arrays),
              "file": path.name, "file_bytes": int(path.stat().st_size)}
    return reloaded, record


def apply_fixed_edit(params: Mapping[str, np.ndarray], work_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Write the test edit to ``work_dir/patch.json``, apply it with ``apply_patch_file`` (``FIXED_EDIT_LIMITS``), save
    the child to ``work_dir/edited.npz`` and reload it. Returns the reloaded child and a record."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    parent_sha = params_sha256(params)
    patch_path = work_dir / PATCH_FILE
    patch_path.write_text(json.dumps(fixed_edit_patch(parent_sha), indent=2) + "\n", encoding="utf-8")
    result = apply_patch_file(params, patch_path, FIXED_EDIT_LIMITS, parent_sha)
    reloaded, saved = save_reload(result.child, work_dir / "edited.npz")
    if saved["reloaded_sha256"] != result.child_sha256:
        raise RuntimeError("the reloaded edited params hash differently from the patch result")
    changed = {name: stats["changed"] for name, stats in result.per_tensor.items() if stats["changed"]}
    record = {"parent_sha256": parent_sha, "child_sha256": result.child_sha256, "n_changed": result.n_changed,
              "edit_size": result.edit_size, "changed_per_tensor": changed, "patch_file": PATCH_FILE,
              "save_reload": saved}
    return reloaded, record


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Same dtype, same shape and the same bytes."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(
        b).tobytes()


def parity_record(mine: np.ndarray, exact: np.ndarray, float32: np.ndarray) -> dict[str, Any]:
    """``PARITY_RULE`` as ``check_logit_parity`` decides it, with the measured values also on a failure, and the
    float32 argmax disagreements."""
    mine64, exact64, f32 = (np.asarray(a, dtype=np.float64) for a in (mine, exact, float32))
    record: dict[str, Any] = {"rows": int(mine64.shape[0]), "rule": dict(PARITY_RULE)}
    try:
        record.update(check_logit_parity(mine64, exact64, f32))
        record["verdict"] = "pass"
    except ParityError as error:
        record.update({
            "max_abs_diff": float(np.max(np.abs(mine64 - f32))),
            "tolerance": float32_tolerance(exact64),
            "logit_scale": logit_scale(exact64),
            "float64_max_abs_diff": float(np.max(np.abs(mine64 - exact64))),
            "float64_max_rel_diff": float(np.max(np.abs(mine64 - exact64) / np.maximum(1.0, np.abs(exact64)))),
            "float64_tolerance": FLOAT32_STEP,
            "verdict": "fail",
            "error": str(error),
        })
    record["float32_argmax_disagreements"] = int(np.count_nonzero(mine64.argmax(1) != f32.argmax(1)))
    record["float64_argmax_disagreements"] = int(np.count_nonzero(mine64.argmax(1) != exact64.argmax(1)))
    return record


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Percent of rows whose label is among the k largest logits (ties broken by numpy's argsort order)."""
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError(f"logits {logits.shape} and labels {labels.shape} do not match")
    top = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    return 100.0 * float(np.mean(np.any(top == labels[:, None], axis=1)))


def accuracy_record(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Pass item (4): top-1 (argmax) accuracy in percent against the band; top-5 is descriptive."""
    labels = np.asarray(labels)
    pred = np.asarray(logits).argmax(1)
    top1 = 100.0 * float(np.mean(pred == labels))
    low, high = ACCURACY_REFERENCE - ACCURACY_BAND, ACCURACY_REFERENCE + ACCURACY_BAND
    n = int(labels.shape[0])
    p = top1 / 100.0
    return {"rows": n, "top1_percent": top1, "n_correct": int(np.count_nonzero(pred == labels)),
            "top5_percent": top_k_accuracy(logits, labels, 5), "reference_percent": ACCURACY_REFERENCE,
            "band": [low, high], "binomial_se_percent": 100.0 * float(np.sqrt(p * (1.0 - p) / n)) if n else None,
            "verdict": "pass" if low <= top1 <= high else "fail"}


__all__ = [
    "ACCURACY_BAND", "ACCURACY_REFERENCE", "PATCH_FILE", "FIXED_EDIT_LIMITS", "FIXED_EDIT_OPS", "accuracy_record",
    "apply_fixed_edit", "bitwise_equal", "parity_record", "save_reload", "fixed_edit_patch", "top_k_accuracy",
]
