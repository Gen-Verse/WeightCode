"""Severity calibration of the drawn-parameter conditions on the train split (contract section 2.1).

The rule, fixed before any run (user decision 2026-11-04):

* The target drop ``D*`` is the mean, over the dev models (seeds 100 and 101) and the reference conditions
  ``contrast_s3`` and ``gaussian_noise_s3``, of clean train accuracy minus corrupted train accuracy.
* For a drawn condition, ``s`` is the smallest value of its grid whose mean drop over the two dev models reaches
  ``D*``. Colormix values with ``cond_2(A) > 10`` are not eligible. If no eligible value reaches ``D*``, the largest
  eligible value is used and the condition is recorded as below target.

Only the train split is read: the clean one, the reference conditions' stored corrupted ones, and the drawn
transform of the clean one at each grid value (the same ``apply_drawn`` bytes that the build stores). Accuracies come
from the base_eval path of ``weightcode.models.importer`` (the dataset's checked numpy forward.py and ``predict``).
"Reaches" is decided on integer counts, ``n_pairs * drop_count(s) >= n_models * reference_drop_count``, which is
``mean drop(s) >= D*`` without rounding. Every grid value is evaluated and logged, eligible or not.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import Params
from weightcode.data import conditions as C
from weightcode.data import store
from weightcode.evalkit.patch_check import ForwardFn, predict
from weightcode.models.importer import EVAL_BATCH_SIZE, checked_forward, load_indexed_models

CALIBRATION_SCHEMA = "weightcode.severity_calibration.v0"
CALIBRATION_SPLIT = "train"
CALIBRATION_MODELS: Mapping[str, tuple[int, ...]] = {"dev": (100, 101)}
"""The models of the rule: the dev models 100 and 101 of the dataset's models/index.json."""
REFERENCE_CONDITIONS: tuple[str, ...] = ("contrast_s3", "gaussian_noise_s3")
RULE = (
    "D* = mean over the dev models 100 and 101 and the conditions contrast_s3 and gaussian_noise_s3 of clean train "
    "accuracy minus corrupted train accuracy; s = the smallest grid value whose mean train accuracy drop over the "
    "two dev models is >= D*, among eligible values (colormix: cond_2(A) <= 10); if none reaches D*, the largest "
    "eligible value, recorded as below target"
)
RULE_SOURCE = "docs/CONTRACT_v0_DRAFT.md section 2.1 (user decision 2026-11-04)"
EVALUATION = (
    "argmax of the dataset's numpy forward.py (sha256 checked) via weightcode.evalkit.patch_check.predict, batch "
    f"{EVAL_BATCH_SIZE}: the base_eval path of weightcode.models.importer"
)


def load_calibration_split(dataset: str, split: str, condition: str | None = None) -> dict[str, np.ndarray]:
    """The train split of ``dataset`` (or of its ``condition``), checked against its dataset.json hash.

    Any other split is refused: the calibration never reads V_select, V_evidence, V_check or test.
    """
    if split != CALIBRATION_SPLIT:
        raise ValueError(f"the severity calibration reads only the {CALIBRATION_SPLIT} split, not {split!r}")
    manifest = store.read_dataset_json(dataset, condition)
    arrays = store.load_split(dataset, split, condition)
    if store.split_sha256(arrays) != manifest["splits"][split]["sha256"]:
        raise RuntimeError(f"{dataset}/{condition or ''} {split}.npz does not match its dataset.json hash")
    return arrays


def correct_count(forward: ForwardFn, params: Params, arrays: Mapping[str, np.ndarray]) -> int:
    """Number of images whose forward.py argmax equals the label."""
    return int(np.sum(predict(forward, params, arrays["x"], batch_size=EVAL_BATCH_SIZE) == arrays["y"]))


def select_severity(
    curve: Sequence[Mapping[str, Any]], reference_drop_count: int, n_pairs: int, n_models: int
) -> dict[str, Any]:
    """Apply the rule to a curve of ``{"severity", "eligible", "drop_count"}`` rows in increasing severity.

    ``drop_count`` sums clean minus transformed correct counts over the ``n_models`` models, and
    ``reference_drop_count`` sums the same over the ``n_pairs`` (model, reference condition) pairs.
    """
    severities = [row["severity"] for row in curve]
    if severities != sorted(severities) or len(set(severities)) != len(severities):
        raise ValueError("the curve must list distinct severities in increasing order")
    eligible = [row for row in curve if row["eligible"]]
    if not eligible:
        raise ValueError("no grid value is eligible; the rule defines no severity")
    reaching = [row for row in eligible if n_pairs * row["drop_count"] >= n_models * reference_drop_count]
    chosen, status = (reaching[0], "reached_target") if reaching else (eligible[-1], "below_target")
    return {"severity": chosen["severity"], "status": status}


def _model_rows(
    models: Sequence[tuple[Mapping[str, Any], Params]], counts: Sequence[int], clean: Sequence[int], n: int
) -> list[dict[str, Any]]:
    return [
        {
            "model_id": entry["model_id"],
            "correct": count,
            "acc": count / n,
            "drop": (base - count) / n,
        }
        for (entry, _), count, base in zip(models, counts, clean)
    ]


def calibrate(
    name: str, run_info: Mapping[str, Any], log_fn: Callable[[str], None] = print
) -> dict[str, Any]:
    """Run the rule for one drawn condition and return its calibration log, which ``write_calibration`` writes.

    Refuses a condition whose dataset.json already exists: the severity is fixed before any split is written.
    """
    condition = C.get_condition(name)
    if not isinstance(condition, C.DrawnCondition):
        raise TypeError(f"{name} is not a drawn-parameter condition")
    dataset = condition.dataset
    if store.dataset_json_path(dataset, name).exists():
        raise FileExistsError(f"{store.dataset_json_path(dataset, name)} exists; the calibration precedes the build")
    clean_manifest = store.read_dataset_json(dataset)
    forward, forward_sha = checked_forward(dataset, clean_manifest)
    models = load_indexed_models(dataset, CALIBRATION_MODELS)
    clean = load_calibration_split(dataset, CALIBRATION_SPLIT)
    n = int(clean["y"].size)
    clean_counts = [correct_count(forward, params, clean) for _, params in models]
    log_fn(f"[{name}] clean train acc " + ", ".join(f"{c / n:.4f}" for c in clean_counts))

    reference = []
    reference_drop_count = 0
    for ref in REFERENCE_CONDITIONS:
        ref_manifest = store.read_dataset_json(dataset, ref)
        if ref_manifest["condition"]["clean"]["splits"][CALIBRATION_SPLIT] != clean_manifest["splits"][CALIBRATION_SPLIT]["sha256"]:
            raise RuntimeError(f"{ref} was not built from the current clean train split")
        arrays = load_calibration_split(dataset, CALIBRATION_SPLIT, ref)
        if not (np.array_equal(arrays["y"], clean["y"]) and np.array_equal(arrays["idx"], clean["idx"])):
            raise RuntimeError(f"{ref} train rows differ from the clean train rows")
        counts = [correct_count(forward, params, arrays) for _, params in models]
        del arrays
        reference_drop_count += sum(base - count for base, count in zip(clean_counts, counts))
        reference.append(
            {
                "condition": ref,
                "train_sha256": ref_manifest["splits"][CALIBRATION_SPLIT]["sha256"],
                "per_model": _model_rows(models, counts, clean_counts, n),
            }
        )
        log_fn(f"[{name}] reference {ref} train acc " + ", ".join(f"{c / n:.4f}" for c in counts))
    n_pairs = len(models) * len(REFERENCE_CONDITIONS)
    target = reference_drop_count / (n_pairs * n)
    log_fn(f"[{name}] D* = {target:.6f}")

    params = condition.parameters()
    image_hw = (int(clean_manifest["image_shape"][1]), int(clean_manifest["image_shape"][2]))
    curve = []
    for s in condition.grid:
        diagnostics = C.operator_diagnostics(condition.transform, params, s, image_hw)
        x = C.apply_drawn(clean["x"], condition.transform, params, s)
        counts = [correct_count(forward, p, {"x": x, "y": clean["y"]}) for _, p in models]
        del x
        drop_count = sum(base - count for base, count in zip(clean_counts, counts))
        row = {
            **diagnostics,
            "per_model": _model_rows(models, counts, clean_counts, n),
            "drop_count": drop_count,
            "mean_drop": drop_count / (len(models) * n),
            "reaches_target": bool(n_pairs * drop_count >= len(models) * reference_drop_count),
        }
        curve.append(row)
        extra = (
            f" cond_2(A) {row['cond_2']:.3f}" if "cond_2" in row
            else f" min|FFT(K)| {row['min_abs_fft']:.4f}" if "min_abs_fft" in row
            else f" gammas {np.round(row['gammas'], 4).tolist()}"
        )
        log_fn(
            f"[{name}] s={s:g} mean drop {row['mean_drop']:.4f} eligible {row['eligible']} "
            f"reaches {row['reaches_target']}{extra}"
        )
    choice = select_severity(curve, reference_drop_count, n_pairs, len(models))
    chosen_row = next(row for row in curve if row["severity"] == choice["severity"])
    return {
        "schema": CALIBRATION_SCHEMA,
        "condition": name,
        "dataset": dataset,
        "transform": condition.transform,
        "split": CALIBRATION_SPLIT,
        "rule": RULE,
        "rule_source": RULE_SOURCE,
        "formula": C.DRAWN_FORMULAS[condition.transform],
        "computation": C.DRAWN_COMPUTATION[condition.transform],
        "storage": C.STORAGE_RULE,
        "parameters": {
            "split_seed": C.SPLIT_SEED,
            "stream": condition.parameter_stream,
            "draw": C.PARAMETER_DRAWS[condition.transform],
            "values": {key: value.tolist() for key, value in params.items()},
            "numpy": np.__version__,
        },
        "grid": list(condition.grid),
        "colormix_max_cond_2": C.COLORMIX_MAX_COND if condition.transform == "colormix" else None,
        "evaluation": EVALUATION,
        "forward_sha256": forward_sha,
        "clean_train_sha256": clean_manifest["splits"][CALIBRATION_SPLIT]["sha256"],
        "n_train": n,
        "models": [
            {
                "model_id": entry["model_id"],
                "role": entry["role"],
                "seed": entry["seed"],
                "params_sha256": entry["params_sha256"],
                "clean_train_correct": base,
                "clean_train_acc": base / n,
            }
            for (entry, _), base in zip(models, clean_counts)
        ],
        "target_drop": {
            "value": target,
            "drop_count": reference_drop_count,
            "n_pairs": n_pairs,
            "reference": reference,
        },
        "curve": curve,
        "chosen": {
            "severity": choice["severity"],
            "status": choice["status"],
            "mean_drop": chosen_row["mean_drop"],
            "drop_count": chosen_row["drop_count"],
        },
        "created_by": dict(run_info),
    }


def write_calibration(log: Mapping[str, Any]) -> Path:
    """Write a calibration log next to its condition's dataset.json. An existing log is never overwritten."""
    condition = C.get_condition(str(log["condition"]))
    if not isinstance(condition, C.DrawnCondition):
        raise TypeError(f"{log['condition']} is not a drawn-parameter condition")
    path = C.calibration_path(condition)
    if path.exists():
        raise FileExistsError(f"{path} exists; a calibration is written once")
    store.write_json(path, log)
    return path
