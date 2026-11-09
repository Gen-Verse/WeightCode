"""The harness side of rule fewshot-v1 (PREREG E2): the one access point of the editor harness to
``weightcode.evalkit.fewshot``, which owns the rule (the instances, their evidence, the class cut of V_check and the V_select
readout).

A fewshot instance runs in the LEARN arm only. Compared with an evidence-v1 LEARN session:

- the instance id is ``f`` + 8 hex and the instance lives under ``$WEIGHTCODE_PRIVATE/fewshot_instances``; the runner refuses
  it unless ``check_fewshot_current`` accepts it (active in its index, task.json and files as the current code writes
  them);
- the experience is the instance's evidence.npz itself, which already has the LEARN keys (the labelled examples of the
  new classes and the old-class replay rows), with every ``pred_*`` re-checked against the student;
- the check is V_check cut by class (``fewshot_check_parts``): S_new over the rows whose label is a new class, S_old
  over the rest. The session ledger cuts the session's V_check itself, from the class lists recorded in session.json
  (``session_record``), so the MCP server needs no store variable;
- the readout is ``fewshot_readout`` on V_select: accuracy on the new-class rows, the old-class rows and all rows,
  child against base (the PREREG E2 gate reads the all-class change).

There is no decoy, no corruption condition and no clean companion. ``weightcode.evalkit.fewshot`` is imported inside the
functions that need it, so an evidence-v1 session never loads it. (Until 2026-11-07 it imported torch at module level
through ``weightcode.models.export``; it now imports that module only inside its two student-store functions.)
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import params_sha256
from weightcode.common.paths import private_dir
from weightcode.evalkit.patch_check import predict

FEWSHOT_RULE = "fewshot-v1"  # weightcode.evalkit.fewshot.FEWSHOT_RULE; test_harness_fewshot checks that the two agree
INSTANCE_ID_RE = re.compile(r"^f[0-9a-f]{8}$")  # weightcode.evalkit.fewshot.INSTANCE_ID_RE, checked the same way
FEWSHOT_ROOT_NAME = "fewshot_instances"  # weightcode.evalkit.fewshot.fewshot_root
EXPERIENCE_SOURCE = "instance_fewshot_v1"
EXPERIENCE_KEYS = ("x_new", "y_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old")
PREDICT_BATCH = 1024
READOUT_PARTS = ("new", "old", "all")
# private.json fields the harness reads from a fewshot instance; an instance without one of them is refused.
REQUIRED_PRIVATE_FIELDS = (
    "instance_id",
    "split_role",
    "dataset",
    "arch",
    "model_id",
    "evidence_rule",
    "train_classes",
    "new_classes",
    "vcheck_path",
    "vcheck_sha256",
    "n_vcheck",
    "n_vcheck_new",
    "n_vcheck_old",
    "n_base_errors_vcheck_new",
    "n_base_errors_vcheck_old",
    "model_sha256",
    "forward_sha256",
    "task_sha256",
    "evidence_content_sha256",
    "model_spec_sha256",
)
# private.json fields that hold host paths; every one present is a forbidden string of the session.
PRIVATE_PATH_FIELDS = ("vcheck_path", "vselect_path", "vevidence_path", "replay_path", "model_source_path")


class FewshotHarnessError(RuntimeError):
    """A fewshot instance or session record is inconsistent, so no LEARN view, check or readout is built from it."""


def is_fewshot_id(instance_id: object) -> bool:
    """True for an id of the fewshot store's form (``f`` and 8 hex digits)."""
    return isinstance(instance_id, str) and INSTANCE_ID_RE.fullmatch(instance_id) is not None


def is_fewshot(private: Mapping[str, Any]) -> bool:
    """True when private.json says the instance follows rule fewshot-v1."""
    return private.get("evidence_rule") == FEWSHOT_RULE


def instance_dir(instance_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/fewshot_instances/<instance_id>``, the path of ``weightcode.evalkit.fewshot.fewshot_instance_dir``
    computed without creating the store root, so that resolving an id writes nothing; refuses an id of another form."""
    if not is_fewshot_id(instance_id):
        raise FewshotHarnessError(f"{instance_id!r} is not a fewshot instance id ('f' and 8 hex digits)")
    return private_dir() / FEWSHOT_ROOT_NAME / instance_id


def check_current(path: str | Path) -> dict[str, Any]:
    """``weightcode.evalkit.fewshot.check_fewshot_current``: the instance's task, or ``StaleFewshotInstance``."""
    from weightcode.evalkit.fewshot import check_fewshot_current

    return check_fewshot_current(path)


def stale_error() -> type[Exception]:
    """The evalkit exception for an instance that is not active or not current."""
    from weightcode.evalkit.fewshot import FewshotError

    return FewshotError


def active_index_row(instance_id: str) -> dict[str, Any] | None:
    """The instance's row among the active rows of the fewshot index, or None."""
    from weightcode.evalkit.fewshot import read_fewshot_instance_index

    rows = read_fewshot_instance_index()["instances"]
    found = [dict(row) for row in rows if isinstance(row, dict) and row.get("instance_id") == instance_id]
    return found[0] if found else None


def load_any(instance_id: str) -> Instance:
    """The instance with this id: a fewshot instance (``f`` + 8 hex) or an evidence-v1 instance (``i`` + 8 hex)."""
    if is_fewshot_id(instance_id):
        return load_instance(instance_dir(instance_id))
    from weightcode.evalkit.instances import instances_root

    return load_instance(instances_root() / instance_id)


def _class_list(private: Mapping[str, Any], key: str) -> list[int]:
    values = private.get(key)
    if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise FewshotHarnessError(f"private.json {key} must be a non-empty list of class indices, got {values!r}")
    return [int(v) for v in values]


def experience(instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The LEARN experience of a fewshot instance and its host-side record.

    The arrays are the instance's evidence under ``EXPERIENCE_KEYS``. ``pred_new`` and ``pred_old`` must be the
    student's predictions, ``y_new`` must hold every new class equally often and ``y_old`` only train classes.
    """
    private = instance.private
    if not is_fewshot(private):
        raise FewshotHarnessError(f"{instance.instance_id} follows {private.get('evidence_rule')!r}, not {FEWSHOT_RULE}")
    evidence = instance.evidence
    if sorted(evidence) != sorted(EXPERIENCE_KEYS):
        raise FewshotHarnessError(f"{instance.instance_id}: evidence keys {sorted(evidence)} are not the LEARN keys "
                                  f"{sorted(EXPERIENCE_KEYS)}")
    arrays = {key: np.ascontiguousarray(evidence[key]) for key in EXPERIENCE_KEYS}
    new_classes, train_classes = _class_list(private, "new_classes"), _class_list(private, "train_classes")
    if set(new_classes) & set(train_classes):
        raise FewshotHarnessError(f"new classes {new_classes} and train classes {train_classes} overlap")
    n_wrong = {}
    for part in ("new", "old"):
        x, y, shown = arrays[f"x_{part}"], arrays[f"y_{part}"], arrays[f"pred_{part}"]
        if not (len(x) == len(y) == len(shown) == len(arrays[f"split_ids_{part}"])) or len(y) == 0:
            raise FewshotHarnessError(f"{instance.instance_id}: the {part} part has unequal or zero lengths")
        pred = predict(instance.forward, instance.params, x, PREDICT_BATCH)
        if not np.array_equal(pred, shown):
            raise FewshotHarnessError(f"{instance.instance_id}: pred_{part} disagrees with the student on "
                                      f"{int(np.count_nonzero(pred != shown))} of {len(y)} rows")
        n_wrong[part] = int(np.count_nonzero(pred != y))
    labels, counts = np.unique(arrays["y_new"], return_counts=True)
    if sorted(int(label) for label in labels) != sorted(new_classes) or len(set(counts.tolist())) != 1:
        raise FewshotHarnessError(f"{instance.instance_id}: y_new holds {dict(zip(labels.tolist(), counts.tolist()))}; "
                                  f"the new classes are {new_classes}, each equally often")
    if not np.isin(arrays["y_old"], train_classes).all():
        raise FewshotHarnessError(f"{instance.instance_id}: y_old holds a class outside the train classes")
    ids = [*arrays["split_ids_new"].tolist(), *arrays["split_ids_old"].tolist()]
    if len(set(ids)) != len(ids):
        raise FewshotHarnessError(f"{instance.instance_id}: the split ids are not unique")
    record = {
        "file": "experience.npz",
        "source": EXPERIENCE_SOURCE,
        "rule": FEWSHOT_RULE,
        "content_sha256": params_sha256(arrays),
        "evidence_content_sha256": private["evidence_content_sha256"],
        "student_params_sha256": instance.params_sha256,
        "n_new": int(len(arrays["y_new"])),
        "n_new_fail": n_wrong["new"],
        "n_old": int(len(arrays["y_old"])),
        "n_old_fail": n_wrong["old"],
        "per_class": int(counts[0]),
        "new_classes": new_classes,
        "train_classes": train_classes,
    }
    if record["content_sha256"] != private["evidence_content_sha256"]:
        raise FewshotHarnessError(f"{instance.instance_id}: the experience differs from the recorded evidence")
    return arrays, record


def session_record(private: Mapping[str, Any]) -> dict[str, Any]:
    """The ``fewshot`` section of session.json: what the ledger needs to cut V_check, and the base error counts."""
    missing = [name for name in REQUIRED_PRIVATE_FIELDS if name not in private]
    if missing:
        raise FewshotHarnessError(f"private.json lacks {missing}")
    return {
        "rule": FEWSHOT_RULE,
        "train_classes": _class_list(private, "train_classes"),
        "new_classes": _class_list(private, "new_classes"),
        "n_vcheck_new": int(private["n_vcheck_new"]),
        "n_vcheck_old": int(private["n_vcheck_old"]),
        "n_base_errors_vcheck_new": int(private["n_base_errors_vcheck_new"]),
        "n_base_errors_vcheck_old": int(private["n_base_errors_vcheck_old"]),
    }


def check_parts(x: np.ndarray, y: np.ndarray, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """V_check cut by class (``weightcode.evalkit.fewshot.fewshot_check_parts``), with the part sizes of ``record``."""
    from weightcode.evalkit.fewshot import fewshot_check_parts

    if record.get("rule") != FEWSHOT_RULE:
        raise FewshotHarnessError(f"the session's fewshot record names rule {record.get('rule')!r}")
    parts = fewshot_check_parts({"x": x, "y": y}, [int(c) for c in record["new_classes"]])
    sizes = (len(parts["y_new"]), len(parts["y_old"]))
    if sizes != (int(record["n_vcheck_new"]), int(record["n_vcheck_old"])):
        raise FewshotHarnessError(f"V_check cuts into {sizes} new and old rows; the instance records "
                                  f"{(record['n_vcheck_new'], record['n_vcheck_old'])}")
    if np.isin(parts["y_old"], [int(c) for c in record["train_classes"]], invert=True).any():
        raise FewshotHarnessError("an old-part V_check row has a label outside the train classes")
    return parts


def readout(instance: Instance, child: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
    """``fewshot_readout`` of ``child`` on V_select (``readouts_fewshot``) and, per part (new classes, old classes,
    all), the change in the fields of a dev readout: n, base_acc, child_acc, delta_acc, fixed and broken."""
    from weightcode.evalkit.fewshot import fewshot_readout

    full = fewshot_readout(instance, dict(child), batch_size)
    parts = {part: {"n": full[f"n_{part}"], "base_acc": full[f"acc_{part}_base"],
                    "child_acc": full[f"acc_{part}_child"], "delta_acc": full[f"delta_{part}"],
                    "fixed": full[f"fixed_{part}"], "broken": full[f"broken_{part}"]} for part in READOUT_PARTS}
    return {"readouts_fewshot": full, "parts": parts}


def private_paths(private: Mapping[str, Any]) -> list[str]:
    """The host paths recorded in a fewshot private.json (``PRIVATE_PATH_FIELDS``)."""
    return [str(private[name]) for name in PRIVATE_PATH_FIELDS if private.get(name)]


__all__ = [
    "EXPERIENCE_KEYS", "EXPERIENCE_SOURCE", "FEWSHOT_ROOT_NAME", "FEWSHOT_RULE", "FewshotHarnessError", "INSTANCE_ID_RE",
    "PRIVATE_PATH_FIELDS", "READOUT_PARTS", "REQUIRED_PRIVATE_FIELDS", "active_index_row", "check_current",
    "check_parts", "experience", "instance_dir", "is_fewshot", "is_fewshot_id", "load_any",
    "private_paths", "readout", "session_record", "stale_error",
]
