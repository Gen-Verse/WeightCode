"""The one access point of the harness to clean companions and replay rows (LEARN arm, HARNESS_SPEC section 5).

Every evalkit import of the learning experiments lives here, so the dependency stays in one place:

- ``load_replay(dataset)``: the replay evidence of rule replay-v1, ``weightcode.evalkit.companions.load_replay`` (64 rows of
  the dataset's clean train split, drawn once per dataset, hash-checked; the train split is disjoint from V_evidence,
  so no replay row is the clean companion of an evidence row), with its draw record (``replay_v1.json``: split seed and
  stream, idx and row ids, content and pool hashes).
- ``load_clean_check(instance)``: the clean companion of the instance's V_check,
  ``weightcode.evalkit.companions.load_clean_rows(instance, "V_check")`` (both files hash-checked, rows matched).
- ``joint_oracle(...)``: evalkit's counted ``weightcode.evalkit.patch_check.JointPatchCheckOracle``; ``score`` is S_joint =
  score_new + score_old. The MCP server builds it from the session's private copies (V_check and its companion),
  because the server runs without the ``WEIGHTCODE_*`` variables that ``companions.make_joint_oracle`` needs to find the store.
- ``dev_readout_joint(instance, child, batch_size)``: ``weightcode.evalkit.devreadout.dev_readout_joint``, the readout on
  V_select of the condition (``readouts``) and on its clean companion (``readouts_clean``).
- ``preflight(instance)``: the replay and the V_check companion exist (a ``deploy`` instance has none and is refused),
  checked before a LEARN session writes anything.

No clean companion ever reaches a trial file: the experience holds only the replay rows. ``set_source`` installs a test
double for the three store-reading services (the toy store has only the ``deploy`` condition, which has no companions).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

COMPANIONS_MODULE = "weightcode.evalkit.companions"
EVALKIT_FUNCTIONS = (f"{COMPANIONS_MODULE}.load_replay", f"{COMPANIONS_MODULE}.load_clean_rows",
                     "weightcode.evalkit.patch_check.JointPatchCheckOracle", "weightcode.evalkit.devreadout.dev_readout_joint")
REPLAY_RECORD_KEYS = ("schema", "rule", "dataset", "pool_split", "pool_sha256", "split_seed", "stream", "n", "row_ids",
                      "idx", "content_sha256", "numpy_version")
REPLAY_ARRAY_KEYS = ("x", "y", "idx", "row_ids", "split_ids")
CLEAN_CHECK_KEYS = ("x", "y", "idx")


class CleanRowsUnavailable(RuntimeError):
    """A LEARN session, readout or transfer run cannot get its clean companions or replay rows."""


@dataclass(frozen=True)
class CleanSource:
    """The store-reading services of the module docstring; ``name`` is recorded in session.json."""

    name: str
    load_replay: Callable[[str], tuple[dict[str, np.ndarray], dict[str, Any]]]
    load_clean_check: Callable[[Any], tuple[dict[str, np.ndarray], dict[str, Any]]]
    dev_readout_joint: Callable[[Any, Mapping[str, np.ndarray], int], dict[str, Any]]


_installed: CleanSource | None = None


def _evalkit() -> tuple[Any, Any, Any]:
    try:
        from weightcode.evalkit import companions
        from weightcode.evalkit.devreadout import dev_readout_joint
        from weightcode.evalkit.patch_check import JointPatchCheckOracle
    except ImportError as exc:
        raise CleanRowsUnavailable(f"the LEARN arm needs {', '.join(EVALKIT_FUNCTIONS)}: {exc}") from None
    return companions, dev_readout_joint, JointPatchCheckOracle


def evalkit_source() -> CleanSource:
    """The evalkit implementation; ``CleanRowsUnavailable`` names a missing function."""
    companions, readout_joint, _ = _evalkit()

    def load_replay(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays = companions.load_replay(dataset)
        _, record_path = companions.replay_paths(dataset)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return arrays, {key: record.get(key) for key in REPLAY_RECORD_KEYS}

    def load_clean_check(instance: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        rows = companions.load_clean_rows(instance, "V_check")
        return rows.arrays(), {"clean_sha256": rows.clean_sha256, "condition_sha256": rows.condition_sha256}

    def readout(instance: Any, child: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
        return readout_joint(instance, child, batch_size)  # type: ignore[no-any-return]

    return CleanSource(name=", ".join(EVALKIT_FUNCTIONS), load_replay=load_replay, load_clean_check=load_clean_check,
                       dev_readout_joint=readout)


def set_source(source: CleanSource | None) -> CleanSource | None:
    """Install a test double (None restores evalkit); returns the previous one."""
    global _installed
    previous, _installed = _installed, source
    return previous


def source() -> CleanSource:
    return _installed if _installed is not None else evalkit_source()


def load_replay(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The replay arrays (``REPLAY_ARRAY_KEYS``) and their draw record; the arrays are checked for kind and length."""
    arrays, record = source().load_replay(dataset)
    missing = [key for key in REPLAY_ARRAY_KEYS if key not in arrays]
    if missing:
        raise CleanRowsUnavailable(f"the replay rows of {dataset} lack {missing}")
    n = len(arrays["y"])
    if n == 0 or any(len(arrays[key]) != n for key in REPLAY_ARRAY_KEYS):
        raise CleanRowsUnavailable(f"the replay arrays of {dataset} have unequal or zero lengths")
    if not np.issubdtype(arrays["x"].dtype, np.integer) or not np.issubdtype(arrays["y"].dtype, np.integer):
        raise CleanRowsUnavailable(f"the replay rows of {dataset} must have integer x and y")
    return {key: np.ascontiguousarray(arrays[key]) for key in REPLAY_ARRAY_KEYS}, dict(record)


def load_clean_check(instance: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The clean companion of the instance's V_check (``CLEAN_CHECK_KEYS``) and its hashes."""
    arrays, record = source().load_clean_check(instance)
    if any(key not in arrays for key in CLEAN_CHECK_KEYS) or len(arrays["x"]) != len(arrays["y"]):
        raise CleanRowsUnavailable(f"{instance.instance_id}: the clean V_check rows lack {CLEAN_CHECK_KEYS} or align badly")
    return {key: np.ascontiguousarray(arrays[key]) for key in CLEAN_CHECK_KEYS}, dict(record)


def joint_oracle(x_new: np.ndarray, y_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray, forward: Any,
                 base_params: Mapping[str, np.ndarray], budget: int) -> Any:
    """evalkit's ``JointPatchCheckOracle`` on the two parts of a joint check: the condition's V_check (new) and its
    clean companion (old), or, on a fewshot-v1 instance, the new-class and old-class rows of V_check."""
    _, _, oracle_class = _evalkit()
    return oracle_class(x_new, y_new, x_old, y_old, forward, base_params, budget)


def dev_readout_joint(instance: Any, child: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
    """Readouts of ``child`` on V_select of the condition and on its clean companion."""
    return source().dev_readout_joint(instance, child, batch_size)


def preflight(instance: Any) -> None:
    """Raise ``CleanRowsUnavailable`` unless a LEARN session on ``instance`` can get its replay and joint oracle."""
    try:
        load_replay(str(instance.private["dataset"]))
        load_clean_check(instance)
    except CleanRowsUnavailable:
        raise
    except Exception as exc:  # the evalkit refusal (CompanionError, a missing replay file) names the cause
        raise CleanRowsUnavailable(f"{instance.instance_id}: {type(exc).__name__}: {exc}") from None


def source_name() -> str:
    return source().name


__all__ = ["CLEAN_CHECK_KEYS", "CleanRowsUnavailable", "CleanSource", "EVALKIT_FUNCTIONS", "dev_readout_joint",
           "evalkit_source", "joint_oracle", "load_clean_check", "load_replay", "preflight", "set_source", "source",
           "source_name"]
