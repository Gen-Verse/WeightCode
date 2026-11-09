"""Evidence of rule evidence-v1 drawn for any student on an instance's deployment condition (PREREG E4 rounds).

``draw_evidence_for(instance, params)`` applies rule evidence-v1 (``weightcode.evalkit.evidence.draw_evidence_v1``) to
``params`` in place of the instance's base model. It uses the instance's own V_evidence pool, refused unless its
content hash equals the one the instance recorded, and the instance's forward. It draws K_FAIL rows from all of THIS
student's errors on V_evidence and K_RAND rows from the other rows, with the generator the rule seeds from the
student's parameter hash and the condition (``evidence_seed_v1``).

The draw depends on exactly four things: the student's parameter hash, the condition name, V_evidence (x and y) and
the student's predictions on V_evidence. It does not depend on the base model, the decoy, V_check or any earlier
draw. Two consequences follow:

- Applied to the instance's base params, it reproduces the instance's evidence.npz exactly.
- Applied to another model, it gives the evidence that model's own evidence-v1 instance on the condition holds.
  ``decoy_evidence`` in ``weightcode.evalkit.instances`` relies on the same identity.

Identical params give identical evidence. A student with fewer than K_FAIL errors on V_evidence has no evidence-v1
draw, and the rule raises ValueError.

The result holds the arrays of an evidence.npz (``EVIDENCE_KEYS``), so a program reads it as it reads an instance's
evidence. It also carries the V_evidence rows of the draw (``row_ids``) and their dataset indices (``source_idx``).
Those two are private, like the instance's private.json, and never reach an editor or a program.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import Params, params_sha256
from weightcode.evalkit.evidence import (
    EVIDENCE_RULE_V1,
    K_RAND,
    draw_evidence_v1,
    evidence_content_sha256,
    evidence_seed_v1,
)
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.targets import K_FAIL

EVIDENCE_BATCH = 1024
"""Forward batch size, the default of ``scripts/build_instances.py``, so a base-params draw matches the build."""


class StudentEvidenceError(ValueError):
    """The draw was refused: not an evidence-v1 instance, a changed V_evidence pool or params of another layout."""


@dataclass(frozen=True)
class StudentEvidence:
    """Evidence-v1 of one student on one instance's condition, with the private record of the draw."""

    instance_id: str
    condition: str
    params_sha256: str
    seed: int
    arrays: dict[str, np.ndarray]
    row_ids: np.ndarray
    source_idx: np.ndarray
    n_errors_vevidence: int
    content_sha256: str

    @property
    def n_fail(self) -> int:
        return int(len(self.arrays["y_fail"]))

    @property
    def n_rand(self) -> int:
        return int(len(self.arrays["y_rand"]))

    @property
    def x(self) -> np.ndarray:
        """Images in file order: the fail rows, then the random rows."""
        return np.concatenate([self.arrays["x_fail"], self.arrays["x_rand"]])

    @property
    def y(self) -> np.ndarray:
        return np.concatenate([self.arrays["y_fail"], self.arrays["y_rand"]])

    @property
    def pred(self) -> np.ndarray:
        """The student's predictions on those rows."""
        return np.concatenate([self.arrays["pred_fail"], self.arrays["pred_rand"]])

    @property
    def split_ids(self) -> list[str]:
        return [str(value) for value in self.arrays["split_ids"]]


def _resolve(instance: Instance | str) -> Instance:
    return instance if isinstance(instance, Instance) else load_instance(instances_root() / instance)


def _check_layout(base: Params, params: Params) -> None:
    if set(base) != set(params):
        raise StudentEvidenceError(f"params keys {sorted(params)} differ from the instance model's {sorted(base)}")
    for key, value in base.items():
        other = np.asarray(params[key])
        if other.shape != value.shape or other.dtype != value.dtype:
            raise StudentEvidenceError(
                f"{key} is {other.dtype}{other.shape}, the instance model's is {value.dtype}{value.shape}"
            )


def load_vevidence(instance: Instance) -> dict[str, np.ndarray]:
    """The instance's V_evidence, refused unless its content hash equals the one the instance recorded."""
    path = Path(str(instance.private["vevidence_path"]))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    recorded = instance.private.get("vevidence_content_sha256")
    if recorded is None or params_sha256(arrays) != recorded:
        raise StudentEvidenceError(f"{instance.instance_id}: V_evidence content hash differs from the instance record")
    return arrays


def draw_evidence_for(
    instance: Instance | str, params: Params, batch_size: int = EVIDENCE_BATCH
) -> StudentEvidence:
    """Rule evidence-v1 for ``params`` on the condition of ``instance`` (an Instance or an instance id)."""
    resolved = _resolve(instance)
    if resolved.private.get("evidence_rule") != EVIDENCE_RULE_V1:
        raise StudentEvidenceError(
            f"{resolved.instance_id} follows {resolved.private.get('evidence_rule')!r}; the draw is defined for "
            f"{EVIDENCE_RULE_V1} instances only"
        )
    _check_layout(resolved.params, params)
    v_ev = load_vevidence(resolved)
    pred = predict(resolved.forward, params, v_ev["x"], batch_size)
    digest = params_sha256(params)
    condition = str(resolved.private["condition"])
    seed = evidence_seed_v1(digest, condition)
    evidence = draw_evidence_v1(v_ev["x"], v_ev["y"], pred, seed=seed, k_fail=K_FAIL, k_rand=K_RAND)
    return StudentEvidence(
        instance_id=resolved.instance_id,
        condition=condition,
        params_sha256=digest,
        seed=evidence.seed,
        arrays=evidence.arrays,
        row_ids=evidence.row_ids,
        source_idx=np.asarray(v_ev["idx"])[evidence.row_ids].astype(np.int64),
        n_errors_vevidence=int(np.count_nonzero(pred != v_ev["y"])),
        content_sha256=evidence_content_sha256(evidence.arrays),
    )
