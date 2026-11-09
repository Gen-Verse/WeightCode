"""PREREG D2, the program's structure trained by gradient (user 2026-11-05 about 13:50, main 3bc7838; a control, not
training-free, no LLM call).

The trainable entries are exactly the entries an E11 LEARN final changes on the seed-100 student: the final's derived
patch (saved by its program-transfer run on that student) is applied to the student by the shared executor, the
child is checked against the transfer record's ``child_sha256`` and ``n_changed``, and the mask is ``child != base``
per tensor. The mask is derived from the weights alone (label-free) and deterministically. Every other parameter is
frozen. Training is F1's full fine-tuning recipe: the references' rows (``load_joint_evidence``: the 64 condition rows,
then the 64 clean replay rows, with the references' labels), mean cross-entropy, full-batch Adam (0.9, 0.999, 1e-8),
no weight decay, eval-form forward, fp32, run seed 0, one trajectory per lr checkpointed at the grid's steps, F1's
full-FT grid lr {1e-5, 1e-4} x steps {10, 30, 100, 300}: 8 configurations, one joint check each, the joint oracle and
the kept rule of F1. The mask multiplies every gradient (``grad.adam_checkpoints``), so masked-out entries keep their
start values exactly.

* ``d2_struct_base``: start (i), the base student; the program's structure only.
* ``d2_struct_final``: start (ii), the LLM final's child; the structure and the program's values, refined.

Like F1's methods, neither is held to the E10 program limits; each checked candidate records whether it falls within
them. The final's own readout is its transfer record's (and DEV_RESULTS'), not a D2 candidate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256
from weightcode.patch import Limits, apply_patch, load_patch

from .ftcompare import F1_DATASET, FT_LRS
from .grad import ADAM_BETAS, ADAM_EPS, STEP_CHECKPOINTS, _check_frozen_entries, adam_checkpoints, evidence_loss
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import VITLITE, architecture_from, assert_parity, images_tensor, to_tensors

PREREG_D2 = "PREREG D2 (the program's structure trained by gradient, user 2026-11-05 about 13:50 Beijing, main 3bc7838)"
D2_LABEL = "d2"
"""Run-directory label of D2 runs: ``baselines/<instance_id>/<method>/d2/<seed>/``."""
TRANSFER_SCHEMA = "weightcode.harness.program_transfer.v0"
_OPEN_LIMITS = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None, max_ops=10**9, max_rank=10**9,
                      max_file_bytes=2**62)
"""Only to rebuild the final's child: the executor applies or refuses, it never alters a patch, and the child is
checked against the transfer record's hash."""

_TAIL = f"gradient-trained (not training-free), edit limits ignored; {PREREG_D2}"
D2_SPECS: dict[str, MethodSpec] = {
    "d2_struct_base": MethodSpec(
        method="d2_struct_base",
        label=("D2 (i): Adam from the base on exactly the entries the LLM final changes, condition and replay rows; "
               + _TAIL),
        uses_labels_for_fit=True, edit_limits_enforced=False),
    "d2_struct_final": MethodSpec(
        method="d2_struct_final",
        label=("D2 (ii): Adam from the LLM final's child on exactly the entries it changes, condition and replay rows; "
               + _TAIL),
        uses_labels_for_fit=True, edit_limits_enforced=False),
}
D2_METHODS: tuple[str, ...] = tuple(D2_SPECS)


class D2Error(ValueError):
    """A D2 run was refused: unknown method, a non-E11 instance, or a transfer record that does not match."""


def d2_configs() -> list[dict[str, Any]]:
    """The 8 configurations in check order: lr outer, then steps (F1's full-FT grid)."""
    return [{"lr": lr, "steps": steps} for lr in FT_LRS for steps in STEP_CHECKPOINTS]


def final_child(instance: Instance, transfer_record: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The LLM final's child on ``instance``, rebuilt from its program-transfer record and hash-checked."""
    record = json.loads(Path(transfer_record).read_text())
    if record.get("schema") != TRANSFER_SCHEMA or record.get("status") != "ok":
        raise D2Error(f"{transfer_record}: not an ok {TRANSFER_SCHEMA} record")
    target = record.get("target", {})
    if target.get("instance_id") != instance.instance_id or target.get("params_sha256") != instance.params_sha256:
        raise D2Error(f"{transfer_record}: its target {target.get('instance_id')} is not {instance.instance_id} "
                      "with the instance's weights")
    patch_path = Path(transfer_record).with_suffix("") / "patch" / "derived_patch.json"
    result = apply_patch(instance.params, load_patch(patch_path), patch_path.parent, _OPEN_LIMITS,
                         instance.params_sha256)
    if result.child_sha256 != record["child_sha256"] or result.n_changed != record["n_changed"]:
        raise D2Error(f"{patch_path}: rebuilt child {result.child_sha256[:12]} ({result.n_changed} changed) differs "
                      f"from the record's {str(record['child_sha256'])[:12]} ({record['n_changed']})")
    source = {"transfer_record": str(transfer_record), "patch": str(patch_path),
              "program_sha256": record.get("program", {}).get("sha256") if isinstance(record.get("program"), dict)
              else None, "child_sha256": result.child_sha256, "n_changed": result.n_changed}
    return {k: np.array(v) for k, v in result.child.items()}, source


def structure_mask(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]
                   ) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Trainable tensors (forward order of ``base``), their boolean masks (``child != base``) and a mask record."""
    trainable, masks, per_tensor = [], {}, {}
    for key, parent in base.items():
        changed = np.asarray(child[key]) != np.asarray(parent)
        if changed.any():
            trainable.append(key)
            masks[key] = changed
            per_tensor[key] = {"entries": int(changed.sum()), "size": int(changed.size)}
    total = int(sum(v["entries"] for v in per_tensor.values()))
    if total == 0:
        raise D2Error("the final changes no entry; D2 has nothing to train")
    record = {"n_entries": total, "n_tensors": len(trainable), "per_tensor": per_tensor,
              "mask_sha256": params_sha256({k: m.astype(np.uint8) for k, m in masks.items()}),
              "rule": "child != base per entry, child = the final's derived patch applied by the shared executor"}
    return trainable, masks, record


def d2_generator(instance: Instance, method: str, transfer_record: Path, run_seed: int) -> Generator:
    """Candidate generator of one D2 start on one E11 instance (module docstring)."""
    if method not in D2_SPECS:
        raise D2Error(f"unknown D2 method {method!r}; expected one of {D2_METHODS}")
    configs = d2_configs()

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        base = instance.params
        arch = architecture_from(base, read_model_spec(instance))
        if arch.family != VITLITE:
            raise D2Error(f"D2 runs on {VITLITE} students, not {arch.family}")
        if evidence.n_replay == 0:
            raise D2Error("D2 fits the condition and replay rows; the evidence has no replay rows")
        child, source = final_child(instance, transfer_record)
        trainable, masks, mask_record = structure_mask(base, child)
        start = base if method == "d2_struct_base" else child
        parity = assert_parity(arch, base, instance.forward, evidence.x)
        x = images_tensor(evidence.x)
        y = torch.from_numpy(evidence.y.astype(np.int64))
        info: dict[str, Any] = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity,
                                "fit_rows": f"{len(evidence.y)} rows ({evidence.composition()})", "prereg": PREREG_D2,
                                "start": "base" if method == "d2_struct_base" else "the LLM final's child",
                                "final": source, "mask": mask_record, "trainable": trainable}
        with torch.no_grad():
            info["start_evidence_loss"] = float(evidence_loss(arch, to_tensors(start), x, y))
        candidates: list[CandidatePatch] = []
        for lr in FT_LRS:
            steps = tuple(c["steps"] for c in configs if c["lr"] == lr)
            for step, params, loss in adam_checkpoints(arch, start, x, y, trainable=trainable, lr=lr,
                                                       checkpoints=steps, seed=run_seed, masks=masks):
                _check_frozen_entries(start, params, trainable, masks)
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=params, config={"lr": lr, "steps": step},
                    hypothesis=(f"{method}: Adam lr={lr:g} for {step} full-batch steps on the {mask_record['n_entries']} "
                                f"entries the LLM final changes ({len(evidence.y)} rows, {evidence.composition()}), "
                                f"from {info['start']}; {D2_SPECS[method].label}"),
                    predicted_effect=("fewer errors on held-out condition images if the fit generalizes within the "
                                      f"program's structure; retention is fitted on the {evidence.n_replay} clean "
                                      "replay rows"),
                    notes={"objective_at_checkpoint": {"ce": loss}}))
        if [c.config for c in candidates] != configs:
            raise RuntimeError(f"{method}: candidates {[c.config for c in candidates]} != configurations {configs}")
        return candidates, info

    return generate


def d2_config(method: str, transfer_record: Path, run_seed: int) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {
        "prereg": PREREG_D2, "candidate_order": d2_configs(), "transfer_record": str(transfer_record),
        "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
        "fit_rows": "the 64 condition rows, then the 64 clean replay rows", "batch": "full", "dtype": "float32",
        "seed": run_seed, "start": "base" if method == "d2_struct_base" else "final", "norm_limit": None,
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": "cpu",
    }


def run_d2(instance: Instance, method: str, transfer_record: Path, run_dir: Path, *, run_seed: int = 0) -> dict[str, Any]:
    """Run one D2 start on one E11 dev instance under F1's joint oracle and executor (without the size limits)."""
    if method not in D2_SPECS:
        raise D2Error(f"unknown D2 method {method!r}; expected one of {D2_METHODS}")
    if instance.private.get("dataset") != F1_DATASET:
        raise D2Error(f"D2 runs on {F1_DATASET} instances, got dataset {instance.private.get('dataset')!r}")
    return execute_run(
        instance=instance, spec=D2_SPECS[method], run_dir=run_dir, run_seed=run_seed,
        config=d2_config(method, transfer_record, run_seed),
        generate=d2_generator(instance, method, transfer_record, run_seed),
        allow_main=False, joint=True, replay_in_fit=True, e10=True,
    )


__all__ = ["D2Error", "D2_LABEL", "D2_METHODS", "D2_SPECS", "d2_config", "d2_configs", "d2_generator", "final_child",
           "run_d2", "structure_mask"]
