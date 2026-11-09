"""Arms as trial views (contract 5.4, HARNESS_SPEC sections 4 and 5).

Every arm sees the same model_spec.json, task text and query budget. Arms differ only in the weights shown, the
evidence shown, the model a patch is applied to, and the extra ``predict`` tool of NOWEIGHT. In every arm the
patch-check oracle's base is the real model, so S is always measured against the real base.

M2NL is the label-free M2: the real weights as in M2, evidence reduced to the random deployment sample without labels
(``labels=False``: no ``y_*`` array and no failure sample) and no ``patch_check`` or ``predict`` tool. Without
``patch_check`` its patch-check budget is zero, so contract 5.2 keeps the patch of its last valid declaration
(``weightcode.harness.finalize``).

LEARN (added 2026-11-04) delivers an update program instead of a patch (``program=True``): the real weights, apply mode
real, M2's tools and budgets, an ``experience.npz`` of new-condition and clean examples, and ``patch_check`` runs the
program and scores the joint S. Its deltas are held to limits measured on the change itself (``PROGRAM_LIMIT_FIELDS``,
PREREG section 0.1): M2's ``max_rel_frob`` on tensors of rank >= 2, and ``max_changed_frac`` =
``PROGRAM_MAX_CHANGED_FRAC`` of all entries. A program's dense deltas are counted per changed entry, which has no
equivalent of M2's information-counted ``max_edit_frac`` (M2's own CIFAR-10 noise update changes about 9% of the
entries through per-row scale ops), so the cap is the user's 20% bound for tasks that need new capacity (decision of
2026-11-04). ``max_edit_frac`` counts the numbers a patch specifies, and the derived patch's ``add_dense`` ops
specify whole tensors, so it is not applied to programs (``applied_limits``).

LEARN_BLIND (PREREG A4a, added 2026-11-05) is LEARN without the parameter values in the editor's view
(``blind=True``): the trial shows ``model_shapes.json`` (names, shapes and dtypes) in place of ``model.npz``, and its
``experience.npz`` holds the rows and labels without the student's predictions ``pred_new`` and ``pred_old`` (unless the
session keeps them, ``weightcode.harness.runner.run_session`` ``blind_predictions``), so no activation or output of the
student can be computed in the kernel. A program still runs on the real weights and on the LEARN experience with the
predictions (private copies outside every editor mount, ``weightcode.harness.trial``), so its inputs are those of a LEARN run.
A check returns the scores and the run's exit status and nothing the program prints, raises or writes
(``weightcode.harness.program.run_update`` ``blind``, ``weightcode.harness.tools``). Everything else is LEARN's: apply mode,
limits, budgets, tools and the kept rule.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weightcode.patch import Limits
from weightcode.patch.limits import SIZE_LIMIT_FIELDS, task_limits, with_dense_file_cap

ARMS = ("M2", "M1", "NOWEIGHT", "DECOY", "REWRITE", "M2NL", "LEARN", "LEARN_BLIND")

APPLY_REAL = "real"
APPLY_BY_INDEX_TO_REAL = "by_index_to_real"
APPLY_ZERO = "zero"
APPLY_MODES = (APPLY_REAL, APPLY_BY_INDEX_TO_REAL, APPLY_ZERO)

SHOWN_REAL = "real"
SHOWN_DECOY = "decoy"
SHOWN_ZEROS = "zeros"
SHOWN_SHAPES = "shapes"

PATCH_CHECK_TOOL = "patch_check"
BASE_TOOLS = ("python", PATCH_CHECK_TOOL, "status", "declare_final")
NO_CHECK_TOOLS = ("python", "status", "declare_final")
PREDICT_TOOL = "predict"
MCP_SERVER_NAME = "weightcode"
# The limits a program's deltas are checked against (on child - parent, exactly as on the patch path); every other
# size limit of Limits is set to None for program arms, and max_changed_frac takes M2's edit-size fraction when that is
# tighter. The format limits max_rank and max_file_bytes stay. The derived patch has one add_dense op per changed
# tensor, so its op count is bounded by the student's tensor count: given ``n_tensors``, ``applied_limits`` raises
# max_ops to it, so the format cap never binds before the stated limits (on the 70-tensor seqops student of PREREG E5
# the cap of 64 would have refused an update of more than 64 tensors that the task does not forbid; 2026-11-04). On a
# student of at most max_ops tensors (every E1 to E4 student) the limits are unchanged.
PROGRAM_LIMIT_FIELDS = ("max_rel_frob", "rel_frob_min_ndim", "max_changed_frac")
# Changed-entry cap of program arms (module docstring; PREREG section 0.1).
PROGRAM_MAX_CHANGED_FRAC = 0.20


@dataclass(frozen=True)
class ArmSpec:
    """What one arm shows and how its patches are applied.

    ``shown``: the weights in the trial view (real, decoy, zeros, or shapes only). ``evidence``: whether
    evidence.npz is shown. ``apply_mode``: the model a checked patch is applied to (session.json field).
    ``size_limits``: False disables the edit-size limits (REWRITE). ``tools``: the MCP tools of the arm.
    ``labels``: False shows only the random part of the evidence, without labels (M2NL). ``program``: the editor
    delivers an update program and ``patch_check`` runs it (LEARN). ``blind``: a program arm whose editor view holds no
    parameter value and no output of the student, and whose checks return no output of the program (LEARN_BLIND).
    """

    name: str
    shown: str
    evidence: bool
    apply_mode: str
    size_limits: bool
    tools: tuple[str, ...]
    labels: bool = True
    program: bool = False
    blind: bool = False

    @property
    def checks(self) -> bool:
        """Whether the arm has the patch-check oracle; without it the budget is zero and the declared patch is kept."""
        return PATCH_CHECK_TOOL in self.tools

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """Tool names as Claude Code sees them (``mcp__weightcode__<tool>``)."""
        return tuple(f"mcp__{MCP_SERVER_NAME}__{tool}" for tool in self.tools)


ARM_SPECS: dict[str, ArmSpec] = {
    "M2": ArmSpec("M2", SHOWN_REAL, True, APPLY_REAL, True, BASE_TOOLS),
    "M1": ArmSpec("M1", SHOWN_REAL, False, APPLY_REAL, True, BASE_TOOLS),
    "NOWEIGHT": ArmSpec("NOWEIGHT", SHOWN_SHAPES, True, APPLY_REAL, True, (*BASE_TOOLS, PREDICT_TOOL)),
    "DECOY": ArmSpec("DECOY", SHOWN_DECOY, True, APPLY_BY_INDEX_TO_REAL, True, BASE_TOOLS),
    "REWRITE": ArmSpec("REWRITE", SHOWN_ZEROS, True, APPLY_ZERO, False, BASE_TOOLS),
    "M2NL": ArmSpec("M2NL", SHOWN_REAL, True, APPLY_REAL, True, NO_CHECK_TOOLS, labels=False),
    "LEARN": ArmSpec("LEARN", SHOWN_REAL, True, APPLY_REAL, True, BASE_TOOLS, program=True),
    "LEARN_BLIND": ArmSpec("LEARN_BLIND", SHOWN_SHAPES, True, APPLY_REAL, True, BASE_TOOLS, program=True, blind=True),
}


def arm_spec(arm: str) -> ArmSpec:
    """The spec of one arm; an unknown arm name is an error."""
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}; arms: {', '.join(ARMS)}")
    return ARM_SPECS[arm]


def patch_check_budget(task_budget: int, arm: str) -> int:
    """The arm's patch-check budget: task.json's ``budget_queries``, or zero in an arm without ``patch_check``."""
    return int(task_budget) if arm_spec(arm).checks else 0


def applied_limits(task_limits: Mapping[str, Any], arm: str, n_tensors: int | None = None,
                   dense_bytes: int | None = None) -> Limits:
    """``Limits(**task.json["limits"])``, with the size limits disabled in REWRITE and, in a program arm, every size
    limit outside ``PROGRAM_LIMIT_FIELDS`` disabled and ``max_changed_frac`` set to ``PROGRAM_MAX_CHANGED_FRAC`` (see
    the module docstring).

    ``n_tensors``: the student's tensor count. In a program arm ``max_ops`` becomes ``max(max_ops, n_tensors)``, the
    most ops a derived patch can have (see ``PROGRAM_LIMIT_FIELDS``). Every caller that applies a program's update
    passes it; it changes nothing in a patch arm.

    ``dense_bytes``: ``weightcode.patch.limits.dense_delta_bytes`` of the student. In a program arm ``max_file_bytes`` becomes
    ``max(max_file_bytes, dense_bytes)`` (``with_dense_file_cap``), so the float64 derived patch of a full-model update
    is not refused by the format cap (PREREG E12's ResNet-18: about 89 MB against 64 MiB). It changes nothing on a
    student whose dense delta fits, and nothing in a patch arm.
    """
    if not isinstance(task_limits, Mapping):
        raise TypeError(f"task.json limits must be an object, got {type(task_limits).__name__}")
    if n_tensors is not None and (isinstance(n_tensors, bool) or not isinstance(n_tensors, int) or n_tensors < 1):
        raise ValueError(f"n_tensors must be an int >= 1, got {n_tensors!r}")
    limits = Limits(**dict(task_limits))
    spec = arm_spec(arm)
    if not spec.size_limits:
        return limits.without_size_limits()
    if spec.program:
        disabled = {name: None for name in SIZE_LIMIT_FIELDS if name not in PROGRAM_LIMIT_FIELDS}
        max_ops = limits.max_ops if n_tensors is None else max(limits.max_ops, n_tensors)
        return with_dense_file_cap(
            dataclasses.replace(limits, **disabled, max_changed_frac=PROGRAM_MAX_CHANGED_FRAC, max_ops=max_ops),
            dense_bytes)
    return limits


def limits_dict(limits: Limits) -> dict[str, Any]:
    """The limits as task.json states them (``weightcode.patch.task_limits``): every field, an empty exemption left out."""
    return task_limits(limits)
