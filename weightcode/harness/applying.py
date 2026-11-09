"""Arm rules for applying a patch (HARNESS_SPEC section 4), shared by the tool server and finalisation.

M2, M1 and NOWEIGHT apply to the real model with the real hash; DECOY checks the patch against the decoy hash and
applies it to the real model by index; REWRITE applies it to the zero model with the size limits disabled. The
patch-check oracle's base is always the real model.

Editor-facing rejections (review finding A, 2026-11-03). Invalid patches cost no query, so a rejection message must not
carry any number computed from the parameter values the patch was applied to: in NOWEIGHT and DECOY those are values
the editor is not shown. ``ArmApplier.apply_for_editor`` therefore raises ``EditorRejection``, whose ``messages`` are
the executor's messages verbatim only when the same patch applied to an all-zero model of the same shapes gives exactly
the same messages (then they are a function of the shapes, the patch and its files alone: schema, references, indices,
file problems, edit size). Otherwise the rejection came from a check on the patched values, and the editor is told which
limit failed and its threshold, never the measured value: ``tensor 'w': the relative Frobenius norm of the change is
above the limit 0.5``, ``size: the fraction of parameters that change is above the limit 0.01``, or ``tensor 'w': some
entries are not finite after the patch``. ``executor_messages`` keeps the executor's full text for the logs. The rule is
the same in every arm.

A blind program arm (LEARN_BLIND, PREREG A4a) goes one step further, since there the deltas come from a program that
reads the weights the editor is not shown, so which tensors fail a limit, and any count in a verbatim message, could
carry those values. ``blind_rejection_messages`` reduces the editor text to the kinds of failure, once each in a fixed
order, each naming its limit and threshold and never a tensor or a count.

The protected-entry limit (PREREG T16 follow-up M, ``weightcode.harness.loop``): in a loop round whose session.json ``loop``
states ``protect_logged``, ``apply`` also refuses, with ``ProtectedEntryChange``, a child that passed every other
limit but changes an entry the round's edit log lists, compared with the round's student (the real model). Every
check, replay and final goes through ``apply``, so the limit binds wherever the others do. The editor is shown the
refusal as it is (``REDACTION_PROTECTED``): how many listed entries changed, and in which tensors. A loop round runs in
the LEARN arm, whose editor has the student's weights and its own program's deltas, so those counts carry nothing
the editor cannot compute. A session without the key applies no such limit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.harness import loop
from weightcode.harness.arms import APPLY_BY_INDEX_TO_REAL, APPLY_REAL, APPLY_ZERO
from weightcode.patch import Limits, PatchError, PatchResult, apply_patch

REDACTION_NONE = "verbatim"
REDACTION_LIMITS = "limits"
REDACTION_NON_FINITE = "non_finite"
REDACTION_GENERIC = "generic"
REDACTION_PROTECTED = "protected"  # PREREG T16 follow-up M: the protected-entry limit's refusal, shown as it is
NON_FINITE_TEXT = ("some entries are not finite after the patch (float64 overflow or invalid arithmetic, or values "
                   "beyond the float32 range)")
GENERIC_TEXT = "patch: the patched model fails a check on its parameter values"
REDACTION_BLIND = "blind"
BLIND_OTHER_TEXT = "patch: the update fails a limit of task.json"


def blind_rejection_messages(messages: list[str], limits: Limits) -> list[str]:
    """The editor text of a rejection in a blind program arm (module docstring): ``messages`` (the editor or executor
    text) reduced to one fixed line per kind, in the order size, rel-Frob, non-finite, other."""
    kinds = set()
    for text in messages:
        if "relative Frobenius norm" in text:
            kinds.add("rel_frob")
        elif "not finite" in text:
            kinds.add("non_finite")
        elif text.startswith("size:") and "parameters" in text and "change" in text:
            kinds.add("size")
        else:
            kinds.add("other")
    shown = []
    if "size" in kinds:
        shown.append(f"size: the fraction of parameters that change is above the limit {limits.max_changed_frac!r}")
    if "rel_frob" in kinds:
        shown.append(f"a tensor: the relative Frobenius norm of the change is above the limit {limits.max_rel_frob!r} "
                     "(the program has the weights and can measure it)")
    if "non_finite" in kinds:
        shown.append(f"a tensor: {NON_FINITE_TEXT}")
    if "other" in kinds:
        shown.append(BLIND_OTHER_TEXT)
    return shown


class SessionConfigError(RuntimeError):
    """session.json disagrees with the files it points to."""


class EditorRejection(PatchError):
    """A rejected patch: ``messages`` are safe to show the editor, ``executor_messages`` are the executor's own."""

    def __init__(self, messages: list[str], executor_messages: list[str], redaction: str) -> None:
        super().__init__(messages)
        self.executor_messages = list(executor_messages)
        self.redaction = redaction

    def __reduce__(self) -> tuple[type[EditorRejection], tuple[list[str], list[str], str]]:  # type: ignore[override]
        return (type(self), (list(self.messages), list(self.executor_messages), self.redaction))


class ProtectedEntryChange(PatchError):
    """A child that changes an entry the round's edit log lists (the protected-entry limit, module docstring)."""


def _rel_frob_violations(target: Mapping[str, np.ndarray], result: PatchResult, limits: Limits) -> list[str]:
    if limits.max_rel_frob is None:
        return []
    return [
        f"tensor {name!r}: the relative Frobenius norm of the change is above the limit {limits.max_rel_frob!r}"
        for name, stats in result.per_tensor.items()
        if limits.rel_frob_applies(name, target[name].ndim) and stats["rel_frob"] > limits.max_rel_frob
    ]


class ArmApplier:
    """Applies patches under one session's arm rules; holds the real parameters (the oracle base)."""

    def __init__(self, session: Mapping[str, Any]) -> None:
        self.apply_mode = str(session["apply_mode"])
        self.limits = Limits(**dict(session["limits"]))
        self.real_sha256 = str(session["real_model_sha256"])
        self.expected_parent_sha256 = str(session["shown_model_sha256"])
        real = load_params(Path(str(session["real_model_path"])))
        if params_sha256(real) != self.real_sha256:
            raise SessionConfigError("the real model file does not hash to session.json real_model_sha256")
        self.real: Params = real
        zeros = {key: np.zeros_like(value) for key, value in real.items()}
        if self.apply_mode == APPLY_REAL:
            if self.expected_parent_sha256 != self.real_sha256:
                raise SessionConfigError("apply mode 'real' needs the shown model to be the real model")
            self._target: Params = real
        elif self.apply_mode == APPLY_BY_INDEX_TO_REAL:
            if self.expected_parent_sha256 != session.get("decoy_model_sha256"):
                raise SessionConfigError("apply mode 'by_index_to_real' needs the shown model to be the decoy")
            if self.expected_parent_sha256 == self.real_sha256:
                raise SessionConfigError("the decoy hash equals the real hash")
            self._target = real
        elif self.apply_mode == APPLY_ZERO:
            if params_sha256(zeros) != self.expected_parent_sha256:
                raise SessionConfigError("apply mode 'zero' needs the shown model to be the zero model")
            self._target = zeros
        else:
            raise SessionConfigError(f"unknown apply mode {self.apply_mode!r}")
        self._zeros: Params = zeros
        self.protected: dict[str, np.ndarray] | None = None  # the protected-entry limit (module docstring), or None
        if "protect_logged" in (session.get("loop") or {}):
            if self.apply_mode != APPLY_REAL:
                raise SessionConfigError("the protected-entry limit is a loop round's, which applies to the real model")
            try:
                self.protected = loop.session_protected_entries(session, {k: v.shape for k, v in real.items()})
            except (loop.LoopError, OSError) as exc:
                raise SessionConfigError(f"the protected-entry limit cannot be set up: {exc}") from None

    def apply(self, patch: Mapping[str, Any], patch_dir: Path) -> PatchResult:
        """Validate and apply one patch under the arm rules; raises ``PatchError`` with the executor's messages, or
        ``ProtectedEntryChange`` for a child that passes them and changes a protected entry."""
        result = apply_patch(self._target, patch, patch_dir, self.limits, self.expected_parent_sha256)
        if self.protected is not None:
            changes = loop.protected_changes(self.protected, self._target, result.child)
            if changes:
                raise ProtectedEntryChange(loop.protect_messages(changes))
        return result

    def apply_for_editor(self, patch: Mapping[str, Any], patch_dir: Path) -> PatchResult:
        """``apply``, but a rejection raises ``EditorRejection`` (see the module docstring)."""
        try:
            return self.apply(patch, patch_dir)
        except ProtectedEntryChange as problem:  # shown as it is (module docstring)
            raise EditorRejection(list(problem.messages), list(problem.messages), REDACTION_PROTECTED) from None
        except PatchError as problem:
            executor = list(problem.messages)
        messages, redaction = self.editor_messages(patch, patch_dir, executor)
        raise EditorRejection(messages, executor, redaction)

    def editor_messages(self, patch: Mapping[str, Any], patch_dir: Path, executor: list[str]) -> tuple[list[str], str]:
        """The editor-facing text of the executor's rejection ``executor`` and the redaction applied to it."""
        try:
            apply_patch(self._zeros, patch, patch_dir, self.limits, self.expected_parent_sha256)
        except PatchError as twin:
            if list(twin.messages) == executor:
                return executor, REDACTION_NONE
        relaxed = dataclasses.replace(self.limits, max_rel_frob=None, max_changed_frac=None)
        try:
            result = apply_patch(self._target, patch, patch_dir, relaxed, self.expected_parent_sha256)
        except PatchError as again:  # with those limits off, only the finiteness check of the patched values remains
            names = [name for name in self._target if any(m.startswith(f"tensor {name!r}: ") for m in again.messages)]
            if not names:
                return [GENERIC_TEXT], REDACTION_GENERIC
            return [f"tensor {name!r}: {NON_FINITE_TEXT}" for name in names], REDACTION_NON_FINITE
        messages = []
        if self.limits.max_changed_frac is not None and result.changed_frac > self.limits.max_changed_frac:
            messages.append(f"size: the fraction of parameters that change is above the limit "
                            f"{self.limits.max_changed_frac!r}")
        messages.extend(_rel_frob_violations(self._target, result, self.limits))
        if not messages:
            return [GENERIC_TEXT], REDACTION_GENERIC
        return messages, REDACTION_LIMITS


__all__ = ["ArmApplier", "BLIND_OTHER_TEXT", "EditorRejection", "GENERIC_TEXT", "NON_FINITE_TEXT",
           "ProtectedEntryChange", "REDACTION_BLIND", "REDACTION_GENERIC", "REDACTION_LIMITS", "REDACTION_NONE",
           "REDACTION_NON_FINITE", "REDACTION_PROTECTED", "SessionConfigError", "blind_rejection_messages"]
