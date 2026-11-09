"""Instance selection and the checks that run before any session file is written (HARNESS_SPEC sections 1, 5, 10).

The harness runs evidence-v1 instances: one (dataset, deployment condition, model), no target pair. A v0
instance, a superseded instance and, unless explicitly allowed, a main instance are refused here, before the
runner creates anything. Every read below is side-effect free: no directory is created and no file is written.

It also runs fewshot-v1 instances (PREREG E2, ``weightcode.harness.fewshot``): an id of the form ``f`` + 8 hex resolves in
the fewshot store and is refused unless ``check_fewshot_current`` accepts it; the same main-instance guard applies.
Likewise seqskill-v1 instances (PREREG E5, ``weightcode.harness.seqskill``): an id of the form ``q`` + 8 hex resolves in the
seqskill store and is refused unless ``check_seqskill_current`` accepts it. And graphnew-v1 instances (PREREG E7,
``weightcode.harness.graphnew``): an id of the form ``g`` + 8 hex resolves in the graphnew store and is refused unless
``check_graphnew_current`` accepts it. And fscil-v1 instances (PREREG E9, ``weightcode.harness.fscil``): an id of the form
``c`` + 8 hex resolves in the fscil store and is refused unless ``check_fscil_current`` accepts it. And
policyskill-v1 instances (PREREG E14, ``weightcode.harness.policyskill``): an id of the form ``p`` + 8 hex resolves in the
E14 store and is refused unless ``check_e14_current`` accepts it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weightcode.common.paths import private_dir
from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.evalkit.instances import (
    INSTANCE_ID_RE,
    PRIVATE_SCHEMA_V1,
    TASK_SCHEMA,
    TASK_SCHEMA_V1,
    StaleInstanceError,
    check_task_current,
    read_index,
)
from weightcode.harness import fewshot, fscil, graphnew, policyskill, seqskill
from weightcode.harness.jsonio import read_json
from weightcode.patch import OP_NAMES, task_limits

SPLIT_ROLE_DEV = "dev"
SPLIT_ROLE_MAIN = "main"

# private.json fields the harness reads (HARNESS_SPEC section 1); a v1 instance without one of them is refused.
REQUIRED_PRIVATE_FIELDS = (
    "split_role",
    "condition",
    "dataset",
    "model_id",
    "decoy_model_id",
    "vcheck_path",
    "vcheck_sha256",
    "decoy_model_path",
    "decoy_model_sha256",
    "model_sha256",
    "forward_sha256",
    "n_vcheck",
    "n_base_errors_vcheck",
    "task_sha256",
    "evidence_content_sha256",
    "model_spec_sha256",
)


class InstanceRefused(ValueError):
    """The instance may not be run by the editor harness (v0, superseded, main without permission, malformed)."""


@dataclass(frozen=True)
class InstanceRecord:
    """task.json, private.json and the active index row of one evidence-v1 instance."""

    instance_id: str
    instance_dir: Path
    task: dict[str, Any]
    private: dict[str, Any]
    index_row: dict[str, Any]

    @property
    def split_role(self) -> str:
        return str(self.private["split_role"])

    @property
    def is_fewshot(self) -> bool:
        """True for a fewshot-v1 instance (PREREG E2)."""
        return fewshot.is_fewshot(self.private)

    @property
    def is_seqskill(self) -> bool:
        """True for a seqskill-v1 instance (PREREG E5)."""
        return seqskill.is_seqskill(self.private)

    @property
    def is_graphnew(self) -> bool:
        """True for a graphnew-v1 instance (PREREG E7)."""
        return graphnew.is_graphnew(self.private)

    @property
    def is_fscil(self) -> bool:
        """True for an fscil-v1 instance (PREREG E9)."""
        return fscil.is_fscil(self.private)

    @property
    def is_policyskill(self) -> bool:
        """True for a policyskill-v1 instance (PREREG E14)."""
        return policyskill.is_policyskill(self.private)

    @property
    def has_own_experience(self) -> bool:
        """True for an instance whose evidence file is the LEARN experience and whose check is cut into the two
        check parts (fewshot-v1, seqskill-v1, graphnew-v1, fscil-v1, policyskill-v1): no decoy, no clean companion,
        no replay draw, program arms only."""
        return self.is_fewshot or self.is_seqskill or self.is_graphnew or self.is_fscil or self.is_policyskill


def instance_dir_for(instance_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/instances/<instance_id>``; the id must have the store's form, so no path can escape."""
    if not isinstance(instance_id, str) or INSTANCE_ID_RE.fullmatch(instance_id) is None:
        raise InstanceRefused(f"{instance_id!r} is not an instance id (expected 'i' and 8 hex digits)")
    root = private_dir() / "instances" / instance_id
    if not root.is_dir():
        raise InstanceRefused(f"instance {instance_id} does not exist under $WEIGHTCODE_PRIVATE/instances")
    return root


def refuse_v0(instance_id: str, task: Mapping[str, Any], private: Mapping[str, Any]) -> None:
    """Refuse an evidence-v0 (target-pair) instance with a message that says why."""
    reasons = []
    if task.get("schema") == TASK_SCHEMA:
        reasons.append(f"task.json has schema {TASK_SCHEMA}")
    if "target" in private:
        reasons.append("private.json has a target pair")
    if reasons:
        raise InstanceRefused(
            f"instance {instance_id} is an evidence-v0 target-pair instance ({'; '.join(reasons)}). The editor "
            f"harness runs evidence-v1 instances only (task schema {TASK_SCHEMA_V1}, no target pair); v0 instances "
            "are superseded (HARNESS_SPEC section 10)."
        )
    problems = []
    if task.get("schema") != TASK_SCHEMA_V1:
        problems.append(f"task.json schema is {task.get('schema')!r}, not {TASK_SCHEMA_V1}")
    if private.get("schema") != PRIVATE_SCHEMA_V1:
        problems.append(f"private.json schema is {private.get('schema')!r}, not {PRIVATE_SCHEMA_V1}")
    if private.get("evidence_rule") != EVIDENCE_RULE_V1:
        problems.append(f"private.json evidence_rule is {private.get('evidence_rule')!r}, not {EVIDENCE_RULE_V1}")
    if problems:
        raise InstanceRefused(f"instance {instance_id} is not an evidence-v1 instance: {'; '.join(problems)}")


def _same_value(left: object, right: object) -> bool:
    """Equality of two JSON limit values that never equates a bool with a number or None with anything else."""
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def task_current_problems(task: Mapping[str, Any]) -> list[str]:
    """Differences between a task.json snapshot and the executor now: ``limits`` against
    ``weightcode.patch.task_limits()`` and ``patch_format.allowed_ops`` against ``weightcode.patch.OP_NAMES``.

    task.json is written when the instance is built, so a later change of the limits or the op set would otherwise
    go unnoticed. One message per differing field. The decision belongs to the shared
    ``weightcode.evalkit.instances.check_task_current`` (``refuse_stale``); this function only names the differing fields
    in the refusal message.
    """
    problems: list[str] = []
    expected = task_limits()
    actual = task.get("limits")
    if not isinstance(actual, Mapping):
        problems.append(f"limits: task.json has {type(actual).__name__}, expected an object")
    else:
        for name in sorted(set(expected) | set(actual)):
            if name not in actual:
                problems.append(f"limits.{name}: missing from task.json; Limits() has {expected[name]!r}")
            elif name not in expected:
                problems.append(f"limits.{name}: task.json has {actual[name]!r}; Limits() has no such field")
            elif not _same_value(actual[name], expected[name]):
                problems.append(f"limits.{name}: task.json has {actual[name]!r}; Limits() has {expected[name]!r}")
    patch_format = task.get("patch_format")
    ops = patch_format.get("allowed_ops") if isinstance(patch_format, Mapping) else None
    if not isinstance(ops, list):
        problems.append("patch_format.allowed_ops: missing from task.json or not a list")
    elif ops != list(OP_NAMES):
        missing = [op for op in OP_NAMES if op not in ops]
        extra = [op for op in ops if op not in OP_NAMES]
        detail = []
        if missing:
            detail.append(f"executor ops missing from task.json: {missing}")
        if extra:
            detail.append(f"task.json ops the executor lacks: {extra}")
        if not detail:
            detail.append(f"same ops in a different order: task.json {ops}, executor {list(OP_NAMES)}")
        problems.append("patch_format.allowed_ops: " + "; ".join(detail))
    return problems


def refuse_stale(instance_id: str, instance_dir: Path, task: Mapping[str, Any]) -> None:
    """``weightcode.evalkit.instances.check_task_current``: refuse an instance that is superseded, missing from the active
    index, not evidence-v1, or whose task.json is not what the current code writes (limits, op list, task text).

    The refusal names the id and the reason; for a superseded instance it adds when and why it was superseded, and
    for a stale task.json every differing limit field and the op-list difference.
    """
    try:
        check_task_current(instance_dir)
    except StaleInstanceError as exc:
        details = []
        retired = {row.get("instance_id"): row for row in read_index().get("superseded", []) if isinstance(row, dict)}
        if instance_id in retired:
            row = retired[instance_id]
            details.append(f"superseded at {row.get('superseded_at')!r}, reason: {row.get('superseded_reason')!r}")
        problems = task_current_problems(task)
        if problems:
            details.append(f"{len(problems)} differing fields: " + "; ".join(problems))
        suffix = f" ({'; '.join(details)})" if details else ""
        raise InstanceRefused(f"instance {instance_id} refused: {exc}{suffix}. The harness never runs a stale, "
                              "superseded or unlisted instance.") from None


def read_instance_record(instance_id: str, *, allow_main: bool = False) -> InstanceRecord:
    """Read and check one instance without writing anything.

    Refuses: a malformed id, a missing directory, a v0 or otherwise non-v1 instance, missing private fields, a
    task.json whose limits or op list differ from the executor's current ones, an instance that is not active in
    index.json (listed as superseded, or missing from the active rows) and, unless ``allow_main``, an instance
    whose private.json says ``split_role == "main"`` (the main-instance guard; the flag stays unused until the
    PREREG is frozen).
    """
    if fewshot.is_fewshot_id(instance_id):
        return read_fewshot_record(instance_id, allow_main=allow_main)
    if seqskill.is_seqskill_id(instance_id):
        return read_seqskill_record(instance_id, allow_main=allow_main)
    if graphnew.is_graphnew_id(instance_id):
        return read_graphnew_record(instance_id, allow_main=allow_main)
    if fscil.is_fscil_id(instance_id):
        return read_fscil_record(instance_id, allow_main=allow_main)
    if policyskill.is_policyskill_id(instance_id):
        return read_policyskill_record(instance_id, allow_main=allow_main)
    root = instance_dir_for(instance_id)
    task = read_json(root / "task.json")
    private = read_json(root / "private.json")
    if not isinstance(task, dict) or not isinstance(private, dict):
        raise InstanceRefused(f"instance {instance_id}: task.json and private.json must be JSON objects")
    refuse_v0(instance_id, task, private)
    refuse_stale(instance_id, root, task)
    row = active_index_row(instance_id)
    missing = [field for field in REQUIRED_PRIVATE_FIELDS if field not in private]
    if missing:
        raise InstanceRefused(f"instance {instance_id}: private.json lacks {missing}")
    if private.get("instance_id") != instance_id:
        raise InstanceRefused(f"instance {instance_id}: private.json names instance {private.get('instance_id')!r}")
    _refuse_row_or_role(instance_id, row, private, allow_main)
    return InstanceRecord(instance_id=instance_id, instance_dir=root, task=task, private=private, index_row=row)


def _refuse_row_or_role(instance_id: str, row: Mapping[str, Any], private: Mapping[str, Any], allow_main: bool) -> None:
    """The index row must agree with private.json; a main instance needs ``allow_main``; the role must be known."""
    for field in ("split_role", "dataset", "model_id"):
        if field in row and row[field] != private[field]:
            raise InstanceRefused(f"instance {instance_id}: index.json {field} {row[field]!r} != private.json {private[field]!r}")
    role = private["split_role"]
    if role == SPLIT_ROLE_MAIN and not allow_main:
        raise InstanceRefused(
            f"instance {instance_id} is a main instance (split_role 'main'). No session runs on a main instance "
            "until the PREREG is frozen; pass allow_main=True (--allow-main) only after that."
        )
    if role not in (SPLIT_ROLE_DEV, SPLIT_ROLE_MAIN):
        raise InstanceRefused(f"instance {instance_id} has split_role {role!r}")


def read_fewshot_record(instance_id: str, *, allow_main: bool = False) -> InstanceRecord:
    """``read_instance_record`` for a fewshot-v1 instance (PREREG E2), without writing anything.

    Refuses: a missing directory; an instance that ``check_fewshot_current`` refuses (not active in the fewshot
    index, another rule, task.json or a file other than the current code writes); missing private fields
    (``weightcode.harness.fewshot.REQUIRED_PRIVATE_FIELDS``); an index row that disagrees with private.json; and, unless
    ``allow_main``, a main instance.
    """
    try:
        root = fewshot.instance_dir(instance_id)
    except fewshot.FewshotHarnessError as exc:
        raise InstanceRefused(str(exc)) from None
    if not root.is_dir():
        raise InstanceRefused(f"fewshot instance {instance_id} does not exist under $WEIGHTCODE_PRIVATE/"
                              f"{fewshot.FEWSHOT_ROOT_NAME}")
    try:
        task = fewshot.check_current(root)
    except fewshot.stale_error() as exc:
        raise InstanceRefused(f"instance {instance_id} refused: {exc}. The harness never runs a stale, superseded or "
                              "unlisted instance.") from None
    private = read_json(root / "private.json")
    if not isinstance(task, dict) or not isinstance(private, dict):
        raise InstanceRefused(f"instance {instance_id}: task.json and private.json must be JSON objects")
    if not fewshot.is_fewshot(private):
        raise InstanceRefused(f"instance {instance_id} follows {private.get('evidence_rule')!r}, not "
                              f"{fewshot.FEWSHOT_RULE}")
    missing = [field for field in fewshot.REQUIRED_PRIVATE_FIELDS if field not in private]
    if missing:
        raise InstanceRefused(f"instance {instance_id}: private.json lacks {missing}")
    if private.get("instance_id") != instance_id:
        raise InstanceRefused(f"instance {instance_id}: private.json names instance {private.get('instance_id')!r}")
    row = fewshot.active_index_row(instance_id)
    if row is None:
        raise InstanceRefused(f"instance {instance_id} is missing from the active rows of the fewshot index; no runner "
                              "may select it")
    _refuse_row_or_role(instance_id, row, private, allow_main)
    return InstanceRecord(instance_id=instance_id, instance_dir=root, task=task, private=private, index_row=row)


def read_seqskill_record(instance_id: str, *, allow_main: bool = False) -> InstanceRecord:
    """``read_instance_record`` for a seqskill-v1 instance (PREREG E5), without writing anything.

    Refuses: a missing directory; an instance that ``check_seqskill_current`` refuses (not active in the seqskill
    index, another rule, task.json or a file other than the current code writes); missing private fields
    (``weightcode.harness.seqskill.REQUIRED_PRIVATE_FIELDS``); an index row that disagrees with private.json; and, unless
    ``allow_main``, a main instance.
    """
    try:
        root = seqskill.instance_dir(instance_id)
    except seqskill.SeqskillHarnessError as exc:
        raise InstanceRefused(str(exc)) from None
    if not root.is_dir():
        raise InstanceRefused(f"seqskill instance {instance_id} does not exist under $WEIGHTCODE_PRIVATE/"
                              f"{seqskill.SEQSKILL_ROOT_NAME}")
    try:
        task = seqskill.check_current(root)
    except seqskill.stale_error() as exc:
        raise InstanceRefused(f"instance {instance_id} refused: {exc}. The harness never runs a stale, superseded or "
                              "unlisted instance.") from None
    private = read_json(root / "private.json")
    if not isinstance(task, dict) or not isinstance(private, dict):
        raise InstanceRefused(f"instance {instance_id}: task.json and private.json must be JSON objects")
    if not seqskill.is_seqskill(private):
        raise InstanceRefused(f"instance {instance_id} follows {private.get('evidence_rule')!r}, not "
                              f"{seqskill.SEQSKILL_RULE}")
    if seqskill.is_scan_id(instance_id) != seqskill.is_scan(private):
        raise InstanceRefused(f"instance {instance_id}: its id and its rule {private.get('evidence_rule')!r} name "
                              "different stores")
    missing = [field for field in seqskill.required_fields(private) if field not in private]
    if missing:
        raise InstanceRefused(f"instance {instance_id}: private.json lacks {missing}")
    if private.get("instance_id") != instance_id:
        raise InstanceRefused(f"instance {instance_id}: private.json names instance {private.get('instance_id')!r}")
    row = seqskill.active_index_row(instance_id)
    if row is None:
        raise InstanceRefused(f"instance {instance_id} is missing from the active rows of the seqskill index; no "
                              "runner may select it")
    _refuse_row_or_role(instance_id, row, private, allow_main)
    return InstanceRecord(instance_id=instance_id, instance_dir=root, task=task, private=private, index_row=row)


def read_graphnew_record(instance_id: str, *, allow_main: bool = False) -> InstanceRecord:
    """``read_instance_record`` for a graphnew-v1 instance (PREREG E7), without writing anything.

    Refuses: a missing directory; an instance that ``check_graphnew_current`` refuses (not active in the graphnew
    index, another rule, task.json or a file other than the current code writes, graph.npz included); missing private
    fields (``weightcode.harness.graphnew.REQUIRED_PRIVATE_FIELDS``); an index row that disagrees with private.json; and,
    unless ``allow_main``, a main instance.
    """
    try:
        root = graphnew.instance_dir(instance_id)
    except graphnew.GraphnewHarnessError as exc:
        raise InstanceRefused(str(exc)) from None
    if not root.is_dir():
        raise InstanceRefused(f"graphnew instance {instance_id} does not exist under $WEIGHTCODE_PRIVATE/"
                              f"{graphnew.GRAPHNEW_ROOT_NAME}")
    try:
        task = graphnew.check_current(root)
    except graphnew.stale_error() as exc:
        raise InstanceRefused(f"instance {instance_id} refused: {exc}. The harness never runs a stale, superseded or "
                              "unlisted instance.") from None
    private = read_json(root / "private.json")
    if not isinstance(task, dict) or not isinstance(private, dict):
        raise InstanceRefused(f"instance {instance_id}: task.json and private.json must be JSON objects")
    if not graphnew.is_graphnew(private):
        raise InstanceRefused(f"instance {instance_id} follows {private.get('evidence_rule')!r}, not "
                              f"{graphnew.GRAPHNEW_RULE}")
    missing = [field for field in graphnew.REQUIRED_PRIVATE_FIELDS if field not in private]
    if missing:
        raise InstanceRefused(f"instance {instance_id}: private.json lacks {missing}")
    if private.get("instance_id") != instance_id:
        raise InstanceRefused(f"instance {instance_id}: private.json names instance {private.get('instance_id')!r}")
    row = graphnew.active_index_row(instance_id)
    if row is None:
        raise InstanceRefused(f"instance {instance_id} is missing from the active rows of the graphnew index; no "
                              "runner may select it")
    _refuse_row_or_role(instance_id, row, private, allow_main)
    return InstanceRecord(instance_id=instance_id, instance_dir=root, task=task, private=private, index_row=row)


def read_fscil_record(instance_id: str, *, allow_main: bool = False) -> InstanceRecord:
    """``read_instance_record`` for an fscil-v1 instance (PREREG E9), without writing anything.

    Refuses: a missing directory; an instance that ``check_fscil_current`` refuses (not active in the fscil index,
    another rule, task.json or a file other than the current code writes); missing private fields
    (``weightcode.harness.fscil.REQUIRED_PRIVATE_FIELDS``); an index row that disagrees with private.json; and, unless
    ``allow_main``, a main instance.
    """
    try:
        root = fscil.instance_dir(instance_id)
    except fscil.FscilHarnessError as exc:
        raise InstanceRefused(str(exc)) from None
    if not root.is_dir():
        raise InstanceRefused(f"fscil instance {instance_id} does not exist under "
                              f"$WEIGHTCODE_PRIVATE/{fscil.FSCIL_ROOT_NAME}")
    try:
        task = fscil.check_current(root)
    except fscil.stale_error() as exc:
        raise InstanceRefused(f"instance {instance_id} refused: {exc}. The harness never runs a stale, superseded or "
                              "unlisted instance.") from None
    private = read_json(root / "private.json")
    if not isinstance(task, dict) or not isinstance(private, dict):
        raise InstanceRefused(f"instance {instance_id}: task.json and private.json must be JSON objects")
    if not fscil.is_fscil(private):
        raise InstanceRefused(f"instance {instance_id} follows {private.get('evidence_rule')!r}, not "
                              f"{fscil.FSCIL_RULE}")
    missing = [field for field in fscil.REQUIRED_PRIVATE_FIELDS if field not in private]
    if missing:
        raise InstanceRefused(f"instance {instance_id}: private.json lacks {missing}")
    if private.get("instance_id") != instance_id:
        raise InstanceRefused(f"instance {instance_id}: private.json names instance {private.get('instance_id')!r}")
    row = fscil.active_index_row(instance_id)
    if row is None:
        raise InstanceRefused(f"instance {instance_id} is missing from the active rows of the fscil index; no runner "
                              "may select it")
    for field in ("session", "parent_params_sha256", "parent_instance", "base_model_id"):
        if row.get(field) != private.get(field):
            raise InstanceRefused(f"instance {instance_id}: index.json {field} {row.get(field)!r} != private.json "
                                  f"{private.get(field)!r}")
    _refuse_row_or_role(instance_id, row, private, allow_main)
    return InstanceRecord(instance_id=instance_id, instance_dir=root, task=task, private=private, index_row=row)


def read_policyskill_record(instance_id: str, *, allow_main: bool = False) -> InstanceRecord:
    """``read_instance_record`` for a policyskill-v1 instance (PREREG E14), without writing anything.

    Refuses: a missing directory; an instance that ``check_e14_current`` refuses (not active in the E14 index, another
    rule, task.json or a file other than the current code writes); missing private fields
    (``weightcode.harness.policyskill.REQUIRED_PRIVATE_FIELDS``); an index row that disagrees with private.json; and, unless
    ``allow_main``, a main instance.
    """
    try:
        root = policyskill.instance_dir(instance_id)
    except policyskill.PolicyskillHarnessError as exc:
        raise InstanceRefused(str(exc)) from None
    if not root.is_dir():
        raise InstanceRefused(f"policyskill instance {instance_id} does not exist under $WEIGHTCODE_PRIVATE/"
                              f"{policyskill.POLICYSKILL_ROOT_NAME}")
    try:
        task = policyskill.check_current(root)
    except policyskill.stale_error() as exc:
        raise InstanceRefused(f"instance {instance_id} refused: {exc}. The harness never runs a stale, superseded or "
                              "unlisted instance.") from None
    private = read_json(root / "private.json")
    if not isinstance(task, dict) or not isinstance(private, dict):
        raise InstanceRefused(f"instance {instance_id}: task.json and private.json must be JSON objects")
    if not policyskill.is_policyskill(private):
        raise InstanceRefused(f"instance {instance_id} follows {private.get('evidence_rule')!r}, not "
                              f"{policyskill.POLICYSKILL_RULE}")
    missing = [field for field in policyskill.REQUIRED_PRIVATE_FIELDS if field not in private]
    if missing:
        raise InstanceRefused(f"instance {instance_id}: private.json lacks {missing}")
    if private.get("instance_id") != instance_id:
        raise InstanceRefused(f"instance {instance_id}: private.json names instance {private.get('instance_id')!r}")
    row = policyskill.active_index_row(instance_id)
    if row is None:
        raise InstanceRefused(f"instance {instance_id} is missing from the active rows of the E14 index; no runner "
                              "may select it")
    _refuse_row_or_role(instance_id, row, private, allow_main)
    return InstanceRecord(instance_id=instance_id, instance_dir=root, task=task, private=private, index_row=row)


def active_index_row(instance_id: str) -> dict[str, Any]:
    """The instance's row in index.json ``instances``; refuses an id listed in ``superseded`` or not listed at all."""
    index = read_index()
    active = {row.get("instance_id"): row for row in index["instances"] if isinstance(row, dict)}
    retired = {row.get("instance_id"): row for row in index.get("superseded", []) if isinstance(row, dict)}
    if instance_id in retired:
        row = retired[instance_id]
        raise InstanceRefused(
            f"instance {instance_id} is listed in instances/index.json 'superseded' (at {row.get('superseded_at')!r}, "
            f"reason: {row.get('superseded_reason')!r}); no runner may select it"
        )
    if instance_id not in active:
        raise InstanceRefused(f"instance {instance_id} is missing from the active rows ('instances') of "
                              "instances/index.json; no runner may select it")
    return dict(active[instance_id])


def list_instances(
    *,
    split_role: str | None = SPLIT_ROLE_DEV,
    dataset: str | None = None,
    seed: int | None = None,
    condition: str | None = None,
    evidence_rule: str | None = EVIDENCE_RULE_V1,
) -> list[dict[str, Any]]:
    """Active rows of ``$WEIGHTCODE_PRIVATE/instances/index.json``, by default the evidence-v1 dev instances.

    ``split_role=None`` lists every role; main instances are listed only when asked for. Instances are found by
    rule, dataset, condition, split role and seed, never by a remembered id (ids change when instances are rebuilt).
    """
    if not (private_dir() / "instances").is_dir():
        return []  # read_index would create the directory; listing must not write
    rows = []
    for row in read_index()["instances"]:
        if not isinstance(row, dict):
            continue
        if split_role is not None and row.get("split_role") != split_role:
            continue
        if evidence_rule is not None and row.get("evidence_rule") != evidence_rule:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if condition is not None and row.get("condition") != condition:
            continue
        if seed is not None and row.get("seed") != seed:
            continue
        rows.append(dict(row))
    return rows


def find_instance(*, dataset: str, seed: int, split_role: str = SPLIT_ROLE_DEV, condition: str | None = None) -> str:
    """The id of the single active evidence-v1 instance with this dataset, seed and split role."""
    rows = list_instances(split_role=split_role, dataset=dataset, seed=seed, condition=condition)
    if len(rows) != 1:
        found = [row.get("instance_id") for row in rows]
        raise LookupError(f"expected one active v1 {split_role} instance of {dataset} seed {seed}, found {found}")
    return str(rows[0]["instance_id"])
