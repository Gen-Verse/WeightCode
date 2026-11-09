"""The harness side of rule seqskill-v1 (PREREG E5): the one access point of the editor harness to
``weightcode.evalkit.seqskill``, which owns the rule (the instances, their evidence, the part cut of V_check and the
V_select readout).

A seqskill instance asks an update to teach a sequence student a new operator. It runs in the LEARN arm only and
follows the fewshot-v1 path (``weightcode.harness.fewshot``) step by step:

- the instance id is ``q`` + 8 hex and the instance lives under ``$WEIGHTCODE_PRIVATE/seqskill_instances``; the runner
  refuses it unless ``check_seqskill_current`` accepts it (active in its index, task.json and files as the current
  code writes them);
- the student is a sequence model: inputs are int64 token rows [N, L], targets [N, L] hold the answer symbol at each
  OUT position and IGNORE (-1) elsewhere, and a row is right when every answer position is right (exact match);
- the experience is the instance's evidence.npz itself, which already has the LEARN keys (the new operator's rows and
  the old-operator replay rows). Every ``pred_*`` is re-checked with ``seq_predictions``; every new row must hold
  the new operator alone and no replay row a reserved operator;
- the check is the condition's V_check cut by part (``seqskill_check_parts``): S_new over part 1 (the new operator
  alone, the lengths of the evidence), S_old over part 4 (old operators). The session ledger cuts the session's
  V_check file itself, with its ``part`` array and the part sizes recorded in session.json (``session_record``), so
  the MCP server needs no store variable;
- the readout is ``seqskill_readout`` on V_select: parts i to iv, child against base (the PREREG E5 gate reads
  part i and part iv).

There is no decoy, no corruption condition and no clean companion. ``weightcode.evalkit.seqskill`` is imported inside the
functions that need it, so an evidence-v1 session never loads it.

Variant rule_only (PREREG E8, a rule given as knowledge, no example of it). The same instance, student, checks, kept
rule, limits and readouts; only the teacher's information differs:

- the experience is the old part of the instance's evidence alone (``RULE_ONLY_EXPERIENCE_KEYS``: the 64 replay rows),
  after the whole evidence has been re-checked exactly as in the standard view (``rule_only_experience``);
- the trial's model_spec.json states the new operator's rule in the form the old symbol maps have
  (``rule_only_model_spec``). The table is read from the condition record
  (``$WEIGHTCODE_PRIVATE/<dataset>/conditions/<condition>/dataset.json``, key ``condition.rule.table``), whose sha256 must
  equal private.json ``condition_dataset_json_sha256`` (``new_operator_rule``); it is never typed by hand. The
  instance's own model_spec.json is unchanged and still checked against ``model_spec_sha256``;
- the leak words keep the condition name and drop the words that name its kind of rule ("substitution",
  "permutation"), since the rule is now given (``leak_words``);
- the session record's ``seqskill`` section gets ``"variant": "rule_only"`` and the rule's provenance. A standard
  session's section has no ``variant`` key, so its bytes are those of an E5 session.

Variant scan_rule (PREREG E13, rule scanskill-v1 of ``weightcode.evalkit.scanskill``): the same path for a SCAN student, an
instance id ``n`` + 8 hex under ``$WEIGHTCODE_PRIVATE/scan_instances``. A scanskill instance runs under this variant only,
and this variant runs on a scanskill instance only. What differs:

- the experience is the instance's evidence as it is (``scan_experience``): on addprim_jump the "jump" row as in
  training (``x_prim``, ``y_prim``, ``pred_prim``, ``split_ids_prim``), then the 64 replay rows of old commands; on
  around_right the replay rows alone. Every ``pred_*`` is re-checked; the rule is stated in model_spec.json (the
  student's own, unchanged);
- the check has two files: new V_check (the session's ``vcheck_path``) and the 1000 old V_check rows (old_check), whose
  path and hash the session's ``seqskill`` section records (``session_record``), so the ledger's seqskill branch cuts
  nothing and ``check_parts`` loads the old file itself;
- the leak words are ``scan_leak_tokens`` (the dataset's tokens, the condition names, "addprim", "template" and the
  official file names); the readout is ``scan_readout``; the retention check set is named ``old-command``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256
from weightcode.common.paths import private_dir

SEQSKILL_RULE = "seqskill-v1"  # weightcode.evalkit.seqskill.SEQSKILL_RULE; test_harness_seqskill checks that the two agree
INSTANCE_ID_RE = re.compile(r"^q[0-9a-f]{8}$")  # weightcode.evalkit.seqskill.INSTANCE_ID_RE, checked the same way
SEQSKILL_ROOT_NAME = "seqskill_instances"  # weightcode.evalkit.seqskill.ROOT_NAME
EXPERIENCE_SOURCE = "instance_seqskill_v1"
EXPERIENCE_KEYS = ("x_new", "y_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old")
PREDICT_BATCH = 1024
READOUT_PARTS = ("i", "ii", "iii", "iv", "new", "all")
CHECK_SET = "old-operator"  # the rows behind score_old, as the retention kept rule names them
CHECK_UNIT = "rows"
# private.json fields the harness reads from a seqskill instance; an instance without one of them is refused.
REQUIRED_PRIVATE_FIELDS = (
    "instance_id",
    "split_role",
    "dataset",
    "condition",
    "arch",
    "seed",
    "model_id",
    "evidence_rule",
    "new_operator",
    "new_operator_id",
    "vcheck_path",
    "vcheck_sha256",
    "vcheck_content_sha256",
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

# Session variants of the seqskill-v1 LEARN view: standard (PREREG E5) and rule_only (PREREG E8); scan_rule is the
# scanskill-v1 view (PREREG E13), the only variant of a scanskill instance.
STANDARD = "standard"
RULE_ONLY = "rule_only"
SCAN = "scan_rule"
VARIANTS = (STANDARD, RULE_ONLY, SCAN)

SCAN_RULE = "scanskill-v1"  # weightcode.evalkit.scanskill.SCAN_RULE; test_harness_scan checks that the two agree
SCAN_ID_RE = re.compile(r"^n[0-9a-f]{8}$")  # weightcode.evalkit.scanskill.INSTANCE_ID_RE
SCAN_ROOT_NAME = "scan_instances"  # weightcode.evalkit.scanskill.ROOT_NAME
SCAN_EXPERIENCE_SOURCE = "instance_scanskill_v1"
SCAN_CHECK_SET = "old-command"  # the rows behind score_old on a scanskill instance
SCAN_REQUIRED_PRIVATE_FIELDS = (
    "instance_id", "split_role", "dataset", "condition", "arch", "seed", "model_id", "evidence_rule", "primitive",
    "primitive_row", "primitive_token_id", "vcheck_path", "vcheck_sha256", "vcheck_content_sha256", "oldcheck_path",
    "oldcheck_sha256", "oldcheck_content_sha256", "n_vcheck", "n_vcheck_new", "n_vcheck_old",
    "n_base_errors_vcheck_new", "n_base_errors_vcheck_old", "n_prim", "n_replay", "model_sha256", "forward_sha256",
    "task_sha256", "evidence_content_sha256", "model_spec_sha256",
)
SCAN_PRIVATE_PATH_FIELDS = ("vcheck_path", "vselect_path", "oldcheck_path", "oldselect_path", "train_path",
                            "model_source_path")
RULE_ONLY_EXPERIENCE_KEYS = ("x_old", "y_old", "pred_old", "split_ids_old")
RULE_ONLY_EXPERIENCE_SOURCE = "instance_seqskill_v1_rule_only"
# How the instance's model_spec.json describes a reserved operator token (weightcode.data.seqops.token_meanings), and the
# prefix of it that rule_only replaces with the rule; the rest ("absent from all training rows ...") is kept.
RESERVED_MEANING_PREFIX = "reserved operator token; "
SYMBOL_MAP_KIND = "symbol"  # condition.rule.kind of a symbol map, the only kind rule_only states
SYMBOL_MAP_MEANING = "operator, symbol map: s -> table[s], table = "  # as OP_SUBA's meaning (weightcode.data.seqops)


class SeqskillHarnessError(RuntimeError):
    """A seqskill instance or session record is inconsistent, so no LEARN view, check or readout is built from it."""


def check_variant(variant: str | None) -> str:
    """``variant`` as one of ``VARIANTS``; None is the standard view (PREREG E5)."""
    name = STANDARD if variant is None else variant
    if name not in VARIANTS:
        raise SeqskillHarnessError(f"unknown seqskill variant {variant!r}; one of {VARIANTS}")
    return name


def is_scan_id(instance_id: object) -> bool:
    """True for an id of the scanskill store's form (``n`` and 8 hex digits, PREREG E13)."""
    return isinstance(instance_id, str) and SCAN_ID_RE.fullmatch(instance_id) is not None


def is_seqskill_id(instance_id: object) -> bool:
    """True for an id of the seqskill store's form (``q`` and 8 hex digits) or of the scanskill store's (``n``)."""
    return (isinstance(instance_id, str) and INSTANCE_ID_RE.fullmatch(instance_id) is not None) or is_scan_id(
        instance_id)


def is_scan(private: Mapping[str, Any]) -> bool:
    """True when private.json says the instance follows rule scanskill-v1 (PREREG E13)."""
    return private.get("evidence_rule") == SCAN_RULE


def is_seqskill(private: Mapping[str, Any]) -> bool:
    """True when private.json says the instance follows rule seqskill-v1, or scanskill-v1 (its scan_rule variant)."""
    return private.get("evidence_rule") == SEQSKILL_RULE or is_scan(private)


def check_instance_variant(private: Mapping[str, Any], variant: str | None) -> str:
    """``variant`` checked against the instance: scan_rule on a scanskill instance and on no other."""
    name = check_variant(variant)
    if is_scan(private) != (name == SCAN):
        raise SeqskillHarnessError(f"a scanskill-v1 instance runs under the {SCAN} variant, and that variant on a "
                                   f"scanskill-v1 instance only (instance rule {private.get('evidence_rule')!r}, "
                                   f"variant {name!r})")
    return name


def instance_dir(instance_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/seqskill_instances/<instance_id>`` (or ``scan_instances`` for an ``n`` id), the path of the
    evalkit's instance dir computed without creating the store root, so that resolving an id writes nothing; refuses
    an id of another form."""
    if is_scan_id(instance_id):
        return private_dir() / SCAN_ROOT_NAME / instance_id
    if not is_seqskill_id(instance_id):
        raise SeqskillHarnessError(f"{instance_id!r} is not a seqskill instance id ('q' and 8 hex digits)")
    return private_dir() / SEQSKILL_ROOT_NAME / instance_id


def check_current(path: str | Path) -> dict[str, Any]:
    """``weightcode.evalkit.seqskill.check_seqskill_current`` (``check_scan_current`` for a scanskill instance): the
    instance's task, or the evalkit's stale-instance error."""
    if is_scan_id(Path(path).name):
        from weightcode.evalkit.scanskill import check_scan_current

        return check_scan_current(path)
    from weightcode.evalkit.seqskill import check_seqskill_current

    return check_seqskill_current(path)


def stale_error() -> tuple[type[Exception], ...]:
    """The evalkit exceptions for an instance that is not active or not current (and every other seqskill or
    scanskill refusal), as a tuple an ``except`` clause takes."""
    from weightcode.evalkit.scanskill import ScanskillError
    from weightcode.evalkit.seqskill import SeqskillError

    return (SeqskillError, ScanskillError)


def active_index_row(instance_id: str) -> dict[str, Any] | None:
    """The instance's row among the active rows of the seqskill (or scanskill) index, or None."""
    if is_scan_id(instance_id):
        from weightcode.evalkit.scanskill import read_scan_instance_index

        rows = read_scan_instance_index()["instances"]
    else:
        from weightcode.evalkit.seqskill import read_seqskill_instance_index

        rows = read_seqskill_instance_index()["instances"]
    found = [dict(row) for row in rows if isinstance(row, dict) and row.get("instance_id") == instance_id]
    return found[0] if found else None


def required_fields(private: Mapping[str, Any]) -> tuple[str, ...]:
    """The private.json fields the harness reads from this instance."""
    return SCAN_REQUIRED_PRIVATE_FIELDS if is_scan(private) else REQUIRED_PRIVATE_FIELDS


def experience_keys(private: Mapping[str, Any], variant: str | None = None) -> tuple[str, ...]:
    """The keys of the trial's experience.npz: the scanskill evidence keys of the condition, rule_only's old part, or
    ``EXPERIENCE_KEYS``."""
    if is_scan(private):
        from weightcode.evalkit.scanskill import evidence_keys

        return evidence_keys(str(private["condition"]))
    return RULE_ONLY_EXPERIENCE_KEYS if check_variant(variant) == RULE_ONLY else EXPERIENCE_KEYS


def check_set_name(record: Mapping[str, Any]) -> str:
    """The name of the rows behind score_old in the retention kept rule, from a session's ``seqskill`` section."""
    return SCAN_CHECK_SET if record.get("rule") == SCAN_RULE else CHECK_SET


def leak_words(private: Mapping[str, Any], variant: str | None = None) -> list[str]:
    """``weightcode.evalkit.seqskill.seqskill_leak_tokens`` of the instance's condition: the dataset tokens, the condition
    name, its rule kind and the words that would name the kind of rule (the instance build scans for the same).

    rule_only (PREREG E8) gives the rule to the teacher, so the words that name its kind (the rule kind and
    ``CONDITION_LEAK_WORDS``: "substitution", "permutation") leave the list; the condition name stays."""
    if check_instance_variant(private, variant) == SCAN:  # PREREG E13: weightcode.evalkit.scanskill.scan_leak_tokens
        from weightcode.evalkit.scanskill import scan_leak_tokens

        return scan_leak_tokens()
    from weightcode.data.seqops_build import parse_condition
    from weightcode.evalkit.seqskill import CONDITION_LEAK_WORDS, seqskill_leak_tokens

    condition = str(private["condition"])
    words = seqskill_leak_tokens(condition)
    if check_variant(variant) == STANDARD:
        return words
    _, kind = parse_condition(condition)
    dropped = {*CONDITION_LEAK_WORDS, kind.lower()}
    kept = [word for word in words if word not in dropped]
    if condition.lower() not in kept:
        raise SeqskillHarnessError(f"the rule_only leak words lost the condition name {condition!r}")
    return kept


def _token_ids() -> tuple[int, tuple[int, ...]]:
    from weightcode.data import seqops as S

    return int(S.NONE), tuple(int(t) for t in S.NEW_OP_IDS)


def _check_rows(x: np.ndarray, y: np.ndarray, new_operator_id: int | None, what: str) -> None:
    """Token rows of one part: int64 [N, L] with one target per position; ``new_operator_id`` set: every row holds
    that operator alone (OPA it, OPB NONE); None: no row holds a reserved operator in either slot."""
    none, reserved = _token_ids()
    if x.dtype != np.int64 or x.ndim != 2 or y.shape != x.shape or len(x) == 0:
        raise SeqskillHarnessError(f"{what}: expected int64 token rows [N, L] with targets of the same shape, got "
                                   f"{x.dtype}{x.shape} and {y.shape}")
    if new_operator_id is None:
        if np.isin(x[:, 1:3], reserved).any():
            raise SeqskillHarnessError(f"{what}: a row holds a reserved operator")
    elif not (np.all(x[:, 1] == new_operator_id) and np.all(x[:, 2] == none)):
        raise SeqskillHarnessError(f"{what}: every row must hold operator {new_operator_id} alone (OPA it, OPB NONE)")


def experience(instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The LEARN experience of a seqskill instance and its host-side record.

    The arrays are the instance's evidence under ``EXPERIENCE_KEYS``. ``pred_new`` and ``pred_old`` must be the
    student's predictions (``seq_predictions``: the argmax at each answer position, IGNORE elsewhere); every new row
    holds the new operator alone and no replay row a reserved operator; the ids are unique.
    """
    from weightcode.evalkit.metrics import rows_right
    from weightcode.evalkit.seqskill import seq_predictions

    private = instance.private
    if is_scan(private):
        return scan_experience(instance)
    if not is_seqskill(private):
        raise SeqskillHarnessError(f"{instance.instance_id} follows {private.get('evidence_rule')!r}, not "
                                   f"{SEQSKILL_RULE}")
    evidence = instance.evidence
    if sorted(evidence) != sorted(EXPERIENCE_KEYS):
        raise SeqskillHarnessError(f"{instance.instance_id}: evidence keys {sorted(evidence)} are not the LEARN keys "
                                   f"{sorted(EXPERIENCE_KEYS)}")
    arrays = {key: np.ascontiguousarray(evidence[key]) for key in EXPERIENCE_KEYS}
    new_operator_id = int(private["new_operator_id"])
    n_right = {}
    for part, operator in (("new", new_operator_id), ("old", None)):
        x, y, shown = arrays[f"x_{part}"], arrays[f"y_{part}"], arrays[f"pred_{part}"]
        if not (len(x) == len(y) == len(shown) == len(arrays[f"split_ids_{part}"])):
            raise SeqskillHarnessError(f"{instance.instance_id}: the {part} part has unequal lengths")
        _check_rows(x, y, operator, f"{instance.instance_id} {part} part")
        pred = seq_predictions(instance.forward, instance.params, x, y, PREDICT_BATCH)
        if not np.array_equal(pred, shown):
            raise SeqskillHarnessError(f"{instance.instance_id}: pred_{part} disagrees with the student at "
                                       f"{int(np.count_nonzero(pred != shown))} positions")
        n_right[part] = int(np.count_nonzero(rows_right(y, pred)))
    ids = [*arrays["split_ids_new"].tolist(), *arrays["split_ids_old"].tolist()]
    if len(set(ids)) != len(ids):
        raise SeqskillHarnessError(f"{instance.instance_id}: the split ids are not unique")
    record = {
        "file": "experience.npz",
        "source": EXPERIENCE_SOURCE,
        "rule": SEQSKILL_RULE,
        "content_sha256": params_sha256(arrays),
        "evidence_content_sha256": private["evidence_content_sha256"],
        "student_params_sha256": instance.params_sha256,
        "new_operator": str(private["new_operator"]),
        "new_operator_id": new_operator_id,
        "n_new": int(len(arrays["y_new"])),
        "n_new_right": n_right["new"],
        "n_old": int(len(arrays["y_old"])),
        "n_old_fail": int(len(arrays["y_old"])) - n_right["old"],
    }
    if record["content_sha256"] != private["evidence_content_sha256"]:
        raise SeqskillHarnessError(f"{instance.instance_id}: the experience differs from the recorded evidence")
    return arrays, record


def scan_experience(instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The LEARN experience of a scanskill instance (PREREG E13) and its host-side record: the evidence under the
    condition's keys (``weightcode.evalkit.scanskill.evidence_keys``), every ``pred_*`` re-checked with ``seq_predictions``,
    the primitive part the primitive alone, no replay row holding the condition's primitive words, unique ids, and
    the content hash the instance records."""
    from weightcode.data import scan as S
    from weightcode.evalkit.metrics import rows_right
    from weightcode.evalkit.scanskill import evidence_keys, seq_predictions

    private = instance.private
    condition = str(private["condition"])
    keys = evidence_keys(condition)
    evidence = instance.evidence
    if sorted(evidence) != sorted(keys):
        raise SeqskillHarnessError(f"{instance.instance_id}: evidence keys {sorted(evidence)} are not {sorted(keys)}")
    arrays = {key: np.ascontiguousarray(evidence[key]) for key in keys}
    parts = ("prim", "old") if "x_prim" in arrays else ("old",)
    n_right = {}
    for part in parts:
        x, y, shown = arrays[f"x_{part}"], arrays[f"y_{part}"], arrays[f"pred_{part}"]
        if x.dtype != np.int64 or x.ndim != 2 or y.shape != x.shape or len(x) == 0:
            raise SeqskillHarnessError(f"{instance.instance_id} {part} part: expected int64 token rows [N, L] with "
                                       f"targets of the same shape, got {x.dtype}{x.shape} and {y.shape}")
        if not (len(x) == len(shown) == len(arrays[f"split_ids_{part}"])):
            raise SeqskillHarnessError(f"{instance.instance_id}: the {part} part has unequal lengths")
        commands = S.decode(x)
        if part == "prim" and commands != [str(private["primitive_row"])] * len(x):
            raise SeqskillHarnessError(f"{instance.instance_id}: the primitive part is not {private['primitive_row']!r}")
        if part == "old" and any(S.has_primitive(c, condition) for c in commands):
            raise SeqskillHarnessError(f"{instance.instance_id}: a replay row holds {S.PRIMITIVE[condition]}")
        pred = seq_predictions(instance.forward, instance.params, x, y, PREDICT_BATCH)
        if not np.array_equal(pred, shown):
            raise SeqskillHarnessError(f"{instance.instance_id}: pred_{part} disagrees with the student at "
                                       f"{int(np.count_nonzero(pred != shown))} positions")
        n_right[part] = int(np.count_nonzero(rows_right(y, pred)))
    ids = [i for part in parts for i in arrays[f"split_ids_{part}"].tolist()]
    if len(set(ids)) != len(ids):
        raise SeqskillHarnessError(f"{instance.instance_id}: the split ids are not unique")
    n_prim = int(len(arrays["y_prim"])) if "y_prim" in arrays else 0
    record = {
        "file": "experience.npz",
        "source": SCAN_EXPERIENCE_SOURCE,
        "rule": SCAN_RULE,
        "variant": SCAN,
        "content_sha256": params_sha256(arrays),
        "evidence_content_sha256": private["evidence_content_sha256"],
        "student_params_sha256": instance.params_sha256,
        "condition": condition,
        "primitive": list(private["primitive"]),
        "primitive_row": private["primitive_row"],
        "n_prim": n_prim,
        "n_prim_right": n_right.get("prim", 0),
        "n_new": 0,
        "n_old": int(len(arrays["y_old"])),
        "n_old_fail": int(len(arrays["y_old"])) - n_right["old"],
    }
    if record["content_sha256"] != private["evidence_content_sha256"] or n_prim != int(private["n_prim"]):
        raise SeqskillHarnessError(f"{instance.instance_id}: the experience differs from the recorded evidence")
    return arrays, record


def rule_only_experience(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]
                         ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The rule_only experience (PREREG E8): the old part of ``experience``'s arrays alone, and its record.

    ``arrays`` and ``record`` are what ``experience`` returns, so every row of the instance's evidence was re-checked
    (predictions, operators, ids, content hash) before the new part is withheld. The record keeps the instance's
    evidence hash and states that no new row is shown (``n_new`` 0) and how many were withheld."""
    if sorted(arrays) != sorted(EXPERIENCE_KEYS) or record.get("source") != EXPERIENCE_SOURCE:
        raise SeqskillHarnessError("the rule_only experience is cut from the checked standard experience only")
    shown = {key: np.ascontiguousarray(arrays[key]) for key in RULE_ONLY_EXPERIENCE_KEYS}
    if np.isin(shown["x_old"][:, 1:3], _token_ids()[1]).any():  # pragma: no cover - experience checked it
        raise SeqskillHarnessError("a replay row holds a reserved operator")
    result = {key: value for key, value in record.items() if key != "n_new_right"}
    result.update(source=RULE_ONLY_EXPERIENCE_SOURCE, variant=RULE_ONLY, content_sha256=params_sha256(shown),
                  n_new=0, n_new_withheld=int(record["n_new"]))
    return shown, result


def new_operator_rule(private: Mapping[str, Any]) -> dict[str, Any]:
    """The new operator's rule from the condition record, for rule_only (PREREG E8).

    The record is ``$WEIGHTCODE_PRIVATE/<dataset>/conditions/<condition>/dataset.json`` (``weightcode.data.store``); its bytes
    must hash to private.json ``condition_dataset_json_sha256``. Its ``condition`` must name the instance's condition
    and operator, and ``condition.rule`` must be a symbol map whose ``table`` is a permutation of the symbols; any
    other record is refused. Returns the table and the record's hash."""
    from weightcode.data import seqops as S
    from weightcode.data.store import dataset_json_path

    for name in ("dataset", "condition", "new_operator", "new_operator_id", "condition_dataset_json_sha256"):
        if name not in private:
            raise SeqskillHarnessError(f"private.json lacks {name}, so the new operator's rule cannot be read")
    condition = str(private["condition"])
    data = dataset_json_path(str(private["dataset"]), condition).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != private["condition_dataset_json_sha256"]:
        raise SeqskillHarnessError(f"the condition record of {condition} hashes to {digest}, not private.json "
                                   "condition_dataset_json_sha256")
    record = json.loads(data.decode("utf-8")).get("condition")
    if not isinstance(record, Mapping) or record.get("name") != condition:
        raise SeqskillHarnessError(f"the condition record does not describe condition {condition!r}")
    rule = record.get("rule")
    operator, operator_id = str(private["new_operator"]), int(private["new_operator_id"])
    if (record.get("operator") != operator or record.get("operator_id") != operator_id
            or not isinstance(rule, Mapping) or rule.get("token") != operator):
        raise SeqskillHarnessError(f"the condition record's rule is not bound to {operator} (id {operator_id})")
    if rule.get("kind") != SYMBOL_MAP_KIND:
        raise SeqskillHarnessError(f"rule_only states a symbol map; the condition's rule kind is {rule.get('kind')!r}")
    table = rule.get("table")
    if (not isinstance(table, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in table)
            or sorted(table) != list(range(S.N_SYMBOLS))):
        raise SeqskillHarnessError(f"condition.rule.table is not a permutation of the {S.N_SYMBOLS} symbols")
    return {"table": [int(v) for v in table], "condition_dataset_json_sha256": digest,
            "source": "condition record dataset.json, key condition.rule.table"}


def rule_only_meaning(table: list[int]) -> str:
    """The rule_only meaning of the new operator token, in the form of the old symbol maps' meanings."""
    return SYMBOL_MAP_MEANING + str([int(v) for v in table])


def rule_only_model_spec(model_spec: Mapping[str, Any], private: Mapping[str, Any],
                         rule: Mapping[str, Any]) -> dict[str, Any]:
    """The rule_only model_spec.json (PREREG E8): ``model_spec`` with the new operator token's meaning
    "reserved operator token; absent from all training rows (its embedding row is untrained)" turned into
    "operator, symbol map: s -> table[s], table = [...]; absent from all training rows (its embedding row is
    untrained)", the table being ``rule["table"]`` (``new_operator_rule``). Exactly one token must have the new
    operator's name and id, and its meaning must start with ``RESERVED_MEANING_PREFIX``. Nothing else changes."""
    result = copy.deepcopy(dict(model_spec))
    tokens = result.get("input", {}).get("tokens") if isinstance(result.get("input"), dict) else None
    if not isinstance(tokens, list):
        raise SeqskillHarnessError("model_spec.json has no input.tokens list")
    operator, operator_id = str(private["new_operator"]), int(private["new_operator_id"])
    found = [row for row in tokens if isinstance(row, dict) and row.get("name") == operator]
    if len(found) != 1 or found[0].get("id") != operator_id:
        raise SeqskillHarnessError(f"model_spec.json does not list {operator} once with id {operator_id}")
    meaning = found[0].get("meaning")
    if not isinstance(meaning, str) or not meaning.startswith(RESERVED_MEANING_PREFIX):
        raise SeqskillHarnessError(f"model_spec.json describes {operator} as {meaning!r}, not as a reserved token")
    found[0]["meaning"] = rule_only_meaning(list(rule["table"])) + "; " + meaning[len(RESERVED_MEANING_PREFIX):]
    return result


def session_record(private: Mapping[str, Any], variant: str | None = None,
                   rule: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``seqskill`` section of session.json: what the ledger needs to cut V_check, and the base error counts.

    rule_only adds ``variant`` and ``new_operator_rule`` (the table and the condition record's hash, from
    ``new_operator_rule``); the standard section has neither key. A scanskill instance (variant scan_rule) records its
    rule, condition and primitive, the part sizes, the base errors and the old_check file (path, file hash and
    content hash), which ``check_parts`` loads beside the session's V_check."""
    missing = [name for name in required_fields(private) if name not in private]
    if missing:
        raise SeqskillHarnessError(f"private.json lacks {missing}")
    if check_instance_variant(private, variant) == SCAN:
        if rule is not None:
            raise SeqskillHarnessError("a scan_rule session states its rule in model_spec.json; it takes no table")
        return {
            "rule": SCAN_RULE,
            "variant": SCAN,
            "condition": str(private["condition"]),
            "primitive": list(private["primitive"]),
            "primitive_row": private["primitive_row"],
            "n_vcheck_new": int(private["n_vcheck_new"]),
            "n_vcheck_old": int(private["n_vcheck_old"]),
            "n_base_errors_vcheck_new": int(private["n_base_errors_vcheck_new"]),
            "n_base_errors_vcheck_old": int(private["n_base_errors_vcheck_old"]),
            "old_check_path": str(private["oldcheck_path"]),
            "old_check_sha256": str(private["oldcheck_sha256"]),
            "old_check_content_sha256": str(private["oldcheck_content_sha256"]),
        }
    record: dict[str, Any] = {
        "rule": SEQSKILL_RULE,
        "new_operator": str(private["new_operator"]),
        "new_operator_id": int(private["new_operator_id"]),
        "n_vcheck_new": int(private["n_vcheck_new"]),
        "n_vcheck_old": int(private["n_vcheck_old"]),
        "n_base_errors_vcheck_new": int(private["n_base_errors_vcheck_new"]),
        "n_base_errors_vcheck_old": int(private["n_base_errors_vcheck_old"]),
    }
    if check_variant(variant) == STANDARD:
        if rule is not None:
            raise SeqskillHarnessError("a standard seqskill session states no rule")
        return record
    if rule is None:
        raise SeqskillHarnessError("a rule_only seqskill session records the rule it states")
    record.update(variant=RULE_ONLY, new_operator_rule={"table": list(rule["table"]),
                                                        "condition_dataset_json_sha256":
                                                            str(rule["condition_dataset_json_sha256"])})
    return record


def check_parts(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """V_check cut by part (``weightcode.evalkit.seqskill.seqskill_check_parts``), with the part sizes of ``record``; the
    new part must hold the new operator alone and the old part no reserved operator."""
    from weightcode.evalkit.seqskill import seqskill_check_parts

    if record.get("rule") == SCAN_RULE:
        return _scan_check_parts(arrays, record)
    if record.get("rule") != SEQSKILL_RULE:
        raise SeqskillHarnessError(f"the session's seqskill record names rule {record.get('rule')!r}")
    check_variant(record.get("variant"))  # the cut is the same in every variant
    parts = seqskill_check_parts(arrays)
    sizes = (len(parts["y_new"]), len(parts["y_old"]))
    if sizes != (int(record["n_vcheck_new"]), int(record["n_vcheck_old"])):
        raise SeqskillHarnessError(f"V_check cuts into {sizes} new and old rows; the instance records "
                                   f"{(record['n_vcheck_new'], record['n_vcheck_old'])}")
    _check_rows(parts["x_new"], parts["y_new"], int(record["new_operator_id"]), "the new part of V_check")
    _check_rows(parts["x_old"], parts["y_old"], None, "the old part of V_check")
    return parts


def _scan_check_parts(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The two check parts of a scan_rule session: ``arrays`` (the session's V_check, new rows only) and the old_check
    file the record names, read once from bytes that must hash to its recorded file hash and content hash
    (``weightcode.evalkit.scanskill.check_parts`` then checks each part's ``part`` array); sizes as recorded."""
    import io

    from weightcode.evalkit.scanskill import check_parts as scan_check_parts

    if record.get("variant") != SCAN:
        raise SeqskillHarnessError(f"a {SCAN_RULE} record names variant {record.get('variant')!r}, not {SCAN!r}")
    data = Path(str(record["old_check_path"])).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["old_check_sha256"]:
        raise SeqskillHarnessError("the old_check file does not hash to the session's old_check_sha256")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        old = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if params_sha256(old) != record["old_check_content_sha256"]:
        raise SeqskillHarnessError("the old_check content does not hash to the session's old_check_content_sha256")
    parts = scan_check_parts(arrays, old)
    sizes = (len(parts["y_new"]), len(parts["y_old"]))
    if sizes != (int(record["n_vcheck_new"]), int(record["n_vcheck_old"])):
        raise SeqskillHarnessError(f"the check parts hold {sizes} new and old rows; the instance records "
                                   f"{(record['n_vcheck_new'], record['n_vcheck_old'])}")
    return parts


def readout(instance: Instance, child: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
    """``seqskill_readout`` of ``child`` on V_select (``readouts_seqskill``) and, per part (i to iv, the new parts
    pooled and all rows), the change in the fields of a dev readout: n, base_acc, child_acc, delta_acc, fixed and
    broken, where accuracy is exact match per row. A scanskill instance gives ``scan_readout`` (``readouts_scan``)
    and its two parts, new and old V_select."""
    if is_scan(instance.private):
        from weightcode.evalkit.scanskill import scan_readout

        scan = scan_readout(instance, dict(child), batch_size)
        parts = {part: {"n": scan[part]["n"], "base_acc": scan[part]["em_base"], "child_acc": scan[part]["em_child"],
                        "delta_acc": scan[part]["delta"], "fixed": scan[part]["fixed"], "broken": scan[part]["broken"]}
                 for part in ("new", "old")}
        return {"readouts_scan": scan, "parts": parts}
    from weightcode.evalkit.seqskill import seqskill_readout

    full = seqskill_readout(instance, dict(child), batch_size)
    parts = {part: {"n": full[part]["n"], "base_acc": full[part]["em_base"], "child_acc": full[part]["em_child"],
                    "delta_acc": full[part]["delta"], "fixed": full[part]["fixed"], "broken": full[part]["broken"]}
             for part in READOUT_PARTS}
    return {"readouts_seqskill": full, "parts": parts}


def private_paths(private: Mapping[str, Any]) -> list[str]:
    """The host paths recorded in a seqskill private.json (``PRIVATE_PATH_FIELDS``; a scanskill one's
    ``SCAN_PRIVATE_PATH_FIELDS``)."""
    fields = SCAN_PRIVATE_PATH_FIELDS if is_scan(private) else PRIVATE_PATH_FIELDS
    return [str(private[name]) for name in fields if private.get(name)]


__all__ = [
    "CHECK_SET", "CHECK_UNIT", "EXPERIENCE_KEYS", "EXPERIENCE_SOURCE", "INSTANCE_ID_RE", "PRIVATE_PATH_FIELDS",
    "READOUT_PARTS", "REQUIRED_PRIVATE_FIELDS", "RESERVED_MEANING_PREFIX", "RULE_ONLY", "RULE_ONLY_EXPERIENCE_KEYS",
    "RULE_ONLY_EXPERIENCE_SOURCE", "SCAN", "SCAN_CHECK_SET", "SCAN_EXPERIENCE_SOURCE", "SCAN_ID_RE",
    "SCAN_PRIVATE_PATH_FIELDS", "SCAN_REQUIRED_PRIVATE_FIELDS", "SCAN_ROOT_NAME", "SCAN_RULE", "SEQSKILL_ROOT_NAME",
    "SEQSKILL_RULE", "STANDARD", "SYMBOL_MAP_MEANING", "SeqskillHarnessError", "VARIANTS", "active_index_row",
    "check_current", "check_instance_variant", "check_parts", "check_set_name", "check_variant", "experience",
    "experience_keys", "instance_dir", "is_scan", "is_scan_id", "is_seqskill", "is_seqskill_id", "leak_words",
    "new_operator_rule", "private_paths", "readout", "required_fields", "rule_only_experience", "rule_only_meaning",
    "rule_only_model_spec", "scan_experience", "session_record", "stale_error",
]
