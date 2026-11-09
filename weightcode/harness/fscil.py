"""The harness side of rule fscil-v1 (PREREG E9): the one access point of the editor harness to
``weightcode.evalkit.fscil``, which owns the rule (the instances and their parents, the evidence, the session mask, the check
parts, the limits, the V_select readout and the leak tokens).

An fscil instance is one few-shot class-incremental session t (2 to 9) on one lineage's current student. It runs in the
LEARN arm only and follows the fewshot-v1 path (``weightcode.harness.fewshot``) step by step, with the differences PREREG E9
states:

- the instance id is ``c`` + 8 hex and the instance lives under ``$WEIGHTCODE_PRIVATE/fscil_instances``; the runner refuses it
  unless ``check_fscil_current`` accepts it (active in its index, task.json and files as the current code writes them).
  private.json names the store's base student as ``base_model_id``; the student of the instance is its model.npz (the
  base at session 2, a lineage's output after that), so session.json records ``model_id`` = ``base_model_id`` and the
  student by its hash (``real_model_sha256``);
- every prediction is the masked one, the argmax over the outputs of the classes seen through session t
  (``masked_forward``); forward.py itself returns all 100 logits and model_spec.json states the mask;
- the experience is the instance's evidence.npz itself, in three parts: the 25 shots of session t (new), the 64 base
  replay rows (old) and the shots of sessions 2 to t - 1 with their session (prev, empty at t = 2). Every ``pred_*`` is
  re-checked under the session mask at batch ``EVALUATION_BATCH`` (``experience``);
- the limits are ``fscil_program_limits`` (the head ``fc.weight`` exempt from max_rel_frob; at most 20% of the entries
  changed, BatchNorm statistics counted) with max_ops at least the tensor count, which must equal the LEARN arm's
  ``applied_limits`` of the same task (``program_limits``);
- the check is ``fscil_check_parts``: S_new over the V_check rows of session t's classes, S_old over old_check (the base
  classes) followed by the V_check rows of the classes of sessions 2 to t - 1, both under the session mask and against
  the current student at batch ``EVALUATION_BATCH`` (``check_oracle``, as ``make_fscil_oracle``). The session ledger
  cuts the parts itself, from V_check (session.json ``vcheck_path``, its file hash) and old_check (the ``fscil``
  section, ``session_record``), each checked by bytes and by content, so the MCP server needs no store variable;
- the readout is ``fscil_readout`` on V_select and old_select under the session mask, child against the current
  student (``readout``), and ``fscil_accuracy`` gives one model's absolute readout at a session (``accuracy``);
- the leak scan adds ``fscil_leak_tokens`` (the dataset tokens and words, the store roots, the instance chain's ids, the
  base model id and the library's session and instance ids) to the text scan (``leak_words``); the sandbox self-test
  keeps the base strings, since the runs root is a prefix of every session directory it sees;
- the kept rule is joint-retain-v1 with the floor of session t given in rows (``retention_floor_rows``), which PREREG E9
  computes outside the harness from R, d and the session's old-part sizes;
- the library is the lineage's kept programs of earlier sessions (LEARN v3 format, ``library_entries``): each source
  session ran on an instance of the target's chain, kept a checked program under its kept rule, and is labelled by its
  session.

There is no decoy, no corruption condition and no clean companion. ``weightcode.evalkit.fscil`` imports torch (through
``weightcode.models.export``), so it is imported inside the functions that need it: an evidence-v1 session never loads it.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256
from weightcode.common.paths import private_dir

FSCIL_RULE = "fscil-v1"  # weightcode.evalkit.fscil.FSCIL_RULE; test_harness_fscil checks that the two agree
INSTANCE_ID_RE = re.compile(r"^c[0-9a-f]{8}$")  # weightcode.evalkit.fscil.INSTANCE_ID_RE, checked the same way
FSCIL_ROOT_NAME = "fscil_instances"  # weightcode.evalkit.fscil.ROOT_NAME
EXPERIENCE_SOURCE = "instance_fscil_v1"
EXPERIENCE_KEYS = ("x_new", "y_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old",
                   "x_prev", "y_prev", "pred_prev", "session_prev", "split_ids_prev")  # = evalkit EVIDENCE_KEYS
EXPERIENCE_PARTS = ("new", "old", "prev")
EVALUATION_BATCH = 256  # weightcode.evalkit.fscil.EVALUATION_BATCH: the check, the experience check and the readouts
READOUT_PARTS = ("new", "prev", "old", "seen")
CHECK_SET = "old-class"  # the rows behind score_old, as the retention kept rule names them
CHECK_UNIT = "images"
# private.json fields the harness reads from an fscil instance; an instance without one of them is refused.
REQUIRED_PRIVATE_FIELDS = (
    "instance_id",
    "split_role",
    "dataset",
    "arch",
    "seed",
    "evidence_rule",
    "session",
    "n_seen",
    "new_classes",
    "prev_classes",
    "parent_kind",
    "parent_params_sha256",
    "parent_instance",
    "base_model_id",
    "base_params_sha256",
    "head_tensors",
    "vcheck_path",
    "vcheck_sha256",
    "vcheck_content_sha256",
    "oldcheck_path",
    "oldcheck_sha256",
    "oldcheck_content_sha256",
    "n_vcheck_new",
    "n_vcheck_old",
    "n_base_errors_vcheck_new",
    "n_base_errors_vcheck_old",
    "n_evidence_new",
    "n_evidence_old",
    "n_evidence_prev",
    "model_sha256",
    "forward_sha256",
    "task_sha256",
    "evidence_content_sha256",
    "model_spec_sha256",
)
# private.json fields that hold host paths; every one present is a forbidden string of the session.
PRIVATE_PATH_FIELDS = ("vevidence_path", "vcheck_path", "vselect_path", "oldcheck_path", "oldselect_path",
                       "oldreplay_path", "base_model_source_path")
LIBRARY_LABEL_TEMPLATE = "session {session}"
LIBRARY_KEEPS = ("complete", "timed_out", "crashed")  # weightcode.harness.finalize.KEEPS_FINAL_PATCH


class FscilHarnessError(RuntimeError):
    """An fscil instance or session record is inconsistent, so no LEARN view, check or readout is built from it."""


def is_fscil_id(instance_id: object) -> bool:
    """True for an id of the fscil store's form (``c`` and 8 hex digits)."""
    return isinstance(instance_id, str) and INSTANCE_ID_RE.fullmatch(instance_id) is not None


def is_fscil(private: Mapping[str, Any]) -> bool:
    """True when private.json says the instance follows rule fscil-v1."""
    return private.get("evidence_rule") == FSCIL_RULE


def instance_dir(instance_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/fscil_instances/<instance_id>``, the path of ``weightcode.evalkit.fscil.fscil_instance_dir`` computed
    without creating the store root, so that resolving an id writes nothing; refuses an id of another form."""
    if not is_fscil_id(instance_id):
        raise FscilHarnessError(f"{instance_id!r} is not an fscil instance id ('c' and 8 hex digits)")
    return private_dir() / FSCIL_ROOT_NAME / instance_id


def check_current(path: str | Path) -> dict[str, Any]:
    """``weightcode.evalkit.fscil.check_fscil_current``: the instance's task, or ``StaleFscilInstance``."""
    from weightcode.evalkit.fscil import check_fscil_current

    return check_fscil_current(path)


def stale_error() -> type[Exception]:
    """The evalkit exception for an instance that is not active or not current (and every other fscil refusal)."""
    from weightcode.evalkit.fscil import FscilError

    return FscilError


def _index_path() -> Path:
    return private_dir() / FSCIL_ROOT_NAME / "index.json"


def active_index_row(instance_id: str) -> dict[str, Any] | None:
    """The instance's row among the active rows of the fscil index, or None (read without creating the store root)."""
    path = _index_path()
    if not path.is_file():
        return None
    index = json.loads(path.read_text(encoding="utf-8"))
    rows = index.get("instances", []) if isinstance(index, dict) else []
    found = [dict(row) for row in rows if isinstance(row, dict) and row.get("instance_id") == instance_id]
    return found[0] if found else None


def chain(instance_id: str) -> list[str]:
    """``weightcode.evalkit.fscil.instance_chain``: the instance and its ancestors, newest first, ending at session 2."""
    from weightcode.evalkit.fscil import instance_chain

    return instance_chain(instance_id)


def leak_words(extra: Sequence[str] = ()) -> list[str]:
    """``weightcode.evalkit.fscil.fscil_leak_tokens(extra)``: the dataset tokens and words, the store roots and ``extra``
    (instance, session and model ids), lower case; the instance build scans its files for the same."""
    from weightcode.evalkit.fscil import fscil_leak_tokens

    return fscil_leak_tokens(list(extra))


def scan(text: str, extra: Sequence[str] = ()) -> list[str]:
    """``weightcode.evalkit.fscil.fscil_scan``: the leak tokens found in ``text`` (substrings, case-insensitive)."""
    from weightcode.evalkit.fscil import fscil_scan

    return fscil_scan(text, list(extra))


def session_extra_tokens(private: Mapping[str, Any], library_ids: Sequence[str] = ()) -> list[str]:
    """The ids an fscil session forbids beyond its own instance id: every instance of its chain, the store's base
    model and the library's source sessions and instances (``library_ids``)."""
    ids = [*chain(str(private["instance_id"])), str(private["base_model_id"]), *library_ids]
    return sorted({str(item) for item in ids if item})


def program_limits(task_limits_json: Mapping[str, Any], n_tensors: int) -> Any:
    """``weightcode.evalkit.fscil.fscil_program_limits`` of the task's limits with max_ops raised to the student's tensor
    count (PREREG E9: max_ops is the tensor count), which must equal the LEARN arm's ``applied_limits`` of the same
    task (the limits a session applies); a difference is a harness fault, so the view is refused."""
    import dataclasses

    from weightcode.evalkit.fscil import fscil_program_limits
    from weightcode.harness.arms import applied_limits

    rule = fscil_program_limits(task_limits_json)
    limits = dataclasses.replace(rule, max_ops=max(rule.max_ops, int(n_tensors)))
    if limits != applied_limits(task_limits_json, "LEARN", n_tensors):
        raise FscilHarnessError("fscil_program_limits differs from the LEARN arm's applied limits")
    return limits


def head_tensors(spec: Mapping[str, Any]) -> list[str]:
    """``weightcode.evalkit.fscil.head_tensors``: the cosine head, the last layer of model_spec.json."""
    from weightcode.evalkit.fscil import head_tensors as rule_head

    return rule_head(spec)


def masked(forward: Any, session: int) -> Any:
    """``weightcode.evalkit.fscil.masked_forward``: the logits of the classes seen through ``session``."""
    from weightcode.evalkit.fscil import masked_forward

    return masked_forward(forward, int(session))


def _int_list(private: Mapping[str, Any], key: str, allow_empty: bool = False) -> list[int]:
    values = private.get(key)
    if (not isinstance(values, list) or (not values and not allow_empty)
            or any(isinstance(v, bool) or not isinstance(v, int) for v in values)):
        raise FscilHarnessError(f"private.json {key} must be a list of class indices, got {values!r}")
    return [int(v) for v in values]


def _session(private: Mapping[str, Any]) -> int:
    from weightcode.evalkit.fscil import FIRST_SESSION, LAST_SESSION, n_seen, prev_classes, session_classes

    t = private.get("session")
    if isinstance(t, bool) or not isinstance(t, int) or not FIRST_SESSION <= t <= LAST_SESSION:
        raise FscilHarnessError(f"private.json session must be an int in {FIRST_SESSION}..{LAST_SESSION}, got {t!r}")
    if (_int_list(private, "new_classes") != session_classes(t)
            or _int_list(private, "prev_classes", allow_empty=True) != prev_classes(t)
            or private.get("n_seen") != n_seen(t)):
        raise FscilHarnessError(f"private.json new_classes, prev_classes or n_seen are not those of session {t}")
    return t


def experience(instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The LEARN experience of an fscil instance and its host-side record.

    The arrays are the instance's evidence under ``EXPERIENCE_KEYS``. Every ``pred_*`` must be the student's masked
    prediction at the session (batch ``EVALUATION_BATCH``; an empty prev part is not run). ``y_new`` must hold the
    session's classes, ``N_SHOTS`` each; ``y_old`` only base classes; ``y_prev`` the classes of sessions 2 to t - 1,
    ``N_SHOTS`` each, with ``session_prev`` the session of each; the split ids are unique over the three parts.
    """
    from weightcode.evalkit.fscil import N_BASE_CLASSES, N_SHOTS, prev_classes, session_classes, session_of_class
    from weightcode.evalkit.patch_check import predict

    private = instance.private
    if not is_fscil(private):
        raise FscilHarnessError(f"{instance.instance_id} follows {private.get('evidence_rule')!r}, not {FSCIL_RULE}")
    t = _session(private)
    evidence = instance.evidence
    if sorted(evidence) != sorted(EXPERIENCE_KEYS):
        raise FscilHarnessError(f"{instance.instance_id}: evidence keys {sorted(evidence)} are not the LEARN keys "
                                f"{sorted(EXPERIENCE_KEYS)}")
    arrays = {key: np.ascontiguousarray(evidence[key]) for key in EXPERIENCE_KEYS}
    forward = masked(instance.forward, t)
    n_wrong: dict[str, int] = {}
    for part in EXPERIENCE_PARTS:
        x, y, shown = arrays[f"x_{part}"], arrays[f"y_{part}"], arrays[f"pred_{part}"]
        if not (len(x) == len(y) == len(shown) == len(arrays[f"split_ids_{part}"])):
            raise FscilHarnessError(f"{instance.instance_id}: the {part} part has unequal lengths")
        if len(x) == 0:
            n_wrong[part] = 0
            continue
        pred = predict(forward, instance.params, x, EVALUATION_BATCH)
        if not np.array_equal(pred, shown):
            raise FscilHarnessError(f"{instance.instance_id}: pred_{part} disagrees with the masked student on "
                                    f"{int(np.count_nonzero(pred != shown))} of {len(y)} rows")
        n_wrong[part] = int(np.count_nonzero(pred != y))
    for part, classes in (("new", session_classes(t)), ("prev", prev_classes(t))):
        labels, counts = np.unique(arrays[f"y_{part}"], return_counts=True)
        if labels.tolist() != classes or any(int(count) != N_SHOTS for count in counts):
            held = dict(zip(labels.tolist(), counts.tolist()))
            raise FscilHarnessError(f"{instance.instance_id}: y_{part} holds {held}; session {t} needs classes "
                                    f"{classes}, {N_SHOTS} each")
    if len(arrays["y_old"]) == 0 or np.any(arrays["y_old"] >= N_BASE_CLASSES):
        raise FscilHarnessError(f"{instance.instance_id}: y_old must hold base classes only")
    sessions = np.array([session_of_class(int(c)) for c in arrays["y_prev"]], dtype=np.int64)
    if not np.array_equal(arrays["session_prev"].astype(np.int64), sessions):
        raise FscilHarnessError(f"{instance.instance_id}: session_prev is not the session of each prev row")
    ids = [*arrays["split_ids_new"].tolist(), *arrays["split_ids_old"].tolist(), *arrays["split_ids_prev"].tolist()]
    if len(set(ids)) != len(ids):
        raise FscilHarnessError(f"{instance.instance_id}: the split ids are not unique")
    counts = {part: int(len(arrays[f"y_{part}"])) for part in EXPERIENCE_PARTS}
    if (counts["new"], counts["old"], counts["prev"]) != tuple(
            int(private[f"n_evidence_{part}"]) for part in EXPERIENCE_PARTS):
        raise FscilHarnessError(f"{instance.instance_id}: the evidence parts number {counts}; private.json says "
                                "otherwise")
    record = {
        "file": "experience.npz",
        "source": EXPERIENCE_SOURCE,
        "rule": FSCIL_RULE,
        "content_sha256": params_sha256(arrays),
        "evidence_content_sha256": private["evidence_content_sha256"],
        "student_params_sha256": instance.params_sha256,
        "session": t,
        "n_seen": int(private["n_seen"]),
        "n_new": counts["new"],
        "n_new_fail": n_wrong["new"],
        "n_old": counts["old"],
        "n_old_fail": n_wrong["old"],
        "n_prev": counts["prev"],
        "n_prev_fail": n_wrong["prev"],
        "per_class": N_SHOTS,
        "new_classes": session_classes(t),
        "prev_classes": prev_classes(t),
        "predictions": f"masked argmax over outputs 0 .. {int(private['n_seen']) - 1}, batch {EVALUATION_BATCH}",
    }
    if record["content_sha256"] != private["evidence_content_sha256"]:
        raise FscilHarnessError(f"{instance.instance_id}: the experience differs from the recorded evidence")
    return arrays, record


def _read_npz(path: Path) -> tuple[bytes, dict[str, np.ndarray]]:
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return data, {key: archive[key] for key in archive.files}


def vcheck_rows(private: Mapping[str, Any]) -> int:
    """The row count of the instance's V_check file, once its bytes hash to private.json ``vcheck_sha256`` and its
    content to ``vcheck_content_sha256``; session.json records it as ``n_vcheck`` (private.json has no such field)."""
    data, arrays = _read_npz(Path(str(private["vcheck_path"])))
    if hashlib.sha256(data).hexdigest() != private["vcheck_sha256"]:
        raise FscilHarnessError(f"{private['vcheck_path']}: file hash differs from private.json vcheck_sha256")
    if params_sha256(arrays) != private["vcheck_content_sha256"]:
        raise FscilHarnessError(f"{private['vcheck_path']}: content hash differs from private.json "
                                "vcheck_content_sha256")
    return int(len(arrays["y"]))


def session_record(private: Mapping[str, Any]) -> dict[str, Any]:
    """The ``fscil`` section of session.json: what the ledger needs to cut the check parts (old_check's path and its
    file and content hashes, V_check's content hash), the session and its classes, and the base error counts."""
    missing = [name for name in REQUIRED_PRIVATE_FIELDS if name not in private]
    if missing:
        raise FscilHarnessError(f"private.json lacks {missing}")
    t = _session(private)
    return {
        "rule": FSCIL_RULE,
        "session": t,
        "n_seen": int(private["n_seen"]),
        "new_classes": _int_list(private, "new_classes"),
        "prev_classes": _int_list(private, "prev_classes", allow_empty=True),
        "n_vcheck_new": int(private["n_vcheck_new"]),
        "n_vcheck_old": int(private["n_vcheck_old"]),
        "n_base_errors_vcheck_new": int(private["n_base_errors_vcheck_new"]),
        "n_base_errors_vcheck_old": int(private["n_base_errors_vcheck_old"]),
        "vcheck_content_sha256": str(private["vcheck_content_sha256"]),
        "oldcheck_path": str(private["oldcheck_path"]),
        "oldcheck_sha256": str(private["oldcheck_sha256"]),
        "oldcheck_content_sha256": str(private["oldcheck_content_sha256"]),
        "check_batch": EVALUATION_BATCH,
        "head_tensors": [str(name) for name in private["head_tensors"]],
        "base_model_id": str(private["base_model_id"]),
        "parent_kind": str(private["parent_kind"]),
        "parent_instance": private["parent_instance"],
        "parent_params_sha256": str(private["parent_params_sha256"]),
    }


def check_parts(v_check: Mapping[str, np.ndarray], old_check: Mapping[str, np.ndarray],
                record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The session's check parts (``weightcode.evalkit.fscil.fscil_check_parts``), after the content hashes of V_check and
    old_check are checked against ``record``; the part sizes must be the recorded ones."""
    from weightcode.evalkit.fscil import fscil_check_parts

    if record.get("rule") != FSCIL_RULE:
        raise FscilHarnessError(f"the session's fscil record names rule {record.get('rule')!r}")
    if params_sha256(dict(v_check)) != record.get("vcheck_content_sha256"):
        raise FscilHarnessError("V_check content differs from the session's vcheck_content_sha256")
    if params_sha256(dict(old_check)) != record.get("oldcheck_content_sha256"):
        raise FscilHarnessError("old_check content differs from the session's oldcheck_content_sha256")
    parts = fscil_check_parts(v_check, old_check, int(record["session"]))
    sizes = (len(parts["y_new"]), len(parts["y_old"]))
    if sizes != (int(record["n_vcheck_new"]), int(record["n_vcheck_old"])):
        raise FscilHarnessError(f"the check cuts into {sizes} new and old rows; the instance records "
                                f"{(record['n_vcheck_new'], record['n_vcheck_old'])}")
    return parts


def load_check_parts(record: Mapping[str, Any], vcheck_path: str | Path, vcheck_sha256: str) -> dict[str, np.ndarray]:
    """``check_parts`` of the session's V_check and old_check files, each read once and checked by its file hash."""
    arrays = {}
    for name, path, expected in (("V_check", vcheck_path, vcheck_sha256),
                                 ("old_check", record["oldcheck_path"], record["oldcheck_sha256"])):
        data, arrays[name] = _read_npz(Path(str(path)))
        if hashlib.sha256(data).hexdigest() != expected:
            raise FscilHarnessError(f"{name} file {path}: hash differs from the recorded {expected}")
    return check_parts(arrays["V_check"], arrays["old_check"], record)


def check_oracle(parts: Mapping[str, np.ndarray], forward: Any, base_params: Mapping[str, np.ndarray], budget: int,
                 session: int) -> Any:
    """evalkit's ``JointPatchCheckOracle`` on the two check parts under the session mask, batch ``EVALUATION_BATCH``,
    as ``make_fscil_oracle`` builds it."""
    from weightcode.evalkit.patch_check import JointPatchCheckOracle

    return JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"],
                                 masked(forward, session), base_params, budget, EVALUATION_BATCH)


# ---------------------------------------------------------------------------------------------------------------
# E9 v2: check score fscil-cb-v1 and kept rule fscil-retain-v2 (PREREG "E9 v2"), computed by weightcode.evalkit.fscil


FSCIL_V1 = "v1"
FSCIL_V2 = "v2"
FSCIL_VERSIONS = (FSCIL_V1, FSCIL_V2)
KEPT_RULE_V2 = "fscil-retain-v2"  # weightcode.evalkit.fscil.KEPT_RULE_V2; test_harness_fscil_v2 checks that the two agree
CHECK_SCORE_V2 = "fscil-cb-v1"  # weightcode.evalkit.fscil.CHECK_SCORE_RULE, checked the same way
R_POINTS_V2 = -1.0  # PREREG E9 v2: R = -1.0 per session (V3's R)
V2_CHECK_SET = "base-class"
V2_SHOWN_FIELDS = ("S", "S_new", "S_prev", "S_base", "B_c", "d_c", "floor_c", "eligible")
"""What ``patch_check`` and ``status`` show of a v2 check (points; d_c a fraction; eligible: B_c >= floor_c)."""
V2_COUNT_FIELDS = ("fixed_base", "broken_base", "n_base", "n_seen_classes")
V2_REPLAY_FIELDS = (*V2_SHOWN_FIELDS, *V2_COUNT_FIELDS, "per_class", "check_score")
"""The v2 fields of a check record, compared on replay besides the row-level fields of the joint oracle."""
V2_LIBRARY_FIELDS = ("S", "S_new", "S_prev", "S_base", "B_c", "d_c", "floor_c")


def check_version(version: object) -> str:
    """``version`` as one of ``FSCIL_VERSIONS`` (None is v1)."""
    if version is None:
        return FSCIL_V1
    if version not in FSCIL_VERSIONS:
        raise FscilHarnessError(f"the fscil version is one of {FSCIL_VERSIONS}, not {version!r}")
    return str(version)


def v2_base_rows() -> tuple[int, int]:
    """(n_c, n_s) of fscil-retain-v2: the base-class rows of old_check (V_check's base part) and of old_select
    (V_select's), from the store's dataset.json class counts; 1800 and 1800 on the real store."""
    from weightcode.evalkit.fscil import DATASET, read_dataset_meta

    splits = read_dataset_meta(DATASET)["splits"]
    return (int(np.sum(splits["old_check"]["class_counts"])), int(np.sum(splits["old_select"]["class_counts"])))


def kept_rule_v2(n_c: int, n_s: int) -> dict[str, Any]:
    """The session.json ``kept_rule`` of an E9 v2 session: fscil-retain-v2 on check score fscil-cb-v1, R = -1.0, with
    the base-row counts behind the floor (``v2_base_rows``)."""
    if isinstance(n_c, bool) or isinstance(n_s, bool) or int(n_c) <= 0 or int(n_s) <= 0:
        raise FscilHarnessError(f"fscil-retain-v2 needs positive base-row counts, not n_c={n_c!r}, n_s={n_s!r}")
    return {"name": KEPT_RULE_V2, "check_score": CHECK_SCORE_V2, "r_points": R_POINTS_V2, "n_c": int(n_c),
            "n_s": int(n_s), "check_set": V2_CHECK_SET}


def v2_rule_for(private: Mapping[str, Any]) -> dict[str, Any]:
    """The ``kept_rule_v2`` record of an fscil instance's session: n_c and n_s from ``v2_base_rows``, with the
    instance's old_check file (checked by its file hash) holding exactly n_c rows. Read before a session writes."""
    n_c, n_s = v2_base_rows()
    data, old_check = _read_npz(Path(str(private["oldcheck_path"])))
    if hashlib.sha256(data).hexdigest() != private["oldcheck_sha256"]:
        raise FscilHarnessError(f"{private['oldcheck_path']}: file hash differs from private.json oldcheck_sha256")
    if len(old_check["y"]) != n_c:
        raise FscilHarnessError(f"old_check holds {len(old_check['y'])} rows; the store's base check rows are {n_c}")
    return kept_rule_v2(n_c, n_s)


def session_kept_v2(session: Mapping[str, Any]) -> dict[str, Any] | None:
    """The fscil-retain-v2 kept rule of a session record, checked against ``kept_rule_v2``, or None for a session
    under another kept rule (or none)."""
    rule = session.get("kept_rule")
    if not isinstance(rule, Mapping) or rule.get("name") != KEPT_RULE_V2:
        return None
    if "fscil" not in session:
        raise FscilHarnessError(f"kept rule {KEPT_RULE_V2} is for an fscil-v1 instance")
    try:
        expected = kept_rule_v2(rule["n_c"], rule["n_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FscilHarnessError(f"the {KEPT_RULE_V2} record {dict(rule)!r} is malformed: {exc}") from None
    if dict(rule) != expected:
        raise FscilHarnessError(f"the {KEPT_RULE_V2} record {dict(rule)!r} is not {expected!r}")
    return expected


def _cb_oracle_class() -> type:
    from weightcode.common.identity import count_changed
    from weightcode.evalkit.metrics import summarize
    from weightcode.evalkit.patch_check import (
        JOINT_PARTS,
        BudgetExceeded,
        JointPatchCheckOracle,
        predict,
        validate_child,
    )

    class CbCheckOracle(JointPatchCheckOracle):
        """``JointPatchCheckOracle`` whose every query also carries fscil-cb-v1 (``cb_check_score`` on the joint
        rows, in ``cb_check_rows``' order) and the fscil-retain-v2 floor_c and eligibility of the child.

        The row-level fields (score, fixed, broken, the per-part fields) are the parent's, computed from the same
        predictions: the child's predictions are computed once per part and shared by both scores."""

        def __init__(self, parts: Mapping[str, np.ndarray], forward: Any, base_params: Mapping[str, np.ndarray],
                     budget: int, session: int, rule: Mapping[str, Any]) -> None:
            from weightcode.evalkit.fscil import cb_check_rows

            super().__init__(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"], masked(forward, session),
                             base_params, budget, EVALUATION_BATCH)
            self._session = int(session)
            self._rule = dict(rule)
            self._cb_y, self._cb_part = cb_check_rows(self._y["new"], self._y["old"], self._session)
            n_base = int(np.count_nonzero(self._cb_part == "old"))
            if n_base != int(self._rule["n_c"]):
                raise FscilHarnessError(f"the check holds {n_base} base-class rows; the kept rule's n_c is "
                                        f"{self._rule['n_c']}")
            self._cb_base = np.concatenate([self._base_pred[part] for part in JOINT_PARTS])

        def query(self, child_params: Mapping[str, np.ndarray], patch_id: str) -> dict[str, object]:
            """``JointPatchCheckOracle.query`` with the v2 fields; validation failures raise before any budget is
            consumed."""
            from weightcode.evalkit.fscil import cb_check_score, retain_v2_floor

            child = validate_child(self._base, child_params)
            if any(entry["patch_id"] == patch_id for entry in self._log):
                raise ValueError(f"patch id {patch_id!r} was already checked")
            if self._used >= self._budget:
                raise BudgetExceeded(f"patch-check budget of {self._budget} queries is spent")
            child_pred = {part: predict(self._forward, child, self._x[part], self._batch_size) for part in JOINT_PARTS}
            parts = {part: summarize(self._y[part], self._base_pred[part], child_pred[part]).as_dict()
                     for part in JOINT_PARTS}
            cb = cb_check_score(self._cb_base, np.concatenate([child_pred[part] for part in JOINT_PARTS]), self._cb_y,
                                self._cb_part, self._session)
            floor = retain_v2_floor(float(cb["d_c"]), float(self._rule["r_points"]), int(self._rule["n_c"]),
                                    int(self._rule["n_s"]))
            pooled = {key: sum(int(parts[part][key]) for part in JOINT_PARTS)  # type: ignore[call-overload]
                      for key in ("score", "fixed", "broken", "n", "n_base_errors")}
            if (pooled["fixed"], pooled["broken"]) != (cb["fixed"], cb["broken"]):  # pragma: no cover - same rows
                raise FscilHarnessError("the check score and the row counts disagree on the check's rows")
            query_index = self._used
            self._used += 1
            base_right = pooled["n"] - pooled["n_base_errors"]
            entry: dict[str, object] = {
                "query_index": query_index,
                "patch_id": patch_id,
                "child_sha256": params_sha256(child),
                "n_changed": count_changed(self._base, child),
                **pooled,
                "base_acc": base_right / pooled["n"],
                "child_acc": (base_right + pooled["score"]) / pooled["n"],
                **{f"{key}_{part}": value for part in JOINT_PARTS for key, value in parts[part].items()
                   if value is not None},
                "check_score": cb["rule"],
                **{key: cb[key] for key in ("S", "S_new", "S_prev", "S_base", "B_c", "d_c", *V2_COUNT_FIELDS)},
                "floor_c": floor,
                "eligible": bool(float(cb["B_c"]) >= floor),
                "per_class": cb["per_class"],
                "queries_remaining": self.queries_remaining,
            }
            self._log.append(entry)
            return dict(entry)

    return CbCheckOracle


def cb_check_oracle(parts: Mapping[str, np.ndarray], forward: Any, base_params: Mapping[str, np.ndarray], budget: int,
                    session: int, rule: Mapping[str, Any]) -> Any:
    """The check of an E9 v2 session: ``check_oracle``'s parts, mask, batch and row-level fields, plus fscil-cb-v1
    and floor_c and eligible under ``rule`` (a ``kept_rule_v2`` record) on every query."""
    return _cb_oracle_class()(parts, forward, base_params, budget, int(session), rule)


def make_cb_oracle(instance: Instance | str, budget: int, rule: Mapping[str, Any]) -> Any:
    """``cb_check_oracle`` of an fscil instance against its student, its parts cut as the session ledger cuts them
    (``load_check_parts``: file and content hashes, part sizes); the lineage's checks with no LLM call."""
    from weightcode.evalkit.fscil import load_fscil_instance

    resolved = load_fscil_instance(instance)
    record = session_record(resolved.private)
    parts = load_check_parts(record, resolved.private["vcheck_path"], resolved.private["vcheck_sha256"])
    return cb_check_oracle(parts, resolved.forward, resolved.params, budget, int(record["session"]), rule)


def readout(instance: Instance, child: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """``fscil_readout`` of ``child`` against the instance's student under the session mask (``readouts_fscil``) and,
    per part (session t's classes, the earlier incremental classes, the base classes, all seen classes), the fields of
    a dev readout: n, base_acc, child_acc, delta_acc, fixed and broken (an accuracy over zero rows is None)."""
    from weightcode.evalkit.fscil import fscil_readout

    full = fscil_readout(instance, dict(child), EVALUATION_BATCH)
    parts = {part: {"n": full[f"n_{part}"], "base_acc": full[f"acc_{part}_base"],
                    "child_acc": full[f"acc_{part}_child"], "delta_acc": full[f"delta_{part}"],
                    "fixed": full[f"fixed_{part}"], "broken": full[f"broken_{part}"]} for part in READOUT_PARTS}
    return {"readouts_fscil": full, "parts": parts}


def accuracy(params: Mapping[str, np.ndarray], session: int) -> dict[str, Any]:
    """``fscil_accuracy``: one model's absolute masked readout at ``session`` (1 reads the base classes only)."""
    from weightcode.evalkit.fscil import fscil_accuracy

    return fscil_accuracy(dict(params), int(session), batch_size=EVALUATION_BATCH)


def private_paths(private: Mapping[str, Any]) -> list[str]:
    """The host paths recorded in an fscil private.json (``PRIVATE_PATH_FIELDS`` and the parent's model file)."""
    paths = [str(private[name]) for name in PRIVATE_PATH_FIELDS if private.get(name)]
    created = private.get("created_by")
    if isinstance(created, Mapping) and created.get("parent_params_path"):
        paths.append(str(created["parent_params_path"]))
    return paths


# ---------------------------------------------------------------------------------------------------------------
# The LEARN task view


def _named(indices: Sequence[int], names: Sequence[str]) -> str:
    return ", ".join(f"{index} ({names[index]})" for index in indices)


def _earlier(t: int) -> str:
    return "session 2" if t == 3 else f"sessions 2 to {t - 1}"


def fscil_statement(names: Sequence[str], session: int) -> str:
    """The LEARN statement of session ``session``: the session, the mask, the classes added now and before, the fact
    that the earlier sessions' updates are in the classifier, the three experience parts and the program to write."""
    from weightcode.evalkit.fscil import N_BASE_CLASSES, N_CLASSES, N_REPLAY, N_SHOTS, N_WAYS, n_seen, prev_classes
    from weightcode.evalkit.fscil import session_classes

    t = int(session)
    new, prev, k = session_classes(t), prev_classes(t), n_seen(t)
    history = (f" The classes {_named(prev, names)} were added in {_earlier(t)}, and the classifier given here is the "
               "one those sessions' updates produced." if prev else
               " No session before this one added classes, so the classifier given here is the one trained on the base "
               "classes.")
    prev_text = (f", the {N_SHOTS} labelled images of each class added in the earlier sessions, with their session,"
                 if prev else "")
    return (
        "Make a trained image classifier recognise new classes from a few labelled images while it keeps the classes "
        f"it already recognises. The classifier has {N_CLASSES} outputs, one per class. Its prediction is the "
        f"highest-scoring output among the classes seen so far, outputs 0 to {k - 1}; the other outputs are ignored. "
        f"It was first trained only on images of the base classes 0 to {N_BASE_CLASSES - 1}. New classes arrive in "
        f"sessions of {N_WAYS} classes with {N_SHOTS} labelled images each. This is session {t}: the classes "
        f"{_named(new, names)} are added now.{history} The experience file holds the {N_SHOTS} labelled images of each "
        f"new class, {N_REPLAY} labelled images of the base classes{prev_text} and the classifier's predictions under "
        "this rule. Write a program that computes an additive update of the classifier's weights from the classifier "
        "and its experience, without training it, so that the updated classifier also classifies images of the new "
        "classes correctly while the images of the classes seen before that it already classifies correctly stay "
        "correct."
    )


FSCIL_READOUT = (
    "each query runs the program once on this classifier and its experience, applies the update within the limits and "
    "returns three numbers, each against the unedited classifier and under the prediction rule of this session: "
    "score_new = fixed - broken on a held-out check set of images of the classes added in this session, score_old = "
    "fixed - broken on a held-out check set of images of the classes seen before (the base classes and the classes of "
    "the earlier sessions), and score = score_new + score_old (S_joint); fixed counts images the unedited classifier "
    "got wrong that the updated one gets right, broken counts images it got right that the updated one gets wrong; no "
    "per-image information"
)
FSCIL_PROGRAM_RULE_FIRST = "in this task the evidence is experience.npz, and its labels are y_new, y_old and y_prev"


def v2_numbers(rule: Mapping[str, Any]) -> dict[str, str]:
    """The numbers of a ``kept_rule_v2`` record as the v2 texts state them: R, R - 0.11, the base rows n_c and n_s,
    and d_zero, the base flip rate from which floor_c is 0 (floor_c = R - 0.11 at d_c = 0)."""
    r, n_c, n_s = float(rule["r_points"]), int(rule["n_c"]), int(rule["n_s"])
    d_zero = ((0.11 - r) / 164.5) ** 2 / (1.0 / n_c + 1.0 / n_s)
    return {"R": f"{r:.1f}", "R_shift": f"{r - 0.11:.2f}", "n_c": str(n_c), "n_s": str(n_s),
            "d_zero": f"{d_zero:.3f}"}


def v2_floor_formula(rule: Mapping[str, Any]) -> str:
    """floor_c's formula with the rule's numbers: ``min(0, R - 0.11 + 164.5 sqrt(d_c / n_c + d_c / n_s))``."""
    v = v2_numbers(rule)
    return f"min(0, R - 0.11 + 164.5 sqrt(d_c / {v['n_c']} + d_c / {v['n_s']})) with R = {v['R']}"


def v2_class_change(subject: str) -> str:
    """The per-class change that every v2 score averages, worded for ``subject`` (the classifier before the update)."""
    return (f"for each class seen so far, its change is (fixed - broken) / n over its check images, where fixed counts "
            f"its check images {subject} got wrong that the updated one gets right, broken those {subject} got right "
            f"that the updated one gets wrong, and n is its number of check images")


def readout_v2(rule: Mapping[str, Any]) -> str:
    """The task.json ``patch_check.readout`` of an E9 v2 session (``FSCIL_READOUT``'s place)."""
    v = v2_numbers(rule)
    return (
        "each query runs the program once on this classifier and its experience, applies the update within the limits "
        "and returns, against the unedited classifier and under the prediction rule of this session, scores in "
        "percentage points on held-out check sets of images of every class seen so far (the classes added in this "
        "session, the classes of the earlier sessions and the base classes): "
        + v2_class_change("the unedited classifier") + "; S = 100 x the mean of the changes over the seen classes, the "
        "change of the class-balanced accuracy over the classes seen so far; S_new, S_prev and S_base are the parts of "
        "S from the classes added in this session, the classes of the earlier sessions and the base classes, and they "
        "sum to S; B_c = 100 x the mean of the changes over the base classes, the change of the base classes' "
        f"class-balanced accuracy; d_c = (fixed + broken) / {v['n_c']} over the {v['n_c']} check images of the base "
        f"classes, the program's own base flip rate; floor_c = {v2_floor_formula(rule)}, the program's own floor on "
        "B_c; eligible is whether B_c >= floor_c; no per-image or per-class information"
    )


def final_selection_v2(rule: Mapping[str, Any]) -> str:
    """The task.json ``patch_check.final_selection`` of an E9 v2 session (kept rule fscil-retain-v2)."""
    return ("kept at the end: among the checked programs with B_c >= floor_c (eligible) and the unedited classifier "
            "(S = 0), the update computed by the one with the highest S; ties go to the earlier query, and a tie with "
            "the unedited classifier to the unedited classifier; a program with S <= 0, a program that is not eligible "
            "and an unchecked program are never kept")


def kept_section_v2(rule: Mapping[str, Any]) -> str:
    """The body of the system prompt's "What is kept" under fscil-retain-v2 (unwrapped; ``weightcode.harness.prompts``
    wraps it)."""
    v = v2_numbers(rule)
    return (
        "After the session, among the programs you checked with `patch_check` whose B_c is at least their own floor_c, "
        "the update of the one with the highest S is kept. Every score is in percentage points on held-out check sets "
        "of images of every class seen so far, the classes added in this session, those of the earlier sessions and "
        "the base classes, all scored under the prediction rule of this session: "
        + v2_class_change("the original model") + ". S = 100 x the mean of the changes over all classes seen so far: "
        "the change of the class-balanced accuracy over the seen classes. S_new, S_prev and S_base are the parts of S "
        "from the classes added in this session, the classes of the earlier sessions and the base classes; they sum "
        "to S. B_c = 100 x the mean of the changes over the base classes: the change of the base classes' "
        f"class-balanced accuracy. floor_c = {v2_floor_formula(rule)}, where d_c = (fixed + broken) / {v['n_c']} over "
        f"the {v['n_c']} check images of the base classes is the program's own base flip rate, so every program has "
        f"its own floor: floor_c rises with d_c, from {v['R_shift']} at d_c = 0 to 0 at d_c = {v['d_zero']} and "
        "above. The check sets are held out and you never see their images. A program with B_c below its floor_c is "
        "never kept, whatever its S. If no checked program with B_c of at least its floor_c has an S > 0, the "
        "original model is kept. Unchecked programs are never kept."
    )


def _prev_description(t: int) -> str:
    from weightcode.evalkit.fscil import N_SHOTS

    if t == 2:
        return ("labelled images of classes added in earlier sessions: no session before this one added classes, so "
                "the prev arrays are empty")
    return (f"labelled images of the classes added in {_earlier(t)}, the {N_SHOTS} of each class that its session "
            "gave, in session order, with the session of each image in session_prev")


def _program_section() -> dict[str, Any]:
    """``weightcode.harness.experience._program_section`` for an incremental session: the same interface, with the forward
    worded for the mask and the transfer for the later sessions of the same classifier (PREREG E9: a kept program is
    run again on later sessions)."""
    from weightcode.harness.experience import _program_section as base_section

    section = base_section()
    section["student"] = {
        "params": "the classifier's weights as in model.npz: {tensor name: float32 array}",
        "forward": "forward(params, x) -> logits [N, number of classes], the function in forward.py, with a logit for "
                   "every class; the prediction is the argmax over the outputs of the classes seen so far "
                   "(student['layout']['prediction'])",
        "layout": "the content of model_spec.json",
    }
    section["transfer"] = (
        "the program may be run unchanged in later sessions of this classifier, on the classifier those sessions start "
        "from and on their own experience and prediction rule (student['layout']['prediction']), so it must compute "
        "the update from the student and the experience it receives, find the tensors and the classes it changes from "
        "the layout and the experience, and not contain this classifier's numbers")
    return section


def learn_task_fscil(task: Mapping[str, Any], shown_sha256: str, limits: Any,
                     record: Mapping[str, Any], kept_rule: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The task.json of a LEARN trial on an fscil instance (``FSCIL_RULE``): ``learn_task_fewshot`` for session t.

    The statement, the three experience parts, the readout, the program's student and transfer and the first line of
    the program rule are worded for the session; the task's session, class lists and prediction rule are kept; the
    final selection, the invalid-program rule and the training-free rule are E2's; the limits are
    ``program_limits`` (``limits``). ``record`` is the experience record of ``experience``. ``kept_rule``, a
    ``kept_rule_v2`` record (E9 v2), replaces the readout and the final selection by ``readout_v2`` and
    ``final_selection_v2``; None is E9's task.json, unchanged.
    """
    from weightcode.harness.experience import (
        LEARN_FINAL_SELECTION,
        LEARN_INVALID,
        LEARN_PROGRAM_RULE,
        ExperienceError,
        program_limits_view,
    )

    required = ("classes", "session", "n_seen", "base_classes", "prev_classes", "new_classes", "prediction_rule",
                "patch_check", "limits", "limit_definitions", "training_free_rule")
    missing = [key for key in required if key not in task]
    if missing or "parent_sha256" in task:
        raise ExperienceError(f"the fscil task.json cannot give the LEARN view: missing {missing}, or it already has "
                              "parent_sha256")
    t = int(task["session"])
    if record.get("session") != t or list(task["new_classes"]) != record.get("new_classes") or list(
            task["prev_classes"]) != record.get("prev_classes"):
        raise ExperienceError(f"the experience record is not that of session {t}")
    names = [str(name) for name in task["classes"]]
    if kept_rule is not None and kept_rule.get("name") != KEPT_RULE_V2:
        raise ExperienceError(f"an fscil task.json states kept rule {KEPT_RULE_V2} or E9's, not {kept_rule!r}")
    if LEARN_PROGRAM_RULE[0] != "in this task the evidence is experience.npz, and its labels are y_new and y_old":
        raise ExperienceError("the LEARN program rule no longer opens with the two-part evidence line")
    result = copy.deepcopy({key: value for key, value in task.items()
                            if key not in ("evidence", "patch_format", "patch_check")})
    shown, definitions = program_limits_view(limits)
    per_class = int(record["per_class"])
    result.update(
        parent_sha256=shown_sha256,
        statement=fscil_statement(names, t),
        experience={
            "new": {"keys": ["x_new", "y_new", "pred_new"], "count": int(record["n_new"]), "per_class": per_class,
                    "classes": list(record["new_classes"]),
                    "description": f"labelled images of the classes added in this session, {per_class} of each, in "
                                   "class order"},
            "old": {"keys": ["x_old", "y_old", "pred_old"], "count": int(record["n_old"]),
                    "classes": list(task["base_classes"]),
                    "description": "labelled images of the base classes, a fixed random draw from held-out images of "
                                   "those classes, the same in every session"},
            "prev": {"keys": ["x_prev", "y_prev", "pred_prev", "session_prev"], "count": int(record["n_prev"]),
                     "per_class": per_class, "classes": list(record["prev_classes"]),
                     "description": _prev_description(t)},
            "ids": {"keys": ["split_ids_new", "split_ids_old", "split_ids_prev"],
                    "description": "one opaque id per example, in x_new, x_old and x_prev order"},
        },
        program=_program_section(),
        patch_check={"budget_queries": task["patch_check"]["budget_queries"],
                     "readout": FSCIL_READOUT if kept_rule is None else readout_v2(kept_rule),
                     "invalid_programs": LEARN_INVALID,
                     "final_selection": LEARN_FINAL_SELECTION if kept_rule is None else final_selection_v2(kept_rule)},
        limits=shown,
        limit_definitions=definitions,
        training_free_rule={**copy.deepcopy(task["training_free_rule"]),
                            "program": [FSCIL_PROGRAM_RULE_FIRST, *LEARN_PROGRAM_RULE[1:]]},
    )
    return result


# ---------------------------------------------------------------------------------------------------------------
# The library: the lineage's kept programs of earlier sessions


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FscilHarnessError(f"{path} is missing") from None


def library_entries(session_ids: Sequence[str], sessions_dir: Path, *, target_instance_id: str,
                    version: str | None = None) -> list[dict[str, Any]]:
    """The checked library entries of an fscil session, in session order (LEARN v3 format, labelled by session).

    Each source is a non-dry-run LEARN session on an fscil instance of the target's chain (an ancestor, one per
    session), whose status keeps its final (complete, timed_out or crashed: the status is recorded with the entry) and
    whose final was chosen by its kept rule over checked programs: final kind checked, its kept rule joint-retain-v1
    with a floor that is not a placeholder, the kept check a consumed query whose program file hashes to the check's
    program_sha256, with the same scores as the selection. Its final child must be the student of the chain's next
    session (the lineage went on from it). ``version`` "v2" (PREREG E9 v2) takes sources under fscil-retain-v2
    instead, whose kept check is eligible with S > 0, and the entries carry the v2 scores (``V2_LIBRARY_FIELDS``).
    """
    from weightcode.harness.library import LIBRARY_PROGRAM_TEMPLATE, MAX_LIBRARY_PROGRAMS, PROGRAM_ARMS, _kept_check
    from weightcode.harness.retention import FLOOR_KIND_ROWS, KEPT_RULE_RETAIN, is_placeholder

    v2 = check_version(version) == FSCIL_V2
    if not session_ids:
        raise FscilHarnessError("a library needs at least one earlier session")
    if len(session_ids) > MAX_LIBRARY_PROGRAMS or len(set(session_ids)) != len(session_ids):
        raise FscilHarnessError(f"a library holds at most {MAX_LIBRARY_PROGRAMS} distinct sessions, not "
                                f"{list(session_ids)}")
    lineage = chain(target_instance_id)  # newest first; refuses an inactive target
    by_session: dict[int, str] = {}
    for instance_id in lineage:
        row = active_index_row(instance_id)
        if row is None:
            raise FscilHarnessError(f"chain instance {instance_id} is not active")
        by_session[int(row["session"])] = instance_id
    params_by_session = {int(active_index_row(i)["session"]): str(active_index_row(i)["parent_params_sha256"])
                         for i in lineage}
    sources = []
    for session_id in session_ids:
        root = Path(sessions_dir) / session_id
        session = _read_json(root / "session.json")
        result = _read_json(root / "session_result.json")
        selection = _read_json(root / "final" / "selection.json")
        instance_id = str(session.get("instance_id"))
        if session.get("arm") not in PROGRAM_ARMS or session.get("dry_run"):
            raise FscilHarnessError(f"{session_id}: arm {session.get('arm')!r}, dry run {session.get('dry_run')}; a "
                                    "library source is a real LEARN session")
        record = session.get("fscil")
        if not isinstance(record, Mapping) or record.get("rule") != FSCIL_RULE:
            raise FscilHarnessError(f"{session_id} did not run on an fscil-v1 instance")
        status = str(result.get("status"))
        if status not in LIBRARY_KEEPS:
            raise FscilHarnessError(f"{session_id} has status {status!r}; its final is not a kept final")
        t = int(record["session"])
        if instance_id == target_instance_id or by_session.get(t) != instance_id:
            raise FscilHarnessError(f"{session_id} ran on {instance_id}, which is not the session-{t} instance of the "
                                    f"chain of {target_instance_id}")
        kept = session.get("kept_rule")
        if v2:
            if session_kept_v2(session) is None:
                raise FscilHarnessError(f"{session_id}: its kept rule {kept!r} is not {KEPT_RULE_V2}")
        elif (not isinstance(kept, Mapping) or kept.get("name") != KEPT_RULE_RETAIN
                or kept.get("floor_kind") != FLOOR_KIND_ROWS or is_placeholder(kept.get("floor_source") or {})):
            raise FscilHarnessError(f"{session_id}: its kept rule {kept!r} is not joint-retain-v1 with a stated floor "
                                    "in rows")
        final = selection.get("final") or {}
        if (final.get("kind") != "checked" or final.get("is_base_noop")
                or result.get("final", {}).get("kind") != "checked"):
            raise FscilHarnessError(f"{session_id} kept the base model; it has no program to add")
        query_index = int(final["query_index"])
        try:
            check = _kept_check(root, query_index)
        except ValueError as exc:
            raise FscilHarnessError(str(exc)) from None
        if v2:
            from weightcode.evalkit.fscil import V2_TIE_POINTS

            if check.get("charged_failure") or check.get("eligible") is not True or float(check["S"]) <= V2_TIE_POINTS:
                raise FscilHarnessError(f"{session_id}: its kept check is not a valid pick of its kept rule")
        elif (check.get("charged_failure") or int(check["score_old"]) < int(kept["s_old_floor"])
                or int(check["score"]) <= 0):
            raise FscilHarnessError(f"{session_id}: its kept check is not a valid pick of its kept rule")
        program_path = root / "final" / str(final["patch_id"])
        if not program_path.is_file():
            raise FscilHarnessError(f"{session_id}: the kept program {program_path} is missing")
        source = program_path.read_bytes()
        if hashlib.sha256(source).hexdigest() != check.get("program_sha256"):
            raise FscilHarnessError(f"{session_id}: {program_path.name} does not hash to its check's program_sha256")
        if v2:
            scores = {key: final[key] for key in V2_LIBRARY_FIELDS}
            if any(check[key] != scores[key] for key in scores):
                raise FscilHarnessError(f"{session_id}: the selection's scores differ from its check's")
        else:
            scores = {key: int(final[key]) for key in ("score_new", "score_old", "score")}
            if any(int(check[key]) != scores[key] for key in scores):
                raise FscilHarnessError(f"{session_id}: the selection's scores differ from its check's")
        following = params_by_session.get(t + 1)
        if following is not None and following != final.get("child_sha256"):
            raise FscilHarnessError(f"{session_id}: its final child is not the student of the chain's session {t + 1}")
        sources.append((t, session_id, instance_id, status, query_index, check, program_path, scores, source))
    sessions = [item[0] for item in sources]
    if len(set(sessions)) != len(sessions):
        raise FscilHarnessError(f"two library sessions ran in the same session: {sessions}")
    entries = []
    for k, (t, session_id, instance_id, status, query_index, check, program_path, scores, source) in enumerate(
            sorted(sources), start=1):
        entries.append({
            "k": k,
            "label": LIBRARY_LABEL_TEMPLATE.format(session=t),
            "session": t,
            "file": LIBRARY_PROGRAM_TEMPLATE.format(k=k),
            **scores,
            "n_check_new": int(check["n_new"]),
            "n_check_old": int(check["n_old"]),
            "status": status,
            "source": {"session_id": session_id, "instance_id": instance_id, "session": t, "status": status,
                       "query_index": query_index, "program_file": program_path.name,
                       "program_sha256": check["program_sha256"], "program_path": str(program_path)},
            "_bytes": source,
        })
    return entries


def library_ids(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """The session and instance ids of the library's sources (forbidden strings of the target session)."""
    return sorted({str(entry["source"][key]) for entry in entries for key in ("session_id", "instance_id")})


def withhold(entries: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> list[dict[str, Any]]:
    """``entries`` with the file of every program whose text holds a forbidden token withheld: such an entry is listed
    with its scores and ``file`` None, and records the tokens found (the trial scan would refuse the session)."""
    from weightcode.evalkit.instances import scan_for_tokens

    shown = []
    for entry in entries:
        found = scan_for_tokens(entry["_bytes"].decode("utf-8", errors="replace"), list(tokens))
        shown.append({**entry, "file": None, "withheld": found} if found else dict(entry))
    return shown


def library_index(entries: Sequence[Mapping[str, Any]], version: str | None = None) -> dict[str, Any]:
    """The trial's ``library_index.json`` of an fscil session: labels, files and kept-check scores only (``version``
    "v2": the fscil-cb-v1 scores of each kept check, ``V2_LIBRARY_FIELDS``)."""
    from weightcode.harness.library import LIBRARY_SCHEMA

    if check_version(version) == FSCIL_V2:
        return {
            "schema": LIBRARY_SCHEMA,
            "about": ("The update programs kept in earlier sessions of this classifier, one per session, each as it "
                      "was written. For each: its session and the scores its own kept patch_check returned in that "
                      "session, in percentage points on the held-out check sets of the classes seen in that session: "
                      "S, the change of the class-balanced accuracy over those classes, its parts S_new, S_prev and "
                      "S_base (that session's classes, the classes of the sessions before it, the base classes), B_c, "
                      "the change of the base classes' class-balanced accuracy, d_c, the program's base flip rate "
                      "there, and floor_c, its floor on B_c there. file is null for a program that is not shown."),
            "programs": [{key: entry[key] for key in ("label", "session", "file", *V2_LIBRARY_FIELDS)}
                         for entry in entries],
        }
    return {
        "schema": LIBRARY_SCHEMA,
        "about": ("The update programs kept in earlier sessions of this classifier, one per session, each as it was "
                  "written. For each: its session and the scores its own kept patch_check returned in that session, "
                  "score_new on the held-out check set of the n_check_new images of that session's classes and "
                  "score_old on the held-out check set of the n_check_old images of the classes seen before it. file "
                  "is null for a program that is not shown."),
        "programs": [{key: entry[key] for key in ("label", "session", "file", "score_new", "score_old", "score",
                                                  "n_check_new", "n_check_old")} for entry in entries],
    }


def library_bullet(entries: Sequence[Mapping[str, Any]], index_file: str, version: str | None = None) -> str:
    """The task prompt's file-list bullet for an fscil library (``weightcode.harness.prompts.library_files_section``'s place,
    worded for the lineage's earlier sessions; ``version`` "v2" names the fscil-cb-v1 scores)."""
    files = [str(entry["file"]) for entry in entries if entry["file"]]
    sessions = [int(entry["session"]) for entry in entries]
    named = (f"session {sessions[0]}" if len(sessions) == 1
             else "sessions " + ", ".join(map(str, sessions[:-1])) + f" and {sessions[-1]}")
    if not files:
        shown = f"`{index_file}`: "
    elif len(files) == 1:
        shown = f"`{index_file}` and `{files[0]}`: "
    else:
        shown = f"`{index_file}` and " + ", ".join(f"`{name}`" for name in files[:-1]) + f" and `{files[-1]}`: "
    what = "the update program" if len(entries) == 1 else "the update programs"
    if check_version(version) == FSCIL_V2:
        return (f"- {shown}{what} kept in {named} of this classifier, each as it was written. `{index_file}` gives "
                "for each its session, its file and the scores that its own kept `patch_check` returned in that "
                "session, in points on the held-out check sets of the classes seen then: S, S_new, S_prev, S_base, "
                "B_c, d_c and floor_c.")
    return (f"- {shown}{what} kept in {named} of this classifier, each as it was written. `{index_file}` gives for "
            "each its session, its file and the scores that its own kept `patch_check` returned in that session: "
            "score_new on the held-out check set of that session's classes and score_old on the held-out check set of "
            "the classes seen before it.")


__all__ = [
    "CHECK_SCORE_V2", "CHECK_SET", "CHECK_UNIT", "EVALUATION_BATCH", "EXPERIENCE_KEYS", "EXPERIENCE_PARTS",
    "EXPERIENCE_SOURCE", "FSCIL_PROGRAM_RULE_FIRST", "FSCIL_READOUT", "FSCIL_ROOT_NAME", "FSCIL_RULE", "FSCIL_V1",
    "FSCIL_V2", "FSCIL_VERSIONS", "FscilHarnessError", "INSTANCE_ID_RE", "KEPT_RULE_V2", "LIBRARY_KEEPS",
    "LIBRARY_LABEL_TEMPLATE", "PRIVATE_PATH_FIELDS", "READOUT_PARTS", "REQUIRED_PRIVATE_FIELDS", "R_POINTS_V2",
    "V2_CHECK_SET", "V2_COUNT_FIELDS", "V2_LIBRARY_FIELDS", "V2_REPLAY_FIELDS", "V2_SHOWN_FIELDS", "accuracy",
    "active_index_row", "cb_check_oracle", "chain", "check_current", "check_oracle", "check_parts", "check_version",
    "experience", "final_selection_v2", "fscil_statement", "head_tensors", "instance_dir", "is_fscil", "is_fscil_id",
    "kept_rule_v2", "kept_section_v2", "leak_words", "learn_task_fscil", "library_bullet", "library_entries",
    "library_ids", "library_index", "load_check_parts", "make_cb_oracle", "masked", "private_paths", "program_limits",
    "readout", "readout_v2", "scan", "session_extra_tokens", "session_kept_v2", "session_record", "stale_error",
    "v2_base_rows", "v2_class_change", "v2_floor_formula", "v2_numbers", "v2_rule_for", "vcheck_rows", "withhold",
]
