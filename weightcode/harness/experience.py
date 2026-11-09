"""The LEARN experience and task view (HARNESS_SPEC section 5): what the update program learns from.

``experience.npz`` has two labelled parts with the student's predictions:

- new: the instance's own evidence-v1 sample, the same arrays M2 gets, as one part: ``x_new = [x_fail; x_rand]`` (16
  rows drawn from the student's errors on V_evidence of the condition, then 48 random rows), ``y_new``, ``pred_new``
  and ``split_ids_new`` (the evidence ids, fail rows first).
- old: the dataset's replay evidence (rule replay-v1, ``weightcode.harness.clean.load_replay``): 64 rows of the clean train
  split, one fixed draw per dataset shared by every model and condition, as ``x_old``, ``y_old``, the student's
  ``pred_old`` and the replay's opaque ``split_ids_old``. The draw record (split seed and stream, idx, row ids, content
  and pool hashes) goes to session.json only, never to a trial file.

No clean companion of any V_evidence, V_check or V_select row is in the experience, and ``build_experience`` refuses a
replay whose idx meets the idx of the instance's evidence rows (the train split is disjoint from V_evidence, so this
never happens with the evalkit replay; the check guards the rule). Trial assembly and the transfer runner
(``scripts/run_program_transfer.py``) both call ``build_experience``, so a transfer target gets its experience exactly
as a trial does. The condition is never named: the task view speaks of "a new input condition" and "clean examples from
the data the classifier was trained on".

The blind view (LEARN_BLIND, PREREG A4a): ``blind_trial_experience`` is the experience without the student's
predictions (``BLIND_HIDDEN_KEYS``), unless the session keeps them; a program still receives the whole experience.
``learn_task_blind`` is ``learn_task`` with the sentences that are false for that view replaced: the statement (no
predictions in the file, when they are hidden), the experience keys, what ``student["params"]`` and ``experience`` hold
for a program, and the added ``program.check_returns``, what a check returns (``BLIND_OUTPUT``). Every other field is
``learn_task``'s.

The rule view (PREREG K1, E11 only, ``weightcode.harness.e11``): ``rule_experience`` is the experience's old part alone
(``RULE_EXPERIENCE_KEYS``: x_old, y_old, pred_old, split_ids_old), cut after ``build_experience`` checked the whole
experience; no row of the new condition is in it. ``learn_task_rule`` is ``learn_task`` with the texts that speak of
new-condition rows replaced: the statement's sentence on the experience file becomes the condition's rule (the
PREREG K1 sentence) and a sentence saying that no image of the condition is in any file and that patch_check scores on
held-out images of the condition; the experience section loses its new part and names split_ids_old alone; the first
item of the program's training-free rule names y_old alone as the labels. Every other field is ``learn_task``'s.

The information ladder (PREREG, E11 gaussian_noise_s5 on the seed-100 student only, ``weightcode.harness.e11``), each view
cut after ``build_experience`` checked the whole experience, each changed text found exactly once or refused:

- U (``unlabelled_experience``, ``learn_task_unlabelled``): the experience without ``y_new``
  (``UNLABELLED_EXPERIENCE_KEYS``), its rows and their order E11's; the statement calls the condition rows unlabelled,
  the new part's keys drop y_new, its description no longer says how the rows were drawn (which of them are the
  student's errors) and its ``n_from_errors`` is removed, and the program's first training-free item names y_old alone
  as the labels.
- K0b and K0a (``named_experience``, ``learn_task_named``): no experience file and no array; a program gets an empty
  dict. The statement's sentence on the experience file becomes the capability sentence and a sentence that no image
  and no label is in any file; the experience section is removed; the program's experience, transfer and
  training-free items and K0b's readout lose the experience (``NAMED_*``). K0a has no ``patch_check`` section and a
  top-level ``final_selection`` (``NAMED_NOCHECK_FINAL_SELECTION``), and its task.json never says "check".
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.targets import deployment_splits, load_checked_split
from weightcode.harness.arms import PROGRAM_LIMIT_FIELDS
from weightcode.harness import fewshot, graphnew, policyskill, seqskill
from weightcode.harness.clean import load_replay, source_name
from weightcode.patch import Limits
from weightcode.sandbox.program import PROGRAM_WALL_S

EXPERIENCE_FILE = "experience.npz"
EXPERIENCE_SOURCE = "instance_evidence_and_replay_v1"
EXPERIENCE_KEYS = ("x_new", "y_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old")
EXPERIENCE_PARTS = ("new", "old")
PREDICT_BATCH = 1024
MODULE_STRINGS = ("HYPOTHESIS", "PREDICTED_EFFECT")

LEARN_STATEMENT = (
    "Make a trained image classifier handle a new input condition while it keeps what it already does. The "
    "experience file holds labelled examples from the new condition, on which the classifier makes errors, and "
    "labelled clean examples from the data it was trained on, with the classifier's predictions on both. Write a "
    "program that computes an additive update of the classifier's weights from the classifier and its experience, "
    "without training it, so that the updated classifier classifies more new-condition images correctly while the "
    "clean images it already classifies correctly stay correct."
)
LEARN_READOUT = (
    "each query runs the program once on this classifier and its experience, applies the update within the limits and "
    "returns three numbers, each against the unedited classifier: score_new = fixed - broken on a held-out check set of "
    "new-condition images, score_old = fixed - broken on a held-out check set of clean images, and score = score_new + "
    "score_old (S_joint); fixed counts images the unedited classifier got wrong that the updated one gets right, "
    "broken counts images it got right that the updated one gets wrong; no per-image information"
)
LEARN_INVALID = "a program that fails to run, returns invalid deltas or fails a limit consumes no query"
LEARN_FINAL_SELECTION = (
    "kept at the end: among the checked programs and the unedited classifier (score = 0), the update computed by the "
    "one with the highest score (S_joint); ties go to fewer changed parameters, then to the earlier query; a program "
    "with score <= 0 and an unchecked program are never kept"
)
LEARN_PROGRAM_RULE = (
    "in this task the evidence is experience.npz, and its labels are y_new and y_old",
    "the rules hold for the update program as for your own cells: the program computes its update in a fixed "
    "sequence of steps from forward passes, activations, weights and label-conditioned statistics of its experience",
    "forbidden in the program: gradients of any loss with respect to parameters, solvers or regression whose targets "
    "derive from experience labels, any .fit( call, and loops that update parameters to reduce a label-dependent loss",
    "one run of the program may score at most 8 explicit candidate settings against the experience labels",
    "every checked program is stored with the session record",
)
PROGRAM_LIMIT_DEFINITIONS = {
    "max_changed_frac": "changed entries over all parameter entries; an entry is changed if its float32 value after "
                        "the update differs from the parent's",
    "max_rel_frob": "per tensor, ||child - parent||_F / ||parent||_F, where child = float32(parent + delta)",
    "rel_frob_min_ndim": "smallest tensor rank that max_rel_frob applies to; tensors of lower rank are exempt",
    "rel_frob_exempt": "tensors that max_rel_frob does not apply to, whatever their rank",
}


class ExperienceError(RuntimeError):
    """The experience cannot be assembled consistently (a harness or data fault)."""


def evidence_idx(instance: Instance) -> np.ndarray:
    """Dataset idx of the instance's evidence rows (fail rows first), from its V_evidence and private.json rows."""
    rows = instance.private.get("evidence_row_ids")
    if not isinstance(rows, list):
        raise ExperienceError(f"{instance.instance_id}: private.json has no evidence_row_ids")
    splits = deployment_splits(str(instance.private["dataset"]), str(instance.private["condition"]))
    arrays = load_checked_split(splits.v_evidence, splits.v_evidence_sha256)
    positions = np.asarray(rows, dtype=np.int64)
    evidence = instance.evidence
    labels = np.concatenate([evidence["y_fail"], evidence["y_rand"]])
    if len(positions) != len(labels) or not np.array_equal(arrays["y"][positions], labels):
        raise ExperienceError(f"{instance.instance_id}: evidence_row_ids do not address the evidence rows")
    return np.asarray(arrays["idx"][positions])


def build_experience(instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The experience of ``instance`` (module docstring) and its host-side record; every prediction is re-checked."""
    evidence = instance.evidence
    n_fail, n_rand = len(evidence["x_fail"]), len(evidence["x_rand"])
    ids_new = np.asarray(evidence["split_ids"])
    if len(ids_new) != n_fail + n_rand:
        raise ExperienceError(f"evidence has {len(ids_new)} split_ids for {n_fail} fail and {n_rand} random rows")
    new = {
        "x_new": np.ascontiguousarray(np.concatenate([evidence["x_fail"], evidence["x_rand"]])),
        "y_new": np.ascontiguousarray(np.concatenate([evidence["y_fail"], evidence["y_rand"]]).astype(np.int64)),
        "pred_new": np.ascontiguousarray(np.concatenate([evidence["pred_fail"], evidence["pred_rand"]]).astype(np.int64)),
        "split_ids_new": np.ascontiguousarray(ids_new),
    }
    replay, record = load_replay(str(instance.private["dataset"]))
    x_old = replay["x"]
    if x_old.shape[1:] != new["x_new"].shape[1:] or x_old.dtype != new["x_new"].dtype:
        raise ExperienceError(f"replay inputs are {x_old.dtype}{x_old.shape[1:]}, the new-condition inputs "
                              f"{new['x_new'].dtype}{new['x_new'].shape[1:]}")
    shared = sorted(set(int(i) for i in replay["idx"]) & set(int(i) for i in evidence_idx(instance)))
    if shared:
        raise ExperienceError(f"{instance.instance_id}: replay rows share dataset idx {shared[:10]} with the evidence "
                              "rows; a clean view of an evidence image must never reach the editor")
    if set(replay["split_ids"].tolist()) & set(ids_new.tolist()):
        raise ExperienceError("a replay id equals an evidence id")
    predicted = predict(instance.forward, instance.params, new["x_new"], PREDICT_BATCH)
    if not np.array_equal(predicted, new["pred_new"]):
        raise ExperienceError(f"pred_new disagrees with the student on {int(np.sum(predicted != new['pred_new']))} rows")
    experience = {**new, "x_old": x_old, "y_old": replay["y"].astype(np.int64),
                  "pred_old": predict(instance.forward, instance.params, x_old, PREDICT_BATCH),
                  "split_ids_old": replay["split_ids"]}
    summary = {
        "file": EXPERIENCE_FILE, "source": EXPERIENCE_SOURCE, "content_sha256": params_sha256(experience),
        "n_new": n_fail + n_rand, "n_new_fail": n_fail, "n_old": int(len(x_old)),
        "replay": {**record, "arrays_sha256": params_sha256(replay), "source": source_name()},
        "replay_idx_disjoint_from_evidence_idx": True,
    }
    return experience, summary


BLIND_HIDDEN_KEYS = ("pred_new", "pred_old")
"""The student's outputs on the experience rows, which the blind trial leaves out unless the session keeps them."""
BLIND_STATEMENT_PREDICTIONS = ", with the classifier's predictions on both."
BLIND_PARAMS = ("the classifier's weights, {tensor name: float32 array} with the names, shapes and dtypes that "
                "model_shapes.json lists; the values are not in /work/trial, and the program receives them when "
                "patch_check runs it")
BLIND_EXPERIENCE_WITH_PREDICTIONS = ("the arrays of experience.npz and the classifier's predictions on its inputs, "
                                     "pred_new and pred_old (int64, in x_new and x_old order): {name: array}")
BLIND_OUTPUT = (
    "the scores and the run's exit status: ok (the program returned its deltas), error (it raised an exception), "
    "timeout (it did not return within the wall limit) or died (it ended without a result, for example at a memory or "
    "file-size limit or by exiting); nothing the program prints, raises or writes is returned, and a program that "
    "fails is told only its exit status and the stage at which it failed"
)


def blind_trial_experience(experience: Mapping[str, np.ndarray], predictions: bool) -> dict[str, np.ndarray]:
    """The experience a blind trial shows: ``experience`` without ``BLIND_HIDDEN_KEYS`` unless ``predictions``."""
    missing = [key for key in EXPERIENCE_KEYS if key not in experience]
    if missing or len(experience) != len(EXPERIENCE_KEYS):
        raise ExperienceError(f"the blind view needs the LEARN experience keys {list(EXPERIENCE_KEYS)}, got "
                              f"{sorted(experience)}")
    return {key: experience[key] for key in EXPERIENCE_KEYS if predictions or key not in BLIND_HIDDEN_KEYS}


def learn_task_blind(task: Mapping[str, Any], shown_sha256: str, limits: Limits, n_new: int, n_new_fail: int,
                     n_old: int, predictions: bool) -> dict[str, Any]:
    """The task.json of a LEARN_BLIND trial (module docstring); ``predictions``: the trial keeps ``pred_*``."""
    result = learn_task(task, shown_sha256, limits, n_new, n_new_fail, n_old)
    program = result["program"]
    program["student"]["params"] = BLIND_PARAMS
    program["check_returns"] = BLIND_OUTPUT
    if predictions:
        return result
    statement = str(result["statement"])
    if statement.count(BLIND_STATEMENT_PREDICTIONS) != 1:  # pragma: no cover - LEARN_STATEMENT is fixed above
        raise ExperienceError("the LEARN statement no longer names the predictions once")
    result["statement"] = statement.replace(BLIND_STATEMENT_PREDICTIONS, ".")
    for part in ("new", "old"):
        keys = result["experience"][part]["keys"]
        result["experience"][part]["keys"] = [key for key in keys if key not in BLIND_HIDDEN_KEYS]
    program["experience"] = BLIND_EXPERIENCE_WITH_PREDICTIONS
    return result


# The rule view (PREREG K1, ``weightcode.harness.e11``): the experience's old part alone, and the LEARN task.json with the
# sentences that speak of new-condition rows replaced. Each replaced text must occur exactly once, or the view is
# refused.
RULE_EXPERIENCE_KEYS = ("x_old", "y_old", "pred_old", "split_ids_old")
RULE_EXPERIENCE_SOURCE = "instance_evidence_and_replay_v1_rule"
RULE_STATEMENT_EXPERIENCE = (
    "The experience file holds labelled examples from the new condition, on which the classifier makes errors, and "
    "labelled clean examples from the data it was trained on, with the classifier's predictions on both."
)
RULE_STATEMENT_NO_IMAGE = (
    "No image of the condition is in any file: the experience file holds labelled clean examples from the data the "
    "classifier was trained on, with the classifier's predictions on them, and patch_check scores each checked "
    "program on held-out images of the condition and on held-out clean images."
)
RULE_PROGRAM_LABELS = ("in this task the evidence is experience.npz, and its labels are y_new and y_old",
                       "in this task the evidence is experience.npz, and its labels are y_old")
RULE_IDS_DESCRIPTION = "one opaque id per example, in x_old order"


def rule_experience(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]
                    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The rule-view experience (PREREG K1): the old part of ``build_experience``'s arrays alone, and its record.

    ``arrays`` and ``record`` are what ``build_experience`` returns, so every prediction of the whole experience was
    re-checked before the new part is withheld. The record keeps the replay record and states that no new row is
    shown (``n_new`` 0), how many were withheld and the content hash of the withheld part."""
    if sorted(arrays) != sorted(EXPERIENCE_KEYS) or record.get("source") != EXPERIENCE_SOURCE:
        raise ExperienceError("the rule-view experience is cut from the checked LEARN experience only")
    shown = {key: np.ascontiguousarray(arrays[key]) for key in RULE_EXPERIENCE_KEYS}
    withheld = {key: arrays[key] for key in EXPERIENCE_KEYS if key not in RULE_EXPERIENCE_KEYS}
    new_rows = {np.ascontiguousarray(row).tobytes() for row in withheld["x_new"]}
    if any(np.ascontiguousarray(row).tobytes() in new_rows for row in shown["x_old"]):
        raise ExperienceError("a shown clean row equals a withheld new-condition row")
    result = {key: value for key, value in record.items() if key != "n_new_fail"}
    result.update(source=RULE_EXPERIENCE_SOURCE, variant="rule", content_sha256=params_sha256(shown), n_new=0,
                  n_new_withheld=int(record["n_new"]), n_new_fail_withheld=int(record["n_new_fail"]),
                  withheld_keys=sorted(withheld), withheld_content_sha256=params_sha256(withheld),
                  learn_content_sha256=str(record["content_sha256"]))
    return shown, result


def _replace_rule_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ExperienceError(f"the LEARN task.json {what} does not say {old!r} once, so the rule view cannot be "
                              "derived from it")
    return text.replace(old, new)


def learn_task_rule(task: Mapping[str, Any], rule_sentence: str, n_old: int) -> dict[str, Any]:
    """The task.json of a rule-view trial (PREREG K1): ``task``, the LEARN task.json of ``learn_task``, with what
    speaks of new-condition rows replaced and nothing else changed.

    * ``statement``: its sentence on the experience file (``RULE_STATEMENT_EXPERIENCE``) becomes ``rule_sentence``
      followed by ``RULE_STATEMENT_NO_IMAGE``;
    * ``experience``: no ``new`` part; ``old`` unchanged (``n_old`` rows); ``ids`` lists ``split_ids_old`` alone;
    * ``training_free_rule.program``: its first item names y_old alone as the labels (``RULE_PROGRAM_LABELS``).

    Each replaced text must occur exactly once, and the rule sentence must not occur in ``task`` already."""
    if json.dumps(task).count(rule_sentence) != 0:
        raise ExperienceError("the LEARN task.json already states the rule")
    experience = task.get("experience")
    if not isinstance(experience, Mapping) or sorted(experience) != ["ids", "new", "old"]:
        raise ExperienceError("the rule view is derived from a LEARN task.json with the new, old and ids parts")
    if int(experience["old"]["count"]) != int(n_old) or experience["ids"]["keys"] != ["split_ids_new",
                                                                                      "split_ids_old"]:
        raise ExperienceError("the LEARN task.json experience does not describe the old part and both id arrays")
    result = copy.deepcopy(dict(task))
    result["statement"] = _replace_rule_once(str(task["statement"]), RULE_STATEMENT_EXPERIENCE,
                                             f"{rule_sentence} {RULE_STATEMENT_NO_IMAGE}", "statement")
    result["experience"] = {"old": copy.deepcopy(dict(experience["old"])),
                            "ids": {"keys": ["split_ids_old"], "description": RULE_IDS_DESCRIPTION}}
    program_rule = list(task["training_free_rule"]["program"])
    if program_rule.count(RULE_PROGRAM_LABELS[0]) != 1:
        raise ExperienceError(f"the LEARN training-free rule does not say {RULE_PROGRAM_LABELS[0]!r} once, so the rule "
                              "view cannot be derived from it")
    result["training_free_rule"]["program"] = [RULE_PROGRAM_LABELS[1] if item == RULE_PROGRAM_LABELS[0] else item
                                               for item in program_rule]
    text = json.dumps(result)
    if "x_new" in text or "y_new" in text or "pred_new" in text or "split_ids_new" in text:
        raise ExperienceError("the rule-view task.json still names a new-condition array")
    if text.count(rule_sentence) != 1:  # pragma: no cover - set once above
        raise ExperienceError("the rule-view task.json does not state the rule exactly once")
    return result


# The unlabelled view (U, the PREREG information ladder, ``weightcode.harness.e11``): the LEARN experience without y_new, and
# the LEARN task.json with the texts that call the condition rows labelled changed. Each changed text must occur
# exactly once, or the view is refused.
UNLABELLED_EXPERIENCE_KEYS = ("x_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old")
UNLABELLED_WITHHELD_KEYS = ("y_new",)
UNLABELLED_EXPERIENCE_SOURCE = "instance_evidence_and_replay_v1_unlabelled"
UNLABELLED_STATEMENT = ("The experience file holds labelled examples from the new condition",
                        "The experience file holds unlabelled examples from the new condition")
UNLABELLED_NEW_KEYS = ["x_new", "pred_new"]
UNLABELLED_ERROR_WORDS = ("n_from_errors", "the classifier's errors", "drawn at random from all",
                          "whether the classifier gets them right or not")
"""What says which condition rows are the student's errors; no U text may contain it. The rows and their order are
E11's (``n_new_fail`` of the ``n_new`` drawn from the errors, first), and that selection is not stated to the editor."""


def _replace_in_learn_task(text: str, old: str, new: str, what: str, view: str) -> str:
    if text.count(old) != 1:
        raise ExperienceError(f"the LEARN task.json {what} does not say {old!r} once, so the {view} view cannot be "
                              "derived from it")
    return text.replace(old, new)


def unlabelled_experience(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]
                          ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The unlabelled-view experience (U): ``build_experience``'s arrays without ``y_new``, and its record.

    Every prediction of the whole experience was re-checked by ``build_experience`` before the labels are withheld.
    The record keeps the counts (the same 64 condition rows in E11's order, the first 16 drawn from the student's
    errors; host-side only, the editor is not told the selection) and the replay record, and states the withheld key
    and its content hash."""
    if sorted(arrays) != sorted(EXPERIENCE_KEYS) or record.get("source") != EXPERIENCE_SOURCE:
        raise ExperienceError("the unlabelled-view experience is cut from the checked LEARN experience only")
    shown = {key: np.ascontiguousarray(arrays[key]) for key in UNLABELLED_EXPERIENCE_KEYS}
    withheld = {key: arrays[key] for key in UNLABELLED_WITHHELD_KEYS}
    result = dict(record)
    result.update(source=UNLABELLED_EXPERIENCE_SOURCE, variant="unlabelled", content_sha256=params_sha256(shown),
                  labels_new_shown=False, withheld_keys=list(UNLABELLED_WITHHELD_KEYS),
                  withheld_content_sha256=params_sha256(withheld), learn_content_sha256=str(record["content_sha256"]))
    return shown, result


def learn_task_unlabelled(task: Mapping[str, Any]) -> dict[str, Any]:
    """The task.json of an unlabelled-view trial (U): ``task``, the LEARN task.json of ``learn_task``, with what calls
    the condition rows labelled changed and nothing else.

    * ``statement``: "labelled examples from the new condition" becomes "unlabelled examples ..." (the clean examples
      stay labelled);
    * ``experience.new.keys``: x_new and pred_new, without y_new;
    * ``experience.new.description``: "examples from the new input condition", without the clause on how the rows
      were drawn (``learn_new_selection``), which says which rows are the student's errors; ``n_from_errors``
      removed;
    * ``training_free_rule.program``: its first item names y_old alone as the labels (``RULE_PROGRAM_LABELS``).

    Each changed text must occur exactly once, and no ``UNLABELLED_ERROR_WORDS`` may remain."""
    experience = task.get("experience")
    if not isinstance(experience, Mapping) or sorted(experience) != ["ids", "new", "old"]:
        raise ExperienceError("the unlabelled view is derived from a LEARN task.json with the new, old and ids parts")
    if list(experience["new"].get("keys", [])) != ["x_new", "y_new", "pred_new"]:
        raise ExperienceError("the LEARN task.json new part does not list x_new, y_new and pred_new")
    result = copy.deepcopy(dict(task))
    result["statement"] = _replace_in_learn_task(str(task["statement"]), *UNLABELLED_STATEMENT, "statement",
                                                 "unlabelled")
    new = result["experience"]["new"]
    new["keys"] = list(UNLABELLED_NEW_KEYS)
    try:
        n_fail, n_new = int(new.pop("n_from_errors")), int(new["count"])
    except (KeyError, TypeError, ValueError):
        raise ExperienceError("the LEARN task.json new part does not give its count and n_from_errors, so the "
                              "unlabelled view cannot be derived from it") from None
    new["description"] = _replace_in_learn_task(str(new["description"]), learn_new_selection(n_fail, n_new - n_fail),
                                                "", "new-part description", "unlabelled")
    program_rule = list(task["training_free_rule"]["program"])
    if program_rule.count(RULE_PROGRAM_LABELS[0]) != 1:
        raise ExperienceError(f"the LEARN training-free rule does not say {RULE_PROGRAM_LABELS[0]!r} once, so the "
                              "unlabelled view cannot be derived from it")
    result["training_free_rule"]["program"] = [RULE_PROGRAM_LABELS[1] if item == RULE_PROGRAM_LABELS[0] else item
                                               for item in program_rule]
    text = json.dumps(result)
    if "y_new" in text:
        raise ExperienceError("the unlabelled-view task.json still names y_new")
    found = [word for word in UNLABELLED_ERROR_WORDS if word in text]
    if found:
        raise ExperienceError(f"the unlabelled-view task.json still says which condition rows are errors: {found}")
    return result


# The no-data views (K0b ``named`` and K0a ``named_nocheck``, the PREREG information ladder, ``weightcode.harness.e11``): no
# experience file and no image of any kind; a program gets an empty experience. The LEARN task.json has the texts that
# speak of the experience changed, each found exactly once, and the statement names the capability.
NAMED_EXPERIENCE_SOURCE = "e11_named_no_data_v1"
NAMED_STATEMENT_EXPERIENCE = RULE_STATEMENT_EXPERIENCE
NAMED_STATEMENT_NO_DATA = (
    "No image and no label is in any file, of the new condition or clean: patch_check scores each checked program on "
    "held-out images of the new condition and on held-out clean images."
)
NAMED_NOCHECK_STATEMENT_NO_DATA = (
    "No image and no label is in any file, of the new condition or clean: the program you declare is scored after "
    "the session on held-out images of the new condition and on held-out clean images, and you see no score."
)
NAMED_STATEMENT_PROGRAM = ("from the classifier and its experience, without training it",
                           "from the classifier, without training it")
NAMED_PROGRAM_EXPERIENCE = "no array: an empty dict, since this task gives no example"
NAMED_TRANSFER = ("(student['layout']) and its own experience drawn by the same rules, so it must compute the update "
                  "from the student and the experience it receives,",
                  "(student['layout']), so it must compute the update from the student it receives,")
NAMED_READOUT = ("each query runs the program once on this classifier and its experience,",
                 "each query runs the program once on this classifier,")
NAMED_PROGRAM_RULES = (
    (RULE_PROGRAM_LABELS[0], "in this task no evidence is given: no example and no label"),
    ("from forward passes, activations, weights and label-conditioned statistics of its experience",
     "from forward passes, activations and weights"),
    ("solvers or regression whose targets derive from experience labels",
     "solvers or regression whose targets derive from labels"),
    ("score at most 8 explicit candidate settings against the experience labels",
     "score at most 8 explicit candidate settings against labels"),
)
NAMED_EXPERIENCE_WORDS = ("experience.npz", "experience file", "experience labels", "its experience",
                          "the experience", *EXPERIENCE_KEYS)
"""What a no-data task.json, task prompt and tool text never say: an experience file, its parts or its labels. The
program interface keeps ``update(student, experience)``, whose ``experience`` is then an empty dict."""
NAMED_NOCHECK_PROGRAM_STORED = ("every checked program is stored with the session record",
                                "every declared program is stored with the session record")
NAMED_NOCHECK_FINAL_SELECTION = (
    "kept at the end: the update of the program named in your last valid declaration (null: the unedited "
    "classifier), computed when you declared it, scored once after the session against the unedited classifier on "
    "held-out images of the new condition (score_new = fixed - broken) and on held-out clean images (score_old = "
    "fixed - broken), with score = score_new + score_old; it is kept only if score > 0 and score_old meets the floor "
    "of the system prompt, and otherwise the unedited classifier is kept; fixed counts images the unedited classifier "
    "got wrong that the updated one gets right, broken counts images it got right that the updated one gets wrong; "
    "no score is shown to you, and a program never validly declared is never kept"
)


def named_experience(record: Mapping[str, Any], variant: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The no-data view's experience (K0b, K0a): no array, and its record.

    ``record`` is ``build_experience``'s, so the whole LEARN experience was assembled and checked before every part
    is withheld; the record keeps the replay record and the content hash of what is withheld, and its
    ``content_sha256`` (of what is shown) is None, since nothing is."""
    if record.get("source") != EXPERIENCE_SOURCE:
        raise ExperienceError("the no-data view is cut from the checked LEARN experience only")
    shown: dict[str, np.ndarray] = {}
    result = {"file": None, "source": NAMED_EXPERIENCE_SOURCE, "variant": variant,
              "content_sha256": None, "n_new": 0, "n_old": 0,
              "n_new_withheld": int(record["n_new"]), "n_new_fail_withheld": int(record["n_new_fail"]),
              "n_old_withheld": int(record["n_old"]), "withheld_keys": sorted(EXPERIENCE_KEYS),
              "learn_content_sha256": str(record["content_sha256"]), "replay": dict(record["replay"]),
              "replay_idx_disjoint_from_evidence_idx": record.get("replay_idx_disjoint_from_evidence_idx")}
    return shown, result


def learn_task_named(task: Mapping[str, Any], capability_sentence: str, checks: bool) -> dict[str, Any]:
    """The task.json of a no-data trial: ``task``, the LEARN task.json of ``learn_task``, with what speaks of the
    experience changed. ``checks`` False is K0a: no ``patch_check`` section, and a top-level ``final_selection``.

    * ``statement``: the sentence on the experience file becomes ``capability_sentence`` and a sentence that no image
      and no label is in any file (``NAMED_STATEMENT_NO_DATA``, or ``NAMED_NOCHECK_STATEMENT_NO_DATA`` for K0a);
      "from the classifier and its experience" becomes "from the classifier";
    * ``experience``: removed;
    * ``program.experience``: an empty dict (``NAMED_PROGRAM_EXPERIENCE``); ``program.transfer``: the program gets
      the student alone (``NAMED_TRANSFER``);
    * ``patch_check.readout`` (K0b): runs on this classifier (``NAMED_READOUT``); K0a: ``patch_check`` removed and
      ``final_selection`` (``NAMED_NOCHECK_FINAL_SELECTION``) at the top level;
    * ``training_free_rule.program``: no evidence and no label (``NAMED_PROGRAM_RULES``); K0a also stores every
      declared program (``NAMED_NOCHECK_PROGRAM_STORED``).

    Each changed text must occur exactly once, and the capability sentence must not occur in ``task`` already."""
    if json.dumps(task).count(capability_sentence) != 0:
        raise ExperienceError("the LEARN task.json already names the capability")
    experience = task.get("experience")
    if not isinstance(experience, Mapping) or sorted(experience) != ["ids", "new", "old"]:
        raise ExperienceError("the no-data view is derived from a LEARN task.json with the new, old and ids parts")
    view = "no-data"
    result = copy.deepcopy(dict(task))
    no_data = NAMED_STATEMENT_NO_DATA if checks else NAMED_NOCHECK_STATEMENT_NO_DATA
    statement = _replace_in_learn_task(str(task["statement"]), NAMED_STATEMENT_EXPERIENCE,
                                       f"{capability_sentence} {no_data}", "statement", view)
    result["statement"] = _replace_in_learn_task(statement, *NAMED_STATEMENT_PROGRAM, "statement", view)
    del result["experience"]
    program = result["program"]
    program["experience"] = NAMED_PROGRAM_EXPERIENCE
    program["transfer"] = _replace_in_learn_task(str(program["transfer"]), *NAMED_TRANSFER, "program transfer", view)
    if checks:
        check = result["patch_check"]
        check["readout"] = _replace_in_learn_task(str(check["readout"]), *NAMED_READOUT, "patch-check readout", view)
    else:
        del result["patch_check"]
        result["final_selection"] = NAMED_NOCHECK_FINAL_SELECTION
    rules = [*NAMED_PROGRAM_RULES, *(() if checks else (NAMED_NOCHECK_PROGRAM_STORED,))]
    items = list(task["training_free_rule"]["program"])
    for old, new in rules:
        hits = [index for index, item in enumerate(items) if old in item]
        if len(hits) != 1 or items[hits[0]].count(old) != 1:
            raise ExperienceError(f"the LEARN training-free rule does not say {old!r} once, so the {view} view cannot "
                                  "be derived from it")
        items[hits[0]] = items[hits[0]].replace(old, new)
    result["training_free_rule"]["program"] = items
    text = json.dumps(result)
    for word in NAMED_EXPERIENCE_WORDS:
        if word in text:
            raise ExperienceError(f"the no-data task.json still names {word!r}")
    if not checks and "check" in text.lower():
        raise ExperienceError("the K0a task.json still says 'check'")
    if text.count(capability_sentence) != 1:  # pragma: no cover - set once above
        raise ExperienceError("the no-data task.json does not name the capability exactly once")
    return result


REL_FROB_NOT_APPLIED = "; max_rel_frob is null, so it applies to no tensor"


def program_limits_view(limits: Limits) -> tuple[dict[str, Any], dict[str, str]]:
    """The limits a program's deltas are held to (``PROGRAM_LIMIT_FIELDS``, plus ``rel_frob_exempt`` when it names
    tensors) and their definitions, for task.json.

    A limit that is None is shown as null and its definition says it is not applied. When ``max_rel_frob`` is None
    (the lifted rel-Frob limit of the PREREG E8 amendment), the two fields that say which tensors it applies to say
    that it applies to none (``REL_FROB_NOT_APPLIED``); with a rel-Frob limit the view is unchanged."""
    shown = {name: getattr(limits, name) for name in PROGRAM_LIMIT_FIELDS}
    definitions = dict(PROGRAM_LIMIT_DEFINITIONS)
    min_ndim = shown.get("rel_frob_min_ndim")
    exempt = "1-D tensors are exempt" if min_ndim == 2 else "tensors of lower rank are exempt"
    if limits.rel_frob_exempt:
        shown["rel_frob_exempt"] = list(limits.rel_frob_exempt)
        exempt += ", and so are the tensors in rel_frob_exempt"
    definitions["max_rel_frob"] = (f"per tensor of rank >= {min_ndim}, ||child - parent||_F / ||parent||_F, where "
                                   f"child = float32(parent + delta); {exempt}")
    for name, value in shown.items():
        if value is None:
            definitions[name] += "; null: this limit is not applied"
    if limits.max_rel_frob is None:
        for name in ("rel_frob_min_ndim", "rel_frob_exempt"):
            definitions[name] += REL_FROB_NOT_APPLIED
    return shown, definitions


LEARN_NEW_DESCRIPTION = "examples from the new input condition"


def learn_new_selection(n_new_fail: int, n_rand: int) -> str:
    """The clause of the LEARN new part's description that says how its rows were drawn, and so which of them are the
    student's errors (the first ``n_new_fail``); the unlabelled view (U) removes it."""
    return (f": the first {n_new_fail} drawn at random from all of the classifier's errors there, the other {n_rand} "
            "drawn at random from the remaining new-condition examples, whether the classifier gets them right or not")


def learn_task(task: Mapping[str, Any], shown_sha256: str, limits: Limits, n_new: int, n_new_fail: int,
               n_old: int) -> dict[str, Any]:
    """The task.json of a LEARN trial: the instance's task with the experience, program and joint-check sections."""
    required = ("statement", "classes", "evidence", "patch_check", "patch_format", "limits", "limit_definitions",
                "training_free_rule")
    missing = [key for key in required if key not in task]
    if missing or "parent_sha256" in task:
        raise ExperienceError(f"the instance task.json cannot give the LEARN view: missing {missing}, or it already "
                              "has parent_sha256")
    result = copy.deepcopy({key: value for key, value in task.items()
                            if key not in ("evidence", "patch_format", "patch_check")})
    shown, definitions = program_limits_view(limits)
    n_rand = n_new - n_new_fail
    result.update(
        parent_sha256=shown_sha256,
        statement=LEARN_STATEMENT,
        experience={
            "new": {"keys": ["x_new", "y_new", "pred_new"], "count": n_new, "n_from_errors": n_new_fail,
                    "description": LEARN_NEW_DESCRIPTION + learn_new_selection(n_new_fail, n_rand)},
            "old": {"keys": ["x_old", "y_old", "pred_old"], "count": n_old,
                    "description": "clean examples drawn at random from the data the classifier was trained on"},
            "ids": {"keys": ["split_ids_new", "split_ids_old"],
                    "description": "one opaque id per example, in x_new and x_old order"},
        },
        program=_program_section(),
        patch_check={"budget_queries": task["patch_check"]["budget_queries"], "readout": LEARN_READOUT,
                     "invalid_programs": LEARN_INVALID, "final_selection": LEARN_FINAL_SELECTION},
        limits=shown,
        limit_definitions=definitions,
        training_free_rule={**copy.deepcopy(task["training_free_rule"]), "program": list(LEARN_PROGRAM_RULE)},
    )
    return result


# Few-shot new classes (PREREG E2, rule fewshot-v1): the student has an output for every class but was trained only on
# the old ones; the experience is the instance's own evidence file, which already holds the LEARN keys.
FEWSHOT_RULE = fewshot.FEWSHOT_RULE
FEWSHOT_EXPERIENCE_SOURCE = fewshot.EXPERIENCE_SOURCE
FEWSHOT_READOUT = (
    "each query runs the program once on this classifier and its experience, applies the update within the limits and "
    "returns three numbers, each against the unedited classifier: score_new = fixed - broken on a held-out check set of "
    "images of the new classes, score_old = fixed - broken on a held-out check set of images of the old classes, and "
    "score = score_new + score_old (S_joint); fixed counts images the unedited classifier got wrong that the updated "
    "one gets right, broken counts images it got right that the updated one gets wrong; no per-image information"
)


def _class_indices(task: Mapping[str, Any], key: str) -> list[int]:
    """``task[key]`` as class indices; the list may give indices or class names."""
    names = [str(name) for name in task["classes"]]
    values = task.get(key)
    if not isinstance(values, list) or not values:
        raise ExperienceError(f"the few-shot task.json has no non-empty {key}")
    indices = [names.index(value) if isinstance(value, str) else int(value) for value in values]
    if any(not 0 <= index < len(names) for index in indices) or len(set(indices)) != len(indices):
        raise ExperienceError(f"task.json {key} {values!r} are not distinct classes of {names}")
    return indices


def _named(indices: Sequence[int], names: Sequence[str]) -> str:
    return ", ".join(f"{index} ({names[index]})" for index in indices)


def fewshot_statement(names: Sequence[str], old: Sequence[int], new: Sequence[int]) -> str:
    """The LEARN statement of a few-shot task."""
    return (
        "Make a trained image classifier recognise classes it was never trained on while it keeps the classes it "
        f"knows. The classifier has an output for each of the {len(names)} classes, but it was trained only on "
        f"images of the old classes {_named(old, names)}; the outputs of the new classes {_named(new, names)} never "
        "saw an image of their class. The experience file holds a few labelled examples of each new class and "
        "labelled examples of the old classes from the data the classifier was trained on, with the classifier's "
        "predictions on both. Write a program that computes an additive update of the classifier's weights from the "
        "classifier and its experience, without training it, so that the updated classifier classifies images of the "
        "new classes correctly while the old-class images it already classifies correctly stay correct."
    )


def learn_task_fewshot(task: Mapping[str, Any], shown_sha256: str, limits: Limits, y_new: np.ndarray,
                       n_old: int) -> dict[str, Any]:
    """The task.json of a LEARN trial on a few-shot instance (``FEWSHOT_RULE``): statement, experience, program,
    joint check (new classes and old classes) and the program limits; ``y_new`` gives the per-class counts."""
    required = ("classes", "train_classes", "new_classes", "patch_check", "limits", "limit_definitions",
                "training_free_rule")
    missing = [key for key in required if key not in task]
    if missing or "parent_sha256" in task:
        raise ExperienceError(f"the few-shot task.json cannot give the LEARN view: missing {missing}, or it already "
                              "has parent_sha256")
    names = [str(name) for name in task["classes"]]
    old, new = _class_indices(task, "train_classes"), _class_indices(task, "new_classes")
    if set(old) & set(new):
        raise ExperienceError(f"old classes {old} and new classes {new} overlap")
    counts = {int(label): int(count) for label, count in zip(*np.unique(np.asarray(y_new), return_counts=True))}
    if set(counts) != set(new) or len(set(counts.values())) != 1:
        raise ExperienceError(f"y_new holds classes {counts}; the task's new classes are {new}, equally often")
    per_class = next(iter(counts.values()))
    result = copy.deepcopy({key: value for key, value in task.items()
                            if key not in ("evidence", "patch_format", "patch_check")})
    shown, definitions = program_limits_view(limits)
    result.update(
        parent_sha256=shown_sha256,
        statement=fewshot_statement(names, old, new),
        experience={
            "new": {"keys": ["x_new", "y_new", "pred_new"], "count": int(len(y_new)), "per_class": per_class,
                    "classes": new,
                    "description": f"labelled examples of the new classes, {per_class} of each, drawn at random from "
                                   "held-out images of that class"},
            "old": {"keys": ["x_old", "y_old", "pred_old"], "count": n_old, "classes": old,
                    "description": "labelled examples of the old classes drawn at random from the data the "
                                   "classifier was trained on"},
            "ids": {"keys": ["split_ids_new", "split_ids_old"],
                    "description": "one opaque id per example, in x_new and x_old order"},
        },
        program=_program_section(),
        patch_check={"budget_queries": task["patch_check"]["budget_queries"], "readout": FEWSHOT_READOUT,
                     "invalid_programs": LEARN_INVALID, "final_selection": LEARN_FINAL_SELECTION},
        limits=shown,
        limit_definitions=definitions,
        training_free_rule={**copy.deepcopy(task["training_free_rule"]), "program": list(LEARN_PROGRAM_RULE)},
    )
    return result


# New classes for a graph network (PREREG E7, rule graphnew-v1): the fewshot-v1 task on the nodes of one citation graph.
# The inputs are int64 node ids into graph.npz, which forward.py reads from its own directory; every forward runs the
# whole graph. The replay nodes are held-out nodes of the old classes, not training nodes (E2 drew its replay from the
# training rows), so the old part is described as such.
GRAPHNEW_RULE = graphnew.GRAPHNEW_RULE
GRAPHNEW_EXPERIENCE_SOURCE = graphnew.EXPERIENCE_SOURCE
GRAPHNEW_READOUT = (
    "each query runs the program once on this classifier and its experience, applies the update within the limits and "
    "returns three numbers, each against the unedited classifier: score_new = fixed - broken on a held-out check set of "
    "nodes of the new classes, score_old = fixed - broken on a held-out check set of nodes of the old classes, and "
    "score = score_new + score_old (S_joint); fixed counts nodes the unedited classifier got wrong that the updated "
    "one gets right, broken counts nodes it got right that the updated one gets wrong; no per-node information"
)
GRAPHNEW_INPUTS = ("x_new and x_old are int64 node ids into graph.npz (task.json graph describes it); forward.py reads "
                   "graph.npz from its own directory and runs the whole graph on every call")


def graphnew_statement(names: Sequence[str], old: Sequence[int], new: Sequence[int], n_nodes: int,
                       n_features: int) -> str:
    """The LEARN statement of a graphnew task (``fewshot_statement`` for the nodes of one citation graph)."""
    return (
        "Make a trained graph classifier recognise classes it was never trained on while it keeps the classes it "
        "knows. It classifies the nodes of one citation graph, one node per paper; graph.npz holds the graph: "
        f"{n_nodes} nodes with {n_features} features each and the normalized adjacency. The classifier has an output "
        f"for each of the {len(names)} classes, but it was trained only on labelled nodes of the old classes "
        f"{', '.join(str(names[c]) for c in old)}; the outputs of the new classes "
        f"{', '.join(str(names[c]) for c in new)} never saw a labelled node of their class. The experience file holds "
        "a few labelled nodes of each new class and labelled nodes of the old classes held out from its training, "
        "given as node ids into graph.npz, with the classifier's predictions on both. Write a program that computes "
        "an additive update of the classifier's weights from the classifier and its experience, without training it, "
        "so that the updated classifier classifies the nodes of the new classes correctly while the old-class nodes "
        "it already classifies correctly stay correct."
    )


def _graphnew_program_section() -> dict[str, Any]:
    """``_program_section`` for a graph network: the same interface, with the forward, the run's inputs and the
    transfer worded for node ids of one graph (PREREG E7 transfer: other seeds of the same architecture)."""
    section = _program_section()
    section["student"] = {
        "params": "the classifier's weights as in model.npz: {tensor name: float32 array}",
        "forward": "forward(params, x) -> logits [N, number of classes] for int64 node ids x [N], the function in "
                   "forward.py; every call runs the whole graph of graph.npz and returns the rows of x",
        "layout": "the content of model_spec.json",
    }
    section["runs"] = (
        "each run gets a fresh copy of exactly these inputs, numpy and the standard library, no network, no file of "
        "/work and no state of your kernel, and a copy of graph.npz beside the run's forward.py, which forward reads; "
        "a program that needs the graph itself loads graph.npz from the directory of "
        "student['forward'].__code__.co_filename; two runs on the same inputs must return the same deltas")
    section["transfer"] = (
        "the program is run unchanged on other independently trained classifiers of the same architecture on the same "
        "graph and classes, each with its own layout (student['layout']) and its own experience drawn by the same "
        "rules, so it must compute the update from the student and the experience it receives, find the tensors it "
        "changes from the layout, and not contain this classifier's numbers")
    return section


def learn_task_graphnew(task: Mapping[str, Any], shown_sha256: str, limits: Limits, y_new: np.ndarray,
                        n_old: int) -> dict[str, Any]:
    """The task.json of a LEARN trial on a graphnew instance (``GRAPHNEW_RULE``): ``learn_task_fewshot`` for the
    nodes of one graph. The statement, the experience descriptions (and ``inputs``, the node ids), the readout and the
    program's student, runs and transfer are worded for the graph; the task's ``graph`` section is kept; the final
    selection, the program limits (``weightcode.harness.graphnew.program_limits``) and the training-free rule are E2's."""
    required = ("classes", "train_classes", "new_classes", "graph", "patch_check", "limits", "limit_definitions",
                "training_free_rule")
    missing = [key for key in required if key not in task]
    if missing or "parent_sha256" in task:
        raise ExperienceError(f"the graphnew task.json cannot give the LEARN view: missing {missing}, or it already "
                              "has parent_sha256")
    names = [str(name) for name in task["classes"]]
    old, new = _class_indices(task, "train_classes"), _class_indices(task, "new_classes")
    if set(old) & set(new):
        raise ExperienceError(f"old classes {old} and new classes {new} overlap")
    counts = {int(label): int(count) for label, count in zip(*np.unique(np.asarray(y_new), return_counts=True))}
    if set(counts) != set(new) or len(set(counts.values())) != 1:
        raise ExperienceError(f"y_new holds classes {counts}; the task's new classes are {new}, equally often")
    per_class = next(iter(counts.values()))
    graph = task["graph"]
    if graph.get("file") != graphnew.GRAPH_FILE:
        raise ExperienceError(f"the graphnew task.json names graph file {graph.get('file')!r}, not {graphnew.GRAPH_FILE}")
    result = copy.deepcopy({key: value for key, value in task.items()
                            if key not in ("evidence", "patch_format", "patch_check")})
    shown, definitions = program_limits_view(limits)
    result.update(
        parent_sha256=shown_sha256,
        statement=graphnew_statement(names, old, new, int(graph["n_nodes"]), int(graph["n_features"])),
        experience={
            "new": {"keys": ["x_new", "y_new", "pred_new"], "count": int(len(y_new)), "per_class": per_class,
                    "classes": new,
                    "description": f"labelled nodes of the new classes, {per_class} of each, drawn at random from "
                                   "held-out nodes of that class"},
            "old": {"keys": ["x_old", "y_old", "pred_old"], "count": n_old, "classes": old,
                    "description": "labelled nodes of the old classes, a fixed random draw from held-out nodes of "
                                   "those classes"},
            "inputs": GRAPHNEW_INPUTS,
            "ids": {"keys": ["split_ids_new", "split_ids_old"],
                    "description": "one opaque id per example, in x_new and x_old order"},
        },
        program=_graphnew_program_section(),
        patch_check={"budget_queries": task["patch_check"]["budget_queries"], "readout": GRAPHNEW_READOUT,
                     "invalid_programs": LEARN_INVALID, "final_selection": LEARN_FINAL_SELECTION},
        limits=shown,
        limit_definitions=definitions,
        training_free_rule={**copy.deepcopy(task["training_free_rule"]), "program": list(LEARN_PROGRAM_RULE)},
    )
    return result


def _program_section() -> dict[str, Any]:
    """The program interface of every LEARN task view: file, signature, student, experience, returns, transfer."""
    return {
        "file": "one Python file in /work/patches, for example update.py",
        "defines": ["update(student: dict, experience: dict) -> dict[str, numpy.ndarray]",
                    *(f"{name}: a non-empty str" for name in MODULE_STRINGS)],
        "student": {"params": "the classifier's weights as in model.npz: {tensor name: float32 array}",
                    "forward": "forward(params, x) -> logits, the function in forward.py",
                    "layout": "the content of model_spec.json"},
        "experience": "the arrays of experience.npz: {name: array}",
        "returns": "additive deltas for the tensors it changes: {tensor name: finite array of that tensor's shape, "
                   "float16, float32 or float64}; the updated tensor is float32(params[name] + delta), computed "
                   "in float64, and a tensor left out is unchanged",
        "transfer": "the program is run unchanged on other independently trained classifiers of the same "
                    "architecture, and on classifiers of other architectures for the same inputs and classes, "
                    "each with its own layout (student['layout']) and its own experience drawn by the same rules, "
                    "so it must compute the update from the student and the experience it receives, find the "
                    "tensors it changes from the layout, and not contain this classifier's numbers",
        "runs": "each run gets a fresh copy of exactly these inputs, numpy and the standard library, no network, no "
                "file of /work and no state of your kernel; two runs on the same inputs must return the same "
                "deltas",
        "wall_limit_s": PROGRAM_WALL_S,
    }


# A new operator for a sequence model (PREREG E5, rule seqskill-v1): the student never saw a row holding the operator
# token; the experience is the instance's own evidence file, which already holds the LEARN keys.
SEQSKILL_RULE = seqskill.SEQSKILL_RULE
SEQSKILL_EXPERIENCE_SOURCE = seqskill.EXPERIENCE_SOURCE
SEQSKILL_READOUT = (
    "each query runs the program once on this model and its experience, applies the update within the limits and "
    "returns three numbers, each against the unedited model: score_new = fixed - broken on a held-out check set of "
    "rows of the new operator alone, at the lengths of the new examples, score_old = fixed - broken on a held-out "
    "check set of old-operator rows, single and composed, and score = score_new + score_old (S_joint); a row is right "
    "when every output slot is right; fixed counts rows the unedited model got wrong that the updated one gets right, "
    "broken counts rows it got right that the updated one gets wrong; no per-row information"
)
SEQSKILL_FINAL_SELECTION = (
    "kept at the end: among the checked programs and the unedited model (score = 0), the update computed by the one "
    "with the highest score (S_joint); ties go to fewer changed parameters, then to the earlier query; a program with "
    "score <= 0 and an unchecked program are never kept"
)


def seqskill_statement(operator: str, variant: str | None = None) -> str:
    """The LEARN statement of a seqskill task with new operator token ``operator``.

    rule_only (PREREG E8) replaces E5's "and what it does is not given: the experience shows it" by a sentence saying
    that the rule is given in model_spec.json and that the experience holds no row of it, and the sentence on the
    experience file names only its old-operator rows; every other word is E5's."""
    if seqskill.check_variant(variant) == seqskill.RULE_ONLY:
        new_operator = (f"{operator} never occurred in the model's training, so the model does not compute it. Its "
                        "rule is given in model_spec.json, and the experience holds no row of it. The experience file "
                        "holds labelled rows of the old operators from the data the model was trained on, with the "
                        "model's predictions on them. ")
    else:
        new_operator = (f"{operator} never occurred in the model's training, so the model does not compute it, and "
                        "what it does is not given: the experience shows it. The experience file holds labelled rows "
                        "of the new operator alone and labelled rows of the old operators from the data the model was "
                        "trained on, with the model's predictions on both. ")
    return (
        "Make a trained sequence model compute an operator it was never trained on while it keeps the operators it "
        "knows. Each row names one or two operators and holds a string of symbols, and the model writes the result, "
        "one symbol per output slot; model_spec.json gives the row format and the meaning of every token. The token "
        + new_operator +
        "Write a program that computes an additive update of the model's weights from the model and its "
        "experience, without training it, so that the updated model answers the rows that use the new operator "
        "correctly (task.json rows describes them) while the old-operator rows it already answers correctly stay "
        "correct."
    )


# rule_only (PREREG E8): the instance task.json's sentences that speak of new-operator rows in the experience, and
# what the rule_only view says instead. Each must occur in the instance text exactly once, or the view is refused.
RULE_ONLY_FACTS = (" What the operator does is not given; the new evidence rows show it",
                   " Its rule is given in model_spec.json, and the experience holds no row of it")
RULE_ONLY_PREDICTIONS = ("pred_new and pred_old hold ", "pred_old holds ")
RULE_ONLY_IDS_DESCRIPTION = "one opaque id per example, in x_old order"
# The rows section and the patch-check readout name the check and composed lengths by the new evidence, which the
# rule_only experience does not have; the view names them as numbers, the new-operator lengths of the check
# (weightcode.data.seqops.NEW_SHORT_LENGTHS, which the instance task.json's patch_check.parts.new must state too).
RULE_ONLY_ROWS_LENGTHS = "at the lengths of the new evidence"
RULE_ONLY_READOUT_LENGTHS = "at the lengths of the new examples"


def _rule_only_lengths(task: Mapping[str, Any]) -> str:
    """"at lengths <lo> to <hi>", the new-operator lengths of the check, as the instance task.json states them."""
    from weightcode.data import seqops as S

    lo, hi = S.NEW_SHORT_LENGTHS
    stated = str(task["patch_check"].get("parts", {}).get("new", ""))
    if not stated.endswith(f"lengths {lo} to {hi}"):
        raise ExperienceError(f"the seqskill task.json patch_check.parts.new {stated!r} does not state the lengths "
                              f"{lo} to {hi}, so the rule_only view cannot name them")
    return f"at lengths {lo} to {hi}"


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ExperienceError(f"the seqskill task.json {what} does not say {old!r} once, so the rule_only view cannot "
                              "be derived from it")
    return text.replace(old, new)


def _seqskill_program_section() -> dict[str, Any]:
    """``_program_section`` for a sequence model: the same interface, with the student and transfer worded for it."""
    section = _program_section()
    section["student"] = {
        "params": "the model's weights as in model.npz: {tensor name: float32 array}",
        "forward": "forward(params, x) -> logits [N, seq_len, number of symbols], the function in forward.py; the "
                   "model's answer at an output slot is the argmax there",
        "layout": "the content of model_spec.json",
    }
    section["transfer"] = (
        "the program may also be run unchanged on other independently trained models for the same rows, each with "
        "its own layout (student['layout']) and its own experience drawn by the same rules, so it must compute the "
        "update from the student and the experience it receives, find the tensors it changes from the layout, and "
        "not contain this model's numbers")
    return section


def learn_task_seqskill(task: Mapping[str, Any], shown_sha256: str, limits: Limits,
                        record: Mapping[str, Any], variant: str | None = None) -> dict[str, Any]:
    """The task.json of a LEARN trial on a seqskill instance (``SEQSKILL_RULE``): statement, experience, program,
    joint check (the new operator alone and the old operators) and the program limits; ``record`` is the experience
    record of ``weightcode.harness.seqskill.experience``.

    rule_only (PREREG E8; ``record`` from ``weightcode.harness.seqskill.rule_only_experience``) differs in five places:
    the statement (``seqskill_statement``), ``new_operator.facts`` (its last sentence, ``RULE_ONLY_FACTS``), the
    experience section, which describes the old part alone (its keys, count and description, the targets, the
    predictions of ``pred_old`` and the ids of ``split_ids_old``), and the two phrases that named lengths by the new
    evidence: "at the lengths of the new evidence" in ``rows.new_operator_rows`` and "at the lengths of the new
    examples" in the patch-check readout both become "at lengths 4 to 8" (``_rule_only_lengths``). Every other part
    of the rows section and the readout, the program, the final selection, the limits and the training-free rule are
    those of the standard view."""
    variant = seqskill.check_variant(variant)
    required = ("classes", "input", "new_operator", "old_operators", "rows", "patch_check", "limits",
                "limit_definitions", "training_free_rule")
    missing = [key for key in required if key not in task]
    if missing or "parent_sha256" in task:
        raise ExperienceError(f"the seqskill task.json cannot give the LEARN view: missing {missing}, or it already "
                              "has parent_sha256")
    operator = str(task["new_operator"]["token"])
    if operator != record["new_operator"] or int(task["new_operator"]["id"]) != int(record["new_operator_id"]):
        raise ExperienceError(f"task.json names new operator {operator!r}, the experience {record['new_operator']!r}")
    if (record.get("variant") == seqskill.RULE_ONLY) != (variant == seqskill.RULE_ONLY):
        raise ExperienceError(f"a {variant} task view needs the experience record of that variant")
    result = copy.deepcopy({key: value for key, value in task.items()
                            if key not in ("evidence", "patch_format", "patch_check")})
    shown, definitions = program_limits_view(limits)
    old = {"keys": ["x_old", "y_old", "pred_old"], "count": int(record["n_old"]),
           "description": str(task["evidence"]["old"]["description"])}
    if variant == seqskill.RULE_ONLY:
        if int(record["n_new"]) != 0:
            raise ExperienceError("a rule_only experience holds no row of the new operator")
        result["new_operator"]["facts"] = _replace_once(str(task["new_operator"]["facts"]), *RULE_ONLY_FACTS,
                                                        "new_operator.facts")
        lengths = _rule_only_lengths(task)
        result["rows"]["new_operator_rows"] = _replace_once(str(task["rows"]["new_operator_rows"]),
                                                            RULE_ONLY_ROWS_LENGTHS, lengths, "rows.new_operator_rows")
        readout = _replace_once(SEQSKILL_READOUT, RULE_ONLY_READOUT_LENGTHS, lengths, "patch-check readout")
        experience = {
            "old": old,
            "targets": str(task["rows"]["targets"]),
            "predictions": _replace_once(str(task["evidence"]["predictions"]), *RULE_ONLY_PREDICTIONS,
                                         "evidence.predictions"),
            "ids": {"keys": ["split_ids_old"], "description": RULE_ONLY_IDS_DESCRIPTION},
        }
    else:
        readout = SEQSKILL_READOUT
        experience = {
            "new": {"keys": ["x_new", "y_new", "pred_new"], "count": int(record["n_new"]),
                    "description": str(task["evidence"]["new"]["description"])},
            "old": old,
            "targets": str(task["rows"]["targets"]),
            "predictions": str(task["evidence"]["predictions"]),
            "ids": {"keys": ["split_ids_new", "split_ids_old"],
                    "description": "one opaque id per example, in x_new and x_old order"},
        }
    result.update(
        parent_sha256=shown_sha256,
        statement=seqskill_statement(operator, variant),
        experience=experience,
        program=_seqskill_program_section(),
        patch_check={"budget_queries": task["patch_check"]["budget_queries"], "readout": readout,
                     "invalid_programs": LEARN_INVALID, "final_selection": SEQSKILL_FINAL_SELECTION},
        limits=shown,
        limit_definitions=definitions,
        training_free_rule={**copy.deepcopy(task["training_free_rule"]), "program": list(LEARN_PROGRAM_RULE)},
    )
    return result


# A known primitive inside old compositions (PREREG E13, rule scanskill-v1, the seqskill variant scan_rule): commands of
# words mapped to action sequences, the interpretation rules stated in model_spec.json; the experience is the
# instance's own evidence file (the primitive row when the condition has one, and 64 replay rows), no target command.
SCAN_RULE = seqskill.SCAN_RULE
SCAN_EXPERIENCE_SOURCE = seqskill.SCAN_EXPERIENCE_SOURCE
SCAN_READOUT = (
    "each query runs the program once on this model and its experience, applies the update within the limits and "
    "returns three numbers, each against the unedited model: score_new = fixed - broken on a held-out check set of "
    "target commands (task.json target), score_old = fixed - broken on a held-out check set of commands of the "
    "training distribution that were never in the model's training rows, and score = score_new + score_old "
    "(S_joint); a row is right when every output slot is right, END included; fixed counts rows the unedited model "
    "got wrong that the updated one gets right, broken counts rows it got right that the updated one gets wrong; no "
    "per-row information"
)


def scan_statement(record: Mapping[str, Any]) -> str:
    """The LEARN statement of a scanskill task (PREREG E13); ``record`` is the experience record of
    ``weightcode.harness.seqskill.scan_experience``, whose primitive part decides the sentence on the experience file."""
    if int(record["n_prim"]) > 0:
        held = (f"the training row of the command {record['primitive_row']} alone and labelled rows of training "
                "commands, with the model's predictions on them")
    else:
        held = "labelled rows of training commands, with the model's predictions on them"
    return (
        "Make a trained sequence model answer a family of commands it fails on while it keeps the commands it knows. "
        "Each row holds a command, a string of words, and the model writes the action sequence the command denotes, "
        "one action per output slot and then END; model_spec.json gives the row format and the meaning of every word "
        "as interpretation rules. task.json target names the commands the model fails on and what it was trained on. "
        f"The experience file holds {held}, and no target command. "
        "Write a program that computes an additive update of the model's weights from the model and its "
        "experience, without training it, so that the updated model answers the target commands correctly while the "
        "commands it already answers correctly stay correct."
    )


def learn_task_scan(task: Mapping[str, Any], shown_sha256: str, limits: Limits,
                    record: Mapping[str, Any]) -> dict[str, Any]:
    """The task.json of a LEARN trial on a scanskill instance (the seqskill variant scan_rule, PREREG E13): the
    instance's data facts (classes, input, target, rows), the statement (``scan_statement``), the experience (the
    primitive part when the condition has one, and the old part, with the counts of ``record``), the program, the
    joint check (target commands and held-out commands of the training distribution) and the program limits;
    ``record`` is the experience record of ``weightcode.harness.seqskill.scan_experience``."""
    required = ("classes", "input", "target", "rows", "evidence", "patch_check", "limits", "limit_definitions",
                "training_free_rule")
    missing = [key for key in required if key not in task]
    if missing or "parent_sha256" in task:
        raise ExperienceError(f"the scanskill task.json cannot give the LEARN view: missing {missing}, or it already "
                              "has parent_sha256")
    if record.get("rule") != SCAN_RULE or record.get("variant") != seqskill.SCAN:
        raise ExperienceError(f"a scan_rule task view needs a scanskill experience record, not rule "
                              f"{record.get('rule')!r} variant {record.get('variant')!r}")
    evidence = task["evidence"]
    prim = int(record["n_prim"]) > 0
    if prim != ("prim" in evidence) or int(record["n_new"]) != 0:
        raise ExperienceError("the scanskill task.json and the experience disagree on the primitive part, or the "
                              "experience holds a target row")
    experience: dict[str, Any] = {}
    for part, count in (("prim", int(record["n_prim"])), ("old", int(record["n_old"]))):
        if part == "prim" and not prim:
            continue
        if int(evidence[part]["count"]) != count:
            raise ExperienceError(f"task.json evidence.{part} counts {evidence[part]['count']} rows, the experience "
                                  f"{count}")
        experience[part] = {"keys": list(evidence[part]["keys"]), "count": count,
                            "description": str(evidence[part]["description"])}
    experience.update(targets=str(task["rows"]["targets"]), predictions=str(evidence["predictions"]),
                      ids=copy.deepcopy(dict(evidence["ids"])))
    result = copy.deepcopy({key: value for key, value in task.items()
                            if key not in ("evidence", "patch_format", "patch_check")})
    shown, definitions = program_limits_view(limits)
    result.update(
        parent_sha256=shown_sha256,
        statement=scan_statement(record),
        experience=experience,
        program=_seqskill_program_section(),
        patch_check={"budget_queries": task["patch_check"]["budget_queries"],
                     "parts": copy.deepcopy(dict(task["patch_check"]["parts"])), "readout": SCAN_READOUT,
                     "invalid_programs": LEARN_INVALID, "final_selection": SEQSKILL_FINAL_SELECTION},
        limits=shown,
        limit_definitions=definitions,
        training_free_rule={**copy.deepcopy(task["training_free_rule"]), "program": list(LEARN_PROGRAM_RULE)},
    )
    return result


# A new skill for a trained policy (PREREG E14, rule policyskill-v1): the student, trained by imitation on the old
# mission only, never saw the instruction words of the new mission and never took the pickup action; the experience is
# the instance's own evidence file, which already holds the LEARN keys (``weightcode.harness.policyskill``).
POLICYSKILL_RULE = policyskill.POLICYSKILL_RULE
POLICYSKILL_EXPERIENCE_SOURCE = policyskill.EXPERIENCE_SOURCE
POLICYSKILL_READOUT = (
    "each query runs the program once on this policy and its experience, applies the update within the limits and "
    "returns three numbers, each against the unedited policy: score_new = fixed - broken on a held-out check set of "
    "episodes of the new mission, score_old = fixed - broken on a held-out check set of episodes of the trained "
    "mission, and score = score_new + score_old (S_joint); every episode is run greedily (the argmax action at every "
    "step, at most {max_steps} steps) in its own room, and it counts as right when its mission is completed; fixed "
    "counts episodes the unedited policy failed that the updated one completes, broken counts episodes it completed "
    "that the updated one fails; no per-episode information"
)
POLICYSKILL_FINAL_SELECTION = (
    "kept at the end: among the checked programs and the unedited policy (score = 0), the update computed by the one "
    "with the highest score (S_joint); ties go to fewer changed parameters, then to the earlier query; a program with "
    "score <= 0 and an unchecked program are never kept"
)
POLICYSKILL_IDS_DESCRIPTION = "one opaque id per row, in x_new and x_old order"


def policyskill_statement(words: Sequence[str]) -> str:
    """The LEARN statement of a policyskill task whose new mission has the untrained instruction ``words`` (task.json
    ``new_mission.words``). The two missions' grammars are referred to task.json, not quoted: they name the object
    attributes, and the prompts are scanned for hint words (``weightcode.harness.leaks.HINT_WORDS``)."""
    if not words:
        raise ExperienceError("a policyskill statement names the new mission's untrained words")
    named = " and ".join(f"\"{word}\"" for word in words)
    return (
        "Make a trained policy perform a new mission it was never trained on while it keeps the mission it knows. At "
        "each step the policy reads one observation row (the fully observed grid of one room, with the agent's "
        "position and direction, and the instruction as word ids) and outputs one logit per action, and it takes the "
        "argmax action; model_spec.json gives the row format, the cell encoding, the actions and the meaning of every "
        "instruction word. It was trained by imitation on one mission only, whose instructions follow the grammar of "
        "task.json trained_mission. The instructions of the new mission follow the grammar of task.json new_mission, "
        f"in the same rooms: the words {named} never occurred in the policy's training, so their embedding rows are "
        "untrained, and the pickup action was never a training target, so the policy does not perform the new "
        "mission, and what completes it is not given: the demonstrations show it. The experience file "
        "holds every step of a few demonstrations of the new mission by a planner, and single steps of the trained "
        "mission from the policy's training demonstrations, with the planner's action and the policy's logits on "
        "both. Write a program that computes an additive update of the policy's weights from the policy and its "
        "experience, without training it, so that the updated policy completes episodes of the new mission (task.json "
        "new_mission describes it) while the episodes of the trained mission it already completes stay completed."
    )


def _policyskill_program_section() -> dict[str, Any]:
    """``_program_section`` for a policy: the same interface, with the student and transfer worded for it."""
    section = _program_section()
    section["student"] = {
        "params": "the policy's weights as in model.npz: {tensor name: float32 array}",
        "forward": "forward(params, x) -> logits [N, number of actions], the function in forward.py; the policy's "
                   "action on a row is the argmax",
        "layout": "the content of model_spec.json",
    }
    section["transfer"] = (
        "the program may also be run unchanged on other independently trained policies for the same missions, each "
        "with its own layout (student['layout']) and its own experience drawn by the same rules, so it must compute "
        "the update from the student and the experience it receives, find the tensors it changes from the layout, and "
        "not contain this policy's numbers")
    return section


def learn_task_policyskill(task: Mapping[str, Any], shown_sha256: str, limits: Limits,
                           record: Mapping[str, Any]) -> dict[str, Any]:
    """The task.json of a LEARN trial on a policyskill instance (``POLICYSKILL_RULE``): statement, experience,
    program, joint check (episodes of the new mission and of the trained mission) and the program limits; ``record`` is
    the experience record of ``weightcode.harness.policyskill.experience``. The instance's data facts (the actions, the input,
    the missions, the rows) are kept as they are; its evidence, patch format and patch-check sections are replaced."""
    from weightcode.data import e14 as D

    required = ("actions", "input", "trained_mission", "new_mission", "rows", "evidence", "patch_check", "limits",
                "limit_definitions", "training_free_rule")
    missing = [key for key in required if key not in task]
    if missing or "parent_sha256" in task:
        raise ExperienceError(f"the policyskill task.json cannot give the LEARN view: missing {missing}, or it already "
                              "has parent_sha256")
    if record.get("source") != POLICYSKILL_EXPERIENCE_SOURCE:
        raise ExperienceError(f"a policyskill task view needs the policyskill experience record, not "
                              f"{record.get('source')!r}")
    evidence = task["evidence"]
    counts = (int(evidence["new"]["count"]), int(evidence["new"]["demonstrations"]), int(evidence["old"]["count"]))
    if counts != (int(record["n_new"]), int(record["n_new_demonstrations"]), int(record["n_old"])):
        raise ExperienceError(f"the policyskill task.json counts {counts} differ from the experience record")
    if [row.get("id") for row in task["new_mission"]["words"]] != list(record["reserved_word_ids"]):
        raise ExperienceError("the policyskill task.json new-mission words are not the experience's reserved words")
    result = copy.deepcopy({key: value for key, value in task.items()
                            if key not in ("evidence", "patch_format", "patch_check")})
    shown, definitions = program_limits_view(limits)
    experience = {
        "new": {"keys": ["x_new", "y_new", "logits_new", "episode_new", "step_new"], "count": counts[0],
                "demonstrations": counts[1], "description": str(evidence["new"]["description"])},
        "old": {"keys": ["x_old", "y_old", "logits_old"], "count": counts[2],
                "description": str(evidence["old"]["description"])},
        "targets": str(task["rows"]["targets"]),
        "logits": str(evidence["logits"]),
        "ids": {"keys": ["split_ids_new", "split_ids_old"], "description": POLICYSKILL_IDS_DESCRIPTION},
    }
    result.update(
        parent_sha256=shown_sha256,
        statement=policyskill_statement([str(row["word"]) for row in task["new_mission"]["words"]]),
        experience=experience,
        program=_policyskill_program_section(),
        patch_check={"budget_queries": task["patch_check"]["budget_queries"],
                     "parts": copy.deepcopy(task["patch_check"]["parts"]),
                     "readout": POLICYSKILL_READOUT.format(max_steps=D.MAX_STEPS),
                     "invalid_programs": LEARN_INVALID, "final_selection": POLICYSKILL_FINAL_SELECTION},
        limits=shown,
        limit_definitions=definitions,
        training_free_rule={**copy.deepcopy(task["training_free_rule"]), "program": list(LEARN_PROGRAM_RULE)},
    )
    return result


__all__ = ["EXPERIENCE_FILE", "EXPERIENCE_KEYS", "EXPERIENCE_PARTS", "EXPERIENCE_SOURCE", "ExperienceError",
           "POLICYSKILL_EXPERIENCE_SOURCE", "POLICYSKILL_FINAL_SELECTION", "POLICYSKILL_IDS_DESCRIPTION",
           "POLICYSKILL_READOUT", "POLICYSKILL_RULE", "learn_task_policyskill", "policyskill_statement",
           "LEARN_FINAL_SELECTION", "LEARN_PROGRAM_RULE", "LEARN_STATEMENT", "MODULE_STRINGS", "build_experience", "evidence_idx",
           "FEWSHOT_EXPERIENCE_SOURCE", "FEWSHOT_READOUT", "FEWSHOT_RULE", "fewshot_statement", "learn_task",
           "learn_task_fewshot", "program_limits_view", "SEQSKILL_EXPERIENCE_SOURCE", "SEQSKILL_FINAL_SELECTION",
           "SCAN_EXPERIENCE_SOURCE", "SCAN_READOUT", "SCAN_RULE", "learn_task_scan", "scan_statement",
           "SEQSKILL_READOUT", "SEQSKILL_RULE", "learn_task_seqskill", "seqskill_statement", "RULE_ONLY_FACTS",
           "RULE_ONLY_IDS_DESCRIPTION", "RULE_ONLY_PREDICTIONS", "RULE_ONLY_READOUT_LENGTHS", "RULE_ONLY_ROWS_LENGTHS",
           "GRAPHNEW_EXPERIENCE_SOURCE", "GRAPHNEW_INPUTS", "GRAPHNEW_READOUT", "GRAPHNEW_RULE", "graphnew_statement",
           "learn_task_graphnew", "BLIND_EXPERIENCE_WITH_PREDICTIONS", "BLIND_HIDDEN_KEYS", "BLIND_OUTPUT",
           "BLIND_PARAMS", "BLIND_STATEMENT_PREDICTIONS", "blind_trial_experience", "learn_task_blind",
           "RULE_EXPERIENCE_KEYS", "RULE_EXPERIENCE_SOURCE", "RULE_IDS_DESCRIPTION", "RULE_PROGRAM_LABELS",
           "RULE_STATEMENT_EXPERIENCE", "RULE_STATEMENT_NO_IMAGE", "learn_task_rule", "rule_experience",
           "UNLABELLED_EXPERIENCE_KEYS", "UNLABELLED_EXPERIENCE_SOURCE", "UNLABELLED_NEW_KEYS", "UNLABELLED_STATEMENT",
           "UNLABELLED_WITHHELD_KEYS", "UNLABELLED_ERROR_WORDS", "LEARN_NEW_DESCRIPTION", "learn_new_selection",
           "learn_task_unlabelled", "unlabelled_experience", "NAMED_EXPERIENCE_SOURCE",
           "NAMED_NOCHECK_FINAL_SELECTION", "NAMED_NOCHECK_PROGRAM_STORED", "NAMED_NOCHECK_STATEMENT_NO_DATA",
           "NAMED_EXPERIENCE_WORDS", "NAMED_PROGRAM_EXPERIENCE", "NAMED_PROGRAM_RULES", "NAMED_READOUT",
           "NAMED_STATEMENT_EXPERIENCE",
           "NAMED_STATEMENT_NO_DATA", "NAMED_STATEMENT_PROGRAM", "NAMED_TRANSFER", "learn_task_named",
           "named_experience"]
