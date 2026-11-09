"""Teacher in the loop (PREREG E4-loop, dev only): a LEARN session on a stream's current student.

Round r >= 2 of a loop stream (``weightcode.harness.loop_stream``) is one LEARN session on round r's dev instance in which
the student is the stream's current student, the student after round r - 1, instead of the instance's base model. A
loop context file (``LOOP_CONTEXT_SCHEMA``), written by the stream outside every trial, says what the session gets:

- ``student``: the current student's .npz and its parameter hash. The trial's model.npz, the check oracle's base and
  the parent of the session's final are this student (``PARENT_LOOP`` in run.json, so a reader that rebuilds finals on
  the instance's base model refuses the record instead of misreading it).
- ``prev``: for each earlier round k, its dev instance and the evidence rows it drew for its condition, as drawn at
  that round (x, y, split ids and dataset idx). The trial's experience.npz holds them after the LEARN keys as
  ``PREV_KEYS``, with the current student's predictions (``loop_experience``). A previous row whose underlying dataset
  image is also among this round's new rows, or among the rows already kept from an earlier round, is left out, so no
  two experience rows show the same image under two conditions; the counts are recorded. The replay rows come from the
  clean train split and may share no dataset idx with any of them.
- ``library``: for each earlier round, its kept program (None for a no-op round) and the part scores of its kept
  check, written to the trial as ``library_index.json`` and ``library_program_<k>.py`` and labelled "round k"
  (``write_loop_library``). No condition name, session id or instance id is written.

The check (``MultiPartOracle``) scores one child on condition r's V_check (part ``new``), its clean companion (``old``)
and the V_check of each earlier round's condition (``prev_k``), every part fixed - broken against the current student;
``score`` is the sum of the parts. The kept rule loop-retain-v1 (``loop_kept_rule``) is the argmax of score over the
checked programs with score_old >= ``s_old_floor`` and every score_prev_k >= its ``s_prev_floor``, else the current
student unchanged. The floors are fractions of each part's rows: 0.003 of the clean rows (-12 of 4000) and 0.005 of
each earlier condition's rows (-20 of 4000), from the E4 retention criteria (PREREG E4-loop).

The kept rule loop-retain-v2 (``loop_kept_rule_v2``) changes only the clean floor: it is the stream's remaining clean
budget. Over the whole stream the kept updates may lose at most 0.015 of the clean V_check rows (-60 of 4000) net, so
round r's floor is ``-60 - (sum of the kept score_old of rounds 1 to r - 1)``, round 1 included (a no-op round counts
0). Every round's clean part is the same clean V_check rows, each score_old is fixed - broken against the student the
round found, so the sum telescopes to the stream's net clean change on those rows against the base. The stream writes
the earlier rounds' kept score_old into the loop context (``kept_rule``, ``context_kept_rule``); a context without that
key is a v1 round, unchanged. The earlier-condition floors stay v1's.

The texts below (statement, readout, final selection, the "What is kept" section, the tool descriptions and the two
file-list bullets) replace the LEARN wording only where it is false for a loop round, and they use no hint word.

Edit log (PREREG T16, arm L; opt in): a context with an ``edit_log`` key gives the round one more trial file,
``edit_log.json`` (``EDIT_LOG_FILE``), and one more sentence in the system prompt (``EDIT_LOG_SENTENCE``, word for
word, as its own paragraph at the end of the prompt's first section, ``edit_log_section``). Nothing else changes: no
task-prompt bullet, no task.json field, no tool text. The file lists, for each earlier round k, every parameter entry
that round's kept update changed (the entries where the student after round k differs from the student before it):
the tensor name and its flat indices (C order), and for each 2-D tensor the rows and the columns that hold a change
(``edit_log_document``). The stream derives it from its saved students and states the file's sha256 and per-round
counts in the context (``EDIT_LOG_SCHEMA``); the session reads it hash-checked, checks it against the current student's
shapes (``check_edit_log``), copies it into the trial and records it in session.json ``loop.edit_log``. A context
without the key is arm N (no log): the round's trial and prompts are as before, and session.json says
``loop.edit_log.enabled`` false.

Protected-entry limit (PREREG T16 follow-up M; opt in, with the edit log only): a context with a ``protect_logged``
key (``PROTECT_SCHEMA``: the edit log's sha256 and the number of entries it lists over all earlier rounds,
``check_context_protect``) adds a limit and one more sentence. The limit refuses every update whose child changes an
entry the edit log lists (``protected_entries``), compared with the round's student entry by entry in float32, as the
log's derivation compares (``protected_changes``). It is applied where every other limit is applied
(``weightcode.harness.applying.ArmApplier.apply``): a checked program it refuses is a limits refusal that uses no query, with
a message naming how many listed entries changed and in which tensors (``protect_messages``), and finalisation applies
it again when it rebuilds the kept child, so a final that changed a listed entry is refused as any final that fails
its re-application. The sentence (``PROTECT_SENTENCE``) follows the edit-log sentence in its paragraph
(``protect_section``); nothing else changes. session.json ``loop.protect_logged`` records the limit (``round_protect``).
A context without the key adds nothing, and its session.json has no ``loop.protect_logged``.
"""

from __future__ import annotations

import io
import json
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import count_changed, load_params, params_sha256
from weightcode.evalkit.metrics import summarize
from weightcode.evalkit.patch_check import BudgetExceeded, ForwardFn, predict, validate_child
from weightcode.harness.jsonio import bytes_sha256
from weightcode.harness.retention import s_old_floor

LOOP_CONTEXT_SCHEMA = "weightcode.harness.loop_context.v0"
LOOP_LIBRARY_SCHEMA = "weightcode.harness.loop_library.v0"
LOOP_RECORD_SCHEMA = "weightcode.harness.loop_session.v0"
KEPT_RULE_LOOP = "loop-retain-v1"
KEPT_RULE_LOOP_V2 = "loop-retain-v2"
KEPT_RULES = (KEPT_RULE_LOOP, KEPT_RULE_LOOP_V2)
LOOP_CLEAN_FRAC = 0.003  # -12 of 4000 clean V_check rows per round
LOOP_PREV_FRAC = 0.005  # -20 of 4000 V_check rows of each earlier condition per round
LOOP_CLEAN_BUDGET_FRAC = 0.015  # loop-retain-v2: -60 of 4000 clean V_check rows over the whole stream
PARENT_LOOP = "loop_student"
PREV_KEYS = ("x_prev", "y_prev", "pred_prev", "round_prev", "split_ids_prev")
EVIDENCE_KEYS = ("x", "y", "split_ids", "idx")
STUDENT_FILE = "loop_student.npz"  # the session's private copy of the current student
PREV_CHECK_TEMPLATE = "vcheck_prev_{k}.npz"  # the session's private copies of the earlier conditions' V_check
LOOP_LIBRARY_INDEX = "library_index.json"
LOOP_PROGRAM_TEMPLATE = "library_program_{k}.py"
ROUND_LABEL = "round {k}"
MAX_ROUNDS = 8
PREDICT_BATCH = 1024
EDIT_LOG_FILE = "edit_log.json"  # PREREG T16: the trial file of arm L
EDIT_LOG_SCHEMA = "weightcode.harness.edit_log.v0"  # the host-side record of the file in the loop context (never in the trial)
EDIT_LOG_SENTENCE = ("edit_log.json lists every parameter entry that earlier rounds changed; those entries hold the "
                     "capabilities installed by earlier rounds.")  # PREREG T16, word for word
EDIT_LOG_ANCHOR = "\n\n## What is kept\n"  # the sentence ends the system prompt's first section
PROTECT_SCHEMA = "weightcode.harness.protect_logged.v0"  # PREREG T16 follow-up M: the protected-entry limit
PROTECT_SENTENCE = ("A candidate or final that changes any entry listed in edit_log.json is refused by the "
                    "limits.")  # follow-up M, to be fixed in PREREG before any M session
PROTECT_COMPARE = ("an entry is changed when its float32 value in the child differs from that in the round's student "
                   "(!=), as in the edit log's derivation")


class LoopError(ValueError):
    """A loop context, part or record that is not a usable loop round."""


def prev_part(k: int) -> str:
    """The check-part name of earlier round ``k``."""
    return f"prev_{int(k)}"


def prev_score_keys(rounds: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"score_{prev_part(k)}" for k in rounds)


def load_npz_checked(path: Path, expected_sha256: str, what: str) -> dict[str, np.ndarray]:
    """Every array of an .npz read once from bytes that must hash to ``expected_sha256``."""
    data = Path(path).read_bytes()
    digest = bytes_sha256(data)
    if digest != expected_sha256:
        raise LoopError(f"{what}: file hash {digest} != recorded {expected_sha256}")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# ---------------------------------------------------------------------------------------------------------------
# The context file


def read_loop_context(path: Path) -> dict[str, Any]:
    """The loop context at ``path``, checked for form and for every file hash it names (module docstring)."""
    path = Path(path)
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LoopError(f"{path} is missing") from None
    if not isinstance(context, dict) or context.get("schema") != LOOP_CONTEXT_SCHEMA:
        raise LoopError(f"{path} is not a {LOOP_CONTEXT_SCHEMA} record")
    r = context.get("round")
    if not isinstance(r, int) or isinstance(r, bool) or not 2 <= r <= MAX_ROUNDS:
        raise LoopError(f"a loop session is round 2 to {MAX_ROUNDS}, not {r!r}")
    prev, library = context.get("prev"), context.get("library")
    if not isinstance(prev, list) or [item.get("round") for item in prev] != list(range(1, r)):
        raise LoopError(f"round {r} needs the evidence of rounds 1 to {r - 1}, in order")
    if not isinstance(library, list) or [item.get("round") for item in library] != list(range(1, r)):
        raise LoopError(f"round {r} needs the library entries of rounds 1 to {r - 1}, in order")
    ids = [str(item["instance_id"]) for item in prev] + [str(context.get("instance_id"))]
    if len(set(ids)) != len(ids):
        raise LoopError(f"an instance appears twice in the loop context: {ids}")
    student = context.get("student") or {}
    params = load_params(Path(str(student.get("path"))))
    if params_sha256(params) != student.get("sha256"):
        raise LoopError("the loop student file does not hash to its recorded parameter hash")
    for item in prev:
        arrays = load_npz_checked(Path(str(item["evidence"]["path"])), str(item["evidence"]["sha256"]),
                                  f"round {item['round']} evidence")
        if sorted(arrays) != sorted(EVIDENCE_KEYS) or len({len(arrays[key]) for key in EVIDENCE_KEYS}) != 1:
            raise LoopError(f"round {item['round']} evidence must hold {EVIDENCE_KEYS} of one length")
    for item in library:
        program = item.get("program")
        if program is not None and bytes_sha256(Path(str(program["path"])).read_bytes()) != program["sha256"]:
            raise LoopError(f"round {item['round']}: the kept program does not hash to its record")
        scores = item.get("scores") or {}
        expected = {"score_new", "score_old", "score", *prev_score_keys(range(1, int(item["round"])))}
        if program is not None and set(scores) != expected:
            raise LoopError(f"round {item['round']}: library scores {sorted(scores)} are not {sorted(expected)}")
    if "kept_rule" in context:
        context_clean_used(context)
    if "edit_log" in context:
        read_context_edit_log(context)
    if "protect_logged" in context:
        check_context_protect(context)
    return context


ROUND1_SESSION_CHECK = "round1_session_check"
"""``clean_used`` source of round 1: the kept check of the round-1 session, on the same clean rows and the same child."""


def context_clean_used(context: Mapping[str, Any]) -> dict[int, int]:
    """The kept score_old of each earlier round that a v2 loop context states (``kept_rule.clean_used``), checked
    against its library: a round that kept no update counts 0, a round from 2 on counts its kept check's score_old,
    and round 1 counts its session's kept score_old when that is the stated source (else the stream's own count on
    the loop's clean rows, with its source). Raises ``LoopError``."""
    spec = context.get("kept_rule")
    r = int(context["round"])
    if not isinstance(spec, Mapping) or spec.get("name") != KEPT_RULE_LOOP_V2:
        raise LoopError(f"a loop context's kept_rule names {KEPT_RULE_LOOP_V2}, not {spec!r}")
    rows = spec.get("clean_used")
    if not isinstance(rows, list) or [item.get("round") for item in rows] != list(range(1, r)):
        raise LoopError(f"round {r} needs the kept score_old of rounds 1 to {r - 1}, in order")
    used: dict[int, int] = {}
    for item, entry in zip(rows, context["library"]):
        k, value = int(item["round"]), item.get("s_old")
        if not isinstance(value, int) or isinstance(value, bool):
            raise LoopError(f"round {k}: the kept score_old is {value!r}, not an int")
        program = entry.get("program")
        if program is None and value != 0:
            raise LoopError(f"round {k} kept no update, so it counts 0 of the clean budget, not {value}")
        from_library = k >= 2 or (item.get("source") or {}).get("kind") == ROUND1_SESSION_CHECK
        if program is not None and from_library and value != int(entry["scores"]["score_old"]):
            raise LoopError(f"round {k}: the kept score_old {value} is not the library's {entry['scores']['score_old']}")
        used[k] = value
    return used


def load_student(context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    student = context["student"]
    params = load_params(Path(str(student["path"])))
    if params_sha256(params) != student["sha256"]:
        raise LoopError("the loop student file does not hash to its recorded parameter hash")
    return params


def load_prev_evidence(context: Mapping[str, Any]) -> list[tuple[int, dict[str, np.ndarray]]]:
    return [(int(item["round"]), load_npz_checked(Path(str(item["evidence"]["path"])), str(item["evidence"]["sha256"]),
                                                  f"round {item['round']} evidence"))
            for item in context["prev"]]


# ---------------------------------------------------------------------------------------------------------------
# Experience


def kept_prev_rows(prev: Sequence[tuple[int, Mapping[str, np.ndarray]]], new_idx: np.ndarray
                   ) -> tuple[list[tuple[int, np.ndarray]], dict[str, int]]:
    """For each earlier round, the row positions kept by the one-image rule (module docstring), and the dropped
    counts per round. Rounds are taken in order; a row is kept when its dataset idx is in no new row and in no row
    kept so far."""
    seen = {int(value) for value in np.asarray(new_idx).tolist()}
    kept: list[tuple[int, np.ndarray]] = []
    dropped: dict[str, int] = {}
    for k, arrays in prev:
        positions = []
        for position, value in enumerate(np.asarray(arrays["idx"]).tolist()):
            if int(value) in seen:
                continue
            seen.add(int(value))
            positions.append(position)
        kept.append((k, np.asarray(positions, dtype=np.int64)))
        dropped[str(k)] = int(len(arrays["idx"]) - len(positions))
    return kept, dropped


def loop_experience(instance: Any, params: Mapping[str, np.ndarray],
                    prev: Sequence[tuple[int, Mapping[str, np.ndarray]]]) -> tuple[dict[str, np.ndarray],
                                                                                     dict[str, Any]]:
    """Round r's experience on the current student ``params``: ``stream_experience`` plus the earlier rounds' rows
    under ``PREV_KEYS`` with this student's predictions (module docstring); the host-side record says what was kept."""
    from weightcode.evalkit.student_evidence import draw_evidence_for
    from weightcode.harness.program_stream import stream_experience

    experience, summary = stream_experience(instance, params)
    drawn = draw_evidence_for(instance, dict(params))
    if not np.array_equal(np.asarray(drawn.arrays["split_ids"]), experience["split_ids_new"]):
        raise LoopError("a second evidence draw for the same student gives other rows")
    new_idx = np.asarray(drawn.source_idx, dtype=np.int64)
    replay_idx = {int(value) for value in np.asarray(summary["replay"]["idx"]).tolist()}
    kept, dropped = kept_prev_rows(prev, new_idx)
    parts = {key: [] for key in ("x", "y", "split_ids", "round", "idx")}
    for (k, arrays), (_, positions) in zip(prev, kept):
        if replay_idx & {int(value) for value in np.asarray(arrays["idx"]).tolist()}:
            raise LoopError(f"round {k} evidence shares a dataset idx with the replay rows")
        parts["x"].append(np.asarray(arrays["x"])[positions])
        parts["y"].append(np.asarray(arrays["y"])[positions].astype(np.int64))
        parts["split_ids"].append(np.asarray(arrays["split_ids"])[positions])
        parts["round"].append(np.full(len(positions), k, dtype=np.int64))
        parts["idx"].append(np.asarray(arrays["idx"])[positions].astype(np.int64))
    x_prev = np.ascontiguousarray(np.concatenate(parts["x"]))
    if x_prev.shape[1:] != experience["x_new"].shape[1:] or x_prev.dtype != experience["x_new"].dtype:
        raise LoopError(f"earlier rows are {x_prev.dtype}{x_prev.shape[1:]}, the new rows "
                        f"{experience['x_new'].dtype}{experience['x_new'].shape[1:]}")
    split_ids_prev = np.ascontiguousarray(np.concatenate(parts["split_ids"]))
    others = set(experience["split_ids_new"].tolist()) | set(experience["split_ids_old"].tolist())
    if len(set(split_ids_prev.tolist())) != len(split_ids_prev) or others & set(split_ids_prev.tolist()):
        raise LoopError("an earlier row's id repeats, or equals a new or replay id")
    prev_arrays = {
        "x_prev": x_prev,
        "y_prev": np.ascontiguousarray(np.concatenate(parts["y"])),
        "pred_prev": predict(instance.forward, params, x_prev, PREDICT_BATCH),
        "round_prev": np.ascontiguousarray(np.concatenate(parts["round"])),
        "split_ids_prev": split_ids_prev,
    }
    result = {**experience, **prev_arrays}
    per_round = {str(k): int(len(positions)) for k, positions in kept}
    record = {
        **summary,
        "content_sha256": params_sha256(result),
        "stream_content_sha256": summary["content_sha256"],
        "n_prev": int(len(x_prev)),
        "prev_per_round": per_round,
        "prev_dropped_shared_image": dropped,
        "prev_idx": np.concatenate(parts["idx"]).astype(int).tolist(),
        "new_source_idx": new_idx.astype(int).tolist(),
    }
    return result, record


def round_evidence(experience: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The evidence rows a round drew for its own condition (``EVIDENCE_KEYS``), kept for the later rounds."""
    return {"x": np.ascontiguousarray(experience["x_new"]), "y": np.ascontiguousarray(experience["y_new"]),
            "split_ids": np.ascontiguousarray(experience["split_ids_new"]),
            "idx": np.asarray(record["new_source_idx"], dtype=np.int64)}


# ---------------------------------------------------------------------------------------------------------------
# The check


class MultiPartOracle:
    """Counted check on several image sets under one budget (PREREG E4-loop; ``JointPatchCheckOracle`` with more parts).

    ``parts`` maps ``new``, ``old`` and then ``prev_1``, ``prev_2``, ... to (inputs, labels). Each part is
    S = fixed - broken over its own images against ``base_params``; ``score`` is the sum over all parts, and the
    unsuffixed ``fixed``, ``broken``, ``n`` and ``n_base_errors`` are pooled. Per-part fields carry the suffix
    ``_<part>``, so ``new`` and ``old`` read as in ``JointPatchCheckOracle``. Validation failures raise before any
    budget is consumed; callers get aggregates only.
    """

    def __init__(self, parts: Mapping[str, tuple[np.ndarray, np.ndarray]], forward: ForwardFn,
                 base_params: Mapping[str, np.ndarray], budget: int, batch_size: int = 1024) -> None:
        names = list(parts)
        expected = ["new", "old", *(prev_part(k) for k in range(1, len(names) - 1))]
        if names != expected:
            raise ValueError(f"the check parts must be {expected}, in order, not {names}")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self._x = {name: x for name, (x, _) in parts.items()}
        self._y = {name: np.asarray(y, dtype=np.int64) for name, (_, y) in parts.items()}
        for name in names:
            if not np.issubdtype(self._x[name].dtype, np.integer):
                raise TypeError(f"{name} check inputs must be an integer array, got {self._x[name].dtype}")
            if len(self._x[name]) != len(self._y[name]) or len(self._y[name]) == 0:
                raise ValueError(f"{name} check set has {len(self._x[name])} inputs and {len(self._y[name])} labels")
        self.part_names = tuple(names)
        self._forward = forward
        self._base = {key: np.ascontiguousarray(value) for key, value in base_params.items()}
        self._base_sha = params_sha256(self._base)
        self._budget = int(budget)
        self._batch_size = batch_size
        self._base_pred = {name: predict(forward, self._base, self._x[name], batch_size) for name in names}
        self._used = 0
        self._log: list[dict[str, object]] = []

    @property
    def base_sha256(self) -> str:
        return self._base_sha

    @property
    def queries_remaining(self) -> int:
        return self._budget - self._used

    @property
    def log(self) -> list[dict[str, object]]:
        return [dict(entry) for entry in self._log]

    def n_rows(self) -> dict[str, int]:
        return {name: int(len(self._y[name])) for name in self.part_names}

    def query(self, child_params: Mapping[str, np.ndarray], patch_id: str) -> dict[str, object]:
        """Evaluate one child on every part. Validation failures raise before any budget is consumed."""
        child = validate_child(self._base, child_params)
        if any(entry["patch_id"] == patch_id for entry in self._log):
            raise ValueError(f"patch id {patch_id!r} was already checked")
        if self._used >= self._budget:
            raise BudgetExceeded(f"patch-check budget of {self._budget} queries is spent")
        parts = {name: summarize(self._y[name], self._base_pred[name],
                                 predict(self._forward, child, self._x[name], self._batch_size)).as_dict()
                 for name in self.part_names}
        query_index = self._used
        self._used += 1
        pooled = {key: sum(int(parts[name][key]) for name in self.part_names)  # type: ignore[call-overload]
                  for key in ("score", "fixed", "broken", "n", "n_base_errors")}
        base_right = pooled["n"] - pooled["n_base_errors"]
        entry: dict[str, object] = {
            "query_index": query_index,
            "patch_id": patch_id,
            "child_sha256": params_sha256(child),
            "n_changed": count_changed(self._base, child),
            **pooled,
            "base_acc": base_right / pooled["n"],
            "child_acc": (base_right + pooled["score"]) / pooled["n"],
            **{f"{key}_{name}": value for name in self.part_names for key, value in parts[name].items()
               if value is not None},
            "queries_remaining": self.queries_remaining,
        }
        self._log.append(entry)
        return dict(entry)


def part_fields(n_prev: int) -> tuple[str, ...]:
    """The per-part score, fixed and broken fields of a loop check with ``n_prev`` earlier rounds."""
    return tuple(f"{key}_{prev_part(k)}" for k in range(1, n_prev + 1) for key in ("score", "fixed", "broken"))


def shown_check_fields(n_prev: int) -> tuple[str, ...]:
    """The check fields the teacher sees: score_new, score_old, score_prev_k for every earlier round, score."""
    return ("score_new", "score_old", *prev_score_keys(range(1, n_prev + 1)), "score")


# ---------------------------------------------------------------------------------------------------------------
# The kept rule


def loop_kept_rule(n_old: int, n_prev: Mapping[int, int], clean_frac: float = LOOP_CLEAN_FRAC,
                   prev_frac: float = LOOP_PREV_FRAC) -> dict[str, Any]:
    """The session.json ``kept_rule`` of a loop round: the clean floor and one floor per earlier round."""
    if not n_prev:
        raise LoopError("a loop round has at least one earlier round")
    return {
        "name": KEPT_RULE_LOOP, "clean_frac": float(clean_frac), "prev_frac": float(prev_frac), "n_old": int(n_old),
        "n_prev": {str(k): int(n) for k, n in sorted(n_prev.items())}, "check_set": "clean",
        "s_old_floor": s_old_floor(clean_frac, n_old),
        "s_prev_floor": {str(k): s_old_floor(prev_frac, n) for k, n in sorted(n_prev.items())},
    }


def loop_kept_rule_v2(n_old: int, n_prev: Mapping[int, int], clean_used: Mapping[int, int],
                      budget_frac: float = LOOP_CLEAN_BUDGET_FRAC, prev_frac: float = LOOP_PREV_FRAC) -> dict[str, Any]:
    """The ``kept_rule`` of a round under loop-retain-v2 (module docstring): the clean floor is the stream's clean
    budget, ``budget_frac`` of the ``n_old`` clean rows, less the kept score_old of every earlier round
    (``clean_used``, one entry per round in ``n_prev``; none in a round 1 that is selected under the rule, as in
    control (b)); the earlier-condition floors are v1's. A budget already overspent (a floor above 0) is refused."""
    if sorted(int(k) for k in clean_used) != sorted(int(k) for k in n_prev):
        raise LoopError(f"the clean budget counts every earlier round: rounds {sorted(n_prev)}, counted "
                        f"{sorted(clean_used)}")
    budget = s_old_floor(budget_frac, n_old)
    total = int(sum(int(value) for value in clean_used.values()))
    floor = budget - total
    if floor > 0:
        raise LoopError(f"the earlier rounds lost {-total} clean rows, more than the stream's budget of {-budget}")
    return {
        "name": KEPT_RULE_LOOP_V2, "clean_budget_frac": float(budget_frac), "prev_frac": float(prev_frac),
        "n_old": int(n_old), "n_prev": {str(k): int(n) for k, n in sorted(n_prev.items())}, "check_set": "clean",
        "clean_budget_floor": budget, "clean_used": {str(k): int(v) for k, v in sorted(clean_used.items())},
        "clean_used_total": total, "s_old_floor": floor,
        "s_prev_floor": {str(k): s_old_floor(prev_frac, n) for k, n in sorted(n_prev.items())},
    }


def kept_rule_for(name: str, n_old: int, n_prev: Mapping[int, int],
                  clean_used: Mapping[int, int] | None = None) -> dict[str, Any]:
    """The ``kept_rule`` of a loop round under the rule ``name``: v1 takes no ``clean_used``, v2 needs it."""
    if name == KEPT_RULE_LOOP:
        if clean_used is not None:
            raise LoopError(f"{KEPT_RULE_LOOP} has a per-round clean floor; it counts no earlier score_old")
        return loop_kept_rule(n_old, n_prev)
    if name == KEPT_RULE_LOOP_V2:
        if clean_used is None:
            raise LoopError(f"{KEPT_RULE_LOOP_V2} needs the kept score_old of every earlier round")
        return loop_kept_rule_v2(n_old, n_prev, clean_used)
    raise LoopError(f"the loop kept rule is one of {KEPT_RULES}, not {name!r}")


def context_kept_rule(context: Mapping[str, Any], n_old: int, n_prev: Mapping[int, int]) -> dict[str, Any]:
    """The kept rule a loop context asks for: v1 without a ``kept_rule`` key, else v2 with the context's counts; the
    budget fraction, the budget and the floor the stream recorded must be the ones the rule gives."""
    spec = context.get("kept_rule")
    if spec is None:
        return loop_kept_rule(n_old, n_prev)
    rule = loop_kept_rule_v2(n_old, n_prev, context_clean_used(context), float(spec["clean_budget_frac"]))
    for key in ("clean_budget_floor", "s_old_floor"):
        if key in spec and int(spec[key]) != int(rule[key]):
            raise LoopError(f"the loop context states {key} {spec[key]}, the rule gives {rule[key]}")
    return rule


def session_loop_rule(session: Mapping[str, Any]) -> dict[str, Any] | None:
    """The kept rule of a loop session (floors recomputed and compared), or None for a session that is not one."""
    if "loop" not in session:
        return None
    rule = session.get("kept_rule")
    if not isinstance(rule, Mapping) or rule.get("name") not in KEPT_RULES:
        raise LoopError(f"a loop session has kept rule {KEPT_RULE_LOOP} or {KEPT_RULE_LOOP_V2}, not "
                        f"{None if rule is None else rule.get('name')!r}")
    n_prev = {int(k): int(n) for k, n in rule["n_prev"].items()}
    if rule["name"] == KEPT_RULE_LOOP_V2:
        again = loop_kept_rule_v2(int(rule["n_old"]), n_prev, {int(k): int(v) for k, v in rule["clean_used"].items()},
                                  float(rule["clean_budget_frac"]), float(rule["prev_frac"]))
    else:
        again = loop_kept_rule(int(rule["n_old"]), n_prev, float(rule["clean_frac"]), float(rule["prev_frac"]))
    if again != dict(rule):
        raise LoopError("the loop kept rule's floors do not follow from its fractions and row counts")
    return dict(rule)


def eligible(entry: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    """Whether a checked program clears both floors of ``rule``."""
    if int(entry["score_old"]) < int(rule["s_old_floor"]):
        return False
    return all(int(entry[f"score_{prev_part(int(k))}"]) >= int(floor) for k, floor in rule["s_prev_floor"].items())


def _floors_text(rule: Mapping[str, Any]) -> tuple[int, int]:
    floors = {int(value) for value in rule["s_prev_floor"].values()}
    if len(floors) != 1:
        raise LoopError("the texts state one floor for every earlier round; these rounds have different row counts")
    return int(rule["s_old_floor"]), floors.pop()


def _budget_text(rule: Mapping[str, Any]) -> tuple[int, str] | None:
    """For a v2 rule, the stream's clean budget in images and what the earlier rounds' kept updates did with it; None
    for v1."""
    if rule.get("name") != KEPT_RULE_LOOP_V2:
        return None
    total = int(rule["clean_used_total"])
    if total < 0:
        used = f"the updates kept in the earlier rounds used {-total} of it"
    elif total == 0:
        used = "the updates kept in the earlier rounds used none of it"
    else:
        used = f"the updates kept in the earlier rounds fixed {total} more than they broke, which adds {total} to it"
    return -int(rule["clean_budget_floor"]), used


# ---------------------------------------------------------------------------------------------------------------
# Texts


def loop_statement(n_earlier: int) -> str:
    rounds = "one earlier round" if n_earlier == 1 else f"{n_earlier} earlier rounds"
    return (
        f"A trained image classifier has been updated in {rounds} of a stream, each round for another input "
        "condition. Make it handle a new input condition while it keeps what it already does, on clean images and "
        "on the conditions of the earlier rounds. The experience file holds labelled examples from the new condition, "
        "on which the classifier makes errors, labelled clean examples from the data it was trained on, and labelled "
        "examples from the conditions of the earlier rounds, with the classifier's predictions on all of them. Write "
        "a program that computes an additive update of the classifier's weights from the classifier and its "
        "experience, without training it, so that the updated classifier classifies more new-condition images "
        "correctly while the clean images and the earlier conditions' images it already classifies correctly stay "
        "correct."
    )


LOOP_READOUT = (
    "each query runs the program once on this classifier and its experience, applies the update within the limits and "
    "returns, each against the unedited classifier (this round's classifier, as the earlier rounds left it): score_new "
    "= fixed - broken on a held-out check set of new-condition images, score_old = fixed - broken on a held-out check "
    "set of clean images, score_prev_k = fixed - broken on a held-out check set of round k's condition for each "
    "earlier round k, and score = the sum of all of them; fixed counts images the unedited classifier got wrong that "
    "the updated one gets right, broken counts images it got right that the updated one gets wrong; no per-image "
    "information"
)


def loop_final_selection(rule: Mapping[str, Any]) -> str:
    s_old, s_prev = _floors_text(rule)
    budget = _budget_text(rule)
    clean = "" if budget is None else (
        f" (the rest of this stream's clean budget: over all its rounds the kept updates may break at most "
        f"{budget[0]} more clean check-set images than they fix, and {budget[1]})")
    return (
        f"kept at the end: among the checked programs with score_old >= {s_old}{clean} and score_prev_k >= {s_prev} "
        "for every earlier round k, and the unedited classifier (score = 0), the update computed by the one with the "
        "highest score; ties go to fewer changed parameters, then to the earlier query; a program below either floor, "
        "a program with score <= 0 and an unchecked program are never kept"
    )


def loop_task(task: Mapping[str, Any], rule: Mapping[str, Any], n_earlier: int, n_prev: int,
              per_round: Mapping[str, int]) -> dict[str, Any]:
    """The LEARN task.json of a loop round: the statement, the earlier rows, the readout and the final selection."""
    result = json.loads(json.dumps(task))
    result["statement"] = loop_statement(n_earlier)
    result["experience"]["prev"] = {
        "keys": ["x_prev", "y_prev", "pred_prev", "round_prev"], "count": int(n_prev),
        "per_round": {str(k): int(n) for k, n in per_round.items()},
        "description": ("examples from the input conditions of the earlier rounds, the ones each round drew for its "
                        "own condition, with this classifier's predictions; round_prev is each example's round"),
    }
    result["experience"]["ids"]["keys"] = ["split_ids_new", "split_ids_old", "split_ids_prev"]
    result["experience"]["ids"]["description"] = "one opaque id per example, in x_new, x_old and x_prev order"
    result["patch_check"]["readout"] = LOOP_READOUT
    result["patch_check"]["final_selection"] = loop_final_selection(rule)
    return result


V1_KEPT_BODY = (
    "After the session, among the programs you checked with `patch_check`, the update of the one with the highest "
    "score = score_new + score_old is kept. Each part is fixed - broken on its own held-out check set, one of "
    "new-condition images (score_new) and one of clean images (score_old). fixed counts check-set images the original "
    "model got wrong and the updated model gets right; broken counts check-set images the original model got right "
    "and the updated model gets wrong. The check sets are held out and you never see their images. If no checked "
    "program has a score > 0, the original model is kept. Unchecked programs are never kept."
)
KEPT_HEADING = "## What is kept\n\n"
PROMPT_WIDTH = 116


def loop_kept_section(text: str, rule: Mapping[str, Any]) -> str:
    """``text``, the LEARN system prompt, with its "What is kept" section stating the loop rule; every other byte is
    kept. The section must state the v1 rule word for word (``V1_KEPT_BODY``)."""
    s_old, s_prev = _floors_text(rule)
    start = text.index(KEPT_HEADING) + len(KEPT_HEADING)
    end = text.index("\n\n## ", start)
    if " ".join(text[start:end].split()) != V1_KEPT_BODY:
        raise LoopError("the What is kept section does not state the v1 kept rule word for word")
    budget = _budget_text(rule)
    clean = "" if budget is None else (
        f"The floor {s_old} on score_old is what is left of this stream's clean budget: over all its rounds together, "
        f"the kept updates may break at most {budget[0]} more clean check-set images than they fix, and {budget[1]}, "
        f"so this round's update may break at most {-s_old} more than it fixes. ")
    body = (
        f"After the session, among the programs you checked with `patch_check` whose score_old is at least {s_old} "
        f"and whose score_prev_k is at least {s_prev} for every earlier round k, the update of the one with the highest "
        f"score = score_new + score_old + the score_prev_k of every earlier round is kept. {clean}Each part is fixed - broken "
        "on its own held-out check set: one of new-condition images (score_new), one of clean images (score_old) and "
        "one of each earlier round's condition (score_prev_k). fixed counts check-set images the original model got "
        "wrong and the updated model gets right; broken counts check-set images the original model got right and the "
        "updated model gets wrong. The original model is this round's classifier, as the earlier rounds left it. The "
        "check sets are held out and you never see their images. A program below either floor is never kept, whatever "
        f"its score. If no checked program at or above both floors has a score > 0, the original model is kept. "
        "Unchecked programs are never kept."
    )
    wrapped = textwrap.fill(body, width=PROMPT_WIDTH, break_long_words=False, break_on_hyphens=False)
    return text[:start] + wrapped + text[end:]


def loop_check_description(base: str, check_sets_phrase: str) -> str:
    """The program arm's ``patch_check`` description ``base`` with the loop's check sets and scores."""
    old_sets = "evaluated on two held-out check sets, " + check_sets_phrase + ". Returns score_new and score_old "
    old_sets += "(fixed - broken on each set), score = score_new + score_old (S_joint),"
    if base.count(old_sets) != 1:
        raise LoopError("the program patch_check description no longer states its two check sets once")
    return base.replace(old_sets, (
        "evaluated on held-out check sets: one of new-condition images, one of clean images and one of each earlier "
        "round's condition. Returns score_new, score_old and score_prev_k for each earlier round k (fixed - broken on "
        "each set, against this classifier), score = the sum of all of them,"))


LOOP_STATUS_DESCRIPTION = ("Queries remaining, Python cells remaining, every checked program with its score_new, "
                           "score_old, score_prev_k and score, and your last declaration.")


def loop_declare_description(base: str, kept_phrase: str, rule: Mapping[str, Any]) -> str:
    s_old, s_prev = _floors_text(rule)
    if base.count(kept_phrase) != 1:
        raise LoopError("declare_final does not state the v1 kept rule")
    budget = _budget_text(rule)
    clean = "" if budget is None else (
        f" (what is left of this stream's clean budget of {budget[0]} clean check-set images, broken minus fixed, over "
        "all its rounds)")
    return base.replace(kept_phrase, (
        f"the checked program with the highest score among those with score_old of at least {s_old}{clean} and every "
        f"score_prev_k of at least {s_prev} (the original model if none of them has score > 0)"))


FILES_ANCHOR = "\n\n## Tools\n"


def loop_files_section(text: str, n_prev: int, library_files: int) -> str:
    """``text``, a rendered LEARN task prompt, with bullets for the earlier rows and the round library at the end of
    its file list; every other byte is kept."""
    if text.count(FILES_ANCHOR) != 1:
        raise LoopError("the task prompt has no single file list ending before its Tools section")
    bullets = [
        f"- `experience.npz` also holds `x_prev`, `y_prev`, `pred_prev`, `round_prev` and `split_ids_prev`: {n_prev} "
        "labelled examples from the input conditions of the earlier rounds of this stream, the ones each round drew "
        "for its own condition, with this classifier's predictions; `round_prev` gives each example's round.",
    ]
    if library_files:
        bullets.append(
            f"- `{LOOP_LIBRARY_INDEX}` and the `library_program_<k>.py` files: the update program kept in each earlier "
            "round k of this stream, as it was written, with the scores of its kept `patch_check` in that round, "
            "against the classifier as that round found it. A round that kept no update has no program file.")
    wrapped = "\n".join(textwrap.fill(bullet, width=PROMPT_WIDTH, subsequent_indent="  ", break_long_words=False,
                                      break_on_hyphens=False) for bullet in bullets)
    at = text.index(FILES_ANCHOR)
    return text[:at] + "\n" + wrapped + text[at:]


# ---------------------------------------------------------------------------------------------------------------
# The round library


def withheld_tokens(library: Sequence[Mapping[str, Any]], forbidden: Sequence[str]) -> dict[int, list[str]]:
    """For each earlier round whose kept program text holds a forbidden string, those strings. Such a program is
    withheld from the trial (its scores are still listed), because every trial file is leak-scanned. ``forbidden`` is
    the list the trial scan applies to library program files (``ForbiddenStrings.library_scan``: on E10, E4-loop's
    standard without the E10 leak words), and the match is that scan's own (``weightcode.evalkit.instances.scan_for_tokens``:
    case-insensitive, short tokens as whole words), so a program is withheld exactly when the scan would refuse the
    session for it."""
    from weightcode.evalkit.instances import scan_for_tokens

    withheld: dict[int, list[str]] = {}
    for item in library:
        program = item.get("program")
        if program is None:
            continue
        text = Path(str(program["path"])).read_bytes().decode("utf-8", errors="replace")
        hits = scan_for_tokens(text, [token for token in forbidden if token])
        if hits:
            withheld[int(item["round"])] = hits
    return withheld


def library_document(library: Sequence[Mapping[str, Any]], n_check: int,
                     withheld: Mapping[int, Sequence[str]] | None = None) -> dict[str, Any]:
    """The trial's ``library_index.json`` for the earlier rounds: labels, files and part scores only."""
    withheld = dict(withheld or {})
    rounds = []
    for item in library:
        k = int(item["round"])
        program = item.get("program")
        rounds.append({
            "label": ROUND_LABEL.format(k=k), "round": k,
            "file": None if program is None or k in withheld else LOOP_PROGRAM_TEMPLATE.format(k=k),
            "kept_update": program is not None,
            "scores": None if program is None else {key: int(value) for key, value in item["scores"].items()},
        })
    return {
        "schema": LOOP_LIBRARY_SCHEMA,
        "about": ("The update program kept in each earlier round of this stream, as it was written, with the scores "
                  "of its kept patch_check in that round. Each round's check was against the classifier as that "
                  "round found it: score_new on that round's own condition, score_old on the same held-out clean check "
                  "set as this task's, and score_prev_j on the condition of each round j before it. Every check set "
                  "has n_check images. kept_update false: the round kept no update. kept_update true with file "
                  "null: the program is not shown here, and only its scores are."),
        "n_check": int(n_check),
        "rounds": rounds,
    }


def write_loop_library(trial_dir: Path, library: Sequence[Mapping[str, Any]], n_check: int,
                       withheld: Mapping[int, Sequence[str]] | None = None) -> dict[str, Any]:
    """Write the round library into ``trial_dir``; return the session.json ``library`` record."""
    trial_dir = Path(trial_dir)
    withheld = dict(withheld or {})
    files: dict[str, str] = {}
    for item in library:
        program = item.get("program")
        if program is None or int(item["round"]) in withheld:
            continue
        data = Path(str(program["path"])).read_bytes()
        if bytes_sha256(data) != program["sha256"]:
            raise LoopError(f"round {item['round']}: the kept program changed")
        path = trial_dir / LOOP_PROGRAM_TEMPLATE.format(k=int(item["round"]))
        if path.exists():
            raise LoopError(f"{path} already exists")
        path.write_bytes(data)
        files[path.name] = bytes_sha256(path.read_bytes())
    index_path = trial_dir / LOOP_LIBRARY_INDEX
    if index_path.exists():
        raise LoopError(f"{index_path} already exists")
    index = library_document(library, n_check, withheld)
    index_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    files[LOOP_LIBRARY_INDEX] = bytes_sha256(index_path.read_bytes())
    return {"schema": LOOP_LIBRARY_SCHEMA, "files": files,
            "withheld": {str(k): list(tokens) for k, tokens in withheld.items()},
            "entries": [{"round": int(item["round"]), "program": item.get("program"), "scores": item.get("scores"),
                         "source": item.get("source")} for item in library]}


# ---------------------------------------------------------------------------------------------------------------
# The edit log (PREREG T16, arm L)


def edit_log_round(k: int, before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Round ``k``'s entry of the edit log: every parameter entry where ``after`` (the student round k kept) differs
    from ``before`` (the student round k found), per tensor in the model's tensor order, with the flat indices (C
    order) and, for a 2-D tensor, the rows and the columns that hold a change, in the tensor order of ``before``. A
    tensor with no change is not listed."""
    if set(before) != set(after):
        raise LoopError(f"round {k}: the students before and after have other tensors")
    tensors = []
    for name, value in before.items():
        other = np.asarray(after[name])
        if np.asarray(value).shape != other.shape:
            raise LoopError(f"round {k}: tensor {name} changed shape")
        flat = np.flatnonzero(np.asarray(value) != other)
        if not len(flat):
            continue
        item: dict[str, Any] = {"tensor": str(name), "flat_indices": flat.astype(int).tolist()}
        if other.ndim == 2:
            rows, columns = np.unravel_index(flat, other.shape)
            item["rows"] = np.unique(rows).astype(int).tolist()
            item["columns"] = np.unique(columns).astype(int).tolist()
        tensors.append(item)
    return {"round": int(k), "tensors": tensors}


def edit_log_document(rounds: Sequence[tuple[int, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]]
                      ) -> dict[str, Any]:
    """The trial's ``edit_log.json`` for ``rounds`` = (k, student before round k, student after round k), rounds 1 to
    r - 1 in order: ``{"rounds": [edit_log_round, ...]}`` and nothing else."""
    if [int(k) for k, _, _ in rounds] != list(range(1, len(rounds) + 1)):
        raise LoopError("the edit log lists rounds 1 to r - 1, in order")
    return {"rounds": [edit_log_round(int(k), before, after) for k, before, after in rounds]}


def edit_log_bytes(document: Mapping[str, Any]) -> bytes:
    """The file's bytes: one JSON line, keys in the order written, so that equal logs give equal files."""
    return (json.dumps(document, separators=(", ", ": ")) + "\n").encode("utf-8")


def edit_log_counts(document: Mapping[str, Any]) -> dict[str, int]:
    """The number of logged entries of each round."""
    return {str(item["round"]): int(sum(len(t["flat_indices"]) for t in item["tensors"]))
            for item in document["rounds"]}


def check_edit_log(data: bytes, r: int, shapes: Mapping[str, tuple[int, ...]] | None = None) -> dict[str, Any]:
    """The edit log of round ``r`` read from ``data``: exactly the bytes ``edit_log_bytes`` gives for its document,
    rounds 1 to r - 1 in order, each tensor named once, flat indices strictly increasing, and (given ``shapes``, the
    current student's) every tensor known, every index in range and the rows and columns of a 2-D tensor exactly those
    of its indices. Raises ``LoopError``."""
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoopError(f"the edit log is not JSON: {exc}") from None
    if not isinstance(document, dict) or list(document) != ["rounds"] or edit_log_bytes(document) != data:
        raise LoopError("the edit log is not a {\"rounds\": [...]} document in its canonical form")
    rounds = document["rounds"]
    if not isinstance(rounds, list) or [item.get("round") for item in rounds] != list(range(1, r)):
        raise LoopError(f"the edit log of round {r} lists rounds 1 to {r - 1}, in order")
    for item in rounds:
        names = [t.get("tensor") for t in item.get("tensors") or []]
        if len(set(names)) != len(names):
            raise LoopError(f"round {item['round']}: a tensor is listed twice in the edit log")
        for tensor in item["tensors"]:
            flat = np.asarray(tensor.get("flat_indices"), dtype=np.int64)
            if not len(flat) or (len(flat) > 1 and not np.all(np.diff(flat) > 0)) or flat[0] < 0:
                raise LoopError(f"round {item['round']}, {tensor['tensor']}: flat indices must be non-empty and "
                                "strictly increasing")
            if shapes is None:
                continue
            if tensor["tensor"] not in shapes:
                raise LoopError(f"round {item['round']}: {tensor['tensor']} is not a tensor of the student")
            shape = tuple(shapes[tensor["tensor"]])
            if flat[-1] >= int(np.prod(shape)):
                raise LoopError(f"round {item['round']}, {tensor['tensor']}: a flat index is out of range")
            keys = ["tensor", "flat_indices", *(["rows", "columns"] if len(shape) == 2 else [])]
            if list(tensor) != keys:
                raise LoopError(f"round {item['round']}, {tensor['tensor']}: the entry's keys are not {keys}")
            if len(shape) == 2:
                rows, columns = np.unravel_index(flat, shape)
                if tensor["rows"] != np.unique(rows).tolist() or tensor["columns"] != np.unique(columns).tolist():
                    raise LoopError(f"round {item['round']}, {tensor['tensor']}: rows or columns are not those of "
                                    "its flat indices")
    return document


def read_context_edit_log(context: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]] | None = None
                          ) -> tuple[bytes, dict[str, Any]]:
    """The edit-log bytes a loop context names (``edit_log``: schema, path, sha256, per-round counts), hash-checked and
    checked by ``check_edit_log``; the stated counts must be the file's. Raises ``LoopError``."""
    spec = context.get("edit_log")
    if not isinstance(spec, Mapping) or spec.get("schema") != EDIT_LOG_SCHEMA:
        raise LoopError(f"a loop context's edit_log is a {EDIT_LOG_SCHEMA} record, not {spec!r}")
    try:
        data = Path(str(spec["path"])).read_bytes()
    except (KeyError, OSError) as exc:
        raise LoopError(f"the edit log cannot be read: {exc}") from None
    if bytes_sha256(data) != spec.get("sha256"):
        raise LoopError("the edit log does not hash to the loop context's record")
    document = check_edit_log(data, int(context["round"]), shapes)
    if edit_log_counts(document) != spec.get("n_entries"):
        raise LoopError(f"the loop context states {spec.get('n_entries')} logged entries, the file holds "
                        f"{edit_log_counts(document)}")
    return data, document


def edit_log_section(text: str) -> str:
    """``text``, a loop round's system prompt, with ``EDIT_LOG_SENTENCE`` as its own paragraph at the end of the first
    section (before "## What is kept"); every other byte is kept."""
    if text.count(EDIT_LOG_ANCHOR) != 1 or EDIT_LOG_SENTENCE in text:
        raise LoopError("the system prompt has no single What is kept heading, or already states the edit log")
    at = text.index(EDIT_LOG_ANCHOR)
    return text[:at] + "\n\n" + EDIT_LOG_SENTENCE + text[at:]


def write_edit_log(trial_dir: Path, data: bytes) -> dict[str, str]:
    """Write the edit log into ``trial_dir``; return ``{file name: sha256}`` for session.json ``trial_files_sha256``."""
    path = Path(trial_dir) / EDIT_LOG_FILE
    if path.exists():
        raise LoopError(f"{path} already exists")
    path.write_bytes(data)
    if bytes_sha256(path.read_bytes()) != bytes_sha256(data):
        raise LoopError(f"{path} does not hold the edit log")
    return {EDIT_LOG_FILE: bytes_sha256(data)}


# ---------------------------------------------------------------------------------------------------------------
# The protected-entry limit (PREREG T16 follow-up M; with the edit log only)


def protected_entries(document: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Every entry an edit log lists, over all its rounds: per tensor, in the order the log first names it, the sorted
    flat indices (C order), each once."""
    union: dict[str, set[int]] = {}
    for item in document["rounds"]:
        for tensor in item["tensors"]:
            union.setdefault(str(tensor["tensor"]), set()).update(int(i) for i in tensor["flat_indices"])
    return {name: np.asarray(sorted(indices), dtype=np.int64) for name, indices in union.items()}


def protected_counts(protected: Mapping[str, np.ndarray]) -> dict[str, int]:
    """The number of protected entries of each tensor."""
    return {name: int(len(indices)) for name, indices in protected.items()}


def protected_changes(protected: Mapping[str, np.ndarray], student: Mapping[str, np.ndarray],
                      child: Mapping[str, np.ndarray]) -> dict[str, int]:
    """For each tensor with protected entries, in ``student``'s tensor order, how many of them ``child`` changes;
    tensors with none are left out. An entry is changed when its float32 value in ``child`` differs from that in
    ``student`` (``!=``), the comparison the edit log's derivation uses (``edit_log_round``): a delta that leaves the
    float32 value as it was changes nothing, and -0.0 equals 0.0."""
    unknown = sorted(set(protected) - set(student))
    if unknown:
        raise LoopError(f"protected entries name tensors the student does not have: {unknown}")
    changes: dict[str, int] = {}
    for name, value in student.items():
        indices = protected.get(name)
        if indices is None:
            continue
        before = np.asarray(value).reshape(-1)[indices]
        after = np.asarray(child[name]).reshape(-1)[indices]
        count = int(np.count_nonzero(before != after))
        if count:
            changes[name] = count
    return changes


def protect_messages(changes: Mapping[str, int]) -> list[str]:
    """The limits refusal of an update that changes protected entries: how many, and in which tensors."""
    total = int(sum(changes.values()))
    where = ", ".join(f"{int(n)} in tensor {name!r}" for name, n in changes.items())
    return [f"protected entries: the update changes {total} {'entry' if total == 1 else 'entries'} listed in "
            f"{EDIT_LOG_FILE} ({where}); the limits refuse any change to a listed entry"]


def protect_section(text: str) -> str:
    """``text``, a loop round's system prompt with the edit-log sentence, with ``PROTECT_SENTENCE`` right after that
    sentence, in its paragraph; every other byte is kept."""
    if text.count(EDIT_LOG_SENTENCE) != 1 or PROTECT_SENTENCE in text:
        raise LoopError("the system prompt does not state the edit log once, or already states the protected-entry "
                        "limit")
    return text.replace(EDIT_LOG_SENTENCE, EDIT_LOG_SENTENCE + " " + PROTECT_SENTENCE, 1)


def check_context_protect(context: Mapping[str, Any]) -> dict[str, Any]:
    """A loop context's ``protect_logged`` record: the schema, the edit log it protects (the context's own, by
    sha256) and the number of entries that log lists. Raises ``LoopError``."""
    spec = context.get("protect_logged")
    if not isinstance(spec, Mapping) or spec.get("schema") != PROTECT_SCHEMA:
        raise LoopError(f"a loop context's protect_logged is a {PROTECT_SCHEMA} record, not {spec!r}")
    log = context.get("edit_log")
    if not isinstance(log, Mapping):
        raise LoopError("the protected-entry limit protects the entries of the round's edit log; the context has none")
    if spec.get("edit_log_sha256") != log.get("sha256"):
        raise LoopError("the protected-entry limit names another edit log than the context's")
    n = spec.get("n_protected")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise LoopError(f"n_protected is {n!r}, not a count")
    return dict(spec)


def round_protect(context: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    """session.json ``loop.protect_logged`` of a round whose context asks for the limit: the edit log protected (its
    sha256 and the trial file), the round's student every child is compared with (the student the log's last round
    kept, so that the log ends where this round starts), the protected entries per tensor and in all (the context's
    count must be the log's), the comparison, the sentence and the context's record."""
    spec = check_context_protect(context)
    log = context["edit_log"]
    student = str(context["student"]["sha256"])
    rounds = log.get("rounds") or []
    if not rounds or rounds[-1].get("child_sha256") != student:
        raise LoopError("the edit log does not end at this round's student: its last round kept another student")
    counts = protected_counts(protected_entries(document))
    if sum(counts.values()) != spec["n_protected"]:
        raise LoopError(f"the loop context states {spec['n_protected']} protected entries, the edit log lists "
                        f"{sum(counts.values())}")
    return {"enabled": True, "schema": PROTECT_SCHEMA, "file": EDIT_LOG_FILE, "edit_log_sha256": str(log["sha256"]),
            "student_sha256": student, "n_protected": int(spec["n_protected"]), "per_tensor": counts,
            "compare": PROTECT_COMPARE, "sentence": PROTECT_SENTENCE, "source": spec}


def session_protected_entries(session: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]]
                              ) -> dict[str, np.ndarray]:
    """The protected entries of a session whose session.json ``loop`` states ``protect_logged``, read from its trial
    edit_log.json: the file must hash to the session's edit-log record, its trial record and the limit's record, the
    session's student must be the limit's, and the entries per tensor must be the ones recorded. Raises
    ``LoopError``."""
    record = session.get("loop") or {}
    spec, log = record.get("protect_logged"), record.get("edit_log") or {}
    if not isinstance(spec, Mapping) or spec.get("enabled") is not True or spec.get("schema") != PROTECT_SCHEMA:
        raise LoopError(f"session.json loop.protect_logged is not an enabled {PROTECT_SCHEMA} record")
    if log.get("enabled") is not True:
        raise LoopError("the protected-entry limit needs the round's edit log, and the session has none")
    data = (Path(str(session["paths"]["trial"])) / EDIT_LOG_FILE).read_bytes()
    digest = bytes_sha256(data)
    if not digest == log.get("sha256") == spec.get("edit_log_sha256") == (session.get("trial_files_sha256")
                                                                          or {}).get(EDIT_LOG_FILE):
        raise LoopError("the trial edit_log.json does not hash to the session's edit log and limit records")
    if session.get("real_model_sha256") != spec.get("student_sha256"):
        raise LoopError("the protected-entry limit compares with another student than the session's")
    protected = protected_entries(check_edit_log(data, int(record["round"]), shapes))
    if protected_counts(protected) != spec.get("per_tensor") or sum(protected_counts(protected).values()) != spec.get(
            "n_protected"):
        raise LoopError("the protected entries of the trial edit_log.json are not the ones session.json records")
    return protected


# ---------------------------------------------------------------------------------------------------------------
# One round, prepared before the session writes anything


@dataclass(frozen=True)
class PreparedRound:
    """Everything a loop session needs, read and checked before any file of the session is written."""

    context: dict[str, Any]
    context_path: Path
    student: dict[str, np.ndarray]
    experience: dict[str, np.ndarray]
    experience_record: dict[str, Any]
    prev_checks: list[dict[str, Any]]  # round, instance_id, source path, sha256, n
    kept_rule: dict[str, Any]
    extra_forbidden: tuple[str, ...]
    edit_log: bytes | None = None  # PREREG T16 arm L: the trial's edit_log.json; None in arm N and before T16
    protect: dict[str, Any] | None = None  # T16 follow-up M: session.json loop.protect_logged; None without the limit

    @property
    def round(self) -> int:
        return int(self.context["round"])

    @property
    def n_prev(self) -> int:
        return int(self.experience_record["n_prev"])

    def task_rewrite(self, task: dict[str, Any]) -> dict[str, Any]:
        return loop_task(task, self.kept_rule, self.round - 1, self.n_prev, self.experience_record["prev_per_round"])


def prepare_round(context_path: Path, instance_id: str, target_private: Mapping[str, Any]) -> PreparedRound:
    """Read and check a loop round for the session on ``instance_id`` (raises ``LoopError``; writes nothing)."""
    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import instances_root
    from weightcode.harness.instance import InstanceRefused, read_instance_record

    context = read_loop_context(context_path)
    if str(context["instance_id"]) != instance_id:
        raise LoopError(f"the loop context is for {context['instance_id']}, not {instance_id}")
    target = load_instance(instances_root() / instance_id)
    if target.params_sha256 != context.get("base_params_sha256"):
        raise LoopError("the loop context's base model is not the instance's")
    student = load_student(context)
    prev_checks: list[dict[str, Any]] = []
    extra = [str(Path(context_path).resolve()), str(Path(str(context["student"]["path"])).resolve().parent)]
    for item in context["prev"]:
        prev_id = str(item["instance_id"])
        try:
            record = read_instance_record(prev_id)
        except InstanceRefused as exc:
            raise LoopError(f"round {item['round']}: {exc}") from None
        if record.has_own_experience or record.split_role != "dev":
            raise LoopError(f"round {item['round']}: {prev_id} is not an evidence-v1 dev instance")
        if load_instance(record.instance_dir).params_sha256 != target.params_sha256:
            raise LoopError(f"round {item['round']}: {prev_id} has another base model")
        if str(record.private["model_id"]) != str(target_private["model_id"]):
            raise LoopError(f"round {item['round']}: {prev_id} has another model id")
        path = Path(str(record.private["vcheck_path"]))
        if bytes_sha256(path.read_bytes()) != record.private["vcheck_sha256"]:
            raise LoopError(f"round {item['round']}: the V_check file of {prev_id} does not hash to its record")
        prev_checks.append({"round": int(item["round"]), "instance_id": prev_id, "source": str(path),
                            "sha256": str(record.private["vcheck_sha256"]), "n": int(record.private["n_vcheck"])})
        extra += [prev_id, str(record.instance_dir), str(path), str(Path(str(item["evidence"]["path"])).resolve())]
    for item in context["library"]:
        program = item.get("program")
        if program is not None:
            extra += [str(Path(str(program["path"])).resolve().parent)]
    try:
        experience, experience_record = loop_experience(target, student, load_prev_evidence(context))
    except ValueError as exc:  # StudentEvidenceError and LoopError are ValueErrors
        raise LoopError(f"the round's experience cannot be drawn: {exc}") from None
    rule = context_kept_rule(context, int(target_private["n_vcheck"]),
                             {item["round"]: item["n"] for item in prev_checks})
    edit_log: bytes | None = None
    protect: dict[str, Any] | None = None
    if "edit_log" in context:  # PREREG T16 arm L: checked against the current student's tensors
        edit_log, document = read_context_edit_log(context, {name: value.shape for name, value in student.items()})
        extra.append(str(Path(str(context["edit_log"]["path"])).resolve()))
        if "protect_logged" in context:  # follow-up M: the limit on the entries this log lists
            protect = round_protect(context, document)
    return PreparedRound(context=context, context_path=Path(context_path), student=student, experience=experience,
                         experience_record=experience_record, prev_checks=prev_checks, kept_rule=rule,
                         extra_forbidden=tuple(sorted(set(extra))), edit_log=edit_log, protect=protect)


def session_edit_log(prepared: PreparedRound, trial_files: Mapping[str, str] | None) -> dict[str, Any]:
    """session.json ``loop.edit_log``: which T16 arm the round ran (``enabled``) and, in arm L, the trial file, its
    sha256, the per-round counts, the sentence and the context's record."""
    if prepared.edit_log is None:
        return {"enabled": False}
    spec = prepared.context["edit_log"]
    return {"enabled": True, "file": EDIT_LOG_FILE, "sha256": bytes_sha256(prepared.edit_log),
            "trial_sha256": None if trial_files is None else trial_files.get(EDIT_LOG_FILE),
            "n_entries": dict(spec["n_entries"]), "sentence": EDIT_LOG_SENTENCE, "source": dict(spec)}


def write_round_private(prepared: PreparedRound, private_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Copy the loop student and the earlier conditions' V_check into the session's private directory; return the
    student copy and the session.json ``loop.prev_checks`` records (each copy hash-checked)."""
    from weightcode.common.identity import save_params

    student_path = Path(private_dir) / STUDENT_FILE
    if save_params(student_path, prepared.student) != prepared.context["student"]["sha256"]:
        raise LoopError("the private copy of the loop student does not hash to the student")
    records = []
    for item in prepared.prev_checks:
        target = Path(private_dir) / PREV_CHECK_TEMPLATE.format(k=item["round"])
        data = Path(item["source"]).read_bytes()
        if bytes_sha256(data) != item["sha256"]:
            raise LoopError(f"round {item['round']}: the V_check file changed")
        target.write_bytes(data)
        records.append({"round": item["round"], "instance_id": item["instance_id"], "path": str(target),
                        "sha256": item["sha256"], "n": item["n"]})
    return student_path, records


__all__ = ["EDIT_LOG_FILE", "EDIT_LOG_SCHEMA", "EDIT_LOG_SENTENCE", "EVIDENCE_KEYS", "KEPT_RULES", "KEPT_RULE_LOOP", "KEPT_RULE_LOOP_V2", "LOOP_CLEAN_BUDGET_FRAC",
           "LOOP_CLEAN_FRAC", "LOOP_CONTEXT_SCHEMA", "LOOP_LIBRARY_SCHEMA", "LOOP_PREV_FRAC", "LOOP_READOUT",
           "LOOP_RECORD_SCHEMA", "LOOP_STATUS_DESCRIPTION", "ROUND1_SESSION_CHECK", "LoopError", "MultiPartOracle",
           "PARENT_LOOP", "PREV_CHECK_TEMPLATE", "PREV_KEYS", "PROTECT_COMPARE", "PROTECT_SCHEMA", "PROTECT_SENTENCE",
           "STUDENT_FILE", "check_context_protect", "check_edit_log", "context_clean_used",
           "context_kept_rule", "edit_log_bytes", "edit_log_counts", "edit_log_document", "edit_log_round",
           "edit_log_section", "eligible", "kept_prev_rows", "kept_rule_for", "library_document", "load_npz_checked",
           "load_prev_evidence", "load_student", "loop_check_description", "loop_declare_description",
           "loop_experience", "loop_files_section", "loop_final_selection", "loop_kept_rule", "loop_kept_rule_v2",
           "loop_kept_section", "loop_statement", "loop_task", "part_fields", "prev_part", "prev_score_keys",
           "protect_messages", "protect_section", "protected_changes", "protected_counts", "protected_entries",
           "read_context_edit_log", "read_loop_context", "round_protect", "session_edit_log", "session_loop_rule",
           "session_protected_entries", "shown_check_fields", "write_edit_log", "write_loop_library"]
