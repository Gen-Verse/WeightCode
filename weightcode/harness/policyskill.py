"""The harness side of rule policyskill-v1 (PREREG E14): the one access point of the editor harness to
``weightcode.evalkit.e14``, which owns the rule (the instances, their evidence, the V_check seed file, the rollout oracle and
the V_select readout).

A policyskill instance asks an update to teach a trained policy a new skill: a film_policy student trained by imitation
on the old mission ("go to {the|a} {color} {type}") is to perform the new one ("pick up {the|a} {color} {type}") in the
same rooms while it keeps the old one. It runs in the LEARN arm only and follows the seqskill-v1 path
(``weightcode.harness.seqskill``) step by step:

- the instance id is ``p`` + 8 hex and the instance lives under ``$WEIGHTCODE_PRIVATE/e14_instances``; the runner refuses it
  unless ``check_e14_current`` accepts it (active in its index, task.json and files as the current code writes them);
- the student is a policy: an input is one int64 observation row [197] (the fully observed grid, the agent's position
  and direction, and the instruction as 5 word ids), the output 7 action logits, and the policy takes the argmax;
- the experience is the instance's evidence.npz itself (``EXPERIENCE_KEYS``): every step of the 8 new-mission
  demonstrations with the bot's action, the demonstration and step numbers, and 64 old-mission replay steps, each with
  the student's float32 logits and an opaque id. ``experience`` re-checks every logit row against forward.py (argmax
  equal, values within ``LOGIT_ATOL``), the actions (one pickup closing each demonstration, no pickup among the replay
  steps), the instruction words (the reserved words in every new row and in no replay row) and the content hash;
- the check is greedy forward.py rollouts on the V_check episodes (``check_oracle``: evalkit's ``RolloutJointOracle``):
  S_new = fixed - broken over the new-mission episodes, S_old the same over the old-mission episodes, child against
  base, S = S_new + S_old. The session ledger reads the seeds from the session's V_check file (x = seeds, mission = 1
  new and 0 old), checked by file hash, content hash and per-part seed hashes against the instance record
  (``check_parts``), and the base's per-episode success against the record's hashes, so the MCP server needs no store
  variable and no seed range from the code;
- the readout is ``e14_readout`` on V_select (``readout``): new and old success, child against base, with lengths,
  pickups and the breakdowns; ``parts`` gives each in a dev readout's fields.

The MCP timeouts are E10's floors (``mcp_timeouts``: 1,200 s per tool call, 900 s to connect, refused below). The
server builds the oracle at start, which runs the base's V_check rollouts, and a restarted server replays every logged
check; a check is the program run (at most 300 s) plus two rollout sets of about 17 s at 4 threads (31 s worst case,
every episode at 64 steps; PREREG E14 as built). The MCP server's interpreter needs minigrid (PREREG E14: a worktree's
own venv), the sandbox numpy only: the editor reads rows, and the checks run the rollouts.

There is no decoy, no corruption condition and no clean companion. ``weightcode.evalkit.e14`` is imported inside the
functions that need it, so no other session loads it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import Params, params_sha256
from weightcode.common.paths import private_dir

POLICYSKILL_RULE = "policyskill-v1"  # weightcode.evalkit.e14.POLICYSKILL_RULE; test_harness_policyskill checks they agree
INSTANCE_ID_RE = re.compile(r"^p[0-9a-f]{8}$")  # weightcode.evalkit.e14.INSTANCE_ID_RE, checked the same way
POLICYSKILL_ROOT_NAME = "e14_instances"  # weightcode.evalkit.e14.ROOT_NAME
EXPERIENCE_SOURCE = "instance_policyskill_v1"
EXPERIENCE_KEYS = ("x_new", "y_new", "logits_new", "episode_new", "step_new", "split_ids_new",
                   "x_old", "y_old", "logits_old", "split_ids_old")  # weightcode.evalkit.e14.EVIDENCE_KEYS
PART_KEYS = {"new": ("x_new", "y_new", "logits_new", "split_ids_new"), "old": ("x_old", "y_old", "logits_old",
                                                                           "split_ids_old")}
LOGIT_ATOL = 1e-4
"""Largest absolute difference between a recorded logit and forward.py's, which may differ in the last bits with the
BLAS thread count; the argmax must be equal."""
VCHECK_KEYS = ("mission", "x", "y")
CHECK_SET = "old-mission"  # the episodes behind score_old, as the retention kept rule names them
CHECK_UNIT = "episodes"
MCP_TOOL_TIMEOUT_MS = 1_200_000
MCP_CONNECT_TIMEOUT_MS = 900_000
# private.json fields the harness reads from a policyskill instance; an instance without one of them is refused.
REQUIRED_PRIVATE_FIELDS = (
    "instance_id",
    "split_role",
    "dataset",
    "arch",
    "seed",
    "model_id",
    "evidence_rule",
    "vcheck_path",
    "vcheck_sha256",
    "vcheck_content_sha256",
    "vcheck_seeds_sha256",
    "base_vcheck_success_sha256",
    "n_vcheck",
    "n_vcheck_new",
    "n_vcheck_old",
    "n_base_errors_vcheck_new",
    "n_base_errors_vcheck_old",
    "n_new_rows",
    "n_new_demonstrations",
    "n_replay",
    "pickup_action",
    "reserved_word_ids",
    "model_sha256",
    "forward_sha256",
    "task_sha256",
    "evidence_content_sha256",
    "model_spec_sha256",
)
# private.json fields that hold host paths; every one present is a forbidden string of the session.
PRIVATE_PATH_FIELDS = ("vcheck_path", "vevidence_path", "replay_path", "model_source_path")


class PolicyskillHarnessError(RuntimeError):
    """A policyskill instance or session record is inconsistent, so no LEARN view, check or readout is built."""


def is_policyskill_id(instance_id: object) -> bool:
    """True for an id of the E14 store's form (``p`` and 8 hex digits)."""
    return isinstance(instance_id, str) and INSTANCE_ID_RE.fullmatch(instance_id) is not None


def is_policyskill(private: Mapping[str, Any]) -> bool:
    """True when private.json says the instance follows rule policyskill-v1."""
    return private.get("evidence_rule") == POLICYSKILL_RULE


def instance_dir(instance_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/e14_instances/<instance_id>``, the path of ``weightcode.evalkit.e14.e14_instance_dir`` computed without
    creating the store root, so that resolving an id writes nothing; refuses an id of another form."""
    if not is_policyskill_id(instance_id):
        raise PolicyskillHarnessError(f"{instance_id!r} is not a policyskill instance id ('p' and 8 hex digits)")
    return private_dir() / POLICYSKILL_ROOT_NAME / instance_id


def check_current(path: str | Path) -> dict[str, Any]:
    """``weightcode.evalkit.e14.check_e14_current``: the instance's task, or ``StaleE14Instance``."""
    from weightcode.evalkit.e14 import check_e14_current

    return check_e14_current(path)


def stale_error() -> type[Exception]:
    """The evalkit exception for an instance that is not active or not current (and every other E14 refusal)."""
    from weightcode.evalkit.e14 import E14InstanceError

    return E14InstanceError


def active_index_row(instance_id: str) -> dict[str, Any] | None:
    """The instance's row among the active rows of the E14 index, or None."""
    from weightcode.evalkit.e14 import read_e14_instance_index

    rows = read_e14_instance_index()["instances"]
    found = [dict(row) for row in rows if isinstance(row, dict) and row.get("instance_id") == instance_id]
    return found[0] if found else None


def leak_words() -> list[str]:
    """``weightcode.evalkit.e14.e14_leak_tokens``: the dataset tokens, the three level names, pickupinstr, babyai and minigrid
    (PREREG E14 as built, decision b); the instance build scans for the same."""
    from weightcode.evalkit.e14 import e14_leak_tokens

    return e14_leak_tokens()


def mcp_timeouts(private: Mapping[str, Any], tool_ms: int | None, connect_ms: int | None,
                 defaults: tuple[int, int]) -> tuple[int, int]:
    """The (tool, connect) MCP timeouts of a session: the given values, else E10's floors (``MCP_TOOL_TIMEOUT_MS``,
    ``MCP_CONNECT_TIMEOUT_MS``) on a policyskill instance, whose values below them are refused (``ValueError``), and
    ``defaults`` otherwise (``weightcode.harness.e10.floored_mcp_timeouts``)."""
    from weightcode.harness import e10

    if (e10.MCP_TOOL_TIMEOUT_MS, e10.MCP_CONNECT_TIMEOUT_MS) != (MCP_TOOL_TIMEOUT_MS, MCP_CONNECT_TIMEOUT_MS):
        raise PolicyskillHarnessError("the policyskill MCP floors are E10's")  # pragma: no cover - fixed constants
    return e10.floored_mcp_timeouts(is_policyskill(private), "E14 policyskill-v1",
                                    "the server's base rollouts and the replay of logged checks", tool_ms, connect_ms,
                                    defaults)


def _words(x: np.ndarray) -> np.ndarray:
    from weightcode.data import e14 as D

    return x[:, D.OBS_LEN:D.OBS_LEN + D.INSTR_LEN]


def _check_rows(x: np.ndarray, y: np.ndarray, logits: np.ndarray, ids: np.ndarray, what: str) -> None:
    from weightcode.data import e14 as D

    n = len(y)
    if x.dtype != np.int64 or x.ndim != 2 or x.shape[1] != D.ROW_LEN or n == 0:
        raise PolicyskillHarnessError(f"{what}: expected int64 observation rows [N, {D.ROW_LEN}], got {x.dtype}"
                                      f"{x.shape}")
    if y.shape != (n,) or len(x) != n or logits.shape != (n, len(D.ACTIONS)) or len(ids) != n:
        raise PolicyskillHarnessError(f"{what}: unequal lengths or shapes ({x.shape}, {y.shape}, {logits.shape}, "
                                      f"{ids.shape})")
    if logits.dtype != np.float32:
        raise PolicyskillHarnessError(f"{what}: logits are {logits.dtype}, not float32")


def experience(instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The LEARN experience of a policyskill instance and its host-side record.

    The arrays are the instance's evidence under ``EXPERIENCE_KEYS``. Every logit row must be forward.py's on the
    student (argmax equal, within ``LOGIT_ATOL``); the new part holds every step of ``n_new_demonstrations``
    demonstrations numbered 0 to k - 1 in order, steps 0, 1, ... within each, each closed by the one pickup of its
    demonstration, and every new row names the reserved words; no replay row names one or is a pickup; the ids are
    unique; and the content hash is private.json's."""
    from weightcode.data import e14 as D
    from weightcode.evalkit.e14 import evidence_content_sha256, policy_logits

    private = instance.private
    if not is_policyskill(private):
        raise PolicyskillHarnessError(f"{instance.instance_id} follows {private.get('evidence_rule')!r}, not "
                                      f"{POLICYSKILL_RULE}")
    evidence = instance.evidence
    if sorted(evidence) != sorted(EXPERIENCE_KEYS):
        raise PolicyskillHarnessError(f"{instance.instance_id}: evidence keys {sorted(evidence)} are not the LEARN "
                                      f"keys {sorted(EXPERIENCE_KEYS)}")
    arrays = {key: np.ascontiguousarray(evidence[key]) for key in EXPERIENCE_KEYS}
    if evidence_content_sha256(arrays) != private["evidence_content_sha256"]:
        raise PolicyskillHarnessError(f"{instance.instance_id}: the experience differs from the recorded evidence")
    pickup, reserved = int(private["pickup_action"]), [int(i) for i in private["reserved_word_ids"]]
    if pickup != D.PICKUP or reserved != [D.WORD_ID[w] for w in D.RESERVED_WORDS]:
        raise PolicyskillHarnessError(f"{instance.instance_id}: private.json pickup_action {pickup} and reserved words "
                                      f"{reserved} are not the code's")
    n_right, max_diff = {}, 0.0
    for part, keys in PART_KEYS.items():
        x, y, logits, ids = (arrays[key] for key in keys)
        _check_rows(x, y, logits, ids, f"{instance.instance_id} {part} part")
        recomputed = np.asarray(policy_logits(instance.forward, instance.params, x), dtype=np.float32)
        if not np.array_equal(recomputed.argmax(axis=1), logits.argmax(axis=1)):
            raise PolicyskillHarnessError(f"{instance.instance_id}: logits_{part} disagree with the student's action at "
                                          f"{int(np.count_nonzero(recomputed.argmax(1) != logits.argmax(1)))} rows")
        diff = float(np.max(np.abs(recomputed - logits)))
        if not diff <= LOGIT_ATOL:
            raise PolicyskillHarnessError(f"{instance.instance_id}: logits_{part} differ from forward.py by {diff:.3g} "
                                          f"(> {LOGIT_ATOL:g})")
        max_diff = max(max_diff, diff)
        n_right[part] = int(np.count_nonzero(logits.argmax(axis=1) == y))
    words_new, words_old = _words(arrays["x_new"]), _words(arrays["x_old"])
    if not all(np.isin(reserved, row).all() for row in words_new) or np.isin(words_old, reserved).any():
        raise PolicyskillHarnessError(f"{instance.instance_id}: a new row lacks the reserved words or a replay row "
                                      "holds one")
    if not np.isin(arrays["y_old"], D.OLD_ACTIONS).all():
        raise PolicyskillHarnessError(f"{instance.instance_id}: a replay step's action is not an old-mission action")
    k = int(private["n_new_demonstrations"])
    episode, step, y_new = arrays["episode_new"], arrays["step_new"], arrays["y_new"]
    order = [int(e) for e in dict.fromkeys(episode.tolist())]
    if order != list(range(k)) or not np.all(np.diff(episode) >= 0):
        raise PolicyskillHarnessError(f"{instance.instance_id}: the new part is not demonstrations 0 to {k - 1} in "
                                      "order")
    for number in range(k):
        rows = np.flatnonzero(episode == number)
        if not np.array_equal(step[rows], np.arange(len(rows))):
            raise PolicyskillHarnessError(f"{instance.instance_id}: demonstration {number}'s steps are not 0, 1, ...")
        if int(np.count_nonzero(y_new[rows] == pickup)) != 1 or int(y_new[rows[-1]]) != pickup:
            raise PolicyskillHarnessError(f"{instance.instance_id}: demonstration {number} does not end in its one "
                                          "pickup")
    ids = [*arrays["split_ids_new"].tolist(), *arrays["split_ids_old"].tolist()]
    if len(set(ids)) != len(ids):
        raise PolicyskillHarnessError(f"{instance.instance_id}: the split ids are not unique")
    counts = (len(y_new), len(arrays["y_old"]))
    if counts != (int(private["n_new_rows"]), int(private["n_replay"])):
        raise PolicyskillHarnessError(f"{instance.instance_id}: {counts} new and replay rows, private.json says "
                                      f"{(private['n_new_rows'], private['n_replay'])}")
    record = {
        "file": "experience.npz",
        "source": EXPERIENCE_SOURCE,
        "rule": POLICYSKILL_RULE,
        "content_sha256": params_sha256(arrays),
        "evidence_content_sha256": private["evidence_content_sha256"],
        "student_params_sha256": instance.params_sha256,
        "n_new": counts[0],
        "n_new_demonstrations": k,
        "n_new_right": n_right["new"],
        "n_old": counts[1],
        "n_old_fail": counts[1] - n_right["old"],
        "pickup_action": pickup,
        "reserved_word_ids": reserved,
        "logits_max_abs_diff": max_diff,
    }
    return arrays, record


def session_record(private: Mapping[str, Any]) -> dict[str, Any]:
    """The ``policyskill`` section of session.json: what the ledger needs to read the V_check seeds and the base's
    episodes, and the base error counts."""
    missing = [name for name in REQUIRED_PRIVATE_FIELDS if name not in private]
    if missing:
        raise PolicyskillHarnessError(f"private.json lacks {missing}")
    return {
        "rule": POLICYSKILL_RULE,
        "check_unit": CHECK_UNIT,
        "n_vcheck_new": int(private["n_vcheck_new"]),
        "n_vcheck_old": int(private["n_vcheck_old"]),
        "n_base_errors_vcheck_new": int(private["n_base_errors_vcheck_new"]),
        "n_base_errors_vcheck_old": int(private["n_base_errors_vcheck_old"]),
        "vcheck_content_sha256": str(private["vcheck_content_sha256"]),
        "vcheck_seeds_sha256": {part: str(private["vcheck_seeds_sha256"][part]) for part in ("new", "old")},
        "base_vcheck_success_sha256": {part: str(private["base_vcheck_success_sha256"][part])
                                       for part in ("new", "old")},
    }


def check_parts(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The new and old V_check seeds of the session's V_check file (keys ``VCHECK_KEYS``: x = seeds, y = 1, mission =
    1 new and 0 old), checked against the session's ``policyskill`` record: the content hash, the part sizes and each
    part's seed hash."""
    from weightcode.evalkit.e14 import seeds_sha256

    if record.get("rule") != POLICYSKILL_RULE:
        raise PolicyskillHarnessError(f"the session's policyskill record names rule {record.get('rule')!r}")
    if sorted(arrays) != sorted(VCHECK_KEYS):
        raise PolicyskillHarnessError(f"the V_check file has keys {sorted(arrays)}, not {list(VCHECK_KEYS)}")
    if params_sha256(dict(arrays)) != record["vcheck_content_sha256"]:
        raise PolicyskillHarnessError("the V_check file's content differs from the instance record")
    x, y, mission = (np.asarray(arrays[key]) for key in ("x", "y", "mission"))
    if x.dtype != np.int64 or not (x.shape == y.shape == mission.shape) or not np.all(y == 1):
        raise PolicyskillHarnessError("the V_check file is not int64 seeds with y = 1 and a mission per seed")
    parts = {"new": x[mission == 1], "old": x[mission == 0]}
    sizes = (len(parts["new"]), len(parts["old"]))
    if sizes != (int(record["n_vcheck_new"]), int(record["n_vcheck_old"])) or sum(sizes) != len(x):
        raise PolicyskillHarnessError(f"V_check cuts into {sizes} new and old episodes; the instance records "
                                      f"{(record['n_vcheck_new'], record['n_vcheck_old'])}")
    for part, seeds in parts.items():
        if seeds_sha256(seeds) != record["vcheck_seeds_sha256"][part]:
            raise PolicyskillHarnessError(f"the {part} V_check seeds differ from the instance record")
    return parts


def check_oracle(parts: Mapping[str, np.ndarray], forward: Any, base: Params, budget: int,
                 record: Mapping[str, Any]) -> Any:
    """``weightcode.evalkit.e14.RolloutJointOracle`` on the V_check seeds of ``check_parts``, its base's per-episode success
    checked against the record's hashes (the ``JointPatchCheckOracle`` fields, counted in episodes)."""
    from weightcode.evalkit.e14 import E14InstanceError, RolloutJointOracle

    try:
        return RolloutJointOracle(parts["new"], parts["old"], forward, base, budget,
                                  dict(record["base_vcheck_success_sha256"]))
    except E14InstanceError as exc:
        raise PolicyskillHarnessError(str(exc)) from None


def _part(full: Mapping[str, Any]) -> dict[str, Any]:
    return {"n": full["n"], "base_acc": full["success_base"], "child_acc": full["success_child"],
            "delta_acc": full["delta"], "fixed": full["fixed"], "broken": full["broken"]}


def readout(instance: Instance, child: Params) -> dict[str, Any]:
    """``e14_readout`` of ``child`` on V_select (``readouts_policyskill``: new and old success, child against base,
    with lengths, pickups, the breakdowns, gain, clean and joint) and, per part (new, old), the change in the fields
    of a dev readout: n, base_acc, child_acc, delta_acc, fixed and broken, where accuracy is success per episode."""
    from weightcode.evalkit.e14 import e14_readout

    full = e14_readout(instance, dict(child))
    return {"readouts_policyskill": full, "parts": {part: _part(full[part]) for part in ("new", "old")}}


def private_paths(private: Mapping[str, Any]) -> list[str]:
    """The host paths recorded in a policyskill private.json (``PRIVATE_PATH_FIELDS``)."""
    return [str(private[name]) for name in PRIVATE_PATH_FIELDS if private.get(name)]


__all__ = [
    "CHECK_SET", "CHECK_UNIT", "EXPERIENCE_KEYS", "EXPERIENCE_SOURCE", "INSTANCE_ID_RE", "LOGIT_ATOL",
    "MCP_CONNECT_TIMEOUT_MS", "MCP_TOOL_TIMEOUT_MS", "PART_KEYS", "POLICYSKILL_ROOT_NAME", "POLICYSKILL_RULE",
    "PRIVATE_PATH_FIELDS", "PolicyskillHarnessError", "REQUIRED_PRIVATE_FIELDS", "VCHECK_KEYS", "active_index_row",
    "check_current", "check_oracle", "check_parts", "experience", "instance_dir", "is_policyskill", "is_policyskill_id",
    "leak_words", "mcp_timeouts", "private_paths", "readout", "session_record", "stale_error",
]
