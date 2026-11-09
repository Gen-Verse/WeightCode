"""The counted patch-check state of one session, rebuilt from ``logs/checks.jsonl`` (HARNESS_SPEC sections 4 and 5).

``CheckLedger`` builds the oracle as ``weightcode.baselines.instance_io.make_oracle`` does (V_check from
``private.json["vcheck_path"]``, the forward the editor sees, the real model as base, ``target=None`` for evidence-v1),
after checking the V_check file's bytes against ``private.json["vcheck_sha256"]``. It then replays every logged query:
the private snapshot of the checked patch is applied again under the arm rules and queried again, and every logged
number must come out the same. The tool server uses the ledger so that a restarted server never hands out a fresh
budget; finalisation uses it to rebuild the oracle state for the final-patch rule.

A query that passed validation but produced non-finite logits on V_check is a "charged failure": the oracle does not
count it, but its outcome depends on the check set, so the harness charges one query for it.

LEARN (a program arm): the oracle is evalkit's ``JointPatchCheckOracle`` (``weightcode.harness.clean.joint_oracle``) on V_check
and its clean companion, the session's private copy ``session.json["clean_check"]`` (bytes hashed; ``y`` and ``idx``
equal to V_check's in order). Its ``score`` is S_joint = score_new + score_old, so the final-patch rule (argmax score,
ties to fewer changed parameters, then the earlier query) is the joint rule; the replay also compares the part scores.
On a fewshot-v1 instance (PREREG E2, ``session.json["fewshot"]``) the two parts are V_check cut by class
(``weightcode.harness.fewshot.check_parts``): the rows of the new classes and the rows of the old classes. On a seqskill-v1
instance (PREREG E5, ``session.json["seqskill"]``) they are V_check cut by its ``part`` array
(``weightcode.harness.seqskill.check_parts``): the rows of the new operator alone and the old-operator rows. On a graphnew-v1
instance (PREREG E7, ``session.json["graphnew"]``) they are V_check cut by class (``weightcode.harness.graphnew.check_parts``),
as node ids of the graph that the private forward.py copy reads beside itself; that private graph.npz must hash to the
recorded ``graph_file_sha256``, and each part is one full-graph forward (``weightcode.harness.graphnew.check_oracle``). On an
fscil-v1 instance (PREREG E9, ``session.json["fscil"]``) the new part is the V_check rows of session t's classes and the
old part old_check (the base classes, the file the ``fscil`` section names, checked by bytes and content) followed by
the V_check rows of the classes of sessions 2 to t - 1 (``weightcode.harness.fscil.load_check_parts``), scored under the
session mask at batch 256 (``weightcode.harness.fscil.check_oracle``). Under the E9 v2 kept rule fscil-retain-v2
(``session.json["kept_rule"]``, ``weightcode.harness.fscil.session_kept_v2``) the same check also carries fscil-cb-v1 and
floor_c and eligible (``weightcode.harness.fscil.cb_check_oracle``), and the replay compares those too. A loop round
(PREREG E4-loop, ``session.json["loop"]``) uses ``weightcode.harness.loop.MultiPartOracle``: V_check, its clean companion and
the private copy of each earlier round's condition V_check, all against the loop student, which is the session's real
model; its replay compares every part's score, fixed and broken. On a policyskill-v1 instance (PREREG E14,
``session.json["policyskill"]``) V_check holds environment seeds, cut by mission into the new and the trained
mission (``weightcode.harness.policyskill.check_parts``, checked by content and by each part's seed hash), and each part
is scored by greedy forward.py rollouts, one episode per seed, right when it succeeds
(``weightcode.harness.policyskill.check_oracle``); the base's per-episode success must match the instance record.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.evalkit.patch_check import PatchCheckOracle, load_forward
from weightcode.harness import clean, fewshot, fscil, graphnew, loop, policyskill, seqskill
from weightcode.harness.applying import ArmApplier, SessionConfigError
from weightcode.harness.arms import arm_spec
from weightcode.harness.jsonio import bytes_sha256, file_sha256, read_jsonl
from weightcode.harness.session import SessionPaths
from weightcode.patch import PatchResult, load_patch

CHECK_RECORD_SCHEMA = "weightcode.harness.check.v0"
REPLAY_FIELDS = ("query_index", "patch_id", "child_sha256", "score", "fixed", "broken", "n_changed")
JOINT_REPLAY_FIELDS = ("score_new", "score_old", "fixed_new", "broken_new", "fixed_old", "broken_old")


def load_checked_npz(path: Path, expected_sha256: str, what: str) -> dict[str, np.ndarray]:
    """Every array of an .npz read once from bytes that must hash to ``expected_sha256``."""
    data = path.read_bytes()
    digest = bytes_sha256(data)
    if digest != expected_sha256:
        raise SessionConfigError(f"{what} file hash {digest} != recorded {expected_sha256}")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_vcheck(path: Path, expected_sha256: str, expected_n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """V_check inputs and labels, read once; the bytes must hash to ``private.json["vcheck_sha256"]``."""
    data = path.read_bytes()
    digest = bytes_sha256(data)
    if digest != expected_sha256:
        raise SessionConfigError(f"V_check file hash {digest} != private.json vcheck_sha256 {expected_sha256}")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        x, y = archive["x"], archive["y"]
    if expected_n is not None and len(y) != expected_n:
        raise SessionConfigError(f"V_check has {len(y)} examples, private.json n_vcheck says {expected_n}")
    return x, y


class CheckLedger:
    """Oracle, arm applier and the replayed check log of one session."""

    def __init__(self, session: Mapping[str, Any], paths: SessionPaths) -> None:
        self.paths = paths
        self.applier = ArmApplier(session)
        self.forward, forward_sha = load_forward(Path(str(session["forward_path"])))
        if forward_sha != session["forward_sha256"]:
            raise SessionConfigError("the private forward.py copy does not hash to session.json forward_sha256")
        x, y = load_vcheck(Path(str(session["vcheck_path"])), str(session["vcheck_sha256"]), int(session["n_vcheck"]))
        self.budget = int(session["budget"]["patch_checks"])
        self.joint = arm_spec(str(session["arm"])).program
        self.loop = "loop" in session
        self.n_prev = len(session["loop"]["prev_checks"]) if self.loop else 0
        rules = [name for name in ("fewshot", "seqskill", "graphnew", "fscil", "policyskill") if name in session]
        if len(rules) > 1:
            raise SessionConfigError(f"a session follows one of fewshot-v1, seqskill-v1, graphnew-v1, fscil-v1 and "
                                     f"policyskill-v1, not {rules}")
        if rules and not self.joint:
            raise SessionConfigError(f"a {rules[0]}-v1 session is a program arm")
        if self.loop and (not self.joint or rules):
            raise SessionConfigError("a loop session is a program arm on an evidence-v1 instance")
        try:  # PREREG E9 v2: the fscil-retain-v2 kept rule (checked against its definition), or None
            self.fscil_v2 = fscil.session_kept_v2(session)
        except fscil.FscilHarnessError as exc:
            raise SessionConfigError(str(exc)) from None
        self.oracle: Any
        if not self.joint:
            self.oracle = PatchCheckOracle(
                x=x, y=y, forward=self.forward, base_params=self.applier.real, target=None, budget=self.budget
            )
        elif "fewshot" in session:  # PREREG E2: V_check cut by class, S_new over the new classes, S_old over the rest
            try:
                parts = fewshot.check_parts(x, y, session["fewshot"])
            except (fewshot.FewshotHarnessError, fewshot.stale_error()) as exc:
                raise SessionConfigError(f"the fewshot V_check cut fails: {exc}") from None
            self.oracle = clean.joint_oracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"],
                                             self.forward, self.applier.real, self.budget)
        elif "graphnew" in session:  # PREREG E7: V_check cut by class, as node ids; one forward per part
            record = session["graphnew"]
            graph_copy = Path(str(session["forward_path"])).parent / graphnew.GRAPH_FILE
            if not graph_copy.is_file() or file_sha256(graph_copy) != record["graph_file_sha256"]:
                raise SessionConfigError(f"the private {graphnew.GRAPH_FILE} beside forward.py does not hash to "
                                         "session.json graphnew graph_file_sha256")
            arrays = load_checked_npz(Path(str(session["vcheck_path"])), str(session["vcheck_sha256"]), "V_check")
            try:
                parts = graphnew.check_parts(arrays, record)
            except (graphnew.GraphnewHarnessError, graphnew.stale_error()) as exc:
                raise SessionConfigError(f"the graphnew V_check cut fails: {exc}") from None
            self.oracle = graphnew.check_oracle(parts, self.forward, self.applier.real, self.budget)
        elif "fscil" in session:  # PREREG E9: session t's classes against old_check and the earlier sessions' rows
            record = session["fscil"]
            try:
                parts = fscil.load_check_parts(record, Path(str(session["vcheck_path"])), str(session["vcheck_sha256"]))
            except (fscil.FscilHarnessError, fscil.stale_error(), OSError) as exc:
                raise SessionConfigError(f"the fscil check cut fails: {exc}") from None
            if self.fscil_v2 is None:
                self.oracle = fscil.check_oracle(parts, self.forward, self.applier.real, self.budget,
                                                 int(record["session"]))
            else:  # E9 v2: the same parts and row fields, plus fscil-cb-v1 with floor_c and eligible
                try:
                    self.oracle = fscil.cb_check_oracle(parts, self.forward, self.applier.real, self.budget,
                                                        int(record["session"]), self.fscil_v2)
                except fscil.FscilHarnessError as exc:
                    raise SessionConfigError(f"the fscil v2 check fails: {exc}") from None
        elif "policyskill" in session:  # PREREG E14: V_check seeds cut by mission, each part run as greedy episodes
            arrays = load_checked_npz(Path(str(session["vcheck_path"])), str(session["vcheck_sha256"]), "V_check")
            try:
                parts = policyskill.check_parts(arrays, session["policyskill"])
                self.oracle = policyskill.check_oracle(parts, self.forward, self.applier.real, self.budget,
                                                       session["policyskill"])
            except (policyskill.PolicyskillHarnessError, policyskill.stale_error()) as exc:
                raise SessionConfigError(f"the policyskill V_check fails: {exc}") from None
        elif "seqskill" in session:  # PREREG E5: V_check cut by part, S_new over part 1, S_old over part 4
            arrays = load_checked_npz(Path(str(session["vcheck_path"])), str(session["vcheck_sha256"]), "V_check")
            try:
                parts = seqskill.check_parts(arrays, session["seqskill"])
            except (seqskill.SeqskillHarnessError, seqskill.stale_error()) as exc:
                raise SessionConfigError(f"the seqskill V_check cut fails: {exc}") from None
            self.oracle = clean.joint_oracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"],
                                             self.forward, self.applier.real, self.budget)
        else:
            record = session["clean_check"]
            new = load_checked_npz(Path(str(session["vcheck_path"])), str(session["vcheck_sha256"]), "V_check")
            old = load_checked_npz(Path(str(record["path"])), str(record["sha256"]), "clean V_check")
            if not (np.array_equal(old["y"], y) and "idx" in new and np.array_equal(old["idx"], new["idx"])):
                raise SessionConfigError("the clean V_check rows are not V_check's rows (y and idx in order)")
            if self.loop:  # PREREG E4-loop: the earlier rounds' conditions are further parts of the same check
                parts = {"new": (x, y), "old": (old["x"], old["y"])}
                for item in session["loop"]["prev_checks"]:
                    arrays = load_checked_npz(Path(str(item["path"])), str(item["sha256"]),
                                              f"round {item['round']} V_check")
                    if len(arrays["y"]) != int(item["n"]):
                        raise SessionConfigError(f"round {item['round']} V_check has {len(arrays['y'])} rows, "
                                                 f"session.json says {item['n']}")
                    parts[loop.prev_part(int(item["round"]))] = (arrays["x"], arrays["y"])
                self.oracle = loop.MultiPartOracle(parts, self.forward, self.applier.real, self.budget)
            else:
                self.oracle = clean.joint_oracle(x, y, old["x"], old["y"], self.forward, self.applier.real,
                                                 self.budget)
        self.replay_fields = (REPLAY_FIELDS + (JOINT_REPLAY_FIELDS if self.joint else ())
                              + (loop.part_fields(self.n_prev) if self.loop else ())
                              + (fscil.V2_REPLAY_FIELDS if self.fscil_v2 is not None else ()))
        if self.oracle.base_sha256 != self.applier.real_sha256:
            raise SessionConfigError("the oracle base is not the real model")
        self.entries: list[dict[str, Any]] = []  # checks.jsonl records, consumed queries and charged failures
        self.charged: list[dict[str, Any]] = []
        self._replay()

    @property
    def queries_remaining(self) -> int:
        """Queries left after the oracle's consumed queries and the charged failures."""
        return max(0, self.oracle.queries_remaining - len(self.charged))

    @property
    def queries_used(self) -> int:
        return self.budget - self.queries_remaining

    def checked(self) -> list[dict[str, Any]]:
        """Records of the consumed oracle queries, in query order."""
        return [entry for entry in self.entries if not entry.get("charged_failure")]

    def entry_for(self, patch_id: str) -> dict[str, Any]:
        """The checks.jsonl record of one consumed query."""
        for entry in self.checked():
            if entry["patch_id"] == patch_id:
                return entry
        raise KeyError(f"no consumed query for patch {patch_id!r}")

    def reapply(self, entry: Mapping[str, Any]) -> PatchResult:
        """Apply the private snapshot of a logged check again; the child must hash as logged."""
        attempt_dir = self.paths.attempts / str(entry["attempt"])
        if "program_file" in entry and file_sha256(attempt_dir / str(entry["program_file"])) != entry["program_sha256"]:
            raise SessionConfigError(f"the program snapshot of {entry['patch_id']!r} changed after its check")
        result = self.applier.apply(load_patch(attempt_dir / str(entry["patch_file"])), attempt_dir)
        if result.child_sha256 != entry["child_sha256"]:
            raise SessionConfigError(f"re-applying the snapshot of {entry['patch_id']!r} gives another child")
        return result

    def record(self, entry: dict[str, Any]) -> None:
        """Add a record that the tool server has just appended to checks.jsonl."""
        self.entries.append(entry)
        if entry.get("charged_failure"):
            self.charged.append(entry)

    def _replay(self) -> None:
        for number, entry in enumerate(read_jsonl(self.paths.checks_log), start=1):
            if not isinstance(entry, dict) or entry.get("schema") != CHECK_RECORD_SCHEMA:
                raise SessionConfigError(f"checks.jsonl line {number} is not a {CHECK_RECORD_SCHEMA} record")
            if entry.get("charged_failure"):
                self.record(entry)
                continue
            result = self.reapply(entry)
            replay = self.oracle.query(result.child, str(entry["patch_id"]))
            for key in self.replay_fields:
                if replay[key] != entry[key]:
                    raise SessionConfigError(f"replay of checks.jsonl line {number} differs in {key}: "
                                             f"{replay[key]!r} != {entry[key]!r}")
            self.record(entry)


__all__ = ["CHECK_RECORD_SCHEMA", "CheckLedger", "load_vcheck"]
