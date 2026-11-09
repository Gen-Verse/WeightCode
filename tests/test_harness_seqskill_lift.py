"""The rel-Frob limit lifted for a seqskill-v1 LEARN session (PREREG E8 amendment of 2026-11-04 15:37): max_rel_frob is
null in the trial task.json, the task prompt, session.json and every check, declaration, final and readout replay; the
20% changed-entry cap and every other limit stay; anything but a seqskill-v1 instance in the LEARN arm is refused.

The seqskill store is the evalkit toy seqops store (``test_evalkit_seqskill.store``), the evidence-v1 one the harness
toy store (``test_harness_fixtures.harness_store``). No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from test_evalkit_seqskill import CONDITION, store  # noqa: F401
from test_harness_fixtures import HarnessStore, harness_store, needs_sandbox, read_jsonl, scripted_session  # noqa: F401

from weightcode.analysis.session_info import session_info
from weightcode.evalkit import seqskill as Q
from weightcode.evalkit.devreadout import run_limits
from weightcode.evalkit.instances import instances_root
from weightcode.harness import seqskill
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import PROGRAM_MAX_CHANGED_FRAC
from weightcode.harness.experience import REL_FROB_NOT_APPLIED
from weightcode.harness.finalize import KIND_CHECKED, write_final
from weightcode.harness.instance import InstanceRefused
from weightcode.harness.jsonio import read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.prompts import Budgets, render_system, render_task
from weightcode.harness.runner import run_session
from weightcode.harness.session import SessionPaths
from weightcode.harness.trial import TrialError, assemble
from weightcode.patch import PatchError, load_patch

RULE_ONLY = seqskill.RULE_ONLY
STRINGS = 'HYPOTHESIS = "one large entry"\nPREDICTED_EFFECT = "changes the head only"\nimport numpy as np\n'
# One entry of head.weight moved by ten times the tensor's norm: rel-Frob about 10, far below the 20% entry cap.
BIG = STRINGS + '''def update(student, experience):
    w = student["params"]["head.weight"]
    delta = np.zeros(w.shape, dtype=np.float64)
    delta[0, 0] = 10.0 * float(np.linalg.norm(w))
    return {"head.weight": delta}
'''
# Every entry of every tensor moved a little: every entry changes, far above the 20% entry cap.
WIDE = STRINGS + '''def update(student, experience):
    return {name: np.full(value.shape, 1e-3) for name, value in student["params"].items()}
'''


@pytest.fixture
def seq_id(store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:  # noqa: F811
    """The toy seqskill instance built, with a runs root for sessions."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(runs))
    (row,) = Q.build_seqskill_instances(CONDITION)
    return str(row["instance_id"])


def _view(root: Path, directory: Path, **kwargs):  # type: ignore[no-untyped-def]
    (directory / "private").mkdir(parents=True)
    return assemble(root, "LEARN", directory, **kwargs)


def _cell(programs: dict[str, str]) -> str:
    payload = json.dumps(json.dumps(programs))
    return ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n")


@pytest.mark.parametrize("variant", [None, RULE_ONLY])
def test_the_lifted_view_states_no_rel_frob_limit(seq_id: str, tmp_path: Path, variant: str | None) -> None:
    root = Q.seqskill_instance_dir(seq_id)
    plain = _view(root, tmp_path / "plain", seqskill_variant=variant)
    off = _view(root, tmp_path / "off", seqskill_variant=variant, lift_rel_frob=False)
    lifted = _view(root, tmp_path / "lifted", seqskill_variant=variant, lift_rel_frob=True)
    assert off.files == plain.files and "lift_rel_frob" not in plain.as_dict()  # the default path is unchanged
    assert plain.limits["max_rel_frob"] == 0.5 and lifted.limits["max_rel_frob"] is None
    assert {k: v for k, v in lifted.limits.items() if k != "max_rel_frob"} == {
        k: v for k, v in plain.limits.items() if k != "max_rel_frob"}
    assert lifted.limits["max_changed_frac"] == PROGRAM_MAX_CHANGED_FRAC and lifted.limits["max_ops"] == plain.limits[
        "max_ops"]
    assert lifted.as_dict()["lift_rel_frob"] is True and lifted.as_dict()["limits"] == lifted.limits
    # only task.json differs, and in it only the limits and their definitions
    assert {k: v for k, v in lifted.files.items() if k != "task.json"} == {
        k: v for k, v in plain.files.items() if k != "task.json"}
    assert sorted(key for key in lifted.task if lifted.task[key] != plain.task[key]) == ["limit_definitions",
                                                                                       "limits"]
    assert lifted.task["limits"] == {**plain.task["limits"], "max_rel_frob": None}
    definitions, base = lifted.task["limit_definitions"], plain.task["limit_definitions"]
    assert definitions["max_rel_frob"] == base["max_rel_frob"] + "; null: this limit is not applied"
    for name in ("rel_frob_min_ndim", "rel_frob_exempt"):
        assert definitions[name] == base[name] + REL_FROB_NOT_APPLIED
    assert definitions["max_changed_frac"] == base["max_changed_frac"]
    # the prompts: the task prompt's limits state that there is none; the system prompt states no rel-Frob limit
    budgets = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=5400.0)
    shown = render_task("LEARN", lifted.task, lifted.limits, budgets, seqskill=True, seqskill_variant=variant)
    e5 = render_task("LEARN", plain.task, plain.limits, budgets, seqskill=True, seqskill_variant=variant)
    limits_text = shown.split("## Limits\n", 1)[1]
    assert "- `max_rel_frob` = not applied: " in limits_text and "null: this limit is not applied" in limits_text
    assert "- `max_rel_frob` = 0.5" in e5 and "0.5" not in limits_text and "- `max_changed_frac` = 0.2" in limits_text
    assert shown.split("## Limits\n", 1)[0] == e5.split("## Limits\n", 1)[0]
    system = render_system("LEARN", seqskill=True, seqskill_variant=variant)
    assert "frob" not in system.lower() and "0.5" not in system  # nothing there to state


def test_the_lift_is_refused_outside_a_seqskill_learn_session(seq_id: str, tmp_path: Path) -> None:
    root = Q.seqskill_instance_dir(seq_id)
    with pytest.raises(ValueError, match="LEARN arm only"):
        run_session(seq_id, "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True, lift_rel_frob=True)
    with pytest.raises(ValueError, match="must be a bool"):
        run_session(seq_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True, lift_rel_frob="yes")
    (tmp_path / "m2" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="LEARN arm only"):
        assemble(root, "M2", tmp_path / "m2", lift_rel_frob=True)
    assert not list(Path(os.environ["WEIGHTCODE_RUNS"]).rglob("session.json"))  # refused before any write


def test_the_lift_is_refused_for_an_evidence_v1_instance(harness_store: HarnessStore,  # noqa: F811
                                                         tmp_path: Path) -> None:
    instance_id = harness_store.dev(100)
    with pytest.raises(InstanceRefused, match="not a seqskill-v1 instance"):
        run_session(instance_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True, lift_rel_frob=True)
    assert not list(harness_store.store.runs.rglob("session.json"))  # refused before any write
    (tmp_path / "s" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="seqskill-v1 instance only"):
        assemble(instances_root() / instance_id, "LEARN", tmp_path / "s", lift_rel_frob=True)


def _refusal(reply) -> str:  # type: ignore[no-untyped-def]
    assert reply.is_error, reply.payload
    return " ".join(reply.payload["error"])


@needs_sandbox
def test_a_large_concentrated_update_passes_only_with_the_lift(seq_id: str) -> None:
    steps = [("python", {"code": _cell({"big.py": BIG, "wide.py": WIDE})}),
             ("patch_check", {"program_file": "wide.py"}), ("patch_check", {"program_file": "big.py"}),
             ("declare_final", {"program_file": "big.py", "rationale": "the large entry"})]
    lifted, replies = scripted_session(seq_id, "LEARN", steps, retention_floor_frac=0.005, seqskill_variant=RULE_ONLY,
                                       lift_rel_frob=True)
    assert lifted.status == "complete", lifted.record.get("infra_reasons")
    _, wide, big, declared = replies
    refused = _refusal(wide)  # the cap still refuses
    assert refused.startswith("size: ") and "above the limit 0.2" in refused and "Frobenius" not in refused
    assert not big.is_error and big.payload["tensors"] == ["head.weight"]
    assert declared.payload["valid"] and declared.payload["checked"] is True  # a checked program, declared
    paths = SessionPaths(lifted.root)
    session = read_json(paths.session_json)
    assert session["lift_rel_frob"] is True and session["limits"]["max_rel_frob"] is None
    assert session["limits"]["max_changed_frac"] == PROGRAM_MAX_CHANGED_FRAC
    assert session["trial"]["limits"] == session["limits"] and session["trial"]["lift_rel_frob"] is True
    assert read_json(paths.trial / "task.json")["limits"]["max_rel_frob"] is None
    (entry,) = read_jsonl(paths.checks_log)
    attempt_dir = lifted.root / "attempts" / str(entry["attempt"])
    patch = load_patch(attempt_dir / str(entry["patch_file"]))
    result = ArmApplier(session).apply(patch, attempt_dir)
    assert result.per_tensor["head.weight"]["rel_frob"] > 0.5 and result.changed_frac < PROGRAM_MAX_CHANGED_FRAC
    capped = copy.deepcopy(session)
    capped["limits"]["max_rel_frob"] = 0.5
    with pytest.raises(PatchError, match="relative Frobenius"):  # the same update under the limit: refused
        ArmApplier(capped).apply(patch, attempt_dir)
    # the final and the readout replays apply the session's lifted limits
    ledger = CheckLedger(session, paths)
    assert ledger.reapply(ledger.entry_for(str(entry["patch_id"]))).child_sha256 == entry["child_sha256"]
    final = write_final(paths, ledger, {"kind": KIND_CHECKED, "patch_id": entry["patch_id"]}, None, "test")
    assert final["child_sha256"] == entry["child_sha256"]
    limits, source = run_limits({"session_result": "session_result.json"}, lifted.root)
    assert limits.max_rel_frob is None and limits.max_changed_frac == PROGRAM_MAX_CHANGED_FRAC and source == (
        "session.json")
    info = session_info(lifted.root)
    assert [row["child_sha256"] for row in info["trajectory"]] == [entry["child_sha256"]]

    # the same session without the lift refuses the same program at the check, so it is declared unchecked
    steps = [("python", {"code": _cell({"big.py": BIG})}), ("patch_check", {"program_file": "big.py"}),
             ("declare_final", {"program_file": "big.py", "rationale": "the large entry"})]
    plain, replies = scripted_session(seq_id, "LEARN", steps, retention_floor_frac=0.005, seqskill_variant=RULE_ONLY)
    assert plain.status == "complete", plain.record.get("infra_reasons")
    _, check, declared = replies
    assert "relative Frobenius norm of the change is above the limit 0.5" in _refusal(check)
    assert declared.payload["checked"] is False
    session = read_json(SessionPaths(plain.root).session_json)
    assert "lift_rel_frob" not in session and session["limits"]["max_rel_frob"] == 0.5
    log = SessionPaths(plain.root).checks_log
    assert not log.exists() or not read_jsonl(log)  # no query was used
