"""The editor harness on fewshot-v1 instances (PREREG E2): id resolution and refusals, the LEARN view, the prompts,
the check on V_check cut by class, the session record, session_info and program transfer.

The store is the evalkit toy fewshot store (``test_evalkit_fewshot.fewshot_store``): three random-weight CIFAR-10
students (cnn seeds 100 and 101, cnn4 seed 100) whose new-class head biases are -50, so they never predict a new
class. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_fewshot import NEW, fewshot_store  # noqa: F401
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session

from weightcode.analysis.session_info import format_table, session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256
from weightcode.evalkit import fewshot as F
from weightcode.harness import fewshot
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import PROGRAM_MAX_CHANGED_FRAC, arm_spec
from weightcode.harness.experience import EXPERIENCE_FILE, EXPERIENCE_KEYS
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.prompts import render_system
from weightcode.harness.runner import SANDBOX_PYTHON, run_session
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import FEWSHOT_CHECK_SETS, PROGRAM_CHECK_SETS, tool_specs
from weightcode.harness.transfer import run_transfer
from weightcode.harness.trial import CLEAN_VCHECK_FILE, TrialError, assemble
from weightcode.sandbox.program import ProgramSandbox

OLD = [0, 1, 2, 3, 4]
# Sets every new-class head bias to the mean old-class bias: an architecture-agnostic edit (the head is the last
# linear layer of the layout), so the same file runs on the cnn and the cnn4 students.
LIFT = '''HYPOTHESIS = "the new-class outputs are silenced by their head biases"
PREDICTED_EFFECT = "some new-class images become correct and some old-class images break"
import numpy as np
def update(student, experience):
    head = [layer for layer in student["layout"]["layers"] if layer.get("op") == "linear"][-1]
    bias = student["params"][head["bias"]]
    new = sorted(set(int(c) for c in experience["y_new"]))
    old = [c for c in range(len(bias)) if c not in new]
    delta = np.zeros(bias.shape, dtype=np.float64)
    delta[new] = float(np.mean(bias[old])) - bias[new]
    print("lifted", new)
    return {head["bias"]: delta}
'''


@pytest.fixture
def fewshot_ids(fewshot_store: list[dict[str, object]], tmp_path: Path,  # noqa: F811
                monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, int], str]:
    """The toy fewshot instances built, with a runs root for sessions: {(arch, seed): instance id}."""
    runs = tmp_path / "store" / "runs"
    runs.mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(runs))
    ids = {}
    for outcome in F.build_fewshot_instances("cifar10"):
        private = read_json(F.fewshot_instance_dir(outcome["instance_id"]) / "private.json")
        ids[(str(private["arch"]), int(private["seed"]))] = outcome["instance_id"]
    return ids


def _paragraphs(text: str) -> list[str]:
    return [block for block in text.split("\n\n") if block.strip()]


def _program_cell(programs: dict[str, str]) -> str:
    payload = json.dumps(json.dumps(programs))
    return ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n"
            f"print(sorted(json.loads({payload})))\n")


def _sandbox() -> ProgramSandbox:
    return ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)


def test_the_harness_names_agree_with_the_evalkit_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path))
    assert fewshot.FEWSHOT_RULE == F.FEWSHOT_RULE
    assert fewshot.INSTANCE_ID_RE.pattern == F.INSTANCE_ID_RE.pattern
    assert fewshot.EXPERIENCE_KEYS == F.EVIDENCE_KEYS == EXPERIENCE_KEYS
    assert fewshot.instance_dir("f0123abcd") == F.fewshot_instance_dir("f0123abcd")
    assert fewshot.is_fewshot_id("f0123abcd") and not fewshot.is_fewshot_id("i0123abcd")
    with pytest.raises(fewshot.FewshotHarnessError, match="not a fewshot instance id"):
        fewshot.instance_dir("i0123abcd")


def test_a_fewshot_id_resolves_in_its_own_store_and_is_refused_when_not_current(
        fewshot_ids: dict[tuple[str, int], str]) -> None:
    instance_id = fewshot_ids[("cnn", 100)]
    record = read_instance_record(instance_id)
    assert record.is_fewshot and record.instance_dir == F.fewshot_instance_dir(instance_id)
    assert record.index_row["instance_id"] == instance_id and record.split_role == "dev"
    with pytest.raises(InstanceRefused, match="does not exist"):
        read_instance_record("f00000000")
    with pytest.raises(InstanceRefused, match="program arm"):
        run_session(instance_id, "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True)
    assert not list(Path(os.environ["WEIGHTCODE_RUNS"]).rglob("session.json"))  # refused before any write
    other = fewshot_ids[("cnn", 101)]
    task_path = F.fewshot_instance_dir(other) / "task.json"
    task_path.write_text(task_path.read_text(encoding="utf-8").replace("trained", "taught", 1), encoding="utf-8")
    with pytest.raises(InstanceRefused, match="never runs a stale"):
        read_instance_record(other)


def test_the_learn_view_is_the_evidence_with_the_class_lists(fewshot_ids: dict[tuple[str, int], str],
                                                             tmp_path: Path) -> None:
    instance_id = fewshot_ids[("cnn", 100)]
    root = F.fewshot_instance_dir(instance_id)
    session_dir = tmp_path / "session"
    (session_dir / "private").mkdir(parents=True)
    view = assemble(root, "LEARN", session_dir)
    instance = load_instance(root)
    with np.load(view.trial_dir / EXPERIENCE_FILE, allow_pickle=False) as archive:
        shown = {key: archive[key] for key in archive.files}
    assert sorted(shown) == sorted(F.EVIDENCE_KEYS)
    assert all(np.array_equal(shown[key], instance.evidence[key]) for key in F.EVIDENCE_KEYS)
    assert sorted(path.name for path in view.trial_dir.iterdir()) == sorted(
        ["model.npz", "model_spec.json", "forward.py", "task.json", EXPERIENCE_FILE])
    record = view.experience
    assert record is not None and record["source"] == fewshot.EXPERIENCE_SOURCE
    assert (record["n_new"], record["per_class"], record["n_old"]) == (50, 10, 64)
    assert record["n_new_fail"] == 50  # the student never predicts a new class
    assert record["content_sha256"] == instance.private["evidence_content_sha256"]
    assert view.decoy_model_sha256 is None and view.clean_check is None
    assert not (session_dir / "private" / CLEAN_VCHECK_FILE).exists()
    private = instance.private
    assert view.fewshot == {"rule": F.FEWSHOT_RULE, "train_classes": OLD, "new_classes": NEW,
                            "n_vcheck_new": private["n_vcheck_new"], "n_vcheck_old": private["n_vcheck_old"],
                            "n_base_errors_vcheck_new": private["n_base_errors_vcheck_new"],
                            "n_base_errors_vcheck_old": private["n_base_errors_vcheck_old"]}
    task = view.task
    assert task["parent_sha256"] == instance.params_sha256
    assert "evidence" not in task and "patch_format" not in task and task["train_classes"] == OLD
    assert task["experience"]["new"]["classes"] == NEW and task["experience"]["old"]["classes"] == OLD
    assert task["experience"]["new"]["count"] == 50 and task["experience"]["old"]["count"] == 64
    assert "never saw an image of their class" in task["statement"]
    assert view.limits["rel_frob_exempt"] == ["fc2.weight", "fc2.bias"]
    assert view.limits["max_changed_frac"] == PROGRAM_MAX_CHANGED_FRAC and view.limits["max_edit_frac"] is None
    (tmp_path / "other" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="program arm only"):
        assemble(root, "M2", tmp_path / "other")


def test_the_fewshot_cut_refuses_part_sizes_other_than_recorded(fewshot_ids: dict[tuple[str, int], str]) -> None:
    instance = load_instance(F.fewshot_instance_dir(fewshot_ids[("cnn", 100)]))
    record = fewshot.session_record(instance.private)
    with np.load(instance.private["vcheck_path"], allow_pickle=False) as archive:
        x, y = archive["x"], archive["y"]
    parts = fewshot.check_parts(x, y, record)
    expected = F.fewshot_check_parts({"x": x, "y": y}, NEW)
    assert all(np.array_equal(parts[key], expected[key]) for key in F.CHECK_PART_KEYS)
    with pytest.raises(fewshot.FewshotHarnessError, match="cuts into"):
        fewshot.check_parts(x, y, {**record, "n_vcheck_new": record["n_vcheck_new"] + 1})
    with pytest.raises(fewshot.FewshotHarnessError, match="rule"):
        fewshot.check_parts(x, y, {**record, "rule": "evidence-v1"})


def test_the_fewshot_prompts_change_only_the_sentences_false_for_new_classes() -> None:
    learn, few = render_system("LEARN"), render_system("LEARN", fewshot=True)
    learn_blocks, few_blocks = _paragraphs(learn), _paragraphs(few)
    assert len(learn_blocks) == len(few_blocks)
    changed = [index for index, (a, b) in enumerate(zip(learn_blocks, few_blocks)) if a != b]
    assert len(changed) == 2
    assert "new-condition" not in few and "clean images" not in few and "deployment" not in few
    assert "new classes (score_new)" in few and "old classes (score_old)" in few
    with pytest.raises(ValueError, match="program arm"):
        render_system("M2", fewshot=True)
    tools = arm_spec("LEARN").tools
    plain = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True)}
    shown = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True, fewshot=True)}
    assert shown["patch_check"] == plain["patch_check"].replace(PROGRAM_CHECK_SETS, FEWSHOT_CHECK_SETS)
    assert shown["patch_check"] != plain["patch_check"]
    assert {k: v for k, v in shown.items() if k != "patch_check"} == {k: v for k, v in plain.items()
                                                                      if k != "patch_check"}
    with pytest.raises(ValueError, match="program arm"):
        tool_specs(arm_spec("M2").tools, fewshot=True)


@needs_sandbox
def test_a_fewshot_learn_session_scores_programs_on_the_class_cut(fewshot_ids: dict[tuple[str, int], str],
                                                                  tmp_path: Path) -> None:
    instance_id = fewshot_ids[("cnn", 100)]
    steps = [
        ("python", {"code": _program_cell({"lift.py": LIFT})}),
        ("patch_check", {"program_file": "lift.py"}),
        ("status", {}),
        ("declare_final", {"program_file": "lift.py", "rationale": "the only program"}),
    ]
    result, replies = scripted_session(instance_id, "LEARN", steps)
    assert result.status == "complete", result.record.get("infra_reasons")
    _, check, status, declared = replies
    assert not check.is_error, check.payload
    assert check.payload["score"] == check.payload["score_new"] + check.payload["score_old"]
    assert check.payload["tensors"] == ["fc2.bias"] and check.payload["program_stdout"].startswith("lifted")
    assert declared.payload["valid"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    assert session["fewshot"]["new_classes"] == NEW and session["clean_check"] is None
    assert session["condition"] is None and session["decoy_model_id"] is None
    assert session["leak_scan"]["passed"] is True
    assert paths.system_prompt.read_text(encoding="utf-8") == render_system("LEARN", fewshot=True)
    entry = read_jsonl(paths.checks_log)[0]
    attempt_dir = result.root / "attempts" / str(entry["attempt"])
    child = ArmApplier(session).apply(read_json(attempt_dir / str(entry["patch_file"])), attempt_dir).child
    assert params_sha256(child) == entry["child_sha256"]
    expected = F.make_fewshot_oracle(instance_id).query(child, "lift")  # evalkit's own class-cut oracle
    assert (check.payload["score_new"], check.payload["score_old"]) == (expected["score_new"], expected["score_old"])
    ledger = CheckLedger(session, paths)  # rebuilt from session.json alone, as the MCP server builds it
    assert [row["score"] for row in ledger.checked()] == [check.payload["score"]]

    info = session_info(result.root)
    row = info["trajectory"][0]
    assert row["readouts_fewshot"] == F.fewshot_readout(instance_id, child)
    assert row["vselect"]["delta_acc"] == row["readouts_fewshot"]["delta_all"]
    assert row["vselect_new"]["n"] + row["vselect_old"]["n"] == row["vselect"]["n"]
    assert "vselect_clean" not in row and "dacc_all" in format_table(info)

    program = tmp_path / "lift.py"
    program.write_text(LIFT, encoding="utf-8")
    targets = [fewshot_ids[("cnn", 101)], fewshot_ids[("cnn4", 100)]]
    records = [json.loads(path.read_text()) for path in run_transfer(program, targets, tmp_path / "transfer",
                                                                      _sandbox())]
    assert [record["target"]["instance_id"] for record in records] == targets
    assert [record["tensors"] for record in records] == [["fc2.bias"], ["fc1.bias"]]  # cnn4's head is fc1
    for record in records:
        assert record["status"] == "ok" and record["within_limits"] is True
        assert record["experience"]["source"] == fewshot.EXPERIENCE_SOURCE
        assert record["readouts"]["delta_acc"] == record["readouts_fewshot"]["delta_all"]
        assert "readouts_clean" not in record
