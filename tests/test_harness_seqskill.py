"""The editor harness on seqskill-v1 instances (PREREG E5): id resolution and refusals, the LEARN view, the prompts,
the check on V_check cut by part, the retention kept rule on old-operator rows, the session record, session_info and
program transfer.

The store is the evalkit toy seqops store (``test_evalkit_seqskill.store``): a tiny tfm trained for a few steps, which
answers nothing, and the condition new0_substitution. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_seqskill import CONDITION, store  # noqa: F401
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session

from weightcode.analysis.session_info import format_table, session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256
from weightcode.data import seqops as S
from weightcode.evalkit import seqskill as Q
from weightcode.harness import seqskill
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import PROGRAM_MAX_CHANGED_FRAC, arm_spec
from weightcode.harness.experience import EXPERIENCE_FILE, EXPERIENCE_KEYS
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.leaks import HINT_WORDS
from weightcode.harness.prompts import Budgets, render_system, render_task
from weightcode.harness.retention import kept_rule_record
from weightcode.harness.runner import SANDBOX_PYTHON, run_session
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import PROGRAM_CHECK_SETS, SEQSKILL_CHECK_SETS, tool_specs
from weightcode.harness.transfer import run_transfer
from weightcode.harness.trial import CLEAN_VCHECK_FILE, TrialError, assemble
from weightcode.sandbox.program import ProgramSandbox

# Gives the new operator token the embedding row of an old symbol map, found by name in the layout: an
# architecture-agnostic edit that changes one row of one tensor.
COPY = '''HYPOTHESIS = "the new operator token has an untrained embedding row, so the model does not route it"
PREDICTED_EFFECT = "the new rows change and some old-operator rows may break"
import numpy as np
def update(student, experience):
    ids = {row["name"]: row["id"] for row in student["layout"]["input"]["tokens"]}
    new = int(experience["x_new"][0, 1])
    emb = student["params"]["tok_emb.weight"]
    delta = np.zeros(emb.shape, dtype=np.float64)
    delta[new] = emb[ids["OP_SUBA"]] - emb[new]
    print("copied", new)
    return {"tok_emb.weight": delta}
'''
LEAK_WORDS = ("substitution", "permutation", "seqops", "seed", "new0_substitution")


@pytest.fixture
def seq_id(store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:  # noqa: F811
    """The toy seqskill instance built, with a runs root for sessions."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(runs))
    (row,) = Q.build_seqskill_instances(CONDITION)
    return str(row["instance_id"])


def _paragraphs(text: str) -> list[str]:
    return [block for block in text.split("\n\n") if block.strip()]


def _program_cell(programs: dict[str, str]) -> str:
    payload = json.dumps(json.dumps(programs))
    return ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n"
            f"print(sorted(json.loads({payload})))\n")


def test_the_harness_names_agree_with_the_evalkit_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path))
    assert seqskill.SEQSKILL_RULE == Q.SEQSKILL_RULE
    assert seqskill.INSTANCE_ID_RE.pattern == Q.INSTANCE_ID_RE.pattern
    assert seqskill.EXPERIENCE_KEYS == Q.EVIDENCE_KEYS == EXPERIENCE_KEYS
    assert seqskill.SEQSKILL_ROOT_NAME == Q.ROOT_NAME
    assert seqskill.instance_dir("q0123abcd") == Q.seqskill_instance_dir("q0123abcd")
    assert seqskill.is_seqskill_id("q0123abcd") and not seqskill.is_seqskill_id("f0123abcd")
    assert set(seqskill.READOUT_PARTS) >= set(Q.READOUT_PARTS)
    with pytest.raises(seqskill.SeqskillHarnessError, match="not a seqskill instance id"):
        seqskill.instance_dir("i0123abcd")


def test_a_seqskill_id_resolves_in_its_own_store_and_is_refused_when_not_current(seq_id: str) -> None:
    record = read_instance_record(seq_id)
    assert record.is_seqskill and record.has_own_experience and not record.is_fewshot
    assert record.instance_dir == Q.seqskill_instance_dir(seq_id) and record.split_role == "dev"
    assert record.index_row["instance_id"] == seq_id
    with pytest.raises(InstanceRefused, match="does not exist"):
        read_instance_record("q00000000")
    with pytest.raises(InstanceRefused, match="program arm"):
        run_session(seq_id, "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True)
    with pytest.raises(InstanceRefused, match="library is for evidence-v1"):
        run_session(seq_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                    library_sessions=["s0123456789abcdef"])
    assert not list(Path(os.environ["WEIGHTCODE_RUNS"]).rglob("session.json"))  # refused before any write
    task_path = Q.seqskill_instance_dir(seq_id) / "task.json"
    task_path.write_text(task_path.read_text(encoding="utf-8").replace("trained", "taught", 1), encoding="utf-8")
    with pytest.raises(InstanceRefused, match="never runs a stale"):
        read_instance_record(seq_id)


def test_the_learn_view_is_the_evidence_with_the_operator_and_the_rows(seq_id: str, tmp_path: Path) -> None:
    root = Q.seqskill_instance_dir(seq_id)
    session_dir = tmp_path / "session"
    (session_dir / "private").mkdir(parents=True)
    view = assemble(root, "LEARN", session_dir)
    instance = load_instance(root)
    with np.load(view.trial_dir / EXPERIENCE_FILE, allow_pickle=False) as archive:
        shown = {key: archive[key] for key in archive.files}
    assert sorted(shown) == sorted(Q.EVIDENCE_KEYS)
    assert all(np.array_equal(shown[key], instance.evidence[key]) for key in Q.EVIDENCE_KEYS)
    assert sorted(path.name for path in view.trial_dir.iterdir()) == sorted(
        ["model.npz", "model_spec.json", "forward.py", "task.json", EXPERIENCE_FILE])
    record = view.experience
    assert record is not None and record["source"] == seqskill.EXPERIENCE_SOURCE
    assert (record["n_new"], record["n_old"], record["n_new_right"]) == (Q.N_NEW, Q.N_REPLAY, 0)
    assert record["new_operator"] == "OP_NEW0" and record["new_operator_id"] == S.TOKEN_ID["OP_NEW0"]
    assert record["content_sha256"] == instance.private["evidence_content_sha256"]
    assert view.decoy_model_sha256 is None and view.clean_check is None and view.fewshot is None
    assert not (session_dir / "private" / CLEAN_VCHECK_FILE).exists()
    private = instance.private
    assert view.seqskill == {"rule": Q.SEQSKILL_RULE, "new_operator": "OP_NEW0",
                             "new_operator_id": S.TOKEN_ID["OP_NEW0"], "n_vcheck_new": 4000, "n_vcheck_old": 4000,
                             "n_base_errors_vcheck_new": private["n_base_errors_vcheck_new"],
                             "n_base_errors_vcheck_old": private["n_base_errors_vcheck_old"]}
    task = view.task
    assert task["parent_sha256"] == instance.params_sha256
    assert "evidence" not in task and "patch_format" not in task
    assert task["rows"] == instance.task["rows"] and task["new_operator"] == instance.task["new_operator"]
    assert task["experience"]["new"]["count"] == Q.N_NEW and task["experience"]["old"]["count"] == Q.N_REPLAY
    assert "OP_NEW0 never occurred" in task["statement"]
    assert "classifier" not in json.dumps(task["program"]) and "other independently trained models" in json.dumps(
        task["program"])
    assert view.limits["max_rel_frob"] == 0.5 and view.limits["rel_frob_min_ndim"] == 2
    assert view.limits["max_changed_frac"] == PROGRAM_MAX_CHANGED_FRAC and view.limits["max_edit_frac"] is None
    assert not view.limits.get("rel_frob_exempt")  # no exemption: the new token's row is counted in tok_emb.weight
    budgets = Budgets(patch_checks=8, python_cells=60, cell_timeout_s=120.0, wall_timeout_s=5400.0)
    prompt = render_task("LEARN", task, view.limits, budgets, view.input_files, seqskill=True)
    assert "classifier" not in prompt and "Output symbols by index" in prompt and "-1 elsewhere" in prompt
    texts = (prompt + render_system("LEARN", seqskill=True)).lower()
    assert not [word for word in (*LEAK_WORDS, *HINT_WORDS) if word in texts]
    (tmp_path / "other" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="program arm only"):
        assemble(root, "M2", tmp_path / "other")


def test_the_part_cut_refuses_other_sizes_rules_and_rows(seq_id: str) -> None:
    instance = load_instance(Q.seqskill_instance_dir(seq_id))
    record = seqskill.session_record(instance.private)
    with np.load(instance.private["vcheck_path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    parts = seqskill.check_parts(arrays, record)
    expected = Q.seqskill_check_parts(arrays)
    assert all(np.array_equal(parts[key], expected[key]) for key in Q.CHECK_PART_KEYS)
    with pytest.raises(seqskill.SeqskillHarnessError, match="cuts into"):
        seqskill.check_parts(arrays, {**record, "n_vcheck_new": record["n_vcheck_new"] + 1})
    with pytest.raises(seqskill.SeqskillHarnessError, match="rule"):
        seqskill.check_parts(arrays, {**record, "rule": "fewshot-v1"})
    with pytest.raises(seqskill.SeqskillHarnessError, match="alone"):
        seqskill.check_parts(arrays, {**record, "new_operator_id": S.TOKEN_ID["OP_NEW1"]})
    tampered = {**arrays, "x": arrays["x"].copy()}
    old_row = int(np.flatnonzero(arrays["part"] == S.PART_OLD)[0])
    tampered["x"][old_row, 1] = S.TOKEN_ID["OP_NEW2"]
    with pytest.raises(seqskill.SeqskillHarnessError, match="reserved operator"):
        seqskill.check_parts(tampered, record)


def test_the_seqskill_prompts_change_only_what_is_false_for_a_sequence_model() -> None:
    learn, seq = render_system("LEARN"), render_system("LEARN", seqskill=True)
    learn_blocks, seq_blocks = _paragraphs(learn), _paragraphs(seq)
    assert len(learn_blocks) == len(seq_blocks)
    changed = [index for index, (a, b) in enumerate(zip(learn_blocks, seq_blocks)) if a != b]
    assert changed == [0, 1, 3]  # the title, the role and the kept-rule paragraph
    assert "new-condition" not in seq and "clean images" not in seq and "deployment" not in seq
    flat = " ".join(seq.split())
    assert "new operator alone (score_new)" in flat and "old-operator rows (score_old)" in flat
    rule = kept_rule_record(0.005, 4000, seqskill.CHECK_SET)
    assert rule["s_old_floor"] == -20
    kept = render_system("LEARN", seqskill=True, kept_rule=rule)
    assert "at most 20 more old-operator check-set rows broken than fixed" in " ".join(kept.split())
    assert "score_old below -20 is never kept" in " ".join(kept.split())
    assert max(len(line) for line in kept.splitlines()) <= 116
    with pytest.raises(ValueError, match="program arm"):
        render_system("M2", seqskill=True)
    with pytest.raises(ValueError, match="not both"):
        render_system("LEARN", fewshot=True, seqskill=True)
    tools = arm_spec("LEARN").tools
    plain = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True)}
    shown = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True, seqskill=True)}
    assert SEQSKILL_CHECK_SETS in shown["patch_check"] and PROGRAM_CHECK_SETS not in shown["patch_check"]
    assert "classifier" not in shown["patch_check"]
    assert {k: v for k, v in shown.items() if k != "patch_check"} == {k: v for k, v in plain.items()
                                                                      if k != "patch_check"}
    with pytest.raises(ValueError, match="program arm"):
        tool_specs(arm_spec("M2").tools, seqskill=True)


@needs_sandbox
def test_a_seqskill_learn_session_scores_programs_on_the_part_cut(seq_id: str, tmp_path: Path) -> None:
    steps = [
        ("python", {"code": _program_cell({"copy.py": COPY})}),
        ("patch_check", {"program_file": "copy.py"}),
        ("status", {}),
        ("declare_final", {"program_file": "copy.py", "rationale": "the only program"}),
    ]
    result, replies = scripted_session(seq_id, "LEARN", steps, retention_floor_frac=0.005)
    assert result.status == "complete", result.record.get("infra_reasons")
    _, check, status, declared = replies
    assert not check.is_error, check.payload
    assert check.payload["score"] == check.payload["score_new"] + check.payload["score_old"]
    assert check.payload["tensors"] == ["tok_emb.weight"] and check.payload["program_stdout"].startswith("copied")
    assert declared.payload["valid"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    assert session["seqskill"]["new_operator"] == "OP_NEW0" and session["clean_check"] is None
    assert session["condition"] == CONDITION and session["decoy_model_id"] is None
    assert session["kept_rule"] == kept_rule_record(0.005, 4000, seqskill.CHECK_SET)
    assert session["leak_scan"]["passed"] is True
    assert set(LEAK_WORDS) <= set(session["forbidden_strings"]["text"])
    assert paths.system_prompt.read_text(encoding="utf-8") == render_system("LEARN", seqskill=True,
                                                                           kept_rule=session["kept_rule"])
    tools = {spec["name"]: spec for spec in tool_specs(arm_spec("LEARN").tools, program=True, seqskill=True,
                                                       s_old_floor=-20)}
    assert "score_old of at least -20" in tools["declare_final"]["description"]
    entry = read_jsonl(paths.checks_log)[0]
    attempt_dir = result.root / "attempts" / str(entry["attempt"])
    child = ArmApplier(session).apply(read_json(attempt_dir / str(entry["patch_file"])), attempt_dir).child
    assert params_sha256(child) == entry["child_sha256"]
    expected = Q.make_seqskill_oracle(seq_id).query(child, "copy")  # evalkit's own part-cut oracle
    assert (check.payload["score_new"], check.payload["score_old"]) == (expected["score_new"], expected["score_old"])
    ledger = CheckLedger(session, paths)  # rebuilt from session.json alone, as the MCP server builds it
    assert [row["score"] for row in ledger.checked()] == [check.payload["score"]]

    info = session_info(result.root)
    row = info["trajectory"][0]
    assert row["readouts_seqskill"] == Q.seqskill_readout(seq_id, child)
    assert row["vselect"]["delta_acc"] == row["readouts_seqskill"]["i"]["delta"]
    assert row["vselect_old"]["n"] == 4000 and "vselect_clean" not in row
    assert "dem_iii" in format_table(info)

    program = tmp_path / "copy.py"
    program.write_text(COPY, encoding="utf-8")
    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)
    (path,) = run_transfer(program, [seq_id], tmp_path / "transfer", sandbox)
    record = json.loads(path.read_text())
    assert record["status"] == "ok" and record["within_limits"] is True and record["tensors"] == ["tok_emb.weight"]
    assert record["experience"]["source"] == seqskill.EXPERIENCE_SOURCE
    assert record["child_sha256"] == entry["child_sha256"]  # the same program on the same student and experience
    assert record["readouts_seqskill"] == row["readouts_seqskill"] and "readouts_clean" not in record
