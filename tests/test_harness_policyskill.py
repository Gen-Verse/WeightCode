"""The editor harness on policyskill-v1 instances (PREREG E14): the names against the evalkit rule, id resolution and
refusals, the LEARN view, the prompts and tool texts, the V_check seed cut, the MCP floors, and a scripted LEARN session
with its rollout check, ledger rebuild and session_info readout.

The store is the scratch E14 store of ``test_e14_refs`` (an untrained film_policy recorded as a gated base, V_check and
V_select cut to a few episodes each). No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")

from test_e14_refs import N_EPISODES, instance_id, store  # noqa: E402,F401
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session  # noqa: E402

from weightcode.analysis.session_info import format_table, session_info  # noqa: E402
from weightcode.baselines.instance_io import load_instance  # noqa: E402
from weightcode.common.identity import params_sha256  # noqa: E402
from weightcode.data import e14 as D  # noqa: E402
from weightcode.harness import e10, policyskill  # noqa: E402
from weightcode.harness.applying import ArmApplier  # noqa: E402
from weightcode.harness.arms import arm_spec  # noqa: E402
from weightcode.harness.experience import EXPERIENCE_FILE  # noqa: E402
from weightcode.harness.instance import InstanceRefused, read_instance_record  # noqa: E402
from weightcode.harness.jsonio import read_json  # noqa: E402
from weightcode.harness.ledger import CheckLedger  # noqa: E402
from weightcode.harness.leaks import HINT_WORDS  # noqa: E402
from weightcode.harness.prompts import Budgets, render_system, render_task  # noqa: E402
from weightcode.harness.retention import kept_rule_record_rows  # noqa: E402
from weightcode.harness.runner import run_session, session_forbidden_strings  # noqa: E402
from weightcode.harness.session import SessionPaths  # noqa: E402
from weightcode.harness.tools import POLICYSKILL_CHECK_SETS, PROGRAM_CHECK_SETS, tool_specs  # noqa: E402
from weightcode.harness.trial import TrialError, assemble  # noqa: E402

# Gives the two untrained instruction words the embedding rows of the trained verbs, found by name in the layout: an
# architecture-agnostic edit of one tensor.
COPY = '''HYPOTHESIS = "the untrained instruction words have untrained embedding rows, so the policy misreads them"
PREDICTED_EFFECT = "new-mission instructions read as the trained verbs; the trained mission is unchanged"
import numpy as np
def update(student, experience):
    words = {row["word"]: row["id"] for row in student["layout"]["input"]["instruction"]["words"]}
    emb = student["params"]["word_emb.weight"]
    delta = np.zeros(emb.shape, dtype=np.float64)
    for new, old in (("pick", "go"), ("up", "to")):
        delta[words[new]] = emb[words[old]] - emb[words[new]]
    print("copied", len(words))
    return {"word_emb.weight": delta}
'''
FLOOR = -3  # PREREG E14: joint-retain-v1 with S_old >= -3 of the old-mission check episodes


def _paragraphs(text: str) -> list[str]:
    return [block for block in text.split("\n\n") if block.strip()]


def _program_cell(programs: dict[str, str]) -> str:
    payload = json.dumps(json.dumps(programs))
    return ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n"
            f"print(sorted(json.loads({payload})))\n")


def _runs() -> Path:
    return Path(os.environ["WEIGHTCODE_RUNS"])


def test_the_harness_names_agree_with_the_evalkit_rule(store) -> None:  # noqa: F811
    from weightcode.evalkit import e14 as E

    assert policyskill.POLICYSKILL_RULE == E.POLICYSKILL_RULE
    assert policyskill.INSTANCE_ID_RE.pattern == E.INSTANCE_ID_RE.pattern
    assert policyskill.EXPERIENCE_KEYS == tuple(E.EVIDENCE_KEYS)
    assert policyskill.POLICYSKILL_ROOT_NAME == E.ROOT_NAME
    assert policyskill.instance_dir("p0123abcd") == E.e14_instance_dir("p0123abcd")
    assert policyskill.is_policyskill_id("p0123abcd") and not policyskill.is_policyskill_id("q0123abcd")
    assert policyskill.leak_words() == E.e14_leak_tokens()
    with pytest.raises(policyskill.PolicyskillHarnessError, match="not a policyskill instance id"):
        policyskill.instance_dir("i0123abcd")
    floors = (e10.MCP_TOOL_TIMEOUT_MS, e10.MCP_CONNECT_TIMEOUT_MS)
    own = {"evidence_rule": policyskill.POLICYSKILL_RULE}
    assert (policyskill.MCP_TOOL_TIMEOUT_MS, policyskill.MCP_CONNECT_TIMEOUT_MS) == floors
    assert policyskill.mcp_timeouts(own, None, None, (600_000, 120_000)) == floors
    assert policyskill.mcp_timeouts({"evidence_rule": "evidence-v1"}, None, None, (600_000, 120_000)) == (600_000,
                                                                                                        120_000)
    with pytest.raises(ValueError, match="at least 1200000"):
        policyskill.mcp_timeouts(own, 600_000, None, (600_000, 120_000))


def test_a_policyskill_id_resolves_and_is_refused_outside_its_session(store, instance_id: str) -> None:  # noqa: F811
    record = read_instance_record(instance_id)
    assert record.is_policyskill and record.has_own_experience and not record.is_seqskill
    assert record.instance_dir == policyskill.instance_dir(instance_id) and record.split_role == "dev"
    assert record.index_row["instance_id"] == instance_id
    with pytest.raises(InstanceRefused, match="does not exist"):
        read_instance_record("p00000000")
    before = sorted(_runs().rglob("session.json"))
    with pytest.raises(InstanceRefused, match="program arm"):
        run_session(instance_id, "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True)
    with pytest.raises(InstanceRefused, match="floor in episodes"):
        run_session(instance_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True)
    with pytest.raises(InstanceRefused, match="not a fraction"):
        run_session(instance_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                    retention_floor_frac=0.006)
    with pytest.raises(InstanceRefused, match="MCP timeouts"):
        run_session(instance_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                    retention_floor_rows=FLOOR, mcp_tool_timeout_ms=600_000)
    assert sorted(_runs().rglob("session.json")) == before  # refused before any write
    forbidden = session_forbidden_strings(record)
    assert set(policyskill.private_paths(record.private)) <= set(forbidden.text)
    assert set(policyskill.leak_words()) <= set(forbidden.text)


def test_the_learn_view_is_the_evidence_with_the_actions_and_the_missions(store, instance_id: str,  # noqa: F811
                                                                          tmp_path: Path) -> None:
    root = policyskill.instance_dir(instance_id)
    session_dir = tmp_path / "session"
    (session_dir / "private").mkdir(parents=True)
    view = assemble(root, "LEARN", session_dir)
    instance = load_instance(root)
    with np.load(view.trial_dir / EXPERIENCE_FILE, allow_pickle=False) as archive:
        shown = {key: archive[key] for key in archive.files}
    assert sorted(shown) == sorted(policyskill.EXPERIENCE_KEYS)
    assert all(np.array_equal(shown[key], instance.evidence[key]) for key in policyskill.EXPERIENCE_KEYS)
    assert sorted(path.name for path in view.trial_dir.iterdir()) == sorted(
        ["model.npz", "model_spec.json", "forward.py", "task.json", EXPERIENCE_FILE])
    record = view.experience
    assert record is not None and record["source"] == policyskill.EXPERIENCE_SOURCE
    assert (record["n_new_demonstrations"], record["n_old"]) == (8, 64)
    assert record["n_new"] == instance.private["n_new_rows"] and record["logits_max_abs_diff"] <= policyskill.LOGIT_ATOL
    assert record["reserved_word_ids"] == [int(D.WORD_ID[word]) for word in D.RESERVED_WORDS]
    assert view.decoy_model_sha256 is None and view.clean_check is None and view.seqskill is None
    assert view.policyskill == policyskill.session_record(instance.private)
    assert view.policyskill["n_vcheck_new"] == view.policyskill["n_vcheck_old"] == N_EPISODES
    task = view.task
    assert task["parent_sha256"] == instance.params_sha256
    assert "evidence" not in task and "patch_format" not in task and "classes" not in task
    assert task["actions"] == instance.task["actions"] and task["new_mission"] == instance.task["new_mission"]
    assert task["experience"]["new"]["demonstrations"] == 8 and task["experience"]["old"]["count"] == 64
    assert '"pick" and "up" never occurred' in task["statement"]
    assert "classifier" not in json.dumps(task["program"]) and "other independently trained policies" in json.dumps(
        task["program"])
    from weightcode.evalkit import e14 as E

    assert view.limits["max_rel_frob"] is None and view.limits["max_changed_frac"] == E.E14_MAX_CHANGED_FRAC
    assert view.limits["max_ops"] >= len(instance.params)
    budgets = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=7200.0)
    prompt = render_task("LEARN", task, view.limits, budgets, view.input_files, policyskill=True)
    assert "classifier" not in prompt and f"Actions by index: {json.dumps(list(D.ACTIONS))}" in prompt
    assert "`logits_new`" in prompt and "`episode_new`" in prompt and "pred_new" not in prompt
    tools = tool_specs(arm_spec("LEARN").tools, program=True, s_old_floor=FLOOR, policyskill=True)
    texts = (prompt + render_system("LEARN", policyskill=True) + json.dumps(tools)).lower()
    assert not [word for word in (*policyskill.leak_words(), *HINT_WORDS) if word in texts]
    (tmp_path / "other" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="program arm only"):
        assemble(root, "M2", tmp_path / "other")


def test_the_seed_cut_refuses_other_files_sizes_and_rules(store, instance_id: str) -> None:  # noqa: F811
    instance = load_instance(policyskill.instance_dir(instance_id))
    record = policyskill.session_record(instance.private)
    with np.load(instance.private["vcheck_path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    parts = policyskill.check_parts(arrays, record)
    assert np.array_equal(parts["new"], D.SEED_RANGES["V_check_new"].seeds)
    assert np.array_equal(parts["old"], D.SEED_RANGES["V_check_old"].seeds)
    with pytest.raises(policyskill.PolicyskillHarnessError, match="cuts into"):
        policyskill.check_parts(arrays, {**record, "n_vcheck_new": record["n_vcheck_new"] + 1})
    with pytest.raises(policyskill.PolicyskillHarnessError, match="rule"):
        policyskill.check_parts(arrays, {**record, "rule": "seqskill-v1"})
    tampered = {**arrays, "x": arrays["x"].copy()}
    tampered["x"][0] += 1
    with pytest.raises(policyskill.PolicyskillHarnessError, match="content"):
        policyskill.check_parts(tampered, record)
    with pytest.raises(policyskill.PolicyskillHarnessError, match="seeds differ"):
        policyskill.check_parts(arrays, {**record, "vcheck_seeds_sha256": {"new": "0" * 64, "old": "0" * 64}})


def test_the_policyskill_prompts_change_only_what_is_false_for_a_policy() -> None:
    learn, pol = render_system("LEARN"), render_system("LEARN", policyskill=True)
    learn_blocks, pol_blocks = _paragraphs(learn), _paragraphs(pol)
    assert len(learn_blocks) == len(pol_blocks)
    changed = [index for index, (a, b) in enumerate(zip(learn_blocks, pol_blocks)) if a != b]
    assert changed == [0, 1, 3]  # the title, the role and the kept-rule paragraph
    assert "new-condition" not in pol and "clean images" not in pol and "classif" not in pol.split("## Rules")[0]
    flat = " ".join(pol.split())
    assert "episodes of the new mission (score_new)" in flat and "episodes of the trained mission (score_old)" in flat
    rule = kept_rule_record_rows(FLOOR, N_EPISODES, policyskill.CHECK_SET)
    kept = " ".join(render_system("LEARN", policyskill=True, kept_rule=rule).split())
    assert "at most 3 more old-mission check-set episodes broken than fixed" in kept
    assert "score_old below -3 is never kept" in kept
    assert max(len(line) for line in render_system("LEARN", policyskill=True, kept_rule=rule).splitlines()) <= 116
    with pytest.raises(ValueError, match="program arm"):
        render_system("M2", policyskill=True)
    with pytest.raises(ValueError, match="not both"):
        render_system("LEARN", seqskill=True, policyskill=True)
    tools = arm_spec("LEARN").tools
    plain = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True)}
    shown = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True, policyskill=True)}
    assert POLICYSKILL_CHECK_SETS in shown["patch_check"] and PROGRAM_CHECK_SETS not in shown["patch_check"]
    assert "classifier" not in shown["patch_check"] and "updated policy is evaluated" in shown["patch_check"]
    assert {k: v for k, v in shown.items() if k != "patch_check"} == {k: v for k, v in plain.items()
                                                                      if k != "patch_check"}
    with pytest.raises(ValueError, match="program arm"):
        tool_specs(arm_spec("M2").tools, policyskill=True)


@needs_sandbox
def test_a_policyskill_learn_session_scores_programs_by_rollouts(store, instance_id: str) -> None:  # noqa: F811
    from weightcode.evalkit import e14 as E

    steps = [
        ("python", {"code": _program_cell({"copy.py": COPY})}),
        ("patch_check", {"program_file": "copy.py"}),
        ("status", {}),
        ("declare_final", {"program_file": "copy.py", "rationale": "the only program"}),
    ]
    result, replies = scripted_session(instance_id, "LEARN", steps, retention_floor_rows=FLOOR,
                                       retention_floor_source={"prereg": "E14", "rule": "S_old >= -3 of 500"})
    assert result.status == "complete", result.record.get("infra_reasons")
    _, check, status, declared = replies
    assert not check.is_error, check.payload
    assert check.payload["score"] == check.payload["score_new"] + check.payload["score_old"]
    assert check.payload["tensors"] == ["word_emb.weight"] and check.payload["program_stdout"].startswith("copied")
    assert declared.payload["valid"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    assert session["policyskill"] == policyskill.session_record(load_instance(
        policyskill.instance_dir(instance_id)).private)
    assert session["clean_check"] is None and session["decoy_model_id"] is None and session["condition"] is None
    assert session["kept_rule"]["s_old_floor"] == FLOOR and session["kept_rule"]["check_set"] == "old-mission"
    assert session["mcp"]["tool_timeout_ms"] == 1_200_000 and session["mcp"]["connect_timeout_ms"] == 900_000
    assert session["leak_scan"]["passed"] is True
    assert set(policyskill.leak_words()) <= set(session["forbidden_strings"]["text"])
    assert paths.system_prompt.read_text(encoding="utf-8") == render_system("LEARN", policyskill=True,
                                                                           kept_rule=session["kept_rule"])
    entry = read_jsonl(paths.checks_log)[0]
    attempt_dir = result.root / "attempts" / str(entry["attempt"])
    child = ArmApplier(session).apply(read_json(attempt_dir / str(entry["patch_file"])), attempt_dir).child
    assert params_sha256(child) == entry["child_sha256"]
    expected = E.make_e14_oracle(instance_id).query(child, "copy")  # evalkit's own rollout oracle
    assert (check.payload["score_new"], check.payload["score_old"]) == (expected["score_new"], expected["score_old"])
    ledger = CheckLedger(session, paths)  # rebuilt from session.json alone, as the MCP server builds it
    assert [row["score"] for row in ledger.checked()] == [check.payload["score"]]

    info = session_info(result.root)
    row = info["trajectory"][0]
    assert row["readouts_policyskill"] == E.e14_readout(instance_id, child)
    assert row["vselect"]["child_acc"] == row["readouts_policyskill"]["new"]["success_child"]
    assert row["vselect_old"]["n"] == N_EPISODES and "vselect_clean" not in row
    assert "gain" in format_table(info)
