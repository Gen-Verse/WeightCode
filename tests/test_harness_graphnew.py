"""The editor harness on graphnew-v1 instances (PREREG E7): id resolution and refusals, the LEARN view with graph.npz,
the limits with the head exempt, the prompts and their derivation, the leak scan, the floor in rows, one forward per
check set, a scripted session whose ledger rebuilds from session.json, session_info, the dev readout and transfer.

The store is the evalkit toy graphnew store (``test_evalkit_graphnew.store``): a random inductive graph of 410 nodes
and one newclass GCN student (seed 100). Every directory lives under pytest tmp dirs; no test touches the real store
or starts Claude Code. The sandbox self-test matches forbidden strings as substrings of the session's host paths, and
those hold the test's name, so no test name here contains one (the dataset parts "ind" and "arxiv", "seed", an arm
word or a condition name).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_graphnew import store  # noqa: F401
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session

from weightcode.analysis.session_info import format_table, session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256
from weightcode.evalkit import graphnew as GN
from weightcode.evalkit.devreadout import readout_run
from weightcode.harness import graphnew
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import PROGRAM_MAX_CHANGED_FRAC, applied_limits, arm_spec
from weightcode.harness.experience import EXPERIENCE_FILE, EXPERIENCE_KEYS
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import file_sha256, read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.leaks import file_text
from weightcode.harness.prompts import PROMPT_DIR, render_system
from weightcode.harness.retention import (
    RetentionRuleError,
    check_floor_rows,
    kept_rule_record,
    kept_rule_record_rows,
    session_floor,
)
from weightcode.harness.runner import SANDBOX_PYTHON, run_session, session_forbidden_strings
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import GRAPHNEW_CHECK_SETS, PROGRAM_CHECK_SETS, tool_specs
from weightcode.harness.transfer import run_transfer
from weightcode.harness.trial import CLEAN_VCHECK_FILE, TrialError, assemble
from weightcode.patch import Limits, PatchError, apply_patch
from weightcode.sandbox.program import ProgramSandbox

HEAD = ["conv3.weight", "conv3.bias"]
SOURCE = {"rule": "E7 formula", "R": -4.93, "d": 0.1, "n_c": 3002, "n_s": 3002}  # the toy numbers are made up
PLACEHOLDER = {"placeholder": True, "note": "a stand-in floor for a dry run, not the PREREG E7 floor"}
WHY = {"hypothesis": "a limit probe", "predicted_effect": "accepted or refused by the limits"}
TRIAL_FILES = ["model.npz", "model_spec.json", "forward.py", "task.json", EXPERIENCE_FILE, "graph.npz"]
# The templates of the fewshot-v1 (E2), seqskill-v1 (E5) and rule_only (E8) sessions: the graphnew path keeps them.
# system_LEARN_fewshot.md and task_LEARN.md are those of E2 session s93c1bda80ae82274.
KEPT_TEMPLATE_SHA256 = {
    "system.md": "5f43f8d6a54ff2eeae729b68cd2b99a3fee2d46766968232abab3453e2fcbaab",
    "system_LEARN.md": "206fc7e3fd3239825c186bbe9906c046a1d93d11a95454458496603c608623d1",
    "system_LEARN_fewshot.md": "bb17d52ff6fe65bd4b1eeeff0f0041b316ef6c8ceb922869543d38fbb054382b",
    # task_LEARN.md of E2 session s93c1bda80ae82274 was e6ce25c2d7c8...; since 2026-11-04 it adds the clock line
    # (wall_s_elapsed, wall_s_remaining in every tool result; WEIGHTCODE-main 18:05) and nothing else.
    "task_LEARN.md": "96a563fd7f234b2af9ca2ab821bc951b65f6c457fc9e8e9579e61ddde07e6c1c",
    "system_LEARN_seqskill.md": "972ae9dede30df9017a8a1e285351829d6f1e63f0749d5170004e500b18d0563",
    "task_LEARN_seqskill.md": "b5c97584b11dd6035978777df4e61010ac2bdeb4952c9566e45da9f4de6f5df4",
    "system_LEARN_seqskill_rule.md": "0650d491e314fc99246e91e40024b119fd379c19e104678b933889a6ccf0adcf",
    "task_LEARN_seqskill_rule.md": "d64730fb5bdb1723e2ef01908a06f1a1186b44ba7cced9e361c347e958bd7328",
}
# Lifts every new-class head bias to the mean old-class bias, after one forward of the student on the new nodes (the
# run's forward reads the graph.npz copy beside it); the head is the last graph_conv layer of the layout.
LIFT = '''HYPOTHESIS = "the new-class outputs are silenced by their head biases"
PREDICTED_EFFECT = "some new-class nodes become correct and some old-class nodes break"
import numpy as np
def update(student, experience):
    head = [layer for layer in student["layout"]["layers"] if layer.get("op") == "graph_conv"][-1]
    bias = student["params"][head["bias"]]
    logits = student["forward"](student["params"], experience["x_new"])
    new = sorted(set(int(c) for c in experience["y_new"]))
    old = [c for c in range(len(bias)) if c not in new]
    delta = np.zeros(bias.shape, dtype=np.float64)
    delta[new] = float(np.mean(bias[old])) - bias[new]
    print("lifted", new, "rows", logits.shape[0])
    return {head["bias"]: delta}
'''


@pytest.fixture
def gid(store: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:  # noqa: F811
    """The toy graphnew instance built, with a runs root for sessions."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(runs))
    (outcome,) = GN.build_graphnew_instances()
    return str(outcome["instance_id"])


def _floor(instance_id: str) -> int:
    """A floor in rows that the toy old part admits (the toy has no PREREG floor)."""
    return -max(1, int(load_instance(GN.graphnew_instance_dir(instance_id)).private["n_vcheck_old"]) // 4)


def _program_cell(programs: dict[str, str]) -> str:
    payload = json.dumps(json.dumps(programs))
    return ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n"
            f"print(sorted(json.loads({payload})))\n")


def _paragraphs(text: str) -> list[str]:
    return [block for block in text.split("\n\n") if block.strip()]


def test_graphnew_names_agree_with_the_evalkit_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path))
    assert graphnew.GRAPHNEW_RULE == GN.GRAPHNEW_RULE and graphnew.GRAPH_FILE == GN.GRAPH_FILE
    assert graphnew.INSTANCE_ID_RE.pattern == GN.INSTANCE_ID_RE.pattern
    assert graphnew.GRAPHNEW_ROOT_NAME == GN.ROOT_NAME
    assert graphnew.EXPERIENCE_KEYS == GN.EVIDENCE_KEYS == EXPERIENCE_KEYS
    assert graphnew.instance_dir("g0123abcd") == GN.graphnew_instance_dir("g0123abcd")
    assert graphnew.is_graphnew_id("g0123abcd") and not graphnew.is_graphnew_id("f0123abcd")
    assert graphnew.leak_words() == GN.graphnew_leak_tokens()
    with pytest.raises(graphnew.GraphnewHarnessError, match="graphnew instance id"):
        graphnew.instance_dir("f0123abcd")


def test_a_graphnew_id_resolves_in_its_store_and_refusals_come_before_writes(gid: str) -> None:
    record = read_instance_record(gid)
    assert record.is_graphnew and record.has_own_experience and not record.is_fewshot
    assert record.instance_dir == GN.graphnew_instance_dir(gid) and record.split_role == "dev"
    with pytest.raises(InstanceRefused, match="does not exist"):
        read_instance_record("g00000000")
    floor = _floor(gid)
    refusals = [
        (InstanceRefused, "program arm", dict(arm="M2")),
        (ValueError, "program arm only", dict(arm="M2", retention_floor_rows=floor)),
        (InstanceRefused, "retention_floor_rows", dict(arm="LEARN")),
        (ValueError, "not both", dict(arm="LEARN", retention_floor_rows=floor, retention_floor_frac=0.01)),
        (InstanceRefused, "not a fraction", dict(arm="LEARN", retention_floor_frac=0.01)),
        (ValueError, "at most 0", dict(arm="LEARN", retention_floor_rows=1)),
        (ValueError, "an int", dict(arm="LEARN", retention_floor_rows=-1.5)),
        (InstanceRefused, "lies in", dict(arm="LEARN", retention_floor_rows=-10**6)),
        (ValueError, "goes with a floor in rows",
         dict(arm="LEARN", retention_floor_frac=0.01, retention_floor_source=SOURCE)),
        (ValueError, "JSON object", dict(arm="LEARN", retention_floor_rows=floor, retention_floor_source=[1])),
        (InstanceRefused, "rule_only", dict(arm="LEARN", retention_floor_rows=floor, seqskill_variant="rule_only")),
    ]
    for error, message, kwargs in refusals:
        arm = kwargs.pop("arm")
        with pytest.raises(error, match=message):
            run_session(gid, arm, 0, effort="low", wall_timeout_s=60.0, dry_run=True, **kwargs)
    with pytest.raises(ValueError, match="dry run only"):  # a placeholder floor never reaches a real session
        run_session(gid, "LEARN", 0, effort="low", wall_timeout_s=60.0, retention_floor_rows=floor,
                    retention_floor_source=PLACEHOLDER)
    assert not list(Path(os.environ["WEIGHTCODE_RUNS"]).rglob("session.json"))  # refused before any write
    task_path = GN.graphnew_instance_dir(gid) / "task.json"
    task_path.write_text(task_path.read_text(encoding="utf-8").replace("trained", "taught", 1), encoding="utf-8")
    with pytest.raises(InstanceRefused, match="never runs a stale"):
        read_instance_record(gid)


def test_the_graphnew_view_is_the_evidence_and_the_graph(gid: str, tmp_path: Path) -> None:
    root = GN.graphnew_instance_dir(gid)
    session_dir = tmp_path / "session"
    (session_dir / "private").mkdir(parents=True)
    view = assemble(root, "LEARN", session_dir)
    instance = load_instance(root)
    private = instance.private
    assert sorted(path.name for path in view.trial_dir.iterdir()) == sorted(TRIAL_FILES)
    with np.load(view.trial_dir / EXPERIENCE_FILE, allow_pickle=False) as archive:
        shown = {key: archive[key] for key in archive.files}
    assert sorted(shown) == sorted(GN.EVIDENCE_KEYS) and shown["x_new"].dtype == np.int64
    assert all(np.array_equal(shown[key], instance.evidence[key]) for key in GN.EVIDENCE_KEYS)
    for folder in (view.trial_dir, session_dir / "private"):  # forward.py reads the copy beside itself
        assert file_sha256(folder / "graph.npz") == private["graph_file_sha256"]
    assert view.input_files == ("graph.npz",) and view.files["graph.npz"] == private["graph_file_sha256"]
    record = view.experience
    assert record is not None and record["source"] == graphnew.EXPERIENCE_SOURCE
    assert (record["n_new"], record["n_old"]) == (len(shown["y_new"]), len(shown["y_old"]))
    assert record["content_sha256"] == private["evidence_content_sha256"]
    assert view.decoy_model_sha256 is None and view.clean_check is None and view.fewshot is None
    assert not (session_dir / "private" / CLEAN_VCHECK_FILE).exists()
    assert view.graphnew == graphnew.session_record(private)
    assert view.graphnew["graph_file_sha256"] == private["graph_file_sha256"]
    task = view.task
    assert task["parent_sha256"] == instance.params_sha256 and task["graph"]["file"] == "graph.npz"
    assert "evidence" not in task and "patch_format" not in task
    assert task["experience"]["new"]["classes"] == private["new_classes"]
    assert task["experience"]["old"]["count"] == record["n_old"] and "node ids" in task["experience"]["inputs"]
    assert "citation graph" in task["statement"] and "nodes" in task["patch_check"]["readout"]
    for name in TRIAL_FILES:  # E6's list: no category code, year, split name or dataset word in the trial
        assert graphnew.scan(name + "\n" + file_text(view.trial_dir / name), graphnew.leak_words()) == [], name
    (tmp_path / "other" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="program arm only"):
        assemble(root, "M2", tmp_path / "other")
    (tmp_path / "third" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="rule_only"):
        assemble(root, "LEARN", tmp_path / "third", seqskill_variant="rule_only")


def test_the_graphnew_limits_exempt_the_head_and_count_batchnorm(gid: str, tmp_path: Path) -> None:
    root = GN.graphnew_instance_dir(gid)
    (tmp_path / "session" / "private").mkdir(parents=True)
    view = assemble(root, "LEARN", tmp_path / "session")
    instance = load_instance(root)
    params, n_tensors = instance.params, len(instance.params)
    expected = GN.graphnew_program_limits(instance.task["limits"], n_tensors)
    assert Limits(**view.limits) == expected == applied_limits(instance.task["limits"], "LEARN", n_tensors)
    assert view.limits["rel_frob_exempt"] == HEAD == GN.head_tensors(read_json(root / "model_spec.json"))
    assert view.limits["max_changed_frac"] == PROGRAM_MAX_CHANGED_FRAC and view.limits["max_edit_frac"] is None
    assert view.limits["max_ops"] >= n_tensors
    limits = Limits(**view.limits)

    def one_entry(tensor: str, value: float) -> dict:
        index = [0] * params[tensor].ndim
        return {"schema": "weightcode.patch.v0", "parent_sha256": instance.params_sha256, **WHY,
                "ops": [{"op": "add", "tensor": tensor, "index": [index], "values": [value]}]}

    big = 1e3
    apply_patch(params, one_entry("conv3.weight", big), None, limits, instance.params_sha256)  # the head is exempt
    apply_patch(params, one_entry("bn1.running_mean", big), None, limits, instance.params_sha256)  # 1-D: no rel-Frob
    with pytest.raises(PatchError, match="rel"):
        apply_patch(params, one_entry("conv2.weight", big), None, limits, instance.params_sha256)
    head = params["conv3.weight"]  # more than 20% of all entries, in the exempt head: the cap still binds
    assert head.size > PROGRAM_MAX_CHANGED_FRAC * sum(value.size for value in params.values())
    whole = {"schema": "weightcode.patch.v0", "parent_sha256": instance.params_sha256, **WHY,
             "ops": [{"op": "add", "tensor": "conv3.weight",
                      "index": [[i, j] for i in range(head.shape[0]) for j in range(head.shape[1])],
                      "values": [0.01] * head.size}]}
    with pytest.raises(PatchError, match="parameters change"):
        apply_patch(params, whole, None, limits, instance.params_sha256)
    n_total = sum(value.size for value in params.values())
    running = [name for name in params if name.endswith("running_mean") or name.endswith("running_var")]
    counted = sum(params[name].size for name in running)
    ops = [{"op": "add", "tensor": name, "index": [[i] for i in range(params[name].size)],
            "values": [0.01] * params[name].size} for name in running]
    patch = {"schema": "weightcode.patch.v0", "parent_sha256": instance.params_sha256, **WHY, "ops": ops}
    if counted > PROGRAM_MAX_CHANGED_FRAC * n_total:  # BatchNorm statistics count in the 20% cap
        with pytest.raises(PatchError):
            apply_patch(params, patch, None, limits, instance.params_sha256)
    else:
        result = apply_patch(params, patch, None, limits, instance.params_sha256)
        assert result.n_changed == counted


def test_one_forward_per_check_set_and_one_for_the_experience(gid: str) -> None:
    instance = load_instance(GN.graphnew_instance_dir(gid))
    private = instance.private
    calls: list[int] = []

    def counting(params: dict, x: np.ndarray) -> np.ndarray:
        calls.append(len(x))
        return instance.forward(params, x)

    graphnew.experience(dataclasses.replace(instance, forward=counting))
    assert calls == [len(instance.evidence["x_new"]) + len(instance.evidence["x_old"])]
    calls.clear()
    with np.load(private["vcheck_path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    parts = graphnew.check_parts(arrays, graphnew.session_record(private))
    sizes = sorted([int(private["n_vcheck_new"]), int(private["n_vcheck_old"])])
    oracle = graphnew.check_oracle(parts, counting, instance.params, 8)
    assert sorted(calls) == sizes  # the base: one forward per check set
    child = {key: value.copy() for key, value in instance.params.items()}
    child["conv3.bias"] = child["conv3.bias"] + np.float32(0.5)
    calls.clear()
    reply = oracle.query(child, "shifted")
    assert sorted(calls) == sizes and reply["score"] == reply["score_new"] + reply["score_old"]
    expected = GN.make_graphnew_oracle(gid).query(child, "shifted")  # evalkit's own oracle
    assert (reply["score_new"], reply["score_old"]) == (expected["score_new"], expected["score_old"])
    record = graphnew.session_record(private)
    with pytest.raises(graphnew.GraphnewHarnessError, match="cuts into"):
        graphnew.check_parts(arrays, {**record, "n_vcheck_new": record["n_vcheck_new"] + 1})
    with pytest.raises(graphnew.GraphnewHarnessError, match="content"):
        graphnew.check_parts({**arrays, "y": arrays["y"][::-1].copy()}, record)
    with pytest.raises(graphnew.GraphnewHarnessError, match="rule"):
        graphnew.check_parts(arrays, {**record, "rule": "fewshot-v1"})
    digest = graphnew.vcheck_file_sha256(private)
    assert digest == file_sha256(Path(private["vcheck_path"]))
    with pytest.raises(graphnew.GraphnewHarnessError):
        graphnew.vcheck_file_sha256({**private, "vcheck_content_sha256": "0" * 64})


def test_the_graphnew_templates_derive_from_the_fewshot_ones() -> None:
    for name, digest in KEPT_TEMPLATE_SHA256.items():  # the fewshot and seqskill templates are untouched
        assert hashlib.sha256((PROMPT_DIR / name).read_bytes()).hexdigest() == digest
    few = (PROMPT_DIR / "system_LEARN_fewshot.md").read_text(encoding="utf-8")
    graph = (PROMPT_DIR / "system_LEARN_graphnew.md").read_text(encoding="utf-8")
    few_blocks, graph_blocks = _paragraphs(few), _paragraphs(graph)
    changed = [index for index, (a, b) in enumerate(zip(few_blocks, graph_blocks)) if a != b]
    assert len(few_blocks) == len(graph_blocks) and len(changed) == 2  # the role and the kept rule's body
    assert graph_blocks[changed[1] - 1] == "## What is kept"
    assert max(len(line) for line in graph.splitlines()) <= 116
    for index in changed:
        assert "images" in few_blocks[index] and "image" not in graph_blocks[index]
    task_few = (PROMPT_DIR / "task_LEARN.md").read_text(encoding="utf-8").split("\n- ")
    task_graph = (PROMPT_DIR / "task_LEARN_graphnew.md").read_text(encoding="utf-8").split("\n- ")
    changed = [index for index, (a, b) in enumerate(zip(task_few, task_graph)) if a != b]
    assert len(task_few) == len(task_graph) and len(changed) == 2
    assert [task_graph[index].split(":")[0] for index in changed] == ["`forward.py`", "`experience.npz`"]
    assert all("graph.npz" in task_graph[index] for index in changed)
    rendered = render_system("LEARN", graphnew=True)
    assert rendered == graph and graphnew.scan(rendered, graphnew.leak_words()) == []
    with pytest.raises(ValueError, match="program arm"):
        render_system("M2", graphnew=True)
    with pytest.raises(ValueError, match="not both"):
        render_system("LEARN", fewshot=True, graphnew=True)
    kept = kept_rule_record_rows(-37, 3002, graphnew.CHECK_SET, SOURCE)
    stated = render_system("LEARN", graphnew=True, kept_rule=kept)
    assert "score_old is at least -37 (at most 37 more old-class check-set nodes broken than fixed)" in " ".join(
        stated.split())
    assert stated.split("## What is kept")[0] == graph.split("## What is kept")[0]
    # a floor in rows is stated exactly as the frac form states the same floor (E2: 0.01 of 2000 rows, -20)
    frac, rows = kept_rule_record(0.01, 2000, "old-class"), kept_rule_record_rows(-20, 2000, "old-class")
    for flags in ({"fewshot": True}, {"graphnew": True}):
        assert render_system("LEARN", kept_rule=rows, **flags) == render_system("LEARN", kept_rule=frac, **flags)
    assert "at least 0 (at most 0 more" in " ".join(render_system(
        "LEARN", graphnew=True, kept_rule=kept_rule_record_rows(0, 3002, graphnew.CHECK_SET)).split())


def test_the_graphnew_tools_name_nodes_and_the_floor() -> None:
    tools = arm_spec("LEARN").tools
    plain = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True)}
    shown = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True, graphnew=True)}
    assert shown["patch_check"] == plain["patch_check"].replace(PROGRAM_CHECK_SETS, GRAPHNEW_CHECK_SETS)
    assert shown["patch_check"] != plain["patch_check"]
    assert {k: v for k, v in shown.items() if k != "patch_check"} == {k: v for k, v in plain.items()
                                                                      if k != "patch_check"}
    floored = {spec["name"]: spec["description"]
               for spec in tool_specs(tools, program=True, graphnew=True, s_old_floor=-37)}
    assert "score_old of at least -37" in floored["declare_final"]
    assert graphnew.scan(json.dumps(list(floored.values())), graphnew.leak_words()) == []
    with pytest.raises(ValueError, match="program arm"):
        tool_specs(arm_spec("M2").tools, graphnew=True)
    with pytest.raises(ValueError, match="one of"):
        tool_specs(tools, program=True, fewshot=True, graphnew=True)


def test_the_floor_in_rows_is_a_checked_count() -> None:
    record = kept_rule_record_rows(-151, 3002, graphnew.CHECK_SET, SOURCE)
    assert record == {"name": "joint-retain-v1", "s_old_floor": -151, "n_old": 3002, "check_set": "old-class",
                      "floor_kind": "rows", "floor_source": SOURCE}
    assert kept_rule_record_rows(-151, 3002, "old-class")["floor_source"] is None
    assert session_floor({"kept_rule": record}) == -151
    assert session_floor({"kept_rule": kept_rule_record_rows(0, 3002, "old-class")}) == 0
    for bad in (1, -3003, True, -1.5, "-20"):
        with pytest.raises(RetentionRuleError):
            check_floor_rows(bad, 3002)
    with pytest.raises(RetentionRuleError, match="JSON object"):
        kept_rule_record_rows(-151, 3002, "old-class", [1])  # type: ignore[arg-type]
    placeholder = kept_rule_record_rows(-100, 3002, graphnew.CHECK_SET, PLACEHOLDER)
    assert session_floor({"kept_rule": placeholder, "dry_run": True}) == -100
    with pytest.raises(RetentionRuleError, match="dry run only"):
        session_floor({"kept_rule": placeholder, "dry_run": False})
    for broken in ({**record, "frac": 0.01}, {**record, "floor_kind": "frac"}, {**record, "s_old_floor": 3},
                   {**record, "s_old_floor": -151.0}):
        with pytest.raises(RetentionRuleError):
            session_floor({"kept_rule": broken})
    frac = kept_rule_record(0.01, 2000, "old-class")  # the E2 record, byte for byte as before
    assert frac == {"name": "joint-retain-v1", "frac": 0.01, "n_old": 2000, "check_set": "old-class",
                    "s_old_floor": -20}
    assert session_floor({"kept_rule": frac}) == -20


def test_the_graphnew_strings_scan_whole_words_in_text_only(gid: str) -> None:
    record = read_instance_record(gid)
    forbidden = session_forbidden_strings(record)
    assert set(graphnew.leak_words()) <= set(forbidden.text)
    assert "2018" not in forbidden.selftest and "valid" not in forbidden.selftest  # mount ids can spell a year
    assert {"arxiv", "ind", "seed"} <= set(forbidden.selftest)
    assert record.private["vcheck_path"] in forbidden.text and record.private["graph_path"] in forbidden.text
    assert graphnew.scan("see cs.LG and the 2018 snapshot", forbidden.text) == ["2018", "cs.lg"]
    assert graphnew.scan("invalid_patches and validation, split e2018ab", forbidden.text) == []


@needs_sandbox
def test_a_graphnew_dry_run_records_its_placeholder_floor(gid: str) -> None:
    floor = _floor(gid)
    result = run_session(gid, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                         retention_floor_rows=floor, retention_floor_source=PLACEHOLDER)
    assert result.status == "dry_run", result.record
    assert result.root.parent.name == "dry_runs"
    session = read_json(SessionPaths(result.root).session_json)
    assert session["kept_rule"] == kept_rule_record_rows(floor, int(session["graphnew"]["n_vcheck_old"]),
                                                         graphnew.CHECK_SET, PLACEHOLDER)
    assert session["leak_scan"]["passed"] is True and result.record["selftest"]["status"] == "pass"
    assert sorted(result.record["trial_files"]) == sorted(TRIAL_FILES)


@needs_sandbox
def test_a_graphnew_learn_session_scores_programs_on_the_class_cut(gid: str, tmp_path: Path) -> None:
    floor = _floor(gid)
    steps = [
        ("python", {"code": _program_cell({"lift.py": LIFT})}),
        ("python", {"code": "import numpy as np\ng = np.load('/work/trial/graph.npz')\nprint(sorted(g.files))\n"}),
        ("patch_check", {"program_file": "lift.py"}),
        ("status", {}),
        ("declare_final", {"program_file": "lift.py", "rationale": "the only program"}),
    ]
    result, replies = scripted_session(gid, "LEARN", steps, retention_floor_rows=floor, retention_floor_source=SOURCE)
    assert result.status == "complete", result.record.get("infra_reasons")
    _, graph_cell, check, _, declared = replies
    assert not graph_cell.is_error and "features" in json.dumps(graph_cell.payload)
    assert not check.is_error, check.payload
    assert check.payload["score"] == check.payload["score_new"] + check.payload["score_old"]
    assert check.payload["tensors"] == ["conv3.bias"] and check.payload["program_stdout"].startswith("lifted")
    assert declared.payload["valid"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    instance = load_instance(GN.graphnew_instance_dir(gid))
    assert session["graphnew"] == graphnew.session_record(instance.private) and session["clean_check"] is None
    assert session["kept_rule"] == kept_rule_record_rows(floor, int(instance.private["n_vcheck_old"]),
                                                         graphnew.CHECK_SET, SOURCE)
    assert session["vcheck_sha256"] == file_sha256(Path(instance.private["vcheck_path"]))
    assert session["leak_scan"]["passed"] is True and "cs.lg" in session["leak_scan"]["tokens"]
    assert paths.system_prompt.read_text(encoding="utf-8") == render_system(
        "LEARN", graphnew=True, kept_rule=session["kept_rule"])
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "int64 node ids into\n  `graph.npz`" in task_prompt and "`graph.npz`" in task_prompt
    entry = read_jsonl(paths.checks_log)[0]
    attempt_dir = result.root / "attempts" / str(entry["attempt"])
    child = ArmApplier(session).apply(read_json(attempt_dir / str(entry["patch_file"])), attempt_dir).child
    assert params_sha256(child) == entry["child_sha256"]
    expected = GN.make_graphnew_oracle(gid).query(child, "lift")  # evalkit's own class-cut oracle
    assert (check.payload["score_new"], check.payload["score_old"]) == (expected["score_new"], expected["score_old"])
    ledger = CheckLedger(session, paths)  # rebuilt from session.json alone, as the MCP server builds it
    assert [row["score"] for row in ledger.checked()] == [check.payload["score"]] == [entry["score"]]
    private_graph = Path(session["forward_path"]).parent / "graph.npz"
    private_graph.write_bytes(private_graph.read_bytes() + b"\0")  # the forward's graph is checked by hash
    with pytest.raises(Exception, match="graph.npz"):
        CheckLedger(session, paths)

    info = session_info(result.root)
    row = info["trajectory"][0]
    assert row["readouts_graphnew"] == graphnew.readout(instance, child)["readouts_graphnew"]
    assert row["vselect"]["delta_acc"] == row["readouts_graphnew"]["delta_all"]
    assert row["vselect_new"]["n"] + row["vselect_old"]["n"] == row["vselect"]["n"]
    assert "hop_new" in row["readouts_graphnew"] and "old_select" in row["readouts_graphnew"]
    assert "vselect_clean" not in row and "dbal_new" in format_table(info)
    json.dumps(info, allow_nan=False, default=str)  # no NaN reaches the record
    dev = readout_run(result.root / "run.json")
    assert dev["rule"] == GN.GRAPHNEW_RULE and "readouts_graphnew" in dev
    json.dumps(dev, allow_nan=False)

    program = tmp_path / "lift.py"
    program.write_text(LIFT, encoding="utf-8")
    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)
    (path,) = run_transfer(program, [gid], tmp_path / "transfer", sandbox)
    record = json.loads(path.read_text())
    assert record["status"] == "ok" and record["within_limits"] is True and record["tensors"] == ["conv3.bias"]
    assert record["experience"]["source"] == graphnew.EXPERIENCE_SOURCE
    assert record["limits"]["rel_frob_exempt"] == HEAD
    assert record["readouts"]["delta_acc"] == record["readouts_graphnew"]["delta_all"]
    assert record["readout_source"] == GN.GRAPHNEW_RULE and "readouts_clean" not in record
    assert sorted(record["student_norms"]) == ["conv1.weight", "conv2.weight", "conv3.weight"]
    assert record["child_sha256"] == entry["child_sha256"]  # the same program on the same student: the same child
