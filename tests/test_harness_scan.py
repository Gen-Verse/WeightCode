"""The editor harness on scanskill-v1 instances (PREREG E13, the seqskill variant scan_rule): the names agree with the
evalkit, id resolution and refusals, the LEARN view for both conditions, the prompts and tool descriptions and their
leak words, the check on new V_check and the old_check file, a scripted LEARN session and its session_info.

The store is the toy SCAN store (``scan_toy.store_with_students``: tiny students with the gate set to 0), so these
tests check the mechanics, never a number of record. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import scan_toy
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session

from weightcode.analysis.session_info import format_table, session_info
from weightcode.common.identity import params_sha256
from weightcode.data import scan as S
from weightcode.evalkit import scanskill as K
from weightcode.harness import seqskill
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import arm_spec
from weightcode.harness.experience import EXPERIENCE_FILE
from weightcode.harness.instance import read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.leaks import scan_for_tokens
from weightcode.harness.prompts import Budgets, render_system, render_task
from weightcode.harness.retention import kept_rule_record
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import SCAN_CHECK_SETS, SEQSKILL_CHECK_SETS, tool_specs
from weightcode.harness.trial import TrialError, assemble

SCAN = seqskill.SCAN
# Gives jump's token-embedding row walk's row, found by name in the layout: one row of one tensor.
WALK_ROW = '''HYPOTHESIS = "jump's embedding row is unused in compositions, so walk's row makes the old composition run it"
PREDICTED_EFFECT = "commands with jump answer as walk's; old commands without jump are unchanged"
import numpy as np
def update(student, experience):
    ids = {row["name"]: row["id"] for row in student["layout"]["input"]["tokens"]}
    emb = student["params"]["tok_emb.weight"]
    delta = np.zeros(emb.shape, dtype=np.float64)
    delta[ids["jump"]] = emb[ids["walk"]] - emb[ids["jump"]]
    print("rows", ids["jump"], ids["walk"])
    return {"tok_emb.weight": delta}
'''


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """The toy SCAN store with one tiny base per condition and its scanskill-v1 instance (condition -> id)."""
    patcher = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("compose_harness_store")  # no leak word in a mount path
    try:
        scan_toy.store_with_students(tmp, patcher)
        ids = {}
        for condition in S.CONDITIONS:
            (row,) = K.build_scan_instances(condition)
            ids[condition] = row["instance_id"]
        yield ids
    finally:
        patcher.undo()


def _view(store: dict[str, str], condition: str, directory: Path, **kwargs):  # type: ignore[no-untyped-def]
    (directory / "private").mkdir(parents=True)
    return assemble(K.scan_instance_dir(store[condition]), "LEARN", directory, **kwargs)


def _text_files(trial: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in trial.iterdir() if path.suffix in (".json", ".py")}


def test_the_harness_names_agree_with_the_evalkit_rule() -> None:
    assert seqskill.SCAN_RULE == K.SCAN_RULE and seqskill.SCAN_ROOT_NAME == K.ROOT_NAME
    assert seqskill.SCAN_ID_RE.pattern == K.INSTANCE_ID_RE.pattern
    assert seqskill.is_scan_id("n0123abcd") and seqskill.is_seqskill_id("n0123abcd")
    assert not seqskill.is_scan_id("q0123abcd")
    with pytest.raises(seqskill.SeqskillHarnessError):
        seqskill.check_instance_variant({"evidence_rule": seqskill.SEQSKILL_RULE}, SCAN)
    with pytest.raises(seqskill.SeqskillHarnessError):
        seqskill.check_instance_variant({"evidence_rule": seqskill.SCAN_RULE}, None)
    assert seqskill.check_instance_variant({"evidence_rule": seqskill.SCAN_RULE}, SCAN) == SCAN


def test_the_record_resolves_and_the_view_refuses_other_variants(store: dict[str, str], tmp_path: Path) -> None:
    for condition, instance_id in store.items():
        record = read_instance_record(instance_id)
        assert record.is_seqskill and seqskill.is_scan(record.private)
        assert record.private["condition"] == condition
    with pytest.raises(TrialError, match="scan_rule"):
        _view(store, "addprim_jump", tmp_path / "standard", lift_rel_frob=True)
    with pytest.raises(TrialError, match="scan_rule"):
        _view(store, "addprim_jump", tmp_path / "rule", seqskill_variant=seqskill.RULE_ONLY, lift_rel_frob=True)
    with pytest.raises(TrialError, match="rel-Frob"):
        _view(store, "addprim_jump", tmp_path / "unlifted", seqskill_variant=SCAN)


@pytest.mark.parametrize("condition", S.CONDITIONS)
def test_the_learn_view_is_the_evidence_with_the_rules_and_no_target_row(store: dict[str, str], tmp_path: Path,
                                                                         condition: str) -> None:
    view = _view(store, condition, tmp_path / "s", seqskill_variant=SCAN, lift_rel_frob=True)
    trial = tmp_path / "s" / "trial"
    keys = K.evidence_keys(condition)
    with np.load(trial / EXPERIENCE_FILE, allow_pickle=False) as archive:
        assert sorted(archive.files) == sorted(keys)
        shown = {key: archive[key] for key in archive.files}
    instance = K.load_scan_instance(store[condition])
    assert params_sha256(shown) == instance.private["evidence_content_sha256"]
    task = json.loads((trial / "task.json").read_text())
    prim = S.PRIMITIVE_ROW[condition]
    assert ("prim" in task["experience"]) == (prim is not None) and "new" not in task["experience"]
    assert task["experience"]["old"]["count"] == K.N_REPLAY
    assert task["limits"]["max_rel_frob"] is None and task["parent_sha256"] == instance.params_sha256
    assert task["target"]["commands"] == K.TARGET_COMMANDS[condition]
    assert view.seqskill == seqskill.session_record(instance.private, SCAN)
    spec = json.loads((trial / "model_spec.json").read_text())
    assert {row["name"]: row["meaning"] for row in spec["input"]["tokens"]} == dict(S.TOKEN_MEANINGS)
    tokens = K.scan_leak_tokens()
    found = {name: scan_for_tokens(text, tokens) for name, text in _text_files(trial).items()}
    assert not any(found.values()), found

    kept = kept_rule_record(0.005, int(view.seqskill["n_vcheck_old"]), seqskill.check_set_name(view.seqskill))
    assert kept["s_old_floor"] == -5 and kept["check_set"] == "old-command"
    budgets = Budgets(patch_checks=8, python_cells=200, cell_timeout_s=120.0, wall_timeout_s=7200.0)
    system = render_system("LEARN", kept_rule=kept, seqskill=True, seqskill_variant=SCAN)
    prompt = render_task("LEARN", view.task, view.limits, budgets, view.input_files, seqskill=True,
                         seqskill_variant=SCAN)
    assert "score_old is at least -5" in system and "old-command check-set rows" in system
    assert ("`x_prim`, `y_prim`, `pred_prim`: 1 row: the command jump alone" in prompt) == (prim is not None)
    assert "$" not in prompt and "interpretation rules" in prompt
    tools = tool_specs(arm_spec("LEARN").tools, program=True, seqskill=True, s_old_floor=-5, scan=True)
    check = {spec["name"]: spec for spec in tools}["patch_check"]["description"]
    assert SCAN_CHECK_SETS in check and SEQSKILL_CHECK_SETS not in check
    for name, text in {"system": system, "task": prompt, "tools": json.dumps(tools)}.items():
        assert not scan_for_tokens(text, tokens), (name, scan_for_tokens(text, tokens))


def test_the_check_cut_loads_the_old_check_file_by_its_hashes(store: dict[str, str], tmp_path: Path) -> None:
    instance = K.load_scan_instance(store["around_right"])
    record = seqskill.session_record(instance.private, SCAN)
    new = K.load_recorded(instance.private, "V_check")
    parts = seqskill.check_parts(new, record)
    assert (len(parts["y_new"]), len(parts["y_old"])) == (record["n_vcheck_new"], record["n_vcheck_old"])
    with pytest.raises(seqskill.SeqskillHarnessError, match="old_check_sha256"):
        seqskill.check_parts(new, {**record, "old_check_sha256": "0" * 64})
    with pytest.raises(seqskill.SeqskillHarnessError, match="variant"):
        seqskill.check_parts(new, {**record, "variant": seqskill.STANDARD})


@needs_sandbox
def test_a_scan_learn_session_scores_programs_on_the_two_check_files(store: dict[str, str]) -> None:
    instance_id = store["addprim_jump"]
    payload = json.dumps(json.dumps({"walk_row.py": WALK_ROW}))
    cell = ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n")
    steps = [
        ("python", {"code": cell}),
        ("patch_check", {"program_file": "walk_row.py"}),
        ("status", {}),
        ("declare_final", {"program_file": "walk_row.py", "rationale": "the only program"}),
    ]
    result, replies = scripted_session(instance_id, "LEARN", steps, retention_floor_frac=0.005,
                                       seqskill_variant=SCAN, lift_rel_frob=True)
    assert result.status == "complete", result.record.get("selftest")
    _, check, _, declared = replies
    assert not check.is_error, check.payload
    assert check.payload["score"] == check.payload["score_new"] + check.payload["score_old"]
    assert check.payload["tensors"] == ["tok_emb.weight"] and declared.payload["valid"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    assert session["seqskill"]["rule"] == seqskill.SCAN_RULE and session["seqskill"]["variant"] == SCAN
    assert session["kept_rule"] == kept_rule_record(0.005, 1000, "old-command")
    assert session["leak_scan"]["passed"] is True
    assert set(K.scan_leak_tokens()) <= set(session["forbidden_strings"]["text"])
    entry = read_jsonl(paths.checks_log)[0]
    attempt_dir = result.root / "attempts" / str(entry["attempt"])
    child = ArmApplier(session).apply(read_json(attempt_dir / str(entry["patch_file"])), attempt_dir).child
    assert params_sha256(child) == entry["child_sha256"]
    expected = K.make_scan_oracle(instance_id).query(child, "walk_row")  # the evalkit's own joint oracle
    assert (check.payload["score_new"], check.payload["score_old"]) == (expected["score_new"], expected["score_old"])
    ledger = CheckLedger(session, paths)  # rebuilt from session.json alone, as the MCP server builds it
    assert [row["score"] for row in ledger.checked()] == [check.payload["score"]]

    info = session_info(result.root)
    row = info["trajectory"][0]
    readout = K.scan_readout(instance_id, child)
    assert row["readouts_scan"]["joint_points"] == pytest.approx(readout["joint_points"])
    assert row["vselect_new"]["n"] == readout["new"]["n"] and row["vselect_old"]["n"] == readout["old"]["n"]
    assert "gain" in format_table(info)
