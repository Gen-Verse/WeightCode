"""The seqskill-v1 LEARN variant rule_only (PREREG E8): the new operator's rule given in model_spec.json, no row of
it in the experience, everything else the E5 session.

The store is the evalkit toy seqops store (``test_evalkit_seqskill.store``), as in ``test_harness_seqskill``. No test
touches the real store or starts Claude Code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_seqskill import CONDITION, store  # noqa: F401
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256
from weightcode.data import seqops as S
from weightcode.data.store import dataset_json_path, read_dataset_json
from weightcode.evalkit import seqskill as Q
from weightcode.harness import seqskill
from weightcode.harness.experience import EXPERIENCE_FILE, ExperienceError, learn_task_seqskill, seqskill_statement
from weightcode.harness.instance import read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.leaks import HINT_WORDS, scan_trial
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.prompts import PROMPT_DIR, Budgets, render_system, render_task
from weightcode.harness.retention import kept_rule_record
from weightcode.harness.runner import run_session, session_forbidden_strings
from weightcode.harness.session import SessionPaths
from weightcode.harness.trial import TrialError, assemble

RULE_ONLY = seqskill.RULE_ONLY
KIND_WORDS = ("substitution", "permutation")
# The E5 session's prompt templates (s92c8b990b661d23a provenance.prompt_templates_sha256): the standard view keeps them.
E5_TEMPLATE_SHA256 = {
    "system_LEARN_seqskill.md": "972ae9dede30df9017a8a1e285351829d6f1e63f0749d5170004e500b18d0563",
    "task_LEARN_seqskill.md": "b5c97584b11dd6035978777df4e61010ac2bdeb4952c9566e45da9f4de6f5df4",
}
# The E5 task statement as session s92c8b990b661d23a's trial/task.json has it.
E5_STATEMENT = (
    "Make a trained sequence model compute an operator it was never trained on while it keeps the operators it "
    "knows. Each row names one or two operators and holds a string of symbols, and the model writes the result, one "
    "symbol per output slot; model_spec.json gives the row format and the meaning of every token. The token OP_NEW0 "
    "never occurred in the model's training, so the model does not compute it, and what it does is not given: the "
    "experience shows it. The experience file holds labelled rows of the new operator alone and labelled rows of the "
    "old operators from the data the model was trained on, with the model's predictions on both. Write a program "
    "that computes an additive update of the model's weights from the model and its experience, without training "
    "it, so that the updated model answers the rows that use the new operator correctly (task.json rows describes "
    "them) while the old-operator rows it already answers correctly stay correct."
)
E5_SCORING = "Do not score more than 8 distinct candidate patches against the evidence labels in total."
E8_SCORING = ("Do not score more than 8 distinct candidate patches against the evidence labels in total, and scoring a "
              "patch against answers computed from the new operator's rule counts toward the same limit of 8.")
# Reads the new operator's rule from the layout and gives its token the embedding row of the old table map.
RULE_COPY = '''HYPOTHESIS = "the new operator is a symbol table like OP_SUBA, so its token can borrow that row"
PREDICTED_EFFECT = "the new rows change and some old-operator rows may break"
import numpy as np
def update(student, experience):
    tokens = {row["name"]: row for row in student["layout"]["input"]["tokens"]}
    new = tokens["OP_NEW0"]
    table = new["meaning"].split("table = ")[1].split("]")[0] + "]"
    emb = student["params"]["tok_emb.weight"]
    delta = np.zeros(emb.shape, dtype=np.float64)
    delta[new["id"]] = emb[tokens["OP_SUBA"]["id"]] - emb[new["id"]]
    print("rule", table, "keys", ",".join(sorted(experience)))
    return {"tok_emb.weight": delta}
'''


@pytest.fixture
def seq_id(store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:  # noqa: F811
    """The toy seqskill instance built, with a runs root for sessions."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(runs))
    (row,) = Q.build_seqskill_instances(CONDITION)
    return str(row["instance_id"])


def _record_table() -> list[int]:
    """The condition's rule table, read here independently of the harness."""
    return list(read_dataset_json(S.DATASET, CONDITION)["condition"]["rule"]["table"])


def _views(seq_id: str, tmp_path: Path):  # type: ignore[no-untyped-def]
    root = Q.seqskill_instance_dir(seq_id)
    views = {}
    for name, variant in (("standard", None), ("rule", RULE_ONLY)):
        session_dir = tmp_path / name
        (session_dir / "private").mkdir(parents=True)
        views[name] = assemble(root, "LEARN", session_dir, seqskill_variant=variant)
    return root, views["standard"], views["rule"]


def test_the_rule_only_view_shows_the_old_part_and_states_the_rule(seq_id: str, tmp_path: Path) -> None:
    root, standard, view = _views(seq_id, tmp_path)
    instance = load_instance(root)
    with np.load(view.trial_dir / EXPERIENCE_FILE, allow_pickle=False) as archive:
        shown = {key: archive[key] for key in archive.files}
    assert sorted(shown) == sorted(seqskill.RULE_ONLY_EXPERIENCE_KEYS) and "x_new" not in shown
    assert all(np.array_equal(shown[key], instance.evidence[key]) for key in shown)  # E5's 64 replay rows
    record = view.experience
    assert record is not None and record["source"] == seqskill.RULE_ONLY_EXPERIENCE_SOURCE
    assert record["variant"] == RULE_ONLY and record["n_new"] == 0 and record["n_new_withheld"] == Q.N_NEW
    assert record["n_old"] == Q.N_REPLAY and record["content_sha256"] == params_sha256(shown)
    assert record["evidence_content_sha256"] == instance.private["evidence_content_sha256"]
    assert view.evidence_content_sha256 == params_sha256(shown) and view.evidence_source == record["source"]
    assert sorted(path.name for path in view.trial_dir.iterdir()) == sorted(path.name for path in
                                                                          standard.trial_dir.iterdir())

    # model_spec.json: one token meaning differs, and it carries the table of the condition record
    table = _record_table()
    spec, base = read_json(view.trial_dir / "model_spec.json"), read_json(standard.trial_dir / "model_spec.json")
    rows = {row["name"]: row for row in spec["input"]["tokens"]}
    expected = ("operator, symbol map: s -> table[s], table = " + str(table)
                + "; absent from all training rows (its embedding row is untrained)")
    assert rows["OP_NEW0"]["meaning"] == expected and rows["OP_NEW0"]["id"] == S.TOKEN_ID["OP_NEW0"]
    assert rows["OP_SUBA"]["meaning"].startswith(seqskill.SYMBOL_MAP_MEANING)  # the same form as an old table map
    for row in base["input"]["tokens"]:
        if row["name"] == "OP_NEW0":
            row["meaning"] = expected
    assert spec == base
    assert rows["OP_NEW1"]["meaning"].startswith(seqskill.RESERVED_MEANING_PREFIX)  # other reserved tokens stay
    assert (root / "model_spec.json").read_bytes() != (view.trial_dir / "model_spec.json").read_bytes()

    # task.json: the statement, new_operator.facts, the experience section and two length phrases differ
    task, e5 = view.task, standard.task
    assert sorted(task) == sorted(e5)
    assert sorted(key for key in task if task[key] != e5[key]) == ["experience", "new_operator", "patch_check", "rows",
                                                                   "statement"]
    assert [key for key in task["rows"] if task["rows"][key] != e5["rows"][key]] == ["new_operator_rows"]
    assert [key for key in task["patch_check"] if task["patch_check"][key] != e5["patch_check"][key]] == ["readout"]
    assert task["statement"] == seqskill_statement("OP_NEW0", RULE_ONLY)
    assert "Its rule is given in model_spec.json, and the experience holds no row of it." in task["statement"]
    assert "the experience shows it" not in task["statement"] and "new operator alone and" not in task["statement"]
    assert task["new_operator"]["facts"].endswith("Its rule is given in model_spec.json, and the experience holds "
                                                  "no row of it")
    assert {k: v for k, v in task["new_operator"].items() if k != "facts"} == {
        k: v for k, v in e5["new_operator"].items() if k != "facts"}
    assert "new" not in task["experience"] and task["experience"]["old"] == e5["experience"]["old"]
    assert task["experience"]["targets"] == e5["experience"]["targets"]
    assert task["experience"]["ids"] == {"keys": ["split_ids_old"], "description": "one opaque id per example, in "
                                                                                    "x_old order"}
    assert task["experience"]["predictions"].startswith("pred_old holds the model's argmax")
    assert "x_new" not in json.dumps(task) and "pred_new" not in json.dumps(task)
    assert view.limits == standard.limits and view.shown_model_sha256 == standard.shown_model_sha256

    # the session record: the standard one plus the variant and the rule's provenance
    assert standard.seqskill is not None and "variant" not in standard.seqskill
    assert view.seqskill == {**standard.seqskill, "variant": RULE_ONLY, "new_operator_rule": {
        "table": table, "condition_dataset_json_sha256": instance.private["condition_dataset_json_sha256"]}}
    assert hashlib.sha256(dataset_json_path(S.DATASET, CONDITION).read_bytes()).hexdigest() == \
        instance.private["condition_dataset_json_sha256"]


def test_the_rule_only_view_names_the_lengths_the_new_evidence_named(seq_id: str, tmp_path: Path) -> None:
    _, standard, view = _views(seq_id, tmp_path)
    budgets = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=5400.0)
    e5_rows, e5_readout = "at the lengths of the new evidence", "at the lengths of the new examples"
    rows, readout = view.task["rows"]["new_operator_rows"], view.task["patch_check"]["readout"]
    assert rows == standard.task["rows"]["new_operator_rows"].replace(e5_rows, "at lengths 4 to 8")
    assert readout == standard.task["patch_check"]["readout"].replace(e5_readout, "at lengths 4 to 8")
    assert "at lengths 4 to 8" in rows and "at lengths 4 to 8" in readout
    prompt = render_task("LEARN", view.task, view.limits, budgets, view.input_files, seqskill=True,
                         seqskill_variant=RULE_ONLY)
    for text in (json.dumps(view.task), prompt):
        assert "new evidence" not in text and "new examples" not in text
    assert readout in prompt
    # the standard view keeps E5's phrases, in task.json and in the rendered prompt
    assert e5_rows in standard.task["rows"]["new_operator_rows"] and e5_readout in standard.task["patch_check"][
        "readout"]
    e5_prompt = render_task("LEARN", standard.task, standard.limits, budgets, standard.input_files, seqskill=True)
    assert e5_readout in e5_prompt and "at lengths 4 to 8" not in e5_prompt
    assert standard.task["rows"] == load_instance(Q.seqskill_instance_dir(seq_id)).task["rows"]
    # a change in E5's wording, or in the stated check lengths, fails loudly instead of passing through
    instance = load_instance(Q.seqskill_instance_dir(seq_id))
    arrays, record = seqskill.experience(instance)
    _, cut = seqskill.rule_only_experience(arrays, record)
    limits = Q.seqskill_program_limits(instance.task["limits"], len(instance.params))
    reworded = json.loads(json.dumps(instance.task))
    reworded["rows"]["new_operator_rows"] = reworded["rows"]["new_operator_rows"].replace(e5_rows, "at short lengths")
    with pytest.raises(ExperienceError, match="rows.new_operator_rows"):
        learn_task_seqskill(reworded, instance.params_sha256, limits, cut, RULE_ONLY)
    relengthed = json.loads(json.dumps(instance.task))
    relengthed["patch_check"]["parts"]["new"] = "held-out rows of the new operator alone, lengths 4 to 9"
    with pytest.raises(ExperienceError, match="lengths 4 to 8"):
        learn_task_seqskill(relengthed, instance.params_sha256, limits, cut, RULE_ONLY)


def test_the_rule_only_prompts_and_leak_scan(seq_id: str, tmp_path: Path) -> None:
    _, _, view = _views(seq_id, tmp_path)
    budgets = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=5400.0)
    prompt = render_task("LEARN", view.task, view.limits, budgets, view.input_files, seqskill=True,
                         seqskill_variant=RULE_ONLY)
    assert "$" not in prompt and "x_new" not in prompt and "split_ids_new" not in prompt
    assert "labelled examples of the old operators with the model's predictions, and no row of the new" in prompt
    assert view.task["statement"] in prompt and "Output symbols by index" in prompt
    with pytest.raises(KeyError):  # the standard template needs the new part, which this task.json does not have
        render_task("LEARN", view.task, view.limits, budgets, view.input_files, seqskill=True)
    rule = kept_rule_record(0.005, 4000, seqskill.CHECK_SET)
    system = render_system("LEARN", seqskill=True, kept_rule=rule, seqskill_variant=RULE_ONLY)
    assert E8_SCORING in " ".join(system.split()) and "score_old below -20 is never kept" in " ".join(system.split())

    record = read_instance_record(seq_id)
    forbidden = session_forbidden_strings(record, seqskill_variant=RULE_ONLY)
    standard = session_forbidden_strings(record)
    assert CONDITION in forbidden.text and "seqops" in forbidden.text
    assert not set(KIND_WORDS) & set(forbidden.text) and set(KIND_WORDS) <= set(standard.text)
    assert set(standard.text) - set(forbidden.text) == set(KIND_WORDS)
    assert not {word.capitalize() for word in KIND_WORDS} & set(forbidden.selftest)
    assert seqskill.leak_words(record.private, RULE_ONLY) == [
        word for word in seqskill.leak_words(record.private) if word not in KIND_WORDS]
    assert not scan_trial(view.trial_dir, forbidden.text)
    texts = (prompt + system).lower()
    assert not [word for word in (*forbidden.text, *HINT_WORDS) if word.lower() in texts]


def test_the_rule_only_templates_change_only_what_the_variant_changes() -> None:
    for name, digest in E5_TEMPLATE_SHA256.items():  # the E5 templates themselves are untouched
        assert hashlib.sha256((PROMPT_DIR / name).read_bytes()).hexdigest() == digest
    e5, e8 = render_system("LEARN", seqskill=True), render_system("LEARN", seqskill=True, seqskill_variant=RULE_ONLY)
    assert " ".join(e8.split()) == " ".join(e5.split()).replace(E5_SCORING, E8_SCORING)
    e5_lines, e8_lines = e5.splitlines(), e8.splitlines()
    (at,) = [i for i, line in enumerate(e5_lines) if line.startswith("evidence labels in total.")]
    assert e8_lines[:at] == e5_lines[:at] and e8_lines[at + 2:] == e5_lines[at + 1:]  # one line became two
    assert max(len(line) for line in e8_lines) <= 116
    e5_task = (PROMPT_DIR / "task_LEARN_seqskill.md").read_text(encoding="utf-8").split("\n- ")
    e8_task = (PROMPT_DIR / "task_LEARN_seqskill_rule.md").read_text(encoding="utf-8").split("\n- ")
    changed = [index for index, (a, b) in enumerate(zip(e5_task, e8_task)) if a != b]
    assert len(e5_task) == len(e8_task) and len(changed) == 1
    assert e8_task[changed[0]].startswith("`experience.npz`") and "$n_new" not in e8_task[changed[0]]
    with pytest.raises(ValueError, match="seqskill-v1 session only"):
        render_system("LEARN", seqskill_variant=RULE_ONLY)
    with pytest.raises(ValueError, match="unknown seqskill variant"):
        render_system("LEARN", seqskill=True, seqskill_variant="rules")


def test_the_standard_view_is_unchanged(seq_id: str, tmp_path: Path) -> None:
    assert seqskill_statement("OP_NEW0") == E5_STATEMENT == seqskill_statement("OP_NEW0", seqskill.STANDARD)
    root = Q.seqskill_instance_dir(seq_id)
    views = []
    for name, variant in (("none", None), ("named", seqskill.STANDARD)):
        (tmp_path / name / "private").mkdir(parents=True)
        views.append(assemble(root, "LEARN", tmp_path / name, seqskill_variant=variant))
    assert views[0].files == views[1].files and views[0].seqskill == views[1].seqskill
    assert views[0].task["statement"] == E5_STATEMENT and "variant" not in views[0].seqskill
    assert sorted(views[0].task["experience"]) == ["ids", "new", "old", "predictions", "targets"]
    instance = load_instance(root)
    assert seqskill.session_record(instance.private) == views[0].seqskill
    with np.load(views[0].trial_dir / EXPERIENCE_FILE, allow_pickle=False) as archive:
        assert sorted(archive.files) == sorted(seqskill.EXPERIENCE_KEYS)
    spec = read_json(views[0].trial_dir / "model_spec.json")
    assert spec == {k: v for k, v in read_json(root / "model_spec.json").items() if k != "model_id"}


def test_the_rule_only_view_refuses_what_it_cannot_derive(seq_id: str, tmp_path: Path) -> None:
    root = Q.seqskill_instance_dir(seq_id)
    instance = load_instance(root)
    private = instance.private
    with pytest.raises(seqskill.SeqskillHarnessError, match="hashes to"):
        seqskill.new_operator_rule({**private, "condition_dataset_json_sha256": "0" * 64})
    with pytest.raises(seqskill.SeqskillHarnessError, match="not bound"):
        seqskill.new_operator_rule({**private, "new_operator": "OP_NEW1"})
    rule = seqskill.new_operator_rule(private)
    assert rule["table"] == _record_table() and sorted(rule["table"]) == list(range(S.N_SYMBOLS))
    spec = read_json(root / "model_spec.json")
    once = seqskill.rule_only_model_spec(spec, private, rule)
    with pytest.raises(seqskill.SeqskillHarnessError, match="not as a reserved token"):
        seqskill.rule_only_model_spec(once, private, rule)  # a stated rule is never restated
    arrays, record = seqskill.experience(instance)
    with pytest.raises(ExperienceError, match="experience record of that variant"):
        learn_task_seqskill(instance.task, instance.params_sha256, Q.seqskill_program_limits(
            instance.task["limits"], len(instance.params)), record, RULE_ONLY)
    _, cut = seqskill.rule_only_experience(arrays, record)
    with pytest.raises(seqskill.SeqskillHarnessError, match="checked standard experience"):
        seqskill.rule_only_experience({key: arrays[key] for key in seqskill.RULE_ONLY_EXPERIENCE_KEYS}, cut)
    with pytest.raises(seqskill.SeqskillHarnessError, match="states no rule"):
        seqskill.session_record(private, None, rule)
    with pytest.raises(seqskill.SeqskillHarnessError, match="records the rule"):
        seqskill.session_record(private, RULE_ONLY)
    (tmp_path / "s" / "private").mkdir(parents=True)
    with pytest.raises(TrialError, match="unknown seqskill variant"):
        assemble(root, "LEARN", tmp_path / "s", seqskill_variant="rules")
    with pytest.raises(ValueError, match="unknown seqskill variant"):
        run_session(seq_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True, seqskill_variant="rules")
    assert not list(Path(os.environ["WEIGHTCODE_RUNS"]).rglob("session.json"))  # refused before any write


@needs_sandbox
def test_a_rule_only_session_gives_the_program_the_rule_and_the_old_rows(seq_id: str) -> None:
    payload = json.dumps(json.dumps({"rule.py": RULE_COPY}))
    cell = ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n")
    steps = [("python", {"code": cell}), ("patch_check", {"program_file": "rule.py"}),
             ("declare_final", {"program_file": "rule.py", "rationale": "the only program"})]
    result, replies = scripted_session(seq_id, "LEARN", steps, retention_floor_frac=0.005,
                                       seqskill_variant=RULE_ONLY)
    assert result.status == "complete", result.record.get("infra_reasons")
    _, check, declared = replies
    assert not check.is_error, check.payload
    stdout = check.payload["program_stdout"]
    assert stdout.startswith(f"rule {_record_table()} keys ")  # the program read the rule from its layout
    assert stdout.split("keys ")[1].split() == [",".join(sorted(seqskill.RULE_ONLY_EXPERIENCE_KEYS))]
    assert declared.payload["valid"]
    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    assert session["seqskill"]["variant"] == RULE_ONLY and session["leak_scan"]["passed"] is True
    assert session["experience"]["source"] == seqskill.RULE_ONLY_EXPERIENCE_SOURCE
    assert session["evidence_source"] == seqskill.RULE_ONLY_EXPERIENCE_SOURCE
    assert CONDITION in session["forbidden_strings"]["text"]
    assert not set(KIND_WORDS) & set(session["forbidden_strings"]["text"])
    assert session["kept_rule"] == kept_rule_record(0.005, 4000, seqskill.CHECK_SET)
    assert paths.system_prompt.read_text(encoding="utf-8") == render_system(
        "LEARN", seqskill=True, kept_rule=session["kept_rule"], seqskill_variant=RULE_ONLY)
    entry = read_jsonl(paths.checks_log)[0]
    ledger = CheckLedger(session, paths)  # rebuilt from session.json alone, as the MCP server builds it
    assert [row["score"] for row in ledger.checked()] == [check.payload["score"]] == [entry["score"]]
