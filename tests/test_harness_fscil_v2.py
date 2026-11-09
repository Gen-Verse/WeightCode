"""The E9 v2 harness (PREREG "E9 v2"): check score fscil-cb-v1 and kept rule fscil-retain-v2 through patch_check, the
ledger, the final selection, the prompts, the library and the lineages T2, F2 and C2.

The score functions are ``weightcode.evalkit.fscil``'s (``cb_check_rows``, ``cb_check_score``, ``retain_v2_floor``,
``retain_v2_pick``); these tests check that the harness feeds them the check's rows in the oracle's order, shows what
they return, keeps what ``retain_v2_pick`` picks, and leaves E9's (v1) sessions as they were.

The store is the toy cifar100_fscil store (``fscil_toy``): a random ResNet-20 student, so near chance. The fixture
programs carry toy check images: ``imprint_program`` (``test_harness_fscil``) imprints the new classes' head rows and
breaks base-class images (S > 0, not eligible); ``cover_program`` imprints the head rows of some base classes on their
old_check images (eligible, S > 0), with ``new`` also the new classes' rows, and flips the base rows at session 6 when
asked (so a check refuses it there and the frozen lineage applies it). They are test fixtures, not methods.
Every directory lives under a neutral pytest tmp dir, as in ``test_harness_fscil``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fscil_toy import ToyFscil, build_toy_fscil_store
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session
from test_harness_fscil import imprint_program, needs_neutral_repo, program_cell, toy_floor, toy_source

from weightcode.analysis.session_info import format_table, session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params, save_params
from weightcode.data import store as data_store
from weightcode.evalkit import fscil as FS
from weightcode.evalkit.patch_check import BudgetExceeded, predict
from weightcode.harness import fscil
from weightcode.harness import fscil_lineage as L
from weightcode.harness.applying import SessionConfigError
from weightcode.harness.arms import arm_spec
from weightcode.harness.finalize import FINAL_RULE_FSCIL_V2, FinalizeError, select_final_v2
from weightcode.harness.instance import read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.library import LIBRARY_INDEX_FILE
from weightcode.harness.prompts import PROMPT_DIR, Budgets, render_system, render_task
from weightcode.harness.runner import SANDBOX_PYTHON, run_session
from weightcode.harness.scripted import ScriptedLauncher, call_and_record
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import tool_specs
from weightcode.harness.trial import assemble
from weightcode.sandbox.program import ProgramSandbox

TOY_ROWS = (120, 120)  # old_check and old_select rows of the toy store (2 per base class)
COVER_TEMPLATE = '''HYPOTHESIS = "a class is recognised when its head row points at the mean feature of its images"
PREDICTED_EFFECT = "the check images of the imprinted classes become correct"
import numpy as np
IMAGES = np.frombuffer(bytes.fromhex("{images}"), dtype=np.uint8).reshape({shape})
LABELS = np.array({labels}, dtype=np.int64)
BASE = {base}
NEW = {new}
SPOIL_FIRST_CLASS = {spoil}


def update(student, experience):
    params = student["params"]
    head = student["layout"]["layers"][-1]["weight"]
    weight = params[head]
    n_out, width = weight.shape
    probe = dict(params)
    basis = np.zeros((n_out, width), dtype=np.float32)
    basis[:width] = np.eye(width, dtype=np.float32)
    basis[width:, 0] = 1.0
    probe[head] = basis
    delta = np.zeros((n_out, width), dtype=np.float64)
    new = sorted(set(int(c) for c in experience["y_new"])) if NEW else []
    classes = list(BASE) + new
    for c in classes:
        z = np.asarray(student["forward"](probe, IMAGES[LABELS == c]), dtype=np.float64)[:, :width]
        mean = (z / np.linalg.norm(z, axis=1, keepdims=True)).mean(axis=0)
        delta[c] = mean / np.linalg.norm(mean) - weight[c]
    if new and new[0] == SPOIL_FIRST_CLASS:
        for c in BASE:
            delta[c] = -2.0 * weight[c]
    print("rows", classes)
    return {{head: delta}}
'''
SMALL_BASE = [0, 1, 2]  # three base classes: on the toy, an eligible program with a small S (1.5 points)
SPOIL = 80  # the first class of session 6


def cover_program(base: list[int], new: bool, spoil: int | None = None) -> str:
    """The head rows of ``base`` imprinted on their toy old_check images and, with ``new``, those of the experience's
    new classes on their toy V_check images (module docstring). At the session whose first new class is ``spoil`` the
    rows of ``base`` are flipped instead, which breaks their images (B_c far below any floor)."""
    old = data_store.load_split(FS.DATASET, "old_check")
    keep = np.isin(old["y"], base)
    xs, ys = [old["x"][keep]], [old["y"][keep]]
    if new:
        v_check = data_store.load_split(FS.DATASET, "V_check")
        inc = v_check["y"] >= FS.N_BASE_CLASSES
        xs.append(v_check["x"][inc])
        ys.append(v_check["y"][inc])
    images = np.ascontiguousarray(np.concatenate(xs)).astype(np.uint8)
    return COVER_TEMPLATE.format(images=images.tobytes().hex(), shape=(-1, *images.shape[1:]),
                                 labels=np.concatenate(ys).astype(int).tolist(), base=list(base), new=new,
                                 spoil=spoil)


def child_of(program: str, instance: Any) -> dict[str, np.ndarray]:
    """The child of ``program`` on the instance's student, computed in this process as the harness applies deltas."""
    namespace: dict[str, Any] = {}
    exec(compile(program, "<fixture>", "exec"), namespace)  # noqa: S102 - a fixture program of this file
    arrays, _ = fscil.experience(instance)
    student = {"params": dict(instance.params), "forward": instance.forward,
               "layout": read_json(instance.instance_dir / "model_spec.json")}
    child = dict(instance.params)
    for name, delta in namespace["update"](student, arrays).items():
        child[name] = (instance.params[name].astype(np.float64) + delta).astype(np.float32)
    return child


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ToyFscil]:
    with pytest.MonkeyPatch.context() as patch:
        yield build_toy_fscil_store(tmp_path_factory.mktemp("hs"), patch)


@pytest.fixture(scope="module")
def cid(toy: ToyFscil) -> str:
    return str(FS.build_fscil_instance(2, lineage="T2", run_id="r0")["instance_id"])


@pytest.fixture(scope="module")
def rule(toy: ToyFscil, cid: str) -> dict[str, Any]:
    return fscil.v2_rule_for(read_instance_record(cid).private)


def _paragraphs(text: str) -> list[str]:
    return [block for block in text.split("\n\n") if block.strip()]


def test_the_v2_names_and_kept_rule_agree_with_the_evalkit_rule(toy: ToyFscil, cid: str,
                                                                rule: dict[str, Any]) -> None:
    assert fscil.KEPT_RULE_V2 == FS.KEPT_RULE_V2 and fscil.CHECK_SCORE_V2 == FS.CHECK_SCORE_RULE
    assert fscil.v2_base_rows() == TOY_ROWS
    assert rule == fscil.kept_rule_v2(*TOY_ROWS) == {"name": "fscil-retain-v2", "check_score": "fscil-cb-v1",
                                                    "r_points": -1.0, "n_c": 120, "n_s": 120,
                                                    "check_set": "base-class"}
    assert fscil.session_kept_v2({"fscil": {}, "kept_rule": rule}) == rule
    assert fscil.session_kept_v2({"fscil": {}, "kept_rule": {"name": "joint-retain-v1"}}) is None
    assert fscil.session_kept_v2({}) is None
    for bad, message in (({"fscil": {}, "kept_rule": {**rule, "r_points": -2.0}}, "is not"),
                         ({"fscil": {}, "kept_rule": {**rule, "n_c": 0}}, "positive base-row counts"),
                         ({"fscil": {}, "kept_rule": {k: v for k, v in rule.items() if k != "n_s"}}, "malformed"),
                         ({"kept_rule": rule}, "for an fscil-v1 instance")):
        with pytest.raises(fscil.FscilHarnessError, match=message):
            fscil.session_kept_v2(bad)
    assert fscil.check_version(None) == "v1" and fscil.check_version("v2") == "v2"
    with pytest.raises(fscil.FscilHarnessError, match="one of"):
        fscil.check_version("v3")
    real = fscil.kept_rule_v2(1800, 1800)  # the real store's numbers, as the prompts state them
    numbers = fscil.v2_numbers(real)
    assert numbers == {"R": "-1.0", "R_shift": "-1.11", "n_c": "1800", "n_s": "1800", "d_zero": "0.041"}
    assert FS.retain_v2_floor(0.040, -1.0, 1800, 1800) < 0.0 == FS.retain_v2_floor(0.042, -1.0, 1800, 1800)
    assert FS.retain_v2_floor(0.0) == pytest.approx(-1.11)
    assert fscil.v2_floor_formula(real) == "min(0, R - 0.11 + 164.5 sqrt(d_c / 1800 + d_c / 1800)) with R = -1.0"


def _expected_v2(instance: Any, child: dict[str, np.ndarray], rule: dict[str, Any]) -> dict[str, Any]:
    """fscil-cb-v1 of ``child`` from predictions computed here (session mask, the session's check rows)."""
    private = instance.private
    t = int(private["session"])
    parts = fscil.load_check_parts(fscil.session_record(private), private["vcheck_path"], private["vcheck_sha256"])
    masked = FS.masked_forward(instance.forward, t)
    x = np.concatenate([parts["x_new"], parts["x_old"]])
    y, part = FS.cb_check_rows(parts["y_new"], parts["y_old"], t)
    score = FS.cb_check_score(predict(masked, instance.params, x, 64), predict(masked, child, x, 64), y, part, t)
    score["floor_c"] = FS.retain_v2_floor(score["d_c"], rule["r_points"], rule["n_c"], rule["n_s"])
    return score


def test_the_v2_oracle_adds_the_class_balanced_score_to_the_joint_rows(toy: ToyFscil, cid: str,
                                                                       rule: dict[str, Any]) -> None:
    parent = toy.root / "v2_parent.npz"
    save_params(parent, toy.base_params)
    four = FS.build_fscil_instance(3, lineage="X2", parent_params=parent, parent_instance=cid, run_id="r0")
    four = FS.build_fscil_instance(4, lineage="X2", parent_params=parent, parent_instance=four["instance_id"],
                                   run_id="r0")
    for instance_id in (cid, str(four["instance_id"])):
        instance = load_instance(fscil.instance_dir(instance_id))
        programs = {"imprint": imprint_program(), "small": cover_program(SMALL_BASE, new=False),
                    "cover": cover_program(list(range(FS.N_BASE_CLASSES)), new=True)}
        oracle = fscil.make_cb_oracle(instance_id, len(programs), rule)
        joint = FS.make_fscil_oracle(instance_id, len(programs))
        entries = []
        for name, program in programs.items():
            child = child_of(program, instance)
            entry, row = oracle.query(child, name), joint.query(child, name)
            assert {key: entry[key] for key in row} == row  # the row-level fields are the joint oracle's
            expected = _expected_v2(instance, child, rule)
            assert entry["check_score"] == "fscil-cb-v1"
            for key in ("S", "S_new", "S_prev", "S_base", "B_c", "d_c", "floor_c", "fixed_base", "broken_base",
                        "n_base", "n_seen_classes", "per_class"):
                assert entry[key] == expected[key], key
            assert entry["S"] == pytest.approx(entry["S_new"] + entry["S_prev"] + entry["S_base"])
            assert entry["eligible"] == (entry["B_c"] >= entry["floor_c"]) and entry["n_base"] == rule["n_c"]
            assert len(entry["per_class"]) == FS.n_seen(int(instance.private["session"]))
            json.dumps(entry, allow_nan=False)
            entries.append({**entry, "patch_id": name})
        by_name = {entry["patch_id"]: entry for entry in entries}
        assert by_name["imprint"]["S"] > 0 and not by_name["imprint"]["eligible"]  # breaks base images
        assert by_name["small"]["eligible"] and 0 < by_name["small"]["S"]
        pick = select_final_v2(entries, rule)
        index = FS.retain_v2_pick(entries, rule["r_points"], rule["n_c"], rule["n_s"])
        assert pick == {"kind": "checked", "patch_id": entries[index]["patch_id"],
                        "query_index": entries[index]["query_index"]}
        assert select_final_v2([entries[0]], rule)["kind"] == "base"  # an ineligible program alone: the student
        prev = FS.prev_classes(int(instance.private["session"]))  # empty at session 2, sessions 2 and 3 at 4
        assert all(by_name["cover"]["per_class"][str(c)]["n"] > 0 for c in prev)
        with pytest.raises(BudgetExceeded):
            oracle.query(instance.params, "more")
        again = fscil.make_cb_oracle(instance_id, 2, rule)
        again.query(instance.params, "a")
        with pytest.raises(ValueError, match="already checked"):
            again.query(instance.params, "a")
    with pytest.raises(fscil.FscilHarnessError, match="base-class rows"):
        fscil.make_cb_oracle(cid, 1, fscil.kept_rule_v2(1800, 1800))
    with pytest.raises(FinalizeError, match="not in query order"):
        select_final_v2([{**entries[1], "query_index": 1}, {**entries[0], "query_index": 0}], rule)


def test_the_v2_prompts_change_only_the_scores_and_the_kept_rule(toy: ToyFscil, cid: str, rule: dict[str, Any],
                                                                  tmp_path: Path) -> None:
    v1 = (PROMPT_DIR / "system_LEARN_fscil.md").read_text(encoding="utf-8")
    v2 = render_system("LEARN", fscil=True, kept_rule=rule)
    one, two = _paragraphs(v1), _paragraphs(v2)
    changed = [index for index, (a, b) in enumerate(zip(one, two)) if a != b]
    assert len(one) == len(two) and len(changed) == 1 and two[changed[0] - 1] == "## What is kept"
    stated = " ".join(two[changed[0]].split())
    for phrase in ("whose B_c is at least their own floor_c, the update of the one with the highest S is kept",
                   "floor_c = min(0, R - 0.11 + 164.5 sqrt(d_c / 120 + d_c / 120)) with R = -1.0",
                   "the program's own base flip rate", "from -1.11 at d_c = 0", "S_new, S_prev and S_base",
                   "class-balanced accuracy", "If no checked program with B_c of at least its floor_c has an S > 0"):
        assert phrase in stated, phrase
    assert max(len(line) for line in v2.splitlines()) <= 116 and fscil.scan(v2) == []
    real = " ".join(render_system("LEARN", fscil=True, kept_rule=fscil.kept_rule_v2(1800, 1800)).split())
    assert "sqrt(d_c / 1800 + d_c / 1800)) with R = -1.0" in real and "to 0 at d_c = 0.041 and above" in real
    with pytest.raises(ValueError, match="fscil-v1 session"):
        render_system("LEARN", kept_rule=rule)

    views = {}
    for name, kept in (("one", None), ("two", rule)):
        (tmp_path / name / "private").mkdir(parents=True)
        views[name] = assemble(fscil.instance_dir(cid), "LEARN", tmp_path / name, fscil_kept_rule=kept)
    task_one, task_two = (read_json(views[name].trial_dir / "task.json") for name in ("one", "two"))
    assert task_two["patch_check"] == {**task_one["patch_check"], "readout": fscil.readout_v2(rule),
                                       "final_selection": fscil.final_selection_v2(rule)}
    assert {k: v for k, v in task_two.items() if k != "patch_check"} == {
        k: v for k, v in task_one.items() if k != "patch_check"}
    assert fscil.scan(json.dumps(task_two)) == []
    budgets = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=5400.0)
    texts = {name: render_task("LEARN", views[name].task, views[name].limits, budgets, fscil=True)
             for name in ("one", "two")}
    one, two = _paragraphs(texts["one"]), _paragraphs(texts["two"])
    changed = [index for index, (a, b) in enumerate(zip(one, two)) if a != b]
    assert len(one) == len(two) and [two[index - 1] for index in changed] == ["## Patch-check readout",
                                                                              "## Final selection"]


def test_the_v2_tools_state_the_v2_scores_and_nothing_else_changes(rule: dict[str, Any]) -> None:
    tools = arm_spec("LEARN").tools
    one = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True, fscil=True)}
    two = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True, fscil=True, fscil_v2=rule)}
    assert {k for k in one if one[k] != two[k]} == {"patch_check", "status", "declare_final"}
    head, tail = one["patch_check"].split("and the updated classifier is evaluated on")
    assert two["patch_check"].startswith(head) and two["patch_check"].endswith(tail.split("(S_joint), ")[1])
    for phrase in ("S (the change of the class-balanced accuracy over the classes seen so far)", "B_c", "d_c",
                   "floor_c = min(0, R - 0.11 + 164.5 sqrt(d_c / 120 + d_c / 120)) with R = -1.0",
                   "eligible (B_c >= floor_c)"):
        assert phrase in two["patch_check"], phrase
    assert "score_old" not in two["patch_check"] + two["status"] + two["declare_final"]
    assert "the eligible checked program (B_c >= floor_c) with the highest S" in two["declare_final"]
    for kwargs, message in (({"fscil_v2": rule}, "fscil-v1 session"),
                            ({"fscil": True, "fscil_v2": rule, "s_old_floor": 0}, "only kept rule")):
        with pytest.raises(ValueError, match=message):
            tool_specs(tools, program=True, **kwargs)


def test_a_v2_session_refuses_a_floor_and_another_version_before_any_write(toy: ToyFscil, cid: str) -> None:
    runs = Path(os.environ["WEIGHTCODE_RUNS"])
    before = sorted(path.name for path in runs.iterdir())
    refusals = [
        ("retention floor", {"fscil_version": "v2", "retention_floor_rows": 0}),
        ("retention floor", {"fscil_version": "v2", "retention_floor_frac": 0.01}),
        ("one of", {"fscil_version": "v3"}),
    ]
    for message, kwargs in refusals:
        with pytest.raises(ValueError, match=message):
            run_session(cid, "LEARN", 0, effort="low", wall_timeout_s=60.0, **kwargs)
    with pytest.raises(ValueError, match="program arm"):
        run_session(cid, "M2", 0, effort="low", wall_timeout_s=60.0, fscil_version="v2")
    assert sorted(path.name for path in runs.iterdir()) == before  # nothing written


@pytest.fixture(scope="module")
def v2_session(toy: ToyFscil, cid: str) -> tuple[Any, list[Any]]:
    """A scripted E9 v2 session at session 2: an ineligible program with the higher S, then an eligible one."""
    programs = {"imprint.py": imprint_program(), "small.py": cover_program(SMALL_BASE, new=False)}
    steps = [("python", {"code": program_cell(programs)}), ("patch_check", {"program_file": "imprint.py"}),
             ("patch_check", {"program_file": "small.py"}), ("status", {}),
             ("declare_final", {"program_file": "imprint.py", "rationale": "the higher S"})]
    return scripted_session(cid, "LEARN", steps, fscil_version="v2")


@needs_sandbox
@needs_neutral_repo
def test_a_scripted_v2_session_shows_the_v2_scores_and_keeps_by_fscil_retain_v2(
        toy: ToyFscil, cid: str, rule: dict[str, Any], v2_session: tuple[Any, list[Any]]) -> None:
    result, replies = v2_session
    assert result.status == "complete", result.record.get("infra_reasons")
    _, imprint, small, status, declared = replies
    for reply in (imprint, small):
        assert not reply.is_error, reply.payload
        assert set(fscil.V2_SHOWN_FIELDS) <= set(reply.payload)
        assert not {"score", "score_new", "score_old", "per_class", "fixed_base"} & set(reply.payload)
    assert imprint.payload["S"] > small.payload["S"] > 0
    assert imprint.payload["eligible"] is False and small.payload["eligible"] is True
    assert [sorted(row) for row in status.payload["checked"]] == [
        sorted(("program_file", "query_index", *fscil.V2_SHOWN_FIELDS))] * 2
    assert declared.payload["valid"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    assert session["kept_rule"] == rule and session["fscil"]["session"] == 2
    assert paths.system_prompt.read_text(encoding="utf-8") == render_system("LEARN", fscil=True, kept_rule=rule)
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert fscil.readout_v2(rule) in task_prompt and fscil.final_selection_v2(rule) in task_prompt
    checks = read_jsonl(paths.checks_log)
    assert all(set(fscil.V2_REPLAY_FIELDS) <= set(entry) for entry in checks)
    ledger = CheckLedger(session, paths)  # rebuilt from session.json, every v2 field replayed
    assert ledger.fscil_v2 == rule and [row["S"] for row in ledger.checked()] == [imprint.payload["S"],
                                                                                  small.payload["S"]]
    selection = read_json(paths.final / "selection.json")
    assert selection["rule"] == FINAL_RULE_FSCIL_V2 and selection["kept_rule"] == rule
    assert "s_old_floor" not in selection
    final = selection["final"]
    assert final["kind"] == "checked" and final["patch_id"] == "small.py"  # not the higher S: not eligible
    assert selection["rule_final"] == select_final_v2(checks, rule)
    assert {key: final[key] for key in fscil.V2_SHOWN_FIELDS} == {key: small.payload[key]
                                                                  for key in fscil.V2_SHOWN_FIELDS}
    assert [c["eligible"] for c in selection["candidates"]] == [False, True]
    assert "fscil-retain-v2" in format_table(session_info(result.root))

    tampered = paths.root / "logs" / "checks.jsonl"
    original = tampered.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    rows[1]["S"] = float(rows[1]["S"]) + 1.0
    tampered.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    try:
        with pytest.raises(SessionConfigError, match="differs in S"):
            CheckLedger(session, paths)
    finally:
        tampered.write_text(original, encoding="utf-8")


@needs_sandbox
@needs_neutral_repo
def test_a_v2_library_holds_the_v2_kept_program_and_refuses_a_v1_one(
        toy: ToyFscil, cid: str, rule: dict[str, Any], v2_session: tuple[Any, list[Any]]) -> None:
    result, replies = v2_session
    sessions_dir = Path(os.environ["WEIGHTCODE_RUNS"]) / "sessions"
    row = FS.build_fscil_instance(3, lineage="T2", parent_params=result.root / "final" / "model.npz",
                                  parent_instance=cid, run_id=result.session_id)
    target = str(row["instance_id"])
    (entry,) = fscil.library_entries([result.session_id], sessions_dir, target_instance_id=target, version="v2")
    small = replies[2].payload
    assert {key: entry[key] for key in fscil.V2_LIBRARY_FIELDS} == {key: small[key]
                                                                     for key in fscil.V2_LIBRARY_FIELDS}
    assert entry["label"] == "session 2" and "score_old" not in entry
    with pytest.raises(fscil.FscilHarnessError, match="not joint-retain-v1"):
        fscil.library_entries([result.session_id], sessions_dir, target_instance_id=target)
    v1 = scripted_session(cid, "LEARN", [("python", {"code": program_cell({"small.py": small_source()})}),
                                         ("patch_check", {"program_file": "small.py"})],
                          retention_floor_rows=toy_floor(2), retention_floor_source=toy_source(2))[0]
    with pytest.raises(fscil.FscilHarnessError, match="not fscil-retain-v2"):
        fscil.library_entries([v1.session_id], sessions_dir, target_instance_id=target, version="v2")
    dry = run_session(target, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True, fscil_version="v2",
                      library_sessions=[result.session_id])
    assert dry.status == "dry_run" and dry.record["selftest"]["status"] == "pass", dry.record
    trial = SessionPaths(dry.root).trial
    index = read_json(trial / LIBRARY_INDEX_FILE)
    assert [sorted(item) for item in index["programs"]] == [sorted(("label", "session", "file",
                                                                    *fscil.V2_LIBRARY_FIELDS))]
    assert "percentage points" in index["about"] and result.session_id not in json.dumps(index)
    task_prompt = " ".join(SessionPaths(dry.root).task_prompt.read_text(encoding="utf-8").split())
    assert "This is session 3" in task_prompt and "S, S_new, S_prev, S_base, B_c, d_c and floor_c." in task_prompt
    assert read_json(SessionPaths(dry.root).session_json)["kept_rule"] == fscil.v2_rule_for(
        read_instance_record(target).private)


def small_source() -> str:
    return cover_program(SMALL_BASE, new=False)


@needs_sandbox
@needs_neutral_repo
def test_the_v2_lineages_run_with_no_floor_on_their_own_state(toy: ToyFscil, tmp_path: Path) -> None:
    program = cover_program(list(range(FS.N_BASE_CLASSES)), new=True, spoil=SPOIL)
    steps = [("python", {"code": program_cell({"cover.py": program})}),
             ("patch_check", {"program_file": "cover.py"}), ("status", {}),
             ("declare_final", {"program_file": "cover.py", "rationale": "the only program"})]

    def script(client: Any, replies: list[Any]) -> None:
        for name, arguments in steps:
            call_and_record(client, replies, name, arguments)

    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)
    options = {"effort": "low", "wall_timeout_s": 300.0, "threads": 2, "sandbox": sandbox,
               "session_kwargs": {"launcher": ScriptedLauncher(script), "skip_audit": True}}
    sessions_dir = Path(os.environ["WEIGHTCODE_RUNS"]) / "sessions"
    floors = {2: L.FloorRecord(session=2, floor=-12, source={})}
    for lineage in ("T2", "F2", "C2"):
        with pytest.raises(L.LineageError, match="reads no floors"):
            L.LineageDriver(lineage, floors=floors)
    with pytest.raises(L.LineageError, match="may not set"):
        L.LineageDriver("T2", floors=None, session_kwargs={"fscil_version": "v1"})
    with pytest.raises(L.LineageError, match="T2 has no state yet; F2 and C2"):
        L.LineageDriver("F2", floors=None, **options).run()
    with pytest.raises(L.LineageError, match="takes no floor"):
        L.dry_run(floors=None, placeholder_floor=-12, lineage="T2")
    dry = L.dry_run(floors=None, lineage="T2", session_kwargs={"skip_audit": True})
    assert dry.status == "dry_run" and dry.record["selftest"]["status"] == "pass", dry.record
    assert read_json(dry.root / "session.json")["kept_rule"]["name"] == "fscil-retain-v2"
    assert not (L.lineages_root() / f"T2_{toy.base_params_sha256[:12]}" / "state.json").exists()

    teacher = L.LineageDriver("T2", floors=None, busy=lambda: [], **options)
    assert teacher.dir.name == f"T2_{toy.base_params_sha256[:12]}"
    state = teacher.run()
    s2 = state["sessions"]["2"]
    assert s2["kind"] == L.KIND_LEARN and s2["outcome"] == "kept" and s2["final_kind"] == "checked"
    assert s2["kept_rule"] == fscil.kept_rule_v2(*TOY_ROWS) and "floor" not in s2 and s2["library_sessions"] is None
    assert s2["program"]["S"] > 0 and s2["program"]["B_c"] >= s2["program"]["floor_c"]
    session = read_json(sessions_dir / s2["session_id"] / "session.json")
    assert session["kept_rule"]["name"] == "fscil-retain-v2" and "retention_floor" not in json.dumps(session)
    assert state["sessions"]["3"]["library_sessions"] == [s2["session_id"]]
    for t in (3, 4):
        record = state["sessions"][str(t)]
        result = read_json(sessions_dir / record["session_id"] / "final" / "selection.json")
        assert result["rule"] == FINAL_RULE_FSCIL_V2 and (record["outcome"] == "kept") == (
            result["final"]["kind"] == "checked")
    kept = [t for t in (2, 3, 4) if state["sessions"][str(t)]["program"] is not None]
    for t in range(5, 10):
        record = state["sessions"][str(t)]
        assert record["kind"] == L.KIND_REUSE and "floor" not in record and record["pick"]["kept_rule"] == (
            fscil.kept_rule_v2(*TOY_ROWS))
        assert [run["source_session"] for run in record["runs"]] == kept
        checks = [run["check"] for run in record["runs"] if "check" in run]
        index = FS.retain_v2_pick(checks, -1.0, *TOY_ROWS)
        assert (record["outcome"] == "kept") == (index is not None) == (record["pick"]["kind"] == "checked")
        assert record["outcome"] == ("no-op" if t == 6 else "kept")  # the spoiling step: no check is eligible
        assert all(check["eligible"] is (t != 6) for check in checks)
    assert state["sessions"]["6"]["child_sha256"] == state["sessions"]["5"]["child_sha256"]
    launched = sorted(path.name for path in sessions_dir.iterdir())

    frozen = L.LineageDriver("F2", floors=None, **options).run()
    assert frozen["sessions"]["2"]["kind"] == L.KIND_FROM_TEACHER
    assert frozen["sessions"]["2"]["child_sha256"] == s2["child_sha256"]
    assert frozen["sessions"]["2"]["teacher_state"].endswith(f"T2_{toy.base_params_sha256[:12]}/state.json")
    for t in range(3, 10):
        record = frozen["sessions"][str(t)]
        assert record["kind"] == L.KIND_FROZEN and record["pick"] is None and "check" not in record["runs"][0]
        assert record["outcome"] == "kept"
    six = frozen["sessions"]["6"]  # applied unchecked: its own check would not keep it
    assert not fscil.make_cb_oracle(six["instance_id"], 1, fscil.kept_rule_v2(*TOY_ROWS)).query(
        load_params(Path(six["child_path"])), "f")["eligible"]
    checked = L.LineageDriver("C2", floors=None, **options).run()
    for t in range(3, 10):
        record = checked["sessions"][str(t)]
        assert record["kind"] == L.KIND_CHECKED and record["pick"]["budget"] == 1 and "floor" not in record
        (run,) = record["runs"]
        keep = run.get("check") is not None and run["check"]["eligible"] and run["check"]["S"] > 1e-9
        assert (record["outcome"] == "kept") == keep and keep == (t != 6)
    assert checked["sessions"]["6"]["child_sha256"] == checked["sessions"]["5"]["child_sha256"]
    assert checked["sessions"]["3"]["instance_id"] == frozen["sessions"]["3"]["instance_id"]
    assert "c_ratio" in frozen["summary"] and "c_ratio" in checked["summary"]
    assert sorted(path.name for path in sessions_dir.iterdir()) == launched  # F2 and C2 make no LLM call
    json.dumps(checked, allow_nan=False)
