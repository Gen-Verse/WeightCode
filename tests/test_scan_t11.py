"""PREREG E13 T11 SCAN controls and M04 (``weightcode.evalkit.scan_t11`` and ``scripts/e13/t11_scan.py``): command groups,
the external renaming wrappers on a stub interpreter, the isomorphic relabelling (forward, layout, deltas, and a
program rerun through the sandbox on the toy store), the component networks, the traced forward and the slot and state
readouts. Toy data only: never the real SCAN files, instances or finals."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from test_harness_fixtures import needs_sandbox

from weightcode.data import scan as S
from weightcode.evalkit import scan_t11 as T
from weightcode.evalkit.metrics import IGNORE

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------------
# Command groups


def test_command_groups_of_hand_commands() -> None:
    g = T.command_groups("jump around left twice and walk opposite right", "addprim_jump", "new")
    assert g == {"conjunction": "and", "repetition": "twice", "turns": "around", "turn_verb": "no", "words": "8",
                 "actions": "17-24", "jump_unit": "around", "jump_rep": "twice", "other_phrase": "primitive"}
    g = T.command_groups("turn left after jump thrice", "addprim_jump", "new")
    assert (g["conjunction"], g["repetition"], g["turns"], g["turn_verb"], g["jump_unit"], g["jump_rep"],
            g["other_phrase"]) == ("after", "thrice", "plain", "yes", "bare", "thrice", "turn")
    g = T.command_groups("jump and jump right twice", "addprim_jump", "new")
    assert (g["jump_unit"], g["jump_rep"], g["other_phrase"]) == ("two_phrases", "two_phrases", "jump")
    g = T.command_groups("look around right thrice after walk around left", "around_right", "new")
    assert (g["ar_phrases"], g["ar_rep"], g["other_phrase"], g["actions"]) == ("1", "thrice", "around left", "25-48")
    g = T.command_groups("run around right and jump around right twice", "around_right", "new")
    assert (g["ar_phrases"], g["ar_rep"], g["other_phrase"]) == ("2", "two_phrases", "around right")
    old = T.command_groups("walk twice and look thrice", "addprim_jump", "old")
    assert set(old) == set(T.GENERIC_DIMENSIONS) and old["repetition"] == "twice+thrice"


def test_repetition_value_is_in_the_fixed_list_for_every_grammar_command() -> None:
    for command in S.grammar_commands():
        for condition in S.CONDITIONS:
            part = "new" if S.has_primitive(command, condition) and command != "jump" else "old"
            if condition == "addprim_jump" and command == "jump":
                continue
            groups = T.command_groups(command, condition, part)
            for dim, value in groups.items():
                assert value in T.DIMENSION_VALUES[dim], (command, dim, value)


def test_groups_refuse_commands_outside_their_part() -> None:
    with pytest.raises(T.ScanT11Error):
        T.command_groups("jump twice", "addprim_jump", "old")
    with pytest.raises(T.ScanT11Error):
        T.command_groups("walk twice", "addprim_jump", "new")
    with pytest.raises(T.ScanT11Error):
        T.parse_phrase(["twice", "walk"])
    with pytest.raises((T.ScanT11Error, S.ScanError)):
        T.parse_command("walk and look and run")


def test_grouped_readout_counts_every_row_once_per_dimension() -> None:
    commands = ["jump twice", "jump left", "walk and jump", "jump around right thrice"]
    x, y = S.encode(commands)
    base = np.where(y == IGNORE, IGNORE, S.END)
    child = y.copy()
    child[1, y[1] != IGNORE] = S.END  # row 1 wrong
    out = T.grouped_readout(commands, "addprim_jump", "new", y, base, child)
    for dim, values in out.items():
        assert sum(block["n"] for block in values.values()) == len(commands), dim
    assert out["jump_unit"]["plain"]["em_child"] == 0.0 and out["jump_unit"]["bare"]["em_child"] == 1.0
    assert out["conjunction"]["and"]["n"] == 1


# ---------------------------------------------------------------------------------------------------------------
# External renaming wrappers on a stub interpreter


def _interpreter(blind_to: str | None, blind_phrase: tuple[str, str] | None = None):
    """A stub 'base': answers every command by the interpretation function, except commands holding ``blind_to``
    or the two words ``blind_phrase``, which it answers with END everywhere."""

    def predict(x: np.ndarray) -> np.ndarray:
        commands = S.decode(x)
        _, y = S.encode(commands)
        pred = np.where(y == IGNORE, S.END, y)
        for row, command in enumerate(commands):
            words = command.split()
            blind = (blind_to in words) or (blind_phrase is not None and any(
                tuple(words[k:k + 2]) == blind_phrase for k in range(len(words) - 1)))
            if blind:
                pred[row] = S.END
        return pred

    return predict


def test_rename_wrapper_is_exact_on_consistent_rows_and_breaks_on_inconsistent_ones() -> None:
    commands = ["jump twice", "jump around left and walk", "look after jump opposite right", "jump and run thrice"]
    x, y = S.encode(commands)
    predict = _interpreter("jump")
    pred, consistent = T.rename_jump_predictions(predict, x, "auto")
    assert consistent.all()
    assert T.part_readout(y, predict(x), pred)["em_child"] == 1.0
    pred, consistent = T.rename_jump_predictions(predict, x, "walk")
    assert consistent.tolist() == [True, False, True, True]
    right = T.part_readout(y, predict(x), pred, consistent)
    wrong = T.part_readout(y, predict(x), pred, ~consistent)
    assert right["em_child"] == 1.0 and wrong["em_child"] == 0.0  # I_WALK of the command's own walk read as I_JUMP
    assert T.rename_choice("jump and walk", "auto") == "look"
    with pytest.raises(T.ScanT11Error):
        T.rename_choice("walk twice", "auto")


def test_swap_wrapper_renames_around_right_into_around_left() -> None:
    commands = ["walk around right", "look around right twice after turn left", "run around right and jump around left",
                "walk opposite right"]
    x, y = S.encode(commands)
    predict = _interpreter(None, ("around", "right"))
    pred, applied = T.swap_lr_predictions(predict, x)
    assert applied.tolist() == [True, True, True, False]
    left = T.holds_around_left(commands)
    assert left.tolist() == [False, False, True, False]
    assert T.part_readout(y, predict(x), pred, ~left)["em_child"] == 1.0
    assert T.part_readout(y, predict(x), pred, left)["em_child"] == 0.0  # swapped row holds around right again


# ---------------------------------------------------------------------------------------------------------------
# A tiny network on the real forward template


@pytest.fixture(scope="module")
def forward(tmp_path_factory: pytest.TempPathFactory):
    from weightcode.data.scan_build import forward_constants
    from weightcode.evalkit.patch_check import load_forward
    from weightcode.models.forward_gen import render_seq_forward

    path = tmp_path_factory.mktemp("fwd") / "forward.py"
    path.write_text(render_seq_forward(**forward_constants()))
    fn, _ = load_forward(path)
    return fn


def tiny_params(seed: int = 0, d: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def r(*shape: int) -> np.ndarray:
        return rng.normal(0, 0.5, size=shape).astype(np.float32)

    params = {"tok_emb.weight": r(S.VOCAB_SIZE, d), "pos_emb.weight": r(S.SEQ_LEN, d)}
    for block in ("b1", "b2"):
        for ln in ("ln1", "ln2"):
            params[f"{block}.{ln}.weight"] = (1 + r(d)).astype(np.float32)
            params[f"{block}.{ln}.bias"] = r(d)
        for part in "qkvo":
            params[f"{block}.attn.{part}.weight"] = r(d, d)
            params[f"{block}.attn.{part}.bias"] = r(d)
        params[f"{block}.mlp.fc1.weight"] = r(2 * d, d)
        params[f"{block}.mlp.fc1.bias"] = r(2 * d)
        params[f"{block}.mlp.fc2.weight"] = r(d, 2 * d)
        params[f"{block}.mlp.fc2.bias"] = r(d)
    params["ln_f.weight"] = (1 + r(d)).astype(np.float32)
    params["ln_f.bias"] = r(d)
    params["head.weight"] = r(S.N_CLASSES, d)
    params["head.bias"] = r(S.N_CLASSES)
    return params


ROWS = ["jump twice", "walk around left", "look and jump opposite right", "run thrice after turn right",
        "jump around right twice and walk left"]


def test_default_relabel_moves_every_word_and_action() -> None:
    r = T.default_relabel()
    words = range(len(S.SPECIAL_TOKENS), S.VOCAB_SIZE)
    assert all(r.token_map[i] != i for i in words) and all(r.token_map[i] == i for i in range(len(S.SPECIAL_TOKENS)))
    assert all(r.class_map[c] != c for c in range(len(S.ACTIONS))) and r.class_map[S.END] == S.END
    with pytest.raises(T.ScanT11Error):
        T.Relabel(tuple(range(S.VOCAB_SIZE)), (0, 0, 1, 2, 3, 4, 5))


def test_relabelled_network_computes_the_same_function(forward) -> None:
    params = tiny_params()
    r = T.default_relabel()
    x, y = S.encode(ROWS)
    moved = T.relabel_params(params, r)
    logits = np.asarray(forward(params, x))
    logits_moved = np.asarray(forward(moved, T.relabel_tokens(x, r)))
    back = logits_moved[..., np.asarray(r.class_map)]
    assert T.max_abs_difference(back, logits) < 1e-6
    pred_back = T.unrelabel_classes(logits_moved.argmax(-1), r)
    assert np.array_equal(pred_back, logits.argmax(-1))
    assert np.array_equal(T.unrelabel_classes(T.relabel_classes(y, r), r), y)
    deltas = {"tok_emb.weight": np.arange(S.VOCAB_SIZE * 8, dtype=np.float64).reshape(S.VOCAB_SIZE, 8),
              "head.bias": np.arange(S.N_CLASSES, dtype=np.float64)}
    relabelled_deltas = {"tok_emb.weight": deltas["tok_emb.weight"][r.token_inverse],
                         "head.bias": deltas["head.bias"][r.class_inverse]}
    restored = T.unrelabel_deltas(relabelled_deltas, r)
    assert all(np.array_equal(restored[k], deltas[k]) for k in deltas)


def test_relabel_layout_keeps_names_and_meanings_with_new_ids() -> None:
    layout = {"input": {"tokens": S.token_table(), "layout": S.INPUT_LAYOUT}, "classes": list(S.CLASS_NAMES)}
    r = T.default_relabel()
    out = T.relabel_layout(layout, r)
    assert [t["id"] for t in out["input"]["tokens"]] == list(range(S.VOCAB_SIZE))
    by_name = {t["name"]: t for t in out["input"]["tokens"]}
    for old in S.token_table():
        assert by_name[old["name"]]["id"] == r.token_map[old["id"]]
        assert by_name[old["name"]]["meaning"] == old["meaning"]
    assert out["classes"][r.class_map[S.CLASS_ID["I_JUMP"]]] == "I_JUMP" and out["classes"][S.END] == "END"
    assert layout["classes"] == list(S.CLASS_NAMES)  # the input is not modified
    with pytest.raises(T.ScanT11Error):
        T.relabel_layout({"input": {"tokens": S.token_table()[::-1]}, "classes": list(S.CLASS_NAMES)}, r)


def test_traced_forward_matches_forward_bit_for_bit(forward) -> None:
    params = tiny_params(1)
    x, y = S.encode(ROWS)
    keep = (y != IGNORE) & (y != S.END)
    trace = T.traced_forward(forward, params, x, keep)
    assert T.bytes_equal(trace["logits"], np.asarray(forward(params, x)))
    assert len(trace["blocks"]) == 2 and trace["blocks"][0].shape == (int(keep.sum()), 8)
    assert trace["head_in"].shape == (int(keep.sum()), 8)
    lean = T.traced_forward(forward, params, x, keep, with_blocks=False)
    assert lean["blocks"][0].shape == (0, 8) and T.bytes_equal(lean["head_in"], trace["head_in"])


def test_component_children_and_identity_of_h_only(forward) -> None:
    base = tiny_params(2)
    final = {k: v.copy() for k, v in base.items()}
    j, c = S.TOKEN_ID["jump"], S.CLASS_ID["I_JUMP"]
    final["tok_emb.weight"][j] += 1.0
    final["head.weight"][c] -= 0.5
    final["head.bias"][c] += 2.0
    children, record = T.component_children(base, final)
    assert record["n_changed"] == 8 + 8 + 1
    assert all(T.bytes_equal(children["EH"][k], final[k]) for k in final)
    x, _ = S.encode(ROWS)
    every = np.ones(x.shape, dtype=bool)
    h_only = T.traced_forward(forward, children["H_only"], x, every, with_blocks=False)
    plain = T.traced_forward(forward, base, x, every, with_blocks=False)
    assert T.bytes_equal(h_only["head_in"], plain["head_in"])
    others = [k for k in range(S.N_CLASSES) if k != c]
    assert T.max_abs_difference(h_only["logits"][..., others], plain["logits"][..., others]) == 0.0
    bad = {k: v.copy() for k, v in final.items()}
    bad["b1.attn.q.bias"][0] += 1.0
    with pytest.raises(T.ScanT11Error, match="neither component"):
        T.component_children(base, bad)
    bad = {k: v.copy() for k, v in final.items()}
    bad["head.weight"][0, 0] += 1.0
    with pytest.raises(T.ScanT11Error, match="outside"):
        T.component_children(base, bad)


# ---------------------------------------------------------------------------------------------------------------
# Slot and state readouts


def test_slot_readout_on_hand_logits() -> None:
    _, y = S.encode(["jump twice"])  # I_JUMP I_JUMP then END
    logits = np.zeros(y.shape + (S.N_CLASSES,), dtype=np.float32)
    w, look, run, jump = (S.CLASS_ID[a] for a in ("I_WALK", "I_LOOK", "I_RUN", "I_JUMP"))
    slots = np.flatnonzero(y[0] != IGNORE)
    logits[0, slots[0], [w, look, run, jump]] = [3.0, 1.0, 2.0, 5.0]  # jump wins by 2, spread 3 - 2 = 1
    logits[0, slots[1], [w, look, run, jump]] = [4.0, 1.0, 1.0, 0.0]  # walk wins
    logits[0, slots[2:], S.END] = 1.0
    out = T.slot_readout(y, logits)
    assert out["by_target"]["I_JUMP"]["n"] == 2 and out["by_target"]["I_JUMP"]["accuracy"] == 0.5
    assert out["by_target"]["I_JUMP"]["predicted"] == {"I_WALK": 1, "I_JUMP": 1}
    assert out["jump_slots"]["jump_gap"]["max"] == 2.0 and out["jump_slots"]["wlr_spread"]["min"] == 1.0
    assert out["by_target"]["END"]["accuracy"] == 1.0
    assert out["jump_intrusion"]["max"] == -1.0  # at END slots, jump (0) is 1 below END


def test_state_comparison_on_vectors() -> None:
    refs = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.array([[1.0, 1.0]])]
    mean = np.array([[2 / 3, 2 / 3]])
    out = T.state_comparison(mean, refs)
    assert out["cos_to_mean"]["median"] == pytest.approx(1.0) and out["rel_dist_to_mean"]["max"] == pytest.approx(0)
    out = T.state_comparison(np.array([[1.0, 0.1]]), refs)
    assert out["nearest"] == {"walk": 1, "look": 0, "run": 0}


# ---------------------------------------------------------------------------------------------------------------
# The script's relabelled student and program rerun, on the toy store


def _script():
    path = ROOT / "scripts" / "e13" / "t11_scan.py"
    spec = importlib.util.spec_from_file_location("t11_scan_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOY_PROGRAM = '''
import numpy as np
HYPOTHESIS = "toy: jump's row to the centroid of walk, look and run; the I_JUMP head row to their mean, bias above"
PREDICTED_EFFECT = "toy"


def update(student, experience):
    params, layout, fwd = student["params"], student["layout"], student["forward"]
    toks = {t["name"]: t["id"] for t in layout["input"]["tokens"]}
    names = [t["name"] for t in layout["input"]["tokens"]]
    assert names.index("jump") == toks["jump"]
    classes = list(layout["classes"])
    E = np.asarray(params["tok_emb.weight"], dtype=np.float64)
    H = np.asarray(params["head.weight"], dtype=np.float64)
    B = np.asarray(params["head.bias"], dtype=np.float64)
    pid = [toks[p] for p in ("walk", "look", "run")]
    pc = [classes.index(a) for a in ("I_WALK", "I_LOOK", "I_RUN")]
    jc = classes.index("I_JUMP")
    x = np.asarray(experience["x_old"])
    L = np.asarray(fwd(params, x), dtype=np.float64)[x == toks["OUT"]]
    b = float(np.median(L.max(-1) - L[:, pc].mean(-1)))
    dE, dH, dB = np.zeros_like(E), np.zeros_like(H), np.zeros_like(B)
    dE[toks["jump"]] = E[pid].mean(0) - E[toks["jump"]]
    dH[jc] = H[pc].mean(0) - H[jc]
    dB[jc] = B[pc].mean() + 0.5 * b - B[jc]
    print("b", round(b, 6))
    return {"tok_emb.weight": dE, "head.weight": dH, "head.bias": dB}
'''


@pytest.fixture(scope="module")
def toy_instance(tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    import scan_toy

    from weightcode.evalkit import scanskill as K

    patcher = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("t11_store")
    try:
        scan_toy.store_with_students(tmp, patcher)
        (row,) = K.build_scan_instances("addprim_jump")
        yield K.load_scan_instance(row["instance_id"])
    finally:
        patcher.undo()


@needs_sandbox
def test_program_rerun_on_the_relabelled_student_gives_the_same_edit(toy_instance, tmp_path: Path) -> None:
    from weightcode.harness.program import run_update
    from weightcode.harness.transfer import stage_student
    from weightcode.sandbox.program import ProgramSandbox

    script = _script()
    instance = toy_instance
    program = tmp_path / "toy_program.py"
    program.write_text(TOY_PROGRAM)
    from weightcode.harness.runner import SANDBOX_PYTHON

    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=2 * 2**30, threads=1)
    relabel = T.default_relabel()
    iso = script.isomorphism_check(instance, relabel)
    assert iso["argmax_equal"] and iso["max_abs_logit_difference"] < 1e-5
    student, experience_file, _ = stage_student(instance, tmp_path / "plain")
    plain = run_update(student, experience_file, program, tmp_path / "plain_run", sandbox)
    moved_student, moved_experience, staged = script.relabelled_student(instance, tmp_path / "moved", relabel)
    with np.load(moved_experience) as archive:
        assert np.array_equal(T.unrelabel_classes(archive["y_old"], relabel), instance.evidence["y_old"])
    moved = run_update(moved_student, moved_experience, program, tmp_path / "moved_run", sandbox)
    assert plain.record["run"]["stdout"] == moved.record["run"]["stdout"]
    back = T.unrelabel_deltas(moved.deltas, relabel)
    compared = script.deltas_equal(back, plain.deltas)
    assert all(v["max_abs_difference"] < 1e-9 for v in compared["per_tensor"].values()), compared
    assert staged["relabel"]["token_map"]["jump"] == relabel.token_map[S.TOKEN_ID["jump"]]


@needs_sandbox
def test_script_end_to_end_on_the_toy_store(toy_instance, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A toy final (the toy program through the real program path), one rule-generated reference with its retention
    record, and the walk record, laid out as the real records are; then programs, arms, mech and summary."""
    import json
    import os
    import shutil

    from weightcode.baselines.fit import scanrefs as R
    from weightcode.common.identity import save_params
    from weightcode.evalkit import devreadout as D
    from weightcode.evalkit.scanskill import scan_readout
    from weightcode.harness.program import run_update
    from weightcode.harness.runner import SANDBOX_PYTHON
    from weightcode.harness.transfer import stage_student
    from weightcode.patch import Limits
    from weightcode.sandbox.program import ProgramSandbox

    script = _script()
    instance = toy_instance
    runs = Path(os.environ["WEIGHTCODE_RUNS"])
    session = "stoy0000000000001"
    final_dir = runs / "sessions" / session / "final"
    final_dir.mkdir(parents=True)
    program = final_dir / "toy_program.py"
    program.write_text(TOY_PROGRAM)
    limits = Limits(max_changed_frac=0.2, max_edit_frac=None, max_file_bytes=64 * 2**20, max_ops=70, max_rank=8,
                    max_rel_frob=None, rel_frob_min_ndim=2)
    (runs / "sessions" / session / "session.json").write_text(json.dumps({"limits": script.dataclass_dict(limits)}))
    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=2 * 2**30, threads=1)
    student, experience_file, _ = stage_student(instance, tmp_path / "student")
    update = run_update(student, experience_file, program, tmp_path / "run", sandbox)
    work = tmp_path / "apply"
    work.mkdir()
    child, applied = script.apply_deltas(instance.params, update, work, limits)
    save_params(final_dir / "model.npz", child)
    shutil.copyfile(update.deltas_path, final_dir / "deltas.npz")
    (final_dir / "program_run.json").write_text(json.dumps(update.record))
    learn = runs / "devreadouts" / "learn" / session
    learn.mkdir(parents=True)
    (learn / "info.json").write_text(json.dumps(
        {"final": {"query_index": 0, "patch_id": program.name, "child_sha256": applied["child_sha256"]},
         "trajectory": [{"readouts_scan": scan_readout(instance, child)}]}))
    # one rule-generated reference and its retention record (a floor that keeps the best check)
    monkeypatch.setattr(R, "STEP_CHECKPOINTS", (1, 2, 3, 4))
    monkeypatch.setattr(R, "N_RULEGEN", 200)
    monkeypatch.setattr(R, "N_RULEGEN_OLD", 100)
    monkeypatch.setattr(script, "S_OLD_FLOOR", -1000)
    method = "rulegen_mixed_grad_full"
    run_dir = tmp_path / "ref"
    R.run_reference(instance, method, fit_set=R.rulegen_fit_set("addprim_jump", "mixed"), run_dir=run_dir)
    kept = D.readout_retention(run_dir / "run.json", -1000)
    (runs / "devreadouts" / "toy_refs.jsonl").write_text(json.dumps(kept) + "\n")
    (runs / "devreadouts" / "toy_walk.json").write_text(json.dumps(R.walk_diagnostic(instance)))
    with_ref = not kept["is_base_noop"]
    spec = {"instance": instance.instance_id, "session": session, "final_program": program.name, "final_query": 0,
            "refs": {method: kept["final_patch_id"]} if with_ref else {}, "refs_record": "toy_refs.jsonl",
            "walk_record": "toy_walk.json"}
    monkeypatch.setattr(script, "CONDITIONS", {"addprim_jump": spec})
    arms = ("base", "final", "E_only", "H_only", *((f"ref_{method}",) if with_ref else ()), "walk_row",
            "rename_auto", "rename_walk", "relabelled_program")
    monkeypatch.setattr(script, "ARMS", {"addprim_jump": arms})

    assert script.main(["programs", "--conditions", "addprim_jump"]) == 0
    reruns = {arm: json.loads(script.rerun_record_path("addprim_jump", arm).read_text())
              for arm in ("program", "program_relabelled")}
    assert reruns["program"]["checks"]["deltas_vs_final"]["equal"] and reruns["program"]["checks"]["child_equals_final"]
    assert reruns["program_relabelled"]["staged"]["isomorphism_on_evidence_rows"]["argmax_equal"]
    assert reruns["program_relabelled"]["checks"]["stdout_equal_session"]
    with pytest.raises(script.T11ScanError):  # never repeated
        script.main(["programs", "--conditions", "addprim_jump"])

    assert script.main(["arms", "--condition", "addprim_jump"]) == 0
    records = {arm: script.read_line(script.arm_record_path(arm, instance.instance_id)) for arm in arms}
    for arm, record in records.items():
        assert not [k for k, v in record["checks"].items() if v is False], (arm, record["checks"])
        assert record["new"]["all"]["n"] == instance.private["n_vselect_new"]
        assert sum(b["n"] for b in record["new"]["groups"]["jump_unit"].values()) == record["new"]["all"]["n"]
    assert records["E_only"]["components"]["n_changed"] == records["final"]["n_changed"]
    assert records["relabelled_program"]["new"]["right"] == records["final"]["new"]["right"]
    assert records["E_only"]["old"]["right"] == records["base"]["old"]["right"]
    assert records["rename_walk"]["consistent_rows"]["n"] + records["rename_walk"]["consistent_rows"][
        "n_inconsistent"] == records["base"]["new"]["all"]["n"]

    assert script.main(["mech"]) == 0
    m04 = json.loads((runs / "t11scan" / f"m04_{instance.instance_id}.json").read_text())
    assert not [k for k, v in m04["checks"].items() if v is False], m04["checks"]
    assert m04["checks"]["H_only_head_input_max_abs_difference_new"] == 0.0
    assert set(m04["states"]) == {"b1", "head_in"}
    assert m04["states"]["head_in"]["E_only"]["jump_slots"]["n"] == m04["n_action_slots"]["jump"]
    assert set(m04["slots"]) == {"base", "E_only", "H_only", "EH"}

    assert script.main(["summary"]) == 0
    summary = json.loads((runs / "t11scan" / "e13_t11_summary.json").read_text())
    readings = summary["readings"]["addprim_jump"]
    assert readings["relabelled_program_equals_final_row_for_row"] is True
    assert "reading" in readings["m04"] and "rename_auto_matches_final" in readings
