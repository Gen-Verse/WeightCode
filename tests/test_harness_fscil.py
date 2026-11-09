"""The editor harness on fscil-v1 instances (PREREG E9): names, the derived templates and the kept ones, id resolution
and the refusals before any write, the LEARN view at sessions 2 and 4, the limits with the head exempt, the leak scan,
the floor in rows (0 included), a scripted session whose ledger rebuilds from session.json, session_info, the dev
readout and transfer, and the library of a lineage's earlier kept programs at session 3.

The store is the toy cifar100_fscil store (``fscil_toy``): 3 x 8 x 8 random images, so no class signal, and one random
ResNet-20 student with a cosine head. A program that uses only its inputs keeps the chance level on it, so the scripted
editor's program (``imprint_program``) carries the toy's V_check images and imprints the head row of every new class of
its experience on their mean feature: a test fixture, not a method, it gives the kept rule a program to keep. Every
directory lives under pytest tmp dirs with neutral names: the sandbox self-test matches the session's forbidden strings
("fscil", "cifar", "seed", ids) as substrings of its mounts. The repo's own path is mounted too, so a sandboxed test
here passes only in a worktree whose path holds none of them. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fscil_toy import ToyFscil, build_toy_fscil_store
from test_harness_fixtures import needs_sandbox, read_jsonl, scripted_session

from weightcode.analysis.session_info import format_table, session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256, save_params
from weightcode.data import store as data_store
from weightcode.evalkit import fscil as FS
from weightcode.evalkit.devreadout import readout_run
from weightcode.evalkit.instances import leak_tokens, scan_for_tokens
from weightcode.harness import fscil, seqskill
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import applied_limits, arm_spec
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import file_sha256, read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.library import LIBRARY_INDEX_FILE
from weightcode.harness.prompts import PROMPT_DIR, render_system
from weightcode.harness.retention import kept_rule_record_rows
from weightcode.harness.runner import SANDBOX_PYTHON, run_session, session_forbidden_strings
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import FSCIL_CHECK_SETS, PROGRAM_CHECK_SETS, tool_specs
from weightcode.harness.transfer import run_transfer
from weightcode.harness.trial import assemble
from weightcode.patch import PatchError, apply_patch
from weightcode.sandbox.program import ProgramSandbox

# The repo is mounted into every session, so its path must hold none of the self-test's dataset tokens.
REPO_TOKENS = scan_for_tokens(str(Path(__file__).resolve().parents[1]), leak_tokens(FS.DATASET))
needs_neutral_repo = pytest.mark.skipif(bool(REPO_TOKENS), reason=(
    f"the repo path holds {REPO_TOKENS}: an fscil session's sandbox self-test refuses it (it reads the session's "
    "mounts, the repo among them); run these tests from a worktree whose path holds none"))
HEAD = ["fc.weight"]
R_TOY, D_TOY = -10.0, 0.0  # the toy floors (the toy has no PREREG floor): about -12 rows at session 2, no clamp
SOURCE = {"rule": "PREREG E9 joint_retain_floor", "R": R_TOY, "d": D_TOY, "n": 1}
WHY = {"hypothesis": "a limit probe", "predicted_effect": "accepted or refused by the limits"}
PLACEHOLDER = {"placeholder": True, "note": "a stand-in floor for a dry run, not the PREREG E9 floor"}
TRIAL_FILES = ["model.npz", "model_spec.json", "forward.py", "task.json", "experience.npz"]
# The templates of the fewshot-v1 (E2), seqskill-v1 (E5) and rule_only (E8) sessions: the fscil path keeps them.
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
PROGRAM_TEMPLATE = '''HYPOTHESIS = "a class is recognised when its head row points at the mean feature of its images"
PREDICTED_EFFECT = "the check images of the new classes become correct and few earlier images break"
import numpy as np
IMAGES = np.frombuffer(bytes.fromhex("{images}"), dtype=np.uint8).reshape({shape})
LABELS = np.array({labels}, dtype=np.int64)
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
    new = sorted(set(int(c) for c in experience["y_new"]))
    for c in new:
        z = np.asarray(student["forward"](probe, IMAGES[LABELS == c]), dtype=np.float64)[:, :width]
        mean = (z / np.linalg.norm(z, axis=1, keepdims=True)).mean(axis=0)
        delta[c] = mean / np.linalg.norm(mean) - weight[c]
    if new and new[0] == SPOIL_FIRST_CLASS:
        for c in sorted(set(int(c) for c in experience["y_prev"])):
            delta[c] = -2.0 * weight[c]
    print("rows", new)
    return {{head: delta}}
'''


def imprint_program(spoil_first_class: int | None = None) -> str:
    """The scripted editor's program (module docstring): the toy V_check images of every incremental class as hex,
    and the head rows of the experience's new classes set to their mean normalised feature (a probe head reads the
    features through the student's forward). At the session whose first new class is ``spoil_first_class`` it also
    flips the head rows of the earlier sessions' classes, which breaks their images (score_old far below any floor)."""
    v_check = data_store.load_split(FS.DATASET, "V_check")
    keep = v_check["y"] >= FS.N_BASE_CLASSES
    images = np.ascontiguousarray(v_check["x"][keep]).astype(np.uint8)
    return PROGRAM_TEMPLATE.format(images=images.tobytes().hex(), shape=(-1, *images.shape[1:]),
                                   labels=v_check["y"][keep].astype(int).tolist(), spoil=spoil_first_class)


def program_cell(programs: dict[str, str]) -> str:
    payload = json.dumps(json.dumps(programs))
    return ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n"
            f"print(sorted(json.loads({payload})))\n")


def learn_steps(program: str, name: str = "imprint.py") -> list[tuple[str, dict[str, Any]]]:
    """The scripted editor: write the program, check it, read the status and declare it."""
    return [("python", {"code": program_cell({name: program})}), ("patch_check", {"program_file": name}),
            ("status", {}), ("declare_final", {"program_file": name, "rationale": "the only program"})]


def toy_floor(session: int) -> int:
    n_c, n_s = FS.old_part_sizes(session)
    return FS.joint_retain_floor(n_c, n_s, R_TOY, D_TOY)


def toy_source(session: int) -> dict[str, Any]:
    n_c, n_s = FS.old_part_sizes(session)
    return {**SOURCE, "session": session, "floor_t": toy_floor(session), "n_c": n_c, "n_s": n_s}


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ToyFscil]:
    """The toy store for the whole module, under a neutral directory name."""
    with pytest.MonkeyPatch.context() as patch:
        yield build_toy_fscil_store(tmp_path_factory.mktemp("hs"), patch)


@pytest.fixture(scope="module")
def cid(toy: ToyFscil) -> str:
    """The session-2 instance (on the base student)."""
    return str(FS.build_fscil_instance(2, lineage="T", run_id="r0")["instance_id"])


@pytest.fixture(scope="module")
def chain4(toy: ToyFscil, cid: str) -> list[str]:
    """Sessions 2, 3 and 4 on the base student's weights (a lineage whose sessions kept nothing)."""
    parent = toy.root / "parent.npz"
    save_params(parent, toy.base_params)
    ids = [cid]
    for t in (3, 4):
        row = FS.build_fscil_instance(t, lineage="X", parent_params=parent, parent_instance=ids[-1], run_id="r0")
        ids.append(str(row["instance_id"]))
    return ids


@pytest.fixture(scope="module")
def learned(toy: ToyFscil, cid: str) -> tuple[Any, list[Any], str]:
    """One scripted LEARN session at session 2 under the toy floor: (result, replies, program)."""
    program = imprint_program()
    result, replies = scripted_session(cid, "LEARN", learn_steps(program), retention_floor_rows=toy_floor(2),
                                       retention_floor_source=toy_source(2))
    return result, replies, program


def _paragraphs(text: str) -> list[str]:
    return [block for block in text.split("\n\n") if block.strip()]


def test_fscil_names_agree_with_the_evalkit_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_RUNS", "WEIGHTCODE_SEALED"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(name, str(tmp_path / name))
    assert fscil.FSCIL_RULE == FS.FSCIL_RULE and fscil.FSCIL_ROOT_NAME == FS.ROOT_NAME
    assert fscil.INSTANCE_ID_RE.pattern == FS.INSTANCE_ID_RE.pattern
    assert fscil.EXPERIENCE_KEYS == FS.EVIDENCE_KEYS and fscil.EVALUATION_BATCH == FS.EVALUATION_BATCH
    assert fscil.READOUT_PARTS == FS.READOUT_PARTS
    assert fscil.instance_dir("c0123abcd") == FS.fscil_instance_dir("c0123abcd")
    assert fscil.is_fscil_id("c0123abcd") and not fscil.is_fscil_id("g0123abcd")
    assert fscil.leak_words(["c0123abcd"]) == FS.fscil_leak_tokens(["c0123abcd"])
    assert {"fscil", "cifar100_fscil", "seed"} <= set(fscil.leak_words())
    with pytest.raises(fscil.FscilHarnessError, match="fscil instance id"):
        fscil.instance_dir("g0123abcd")


def test_the_fscil_templates_derive_from_the_fewshot_ones_and_the_kept_ones_are_unchanged(toy: ToyFscil) -> None:
    for name, digest in KEPT_TEMPLATE_SHA256.items():  # the fewshot, seqskill and rule_only templates are untouched
        assert hashlib.sha256((PROMPT_DIR / name).read_bytes()).hexdigest() == digest
    assert render_system("LEARN", fewshot=True) == (PROMPT_DIR / "system_LEARN_fewshot.md").read_text(encoding="utf-8")
    assert render_system("LEARN", seqskill=True) == (PROMPT_DIR / "system_LEARN_seqskill.md").read_text(
        encoding="utf-8")
    assert render_system("LEARN", seqskill=True, seqskill_variant=seqskill.RULE_ONLY) == (
        PROMPT_DIR / "system_LEARN_seqskill_rule.md").read_text(encoding="utf-8")
    few = (PROMPT_DIR / "system_LEARN_fewshot.md").read_text(encoding="utf-8")
    inc = (PROMPT_DIR / "system_LEARN_fscil.md").read_text(encoding="utf-8")
    few_blocks, inc_blocks = _paragraphs(few), _paragraphs(inc)
    changed = [index for index, (a, b) in enumerate(zip(few_blocks, inc_blocks)) if a != b]
    assert len(few_blocks) == len(inc_blocks) and len(changed) == 2  # the role and the kept rule's body
    assert inc_blocks[changed[1] - 1] == "## What is kept"
    assert "sessions" in inc_blocks[changed[0]] and "seen so far" in inc_blocks[changed[0]]
    assert "earlier sessions" in inc_blocks[changed[1]] and "prediction rule of this session" in inc_blocks[changed[1]]
    assert max(len(line) for line in inc.splitlines()) <= 116
    task_few = (PROMPT_DIR / "task_LEARN.md").read_text(encoding="utf-8").split("\n- ")
    task_inc = (PROMPT_DIR / "task_LEARN_fscil.md").read_text(encoding="utf-8").split("\n- ")
    changed = [index for index, (a, b) in enumerate(zip(task_few, task_inc)) if a != b]
    assert len(task_few) == len(task_inc) and len(changed) == 2
    assert [task_inc[index].split(":")[0] for index in changed] == ["`forward.py`", "`experience.npz`"]
    assert "`x_prev`" in task_inc[changed[1]] and "`prediction`" in task_inc[changed[0]]
    rendered = render_system("LEARN", fscil=True)
    assert rendered == inc and fscil.scan(rendered) == []
    with pytest.raises(ValueError, match="program arm"):
        render_system("M2", fscil=True)
    with pytest.raises(ValueError, match="not both"):
        render_system("LEARN", fewshot=True, fscil=True)
    stated = " ".join(render_system("LEARN", fscil=True, kept_rule=kept_rule_record_rows(
        -12, 120, fscil.CHECK_SET, SOURCE)).split())
    assert "score_old is at least -12 (at most 12 more old-class check-set images broken than fixed)" in stated
    zero = " ".join(render_system("LEARN", fscil=True, kept_rule=kept_rule_record_rows(
        0, 1800, fscil.CHECK_SET, SOURCE)).split())
    assert "score_old is at least 0 (at most 0 more old-class check-set images broken than fixed)" in zero
    assert "with score_old below 0 is never kept" in zero


def test_the_fscil_tools_name_the_session_check_sets() -> None:
    tools = arm_spec("LEARN").tools
    plain = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True)}
    shown = {spec["name"]: spec["description"] for spec in tool_specs(tools, program=True, fscil=True)}
    assert shown["patch_check"] == plain["patch_check"].replace(PROGRAM_CHECK_SETS, FSCIL_CHECK_SETS)
    assert shown["patch_check"] != plain["patch_check"]
    assert {k: v for k, v in shown.items() if k != "patch_check"} == {k: v for k, v in plain.items()
                                                                      if k != "patch_check"}
    assert "score_old of at least 0" in {spec["name"]: spec["description"] for spec in tool_specs(
        tools, program=True, fscil=True, s_old_floor=0)}["declare_final"]
    with pytest.raises(ValueError, match="program arm"):
        tool_specs(arm_spec("M2").tools, fscil=True)


def test_an_fscil_id_resolves_and_every_refusal_comes_before_a_write(toy: ToyFscil, cid: str) -> None:
    record = read_instance_record(cid)
    assert record.is_fscil and record.has_own_experience and record.private["session"] == 2
    with pytest.raises(InstanceRefused, match="does not exist"):
        read_instance_record("c00000000")
    runs = Path(os.environ["WEIGHTCODE_RUNS"])
    before = sorted(path.name for path in runs.iterdir())
    n_old = int(record.private["n_vcheck_old"])
    refusals = [
        (InstanceRefused, "not a fraction", {"retention_floor_frac": 0.01}),
        (InstanceRefused, "retention_floor_rows", {}),
        (InstanceRefused, f"-{n_old} .. 0", {"retention_floor_rows": -n_old - 1}),
        (ValueError, "never above 0", {"retention_floor_rows": 1}),
        (ValueError, "dry run only", {"retention_floor_rows": -5, "retention_floor_source": PLACEHOLDER}),
    ]
    for error, message, kwargs in refusals:
        with pytest.raises(error, match=message):
            run_session(cid, "LEARN", 0, effort="low", wall_timeout_s=60.0, **kwargs)
    with pytest.raises(InstanceRefused, match="program arm"):
        run_session(cid, "M2", 0, effort="low", wall_timeout_s=60.0)
    assert sorted(path.name for path in runs.iterdir()) == before  # nothing written


def test_the_learn_view_holds_three_parts_at_sessions_2_and_4(toy: ToyFscil, chain4: list[str],
                                                             tmp_path: Path) -> None:
    for t, instance_id in zip((2, 4), (chain4[0], chain4[2])):
        instance = load_instance(fscil.instance_dir(instance_id))
        (tmp_path / f"v{t}" / "private").mkdir(parents=True)
        view = assemble(instance.instance_dir, "LEARN", tmp_path / f"v{t}")
        assert sorted(view.files) == sorted(TRIAL_FILES) and view.fscil == fscil.session_record(instance.private)
        with np.load(view.trial_dir / "experience.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        assert sorted(arrays) == sorted(fscil.EXPERIENCE_KEYS)
        assert sorted(set(arrays["y_new"].tolist())) == FS.session_classes(t) and len(arrays["y_new"]) == 25
        assert len(arrays["y_old"]) == FS.N_REPLAY and int(arrays["y_old"].max()) < FS.N_BASE_CLASSES
        assert sorted(set(arrays["y_prev"].tolist())) == FS.prev_classes(t)
        assert len(arrays["y_prev"]) == 25 * (t - 2) and arrays["x_prev"].shape[1:] == (3, 8, 8)
        assert sorted(set(arrays["session_prev"].tolist())) == list(range(2, t))
        masked = FS.masked_forward(instance.forward, t)
        for part in fscil.EXPERIENCE_PARTS:  # the predictions are the student's under the session mask
            if len(arrays[f"y_{part}"]):
                expected = np.argmax(masked(instance.params, arrays[f"x_{part}"]), axis=1)
                np.testing.assert_array_equal(arrays[f"pred_{part}"], expected)
                assert int(arrays[f"pred_{part}"].max()) < FS.n_seen(t)
        task = read_json(view.trial_dir / "task.json")
        assert task["experience"]["new"]["count"] == 25 and task["experience"]["prev"]["count"] == 25 * (t - 2)
        assert task["experience"]["old"]["count"] == FS.N_REPLAY
        assert view.experience["session"] == t and view.experience["n_seen"] == FS.n_seen(t)
        spec = read_json(view.trial_dir / "model_spec.json")
        assert spec["prediction"]["n_seen"] == FS.n_seen(t) and "model_id" not in spec
        assert fscil.scan(json.dumps(task), fscil.session_extra_tokens(instance.private)) == []


def test_the_fscil_limits_exempt_the_head_and_allow_one_op_per_tensor(toy: ToyFscil, cid: str) -> None:
    instance = load_instance(fscil.instance_dir(cid))
    n_tensors = len(instance.params)
    limits = fscil.program_limits(instance.task["limits"], n_tensors)
    assert limits == applied_limits(instance.task["limits"], "LEARN", n_tensors)
    assert list(limits.rel_frob_exempt) == HEAD == fscil.head_tensors(read_json(instance.instance_dir /
                                                                               "model_spec.json"))
    assert limits.max_ops == n_tensors > 64 and limits.max_rel_frob == 0.5 and limits.max_changed_frac == 0.2

    def patch(tensor: str, scale: float) -> dict:
        value = instance.params[tensor]
        return {"schema": "weightcode.patch.v0", "parent_sha256": instance.params_sha256, **WHY,
                "ops": [{"op": "add", "tensor": tensor, "index": [[0] * value.ndim],
                         "values": [scale * float(np.linalg.norm(value))]}]}

    apply_patch(instance.params, patch("fc.weight", 3.0), None, limits, instance.params_sha256)  # the head: exempt
    conv = next(name for name, value in instance.params.items() if value.ndim == 4)
    with pytest.raises(PatchError, match="rel"):
        apply_patch(instance.params, patch(conv, 3.0), None, limits, instance.params_sha256)
    every = {"schema": "weightcode.patch.v0", "parent_sha256": instance.params_sha256, **WHY,
             "ops": [{"op": "add", "tensor": name, "index": [[0] * value.ndim], "values": [0.0]}
                     for name, value in instance.params.items()]}
    apply_patch(instance.params, every, None, limits, instance.params_sha256)  # one op per tensor fits max_ops
    with pytest.raises(PatchError):
        apply_patch(instance.params, {**every, "ops": [*every["ops"], every["ops"][0]]}, None, limits,
                    instance.params_sha256)


def test_the_fscil_strings_hold_the_chain_and_pass_the_class_names(toy: ToyFscil, chain4: list[str]) -> None:
    record = read_instance_record(chain4[2])
    tokens = fscil.session_extra_tokens(record.private)
    assert set(chain4) <= set(tokens) and toy.base_model_id in tokens
    forbidden = session_forbidden_strings(record, extra_tokens=tokens)
    assert set(fscil.leak_words(tokens)) <= set(forbidden.text)
    assert set(chain4) <= set(forbidden.selftest) and {"fscil", "seed"} <= set(forbidden.selftest)
    assert record.private["vcheck_path"] in forbidden.text and record.private["oldcheck_path"] in forbidden.text
    names = "the plain, plate, poppy, porcupine and possum images, then apple and aquarium_fish"
    assert fscil.scan(names, forbidden.text) == []
    assert fscil.scan(f"the fscil store and {chain4[0]}", forbidden.text) == sorted(["fscil", chain4[0]])


@needs_sandbox
@needs_neutral_repo
def test_a_floor_of_zero_runs_and_keeps_only_a_program_with_no_net_old_loss(toy: ToyFscil, cid: str) -> None:
    source = {**toy_source(2), "floor_t": 0, "d": 0.05}
    dry = run_session(cid, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True, retention_floor_rows=0,
                      retention_floor_source=source)
    assert dry.status == "dry_run" and dry.record["selftest"]["status"] == "pass", dry.record
    session = read_json(SessionPaths(dry.root).session_json)
    assert session["kept_rule"]["s_old_floor"] == 0
    stated = " ".join(SessionPaths(dry.root).system_prompt.read_text(encoding="utf-8").split())
    assert "score_old is at least 0 (at most 0 more old-class check-set images broken than fixed)" in stated
    result, replies = scripted_session(cid, "LEARN", learn_steps(imprint_program()), retention_floor_rows=0,
                                       retention_floor_source=source)
    assert result.status == "complete", result.record.get("infra_reasons")
    check = replies[1].payload
    assert check["score"] > 0 and check["score_old"] < 0  # kept under the toy floor, not under 0
    selection = read_json(SessionPaths(result.root).final / "selection.json")
    assert selection["s_old_floor"] == 0 and selection["final"]["kind"] == "base"


@needs_sandbox
@needs_neutral_repo
def test_an_fscil_dry_run_records_its_placeholder_floor(toy: ToyFscil, cid: str) -> None:
    result = run_session(cid, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                         retention_floor_rows=-12, retention_floor_source=PLACEHOLDER)
    assert result.status == "dry_run", result.record
    assert result.root.parent.name == "dry_runs"
    session = read_json(SessionPaths(result.root).session_json)
    assert session["kept_rule"] == kept_rule_record_rows(-12, int(session["fscil"]["n_vcheck_old"]),
                                                         fscil.CHECK_SET, PLACEHOLDER)
    assert session["leak_scan"]["passed"] is True and result.record["selftest"]["status"] == "pass"
    assert sorted(result.record["trial_files"]) == sorted(TRIAL_FILES)
    assert session["model_id"] == toy.base_model_id and session["fscil"]["session"] == 2


@needs_sandbox
@needs_neutral_repo
def test_a_scripted_learn_session_scores_programs_under_the_session_mask(
        toy: ToyFscil, cid: str, learned: tuple[Any, list[Any], str], tmp_path: Path) -> None:
    result, replies, program = learned
    assert result.status == "complete", result.record.get("infra_reasons")
    cell, check, _, declared = replies
    assert not cell.is_error and not check.is_error, check.payload
    assert check.payload["score"] == check.payload["score_new"] + check.payload["score_old"]
    assert check.payload["tensors"] == HEAD and check.payload["program_stdout"].startswith("rows [60, 61")
    assert check.payload["score"] > 0 and check.payload["score_old"] >= toy_floor(2)
    assert declared.payload["valid"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    instance = load_instance(fscil.instance_dir(cid))
    assert session["fscil"] == fscil.session_record(instance.private) and session["clean_check"] is None
    assert session["kept_rule"] == kept_rule_record_rows(toy_floor(2), int(instance.private["n_vcheck_old"]),
                                                         fscil.CHECK_SET, toy_source(2))
    assert session["model_id"] == toy.base_model_id and session["n_vcheck"] == fscil.vcheck_rows(instance.private)
    assert session["leak_scan"]["passed"] is True and "fscil" in session["leak_scan"]["tokens"]
    assert paths.system_prompt.read_text(encoding="utf-8") == render_system("LEARN", fscil=True,
                                                                            kept_rule=session["kept_rule"])
    task_prompt = paths.task_prompt.read_text(encoding="utf-8")
    assert "This is session 2" in task_prompt and "`x_prev`" in task_prompt and "60 (plain)" in task_prompt
    assert LIBRARY_INDEX_FILE not in task_prompt
    entry = read_jsonl(paths.checks_log)[0]
    attempt_dir = result.root / "attempts" / str(entry["attempt"])
    child = ArmApplier(session).apply(read_json(attempt_dir / str(entry["patch_file"])), attempt_dir).child
    assert params_sha256(child) == entry["child_sha256"]
    expected = FS.make_fscil_oracle(cid).query(child, "imprint")  # evalkit's own oracle under the session mask
    assert (check.payload["score_new"], check.payload["score_old"]) == (expected["score_new"], expected["score_old"])
    ledger = CheckLedger(session, paths)  # rebuilt from session.json alone, as the MCP server builds it
    assert [row["score"] for row in ledger.checked()] == [check.payload["score"]] == [entry["score"]]
    final = read_json(paths.final / "selection.json")["final"]
    assert final["kind"] == "checked" and final["program"]["sha256"] == hashlib.sha256(program.encode()).hexdigest()

    info = session_info(result.root)
    row = info["trajectory"][0]
    assert row["readouts_fscil"] == fscil.readout(instance, child)["readouts_fscil"]
    assert row["vselect"]["n"] == row["vselect_new"]["n"] + row["vselect_prev"]["n"] + row["vselect_old"]["n"]
    assert row["vselect_prev"]["n"] == 0 and "bal_seen_child" in row["readouts_fscil"]
    assert "dbal_seen" in format_table(info)
    assert "x_prev" in info["access"]["evidence_arrays"]
    json.dumps(info, allow_nan=False, default=str)
    dev = readout_run(result.root / "run.json")
    assert dev["rule"] == FS.FSCIL_RULE and dev["readouts_fscil"] == json.loads(json.dumps(
        fscil.readout(instance, child)["readouts_fscil"]))
    json.dumps(dev, allow_nan=False)

    path = tmp_path / "imprint.py"
    path.write_text(program, encoding="utf-8")
    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)
    (record_path,) = run_transfer(path, [cid], tmp_path / "transfer", sandbox)
    record = json.loads(record_path.read_text())
    assert record["status"] == "ok" and record["within_limits"] is True and record["tensors"] == HEAD
    assert record["experience"]["source"] == fscil.EXPERIENCE_SOURCE and record["limits"]["rel_frob_exempt"] == HEAD
    assert record["readout_source"] == FS.FSCIL_RULE and record["readouts_fscil"]["session"] == 2
    assert record["child_sha256"] == entry["child_sha256"]  # the same program on the same student: the same child


@needs_sandbox
@needs_neutral_repo
def test_the_library_of_session_3_is_the_lineage_kept_program(
        toy: ToyFscil, cid: str, learned: tuple[Any, list[Any], str], tmp_path: Path) -> None:
    result, replies, program = learned
    child_path = result.root / "final" / "model.npz"
    row = FS.build_fscil_instance(3, lineage="T", parent_params=child_path, parent_instance=cid,
                                  run_id=result.session_id)
    target = str(row["instance_id"])
    entries = fscil.library_entries([result.session_id], toy_sessions_dir(), target_instance_id=target)
    (entry,) = entries
    check = replies[1].payload
    assert (entry["label"], entry["session"], entry["file"], entry["status"]) == (
        "session 2", 2, "library_program_1.py", "complete")
    assert (entry["score"], entry["score_new"], entry["score_old"]) == (
        check["score"], check["score_new"], check["score_old"])
    assert entry["_bytes"] == program.encode()
    dry = run_session(target, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                      retention_floor_rows=toy_floor(3), retention_floor_source=toy_source(3),
                      library_sessions=[result.session_id])
    assert dry.status == "dry_run" and dry.record["selftest"]["status"] == "pass", dry.record
    trial = SessionPaths(dry.root).trial
    assert (trial / "library_program_1.py").read_bytes() == program.encode()
    index = read_json(trial / LIBRARY_INDEX_FILE)
    assert [item["label"] for item in index["programs"]] == ["session 2"]
    assert result.session_id not in json.dumps(index) and cid not in json.dumps(index)
    session = read_json(SessionPaths(dry.root).session_json)
    assert result.session_id in session["forbidden_strings"]["text"] and cid in session["forbidden_strings"]["text"]
    assert session["library"]["entries"][0]["status"] == "complete"
    task_prompt = SessionPaths(dry.root).task_prompt.read_text(encoding="utf-8")
    assert "This is session 3" in task_prompt and "library_program_1.py" in task_prompt
    assert "kept in session 2 of this classifier" in " ".join(task_prompt.split("## Tools")[0].split())
    with pytest.raises(fscil.FscilHarnessError, match="final child"):  # the lineage did not go on from it
        fscil.library_entries([result.session_id], toy_sessions_dir(), target_instance_id=row_on_base(toy, cid))
    with pytest.raises(InstanceRefused, match="library"):
        run_session(target, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                    retention_floor_rows=toy_floor(3), retention_floor_source=toy_source(3),
                    library_sessions=[dry.root.name])


def toy_sessions_dir() -> Path:
    return Path(os.environ["WEIGHTCODE_RUNS"]) / "sessions"


def row_on_base(toy: ToyFscil, cid: str) -> str:
    """A session-3 instance on another student than the session's final child (so not on its chain's next step)."""
    parent = toy.root / "parent.npz"
    if not parent.exists():
        save_params(parent, toy.base_params)
    other = FS.build_fscil_instance(2, lineage="Y", run_id="r1")["instance_id"]
    assert other == cid
    return str(FS.build_fscil_instance(3, lineage="Y", parent_params=parent, parent_instance=cid,
                                       run_id="r1")["instance_id"])
