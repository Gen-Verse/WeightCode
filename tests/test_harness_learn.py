"""The update-program arm (LEARN): arm spec, prompts, trial assembly of the experience, the program executor, the joint
check and its final rule, scripted sessions over the real MCP server, session_info and the transfer runner.

The store is the evalkit toy store plus its named condition ``toyblur_s2`` (``test_evalkit_instances_store``), so
the clean companions, the joint oracle, the joint dev readout and the replay draw are evalkit's own
(``weightcode.harness.clean`` with its default source). A test double is installed only to plant a replay that shares an
idx with the evidence. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import ToyStore, add_condition, build_condition, build_toy_store
from test_harness_fixtures import harness_store, needs_sandbox, read_jsonl, scripted_session  # noqa: F401

from weightcode.analysis.session_info import format_table, session_info
from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.companions import build_replay, load_clean_rows, load_replay
from weightcode.evalkit.instances import build_task_v1, instances_root
from weightcode.evalkit.patch_check import predict
from weightcode.harness import clean, runner
from weightcode.harness.arms import ARMS, BASE_TOOLS, PROGRAM_MAX_CHANGED_FRAC, applied_limits, arm_spec, limits_dict
from weightcode.harness.experience import (
    EXPERIENCE_KEYS,
    LEARN_FINAL_SELECTION,
    LEARN_PROGRAM_RULE,
    ExperienceError,
    build_experience,
    evidence_idx,
    learn_task,
)
from weightcode.harness.finalize import FINAL_RULE_JOINT, KIND_BASE, KIND_CHECKED, select_final
from weightcode.harness.instance import InstanceRefused
from weightcode.harness.jsonio import file_sha256, read_json
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.leaks import HINT_WORDS, scan_texts, scan_trial
from weightcode.harness.program import ProgramProblem, Student, run_update, write_derived_patch
from weightcode.harness.prompts import Budgets, render_system, render_task
from weightcode.harness.scripted import ScriptedLauncher
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import PROGRAM_DESCRIPTIONS, TOOL_DESCRIPTIONS, tool_specs
from weightcode.harness.transfer import TransferError, run_transfer
from weightcode.harness.trial import CLEAN_VCHECK_FILE, assemble
from weightcode.patch import Limits, PatchError, apply_patch
from weightcode.sandbox.program import ProgramSandbox

CLASSES = ["c0", "c1", "c2", "c3"]
BUDGETS = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=3600.0)
SHOWN_CHECK_KEYS = {"program_file", "query_index", "score_new", "score_old", "score", "queries_remaining", "tensors",
                    "program_stdout", "program_stderr", "duration_s", "wall_s_elapsed", "wall_s_remaining"}
STRINGS = 'HYPOTHESIS = "unit 0 should read the class-0 pixel"\nPREDICTED_EFFECT = "fixes errors, breaks none"\n'
FIX = STRINGS + """import numpy as np

def update(student, experience):
    w = student["params"]["fc1.weight"]
    delta = np.zeros_like(w)
    delta[0, 4], delta[1, 4] = 6.0, -6.0
    print("rows", len(experience["y_new"]), len(experience["y_old"]))
    return {"fc1.weight": delta}
"""
BIG = STRINGS + """import numpy as np

def update(student, experience):
    delta = np.zeros_like(student["params"]["fc1.weight"])
    delta[0, 0] = 1000.0
    return {"fc1.weight": delta}
"""
WORSE = STRINGS + """import numpy as np

def update(student, experience):
    delta = np.zeros_like(student["params"]["fc2.bias"])
    delta[2] = 100.0
    return {"fc2.bias": delta}
"""
WIDE = STRINGS + """import numpy as np

def update(student, experience):
    return {"fc1.weight": np.full(student["params"]["fc1.weight"].shape, 1e-3)}
"""
NAN = STRINGS + """import numpy as np

def update(student, experience):
    return {"fc1.weight": np.full(student["params"]["fc1.weight"].shape, np.nan)}
"""
INVALID = {
    "shape": (STRINGS + "import numpy as np\ndef update(student, experience):\n    return {'fc1.weight': np.zeros((2, 2))}\n",
              "deltas['fc1.weight']: shape [2, 2], but the tensor has shape [8, 16]"),
    "unknown": (STRINGS + "import numpy as np\ndef update(student, experience):\n    return {'conv9.weight': np.zeros(3)}\n",
                "deltas: 'conv9.weight' is not a tensor of the student"),
    "nan": (NAN, "deltas['fc1.weight']: 128 entries are not finite"),
    "dtype": (STRINGS + "import numpy as np\ndef update(student, experience):\n"
              "    return {'fc2.bias': np.zeros(4, dtype=np.int64)}\n",
              "deltas['fc2.bias']: dtype int64; a delta must be float16, float32 or float64"),
    "strings": ("import numpy as np\ndef update(student, experience):\n    return {}\n",
                "the program must define HYPOTHESIS as a non-empty str at module level; it is missing"),
    "raises": (STRINGS + "def update(student, experience):\n    raise KeyError('no such array')\n",
               "the program failed in update (it ran as /work/in/<program>)"),
    "import": (STRINGS + "import numpy as np\nthis is not python\n", "the program failed while the program was imported"),
}
REL_FROB_TEXT = "tensor 'fc1.weight': the relative Frobenius norm of the change is above the limit 0.5"
CHANGED_TEXT = "size: the fraction of parameters that change is above the limit 0.2"


@dataclass
class LearnStore:
    """The toy store with the condition toyblur_s2, its dev instances by seed and the replay draw."""

    store: ToyStore
    ids: dict[int, str]

    def instance(self, seed: int = 100) -> Instance:
        return load_instance(instances_root() / self.ids[seed])

    @property
    def sessions(self) -> Path:
        return self.store.runs / "sessions"


@pytest.fixture
def learn_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> LearnStore:
    """Under a neutral directory name, as ``harness_store`` (the sandbox self-test sees the session's host paths)."""
    store = build_toy_store(tmp_path_factory.mktemp("hs") / "store")
    store.activate(monkeypatch)
    add_condition(store)
    outcomes = build_condition(store, roles=("dev",))
    build_replay(store.dataset, n=8)
    return LearnStore(store, {o.seed: o.instance_id for o in outcomes if o.split_role == "dev"})


def _assembled(store: LearnStore, tmp_path: Path, seed: int = 100) -> tuple[Any, Path]:
    session_dir = tmp_path / f"session{seed}"
    (session_dir / "private").mkdir(parents=True)
    return assemble(store.instance(seed).instance_dir, "LEARN", session_dir), session_dir


def _student(trial: Path) -> Student:
    return Student(params=load_params(trial / "model.npz"), params_file=trial / "model.npz",
                   forward_file=trial / "forward.py", input_files={}, layout=read_json(trial / "model_spec.json"))


def _sandbox() -> ProgramSandbox:
    return ProgramSandbox(python=runner.SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)


def _write_program(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def _paragraphs(text: str) -> list[str]:
    return text.split("\n\n")


def _sentences(paragraph: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", " ".join(paragraph.split()))


# ---------------------------------------------------------------------------------------------------------------
# Arm spec, limits, tools and prompts


def test_the_spec_is_m2_with_a_program_and_the_program_limits() -> None:
    spec, m2 = arm_spec("LEARN"), arm_spec("M2")
    assert "LEARN" in ARMS and spec.program and not m2.program
    assert (spec.shown, spec.evidence, spec.apply_mode, spec.size_limits, spec.labels, spec.tools) == \
        (m2.shown, m2.evidence, m2.apply_mode, m2.size_limits, m2.labels, BASE_TOOLS)
    limits = build_task_v1(CLASSES)["limits"]
    program, patch = applied_limits(limits, "LEARN"), applied_limits(limits, "M2")
    assert program.max_edit_frac is None and patch.max_edit_frac == Limits().max_edit_frac == 0.05
    assert patch.max_changed_frac is None and program.max_changed_frac == PROGRAM_MAX_CHANGED_FRAC == 0.20
    for name in ("max_rel_frob", "rel_frob_min_ndim", "max_ops", "max_rank", "max_file_bytes"):
        assert getattr(program, name) == getattr(patch, name), name

    specs = {item["name"]: item for item in tool_specs(spec.tools, program=True)}
    assert list(specs) == list(BASE_TOOLS)
    assert specs["python"]["description"] == TOOL_DESCRIPTIONS["python"]
    for name in ("patch_check", "status", "declare_final"):
        assert specs[name]["description"] == PROGRAM_DESCRIPTIONS[name]
    assert specs["patch_check"]["inputSchema"]["required"] == ["program_file"]
    assert set(specs["declare_final"]["inputSchema"]["properties"]) == {"program_file", "rationale"}
    texts = {name: item["description"] for name, item in specs.items()}
    assert scan_texts(texts, (*HINT_WORDS, "seed", "rewrite", "decoy", "noweight")) == {}


def test_a_program_may_change_every_tensor_of_a_student_with_more_tensors_than_max_ops(tmp_path: Path) -> None:
    """The derived patch has one op per changed tensor, so in a program arm max_ops is raised to the tensor count
    (the 70-tensor seqops student of PREREG E5); on a student of at most 64 tensors the limits are unchanged."""
    limits = build_task_v1(CLASSES)["limits"]
    base = applied_limits(limits, "LEARN")
    assert base.max_ops == Limits().max_ops == 64
    assert applied_limits(limits, "LEARN", 10) == applied_limits(limits, "LEARN", 64) == base
    wide = applied_limits(limits, "LEARN", 70)
    assert wide.max_ops == 70 and dataclasses.replace(wide, max_ops=base.max_ops) == base
    assert limits_dict(applied_limits(limits, "LEARN", 12)) == limits_dict(base)
    assert applied_limits(limits, "M2", 70) == applied_limits(limits, "M2")
    for bad in (0, -1, True, 7.0, "70"):
        with pytest.raises(ValueError, match="n_tensors"):
            applied_limits(limits, "LEARN", bad)  # type: ignore[arg-type]

    params = {f"t{index:02d}": np.ones((2, 5), np.float32) for index in range(70)}
    deltas = {}
    for name in params:
        delta = np.zeros((2, 5), np.float32)
        delta[0, 0] = 0.01  # one entry of ten per tensor: 10% changed, rel-Frob about 0.003
        deltas[name] = delta
    update = SimpleNamespace(deltas=deltas, hypothesis="h", predicted_effect="p")
    patch = write_derived_patch(update, params, params_sha256(params), tmp_path)  # type: ignore[arg-type]
    assert len(patch["ops"]) == 70
    with pytest.raises(PatchError, match="has 70 ops; at most 64 are allowed"):
        apply_patch(params, patch, tmp_path, base, params_sha256(params))
    result = apply_patch(params, patch, tmp_path, wide, params_sha256(params))
    assert result.changed_frac == pytest.approx(0.1)
    assert all(result.child[name][0, 0] == np.float32(1.01) for name in params)


FALSE_FOR_A_PROGRAM = {
    "After the session, among the patches you checked with `patch_check`, the one with the highest S = fixed - broken "
    "is kept.",
    "fixed counts check-set examples the original model got wrong and the patched model gets right; broken counts "
    "check-set examples the original model got right and the patched model gets wrong.",
    "The check set is held out from the deployment data and you never see its examples.",
    "If no checked patch has S > 0, the original model is kept.",
    "Unchecked patches are never kept.",
    "Every patch states a falsifiable `hypothesis` and a `predicted_effect` (on accuracy, and on the examples the "
    "model already gets right).",
    "Write patches, and the .npy files they reference, to /work/patches, and anything else to /work/scratch.",
    "A patch counts only once you pass its file name to `patch_check`.",
    "When you have finished, call `declare_final` with the patch you want kept (or null for the original model) and "
    "a short rationale, then stop.",
    "The declaration is recorded as your own choice; it does not change which patch is kept under the rule above.",
}


def test_the_system_prompt_changes_only_the_sentences_false_for_a_program() -> None:
    shared, learn = render_system("M2"), render_system("LEARN")
    old, new = _paragraphs(shared), _paragraphs(learn)
    assert len(old) == len(new)
    changed = [index for index, (a, b) in enumerate(zip(old, new)) if a != b]
    headings = [paragraph.splitlines()[0] for paragraph in old]
    assert [headings[index - 1] for index in changed] == ["## What is kept", "## How to work", "## Environment",
                                                          old[headings.index("## Environment") + 1].splitlines()[0]]
    old_sentences = {sentence for index in changed for sentence in _sentences(old[index])}
    new_sentences = {sentence for index in changed for sentence in _sentences(new[index])}
    assert old_sentences - new_sentences == FALSE_FOR_A_PROGRAM
    assert "Unchecked programs are never kept." in new_sentences and "score_new + score_old" in learn
    assert "`HYPOTHESIS`" in learn and "`PREDICTED_EFFECT`" in learn and "patch" not in " ".join(
        new_sentences - old_sentences).replace("patch_check", "").replace("/work/patches", "")
    assert scan_texts({"system": learn}, (*HINT_WORDS, "seed")) == {}


def test_the_task_prompt_states_the_program_interface_and_rules() -> None:
    task = build_task_v1(CLASSES)
    limits = applied_limits(task["limits"], "LEARN")
    view = learn_task(task, "a" * 64, limits, 64, 16, 64)
    text = render_task("LEARN", view, limits_dict(limits), BUDGETS)
    assert "$" not in text and "8 queries in total" in text and "300 seconds" in text
    assert LEARN_FINAL_SELECTION in text and view["patch_check"]["readout"] in text
    assert all(rule in text for rule in LEARN_PROGRAM_RULE) and all(rule in text for rule in
                                                                   task["training_free_rule"]["forbidden"])
    assert "`update(student: dict, experience: dict) -> dict[str, numpy.ndarray]`" in text
    assert "run unchanged on other independently trained classifiers" in text
    assert "x_fail" not in text and "evidence.npz" not in text and "Allowed ops" not in text
    assert "`max_rel_frob` = 0.5" in text and "`max_changed_frac` = 0.2" in text
    assert "max_edit_frac" not in text and "max_ops" not in text
    assert set(view["limits"]) == {"max_rel_frob", "rel_frob_min_ndim", "max_changed_frac"}
    assert "evidence" not in view and "patch_format" not in view and view["parent_sha256"] == "a" * 64
    assert scan_texts({"task": text, "json": json.dumps(view)}, (*HINT_WORDS, "seed")) == {}


def test_exempt_tensors_are_shown_to_a_program_only_when_there_are_any() -> None:
    task = build_task_v1(CLASSES)
    task["limits"] = {**task["limits"], "rel_frob_exempt": ["fc2.weight", "fc2.bias"]}
    limits = applied_limits(json.loads(json.dumps(task["limits"])), "LEARN")
    assert limits.rel_frob_exempt == ("fc2.weight", "fc2.bias")
    view = json.loads(json.dumps(learn_task(task, "a" * 64, limits, 64, 16, 64)))
    assert view["limits"]["rel_frob_exempt"] == ["fc2.weight", "fc2.bias"]
    assert "so are the tensors in rel_frob_exempt" in view["limit_definitions"]["max_rel_frob"]
    text = render_task("LEARN", view, limits_dict(limits), BUDGETS)
    assert '`rel_frob_exempt` = ["fc2.weight", "fc2.bias"]: tensors that max_rel_frob does not apply to' in text
    plain = build_task_v1(CLASSES)
    plain_limits = applied_limits(plain["limits"], "LEARN")
    plain_text = render_task("LEARN", learn_task(plain, "a" * 64, plain_limits, 64, 16, 64), limits_dict(plain_limits),
                             BUDGETS)
    assert "rel_frob_exempt" not in plain_text
    m2_text = render_task("M2", plain, limits_dict(applied_limits(plain["limits"], "M2")), BUDGETS)
    assert "rel_frob_exempt" not in m2_text and "does not apply to" not in m2_text


def test_the_final_rule_is_argmax_joint_with_the_patch_tie_breaks() -> None:
    def entry(name: str, score: int, n_changed: int, query: int) -> dict[str, Any]:
        return {"patch_id": name, "score": score, "n_changed": n_changed, "query_index": query}

    tied = [entry("a.py", 5, 9, 0), entry("b.py", 5, 2, 1), entry("c.py", 4, 1, 2)]
    assert select_final(tied, budget=8, declared=None)["patch_id"] == "b.py"
    same = [entry("a.py", 5, 2, 0), entry("b.py", 5, 2, 1)]
    assert select_final(same, budget=8, declared=None)["patch_id"] == "a.py"
    assert select_final([entry("a.py", 0, 1, 0), entry("b.py", -3, 1, 1)], budget=8, declared=None)["kind"] == KIND_BASE


# ---------------------------------------------------------------------------------------------------------------
# Trial assembly


def test_the_trial_holds_the_experience_and_no_condition_name(learn_store: LearnStore, tmp_path: Path) -> None:
    instance = learn_store.instance()
    view, session_dir = _assembled(learn_store, tmp_path)
    trial = session_dir / "trial"
    assert sorted(path.name for path in trial.iterdir()) == ["experience.npz", "forward.py", "model.npz",
                                                             "model_spec.json", "task.json"]
    with np.load(trial / "experience.npz", allow_pickle=False) as archive:
        experience = {key: archive[key] for key in archive.files}
    assert sorted(experience) == sorted(EXPERIENCE_KEYS)
    evidence, replay = instance.evidence, load_replay(learn_store.store.dataset)
    assert np.array_equal(experience["x_new"], np.concatenate([evidence["x_fail"], evidence["x_rand"]]))
    assert np.array_equal(experience["y_new"], np.concatenate([evidence["y_fail"], evidence["y_rand"]]))
    assert np.array_equal(experience["split_ids_new"], evidence["split_ids"])
    assert np.array_equal(experience["x_old"], replay["x"]) and np.array_equal(experience["y_old"], replay["y"])
    assert np.array_equal(experience["pred_old"], predict(instance.forward, instance.params, replay["x"], 64))
    assert np.array_equal(experience["pred_new"], predict(instance.forward, instance.params, experience["x_new"], 64))
    replay_idx, shown_idx = set(replay["idx"].tolist()), set(evidence_idx(instance).tolist())
    assert len(shown_idx) == len(experience["y_new"]) and not replay_idx & shown_idx
    assert view.experience["replay_idx_disjoint_from_evidence_idx"] is True
    assert view.experience["replay"]["idx"] == replay["idx"].tolist()
    assert view.experience["replay"]["split_seed"] is not None and view.experience["replay"]["content_sha256"]

    task = read_json(trial / "task.json")
    assert task["experience"]["new"]["count"] == len(experience["y_new"])
    assert task["experience"]["old"]["count"] == len(experience["y_old"])
    record = runner.read_instance_record(instance.instance_id)
    forbidden = runner.session_forbidden_strings(record)
    assert "toyblur_s2" in forbidden.text and scan_trial(trial, forbidden.text) == {}

    companion = load_clean_rows(instance, "V_check")
    with np.load(session_dir / "private" / CLEAN_VCHECK_FILE, allow_pickle=False) as archive:
        private = {key: archive[key] for key in archive.files}
    assert all(np.array_equal(private[key], value) for key, value in companion.arrays().items())
    assert view.clean_check["sha256"] == file_sha256(session_dir / "private" / CLEAN_VCHECK_FILE)
    assert view.clean_check["clean_sha256"] == companion.clean_sha256


def test_a_replay_sharing_an_evidence_idx_is_refused(learn_store: LearnStore) -> None:
    instance = learn_store.instance()
    real = clean.evalkit_source()
    planted = int(evidence_idx(instance)[0])

    def load_planted(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays, record = real.load_replay(dataset)
        idx = arrays["idx"].copy()
        idx[0] = planted
        return {**arrays, "idx": idx}, record

    previous = clean.set_source(clean.CleanSource("planted", load_planted, real.load_clean_check,
                                                  real.dev_readout_joint))
    try:
        with pytest.raises(ExperienceError, match="a clean view of an evidence image must never reach the editor"):
            build_experience(instance)
    finally:
        clean.set_source(previous)


def test_a_learn_session_without_companions_is_refused_before_any_write(harness_store: Any) -> None:  # noqa: F811
    instance_id = harness_store.dev()  # condition deploy: no replay draw and no clean companion
    launcher = ScriptedLauncher(lambda client, replies: None)
    with pytest.raises(InstanceRefused, match="refused for LEARN"):
        runner.run_session(instance_id, "LEARN", 0, effort="low", wall_timeout_s=60.0, launcher=launcher)
    assert not harness_store.sessions.exists() or not any(harness_store.sessions.iterdir())


# ---------------------------------------------------------------------------------------------------------------
# Program executor


@needs_sandbox
def test_a_good_program_runs_deterministically(learn_store: LearnStore, tmp_path: Path) -> None:
    _, session_dir = _assembled(learn_store, tmp_path)
    trial, student = session_dir / "trial", _student(session_dir / "trial")
    program = _write_program(tmp_path, "fix.py", FIX)
    first = run_update(student, trial / "experience.npz", program, tmp_path / "run1", _sandbox())
    second = run_update(student, trial / "experience.npz", program, tmp_path / "run2", _sandbox())
    assert first.tensors == ["fc1.weight"] and first.record["deltas_sha256"] == second.record["deltas_sha256"]
    assert first.record["inputs_sha256"] == second.record["inputs_sha256"]
    assert set(first.record["inputs_sha256"]) == {"experience.npz", "fix.py", "forward.py", "layout.json", "params.npz"}
    assert first.deltas["fc1.weight"][0, 4] == 6.0 and np.count_nonzero(first.deltas["fc1.weight"]) == 2
    assert first.hypothesis.startswith("unit 0") and first.record["shown"]["program_stdout"].startswith("rows ")
    assert not (tmp_path / "run1" / "in").exists()


@needs_sandbox
@pytest.mark.parametrize("case", sorted(INVALID))
def test_invalid_programs_are_reported_verbatim(learn_store: LearnStore, tmp_path: Path, case: str) -> None:
    _, session_dir = _assembled(learn_store, tmp_path)
    source, expected = INVALID[case]
    program = _write_program(tmp_path, f"{case}.py", source)
    with pytest.raises(ProgramProblem) as caught:
        run_update(_student(session_dir / "trial"), session_dir / "trial" / "experience.npz", program,
                   tmp_path / "run", _sandbox())
    assert any(message.startswith(expected) for message in caught.value.messages), caught.value.messages
    assert caught.value.run is not None and "shown" in caught.value.run


@needs_sandbox
def test_a_program_past_the_wall_limit_is_stopped(learn_store: LearnStore, tmp_path: Path) -> None:
    _, session_dir = _assembled(learn_store, tmp_path)
    program = _write_program(tmp_path, "slow.py", STRINGS + "import time\ndef update(student, experience):\n"
                                                            "    time.sleep(60)\n    return {}\n")
    with pytest.raises(ProgramProblem) as caught:
        run_update(_student(session_dir / "trial"), session_dir / "trial" / "experience.npz", program,
                   tmp_path / "run", _sandbox(), wall_s=2.0)
    assert caught.value.messages == ["the program did not return within 2 seconds and was stopped"]
    assert caught.value.run is not None and caught.value.run["run"]["status"] == "timeout"


# ---------------------------------------------------------------------------------------------------------------
# Scripted sessions, session_info and transfer


def _program_cell(programs: dict[str, str]) -> str:
    payload = json.dumps(json.dumps(programs))
    return ("import json\n"
            f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n"
            f"print(sorted(json.loads({payload})))\n")


@needs_sandbox
def test_a_learn_session_checks_programs_on_the_joint_score(learn_store: LearnStore) -> None:
    steps = [
        ("python", {"code": _program_cell({"fix.py": FIX, "big.py": BIG, "nan.py": NAN, "worse.py": WORSE,
                                           "wide.py": WIDE})}),
        ("patch_check", {"program_file": "fix.py"}),
        ("patch_check", {"program_file": "big.py"}),
        ("patch_check", {"program_file": "wide.py"}),
        ("patch_check", {"program_file": "nan.py"}),
        ("patch_check", {"program_file": "worse.py"}),
        ("patch_check", {"program_file": "fix.py"}),
        ("patch_check", {"patch_file": "fix.py"}),
        ("status", {}),
        ("declare_final", {"program_file": "fix.py", "rationale": "it scored best"}),
    ]
    result, replies = scripted_session(learn_store.ids[100], "LEARN", steps)
    assert result.status == "complete", result.record.get("infra_reasons")
    python, fix, big, wide, nan, worse, again, wrong_key, status, declared = replies
    assert not python.is_error and not fix.is_error and not worse.is_error
    assert set(fix.payload) == SHOWN_CHECK_KEYS and set(worse.payload) == SHOWN_CHECK_KEYS
    assert fix.payload["score"] == fix.payload["score_new"] + fix.payload["score_old"] > 0
    assert worse.payload["score"] == worse.payload["score_new"] + worse.payload["score_old"] < 0
    assert fix.payload["queries_remaining"] == 7 and worse.payload["queries_remaining"] == 6
    assert fix.payload["tensors"] == ["fc1.weight"] and fix.payload["program_stdout"].startswith("rows ")
    assert big.is_error and big.payload["error"][0] == REL_FROB_TEXT and big.payload["queries_remaining"] == 7
    assert wide.is_error and wide.payload["error"][0] == CHANGED_TEXT and wide.payload["queries_remaining"] == 7
    assert nan.is_error and nan.payload["error"][0].startswith("deltas['fc1.weight']: 128 entries are not finite")
    assert again.is_error and "already checked" in again.payload["error"][0]
    assert wrong_key.is_error and "unknown arguments ['patch_file']" in wrong_key.payload["error"][0]
    assert [row["program_file"] for row in status.payload["checked"]] == ["fix.py", "worse.py"]
    assert set(status.payload["checked"][0]) == {"program_file", "query_index", "score_new", "score_old", "score"}
    assert declared.payload["valid"] and declared.payload["checked"]

    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    assert session["experience"]["replay"]["idx"] and session["clean_check"]["split"] == "V_check"
    assert session["program"]["wall_s"] == 300.0
    final = result.record["final"]
    assert final["kind"] == KIND_CHECKED and final["patch_id"] == "fix.py" and final["score"] == fix.payload["score"]
    assert result.record["queries"] == {"used": 2, "consumed": 2, "charged_failures": 0, "budget": 8}
    assert "kept_rule" not in session and "s_old_floor" not in read_json(paths.selection_json)  # the v1 rule
    assert (final["score_new"], final["score_old"]) == (fix.payload["score_new"], fix.payload["score_old"])
    program = final["program"]
    assert program["sha256"] == file_sha256(result.root / program["file"]) == file_sha256(result.root /
                                                                                          "attempts" / final["attempt"]
                                                                                          / "fix.py")
    assert program["deltas_sha256"] == file_sha256(result.root / program["deltas"])
    assert program["patch_sha256"] == file_sha256(result.root / "final" / "patch.json")
    child = load_params(result.root / "final" / "model.npz")
    real = learn_store.instance().params
    expected = {key: value.copy() for key, value in real.items()}
    expected["fc1.weight"][0, 4] += np.float32(6.0)
    expected["fc1.weight"][1, 4] -= np.float32(6.0)
    assert params_sha256(child) == final["child_sha256"] == params_sha256(expected)
    run = read_json(result.root / "run.json")
    assert run["final"]["program"] == program and run["final"]["score_new"] == final["score_new"]
    selection = read_json(paths.selection_json)
    assert selection["rule"] == FINAL_RULE_JOINT
    assert [row["program_file"] for row in selection["candidates"]] == ["fix.py", "worse.py"]
    checks = read_jsonl(paths.checks_log)
    assert all(entry["patch_file"] == "derived_patch.json" for entry in checks)
    ledger = CheckLedger(session, paths)  # the replay re-runs nothing but re-applies and re-scores every check
    assert [entry["score"] for entry in ledger.checked()] == [fix.payload["score"], worse.payload["score"]]
    rejected = result.record["rejected_attempts"]
    assert {row["program_file"]: row["stage"] for row in rejected} == {"big.py": "limits", "wide.py": "limits",
                                                                         "nan.py": "program"}
    assert any("rel_frob" in message or "Frobenius" in message for message in rejected[0]["messages"])

    info = session_info(result.root)
    rows = info["trajectory"]
    assert [row["patch_id"] for row in rows] == ["fix.py", "worse.py"]
    assert rows[0]["vcheck"]["score_new"] == fix.payload["score_new"] and rows[0]["vselect_clean"] is not None
    assert info["final"]["program"] == program and info["final"]["vselect_clean"] == rows[0]["vselect_clean"]
    assert info["access"]["evidence_arrays"].keys() == set(EXPERIENCE_KEYS)
    assert "S_joint" in format_table(info) and "dacc_clean" in format_table(info)


@needs_sandbox
def test_the_transfer_runner_reads_out_a_frozen_program(learn_store: LearnStore, tmp_path: Path) -> None:
    fix = _write_program(tmp_path, "fix.py", FIX)
    big = _write_program(tmp_path, "big.py", BIG)
    out = tmp_path / "transfer"
    written = run_transfer(fix, [learn_store.ids[100], learn_store.ids[101]], out, _sandbox())
    records = [json.loads(path.read_text()) for path in written]
    assert len(records) == 2 and {record["target"]["instance_id"] for record in records} == set(learn_store.ids.values())
    for record, seed in zip(records, (100, 101)):
        assert record["status"] == "ok" and record["within_limits"] is True and record["readout_child"] == "within_limits"
        assert record["program"]["sha256"] == file_sha256(fix) and record["tensors"] == ["fc1.weight"]
        assert record["readouts"]["delta_acc"] is not None and record["readouts_clean"]["delta_acc"] is not None
        view, _ = _assembled(learn_store, tmp_path, seed)
        assert record["experience"]["content_sha256"] == view.experience["content_sha256"]
    refused = run_transfer(big, [learn_store.ids[100]], out, _sandbox())
    record = json.loads(refused[0].read_text())
    assert record["within_limits"] is False and record["readout_child"] == "unbounded"
    assert record["limit_messages"] and "fc1.weight" in record["limit_messages"][0]
    assert record["readouts"]["delta_acc"] is not None
    with pytest.raises(TransferError, match="never overwritten"):
        run_transfer(fix, [learn_store.ids[100]], out, _sandbox())
