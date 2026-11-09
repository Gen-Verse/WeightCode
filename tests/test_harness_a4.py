"""PREREG A4 in the harness: the blind arm LEARN_BLIND (A4a) and the B arm with at most 2 checks (A4b).

Sessions on the toy E11 store (``e11_toy``) through the real MCP server and sandbox, running the probe programs of
``a4_view`` (each pushes the student's weights out of its run by one route a program has). The B arm matches the namespace-migrated
reference from main 9a65b20 (``a4_view.B_SHOWN_WEIGHTCODE``), and the leak detectors used on the blind arm
fire on it (a positive control). The blind arm shows no parameter value in its trial, its kernel, its prompts, its tool
list or any tool reply; no program output, exception text, numeric exit code, key, count or tensor name from a run
reaches a reply; the host logs keep the full text; its programs get exactly the LEARN inputs; and the final read on
V_select and the session table run as for B. The 2-check arm is
the B arm with the budget 8 lowered to 2, and refuses the third check. Units: the blind delta problems and limit
rejections, the blind task view (predictions hidden by default, kept on request), the template diff, the tool
descriptions and the argument refusals.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from a4_view import B_SHOWN_WEIGHTCODE, PROBE_ORDER, VALUE_MARKERS, probe_session, shown_digest, trial_digest
from e11_toy import PLANT_CONDITION, e11_store  # noqa: F401
from test_harness_fixtures import needs_sandbox
from test_harness_learn import STRINGS, _program_cell

from weightcode.analysis.session_info import format_table, session_info
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.devreadout import readout_run
from weightcode.harness import runner
from weightcode.harness.applying import BLIND_OTHER_TEXT, NON_FINITE_TEXT, blind_rejection_messages
from weightcode.harness.arms import BASE_TOOLS, arm_spec
from weightcode.harness.experience import BLIND_HIDDEN_KEYS, BLIND_OUTPUT, EXPERIENCE_KEYS
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import json_text, read_json
from weightcode.harness.program import (
    BLIND_BAD_KEY,
    BLIND_BAD_SHAPE,
    BLIND_NON_FINITE,
    RUN_RECORD,
    ProgramProblem,
    load_deltas,
)
from weightcode.harness.prompts import PROMPT_DIR, Budgets, render_task
from weightcode.harness.scripted import ScriptedLauncher, call_and_record
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import BLIND_RETURNS, PROGRAM_RETURNS, tool_specs
from weightcode.harness.trial import arm_task, assemble
from weightcode.models import vit_synth as V
from weightcode.patch import Limits

WALL_S = 300.0
BLIND_TRIAL = ["experience.npz", "forward.py", "model_shapes.json", "model_spec.json", "task.json"]
BLIND_EXPERIENCE = sorted(key for key in EXPERIENCE_KEYS if key not in BLIND_HIDDEN_KEYS)
TIMING = ("wall_s_elapsed", "wall_s_remaining")


def _instance(store: V.SyntheticStore) -> str:
    return store.ids[(PLANT_CONDITION, 100)]


def _needles(params: dict[str, np.ndarray]) -> tuple[list[bytes], list[str]]:
    """The first four float32 values of every tensor whose first eight are not near-constant, as raw bytes and as the
    texts numpy and Python print for them (constant tensors, such as LayerNorm ones and zero biases, identify
    nothing)."""
    raw, texts = [], []
    for value in params.values():
        flat = np.asarray(value, dtype=np.float32).ravel()
        if flat.size < 8 or np.unique(flat[:8]).size < 8:
            continue
        raw.append(flat[:4].tobytes())
        for item in flat[:4]:
            texts.extend(text for text in (repr(float(item)), str(item)) if len(text) >= 8)
    assert raw and texts
    return raw, texts


def _texts(root: Path, replies: list[Any], listed: list[Any]) -> str:
    """Every text the editor saw: the prompts, the tool list, every reply and every text file of the trial."""
    parts = [(root / "prompts" / "system.md").read_text(), (root / "prompts" / "task.md").read_text(),
             json.dumps(listed), json.dumps([reply.payload for reply in replies])]
    parts += [path.read_text() for path in sorted((root / "trial").iterdir()) if path.suffix != ".npz"]
    return "\n".join(parts)


def _attempts(root: Path) -> dict[str, Path]:
    """``{program file: attempt directory}`` of a session."""
    found = {}
    for path in sorted((root / "attempts").iterdir()):
        found[read_json(path / "attempt.json")["program_file"]] = path
    return found


KERNEL_PROBE = """_globals = sorted(name for name in globals() if not name.startswith("_"))
import json, os
import numpy as np
_files = sorted(os.path.join(root, name) for root, _, names in os.walk("/work") for name in names)
with np.load("/work/trial/experience.npz") as _archive:
    _keys = sorted(_archive.files)
print(json.dumps({"globals": _globals, "files": _files, "experience": _keys, "store": os.path.exists(STORE),
                  "work_in": os.path.exists("/work/in"), "work_out": os.path.exists("/work/out")}))
"""


@needs_sandbox
def test_the_b_arm_matches_the_namespace_migrated_pre_a4_reference(e11_store: V.SyntheticStore) -> None:
    instance_id = _instance(e11_store)
    result, replies, listed = probe_session(instance_id, "LEARN")
    assert result.status == "complete", result.record.get("infra_reasons")
    assert shown_digest(result.root, replies, listed) == B_SHOWN_WEIGHTCODE
    session = read_json(result.root / "session.json")
    assert "blind" not in session and "check_cap" not in session and "blind" not in session["trial"]
    assert session["budget"]["patch_checks"] == 8
    # positive control: the detectors of the blind test find the weights and the program output in B
    raw, texts = _needles(load_params(read_instance_record(instance_id).instance_dir / "model.npz"))
    assert all(needle in (result.root / "trial" / "model.npz").read_bytes() for needle in raw)
    shown = json.dumps([reply.payload for reply in replies])
    assert all(marker in shown for marker in ("TALK_VALUE", "TALK_ERR", "BOOM_VALUE", "KEY_VALUE", "FIELD_"))
    assert any(text in shown for text in texts)  # talk.py prints qkv values to its stdout
    assert re.search(r"\d+ entries are not finite", shown) and re.search(r"exit status \d+", shown)


@needs_sandbox
def test_a_blind_session_shows_no_parameter_value_and_no_program_output(e11_store: V.SyntheticStore) -> None:
    instance_id = _instance(e11_store)
    probe = f"STORE = {str(e11_store.root)!r}\n" + KERNEL_PROBE
    result, replies, listed = probe_session(instance_id, "LEARN_BLIND", kernel_cells=(probe,))
    assert result.status == "complete", result.record.get("infra_reasons")
    root = result.root
    real = load_params(read_instance_record(instance_id).instance_dir / "model.npz")

    # the trial: shapes in place of weights, the experience without predictions
    assert sorted(path.name for path in (root / "trial").iterdir()) == BLIND_TRIAL
    with np.load(root / "trial" / "experience.npz") as archive:
        assert sorted(archive.files) == BLIND_EXPERIENCE
    shapes = read_json(root / "trial" / "model_shapes.json")
    assert [(item["name"], item["shape"]) for item in shapes["tensors"]] == [
        (name, list(value.shape)) for name, value in real.items()]

    # the kernel: an empty namespace, no file of weights, no host path, no program run directory
    kernel = json.loads(replies[0].payload["stdout"])
    assert kernel["globals"] == ["STORE"] and kernel["experience"] == BLIND_EXPERIENCE
    assert kernel["files"] == [f"/work/trial/{name}" for name in BLIND_TRIAL]
    assert not kernel["store"] and not kernel["work_in"] and not kernel["work_out"]

    # no parameter value in any trial file (raw float32 bytes) or any text the editor saw
    raw, texts = _needles(real)
    for path in (root / "trial").iterdir():
        assert not any(needle in path.read_bytes() for needle in raw), path.name
    seen = _texts(root, replies, listed)
    assert not [text for text in texts if text in seen]
    assert not [marker for marker in VALUE_MARKERS if marker in seen]
    assert not re.search(r"exit status \d", seen) and not re.search(r"\d+ entries", seen)
    tensor_names = [name for name in real if name in json.dumps([reply.payload for reply in replies])]
    assert tensor_names == []  # no reply names a tensor (the program chooses which tensors fail)

    # what each check returned
    checks = dict(zip(PROBE_ORDER, replies[2:2 + len(PROBE_ORDER)]))
    assert [checks[name].is_error for name in PROBE_ORDER] == [False, True, True, True, True, True, False, True]
    for name in ("talk.py", "ln.py"):
        payload = {key: value for key, value in checks[name].payload.items() if key not in TIMING}
        assert sorted(payload) == ["exit_status", "program_file", "queries_remaining", "query_index", "score",
                                   "score_new", "score_old"] and payload["exit_status"] == "ok"
    status = {name: checks[name].payload["exit_status"] for name in PROBE_ORDER}
    assert status == {"talk.py": "ok", "boom.py": "error", "badkey.py": "ok", "dtype.py": "error", "bits.py": "ok",
                      "exit7.py": "died", "ln.py": "ok", "qkv_big.py": "ok"}
    assert checks["badkey.py"].payload["error"][:2] == [BLIND_BAD_KEY, BLIND_BAD_SHAPE]
    assert checks["bits.py"].payload["error"][0] == BLIND_NON_FINITE
    assert checks["qkv_big.py"].payload["error"][0].startswith(
        "a tensor: the relative Frobenius norm of the change is above the limit 0.5")
    assert "exit status error" in checks["boom.py"].payload["error"][0]
    assert "exit status died" in checks["exit7.py"].payload["error"][0]

    # the host logs keep everything the editor was not shown
    attempts = _attempts(root)
    talk_run = read_json(attempts["talk.py"] / RUN_RECORD)
    assert "TALK_VALUE" in talk_run["run"]["stdout"] and "TALK_ERR" in talk_run["run"]["stderr"]
    assert "TALK_FILE" in (attempts["talk.py"] / "run" / "out" / "notes.txt").read_text()
    for name, marker in (("boom.py", "BOOM_VALUE"), ("badkey.py", "KEY_VALUE"), ("dtype.py", "FIELD_"),
                         ("qkv_big.py", "classifier.blocks.0.self_attn.qkv.weight")):
        record = read_json(attempts[name] / "attempt.json")
        assert record["redaction"] == "blind" and marker in json.dumps(record["messages"]), name
        assert marker not in json.dumps(record["editor_messages"]), name

    # the programs got exactly the LEARN inputs: B's weights and B's whole experience, by content and by file
    session = read_json(root / "session.json")
    blind = session["blind"]
    assert blind["params_sha256"] == B_SHOWN_WEIGHTCODE["trial"]["model.npz"].removeprefix("arrays:")
    assert blind["experience_content_sha256"] == B_SHOWN_WEIGHTCODE["trial"]["experience.npz"].removeprefix("arrays:")
    assert blind["predictions_shown"] is False and sorted(blind["trial_experience_keys"]) == BLIND_EXPERIENCE
    assert talk_run["inputs_sha256"]["params.npz"] == blind["params_file_sha256"]
    assert talk_run["inputs_sha256"]["experience.npz"] == blind["experience_file_sha256"]
    for key in ("params_path", "experience_path"):
        assert Path(blind[key]).parent == SessionPaths(root).private
    assert session["budget"]["patch_checks"] == 8 and "check_cap" not in session

    # everything else is B's: the system prompt, the score fields, the final selection
    digest = shown_digest(root, replies, listed)
    assert digest["system_prompt"] == B_SHOWN_WEIGHTCODE["system_prompt"]
    assert {key: digest["trial"][key] for key in ("forward.py", "model_spec.json")} == {
        key: B_SHOWN_WEIGHTCODE["trial"][key] for key in ("forward.py", "model_spec.json")}
    final = read_json(SessionPaths(root).selection_json)["final"]
    assert final["patch_id"] in ("talk.py", "ln.py")
    # the final read on V_select and the session table work as for B
    readout = readout_run(root / "run.json", clean=True)
    assert readout["n"] == readout["n_clean"] == 160 and readout["readouts"]["delta_acc"] > 0
    info = session_info(root)
    assert format_table(info) and info["arm"] == "LEARN_BLIND"


def _variant(factor: float) -> str:
    return STRINGS + f"""import numpy as np

def update(student, experience):
    return {{"classifier.norm.weight": {factor!r} * np.asarray(student["params"]["classifier.norm.weight"],
                                                              dtype=np.float64)}}
"""


QKV_BIG = STRINGS + """import numpy as np

def update(student, experience):
    w = student["params"]["classifier.blocks.0.self_attn.qkv.weight"]
    return {"classifier.blocks.0.self_attn.qkv.weight": 0.6 * np.asarray(w, dtype=np.float64)}
"""


@needs_sandbox
def test_the_two_check_arm_is_b_with_two_checks_and_refuses_a_third(e11_store: V.SyntheticStore) -> None:
    programs = {"a.py": _variant(-0.5), "big.py": QKV_BIG, "b.py": _variant(-0.25), "c.py": _variant(0.1)}
    order = ["a.py", "big.py", "b.py", "c.py"]
    listed: list[Any] = []

    def script(client: Any, replies: list[Any]) -> None:
        listed.extend(client.list_tools())
        call_and_record(client, replies, "python", {"code": _program_cell(programs)})
        for name in order:
            call_and_record(client, replies, "patch_check", {"program_file": name})
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "a.py", "rationale": "it scored"})

    launcher = ScriptedLauncher(script)
    result = runner.run_session(_instance(e11_store), "LEARN", 0, effort="max", wall_timeout_s=WALL_S,
                                launcher=launcher, skip_audit=True, retention_floor_frac=0.005, max_checks=2)
    assert result.status == "complete", result.record.get("infra_reasons")
    replies = launcher.replies
    checks = dict(zip(order, replies[1:1 + len(order)]))
    assert [checks[name].is_error for name in order] == [False, True, False, True]
    assert checks["big.py"].payload["queries_remaining"] == 1  # a refused program uses no query
    third = json.dumps(checks["c.py"].payload)
    assert "the patch-check budget of 2 queries is spent; no query was used" in third
    assert checks["c.py"].payload["queries_remaining"] == 0
    status = replies[1 + len(order)].payload
    assert status["queries_budget"] == 2 and status["queries_remaining"] == 0
    assert [entry["program_file"] for entry in status["checked"]] == ["a.py", "b.py"]
    paths = SessionPaths(result.root)
    assert [json.loads(line)["patch_id"] for line in paths.checks_log.read_text().splitlines()] == ["a.py", "b.py"]
    session = read_json(result.root / "session.json")
    assert session["check_cap"] == {"max_checks": 2, "task_budget_queries": 8}
    assert session["budget"]["patch_checks"] == 2 and session["trial"]["max_checks"] == 2

    # B's view with 8 read as 2: the trial, both prompts and the tools are B's, byte for byte, but for that number
    digest = trial_digest(result.root / "trial")
    b_trial = B_SHOWN_WEIGHTCODE["trial"]
    assert {key: value for key, value in digest.items() if key != "task.json"} == {
        key: value for key, value in b_trial.items() if key != "task.json"}
    task = read_json(result.root / "trial" / "task.json")
    assert task["patch_check"]["budget_queries"] == 2
    task["patch_check"]["budget_queries"] = 8
    assert hashlib.sha256(json_text(task).encode()).hexdigest() == b_trial["task.json"]
    prompt = (result.root / "prompts" / "task.md").read_text()
    assert prompt.count("- `patch_check`: 2 queries in total") == 1
    as_b = prompt.replace("- `patch_check`: 2 queries in total", "- `patch_check`: 8 queries in total")
    assert hashlib.sha256(as_b.encode()).hexdigest() == B_SHOWN_WEIGHTCODE["task_prompt"]
    system = (result.root / "prompts" / "system.md").read_text()
    assert hashlib.sha256(system.encode()).hexdigest() == B_SHOWN_WEIGHTCODE["system_prompt"]
    assert hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest() == B_SHOWN_WEIGHTCODE["tools"]


def test_a4_arguments_are_refused_before_any_write(e11_store: V.SyntheticStore) -> None:
    instance_id = _instance(e11_store)
    sessions = e11_store.root / "runs" / "sessions"
    before = sorted(sessions.iterdir()) if sessions.exists() else []
    cases = [
        ("LEARN", {"max_checks": 0}, ValueError, "positive int"),
        ("LEARN", {"max_checks": True}, ValueError, "positive int"),
        ("M2NL", {"max_checks": 2}, ValueError, "has no patch_check"),
        ("LEARN", {"blind_predictions": True}, ValueError, "for the LEARN_BLIND arm"),
        ("LEARN_BLIND", {"blind_predictions": 1}, ValueError, "must be a bool"),
        ("LEARN_BLIND", {"task_prompt_override": "a prompt"}, ValueError, "prompt override"),
        ("LEARN_BLIND", {"library_sessions": ["s0123456789abcdef"]}, ValueError, "library sessions"),
        ("LEARN_BLIND", {"loop_context": Path("context.json")}, ValueError, "loop context"),
        ("LEARN", {"max_checks": 9}, InstanceRefused, "can only lower it"),
    ]
    for arm, kwargs, error, text in cases:
        with pytest.raises(error, match=text):
            runner.run_session(instance_id, arm, 0, effort="max", wall_timeout_s=WALL_S, skip_audit=True,
                               launcher=ScriptedLauncher(lambda client, replies: None), **kwargs)
    assert (sorted(sessions.iterdir()) if sessions.exists() else []) == before


def _assembled(store: V.SyntheticStore, directory: Path, arm: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    (directory / "private").mkdir(parents=True)
    return assemble(read_instance_record(_instance(store)).instance_dir, arm, directory, **kwargs)


def _bullet(prompt: str) -> str:
    start = prompt.index("- `experience.npz`")
    return prompt[start:prompt.index("- `task.json`", start)]


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append(f"{path}.{key}")
            else:
                found.extend(_differences(left[key], right[key], f"{path}.{key}"))
        return found
    return [] if left == right else [path]


def test_the_blind_view_hides_the_predictions_unless_kept(e11_store: V.SyntheticStore, tmp_path: Path) -> None:
    hidden = _assembled(e11_store, tmp_path / "hidden", "LEARN_BLIND")
    kept = _assembled(e11_store, tmp_path / "kept", "LEARN_BLIND", blind_predictions=True)
    b_view = _assembled(e11_store, tmp_path / "b", "LEARN")
    for view, keys in ((hidden, BLIND_EXPERIENCE), (kept, sorted(EXPERIENCE_KEYS))):
        with np.load(view.trial_dir / "experience.npz") as archive:
            assert sorted(archive.files) == keys
        assert not (view.trial_dir / "model.npz").exists() and view.private_files.keys() >= {"model.npz",
                                                                                              "experience.npz"}
    # kept: the trial experience is B's; hidden: B's without the predictions
    b_experience = B_SHOWN_WEIGHTCODE["trial"]["experience.npz"].removeprefix("arrays:")
    assert kept.blind["trial_experience_content_sha256"] == b_experience
    with np.load(b_view.trial_dir / "experience.npz") as archive:
        b_rows = {key: archive[key] for key in archive.files}
    assert hidden.blind["trial_experience_content_sha256"] == params_sha256(
        {key: value for key, value in b_rows.items() if key not in BLIND_HIDDEN_KEYS})
    # the blind task.json is B's but for the fields that would be false
    assert _differences(hidden.task, b_view.task) == [
        ".experience.new.keys", ".experience.old.keys", ".program.check_returns", ".program.experience",
        ".program.student.params", ".statement"]
    assert _differences(kept.task, b_view.task) == [".program.check_returns", ".program.student.params"]
    assert hidden.task["program"]["check_returns"] == BLIND_OUTPUT
    assert "predictions" not in hidden.task["statement"] and "pred_new" in hidden.task["program"]["experience"]
    assert b_view.task == arm_task(read_instance_record(_instance(e11_store)).task, "LEARN",
                                   b_view.real_model_sha256, int(b_view.experience["n_old"]))
    # the rendered experience bullet: B's when the predictions are kept, none of their keys otherwise
    budgets = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=WALL_S)
    prompts = {name: render_task(view.arm, view.task, view.limits, budgets, view.input_files)
               for name, view in (("hidden", hidden), ("kept", kept), ("b", b_view))}
    assert _bullet(prompts["kept"]) == _bullet(prompts["b"])
    assert "pred_" not in _bullet(prompts["hidden"]) and "not in the\n  file." in _bullet(prompts["hidden"])
    assert "model.npz" not in prompts["hidden"] and "model_shapes.json" in prompts["hidden"]
    assert "What the program prints is returned" in prompts["b"]
    assert "What the program prints" not in prompts["hidden"] and f"A check returns {BLIND_OUTPUT}." in prompts["hidden"]


def test_the_blind_task_template_is_learns_with_three_passages_replaced() -> None:
    learn = (PROMPT_DIR / "task_LEARN.md").read_text().splitlines()
    blind = (PROMPT_DIR / "task_LEARN_BLIND.md").read_text().splitlines()
    removed = [line[2:] for line in difflib.ndiff(learn, blind) if line.startswith("- ")]
    added = [line[2:] for line in difflib.ndiff(learn, blind) if line.startswith("+ ")]
    assert [line.split(":")[0] for line in removed] == [
        "- `model.npz`", "- `experience.npz`", "  $n_new $new_description. `x_old`, `y_old`, `pred_old`",
        "  predictions, `split_ids_new` and `split_ids_old` are opaque ids, and the inputs `x_*` are in the format that",
        "  `model_spec.json` describes.", "applied; then the updated classifier is scored. What the program prints is "
        "returned with the result."]
    assert added == ["- `model_shapes.json`: the name, shape and dtype of every weight tensor. The weights are not in "
                     "/work/trial; the", "  update program receives them when `patch_check` runs it.",
                     "$experience_bullet",
                     "applied; then the updated classifier is scored. A check returns $check_returns."]


def test_blind_delta_problems_name_no_tensor_count_or_order(tmp_path: Path) -> None:
    params = {"a": np.zeros((3, 4), np.float32), "b": np.zeros(5, np.float32), "c": np.zeros(2, np.float32)}
    nan = np.full(5, np.nan)
    sets = {"one": {"zz_1234": np.zeros(1), "a": np.zeros((2, 2)), "b": nan},
            "two": {"c": np.array([np.nan, 1.0]), "a": np.zeros((4, 3)), "qq_98765": np.zeros(7), "b": nan[:5]}}
    shown = {}
    for name, deltas in sets.items():
        path = tmp_path / f"{name}.npz"
        np.savez(path, **deltas)
        with pytest.raises(ProgramProblem) as caught:
            load_deltas(path, params, blind=True)
        shown[name] = caught.value.messages
        host = "\n".join(caught.value.host_messages)
        assert "_1234" in host or "_98765" in host
        with pytest.raises(ProgramProblem) as plain:
            load_deltas(path, params)
        assert plain.value.messages == plain.value.host_messages == caught.value.host_messages
    assert shown["one"] == shown["two"] == [BLIND_BAD_KEY, BLIND_BAD_SHAPE, BLIND_NON_FINITE]


def test_blind_limit_rejections_name_no_tensor_or_count() -> None:
    limits = Limits(max_rel_frob=0.5, max_changed_frac=0.2)
    messages = ["tensor 'w': the relative Frobenius norm of the change is above the limit 0.5",
                "size: 1234 of 5332 parameters change (fraction 0.2314), above the limit 0.2",
                "tensor 'v': the relative Frobenius norm of the change is above the limit 0.5",
                f"tensor 'u': {NON_FINITE_TEXT}", "add_dense op 3: something else"]
    shown = blind_rejection_messages(messages, limits)
    assert shown == blind_rejection_messages(list(reversed(messages)), limits)
    assert shown[0] == "size: the fraction of parameters that change is above the limit 0.2"
    assert shown[1].startswith("a tensor: the relative Frobenius norm of the change is above the limit 0.5")
    assert shown[2] == f"a tensor: {NON_FINITE_TEXT}" and shown[3] == BLIND_OTHER_TEXT and len(shown) == 4
    assert not re.search(r"'[uvw]'|1234|5332|0\.2314|op 3", "\n".join(shown))


def test_the_blind_tool_list_states_what_a_check_returns() -> None:
    blind = {tool["name"]: tool for tool in tool_specs(BASE_TOOLS, program=True, blind=True)}
    learn = {tool["name"]: tool for tool in tool_specs(BASE_TOOLS, program=True)}
    assert sorted(blind) == sorted(learn) == sorted(arm_spec("LEARN_BLIND").tools)
    assert BLIND_RETURNS in blind["patch_check"]["description"] and "stdout" not in blind["patch_check"]["description"]
    assert PROGRAM_RETURNS in learn["patch_check"]["description"]
    assert {name: tool for name, tool in blind.items() if name != "patch_check"} == {
        name: tool for name, tool in learn.items() if name != "patch_check"}
    assert blind["patch_check"]["description"].replace(BLIND_RETURNS, PROGRAM_RETURNS) == \
        learn["patch_check"]["description"]
    with pytest.raises(ValueError, match="blind"):
        tool_specs(BASE_TOOLS, program=False, blind=True)
    with pytest.raises(ValueError, match="blind"):
        tool_specs(BASE_TOOLS, program=True, blind=True, fewshot=True)
