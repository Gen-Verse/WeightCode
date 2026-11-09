"""The PREREG information ladder in the harness: U, K0b and K0a on the E11 gaussian_noise_s5 seed-100 student.

U (``e11_variant="unlabelled"``) is E11's view with ``y_new`` withheld from every file, view and tool; K0b
(``"named"``) gives no data at all and names the capability in one sentence, the one text exempt from the leak and
hint words; K0a (``"named_nocheck"``) is K0b without ``patch_check``: ``declare_final`` runs and validates the program
and scores nothing, and finalisation checks the last valid declaration once on V_check and keeps it only if
joint-retain-v1 makes it eligible. Sessions run on the toy E11 store (``e11_toy``) through the real runner, sandbox and
MCP server. The tests check by hash that no y_new array or label byte pattern reaches a U session (the standard view
is the positive control), that no image array of any split and no label reaches a K0 session, that every changed text
is found exactly once or the view is refused, that no K0a text says "check", that the hidden check and the final rule
pick nothing silently, that ``e11_variant`` is in every record, and that the ladder is refused wherever it does not
apply. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from e11_toy import PLANT_CONDITION, build_e11_store
from test_harness_fixtures import needs_sandbox
from test_harness_k1_rule import BEFORE_K1, PREREG, _differences, _view_digest, needs_prereg
from test_harness_learn import STRINGS, _program_cell

from weightcode.analysis.session_info import session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.data import store as data_store
from weightcode.evalkit.instances import scan_for_tokens
from weightcode.harness import e11, runner
from weightcode.harness.arms import arm_spec
from weightcode.harness.experience import (
    EXPERIENCE_KEYS,
    LEARN_NEW_DESCRIPTION,
    NAMED_EXPERIENCE_WORDS,
    NAMED_NOCHECK_FINAL_SELECTION,
    NAMED_PROGRAM_EXPERIENCE,
    UNLABELLED_ERROR_WORDS,
    UNLABELLED_EXPERIENCE_KEYS,
    ExperienceError,
    learn_new_selection,
    learn_task_named,
    learn_task_unlabelled,
)
from weightcode.harness.finalize import STATUS_DRY_RUN, FinalizeError, run_hidden_check
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import read_json, read_jsonl
from weightcode.harness.leaks import HINT_WORDS, scan_label_arrays
from weightcode.harness.prompts import (
    LADDER_TASK_EDITS,
    PROMPT_DIR,
    Budgets,
    ladder_task_template,
    nocheck_kept_section,
    render_system,
    render_task,
)
from weightcode.harness.scripted import McpError, ScriptedLauncher, call_and_record
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import tool_specs
from weightcode.harness.trial import NAMED_PROGRAM_EXPERIENCE_FILE, StudentOverride, TrialError, assemble
from weightcode.models import vit_synth as V

REPO = Path(__file__).resolve().parents[1]
POST_LEARN = REPO / "scripts" / "dev" / "post_learn.sh"
U, K0B, K0A = e11.UNLABELLED, e11.NAMED, e11.NAMED_NOCHECK
SENTENCE = e11.CAPABILITY_SENTENCES[PLANT_CONDITION]
WALL_S = 300.0
LN = "classifier.norm.weight"
PROBE = STRINGS + f"""import numpy as np

def update(student, experience):
    print("KEYS", ",".join(sorted(experience)))
    w = np.asarray(student["params"][{LN!r}], dtype=np.float64)
    delta = np.zeros_like(w)
    cut = int(round(0.25 * w.size))
    delta[:cut] = -2.0 * w[:cut]
    return {{{LN!r}: delta}}
"""
BREAK = STRINGS + """import numpy as np

def update(student, experience):
    b = np.asarray(student["params"]["classifier.fc.bias"], dtype=np.float64)
    delta = np.zeros_like(b)
    delta[1] = 50.0
    return {"classifier.fc.bias": delta}
"""  # every image to class 1: on the toy student score -15, score_old -11 below the floor -1
CRASH = STRINGS + """def update(student, experience):
    raise RuntimeError("no update")
"""
KERNEL_PROBE = """import json, os
_files = sorted(os.path.join(root, name) for root, _, names in os.walk("/work") for name in names)
print(json.dumps({"files": _files, "store": os.path.exists(STORE)}))
"""


# ---------------------------------------------------------------------------------------------------------------
# The searches (shared with the dry-run check on the real store)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix == ".npy":
        return {"": np.load(path, allow_pickle=False)}
    return {}


def row_keys(rows: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """For each row of ``rows`` (uint8, [N, ...]) one 8-byte key from its middle that is not one repeated byte, as
    ``{key: [(row, offset)]}``; a row with no such window is keyed at its middle."""
    flat = np.ascontiguousarray(rows).reshape(len(rows), -1)
    width = flat.shape[1]
    keys: dict[int, list[tuple[int, int]]] = {}
    for index in range(len(flat)):
        row = flat[index]
        chosen = max(0, width // 2 - 4)
        for offset in range(chosen, width - 8):
            window = row[offset:offset + 8]
            if window.min() != window.max():
                chosen = offset
                break
        key = int(np.frombuffer(row[chosen:chosen + 8].tobytes(), dtype=np.uint64)[0])
        keys.setdefault(key, []).append((index, chosen))
    return keys


def rows_in_bytes(data: bytes, rows: np.ndarray, keys: dict[int, list[tuple[int, int]]] | None = None) -> list[int]:
    """The rows of ``rows`` (uint8, [N, ...]) whose bytes occur anywhere in ``data``: every 8-byte window of ``data``
    at every offset is matched against ``row_keys`` and each candidate is compared in full."""
    flat = np.ascontiguousarray(rows).reshape(len(rows), -1)
    width = flat.shape[1]
    keys = row_keys(rows) if keys is None else keys
    if len(data) < width or not keys:
        return []
    wanted = np.fromiter(keys, dtype=np.uint64, count=len(keys))
    buffer = np.frombuffer(data, dtype=np.uint8)
    found: set[int] = set()
    for shift in range(8):
        count = (len(buffer) - shift) // 8
        if count <= 0:
            continue
        values = np.frombuffer(data, dtype=np.uint64, count=count, offset=shift)
        for position in np.nonzero(np.isin(values, wanted))[0]:
            at = shift + 8 * int(position)
            for index, offset in keys[int(values[position])]:
                start = at - offset
                if 0 <= start <= len(buffer) - width and data[start:start + width] == flat[index].tobytes():
                    found.add(index)
    return sorted(found)


def label_patterns(labels: np.ndarray) -> dict[str, bytes]:
    """The bytes of a label array as int64, int32 and uint8."""
    values = np.asarray(labels)
    return {dtype: np.ascontiguousarray(values.astype(dtype)).tobytes() for dtype in ("int64", "int32", "uint8")}


def labels_in(paths: list[Path], labels: np.ndarray) -> list[str]:
    """Where ``labels`` is found among ``paths``: an array equal to it, or one of its byte patterns inside a file's
    bytes or an array's bytes."""
    patterns = label_patterns(labels)
    found = []
    for path in paths:
        data = path.read_bytes()
        found += [f"{path.name}: {dtype} bytes" for dtype, pattern in patterns.items() if pattern in data]
        for key, array in _arrays(path).items():
            raw = np.ascontiguousarray(array).tobytes()
            found += [f"{path.name}:{key}: {dtype} bytes" for dtype, pattern in patterns.items() if pattern in raw]
            if array.shape == np.shape(labels) and np.array_equal(array, labels):
                found.append(f"{path.name}:{key}: equal")
    return found


def every_split(dataset: str) -> dict[str, dict[str, np.ndarray]]:
    """``{"<condition or clean>/<split>": arrays}`` of every private split of ``dataset`` and of each of its
    conditions (``weightcode.data.store.LOADABLE_SPLITS``; the sealed test set is never among them)."""
    found = {}
    for condition in (None, *data_store.list_conditions(dataset)):
        for split in data_store.LOADABLE_SPLITS:
            path = data_store.split_path(dataset, split, condition)
            if path.is_file():
                with np.load(path, allow_pickle=False) as archive:
                    found[f"{condition or 'clean'}/{split}"] = {key: archive[key] for key in archive.files}
    return found


def image_arrays_in(paths: list[Path], image_shape: tuple[int, ...]) -> list[str]:
    """Every array among ``paths`` whose trailing shape is the image shape."""
    found = []
    for path in paths:
        for key, array in _arrays(path).items():
            if array.ndim >= len(image_shape) and tuple(array.shape[array.ndim - len(image_shape):]) == image_shape:
                found.append(f"{path.name}:{key}")
    return found


def split_rows_in(paths: list[Path], splits: dict[str, dict[str, np.ndarray]]) -> dict[str, list[str]]:
    """``{split: [file:index]}`` for every image row of ``splits`` found in a file's bytes or an array's bytes."""
    blobs = []
    for path in paths:
        blobs.append((path.name, path.read_bytes()))
        blobs += [(f"{path.name}:{key}", np.ascontiguousarray(array).tobytes())
                  for key, array in _arrays(path).items()]
    found: dict[str, list[str]] = {}
    for name, arrays in splits.items():
        keys = row_keys(arrays["x"])
        hits = [f"{where}:{index}" for where, data in blobs for index in rows_in_bytes(data, arrays["x"], keys)]
        if hits:
            found[name] = hits
    return found


# ---------------------------------------------------------------------------------------------------------------
# Fixtures and helpers


@pytest.fixture
def ladder_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> V.SyntheticStore:
    """The toy E11 store under a neutral directory name."""
    return build_e11_store(tmp_path_factory.mktemp("hs") / "store", monkeypatch)


def _id(store: V.SyntheticStore, condition: str = PLANT_CONDITION, seed: int = 100) -> str:
    return store.ids[(condition, seed)]


def _assembled(store: V.SyntheticStore, directory: Path, **kwargs: Any):  # type: ignore[no-untyped-def]
    (directory / "private").mkdir(parents=True)
    return assemble(read_instance_record(_id(store)).instance_dir, "LEARN", directory, **kwargs)


def _sandbox_files(root: Path) -> list[Path]:
    paths = SessionPaths(root)
    return sorted(path for directory in (paths.trial, paths.patches, paths.scratch) for path in directory.rglob("*")
                  if path.is_file())


def _dry(instance_id: str, variant: str | None = None, **kwargs: Any) -> runner.SessionResult:
    result = runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=7200.0, dry_run=True,
                                retention_floor_frac=0.005, e11_variant=variant, **kwargs)
    assert result.status == STATUS_DRY_RUN, result.record
    return result


def _texts(root: Path) -> dict[str, str]:
    """The prompts and the tool descriptions of a session, as the editor gets them."""
    session = read_json(root / "session.json")
    spec = arm_spec(str(session["arm"]))
    tools = tool_specs(session["tools"], spec.program, s_old_floor=int(session["kept_rule"]["s_old_floor"]),
                       e11_variant=session.get("e11_variant"))
    return {"system": (root / "prompts" / "system.md").read_text(), "task": (root / "prompts" / "task.md").read_text(),
            "tools": json.dumps(tools, sort_keys=True)}


def _session(store: V.SyntheticStore, variant: str | None,
             script: Any) -> tuple[runner.SessionResult, ScriptedLauncher]:
    launcher = ScriptedLauncher(script)
    result = runner.run_session(_id(store), "LEARN", 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                                skip_audit=True, retention_floor_frac=0.005, e11_variant=variant)
    return result, launcher


def _assert_variant_recorded(store: V.SyntheticStore, result: runner.SessionResult, variant: str) -> None:
    """``e11_variant`` in session.json, session_result.json, run.json and the session's manifest line."""
    for name in ("session.json", "session_result.json", "run.json"):
        assert read_json(result.root / name)["e11_variant"] == variant, name
    lines = [json.loads(line) for line in (store.root / "runs" / "manifest.jsonl").read_text().splitlines()]
    assert [line["e11_variant"] for line in lines if line["session_id"] == result.root.name] == [variant]


def _budgets(checks: int = 8) -> Budgets:
    return Budgets(patch_checks=checks, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=7200.0)


# ---------------------------------------------------------------------------------------------------------------
# Units


@needs_prereg
def test_the_capability_sentence_is_the_prereg_one() -> None:
    prereg = " ".join(PREREG.read_text(encoding="utf-8").split())
    entry = prereg[prereg.index("Information ladder on one student: K0a, K0b, U"):]
    assert f'"{SENTENCE}"' in entry[:entry.index("K0b: K0a plus")]


def test_the_ladder_has_one_student() -> None:
    assert e11.LADDER_VARIANTS == (U, K0B, K0A) and e11.NAMED_VARIANTS == (K0B, K0A)
    assert e11.VARIANTS == ("standard", "rule", U, K0B, K0A) and e11.LADDER_CONDITION == PLANT_CONDITION
    base = {"dataset": e11.DATASET, "evidence_rule": "evidence-v1", "condition": PLANT_CONDITION, "seed": 100}
    assert e11.ladder_view(base, U) == {"variant": U, "condition": PLANT_CONDITION, "seed": 100}
    assert e11.ladder_view(base, K0B)["capability_sentence"] == SENTENCE and "hidden_check" not in e11.ladder_view(
        base, K0B)
    assert e11.ladder_view(base, K0A)["hidden_check"] == e11.HIDDEN_CHECK
    for private, text in (({**base, "seed": 101}, "seed-100 student only"),
                          ({**base, "condition": "contrast_s5"}, "seed-100 student only"),
                          ({**base, "dataset": "cifar10_r20"}, "E11 evidence-v1 instance"),
                          ({**base, "evidence_rule": "fewshot-v1"}, "E11 evidence-v1 instance")):
        with pytest.raises(e11.E11HarnessError, match=text):
            e11.ladder_view(private, K0B)
    with pytest.raises(e11.E11HarnessError, match="not a ladder view"):
        e11.ladder_view(base, "rule")


def test_each_ladder_task_template_is_derived_by_edits_found_exactly_once() -> None:
    learn = (PROMPT_DIR / "task_LEARN.md").read_text(encoding="utf-8")
    for variant, edits in LADDER_TASK_EDITS.items():
        text = learn
        for old, new in edits:
            assert text.count(old) == 1, (variant, old[:40])
            text = text.replace(old, new)
        derived = ladder_task_template(variant, learn)
        assert derived == text and max(len(line) for line in derived.splitlines() if "$" not in line) <= 116
        doubled = learn.replace(edits[0][0], edits[0][0] + "\n" + edits[0][0])
        with pytest.raises(ValueError, match="exactly once"):
            ladder_task_template(variant, doubled)
    assert "y_new" not in ladder_task_template(U, learn) and "experience.npz" not in ladder_task_template(K0B, learn)
    nocheck = ladder_task_template(K0A, learn)
    assert "check" not in nocheck.lower() and "$readout" not in nocheck and "$budget_queries" not in nocheck
    system = (PROMPT_DIR / "system_LEARN.md").read_text(encoding="utf-8")
    kept = nocheck_kept_section(system, -20)
    assert "check" not in kept.lower() and "at least -20 (at most 20 more clean held-out images" in " ".join(
        kept.split()) and max(len(line) for line in kept.splitlines()) <= 116
    # every paragraph but What is kept and the two closing passages is system_LEARN.md's, byte for byte
    assert kept.split("## Rules")[1].split("## Environment")[0] == system.split("## Rules")[1].split(
        "## Environment")[0]
    with pytest.raises(ValueError, match="word for word"):
        nocheck_kept_section(system.replace("Unchecked programs are never kept.", "Unchecked programs are kept."), -20)
    with pytest.raises(ValueError, match="exactly once"):
        nocheck_kept_section(system.replace("A program counts only once", "A file counts only once"), -20)


def test_the_ladder_task_views_refuse_what_they_cannot_derive(ladder_store: V.SyntheticStore,
                                                              tmp_path: Path) -> None:
    task = _assembled(ladder_store, tmp_path / "learn").task
    assert learn_task_unlabelled(task)["experience"]["new"]["keys"] == ["x_new", "pred_new"]
    for checks in (True, False):
        once = learn_task_named(task, SENTENCE, checks)
        with pytest.raises(ExperienceError, match="already names the capability"):
            learn_task_named(once, SENTENCE, checks)
    for key, edit, view in (
            ("statement",
             lambda s: s.replace("holds labelled examples from the new", "holds some examples from the new"),
             "unlabelled"),
            ("statement", lambda s: s + " The experience file holds labelled examples from the new condition.",
             "unlabelled"),
            ("statement", lambda s: s.replace("with the classifier's predictions on both.", "and more."), "named"),
            ("statement", lambda s: s.replace("its experience, without", "its data, without"), "named")):
        changed = json.loads(json.dumps(task))
        changed[key] = edit(changed[key])
        with pytest.raises(ExperienceError, match="does not say"):
            if view == "unlabelled":
                learn_task_unlabelled(changed)
            else:
                learn_task_named(changed, SENTENCE, True)
    relabelled = json.loads(json.dumps(task))
    relabelled["training_free_rule"]["program"][0] = "in this task the evidence is experience.npz"
    for derive in (learn_task_unlabelled, lambda t: learn_task_named(t, SENTENCE, False)):
        with pytest.raises(ExperienceError, match="training-free rule does not say"):
            derive(relabelled)
    moved = json.loads(json.dumps(task))
    moved["program"]["transfer"] = moved["program"]["transfer"].replace("own experience", "own data")
    with pytest.raises(ExperienceError, match="program transfer does not say"):
        learn_task_named(moved, SENTENCE, True)
    reread = json.loads(json.dumps(task))
    reread["patch_check"]["readout"] = reread["patch_check"]["readout"].replace("and its experience", "and its rows")
    with pytest.raises(ExperienceError, match="readout does not say"):
        learn_task_named(reread, SENTENCE, True)
    assert "readout" not in learn_task_named(reread, SENTENCE, False).get("patch_check", {})  # K0a has none


# ---------------------------------------------------------------------------------------------------------------
# U: the unlabelled view


def test_u_shows_e11s_experience_without_y_new(ladder_store: V.SyntheticStore, tmp_path: Path) -> None:
    standard = _assembled(ladder_store, tmp_path / "standard")
    view = _assembled(ladder_store, tmp_path / "u", e11_variant=U)
    assert sorted(view.files) == sorted(standard.files) and view.limits == standard.limits
    assert {k: v for k, v in view.files.items() if k not in ("experience.npz", "task.json")} == {
        k: v for k, v in standard.files.items() if k not in ("experience.npz", "task.json")}
    assert {k: v for k, v in view.clean_check.items() if k != "path"} == {
        k: v for k, v in standard.clean_check.items() if k != "path"}
    with np.load(view.trial_dir / "experience.npz") as archive:
        shown = {key: archive[key] for key in archive.files}
    with np.load(standard.trial_dir / "experience.npz") as archive:
        full = {key: archive[key] for key in archive.files}
    assert sorted(shown) == sorted(UNLABELLED_EXPERIENCE_KEYS) == sorted(set(EXPERIENCE_KEYS) - {"y_new"})
    assert all(shown[key].dtype == full[key].dtype and shown[key].tobytes() == full[key].tobytes() for key in shown)
    record = view.experience
    assert record["variant"] == U and record["withheld_keys"] == ["y_new"] and record["labels_new_shown"] is False
    assert (record["n_new"], record["n_new_fail"], record["n_old"]) == (64, 16, standard.experience["n_old"])
    assert record["learn_content_sha256"] == standard.experience["content_sha256"]
    assert view.e11_view == {"variant": U, "condition": PLANT_CONDITION, "seed": 100}
    # task.json: the statement, the new part's keys, description and n_from_errors and the program's label item
    # differ, nothing else
    assert _differences(view.task, standard.task) == [".experience.new.description", ".experience.new.keys",
                                                      ".experience.new.n_from_errors", ".statement",
                                                      ".training_free_rule.program"]
    assert "n_from_errors" not in view.task["experience"]["new"] and view.task["experience"]["new"]["count"] == 64
    assert view.task["statement"] == standard.task["statement"].replace(
        "holds labelled examples from the new condition", "holds unlabelled examples from the new condition")
    assert view.task["training_free_rule"]["program"][0].endswith("its labels are y_old")
    assert "y_new" not in (view.trial_dir / "task.json").read_text()
    prompt = render_task("LEARN", view.task, view.limits, _budgets(), view.input_files, e11_variant=U)
    plain = render_task("LEARN", standard.task, standard.limits, _budgets(), standard.input_files)
    assert "y_new" not in prompt and "only the clean part is labelled" in prompt and "$" not in prompt
    assert [line for line in plain.splitlines() if line not in prompt.splitlines()][0].startswith("Make a trained")


def test_u_does_not_say_which_condition_rows_are_errors(ladder_store: V.SyntheticStore, tmp_path: Path) -> None:
    standard = _assembled(ladder_store, tmp_path / "standard")
    view = _assembled(ladder_store, tmp_path / "u", e11_variant=U)
    selection = learn_new_selection(16, 48)
    kept = {"name": "joint-retain-v1", "s_old_floor": -20, "check_set": "clean"}
    texts = {"task.json": (view.trial_dir / "task.json").read_text(),
             "task.md": render_task("LEARN", view.task, view.limits, _budgets(), view.input_files, e11_variant=U),
             "system.md": render_system("LEARN", kept_rule=kept, e11_variant=U),
             "tools": json.dumps(tool_specs(arm_spec("LEARN").tools, True, s_old_floor=-20, e11_variant=U))}
    plain = {"task.json": (standard.trial_dir / "task.json").read_text(),
             "task.md": render_task("LEARN", standard.task, standard.limits, _budgets(), standard.input_files)}
    # positive control: E11's texts say it, in the description (task.json, task.md) and in n_from_errors (task.json)
    assert standard.task["experience"]["new"]["description"] == LEARN_NEW_DESCRIPTION + selection
    assert all(word in plain["task.json"] for word in UNLABELLED_ERROR_WORDS)
    assert [word for word in UNLABELLED_ERROR_WORDS if word in plain["task.md"]] == list(UNLABELLED_ERROR_WORDS[1:])
    assert "the first 16 drawn" in plain["task.md"]
    for name, text in texts.items():
        assert [word for word in UNLABELLED_ERROR_WORDS if word in text] == [], name
        assert "first 16" not in text and "errors there" not in text, name
    assert view.task["experience"]["new"]["description"] == LEARN_NEW_DESCRIPTION
    bullet = " ".join(texts["task.md"].split())
    assert "`x_new`, `pred_new`: 64 examples from the new input condition. `x_old`" in bullet
    # the row selection is E11's, and only the host record states it
    assert (view.experience["n_new"], view.experience["n_new_fail"]) == (64, 16)
    # the removed text must be found exactly once, n_from_errors must be given, and nothing may still say it
    task = standard.task
    for edit, text in (
            (lambda t: t["experience"]["new"].update(description=t["experience"]["new"]["description"] + selection),
             "new-part description does not say"),
            (lambda t: t["experience"]["new"].update(description=t["experience"]["new"]["description"].replace(
                "the first 16", "the first 15")), "new-part description does not say"),
            (lambda t: t["experience"]["new"].pop("n_from_errors"), "does not give its count and n_from_errors"),
            (lambda t: t.update(statement=t["statement"] + " The first rows are the classifier's errors."),
             "still says which condition rows are errors")):
        changed = json.loads(json.dumps(task))
        edit(changed)
        with pytest.raises(ExperienceError, match=text):
            learn_task_unlabelled(changed)
    with pytest.raises(ValueError, match="names"):
        render_task("LEARN", {**view.task, "experience": {**view.task["experience"], "new": {
            **view.task["experience"]["new"], "description": LEARN_NEW_DESCRIPTION + selection}}}, view.limits,
            _budgets(), view.input_files, e11_variant=U)


def test_no_y_new_array_or_label_bytes_reach_u(ladder_store: V.SyntheticStore, tmp_path: Path) -> None:
    instance = load_instance(read_instance_record(_id(ladder_store)).instance_dir)
    y_new = np.concatenate([instance.evidence["y_fail"], instance.evidence["y_rand"]]).astype(np.int64)
    assert len(y_new) == 64 and len(set(y_new.tolist())) > 1
    standard = _assembled(ladder_store, tmp_path / "standard")
    view = _assembled(ladder_store, tmp_path / "u", e11_variant=U)
    # positive control: the detector finds y_new in the standard view's experience, as an array and as bytes
    found = labels_in(sorted(standard.trial_dir.iterdir()), y_new)
    assert "experience.npz:y_new: equal" in found and "experience.npz: int64 bytes" in found
    assert labels_in(sorted(view.trial_dir.iterdir()), y_new) == []
    assert labels_in(sorted((tmp_path / "u" / "private").iterdir()), y_new) == []  # nor in the program inputs


# ---------------------------------------------------------------------------------------------------------------
# K0b and K0a: the no-data views


@pytest.mark.parametrize("variant", [K0B, K0A])
def test_the_no_data_view_holds_the_network_and_task_json_alone(ladder_store: V.SyntheticStore, tmp_path: Path,
                                                                variant: str) -> None:
    standard = _assembled(ladder_store, tmp_path / "standard")
    view = _assembled(ladder_store, tmp_path / variant, e11_variant=variant)
    assert sorted(view.files) == ["forward.py", "model.npz", "model_spec.json", "task.json"]
    assert {k: v for k, v in view.files.items() if k != "task.json"} == {
        k: v for k, v in standard.files.items() if k not in ("experience.npz", "task.json")}
    assert view.limits == standard.limits and view.shown_model_sha256 == standard.shown_model_sha256
    assert view.evidence_source_path is None and view.experience["withheld_keys"] == sorted(EXPERIENCE_KEYS)
    assert (view.experience["n_new"], view.experience["n_old"]) == (0, 0)
    assert {k: v for k, v in view.clean_check.items() if k != "path"} == {
        k: v for k, v in standard.clean_check.items() if k != "path"}
    empty = Path(view.e11_view["program_experience_path"])
    assert empty.parent == tmp_path / variant / "private" and empty.name == NAMED_PROGRAM_EXPERIENCE_FILE
    with np.load(empty) as archive:
        assert archive.files == []
    assert view.e11_view["capability_sentence"] == SENTENCE and view.e11_view["program_experience_keys"] == []
    assert (view.e11_view.get("hidden_check") == e11.HIDDEN_CHECK) == (variant == K0A)
    task = view.task
    assert "experience" not in task and task["program"]["experience"] == NAMED_PROGRAM_EXPERIENCE
    assert task["statement"].count(SENTENCE) == 1 and json.dumps(task).count(SENTENCE) == 1
    assert task["statement"].startswith("Make a trained image classifier handle a new input condition")
    text = (view.trial_dir / "task.json").read_text()
    assert [word for word in NAMED_EXPERIENCE_WORDS if word in text] == []
    differences = _differences(task, standard.task)
    if variant == K0B:
        assert differences == [".experience", ".patch_check.readout", ".program.experience", ".program.transfer",
                               ".statement", ".training_free_rule.program"]
        assert task["patch_check"]["readout"].startswith("each query runs the program once on this classifier, ")
    else:
        assert differences == [".experience", ".final_selection", ".patch_check", ".program.experience",
                               ".program.transfer", ".statement", ".training_free_rule.program"]
        assert task["final_selection"] == NAMED_NOCHECK_FINAL_SELECTION and "check" not in text.lower()
        assert task["training_free_rule"]["program"][-1] == "every declared program is stored with the session record"
    assert task["training_free_rule"]["program"][0] == "in this task no evidence is given: no example and no label"
    assert scan_label_arrays(view.trial_dir) == {}


@pytest.mark.parametrize("variant", [K0B, K0A])
def test_no_image_array_or_label_of_any_split_reaches_the_no_data_view(ladder_store: V.SyntheticStore,
                                                                       tmp_path: Path, variant: str) -> None:
    splits = every_split(e11.DATASET)
    assert {"clean/train", "clean/V_check", f"{PLANT_CONDITION}/V_evidence", f"{PLANT_CONDITION}/V_select"} <= set(
        splits)
    image_shape = tuple(splits["clean/train"]["x"].shape[1:])
    standard = _assembled(ladder_store, tmp_path / "standard")
    view = _assembled(ladder_store, tmp_path / variant, e11_variant=variant)
    standard_files = sorted(standard.trial_dir.iterdir())
    # positive control: the search finds the standard view's condition rows and clean replay rows, and its labels
    control = split_rows_in(standard_files, splits)
    assert f"{PLANT_CONDITION}/V_evidence" in control and "clean/train" in control
    assert image_arrays_in(standard_files, image_shape) == ["experience.npz:x_new", "experience.npz:x_old"]
    # the no-data view and the program's inputs: no image array, no row of any split, no label pattern of any split
    files = [*sorted(view.trial_dir.iterdir()), Path(view.e11_view["program_experience_path"])]
    assert image_arrays_in(files, image_shape) == [] and split_rows_in(files, splits) == {}
    for name, arrays in splits.items():
        assert labels_in(files, arrays["y"][:64]) == [], name


def test_the_no_data_prompts_name_the_capability_once_and_k0a_says_no_check(ladder_store: V.SyntheticStore,
                                                                             tmp_path: Path) -> None:
    for variant in (K0B, K0A):
        view = _assembled(ladder_store, tmp_path / variant, e11_variant=variant)
        prompt = render_task("LEARN", view.task, view.limits, _budgets(1 if variant == K0A else 8), view.input_files,
                             e11_variant=variant)
        assert prompt.count(SENTENCE) == 1 and "$" not in prompt
        assert [word for word in NAMED_EXPERIENCE_WORDS if word in prompt] == []
        kept = {"name": "joint-retain-v1", "s_old_floor": -20, "check_set": "clean"}
        system = render_system("LEARN", kept_rule=kept, e11_variant=variant)
        assert SENTENCE not in system
        if variant == K0A:
            assert "check" not in prompt.lower() and "check" not in system.lower()
            assert "## Final selection" in prompt and "readout" not in prompt.lower()
            with pytest.raises(ValueError, match="needs that kept rule"):
                render_system("LEARN", e11_variant=K0A)
        else:
            assert "patch_check" in prompt and system == render_system("LEARN", kept_rule=kept)
        with pytest.raises(KeyError):  # the standard template needs the experience, which this task.json lacks
            render_task("LEARN", view.task, view.limits, _budgets(), view.input_files)
    for arm in ("LEARN_BLIND", "M2"):
        with pytest.raises(ValueError, match="is a LEARN view"):
            render_task(arm, view.task, view.limits, _budgets(), view.input_files, e11_variant=K0B)
    tools = tool_specs(("python", "status", "declare_final"), True, s_old_floor=-20, e11_variant=K0A)
    assert "check" not in json.dumps(tools).lower() and [tool["name"] for tool in tools] == ["python", "status",
                                                                                            "declare_final"]
    named = tool_specs(arm_spec("LEARN").tools, True, s_old_floor=-20, e11_variant=K0B)
    standard = tool_specs(arm_spec("LEARN").tools, True, s_old_floor=-20)
    assert [a == b for a, b in zip(named, standard)] == [True, False, True, True]
    assert named[1]["description"] == standard[1]["description"].replace("on this classifier and its experience;",
                                                                         "on this classifier;")
    with pytest.raises(ValueError, match="no patch_check"):
        tool_specs(arm_spec("LEARN").tools, True, s_old_floor=-20, e11_variant=K0A)
    with pytest.raises(ValueError, match="no patch_check"):
        tool_specs(("python", "status", "declare_final"), True, e11_variant=K0A)


# ---------------------------------------------------------------------------------------------------------------
# Dry runs


@needs_sandbox
def test_ladder_dry_runs_pass_the_scans_and_keep_the_other_views(ladder_store: V.SyntheticStore) -> None:
    instance_id = _id(ladder_store)
    standard = _dry(instance_id)
    plain = read_json(standard.root / "session.json")
    plain_texts = _texts(standard.root)
    for variant in (U, K0B, K0A):
        result = _dry(instance_id, variant)
        session = read_json(result.root / "session.json")
        texts = _texts(result.root)
        assert session["e11_variant"] == variant and session["e11_view"]["variant"] == variant
        assert session["trial"]["e11_view"] == session["e11_view"]
        assert read_json(result.root / "session_result.json")["e11_variant"] == variant
        scan = session["leak_scan"]
        assert scan["passed"] is True and scan["hits"] == {} and scan["hint_words"] == {}
        staged = _sandbox_files(result.root)
        assert [path.name for path in staged] == sorted(session["trial_files_sha256"])
        if variant == U:
            assert session["forbidden_strings"] == plain["forbidden_strings"] and "sentence_exempt" not in scan
            assert texts["system"] == plain_texts["system"] and texts["tools"] == plain_texts["tools"]
            assert session["tools"] == plain["tools"] and session["budget"] == plain["budget"]
            continue
        forbidden = session["forbidden_strings"]
        assert forbidden["sentence_exempt"] == [SENTENCE] and forbidden["text"] == plain["forbidden_strings"]["text"]
        assert {key: value for key, value in forbidden.items() if not key.startswith("sentence_exempt")} == plain[
            "forbidden_strings"]
        assert scan["sentence_placed"] is True and scan["sentence_counts"]["trial/task.json"] == {SENTENCE: 1}
        assert scan["sentence_counts"]["prompts/task.md"] == {SENTENCE: 1}
        assert all(count == {SENTENCE: 0} for name, count in scan["sentence_counts"].items()
                   if name not in ("trial/task.json", "prompts/task.md"))
        # only the sentence is exempt: its words are leak and hint words everywhere else
        assert sorted(set(scan_for_tokens(SENTENCE, forbidden["text"]))) == ["corrupt", "gaussian", "noise"]
        assert [word for word in HINT_WORDS if word in SENTENCE.lower()] == ["noise", "corrupt"]
        assert scan_for_tokens(texts["task"].replace(SENTENCE, ""), forbidden["text"]) == []
        assert [path.name for path in staged] == ["forward.py", "model.npz", "model_spec.json", "task.json"]
        assert session["experience"]["source"] == "e11_named_no_data_v1"
        if variant == K0B:
            assert texts["system"] == plain_texts["system"] and session["tools"] == plain["tools"]
            assert session["budget"] == plain["budget"]
        else:
            assert session["tools"] == ["python", "status", "declare_final"]
            assert session["allowed_tools"] == [f"mcp__weightcode__{tool}" for tool in session["tools"]]
            assert session["budget"] == {**plain["budget"], "patch_checks": 1}
            assert session["e11_view"]["hidden_check_budget"] == 1
            assert all("check" not in text.lower() for text in texts.values())
    # the standard, blind and two-check views and the rule view's inputs are as at main f155d9b (BEFORE_K1)
    assert _view_digest(standard) == BEFORE_K1["e11_learn"]
    assert _view_digest(runner.run_session(instance_id, "LEARN_BLIND", 0, effort="max", wall_timeout_s=7200.0,
                                           dry_run=True, retention_floor_frac=0.005)) == BEFORE_K1["e11_blind"]


@needs_sandbox
def test_a_leak_word_outside_the_sentence_still_fails_the_scan(ladder_store: V.SyntheticStore) -> None:
    result = _dry(_id(ladder_store), K0B)
    session = read_json(result.root / "session.json")
    paths = SessionPaths(result.root)
    from weightcode.harness.leaks import ForbiddenStrings

    record = session["forbidden_strings"]
    forbidden = ForbiddenStrings(selftest=tuple(record["selftest"]), text=tuple(record["text"]),
                                 text_exempt=tuple(record["text_exempt"]), sentence_exempt=(SENTENCE,))
    texts = _texts(result.root)
    prompts = {"prompts/system.md": texts["system"], "prompts/task.md": texts["task"]}
    tools = json.loads(texts["tools"])
    assert runner.leak_scan(paths, prompts, tools, forbidden)["passed"] is True
    leaked = {**prompts, "prompts/task.md": texts["task"] + "\nThe images carry Gaussian noise.\n"}
    scan = runner.leak_scan(paths, leaked, tools, forbidden)
    assert scan["passed"] is False and scan["hits"]["prompts/task.md"] == ["gaussian", "noise"]
    assert scan["hint_words"]["prompts/task.md"] == ["noise"]
    twice = {**prompts, "prompts/system.md": texts["system"] + "\n" + SENTENCE + "\n"}
    scan = runner.leak_scan(paths, twice, tools, forbidden)
    assert scan["passed"] is False and scan["sentence_placed"] is False and scan["hits"] == {}
    without = runner.leak_scan(paths, prompts, tools, ForbiddenStrings(
        selftest=forbidden.selftest, text=forbidden.text, text_exempt=forbidden.text_exempt))
    assert without["passed"] is False and "gaussian" in without["hits"]["prompts/task.md"]


# ---------------------------------------------------------------------------------------------------------------
# Sessions


@needs_sandbox
def test_a_u_session_never_gives_y_new_and_runs_as_e11(ladder_store: V.SyntheticStore) -> None:
    probe = f"STORE = {str(ladder_store.root)!r}\n" + KERNEL_PROBE

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": probe})
        call_and_record(client, replies, "python", {"code": _program_cell({"probe.py": PROBE})})
        call_and_record(client, replies, "patch_check", {"program_file": "probe.py"})
        call_and_record(client, replies, "declare_final", {"program_file": "probe.py", "rationale": "it scored"})

    result, launcher = _session(ladder_store, U, script)
    assert result.status == "complete", result.record.get("infra_reasons")
    kernel, _, check, declared = launcher.replies
    session = read_json(result.root / "session.json")
    assert json.loads(kernel.payload["stdout"])["files"] == [f"/work/trial/{name}"
                                                             for name in sorted(session["trial_files_sha256"])]
    assert not check.is_error and check.payload["program_stdout"].startswith(
        "KEYS " + ",".join(sorted(UNLABELLED_EXPERIENCE_KEYS)))
    assert check.payload["score_new"] > 0 and declared.payload["valid"]
    run = read_json(result.root / "attempts" / "a0000" / "program_run.json")
    assert run["inputs_sha256"]["experience.npz"] == session["trial_files_sha256"]["experience.npz"]
    instance = load_instance(read_instance_record(_id(ladder_store)).instance_dir)
    y_new = np.concatenate([instance.evidence["y_fail"], instance.evidence["y_rand"]]).astype(np.int64)
    assert labels_in(_sandbox_files(result.root), y_new) == []
    assert read_json(SessionPaths(result.root).selection_json)["final"]["patch_id"] == "probe.py"
    _assert_variant_recorded(ladder_store, result, U)


@needs_sandbox
def test_a_k0b_session_gives_the_program_an_empty_experience(ladder_store: V.SyntheticStore) -> None:
    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"probe.py": PROBE})})
        call_and_record(client, replies, "patch_check", {"program_file": "probe.py"})
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "probe.py", "rationale": "it scored"})

    result, launcher = _session(ladder_store, K0B, script)
    assert result.status == "complete", result.record.get("infra_reasons")
    _, check, status, declared = launcher.replies
    assert not check.is_error and check.payload["program_stdout"].startswith("KEYS \n")
    assert check.payload["score_new"] > 0 and status.payload["checked"][0]["program_file"] == "probe.py"
    session = read_json(result.root / "session.json")
    run = read_json(result.root / "attempts" / "a0000" / "program_run.json")
    assert run["inputs_sha256"]["experience.npz"] == session["e11_view"]["program_experience_file_sha256"]
    assert run["inputs_sha256"]["params.npz"] == session["trial_files_sha256"]["model.npz"]
    assert "experience.npz" not in session["trial_files_sha256"] and declared.payload["valid"]
    assert read_json(SessionPaths(result.root).selection_json)["final"]["patch_id"] == "probe.py"
    _assert_variant_recorded(ladder_store, result, K0B)


@needs_sandbox
def test_a_k0a_session_has_no_check_and_keeps_an_eligible_declaration_after_one(ladder_store: V.SyntheticStore
                                                                               ) -> None:
    refused: list[str] = []

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"probe.py": PROBE, "crash.py": CRASH})})
        try:
            client.call("patch_check", {"program_file": "probe.py"})
        except McpError as exc:
            refused.append(str(exc))
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "probe.py", "rationale": "the edit"})
        call_and_record(client, replies, "declare_final", {"program_file": "crash.py", "rationale": "a crash"})
        call_and_record(client, replies, "status", {})

    result, launcher = _session(ladder_store, K0A, script)
    assert result.status == "complete", result.record.get("infra_reasons")
    assert launcher.tool_names == ["python", "status", "declare_final"] and refused == ["unknown tool: patch_check"]
    _, before, declared, crashed, after = launcher.replies
    replies = json.dumps([reply.payload for reply in launcher.replies])
    assert "check" not in replies.lower() and "score_new" not in replies and "query" not in replies
    assert before.payload["declared_final"] is None and before.payload["last_valid_program_file"] is None
    assert declared.payload["valid"] and declared.payload["tensors"] == [LN]
    assert declared.payload["program_stdout"].startswith("KEYS \n")
    assert crashed.is_error and not crashed.payload["valid"] and "no update" in json.dumps(crashed.payload)
    assert after.payload["declared_final"] == {"program_file": "crash.py", "valid": False}
    assert after.payload["last_valid_program_file"] == "probe.py"
    # finalisation: one hidden check of the last valid declaration, then joint-retain-v1
    paths = SessionPaths(result.root)
    checks = read_jsonl(paths.checks_log)
    assert len(checks) == 1 and checks[0]["hidden"] is True and checks[0]["program_file"] == "probe.py"
    declarations = read_jsonl(paths.declared_log)
    assert checks[0]["attempt"] == declarations[0]["attempt"] and checks[0]["child_sha256"] == declarations[0][
        "child_sha256"]
    selection = read_json(paths.selection_json)
    hidden = selection["hidden_check"]
    assert hidden["ran"] and hidden["declaration"]["program_file"] == "probe.py" and hidden["score_new"] > 0
    floor = int(read_json(result.root / "session.json")["kept_rule"]["s_old_floor"])
    assert hidden["score_old"] >= floor and hidden["score"] > 0
    final = selection["final"]
    assert final["kind"] == "checked" and final["patch_id"] == "probe.py" and final["program"]["file"].endswith(
        "probe.py")
    assert "hidden check (hidden-check-v1)" in final["reason"] and "so it is kept" in final["reason"]
    record = read_json(result.root / "session_result.json")
    assert record["hidden_check"] == hidden and record["queries"]["used"] == 1 and record["queries"]["budget"] == 1
    _assert_variant_recorded(ladder_store, result, K0A)
    # the declared program's V_select readout: session_info reads the hidden check like any check
    info = session_info(result.root)
    assert [row["patch_id"] for row in info["trajectory"]] == ["probe.py"]
    assert info["trajectory"][0]["vselect"]["delta_acc"] > 0 and "vselect_clean" in info["trajectory"][0]
    assert info["final"]["vselect"] == info["trajectory"][0]["vselect"]


@needs_sandbox
@pytest.mark.parametrize("case", ["ineligible", "null", "none", "invalid"])
def test_a_k0a_session_keeps_the_base_otherwise_and_says_why(ladder_store: V.SyntheticStore, case: str) -> None:
    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"break.py": BREAK, "crash.py": CRASH})})
        if case == "ineligible":
            call_and_record(client, replies, "declare_final", {"program_file": "break.py", "rationale": "x"})
        elif case == "null":
            call_and_record(client, replies, "declare_final", {"program_file": "break.py", "rationale": "x"})
            call_and_record(client, replies, "declare_final", {"program_file": None, "rationale": "the original"})
        elif case == "invalid":
            call_and_record(client, replies, "declare_final", {"program_file": "crash.py", "rationale": "x"})

    result, _ = _session(ladder_store, K0A, script)
    assert result.status == "complete", result.record.get("infra_reasons")
    paths = SessionPaths(result.root)
    selection = read_json(paths.selection_json)
    hidden, final = selection["hidden_check"], selection["final"]
    assert final["kind"] == "base" and final["is_base_noop"] and selection["rule_final"]["kind"] == "base"
    checks = read_jsonl(paths.checks_log)
    if case == "ineligible":
        assert hidden["ran"] and len(checks) == 1 and checks[0]["hidden"] is True
        floor = int(read_json(result.root / "session.json")["kept_rule"]["s_old_floor"])
        assert hidden["score_old"] < floor and hidden["declaration"]["program_file"] == "break.py"
        assert "it is not eligible, so the base model is kept" in final["reason"]
        info = session_info(result.root)  # the declared program's V_select readout, beside the base final
        assert [row["patch_id"] for row in info["trajectory"]] == ["break.py"] and info["final"]["vselect"] is None
        assert "delta_acc" in info["trajectory"][0]["vselect"] and "delta_acc" in info["trajectory"][0][
            "vselect_clean"]
    else:
        assert not hidden["ran"] and checks == [] and result.record["queries"]["used"] == 0
        reason = {"null": "names the original model", "none": "no valid declaration was made",
                  "invalid": "no valid declaration was made"}[case]
        assert reason in hidden["reason"] and reason in final["reason"]


def test_the_hidden_check_refuses_a_session_it_cannot_read(tmp_path: Path) -> None:
    class Ledger:
        def __init__(self, entries: list[Any], budget: int) -> None:
            self.entries, self.budget = entries, budget

    paths = SessionPaths(tmp_path)
    kept = {"kept_rule": {"name": "joint-retain-v1"}}
    with pytest.raises(FinalizeError, match="checks logged before"):
        run_hidden_check(paths, kept, Ledger([{"patch_id": "x"}], 1), None)  # type: ignore[arg-type]
    with pytest.raises(FinalizeError, match="one query"):
        run_hidden_check(paths, kept, Ledger([], 8), None)  # type: ignore[arg-type]
    with pytest.raises(FinalizeError, match="one query"):
        run_hidden_check(paths, {}, Ledger([], 1), None)  # type: ignore[arg-type]
    assert run_hidden_check(paths, kept, Ledger([], 1), None)["ran"] is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------------
# Refusals and the command line


def test_the_ladder_is_refused_where_it_does_not_apply(ladder_store: V.SyntheticStore, tmp_path: Path) -> None:
    instance_id = _id(ladder_store)
    runs = ladder_store.root / "runs"
    before = sorted(path.relative_to(runs) for path in runs.rglob("*"))
    launcher = ScriptedLauncher(lambda client, replies: None)
    for variant in (U, K0B, K0A):
        cases = [
            ("LEARN_BLIND", {}, ValueError, "LEARN arm only"),
            ("M2", {"retention_floor_frac": None}, ValueError, "LEARN arm only"),
            ("LEARN", {"loop_context": Path("context.json")}, ValueError, "refused in a loop round"),
            ("LEARN", {"library_sessions": ["s0123456789abcdef"]}, ValueError, "no library"),
            ("LEARN", {"task_prompt_override": "a prompt"}, ValueError, "prompt override"),
            ("LEARN", {"max_checks": 2}, ValueError, "no max_checks"),
        ]
        for arm, kwargs, error, text in cases:
            with pytest.raises(error, match=text):
                runner.run_session(instance_id, arm, 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                                   e11_variant=variant, **{"retention_floor_frac": 0.005, **kwargs})
        for other in (_id(ladder_store, "contrast_s5"), _id(ladder_store, PLANT_CONDITION, 101)):
            with pytest.raises(InstanceRefused, match="seed-100 student only"):
                runner.run_session(other, "LEARN", 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                                   retention_floor_frac=0.005, e11_variant=variant)
            with pytest.raises(TrialError, match="seed-100 student only"):
                directory = tmp_path / f"{variant}_{other}"
                (directory / "private").mkdir(parents=True)
                assemble(read_instance_record(other).instance_dir, "LEARN", directory, e11_variant=variant)
    with pytest.raises(ValueError, match="give its retention floor"):
        runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                           e11_variant=K0A)
    assert sorted(path.relative_to(runs) for path in runs.rglob("*")) == before  # refused before any write
    override = StudentOverride(params={}, experience={}, record={}, keys=(), task=lambda task: task)
    for variant in (U, K0B, K0A):
        for arm, kwargs in (("LEARN_BLIND", {}), ("LEARN", {"override": override}), ("LEARN", {"max_checks": 2})):
            with pytest.raises(TrialError, match="assembled in the LEARN arm on the instance's own student"):
                assemble(tmp_path / "none", arm, tmp_path / "none", e11_variant=variant, **kwargs)


def test_the_session_command_takes_the_ladder_variants() -> None:
    spec = importlib.util.spec_from_file_location("run_editor_session_ladder",
                                                  REPO / "scripts" / "run_editor_session.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = ["--instance", "i00000000", "--arm", "LEARN", "--repeat", "0", "--effort", "max", "--wall-timeout-min",
            "120", "--retention-floor-frac", "0.005"]
    for variant in (U, K0B, K0A):
        assert module.parse_args([*base, "--e11-variant", variant]).e11_variant == variant


@pytest.mark.skipif(not POST_LEARN.is_file(),
                    reason="scripts/dev/post_learn.sh is a local cluster script, not part of the public repository")
def test_post_learn_refuses_e3_for_every_variant_session() -> None:
    text = POST_LEARN.read_text()
    guard = text[text.index("VARIANT=$("):text.index("run_program_transfer.py --program")]
    assert "get('e11_variant') or ''" in guard and 'if [ -n "$VARIANT" ]; then' in guard and "exit 3" in guard
    assert text.index("exit 3") < text.index("run_program_transfer.py --program")
