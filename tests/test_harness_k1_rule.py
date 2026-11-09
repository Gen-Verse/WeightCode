"""PREREG K1 in the harness: the E11 rule view (``e11_variant="rule"``), the knowledge route on the E11 ViT.

The view gives the editor the network, the instance's clean labelled replay rows and the condition's rule as a formula
(the PREREG K1 sentence), and no row or image of the condition in any file or tool it can reach. Sessions run on the
toy E11 store (``e11_toy``) through the real runner, sandbox and MCP server, with the toy gaussian_noise_s5 condition
record stating the official code's fields (``_state_rule``; the toy store writes a stand-in record, which the view
refuses). The tests check that no array the sandbox can read equals or contains a condition row (row hashes and raw
bytes, with the standard view as the positive control), that the rule sentence appears exactly once, that the leak
words are E11's with no exemption, that every other E11 view (and E10's) is byte for byte what it was at main
f155d9b, that the view is refused where it does not apply, and that only a rule-view session carries ``e11_variant``
in session.json, session_result.json, run.json and the manifest line. No test touches the real store or starts
Claude Code.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from a4_view import trial_digest
from e10_toy import build_e10_store
from e11_toy import PLANT_CONDITION, build_e11_store
from test_harness_fixtures import needs_sandbox
from test_harness_learn import STRINGS, _program_cell

from weightcode.baselines.instance_io import load_instance
from weightcode.data import conditions as C
from weightcode.data.store import dataset_json_path
from weightcode.evalkit.devreadout import readout_run
from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.evalkit.instances import scan_for_tokens
from weightcode.evalkit.targets import deployment_splits, load_checked_split
from weightcode.harness import e11, runner
from weightcode.harness.arms import arm_spec
from weightcode.harness.experience import (
    EXPERIENCE_KEYS,
    RULE_EXPERIENCE_KEYS,
    RULE_EXPERIENCE_SOURCE,
    RULE_STATEMENT_EXPERIENCE,
    RULE_STATEMENT_NO_IMAGE,
    ExperienceError,
    learn_task_rule,
)
from weightcode.harness.finalize import STATUS_DRY_RUN
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.leaks import HINT_WORDS
from weightcode.harness.prompts import PROMPT_DIR, Budgets, render_task
from weightcode.harness.scripted import ScriptedLauncher, call_and_record
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import tool_specs
from weightcode.harness.trial import StudentOverride, TrialError, assemble
from weightcode.models import vit_synth as V

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "PREREG_v0_DRAFT.md"
needs_prereg = pytest.mark.skipif(
    not PREREG.is_file(), reason="docs/PREREG_v0_DRAFT.md is a local research record, not part of the public repository")
RULE = e11.RULE
SENTENCE = e11.RULE_SENTENCES[PLANT_CONDITION]
WALL_S = 300.0
LN = "classifier.norm.weight"
OFFICIAL_SOURCE = {"repository": "https://github.com/hendrycks/robustness",
                   "file": "ImageNet-C/create_c/make_cifar_c.py", "commit": "8190fe329f5f072a06e3a2aea02bb5dda69aed9f", "file_sha256": e11.OFFICIAL_FILE_SHA256}
# What each standard view showed at main f155d9b, before any K1 code (dry runs on the toy stores with retention floor
# 0.005; E11 at 120 min, E10 at 90 min; ``_view_digest``), with only the public namespace migrated.
# E11 LEARN equals a4_view.B_SHOWN_WEIGHTCODE on the trial.
BEFORE_K1 = {'e10_learn': {'trial': {'experience.npz': 'arrays:9da973446ac2aed5e280d146d95fde63576a33bdd549741f701508859c808dea',
                         'forward.py': 'af54f5a1733dd54c0d6913dee2bbbba24d3f52f2efbb4db2e86fdbcba2f9a5db',
                         'model.npz': 'arrays:cb106d3b3556df35689c6e1e562e8eaede025f0c42f9feccf4c97d203e0bd04a',
                         'model_spec.json': '63960864c8c60d0921533540aec63aa16be390d89dd12cdc81488a7ab19d0ca0',
                         'task.json': '70f69c717e184a2189a0aa8b6d6634c4edd4d9fc3bdad9f9c8f0767a21a81811'},
               'system_prompt': '36eb671aef35fb567300d576f13379c4b22ae5538d85373b60bcb50a28770a74',
               'task_prompt': 'cf9bbe6d37af333efb1a05b6da9d5ba57e066988c75f9d194c7b2bcf03c284d3',
               'tools': '49907e9747f8dfe47db8fdc0eb00cec0905ff64ecc326c10cfa918316aba4116'},
 'e11_blind': {'trial': {'experience.npz': 'arrays:76037db24622affb0b0f8ccfde3491122e8a971f7398e7a2a18375cf266c7eff',
                         'forward.py': 'bd0bd988862c996a69bbcb39b979c725a48ce6fb109d2799d2be402a2abf7dc6',
                         'model_shapes.json': '4d31696cb8f9c9385bc610f6aae8c8260001b87f883dad037b494960d0542640',
                         'model_spec.json': '5191cfd3f285c6b6e51e0afcca85296f786bd1f75273a644113552b59dd15bf2',
                         'task.json': 'a80b4cf1c3e2a07f9332c58d12b201a833e4bc60b21f073f163ed511b2c2e448'},
               'system_prompt': '7471546d860b8ea98809ca0564968cda671a85943780c389f75afe1d6abfc9d3',
               'task_prompt': '3afa9472b156c943dff4cc43fc40d54b2a007f9c5838e840ef7e1b4643bf0b5e',
               'tools': '0a69bce1eadfa3be38804d0114830c65c1cbfe3109b6967ee0d846b522e43f9f'},
 'e11_learn': {'trial': {'experience.npz': 'arrays:df5eb9b6c167c882373d6c7d71feca35292fe288e0c6fc9e889d12cb703ebd2b',
                         'forward.py': 'bd0bd988862c996a69bbcb39b979c725a48ce6fb109d2799d2be402a2abf7dc6',
                         'model.npz': 'arrays:b680d1cca43a2ba685e4cc2eb7ad91921561168c72dd5cffd4867f91f1766613',
                         'model_spec.json': '5191cfd3f285c6b6e51e0afcca85296f786bd1f75273a644113552b59dd15bf2',
                         'task.json': '812535932a8d09be994678c8fbe8891d078fd39cb73336f7acce0866f1db7445'},
               'system_prompt': '7471546d860b8ea98809ca0564968cda671a85943780c389f75afe1d6abfc9d3',
               'task_prompt': '8e01eb691073c3acb68469f7e802494df5b35eafd77dc9b32b945b6980fa7b32',
               'tools': '39cd5e0e7cbd06e904be8111307c5708299f3087da050054c7b2d6aebfceaa3b'},
 'e11_learn_2checks': {'trial': {'experience.npz': 'arrays:df5eb9b6c167c882373d6c7d71feca35292fe288e0c6fc9e889d12cb703ebd2b',
                                 'forward.py': 'bd0bd988862c996a69bbcb39b979c725a48ce6fb109d2799d2be402a2abf7dc6',
                                 'model.npz': 'arrays:b680d1cca43a2ba685e4cc2eb7ad91921561168c72dd5cffd4867f91f1766613',
                                 'model_spec.json': '5191cfd3f285c6b6e51e0afcca85296f786bd1f75273a644113552b59dd15bf2',
                                 'task.json': '116d67d7a7e057b7108bc61628345d2e3f92ae4928f13b27ddeea115e0e205b9'},
                       'system_prompt': '7471546d860b8ea98809ca0564968cda671a85943780c389f75afe1d6abfc9d3',
                       'task_prompt': 'e791dfcd279851e9fb78914ea0fee55b497b638f020e2cf78ac9f5de59dcc47e',
                       'tools': '39cd5e0e7cbd06e904be8111307c5708299f3087da050054c7b2d6aebfceaa3b'}}

PROBE = STRINGS + f"""import numpy as np

def update(student, experience):
    print("KEYS", ",".join(sorted(experience)), "SHAPES", [list(experience[k].shape) for k in sorted(experience)])
    w = np.asarray(student["params"][{LN!r}], dtype=np.float64)
    delta = np.zeros_like(w)
    cut = int(round(0.25 * w.size))
    delta[:cut] = -2.0 * w[:cut]
    return {{{LN!r}: delta}}
"""
KERNEL_PROBE = """import json, os
import numpy as np
_files = sorted(os.path.join(root, name) for root, _, names in os.walk("/work") for name in names)
with np.load("/work/trial/experience.npz") as _archive:
    _keys = sorted(_archive.files)
print(json.dumps({"files": _files, "experience": _keys, "store": os.path.exists(STORE)}))
"""


def _state_rule(condition: str = PLANT_CONDITION, **fields: Any) -> str:
    """Rewrite the toy condition record so that it states the official code's fields (``fields`` override them);
    returns the record's text before the rewrite."""
    path = dataset_json_path(e11.DATASET, condition)
    before = path.read_text(encoding="utf-8")
    meta = json.loads(before)
    meta["condition"].update({"corruption": "gaussian_noise", "severity": 5, "constants": "c = 0.10",
                              "source": dict(OFFICIAL_SOURCE), **fields})
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return before


@pytest.fixture
def k1_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> V.SyntheticStore:
    """The toy E11 store under a neutral directory name, with the gaussian_noise_s5 record stating the rule."""
    store = build_e11_store(tmp_path_factory.mktemp("hs") / "store", monkeypatch)
    _state_rule()
    return store


def _id(store: V.SyntheticStore, condition: str = PLANT_CONDITION) -> str:
    return store.ids[(condition, 100)]


def _assembled(store: V.SyntheticStore, directory: Path, **kwargs: Any):  # type: ignore[no-untyped-def]
    (directory / "private").mkdir(parents=True)
    return assemble(read_instance_record(_id(store)).instance_dir, "LEARN", directory, **kwargs)


def _condition_rows(instance_id: str) -> dict[str, np.ndarray]:
    """The instance's own condition rows (the evidence sample, x_new of the LEARN experience) and every condition row
    of its V_evidence and V_check."""
    instance = load_instance(read_instance_record(instance_id).instance_dir)
    splits = deployment_splits(str(instance.private["dataset"]), str(instance.private["condition"]))
    evidence = instance.evidence
    return {"x_new": np.concatenate([evidence["x_fail"], evidence["x_rand"]]),
            "V_evidence": load_checked_split(splits.v_evidence, splits.v_evidence_sha256)["x"],
            "V_check": load_checked_split(splits.v_check, splits.v_check_sha256)["x"]}


def _row_hashes(rows: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows}


def _arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix == ".npy":
        return {"": np.load(path, allow_pickle=False)}
    return {}


def condition_rows_in(paths: list[Path], rows: np.ndarray) -> list[str]:
    """Where a row of ``rows`` (uint8 images [N, C, H, W]) is found among ``paths``: an array holding a row that hashes
    like one (every array whose trailing shape is the image shape, after a lossless cast to uint8), or a row's bytes
    inside a file's bytes or an array's bytes."""
    shape, hashes = rows.shape[1:], _row_hashes(rows)
    raw = [np.ascontiguousarray(row).tobytes() for row in rows]
    found = []
    for path in paths:
        data = path.read_bytes()
        if any(row in data for row in raw):
            found.append(f"{path.name}: bytes")
        for key, array in _arrays(path).items():
            if any(row in np.ascontiguousarray(array).tobytes() for row in raw):
                found.append(f"{path.name}:{key}: bytes")
            if array.ndim < len(shape) or tuple(array.shape[array.ndim - len(shape):]) != tuple(shape):
                continue
            flat = array.reshape(-1, *shape)
            if array.dtype.kind in "iuf" and flat.size and np.all(np.isfinite(flat)) and flat.min() >= 0 \
                    and flat.max() <= 255 and np.array_equal(flat, np.round(flat)):
                if _row_hashes(flat.astype(np.uint8)) & hashes:
                    found.append(f"{path.name}:{key}: rows")
    return found


def _sandbox_files(root: Path) -> list[Path]:
    """Every file the kernel sandbox mounts: /work/trial, /work/patches and /work/scratch."""
    paths = SessionPaths(root)
    return sorted(path for directory in (paths.trial, paths.patches, paths.scratch) for path in directory.rglob("*")
                  if path.is_file())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _view_digest(result: runner.SessionResult) -> dict[str, Any]:
    root = result.root
    session = read_json(root / "session.json")
    spec = arm_spec(str(session["arm"]))
    tools = tool_specs(spec.tools, spec.program, s_old_floor=int(session["kept_rule"]["s_old_floor"]),
                       blind=spec.blind)
    return {"trial": trial_digest(root / "trial"),
            "system_prompt": _sha((root / "prompts" / "system.md").read_text(encoding="utf-8")),
            "task_prompt": _sha((root / "prompts" / "task.md").read_text(encoding="utf-8")),
            "tools": _sha(json.dumps(tools, sort_keys=True))}


def _dry(instance_id: str, arm: str = "LEARN", wall_s: float = 7200.0, **kwargs: Any) -> runner.SessionResult:
    result = runner.run_session(instance_id, arm, 0, effort="max", wall_timeout_s=wall_s, dry_run=True,
                                retention_floor_frac=0.005, **kwargs)
    assert result.status == STATUS_DRY_RUN, result.record
    return result


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


# ---------------------------------------------------------------------------------------------------------------
# Units


@needs_prereg
def test_the_rule_is_the_prereg_sentence() -> None:
    prereg = " ".join(PREREG.read_text(encoding="utf-8").split())
    entry = prereg[prereg.index("K1, the knowledge route on a vision network"):]
    assert SENTENCE[0].lower() + SENTENCE[1:] in entry[:entry.index("Why gaussian")]


def test_the_rule_is_the_official_codes_formula() -> None:
    assert e11.OFFICIAL_FILE_SHA256 == C.SOURCE["file_sha256"]
    assert C.CONSTANTS["gaussian_noise"][5 - 1] == 0.10 and e11.CONDITION_RULES[PLANT_CONDITION] == {
        "corruption": "gaussian_noise", "severity": 5, "constants": "c = 0.10"}
    # the sentence's formula, with one draw per value, is the data module's transcription of the official code
    x = np.random.default_rng(5).integers(0, 256, size=(7, 3, 32, 32), dtype=np.uint8)
    draws = np.random.default_rng(6).normal(loc=0.0, scale=0.10, size=(7, 32, 32, 3))
    stated = np.uint8(np.clip(x / 255 + draws.transpose(0, 3, 1, 2), 0, 1) * 255)
    assert np.array_equal(stated, C._gaussian_noise_chunk(x, 0.10, np.random.default_rng(6)))
    assert e11.check_variant(None) == e11.check_variant("standard") == e11.STANDARD and e11.VARIANTS == (
        "standard", "rule", "unlabelled", "named", "named_nocheck")
    with pytest.raises(ValueError, match="unknown E11 variant"):
        e11.check_variant("rules")


def test_the_rule_task_template_is_learns_with_the_experience_bullet_replaced() -> None:
    learn = (PROMPT_DIR / "task_LEARN.md").read_text(encoding="utf-8").split("\n- ")
    rule = (PROMPT_DIR / "task_LEARN_rule.md").read_text(encoding="utf-8").split("\n- ")
    changed = [index for index, (a, b) in enumerate(zip(learn, rule)) if a != b]
    assert len(learn) == len(rule) and len(changed) == 1
    assert rule[changed[0]].startswith("`experience.npz`: labelled clean examples with the classifier's predictions, "
                                       "and no image of the new condition.")
    assert "$n_new" not in rule[changed[0]] and "x_new" not in rule[changed[0]] and "$n_old" in rule[changed[0]]
    assert max(len(line) for line in (PROMPT_DIR / "task_LEARN_rule.md").read_text().splitlines()) <= 117


def test_the_rule_task_view_refuses_what_it_cannot_derive(k1_store: V.SyntheticStore, tmp_path: Path) -> None:
    view = _assembled(k1_store, tmp_path / "learn")
    task, n_old = view.task, int(view.experience["n_old"])
    once = learn_task_rule(task, SENTENCE, n_old)
    with pytest.raises(ExperienceError, match="already states the rule"):
        learn_task_rule(once, SENTENCE, n_old)
    reworded = json.loads(json.dumps(task))
    reworded["statement"] = reworded["statement"].replace("with the classifier's predictions on both.", "and more.")
    with pytest.raises(ExperienceError, match="statement does not say"):
        learn_task_rule(reworded, SENTENCE, n_old)
    twice = json.loads(json.dumps(task))
    twice["statement"] += " " + RULE_STATEMENT_EXPERIENCE
    with pytest.raises(ExperienceError, match="statement does not say"):
        learn_task_rule(twice, SENTENCE, n_old)
    relabelled = json.loads(json.dumps(task))
    relabelled["training_free_rule"]["program"][0] = "in this task the evidence is experience.npz"
    with pytest.raises(ExperienceError, match="training-free rule does not say"):
        learn_task_rule(relabelled, SENTENCE, n_old)
    with pytest.raises(ExperienceError, match="old part and both id arrays"):
        learn_task_rule(task, SENTENCE, n_old + 1)


# ---------------------------------------------------------------------------------------------------------------
# The view


def test_the_rule_view_shows_the_clean_rows_alone_and_states_the_rule(k1_store: V.SyntheticStore,
                                                                     tmp_path: Path) -> None:
    standard = _assembled(k1_store, tmp_path / "standard")
    view = _assembled(k1_store, tmp_path / "rule", e11_variant=RULE)
    named = _assembled(k1_store, tmp_path / "named", e11_variant="standard")
    assert named.files == standard.files and named.task == standard.task and named.e11_rule is None
    assert "e11_rule" not in standard.as_dict() and view.as_dict()["e11_rule"] == view.e11_rule

    # the files: the same names; the weights, layout and forward are the standard view's, byte for byte
    assert sorted(view.files) == sorted(standard.files)
    assert {k: v for k, v in view.files.items() if k not in ("experience.npz", "task.json")} == {
        k: v for k, v in standard.files.items() if k not in ("experience.npz", "task.json")}
    assert view.limits == standard.limits and view.shown_model_sha256 == standard.shown_model_sha256
    # the same clean V_check companion for score_old (in each view's own private directory)
    assert {k: v for k, v in view.clean_check.items() if k != "path"} == {
        k: v for k, v in standard.clean_check.items() if k != "path"}

    # experience.npz: the old part of the standard experience alone
    with np.load(view.trial_dir / "experience.npz") as archive:
        shown = {key: archive[key] for key in archive.files}
    with np.load(standard.trial_dir / "experience.npz") as archive:
        full = {key: archive[key] for key in archive.files}
    assert sorted(shown) == sorted(RULE_EXPERIENCE_KEYS) and sorted(full) == sorted(EXPERIENCE_KEYS)
    assert all(np.array_equal(shown[key], full[key]) for key in shown)
    record = view.experience
    assert record["source"] == RULE_EXPERIENCE_SOURCE == view.evidence_source and record["variant"] == RULE
    assert (record["n_new"], record["n_new_withheld"], record["n_new_fail_withheld"]) == (0, 64, 16)
    assert record["n_old"] == standard.experience["n_old"] and "n_new_fail" not in record
    assert record["learn_content_sha256"] == standard.experience["content_sha256"]
    assert record["replay"] == standard.experience["replay"] and record["withheld_keys"] == [
        "pred_new", "split_ids_new", "x_new", "y_new"]

    # task.json: the statement, the experience section and the program's label item differ, nothing else
    assert _differences(view.task, standard.task) == [".experience.ids.description", ".experience.ids.keys",
                                                      ".experience.new", ".statement", ".training_free_rule.program"]
    assert view.task["statement"] == standard.task["statement"].replace(
        RULE_STATEMENT_EXPERIENCE, f"{SENTENCE} {RULE_STATEMENT_NO_IMAGE}")
    assert view.task["experience"] == {"old": standard.task["experience"]["old"], "ids": {
        "keys": ["split_ids_old"], "description": "one opaque id per example, in x_old order"}}
    program, learn_program = view.task["training_free_rule"]["program"], standard.task["training_free_rule"]["program"]
    assert program[0] == "in this task the evidence is experience.npz, and its labels are y_old"
    assert program[1:] == learn_program[1:] and learn_program[0].endswith("y_new and y_old")
    for name in ("x_new", "y_new", "pred_new", "split_ids_new"):
        assert name not in (view.trial_dir / "task.json").read_text()

    # the rule's provenance: the condition record's fields and hash
    path = dataset_json_path(e11.DATASET, PLANT_CONDITION)
    assert view.e11_rule == {"variant": RULE, "condition": PLANT_CONDITION, "sentence": SENTENCE,
                             "corruption": "gaussian_noise", "severity": 5, "constants": "c = 0.10",
                             "source": OFFICIAL_SOURCE,
                             "condition_dataset_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_the_rule_sentence_appears_exactly_once(k1_store: V.SyntheticStore, tmp_path: Path) -> None:
    standard = _assembled(k1_store, tmp_path / "standard")
    view = _assembled(k1_store, tmp_path / "rule", e11_variant=RULE)
    budgets = Budgets(patch_checks=8, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=7200.0)
    prompt = render_task("LEARN", view.task, view.limits, budgets, view.input_files, e11_variant=RULE)
    plain = render_task("LEARN", standard.task, standard.limits, budgets, standard.input_files)
    assert prompt.count(SENTENCE) == 1 and json.dumps(view.task).count(SENTENCE) == 1
    assert (view.trial_dir / "task.json").read_text().count(SENTENCE) == 1
    assert plain.count(SENTENCE) == 0 and json.dumps(standard.task).count(SENTENCE) == 0
    for path in view.trial_dir.iterdir():
        if path.name != "task.json" and path.suffix != ".npz":
            assert SENTENCE not in path.read_text(), path.name
    assert prompt.count(RULE_STATEMENT_NO_IMAGE) == 1 and RULE_STATEMENT_EXPERIENCE not in prompt
    assert "$" not in prompt and "x_new" not in prompt and "split_ids_new" not in prompt and "y_new" not in prompt
    with pytest.raises(KeyError):  # the standard template needs the new part, which this task.json does not have
        render_task("LEARN", view.task, view.limits, budgets, view.input_files)
    with pytest.raises(ValueError, match="rule_only task.json has a new experience part"):
        render_task("LEARN", standard.task, standard.limits, budgets, standard.input_files, e11_variant=RULE)
    # the prompt is the standard one but for the statement and the experience bullet
    removed = [line for line in difflib.ndiff(plain.splitlines(), prompt.splitlines()) if line.startswith("- ")]
    assert [line[2:12] for line in removed] == ["Make a tra", "- `experie", "  64 examp", "  predicti",
                                                "  `model_s", "- in this "]
    for arm in ("LEARN_BLIND", "M2"):
        with pytest.raises(ValueError, match="E11 rule view is a LEARN view"):
            render_task(arm, view.task, view.limits, budgets, view.input_files, e11_variant=RULE)


def test_no_array_the_sandbox_can_read_holds_a_condition_row(k1_store: V.SyntheticStore, tmp_path: Path) -> None:
    rows = _condition_rows(_id(k1_store))
    assert len(rows["x_new"]) == 64 and rows["x_new"].dtype == np.uint8
    standard = _assembled(k1_store, tmp_path / "standard")
    view = _assembled(k1_store, tmp_path / "rule", e11_variant=RULE)
    standard_files = sorted(standard.trial_dir.iterdir())
    view_files = sorted(view.trial_dir.iterdir())
    # positive control: the detector finds the 64 x_new rows in the standard view's experience
    found = condition_rows_in(standard_files, rows["x_new"])
    assert "experience.npz: bytes" in found and "experience.npz:x_new: rows" in found
    with np.load(standard.trial_dir / "experience.npz") as archive:
        assert _row_hashes(archive["x_new"]) == _row_hashes(rows["x_new"])
    # the rule view: no condition row of the instance, of V_evidence or of V_check, anywhere in the trial
    for name, x in rows.items():
        assert condition_rows_in(view_files, x) == [], name
    with np.load(view.trial_dir / "experience.npz") as archive:
        assert not _row_hashes(archive["x_old"]) & set().union(*(_row_hashes(x) for x in rows.values()))


# ---------------------------------------------------------------------------------------------------------------
# Sessions


@needs_sandbox
def test_a_rule_dry_run_stages_no_condition_row_and_keeps_e11s_leak_words(k1_store: V.SyntheticStore) -> None:
    instance_id = _id(k1_store)
    standard, rule = _dry(instance_id), _dry(instance_id, e11_variant=RULE)
    plain, ours = read_json(standard.root / "session.json"), read_json(rule.root / "session.json")
    # what the sandbox can read: the trial files alone (patches and scratch are empty), none with a condition row
    staged = _sandbox_files(rule.root)
    assert [path.name for path in staged] == sorted(ours["trial_files_sha256"])
    for name, x in _condition_rows(instance_id).items():
        assert condition_rows_in(staged, x) == [], name
    assert "experience.npz:x_new: rows" in condition_rows_in(_sandbox_files(standard.root),
                                                             _condition_rows(instance_id)["x_new"])
    # the leak words: E11's, the condition's name among them, no exemption; the rule sentence passes the scan
    assert ours["forbidden_strings"] == plain["forbidden_strings"] and ours["forbidden_strings"]["text_exempt"] == []
    text = ours["forbidden_strings"]["text"]
    assert {"gaussian", "noise", "gaussian_noise", PLANT_CONDITION, "s5", *e11.leak_words()} <= set(text)
    assert ours["leak_scan"]["passed"] is True and ours["leak_scan"]["hits"] == {}
    assert scan_for_tokens(SENTENCE, text) == [] and [word for word in HINT_WORDS if word in SENTENCE.lower()] == []
    assert scan_for_tokens("each image gets gaussian noise", text) == ["gaussian", "noise"]  # the words stay live
    # the session record and the prompts: the system prompt and tools are the standard view's
    assert ours["e11_rule"] == ours["trial"]["e11_rule"] and ours["e11_rule"]["sentence"] == SENTENCE
    assert "e11_rule" not in plain and "e11_rule" not in plain["trial"]
    assert ours["experience"]["source"] == RULE_EXPERIENCE_SOURCE
    assert {k: v for k, v in ours["clean_check"].items() if k != "path"} == {
        k: v for k, v in plain["clean_check"].items() if k != "path"}
    digest, before = _view_digest(rule), _view_digest(standard)
    assert digest["system_prompt"] == before["system_prompt"] and digest["tools"] == before["tools"]
    task = (rule.root / "prompts" / "task.md").read_text()
    assert task.count(SENTENCE) == 1 and (rule.root / "prompts" / "system.md").read_text().count(SENTENCE) == 0


@needs_sandbox
def test_the_other_e11_views_are_byte_identical_to_before_k1(k1_store: V.SyntheticStore) -> None:
    instance_id = _id(k1_store)
    assert _view_digest(_dry(instance_id)) == BEFORE_K1["e11_learn"]
    assert _view_digest(_dry(instance_id, e11_variant="standard")) == BEFORE_K1["e11_learn"]
    assert _view_digest(_dry(instance_id, "LEARN_BLIND")) == BEFORE_K1["e11_blind"]
    assert _view_digest(_dry(instance_id, max_checks=2)) == BEFORE_K1["e11_learn_2checks"]


@needs_sandbox
def test_the_e10_view_is_byte_identical_to_before_k1_and_refuses_the_rule(tmp_path_factory: pytest.TempPathFactory,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_e10_store(tmp_path_factory.mktemp("hs") / "store", monkeypatch)
    instance_id = store.ids[(PLANT_CONDITION, 100)]
    assert _view_digest(_dry(instance_id, wall_s=5400.0)) == BEFORE_K1["e10_learn"]
    dry_runs = store.store.root / "runs" / runner.DRY_RUNS_DIR
    before = sorted(dry_runs.iterdir())
    with pytest.raises(InstanceRefused, match="for an E11 evidence-v1 instance"):
        _dry(instance_id, e11_variant=RULE)
    assert sorted(dry_runs.iterdir()) == before


def test_the_rule_view_is_refused_where_it_does_not_apply(k1_store: V.SyntheticStore, tmp_path: Path) -> None:
    instance_id = _id(k1_store)
    runs = k1_store.root / "runs"
    before = sorted(path.relative_to(runs) for path in runs.rglob("*"))
    launcher = ScriptedLauncher(lambda client, replies: None)
    cases = [
        ("LEARN", {"e11_variant": "rules"}, ValueError, "unknown E11 variant"),
        ("LEARN_BLIND", {"e11_variant": RULE}, ValueError, "LEARN arm only"),
        ("M2", {"e11_variant": RULE}, ValueError, "LEARN arm only"),
        ("LEARN", {"e11_variant": RULE, "loop_context": Path("context.json")}, ValueError, "refused in a loop round"),
        ("LEARN", {"e11_variant": RULE, "library_sessions": ["s0123456789abcdef"]}, ValueError, "no library"),
        ("LEARN", {"e11_variant": RULE, "task_prompt_override": "a prompt"}, ValueError, "prompt override"),
    ]
    for arm, kwargs, error, text in cases:
        with pytest.raises(error, match=text):
            runner.run_session(instance_id, arm, 0, effort="max", wall_timeout_s=WALL_S, skip_audit=True,
                               launcher=launcher, **kwargs)
    with pytest.raises(InstanceRefused, match=r"states a rule for \['gaussian_noise_s5'\] only, not 'contrast_s5'"):
        runner.run_session(_id(k1_store, "contrast_s5"), "LEARN", 0, effort="max", wall_timeout_s=WALL_S,
                           launcher=launcher, e11_variant=RULE)
    # a condition record that does not state the official code's fields: the toy store's own, a wrong constant, a
    # wrong source file
    path = dataset_json_path(e11.DATASET, PLANT_CONDITION)
    stated = path.read_text()
    original = json.loads(stated)
    original["condition"] = {key: value for key, value in original["condition"].items()
                             if key not in ("constants", "source")}
    for text, problem in ((json.dumps(original), "states"), (None, "states"), ("source", "official code file")):
        if text is None:
            _state_rule(constants="c = 0.08")
        elif text == "source":
            _state_rule(source={**OFFICIAL_SOURCE, "file_sha256": "0" * 64})
        else:
            path.write_text(text)
        with pytest.raises(InstanceRefused, match=problem):
            runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                               e11_variant=RULE)
        with pytest.raises(TrialError, match=problem):
            _assembled(k1_store, tmp_path / f"refused_{problem.replace(' ', '_')}_{len(text or '')}",
                       e11_variant=RULE)
        path.write_text(stated)
    assert sorted(path.relative_to(runs) for path in runs.rglob("*")) == before  # refused before any write
    # assembly refuses another arm and a student override (a loop round) before reading anything
    override = StudentOverride(params={}, experience={}, record={}, keys=(), task=lambda task: task)
    for arm, kwargs in (("LEARN_BLIND", {}), ("LEARN", {"override": override})):
        with pytest.raises(TrialError, match="assembled in the LEARN arm on the instance's own student"):
            assemble(tmp_path / "none", arm, tmp_path / "none", e11_variant=RULE, **kwargs)
    with pytest.raises(TrialError, match="unknown E11 variant"):
        assemble(tmp_path / "none", "LEARN", tmp_path / "none", e11_variant="rules")
    for private in ({"dataset": "cifar10_r20", "evidence_rule": EVIDENCE_RULE_V1, "condition": PLANT_CONDITION},
                    {"dataset": e11.DATASET, "evidence_rule": "fewshot-v1", "condition": PLANT_CONDITION}):
        with pytest.raises(e11.E11HarnessError, match="for an E11 evidence-v1 instance"):
            e11.condition_rule(private)


def test_the_session_command_takes_the_rule_variant() -> None:
    spec = importlib.util.spec_from_file_location("run_editor_session_k1", REPO / "scripts" / "run_editor_session.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = ["--instance", "i00000000", "--arm", "LEARN", "--repeat", "0", "--effort", "max", "--wall-timeout-min",
            "120", "--retention-floor-frac", "0.005"]
    assert module.parse_args(base).e11_variant is None
    assert module.parse_args([*base, "--e11-variant", "rule"]).e11_variant == RULE
    with pytest.raises(SystemExit):
        module.parse_args([*base, "--e11-variant", "rules"])


@needs_sandbox
def test_a_rule_session_gives_the_program_the_clean_rows_alone_and_reads_out(k1_store: V.SyntheticStore) -> None:
    instance_id = _id(k1_store)
    probe = f"STORE = {str(k1_store.root)!r}\n" + KERNEL_PROBE

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": probe})
        call_and_record(client, replies, "python", {"code": _program_cell({"probe.py": PROBE})})
        call_and_record(client, replies, "patch_check", {"program_file": "probe.py"})
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "probe.py", "rationale": "it scored"})

    launcher = ScriptedLauncher(script)
    result = runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                                skip_audit=True, retention_floor_frac=0.005, e11_variant=RULE)
    assert result.status == "complete", result.record.get("infra_reasons")
    kernel, _, check, status, declared = launcher.replies
    seen = json.loads(kernel.payload["stdout"])
    session = read_json(result.root / "session.json")
    assert seen["experience"] == sorted(RULE_EXPERIENCE_KEYS) and not seen["store"]
    assert seen["files"] == [f"/work/trial/{name}" for name in sorted(session["trial_files_sha256"])]
    assert not check.is_error, check.payload
    stdout = check.payload["program_stdout"]
    assert stdout.startswith("KEYS " + ",".join(sorted(RULE_EXPERIENCE_KEYS)) + " SHAPES ")
    assert check.payload["score_new"] > 0 and check.payload["score_old"] >= 0  # the planted LayerNorm edit
    assert declared.payload["valid"] and status.payload["checked"][0]["program_file"] == "probe.py"
    # the program ran on the trial's own files: the weights and the clean rows alone
    run = read_json(result.root / "attempts" / "a0000" / "program_run.json")  # a0001 is the declared final
    assert run["inputs_sha256"]["experience.npz"] == session["trial_files_sha256"]["experience.npz"]
    assert run["inputs_sha256"]["params.npz"] == session["trial_files_sha256"]["model.npz"]
    # no reply holds a condition row; the final and the V_select readout run as in an E11 session
    replies = json.dumps([reply.payload for reply in launcher.replies])
    for x in _condition_rows(instance_id).values():
        assert not any(str(row.ravel()[:16].tolist())[1:-1] in replies for row in x)
    final = read_json(SessionPaths(result.root).selection_json)["final"]
    assert final["patch_id"] == "probe.py"
    readout = readout_run(result.root / "run.json", clean=True)
    assert readout["condition"] == PLANT_CONDITION and readout["readouts"]["delta_acc"] > 0
    assert session["e11_rule"]["sentence"] == SENTENCE and session["evidence_source"] == RULE_EXPERIENCE_SOURCE


@needs_sandbox
def test_a_rule_session_is_marked_in_every_record_and_a_lookup_by_slot_sees_two(k1_store: V.SyntheticStore) -> None:
    """A rule-view session and a standard session on the same (instance, arm, repeat): only the rule session carries
    ``e11_variant``, in session.json, session_result.json, run.json and its manifest line; a lookup by the slot alone
    (``scripts/run_session_chain.py``) finds both and so refuses, and the dry run's records carry it too."""
    instance_id = _id(k1_store)

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"probe.py": PROBE})})
        call_and_record(client, replies, "patch_check", {"program_file": "probe.py"})
        call_and_record(client, replies, "declare_final", {"program_file": "probe.py", "rationale": "it scored"})

    roots = {}
    for variant in (None, RULE):
        result = runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=WALL_S,
                                    launcher=ScriptedLauncher(script), skip_audit=True, retention_floor_frac=0.005,
                                    e11_variant=variant)
        assert result.status == "complete", result.record.get("infra_reasons")
        roots[variant] = result.root
    standard, rule = roots[None], roots[RULE]
    for name in ("session.json", "session_result.json", "run.json"):
        assert read_json(rule / name)["e11_variant"] == RULE, name
        assert "e11_variant" not in read_json(standard / name), name
    assert set(read_json(rule / "run.json")) - set(read_json(standard / "run.json")) == {"e11_variant"}
    lines = [json.loads(line) for line in (k1_store.root / "runs" / "manifest.jsonl").read_text().splitlines()]
    by_session = {line["session_id"]: line for line in lines}
    plain, ours = by_session[standard.name], by_session[rule.name]
    assert ours["e11_variant"] == RULE and "e11_variant" not in plain and set(ours) - set(plain) == {"e11_variant"}
    assert (plain["instance_id"], plain["arm"], plain["repeat"]) == (ours["instance_id"], ours["arm"], ours["repeat"])
    # the chain's lookup by (instance, arm, repeat) sees two directories and stops instead of picking one
    spec = importlib.util.spec_from_file_location("run_session_chain_k1", REPO / "scripts" / "run_session_chain.py")
    assert spec is not None and spec.loader is not None
    chain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chain)
    assert sorted(chain._matching(standard.parent, sorted([standard.name, rule.name]), instance_id, "LEARN", 0)) == \
        sorted([standard.name, rule.name])
    # a dry run's records carry the key as well
    dry_rule, dry_standard = _dry(instance_id, e11_variant=RULE), _dry(instance_id)
    for name in ("session.json", "session_result.json"):
        assert read_json(dry_rule.root / name)["e11_variant"] == RULE
        assert "e11_variant" not in read_json(dry_standard.root / name)
