"""PREREG E11 in the harness (``weightcode.harness.e11``): a ViT-Lite student on E10's conditions.

Units: the dataset name and leak words, the MCP floors, the student check, the LEARN limits on the ViT-Lite-7/4 tensor
shapes themselves (LayerNorm and every bias 1-D: exempt from rel-Frob and counted in the 20%; the class token, the
positional embedding, qkv and the other weights under rel-Frob; max_ops = 85), the synthetic forward.py against an
independent torch float64 rebuild, and the planted labels of the synthetic store. Sessions on the toy E11 store
(``e11_toy``: tiny synthetic ViT-Lite students, dev seeds 100 to 105, E10's four condition names): the forbidden strings,
the lookups, a dry run, a scripted LEARN session that checks a LayerNorm program and a qkv program and is refused on
rel-Frob (qkv, positional embedding, class token) and on the 20% cap, its readout on V_select, the refusal of a
student that fails the check, and the C transfer. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from e11_toy import CLASSES, PLANT_CONDITION, build_e11_store, e11_store  # noqa: F401
from test_harness_fixtures import needs_sandbox
from test_harness_learn import STRINGS, _program_cell

from weightcode.common.identity import params_sha256
from weightcode.evalkit.devreadout import readout_run
from weightcode.evalkit.e10 import e10_program_limits
from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.evalkit.instances import leak_tokens, scan_for_tokens
from weightcode.harness import e10, e11, runner
from weightcode.harness.arms import applied_limits
from weightcode.harness.finalize import KIND_CHECKED, STATUS_DRY_RUN
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.program import write_derived_patch
from weightcode.harness.scripted import ScriptedLauncher, call_and_record
from weightcode.harness.session import SessionPaths
from weightcode.harness.transfer import run_transfer
from weightcode.models import vit_synth as V
from weightcode.patch import Limits, PatchError, apply_patch, task_limits
from weightcode.sandbox.program import ProgramSandbox

WALL_S = 300.0
ARCH_WORDS = ("vit", "transformer", "attention", "layernorm", "patch", "token", "positional", "gelu", "softmax")

LN_PLANT = STRINGS + """import numpy as np

def update(student, experience):
    w = np.asarray(student["params"]["classifier.norm.weight"], dtype=np.float64)
    delta = np.zeros_like(w)
    cut = int(round(0.25 * w.size))
    delta[:cut] = -2.0 * w[:cut]
    return {"classifier.norm.weight": delta}
"""
QKV_Q = STRINGS + """import numpy as np

def update(student, experience):
    w = np.asarray(student["params"]["classifier.blocks.0.self_attn.qkv.weight"], dtype=np.float64)
    delta = np.zeros_like(w)
    d = w.shape[1]
    delta[:d] = 0.3 * w[:d]
    return {"classifier.blocks.0.self_attn.qkv.weight": delta}
"""


def _scaled(name: str, factor: float) -> str:
    return STRINGS + f"""import numpy as np

def update(student, experience):
    return {{{name!r}: {factor!r} * np.asarray(student["params"][{name!r}], dtype=np.float64)}}
"""


QKV_BIG = _scaled("classifier.blocks.0.self_attn.qkv.weight", 0.6)
POS_BIG = _scaled("classifier.positional_emb", 0.6)
CLS_BIG = _scaled("classifier.class_emb", 0.6)
WIDE = STRINGS + """import numpy as np

def update(student, experience):
    p = student["params"]
    out = {name: np.full(p[name].shape, 1e-4) for name in p if ".norm" in name or "pre_norm" in name}
    for name in ("classifier.blocks.0.linear1.weight", "classifier.blocks.0.linear2.weight"):
        out[name] = np.full(p[name].shape, 1e-4)
    return out
"""
EVERY = STRINGS + """import numpy as np

def update(student, experience):
    out = {}
    for name, value in student["params"].items():
        delta = np.zeros(value.shape)
        delta.flat[0] = 1e-3
        out[name] = delta
    return out
"""


def _sandbox() -> ProgramSandbox:
    return ProgramSandbox(python=runner.SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)


def _module(cfg: V.VitConfig) -> types.ModuleType:
    module = types.ModuleType("forward_under_test")
    exec(V.render_forward(cfg, V.SYNTH_MEAN, V.SYNTH_STD), module.__dict__)  # noqa: S102 - our own template
    return module


# ---------------------------------------------------------------------------------------------------------------
# Units


def test_the_dataset_name_and_conditions() -> None:
    assert e11.CONDITIONS == e10.STREAM_CONDITIONS and e11.SOURCE_SEED == 100 and e11.TRANSFER_SEEDS == e10.TRANSFER_SEEDS
    assert e11.DATASET != e10.DATASET
    tokens = leak_tokens(e11.DATASET)
    assert not set(ARCH_WORDS) & set(tokens) and "vit" in leak_tokens("cifar10_vit")  # why the name avoids it
    assert e11.is_e11({"dataset": e11.DATASET, "evidence_rule": EVIDENCE_RULE_V1})
    assert not e11.is_e11({"dataset": e11.DATASET, "evidence_rule": "fewshot-v1"})
    assert not e11.is_e11({"dataset": e10.DATASET, "evidence_rule": EVIDENCE_RULE_V1})
    assert not e10.is_e10({"dataset": e11.DATASET, "evidence_rule": EVIDENCE_RULE_V1})


def test_the_leak_words_are_e10s_and_architecture_words_are_not_secrets() -> None:
    words = e11.leak_words()
    assert words == e10.leak_words()
    scanned = sorted({*words, *leak_tokens(e11.DATASET)})
    text = "ViT-Lite-7/4: a Transformer with attention, LayerNorm, a class token, patch tokens, positional, GELU, softmax"
    assert scan_for_tokens(text, scanned) == []
    assert scan_for_tokens("GELU, the Gaussian error linear unit", scanned) == ["gaussian"]  # a forward.py hazard


def test_an_e11_session_gets_e10s_mcp_floors_and_refuses_shorter_ones() -> None:
    defaults = (runner.DEFAULT_MCP_TOOL_TIMEOUT_MS, runner.DEFAULT_MCP_CONNECT_TIMEOUT_MS)
    ours = {"dataset": e11.DATASET, "evidence_rule": EVIDENCE_RULE_V1}
    assert e11.mcp_timeouts(ours, None, None, defaults) == (e10.MCP_TOOL_TIMEOUT_MS, e10.MCP_CONNECT_TIMEOUT_MS)
    assert (e11.MCP_TOOL_TIMEOUT_MS, e11.MCP_CONNECT_TIMEOUT_MS) == (1_200_000, 900_000)
    with pytest.raises(ValueError, match=r"an E11 session needs .* \(the ViT-Lite check predictions\)"):
        e11.mcp_timeouts(ours, 600_000, None, defaults)
    assert e11.mcp_timeouts({"dataset": "cifar10", "evidence_rule": EVIDENCE_RULE_V1}, None, None, defaults) == defaults
    e10_private = {"dataset": e10.DATASET, "evidence_rule": EVIDENCE_RULE_V1}
    with pytest.raises(ValueError, match=r"an E10 session needs .* \(the ResNet-20 check predictions\), got 600000"):
        e10.mcp_timeouts(e10_private, 600_000, None, defaults)  # E10's message as before the refactor


def test_the_student_check_finds_the_tokens_and_refuses_batchnorm_or_a_squeezed_class_token() -> None:
    params = V.init_params(V.VIT_LITE_7_4, 100)
    summary = e11.check_student(params)
    assert summary == {"n_tensors": 85, "n_entries": 3_716_874, "class_token": "classifier.class_emb",
                       "positional_embedding": "classifier.positional_emb", "n_rel_frob_tensors": 32,
                       "n_exempt_1d_tensors": 53, "n_exempt_1d_entries": 15_114}
    tiny = V.init_params(V.TINY, 100)
    for renamed in ({"cls_token": (1, 1, 16), "pos_embed": (1, 5, 16)},
                    {"class_token": (1, 1, 16), "encoder.pos_embedding": (1, 5, 16)}):
        other = {k: v for k, v in tiny.items() if k not in (V.CLASS_TOKEN, V.POS_EMBED)}
        other.update({name: np.zeros(shape, np.float32) for name, shape in renamed.items()})
        assert e11.check_student(other)["n_tensors"] == 30
    with pytest.raises(e11.E11HarnessError, match="class_emb has shape \\[16\\]: rank < 2"):
        e11.check_student({**tiny, V.CLASS_TOKEN: np.zeros(16, np.float32)})
    with pytest.raises(e11.E11HarnessError, match="BatchNorm statistics"):
        e11.check_student({**tiny, "bn1.running_mean": np.zeros(16, np.float32)})
    with pytest.raises(e11.E11HarnessError, match="2 positional embedding tensors"):
        e11.check_student({**tiny, "pos_embed": np.zeros((1, 5, 16), np.float32)})
    with pytest.raises(e11.E11HarnessError, match="0 class token tensors"):
        e11.check_student({k: v for k, v in tiny.items() if k != V.CLASS_TOKEN})


def test_the_learn_limits_on_the_vit_lite_tensors(tmp_path: Path) -> None:
    shapes = V.param_shapes(V.VIT_LITE_7_4)
    params = {name: np.ones(shape, np.float32) for name, shape in shapes.items()}
    sha = params_sha256(params)
    n_entries = sum(value.size for value in params.values())
    norms = V.layernorm_tensors(V.VIT_LITE_7_4)
    one_d = [name for name in params if params[name].ndim == 1]
    n_norm = sum(params[name].size for name in norms)
    assert (len(params), n_entries, len(one_d), len(norms), n_norm) == (85, 3_716_874, 53, 30, 7680)
    assert all(params[name].ndim >= 2 for name in (V.CLASS_TOKEN, V.POS_EMBED, V.PATCH_WEIGHT, V.HEAD_WEIGHT))
    limits = applied_limits(task_limits(Limits()), "LEARN", len(params))
    assert limits == e10_program_limits(task_limits(Limits()), len(params))
    assert (limits.max_ops, limits.max_rel_frob, limits.rel_frob_min_ndim, limits.max_changed_frac) == (85, 0.5, 2, 0.2)

    def apply(deltas: dict[str, np.ndarray], name: str, limits: Limits = limits) -> Any:
        directory = tmp_path / name
        directory.mkdir()
        update = SimpleNamespace(deltas=deltas, hypothesis="h", predicted_effect="p")
        patch = write_derived_patch(update, params, sha, directory)  # type: ignore[arg-type]
        return apply_patch(params, patch, directory, limits, sha)

    big_norms = {name: np.full(params[name].shape, 9.0) for name in norms}  # 1-D: rel-Frob 9, exempt
    assert apply(big_norms, "norms").n_changed == n_norm
    assert apply({name: np.full(params[name].shape, 9.0) for name in one_d}, "one_d").n_changed == 15_114
    for tensor in (V.POS_EMBED, V.CLASS_TOKEN, "classifier.blocks.3.self_attn.qkv.weight", V.PATCH_WEIGHT):
        with pytest.raises(PatchError, match=f"tensor '{tensor}': relative Frobenius norm of the change is 0.6"):
            apply({tensor: np.full(params[tensor].shape, 0.6)}, f"frob_{tensor}")
    assert apply({V.POS_EMBED: np.full(params[V.POS_EMBED].shape, 0.4)}, "pos_within").n_changed == 65 * 256
    cap = int(0.2 * n_entries)  # 743,374 entries
    wide = [f"classifier.blocks.{i}.linear1.weight" for i in range(5)]
    rest = cap - n_norm - 5 * 131_072
    partial = np.zeros(131_072)
    partial[:rest] = 0.01
    target = "classifier.blocks.5.linear2.weight"
    within = {**big_norms, **{name: np.full(params[name].shape, 0.01) for name in wide},
              target: partial.reshape(params[target].shape)}
    assert apply(within, "at_cap").n_changed == cap
    over = dict(within, **{target: np.where(np.arange(131_072) <= rest, 0.01, 0.0).reshape(params[target].shape)})
    with pytest.raises(PatchError, match="size: 743375 of 3716874 parameters change .*above the limit 0.2"):
        apply(over, "over")
    without_norms = {key: value for key, value in over.items() if key not in norms}
    assert apply(without_norms, "no_norms").n_changed == cap + 1 - n_norm  # the LayerNorm entries broke the cap
    one_each = {name: np.where(np.arange(params[name].size) == 0, 1e-3, 0.0).reshape(params[name].shape)
                for name in params}
    assert apply(one_each, "every_tensor").n_changed == 85  # one op per tensor, max_ops = 85
    with pytest.raises(PatchError, match="85"):
        apply(one_each, "every_tensor_at_64", Limits(**{**task_limits(limits), "max_ops": 64}))


def test_the_forward_matches_an_independent_torch_rebuild() -> None:
    pytest.importorskip("torch")
    for cfg, n in ((V.TINY, 64), (V.VIT_LITE_7_4, 6)):
        module = _module(cfg)
        assert module.expected_shapes() == V.param_shapes(cfg)
        params = V.init_params(cfg, 100)
        x = np.random.default_rng(7).integers(0, 256, size=(n, *cfg.image_shape), dtype=np.uint8)
        ref = V.torch_logits(cfg, V.SYNTH_MEAN, V.SYNTH_STD, params, x)
        ours64 = module.features(params, x) @ params[V.HEAD_WEIGHT].astype(np.float64).T + params[V.HEAD_BIAS]
        assert np.max(np.abs(ours64 - ref)) <= 1e-10 * np.max(np.abs(ref))
        ours = np.asarray(module.forward(params, x), np.float64)
        assert module.forward(params, x).dtype == np.float32
        assert np.max(np.abs(ours - ref)) <= 1e-6 * np.max(np.abs(ref)) and (ours.argmax(1) == ref.argmax(1)).all()
    with pytest.raises(ValueError, match="needs 3 mean"):
        V.render_forward(V.TINY, (0.5,), (0.25,))
    with pytest.raises(KeyError, match="unexpected"):
        _module(V.TINY).forward({**V.init_params(V.TINY, 1), "extra": np.zeros(1)},
                                np.zeros((1, 3, 8, 8), np.uint8))


def test_the_planted_labels_never_let_the_plant_break_a_row() -> None:
    rng = np.random.default_rng(3)
    for n_classes in (4, 10):
        base_clean, plant_clean, base_cond, plant_cond = (rng.integers(0, n_classes, 4000) for _ in range(4))
        y = V.planted_labels(base_clean, plant_clean, base_cond, plant_cond, n_classes, rng)

        def score(base: np.ndarray, child: np.ndarray) -> tuple[int, int]:
            return int(np.sum((base != y) & (child == y))), int(np.sum((base == y) & (child != y)))

        (fixed_new, broken_new), (fixed_old, broken_old) = score(base_cond, plant_cond), score(base_clean, plant_clean)
        assert broken_new == broken_old == 0 and fixed_new > 0 and fixed_old >= 0
    with pytest.raises(ValueError, match="at least 4 classes"):
        V.planted_labels(base_clean, plant_clean, base_cond, plant_cond, 3, rng)


def test_the_synthetic_files_pass_the_e11_text_scan() -> None:
    cfg = V.VIT_LITE_7_4
    tokens = sorted({*e11.leak_words(), *leak_tokens(e11.DATASET)})
    spec = V.build_model_spec("m12345678", cfg, V.CLASS_NAMES_10, V.SYNTH_MEAN, V.SYNTH_STD, V.init_params(cfg, 1))
    assert scan_for_tokens(V.render_forward(cfg, V.SYNTH_MEAN, V.SYNTH_STD), tokens) == []
    assert scan_for_tokens(json.dumps(spec), tokens) == [] and scan_for_tokens(" ".join(spec["tensors"]), tokens) == []
    assert scan_for_tokens(json.dumps(spec), leak_tokens("cifar10_vit")) == ["vit"]


# ---------------------------------------------------------------------------------------------------------------
# Sessions on toy E11 instances


def test_the_forbidden_strings_of_an_e11_session(e11_store: V.SyntheticStore) -> None:
    record = read_instance_record(e11_store.ids[("snow_s5", 100)])
    assert e11.is_e11(record.private) and record.private["condition"] == "snow_s5"
    tokens = e11.self_test_tokens(["s0123456789abcdef"])
    assert sorted(e11_store.ids.values()) == e11.instance_ids() and set(e11.instance_ids()) <= set(tokens)
    forbidden = runner.session_forbidden_strings(record, extra_tokens=tokens)
    assert set(e11.leak_words()) <= set(forbidden.text) and forbidden.library_text is not None
    for token in (*e11_store.ids.values(), "s0123456789abcdef", "gaussian_noise_s5", "defocus_blur", "contrast_s5"):
        assert token in forbidden.text and token in forbidden.selftest, token
    for word in ARCH_WORDS:
        assert word not in forbidden.text and not scan_for_tokens(word, forbidden.text), word


def test_the_lookups(e11_store: V.SyntheticStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                     capsys: pytest.CaptureFixture[str]) -> None:
    assert e11.condition_ids(100) == {c: e11_store.ids[(c, 100)] for c in e11.CONDITIONS}
    assert e11.find("contrast_s5", 103) == e11_store.ids[("contrast_s5", 103)]
    source = e11_store.ids[("defocus_blur_s5", 100)]
    assert e11.transfer_targets(source) == [e11_store.ids[("defocus_blur_s5", s)] for s in e11.TRANSFER_SEEDS]
    with pytest.raises(e11.E11HarnessError, match="C transfers seed 100"):
        e11.transfer_targets(e11_store.ids[("defocus_blur_s5", 101)])
    with pytest.raises(e11.E11HarnessError, match="not an E11 condition"):
        e11.find("fog_s5", 100)
    assert e11.main(["ids", "--condition", "snow_s5"]) == 0
    assert capsys.readouterr().out.strip() == e11_store.ids[("snow_s5", 100)]
    assert e11.main(["ids"]) == 0 and json.loads(capsys.readouterr().out) == e11.condition_ids(100)
    partial = build_e11_store(tmp_path / "partial", monkeypatch, instance_conditions=(PLANT_CONDITION,))
    found = e11.condition_ids(100)  # a condition that missed headroom has no instance: null, not an error
    assert found[PLANT_CONDITION] == partial.ids[(PLANT_CONDITION, 100)]
    assert [found[c] for c in e11.CONDITIONS if c != PLANT_CONDITION] == [None, None, None]
    with pytest.raises(e11.E11HarnessError, match="expected one active"):
        e11.find("snow_s5", 100)


@needs_sandbox
def test_an_e11_dry_run_states_the_floor_the_limits_and_the_clock(e11_store: V.SyntheticStore) -> None:
    instance_id = e11_store.ids[(PLANT_CONDITION, 100)]
    result = runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=5400.0, dry_run=True,
                                retention_floor_frac=0.005)
    assert result.status == STATUS_DRY_RUN and result.root.parent.name == runner.DRY_RUNS_DIR
    session = read_json(result.root / "session.json")
    assert (session["mcp"]["tool_timeout_ms"], session["mcp"]["connect_timeout_ms"]) == (1_200_000, 900_000)
    n_clean = int(session["clean_check"]["n"])
    assert session["kept_rule"]["s_old_floor"] == -round(0.005 * n_clean) and session["leak_scan"]["passed"]
    limits = session["limits"]
    assert (limits["max_ops"], limits["max_rel_frob"], limits["rel_frob_min_ndim"], limits["max_changed_frac"]) == (
        64, 0.5, 2, 0.2)  # 30 tensors in the toy: max(64, 30)
    assert set(e11.leak_words()) <= set(session["forbidden_strings"]["text"])
    assert set(e11_store.ids.values()) <= set(session["forbidden_strings"]["selftest"])
    task = " ".join((result.root / "prompts" / "task.md").read_text().split())
    assert "`wall_s_elapsed` and `wall_s_remaining`: the seconds since this session started" in task
    trial_task = read_json(result.root / "trial" / "task.json")
    assert trial_task["limits"] == {"max_rel_frob": 0.5, "rel_frob_min_ndim": 2, "max_changed_frac": 0.2}
    with pytest.raises(InstanceRefused, match="MCP timeouts of at least"):
        runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=5400.0, dry_run=True,
                           retention_floor_frac=0.005, mcp_tool_timeout_ms=600_000)


@needs_sandbox
def test_a_student_that_fails_the_check_is_refused_before_any_write(e11_store: V.SyntheticStore,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def failing(params: dict[str, np.ndarray]) -> dict[str, Any]:
        seen.append(len(params))
        raise e11.E11HarnessError("not an E11 student: stand-in failure")

    monkeypatch.setattr(e11, "check_student", failing)
    runs = e11_store.root / "runs"
    before = sorted(path.name for path in runs.iterdir())
    with pytest.raises(InstanceRefused, match="stand-in failure"):
        runner.run_session(e11_store.ids[(PLANT_CONDITION, 100)], "LEARN", 0, effort="max", wall_timeout_s=5400.0,
                           dry_run=True, retention_floor_frac=0.005)
    assert seen == [30] and sorted(path.name for path in runs.iterdir()) == before


@needs_sandbox
def test_a_scripted_e11_session_checks_layernorm_and_qkv_programs_and_reads_out(e11_store: V.SyntheticStore) -> None:
    instance_id = e11_store.ids[(PLANT_CONDITION, 100)]
    programs = {"ln.py": LN_PLANT, "qkv.py": QKV_Q, "qkv_big.py": QKV_BIG, "pos.py": POS_BIG, "cls.py": CLS_BIG,
                "wide.py": WIDE, "every.py": EVERY}
    order = ["ln.py", "qkv.py", "qkv_big.py", "pos.py", "cls.py", "wide.py", "every.py"]

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell(programs)})
        for name in order:
            call_and_record(client, replies, "patch_check", {"program_file": name})
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "ln.py", "rationale": "it scored"})

    launcher = ScriptedLauncher(script)
    result = runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                                skip_audit=True, retention_floor_frac=0.005)
    assert result.status == "complete", result.record.get("infra_reasons")
    replies = launcher.replies
    checks = dict(zip(order, replies[1:1 + len(order)]))
    assert [checks[name].is_error for name in order] == [False, False, True, True, True, True, False]
    for name, tensor in (("qkv_big.py", "classifier.blocks.0.self_attn.qkv.weight"),
                         ("pos.py", "classifier.positional_emb"), ("cls.py", "classifier.class_emb")):
        text = json.dumps(checks[name].payload)
        assert f"tensor '{tensor}': the relative Frobenius norm of the change is above the limit 0.5" in text, text
        assert "no query was used" in text  # the editor sees the limit, not the value (an editor rejection)
    wide_text = json.dumps(checks["wide.py"].payload)  # 160 LayerNorm entries + 1024 MLP entries > 0.2 of 5332
    assert "the fraction of parameters that change is above the limit 0.2" in wide_text, wide_text
    ln, qkv, every = (checks[name].payload for name in ("ln.py", "qkv.py", "every.py"))
    assert ln["tensors"] == ["classifier.norm.weight"] and qkv["tensors"] == [
        "classifier.blocks.0.self_attn.qkv.weight"] and len(every["tensors"]) == 30
    assert ln["score_new"] > 0 and ln["score_old"] >= 0  # the planted labels
    for reply in replies:
        assert {"wall_s_elapsed", "wall_s_remaining"} <= set(reply.payload), reply.name
    paths = SessionPaths(result.root)
    logged = [json.loads(line) for line in paths.checks_log.read_text().splitlines()]
    assert [entry["patch_id"] for entry in logged] == ["ln.py", "qkv.py", "every.py"]
    assert [entry["n_changed"] for entry in logged] == [4, 256, 30]
    session = read_json(result.root / "session.json")
    floor = int(session["kept_rule"]["s_old_floor"])
    eligible = [entry for entry in logged if int(entry["score_old"]) >= floor and int(entry["score"]) > 0]
    best = max(eligible, key=lambda entry: int(entry["score"]))
    final = read_json(paths.selection_json)["final"]
    assert final["kind"] == KIND_CHECKED and final["patch_id"] == best["patch_id"]
    readout = readout_run(result.root / "run.json", clean=True)
    assert readout["n"] == readout["n_clean"] == 160 and readout["condition"] == PLANT_CONDITION
    assert final["patch_id"] == "ln.py"  # the planted labels: a V_select gain and no clean row broken
    assert readout["readouts"]["delta_acc"] > 0 and readout["readouts_clean"]["delta_acc"] >= 0


@needs_sandbox
def test_the_c_transfer_runs_the_layernorm_program_on_the_other_students(e11_store: V.SyntheticStore,
                                                                         tmp_path: Path) -> None:
    source = e11_store.ids[(PLANT_CONDITION, 100)]
    program = tmp_path / "ln.py"
    program.write_text(LN_PLANT, encoding="utf-8")
    written = run_transfer(program, [source, *e11.transfer_targets(source)], tmp_path / "transfer", _sandbox())
    records = [json.loads(path.read_text()) for path in written]
    assert len(records) == 6
    for record in records:
        assert record["status"] == "ok" and record["within_limits"] is True
        assert record["tensors"] == ["classifier.norm.weight"]
        assert record["readouts"]["delta_acc"] is not None and record["readouts_clean"]["delta_acc"] is not None
    assert records[0]["readouts"]["delta_acc"] > 0 and records[0]["readouts_clean"]["delta_acc"] >= 0
