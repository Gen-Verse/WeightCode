"""E11 references on the ViT-Lite-7/4 student: the torch rebuild, input_stats' exact patch-projection fold, the tie
plan, tent_ln and sar, and the CLI's E11 dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from weightcode.baselines.fit import cli as C
from weightcode.baselines.fit import lnrefs as L
from weightcode.baselines.fit import rules as RU
from weightcode.baselines.fit import torch_forward as TF
from weightcode.baselines.fit.runner import Evidence
from weightcode.models import vitlite as V
from weightcode.models.forward_gen import render_vitlite_forward

MEAN = [0.4909, 0.4818, 0.4460]
STD = [0.2471, 0.2435, 0.2616]
CLASSES = [f"c{i}" for i in range(10)]


def _student(seed: int = 0) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    model = V.ViTLite(V.ViTLiteConfig(), MEAN, STD)
    with torch.no_grad():  # away from the near-uniform init, so that predictions depend on the input
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return V.export_params(model)


def _forward():
    namespace: dict = {}
    exec(compile(render_vitlite_forward(MEAN, STD, 10, V.LN_EPS), "forward.py", "exec"), namespace)
    return namespace["forward"]


def _images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, 3, 32, 32), dtype=np.uint8)


def _spec(params: dict[str, np.ndarray]) -> dict:
    return V.build_vitlite_spec("m00000000", CLASSES, MEAN, STD, params)


def _arch(params: dict[str, np.ndarray]) -> TF.Architecture:
    return TF.architecture_from(params, _spec(params))


def test_rebuild_matches_forward_and_trainable_set() -> None:
    params = _student()
    arch = _arch(params)
    assert arch.family == TF.VITLITE and arch.last_linear_tensors == ("classifier.fc.weight", "classifier.fc.bias")
    assert arch.conv_layers == ("tokenizer.conv_layers.0.0",) and len(arch.ln_layers) == 15
    x = _images(12)
    assert TF.assert_parity(arch, params, _forward(), x, dtype=torch.float64) < 1e-5
    assert TF.assert_parity(arch, params, _forward(), x) < 1e-4
    assert TF.trainable_tensors(arch, params) == sorted(params)
    feats = TF.penultimate(arch, TF.to_tensors(params, dtype=torch.float64), torch.from_numpy(x))
    model = V.build_vitlite(V.ViTLiteConfig(), MEAN, STD, params, torch.float64)
    with torch.no_grad():
        assert torch.allclose(feats, model.features(torch.from_numpy(x)), rtol=0, atol=1e-10)


def test_input_stats_fold_is_exact_on_the_patch_projection() -> None:
    params = _student(1)
    arch = _arch(params)
    x = _images(48, 1)
    mean, std = RU.input_channel_stats(arch, x)
    for strength in RU.INPUT_STATS_STRENGTHS:
        fold = RU.input_stats_fold(arch, params, mean, std, strength)
        changed = {k for k in params if not np.array_equal(fold.child[k], params[k])}
        assert changed == {"tokenizer.conv_layers.0.0.weight", "tokenizer.conv_layers.0.0.bias"}
        k = fold.k
        assert np.allclose(k, 1.0 + strength * (1.0 / std - 1.0))
        t64 = TF.to_tensors(params, dtype=torch.float64)
        c64 = TF.to_tensors(fold.child, dtype=torch.float64)
        pre = TF._preprocessed(arch, torch.from_numpy(x), torch.float64)
        target_in = torch.tensor(k).view(1, 3, 1, 1) * (pre - strength * torch.tensor(mean).view(1, 3, 1, 1))
        name = "tokenizer.conv_layers.0.0"
        target = F.conv2d(target_in, t64[f"{name}.weight"], t64[f"{name}.bias"], stride=4)
        ours = F.conv2d(pre, c64[f"{name}.weight"], c64[f"{name}.bias"], stride=4)
        assert torch.max(torch.abs(target - ours)) < 1e-5  # float32 storage of the folded tensors
        notes = fold.notes()
        assert notes["patch_fold"]["admissible"] == (notes["rel_frob_first_weight_k_form"] <= 0.5)
        assert [op["op"] for op in fold.ops] == ["scale_cols"] * 3 + ["set"]


def test_tie_plan_groups_the_patch_projection_by_patch_position() -> None:
    params = _student(2)
    plan = RU.tie_plan(_arch(params), params, _spec(params))
    assert plan.tensor == "tokenizer.conv_layers.0.0.weight" and plan.n_channels == 3
    assert plan.groups.tolist() == [j % 16 for j in range(48)]
    child = RU.tie_columns(params[plan.tensor], plan.groups, 1.0)
    assert np.allclose(child, np.broadcast_to(params[plan.tensor].mean(axis=1, keepdims=True), child.shape), atol=1e-7)


def _instance(tmp_path: Path, params: dict[str, np.ndarray]) -> SimpleNamespace:
    (tmp_path / "model_spec.json").write_text(json.dumps(_spec(params)))
    return SimpleNamespace(instance_dir=tmp_path, params=params, forward=_forward(),
                           private={"dataset": C.E11_DATASET})


def _evidence(seed: int) -> Evidence:
    x = _images(64, seed)
    return Evidence(x=x, y=np.zeros(64, dtype=np.int64), n_fail=16, n_rand=48, sha256="0" * 64,
                    base_pred=np.zeros(64, dtype=np.int64), rule="evidence-v1")


@pytest.mark.parametrize("method", ["tent_ln", "sar"])
def test_ln_references_give_eight_layernorm_only_candidates(tmp_path: Path, method: str) -> None:
    params = _student(3)
    instance = _instance(tmp_path, params)
    generator = L.tent_ln_generator(instance, 0) if method == "tent_ln" else L.sar_generator(instance, 0)
    candidates, info = generator(_evidence(3))
    assert [c.config for c in candidates] == [{"lr": lr, "steps": s} for lr in L.LN_LRS for s in L.LN_STEPS]
    affine = set(L.ln_affine(_arch(params)))
    assert len(affine) == 30 and sum(params[k].size for k in affine) == 7680
    for candidate in candidates:
        changed = {k for k in params if not np.array_equal(candidate.child[k], params[k])}
        assert changed <= affine
        assert all(candidate.child[k].dtype == np.float32 for k in candidate.child)
    big = [c for c in candidates if c.config == {"lr": 1e-3, "steps": 30}][0]
    if method == "tent_ln":
        assert big.notes["mean_entropy_on_inputs"] < info["base_mean_entropy_on_inputs"]
    else:
        assert info["margin"] == pytest.approx(0.4 * np.log(10))
        assert len(info["trajectories"]["0.001"]) == 30
        if info["base_n_below_margin"] == 0:
            assert all(np.array_equal(c.child[k], params[k]) for c in candidates for k in params)


def test_sar_updates_when_inputs_pass_the_margin(tmp_path: Path) -> None:
    params = _student(4)
    params["classifier.fc.bias"] = params["classifier.fc.bias"] + np.float32(5.0) * (np.arange(10) == 0)  # confident
    params["classifier.fc.bias"] = params["classifier.fc.bias"].astype(np.float32)
    instance = _instance(tmp_path, params)
    candidates, info = L.sar_generator(instance, 0)(_evidence(4))
    assert info["base_n_below_margin"] > 0
    big = [c for c in candidates if c.config == {"lr": 1e-3, "steps": 30}][0]
    assert big.notes["updates_so_far"] > 0
    assert any(not np.array_equal(big.child[k], params[k]) for k in L.ln_affine(_arch(params)))


def test_cli_dispatches_e11_and_refuses_mixing() -> None:
    assert C.E11_METHODS[-2:] == ("tent_ln", "sar") and "bn_adapt" not in C.E11_METHODS
    assert set(C.E11_METHODS) <= set(C.FIT_METHODS)
    vit = SimpleNamespace(private={"dataset": C.E11_DATASET})
    other = SimpleNamespace(private={"dataset": "cifar10_r20"})
    with pytest.raises(ValueError, match="runs with e11"):
        C.run_method(vit, "grad_last", Path("/nonexistent"), run_seed=0, allow_main=False, joint=True, e10=True)
    with pytest.raises(ValueError, match="need a cifar10_vl7"):
        C.run_method(other, "grad_last", Path("/nonexistent"), run_seed=0, allow_main=False, joint=True, e10=True,
                     e11=True)
    with pytest.raises(ValueError, match="no BatchNorm"):
        C.run_method(vit, "tent", Path("/nonexistent"), run_seed=0, allow_main=False, joint=True, e10=True, e11=True)
    with pytest.raises(ValueError, match="e11 only"):
        C.run_method(other, "sar", Path("/nonexistent"), run_seed=0, allow_main=False, joint=True, e10=True)


def test_second_grid_values_labels_and_run_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from weightcode.baselines.fit import grad as G
    from weightcode.baselines.fit.runner import baseline_run_dir

    assert G.GRID_E1.learning_rates == (1e-3, 1e-2) and G.GRID_E1.interp_alphas == tuple(k / 8 for k in range(1, 9))
    assert G.GRID_E11_SECOND.learning_rates == (1e-5, 1e-4)
    assert G.GRID_E11_SECOND.interp_alphas == tuple(k / 64 for k in range(1, 9))
    assert G.candidate_grid(G.GRID_E11_SECOND) == [(lr, s) for lr in (1e-5, 1e-4) for s in (10, 30, 100, 300)]
    assert G.candidate_grid() == [(lr, s) for lr in (1e-3, 1e-2) for s in (10, 30, 100, 300)]
    for method in G.GRID2_METHODS:
        assert G.grid_spec(method) is G.GRAD_SPECS[method]
        second = G.grid_spec(method, G.GRID_E11_SECOND)
        assert second.label != G.GRAD_SPECS[method].label and "21e3ff2" in second.label
        assert second.edit_limits_enforced == G.GRAD_SPECS[method].edit_limits_enforced
    assert "alpha in {1/64 .. 8/64}" in G.grid_spec("interp", G.GRID_E11_SECOND).label
    with pytest.raises(ValueError, match="no e11_second grid"):
        G.grid_spec("grad_sparse", G.GRID_E11_SECOND)
    config = G.grad_config("interp", 0, joint=True, grid=G.GRID_E11_SECOND)
    assert config["interp_alphas"] == [k / 64 for k in range(1, 9)] and config["grid"]["name"] == "e11_second"
    assert "grid" not in G.grad_config("interp", 0, joint=True)
    assert C.E11_GRID2_METHODS == ("grad_last", "grad_full", "grad_sparse19_sal", "grad_sparse19_traj", "interp")
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(tmp_path))
    first = baseline_run_dir("i00802425", "grad_full", 0, C.E11_REF_LABEL)
    second = baseline_run_dir("i00802425", "grad_full", 0, C.E11_GRID2_LABEL)
    assert first != second and second.parts[-2] == "e11g2" and first.parts[-2] == "e11"


def test_second_grid_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from weightcode.baselines.fit import grad as G

    params = _student(5)
    instance = _instance(tmp_path, params)
    evidence = _evidence(5)
    candidates, info = G.grad_generator(instance, "grad_last", 0, grid=G.GRID_E11_SECOND)(evidence)
    assert [c.config for c in candidates] == [{"lr": lr, "steps": s} for lr in (1e-5, 1e-4) for s in (10, 30, 100, 300)]
    assert info["grid"] == "e11_second"
    head = set(_arch(params).last_linear_tensors)
    for c in candidates:
        assert {k for k in params if not np.array_equal(c.child[k], params[k])} <= head
    monkeypatch.setattr(G, "TRAJ_STEPS", 3)  # a short theta_full: the same target in both grids is what is tested
    first, info1 = G.grad_generator(instance, "interp", 0)(evidence)
    second, info2 = G.grad_generator(instance, "interp", 0, grid=G.GRID_E11_SECOND)(evidence)
    assert info1["theta_full"]["sha256"] == info2["theta_full"]["sha256"]
    assert [c.config["alpha"] for c in second] == [k / 64 for k in range(1, 9)]
    assert "grid" not in info1
    by_alpha = {c.config["alpha"]: c.child for c in first}
    eighth = [c for c in second if c.config["alpha"] == 8 / 64][0]
    assert all(np.array_equal(eighth.child[k], by_alpha[1 / 8][k]) for k in params)  # 8/64 = 1/8 in both grids


def test_cli_refuses_misused_second_grid() -> None:
    vit = SimpleNamespace(private={"dataset": C.E11_DATASET})
    with pytest.raises(ValueError, match="pass e11=True with grid2"):
        C.run_method(vit, "grad_last", Path("/nonexistent"), run_seed=0, allow_main=False, joint=True, e10=True,
                     grid2=True)
    for method in ("ridge_last", "tent_ln", "sar", "prior_shift"):
        with pytest.raises(ValueError, match="no E11 second-grid run"):
            C.run_method(vit, method, Path("/nonexistent"), run_seed=0, allow_main=False, joint=True, e10=True,
                         e11=True, grid2=True)
