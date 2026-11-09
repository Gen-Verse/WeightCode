"""PREREG F2, the label-equivalent amount (``weightcode.baselines.fit.labeleq``): grid, sampler, training plumbing on a
synthetic ViT-Lite-7/4 student, the check generator and N*."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from test_baselines_fit_joint import CHILDREN, joint_instance  # noqa: F401
from test_e11_refs import MEAN, STD, _images, _student
from test_evalkit_instances_store import toy_store  # noqa: F401

from weightcode.baselines.fit import labeleq as F
from weightcode.common.identity import params_sha256, save_params


def test_grid_is_prereg_f2() -> None:
    assert F.F2_NS == (64, 256, 1024, 4096, 16384, 40000) and F.F2_N_SPEC_TOP == 50000
    assert F.f2_configs() == [{"lr": lr, "steps": s} for lr in (1e-5, 3e-5, 1e-4, 3e-4) for s in (100, 1000)]
    assert F.ceiling_configs() == [{"lr": lr, "steps": 5000} for lr in (1e-5, 3e-5, 1e-4, 3e-4)]
    assert F.groups() == ["n64", "n256", "n1024", "n4096", "n16384", "n40000", "ceiling"]
    trajectories = F.f2_trajectories()
    assert len(trajectories) == 48 and len({t.name for t in trajectories}) == 48
    assert [t.steps for t in trajectories[:8]] == [5000] * 8 and all(t.n == 40000 for t in trajectories[:8])
    steps = {(t.condition, t.n, t.lr, s) for t in trajectories for s in t.checkpoints}
    for condition in F.F2_CONDITIONS:
        for group in F.groups():
            n, configs = F.group_rows(group)
            assert all((condition, n, c["lr"], c["steps"]) in steps for c in configs)
    assert sum(t.steps for t in trajectories) == 2 * (20 * 1000 + 4 * 5000)
    assert not F._spec().edit_limits_enforced
    with pytest.raises(F.F2Error):
        F.group_rows("n50000")
    with pytest.raises(F.F2Error):
        F.Trajectory("gaussian_noise", 100, 1e-4, (10,))


def test_balanced_sampler_halves_and_passes() -> None:
    sampler = F.BalancedBatchSampler(128, 64, 64, 6, torch.Generator().manual_seed(0))
    batches = list(sampler)
    assert len(batches) == 6 and all(len(b) == 128 for b in batches)
    for b in batches:
        assert all(0 <= i < 128 for i in b[:64]) and all(128 <= i < 192 for i in b[64:])
    for k in range(0, 6, 2):  # set a: one pass per two steps
        assert sorted(batches[k][:64] + batches[k + 1][:64]) == list(range(128))
    for b in batches:  # set b: one pass per step
        assert sorted(b[64:]) == list(range(128, 192))
    again = list(F.BalancedBatchSampler(128, 64, 64, 6, torch.Generator().manual_seed(0)))
    assert again == batches
    with pytest.raises(F.F2Error):
        F.BalancedBatchSampler(100, 64, 64, 1, torch.Generator())


def _data(params: dict[str, np.ndarray], n: int = 128) -> F.F2Data:
    x = _images(n, 1)
    noisy = np.clip(x.astype(np.int16) + np.random.default_rng(2).integers(-40, 40, x.shape), 0, 255).astype(np.uint8)
    rng = np.random.default_rng(3)
    return F.F2Data(x_clean=x, y=rng.integers(0, 10, n).astype(np.int64),
                    x_corrupt={"gaussian_noise": noisy, "contrast": (x // 3).astype(np.uint8)},
                    perm_corrupt=rng.permutation(n).astype(np.int64), perm_replay=rng.permutation(n).astype(np.int64),
                    mean=MEAN, std=STD, data_json_sha256="0" * 64, base_params=params)


def test_trajectory_rows_mix_corrupt_then_clean() -> None:
    params = _student(0)
    data = _data(params)
    x, y, info = F.trajectory_rows(F.Trajectory("contrast", 64, 1e-4, (1,)), data)
    assert x.shape == (128, 3, 32, 32) and y.shape == (128,)
    np.testing.assert_array_equal(x[:64], data.x_corrupt["contrast"][data.perm_corrupt[:64]])
    np.testing.assert_array_equal(x[64:], data.x_clean[data.perm_replay[:64]])
    np.testing.assert_array_equal(y, np.concatenate([data.y[data.perm_corrupt[:64]], data.y[data.perm_replay[:64]]]))
    assert info["n_corrupt"] == info["n_clean"] == 64


def test_training_is_deterministic_and_checkpoints_equal_short_runs(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from weightcode.models.train import configure_determinism

    params = _student(0)
    monkeypatch.setattr(F, "F2_STUDENT_SHA256", params_sha256(params))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    data = _data(params)
    out = {}
    for tag, checkpoints in (("", (1, 2)), ("dup", (1, 2)), ("short", (1,))):
        configure_determinism(0)
        t = F.Trajectory("gaussian_noise", 64, 1e-4, checkpoints, tag=tag)
        rows = F.train_trajectory(t, data, device=torch.device("cpu"), seed=0, num_workers=0,
                                  out_root=tmp_path / "models", provenance={"test": True})
        out[tag] = {r["steps"]: r["params_sha256"] for r in rows}
        assert [r["steps"] for r in rows] == list(checkpoints)
    assert out[""] == out["dup"] and out[""][1] == out["short"][1]
    assert out[""][1] != out[""][2] != params_sha256(params)
    record = json.loads((tmp_path / "models" / "gaussian_noise/n64/lr0.0001/s2/record.json").read_text())
    assert record["config"] == {"n": 64, "lr": 1e-4, "steps": 2} and record["optimizer"]["weight_decay"] == 3e-2
    assert len(record["device_check"]["logits"]) == F.F2_CHECK_ROWS or len(data.x_clean) < F.F2_CHECK_ROWS


def _import_record(tmp_path: Path, params: dict[str, np.ndarray], condition: str, n: int) -> dict:
    rows = []
    for k, config in enumerate(F.f2_configs()):
        child = {key: (value + np.float32(1e-3 * (k + 1))).astype(np.float32) for key, value in params.items()}
        path = f"{condition}/n{n}/lr{config['lr']:g}/s{config['steps']}"
        (tmp_path / path).mkdir(parents=True)
        digest = save_params(tmp_path / path / "model.npz", child)
        rows.append({"condition": condition, "n": n, **config, "tag": "", "path": path, "params_sha256": digest,
                     "train_loss_mean_last100": 1.0, "device_check": {"max_abs_diff": 0.0}})
    return {"models_root": str(tmp_path), "rows": rows, "run_dir": "r", "index_sha256": "1" * 64,
            "data_json_sha256": "2" * 64}


def test_generator_loads_the_checkpoints_in_check_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    params = _student(1)
    monkeypatch.setattr(F, "F2_STUDENT_SHA256", params_sha256(params))
    record = _import_record(tmp_path, params, "gaussian_noise", 256)
    instance = SimpleNamespace(instance_id="i00000000", params=params, params_sha256=params_sha256(params),
                               private={"dataset": "cifar10_vl7", "condition": "gaussian_noise_s5"})
    candidates, info = F.f2_generator(instance, record, "n256")(None)
    assert [c.config for c in candidates] == [{"n": 256, **c} for c in F.f2_configs()]
    assert [c.slot for c in candidates] == list(range(8))
    assert [params_sha256(c.child) for c in candidates] == [r["params_sha256"] for r in record["rows"]]
    assert info["n"] == 256 and "not used" in info["fit_rows"]
    with pytest.raises(F.F2Error, match="0 imported checkpoints"):
        F.f2_generator(instance, record, "n64")(None)
    record["rows"][3]["params_sha256"] = "f" * 64
    with pytest.raises(F.F2Error, match="params hash"):
        F.f2_generator(instance, record, "n256")(None)
    wrong = SimpleNamespace(**{**vars(instance), "private": {"dataset": "cifar10_vl7", "condition": "snow_s5"}})
    with pytest.raises(F.F2Error):
        F.f2_generator(wrong, record, "n256")


def test_n_star_interpolates_log_linearly() -> None:
    pts = [{"n": 64, "joint": 5.0, "clean": -0.1}, {"n": 256, "joint": 20.0, "clean": -0.2},
           {"n": 1024, "joint": 40.0, "clean": -0.3}]
    result = F.n_star(pts, 30.0)
    assert result["bound"] == "interpolated" and result["between"] == [256, 1024]
    assert result["n_star"] == pytest.approx(512.0)
    assert F.n_star(pts, 3.0)["bound"] == "at_most" and F.n_star(pts, 3.0)["n_star"] == 64
    assert F.n_star(pts, 50.0)["reached"] is False and F.n_star(pts, 50.0)["text"] == "not reached within 1024"
    clean_miss = [{"n": 64, "joint": 35.0, "clean": -1.5}, {"n": 256, "joint": 36.0, "clean": -0.5}]
    assert F.n_star(clean_miss, 30.0)["bound"] == "grid_point" and F.n_star(clean_miss, 30.0)["n_star"] == 256
    edge = [{"n": 64, "joint": 30.0, "clean": -1.0}]
    assert F.n_star(edge, 30.0)["reached"] is True


def test_f2_run_checks_every_checkpoint_and_reads_out_like_the_references(
        joint_instance: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """End to end on the toy joint store: the 8 imported children of one group are checked by the joint oracle in
    configuration order under the E10 limits' executor without size limits, and the project's readouts (the kept
    pick under joint-retain-v1 and the frontier) read them."""
    from weightcode.evalkit import devreadout

    params = dict(joint_instance.params)
    makers = [*CHILDREN] * 4
    rows = []
    for k, (config, make) in enumerate(zip(F.f2_configs(), makers)):
        child = {key: (value + np.asarray(1e-4 * k, dtype=value.dtype)).astype(value.dtype)
                 for key, value in make(dict(params)).items()}
        path = f"gaussian_noise/n64/lr{config['lr']:g}/s{config['steps']}"
        (tmp_path / "models" / path).mkdir(parents=True)
        digest = save_params(tmp_path / "models" / path / "model.npz", child)
        rows.append({"condition": "gaussian_noise", "n": 64, **config, "tag": "", "path": path,
                     "params_sha256": digest, "train_loss_mean_last100": 1.0, "device_check": {"max_abs_diff": 0.0}})
    record = {"models_root": str(tmp_path / "models"), "rows": rows, "run_dir": "r", "index_sha256": "1" * 64,
              "data_json_sha256": "2" * 64}
    monkeypatch.setattr(F, "condition_of", lambda instance: "gaussian_noise")
    run = F.run_f2(joint_instance, record, "n64", tmp_path / "run")
    assert run["status"] == "complete" and run["queries_used"] == 8 and run["flags"]["edit_limits_enforced"] is False
    assert [c["config"] for c in run["candidates"]] == [{"n": 64, **c} for c in F.f2_configs()]
    assert [e["child_sha256"] for e in run["oracle_log"]] == [r["params_sha256"] for r in rows]
    assert run["generation"]["n"] == 64 and "not used" in run["generation"]["fit_rows"]
    run_path = tmp_path / "run" / "run.json"
    kept = devreadout.readout_retention(run_path, s_old_floor=-20)
    frontier = devreadout.readout_candidates(run_path)
    assert [r["config"] for r in frontier] == [{"n": 64, **c} for c in F.f2_configs()]
    if kept["selection"]["patch_id"] is not None:
        (same,) = [r for r in frontier if r["patch_id"] == kept["selection"]["patch_id"]]
        assert same["readouts"] == kept["readouts"] and same["readouts_clean"] == kept["readouts_clean"]
