"""PREREG F1s (``weightcode.baselines.fit.f1s``): configurations, augmentation, the label-free corruption estimate."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_e11_refs import _instance, _student
from test_ft_compare import _joint_evidence

from weightcode.baselines.fit import f1s as S

SHAPE = (3, 32, 32)


def _images(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.15, 0.85, size=(n, 3, 1, 1))
    tex = rng.normal(0, 0.12, size=(n, 3, 32, 32))
    return (np.clip(base + tex, 0, 1) * 255).astype(np.uint8).reshape(n, -1)


def test_grids_are_the_prereg_grids() -> None:
    assert [(c["lr"], c["steps"]) for c in S.f1s_configs("f1s_s1_gentle")] == [
        (lr, s) for lr in (3e-6, 1e-5) for s in (30, 100, 300, 1000)]
    assert [(c["lr"], c["steps"]) for c in S.f1s_configs("f1s_s2_aug")] == [
        (lr, s) for lr in (1e-5, 3e-5) for s in (100, 300, 1000, 3000)]
    assert [(c["lr"], c["steps"]) for c in S.f1s_configs("f1s_s3_lora_aug")] == [
        (lr, s) for lr in (3e-4, 1e-3) for s in (30, 100, 300, 1000)]
    for m in ("f1s_s4_est_prior", "f1s_s5_oracle_prior"):
        cfg = S.f1s_configs(m)
        assert [c["mode"] for c in cfg] == ["full"] * 4 + ["lora"] * 4
        assert [(c["mode"], c["lr"]) for c in cfg[::4]] == [("full", 1e-5), ("lora", 1e-3)]
    assert not any(spec.edit_limits_enforced for spec in S.F1S_SPECS.values())


def test_crop_flip_is_deterministic_and_shape_preserving() -> None:
    x = torch.from_numpy(_images(6, 0))
    a = S.crop_flip(x, SHAPE, torch.Generator().manual_seed(3))
    b = S.crop_flip(x, SHAPE, torch.Generator().manual_seed(3))
    c = S.crop_flip(x, SHAPE, torch.Generator().manual_seed(4))
    assert a.shape == x.shape and a.dtype == torch.uint8 and torch.equal(a, b) and not torch.equal(a, c)


def test_estimate_picks_the_true_family_label_free() -> None:
    x_old = _images(64, 1)
    old = x_old.reshape(-1, *SHAPE).astype(np.float64) / 255
    rng = np.random.default_rng(9)
    noisy = np.uint8(np.clip(old + rng.normal(0, 0.1, old.shape), 0, 1) * 255).reshape(64, -1)
    means = old.mean(axis=(2, 3), keepdims=True)
    flat = np.uint8(np.clip((old - means) * 0.4 + means, 0, 1) * 255).reshape(64, -1)
    x_other = _images(64, 2)  # different images: the estimate uses statistics, not pairing
    _, rec_n = S.estimate_corruption(noisy, x_other, SHAPE, 0)
    _, rec_c = S.estimate_corruption(flat, x_other, SHAPE, 0)
    assert rec_n["chosen"] == "gaussian_noise" and rec_c["chosen"] == "contrast"
    assert abs(rec_n["families"]["gaussian_noise"]["value"] - 0.1) < 0.03
    assert abs(rec_c["families"]["contrast"]["value"] - 0.4) < 0.1
    copies, _ = S.estimate_corruption(noisy, x_other, SHAPE, 0)
    again, _ = S.estimate_corruption(noisy, x_other, SHAPE, 0)
    assert copies.dtype == np.uint8 and copies.shape == x_other.shape and np.array_equal(copies, again)


def test_estimate_has_no_label_input() -> None:
    import inspect
    assert list(inspect.signature(S.estimate_corruption).parameters) == ["x_new", "x_old", "shape", "seed"]


@pytest.mark.parametrize("method", ["f1s_s1_gentle", "f1s_s2_aug", "f1s_s3_lora_aug", "f1s_s4_est_prior"])
def test_generator_candidates(tmp_path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    short = (1, 2)
    monkeypatch.setattr(S, "GRIDS", {k: {**v, "steps": short} for k, v in S.GRIDS.items()})
    params = _student(7)
    instance = _instance(tmp_path, params)
    candidates, info = S.f1s_generator(instance, method, 0)(_joint_evidence(7))
    assert len(candidates) == 2 * len(S.GRIDS[method]["parts"])
    assert all(np.all(np.isfinite(v)) for c in candidates for v in c.child.values())
    assert any(not np.array_equal(c.child[k], params[k]) for c in candidates for k in params)
    if method == "f1s_s3_lora_aug":
        assert all(np.array_equal(c.child[k], params[k]) for c in candidates for k in params
                   if "linear" not in k and "qkv" not in k and "proj.weight" not in k)
    if method == "f1s_s4_est_prior":
        assert info["estimate"]["chosen"] in S.FAMILIES and "192 rows" in info["fit_rows"]


def test_warmup_scales_first_steps(tmp_path) -> None:
    from weightcode.baselines.fit.torch_forward import architecture_from, images_tensor
    from weightcode.baselines.fit.runner import read_model_spec
    params = _student(7)
    instance = _instance(tmp_path, params)
    arch = architecture_from(params, read_model_spec(instance))
    ev = _joint_evidence(7)
    x, y = images_tensor(ev.x), torch.from_numpy(ev.y)
    full = S.train_checkpoints(arch, params, x, y, mode="full", lr=1e-4, checkpoints=(1,), seed=0)[0][1]
    warm = S.train_checkpoints(arch, params, x, y, mode="full", lr=1e-4, checkpoints=(1,), seed=0, warmup=10)[0][1]
    k = "classifier.fc.weight"
    d_full, d_warm = np.abs(full[k] - params[k]).max(), np.abs(warm[k] - params[k]).max()
    assert d_warm == pytest.approx(d_full / 10, rel=1e-3)


# ------------------------------------------------------------------------------------------------ K1 control


def _k1_record(tmp_path, evidence, seed_offsets=None, corrupt_hash: bool = False):
    import hashlib
    import json

    k = evidence.n_deployment
    n_old = len(evidence.y) - k
    rng = np.random.default_rng(11)
    draws = rng.integers(0, 256, (S.K1_DRAWS, n_old, *evidence.x.shape[1:]), dtype=np.uint8)
    file = tmp_path / "gaussian_noise_s5_rule_draws.npz"
    np.savez(file, x=draws)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    seeds = [S.K1_SEED0 + i for i in range(S.K1_DRAWS)] if seed_offsets is None else seed_offsets
    record = {"schema": S.K1_ROWS_SCHEMA, "severity": 5, "official": {"sha256": "f" * 64},
              "replay": {"content_sha256": evidence.replay_sha256},
              "rows": {"gaussian_noise": {"file": file.name, "file_sha256": "0" * 64 if corrupt_hash else digest,
                                          "seeds": seeds}}}
    path = tmp_path / "k1_rule_rows.json"
    path.write_text(json.dumps(record))
    return path, draws


def _k1_instance(tmp_path, params):
    inst = _instance(tmp_path, params)
    inst.private = {**inst.private, "condition": "gaussian_noise_s5"}
    return inst


def test_k1_registered_and_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    assert S.K1_METHOD in S.F1S_SPECS and S.K1_METHOD in S.F1S_METHODS
    cfg = S.f1s_configs("k1_ft_rule")
    assert [(c["mode"], c["lr"], c["steps"]) for c in cfg] == [
        (m, lr, s) for m, lr in (("full", 1e-5), ("lora", 1e-3)) for s in (30, 100, 300, 1000)]
    with pytest.raises(S.F1sError, match="rows record"):
        S.f1s_generator(None, "k1_ft_rule", 0, None)  # type: ignore[arg-type]


def test_k1_rows_loader_checks(tmp_path) -> None:
    ev = _joint_evidence(2)
    path, draws = _k1_record(tmp_path, ev)
    x, info = S.load_k1_rows(path, "gaussian_noise_s5", ev.replay_sha256, 64)
    assert np.array_equal(x, draws) and info["seeds"] == [1005 + i for i in range(8)]
    with pytest.raises(S.F1sError, match="replay content"):
        S.load_k1_rows(path, "gaussian_noise_s5", "9" * 64, 64)
    bad, _ = _k1_record(tmp_path / ".", ev, corrupt_hash=True)
    with pytest.raises(S.F1sError, match="differs from the record"):
        S.load_k1_rows(bad, "gaussian_noise_s5", ev.replay_sha256, 64)
    bad2, _ = _k1_record(tmp_path / ".", ev, seed_offsets=[1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(S.F1sError, match="seeds"):
        S.load_k1_rows(bad2, "gaussian_noise_s5", ev.replay_sha256, 64)


def test_k1_fit_rows_and_no_x_new_row(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S, "GRIDS", {k: {**v, "steps": (1, 2)} for k, v in S.GRIDS.items()})
    monkeypatch.setattr(S, "K1_WORKERS", 1)
    torch.set_num_threads(2)
    params = _student(7)
    inst = _k1_instance(tmp_path, params)
    ev = _joint_evidence(7)
    path, draws = _k1_record(tmp_path, ev)
    c1, info1 = S.f1s_generator(inst, "k1_ft_rule", 0, path)(ev)
    assert info1["x_new_rows_used"] == 0 and "576 rows" in info1["fit_rows"] and "no condition row" in info1["fit_rows"]
    assert [c.config for c in c1] == S.f1s_configs("k1_ft_rule")
    # the same replay rows with different condition rows (x_new and its labels) give bit-identical candidates
    other = _joint_evidence(8)
    k = ev.n_deployment
    ev2 = type(ev)(x=np.concatenate([other.x[:k], ev.x[k:]]), y=np.concatenate([other.y[:k], ev.y[k:]]), n_fail=ev.n_fail,
                   n_rand=ev.n_rand, sha256="2" * 64, base_pred=ev.base_pred, rule=ev.rule, n_replay=ev.n_replay,
                   replay_sha256=ev.replay_sha256)
    c2, info2 = S.f1s_generator(inst, "k1_ft_rule", 0, path)(ev2)
    assert info1["rows_sha256"] == info2["rows_sha256"]
    for a, b in zip(c1, c2, strict=True):
        assert all(np.array_equal(a.child[key], b.child[key]) for key in params)
    assert any(not np.array_equal(c.child[key], params[key]) for c in c1 for key in params)


def test_k1_pool_equals_serial(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S, "GRIDS", {k: {**v, "steps": (1, 2)} for k, v in S.GRIDS.items()})
    torch.set_num_threads(1)
    params = _student(5)
    inst = _k1_instance(tmp_path, params)
    ev = _joint_evidence(5)
    path, _ = _k1_record(tmp_path, ev)
    monkeypatch.setattr(S, "K1_WORKERS", 1)
    a, _ = S.f1s_generator(inst, "k1_ft_rule", 0, path)(ev)
    monkeypatch.setattr(S, "K1_WORKERS", 2)
    b, _ = S.f1s_generator(inst, "k1_ft_rule", 0, path)(ev)
    for x, y in zip(a, b, strict=True):
        assert x.config == y.config and all(np.array_equal(x.child[key], y.child[key]) for key in params)
