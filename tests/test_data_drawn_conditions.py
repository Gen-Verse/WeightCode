"""Drawn-parameter CIFAR-10 conditions of contract section 2.1 (colormix_d1, blur5_d1, gamma_d1).

Checks the transforms (convex colour mixing, unit-sum blur kernel, monotone gamma, identity at s = 0), the parameter
streams, the severity rule, the train-only guard of the calibration, and the calibrate, build, write and verify cycle
on a tiny synthetic store with two tiny dev models. Nothing here reads the real stores.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType

import numpy as np
import pytest

from weightcode.data import colored
from weightcode.data import conditions as C
from weightcode.data import store
from weightcode.data.registry import CIFAR10_CLASSES, SPLIT_NAMES, SPLIT_SEED
from weightcode.models import calibration as K
from weightcode.models.forward_gen import render_forward, write_forward

REPO = Path(__file__).resolve().parents[1]
DRAWN = ("colormix_d1", "blur5_d1", "gamma_d1")


def _images(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)


def _extremes() -> np.ndarray:
    x = _images(6, 3)
    x[0], x[1] = 0, 255
    x[2, 0], x[2, 1], x[2, 2] = 255, 0, 255
    return x


# ---------------------------------------------------------------------------------------------------------------
# Transforms


def test_registry_and_grids() -> None:
    assert set(DRAWN) <= set(C.CONDITIONS) and set(C.CORRUPTION_CONDITIONS) <= set(C.CONDITIONS)
    assert C.SEVERITY_GRIDS["colormix"] == C.SEVERITY_GRIDS["blur5"]
    assert len(C.SEVERITY_GRIDS["colormix"]) == 20 and C.SEVERITY_GRIDS["colormix"][0] == 0.05
    assert C.SEVERITY_GRIDS["colormix"][-1] == 1.0 and C.SEVERITY_GRIDS["colormix"][2] == 0.15
    assert len(C.SEVERITY_GRIDS["gamma"]) == 30 and C.SEVERITY_GRIDS["gamma"][0] == 0.1
    assert C.SEVERITY_GRIDS["gamma"][-1] == 3.0 and C.SEVERITY_GRIDS["gamma"][2] == 0.3
    for name in DRAWN:
        cond = C.get_condition(name)
        assert isinstance(cond, C.DrawnCondition) and cond.dataset == "cifar10"
        assert cond.severity is None or cond.severity in cond.grid
    with pytest.raises(ValueError, match="not on the"):
        C.DrawnCondition("x_d1", "cifar10", "gamma", 999, 0.15)


def test_colormix_rows_convex_invertible_and_in_range() -> None:
    cond = C.get_condition("colormix_d1")
    q = cond.parameters()["Q"]
    assert q.shape == (3, 3) and np.all(q >= 0) and np.allclose(q.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    eligible = []
    for s in cond.grid:
        a = C.colormix_matrix(q, s)
        assert np.all(a >= 0) and np.allclose(a.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        diag = C.operator_diagnostics("colormix", cond.parameters(), s, (32, 32))
        assert diag["eligible"] == (diag["cond_2"] <= C.COLORMIX_MAX_COND)
        if diag["eligible"]:
            eligible.append(s)
            assert abs(np.linalg.det(a)) > 0 and np.allclose(a @ np.linalg.inv(a), np.eye(3), atol=1e-10)
        out = C.apply_drawn(_extremes(), "colormix", cond.parameters(), s, chunk=4)
        assert out.dtype == np.uint8 and out.shape == (6, 3, 32, 32)
        # White stays white up to the storage rule: a row sum of 1 - 1 ulp truncates to 254.
        assert np.all(out[0] == 0) and np.all(out[1] >= 254)
        v = _extremes() / 255.0
        f = C.transform_float(v, "colormix", a)
        assert np.all(f >= 0) and np.all(f <= 1 + 1e-12)
    assert eligible and eligible[0] == cond.grid[0]
    if cond.severity is not None:
        assert cond.severity in eligible


def test_colormix_is_a_per_pixel_matrix_product() -> None:
    q = C.get_condition("colormix_d1").parameters()["Q"]
    x = _images(3, 4)
    out = C.apply_drawn(x, "colormix", {"Q": q}, 0.4)
    a = C.colormix_matrix(q, 0.4)
    expected = np.einsum("ck,nkhw->nchw", a, x / 255.0) * 255
    assert np.max(np.abs(out.astype(np.float64) - expected)) < 1.0 + 1e-9


def test_blur_kernel_sums_to_one_and_convolves() -> None:
    cond = C.get_condition("blur5_d1")
    params = cond.parameters()
    assert np.all(params["E"] > 0) and abs(params["R"].sum() - 1.0) < 1e-12
    for s in cond.grid:
        k = C.blur_kernel(params["R"], s)
        assert k.shape == (5, 5) and np.all(k >= 0) and abs(k.sum() - 1.0) < 1e-12
        out = C.apply_drawn(_extremes(), "blur5", params, s, chunk=4)
        assert np.all(out[0] == 0) and np.all(out[1] >= 254)
        f = C.transform_float(_extremes() / 255.0, "blur5", k)
        assert np.all(f >= 0) and np.all(f <= 1 + 1e-12) and np.allclose(f[1], 1.0, rtol=0, atol=1e-12)
    # True convolution with reflect padding: a kernel that is one impulse at (2, 3) moves content by (0, -1) there
    # (out[i, j] = v[i, j - 1]), which a cross-correlation would move the other way.
    v = _images(1, 7) / 255.0
    impulse = np.zeros((5, 5))
    impulse[2, 3] = 1.0
    out = C.transform_float(v, "blur5", impulse)
    padded = np.pad(v, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="reflect")
    assert np.array_equal(out, padded[:, :, 2:34, 1:33])


def test_gamma_monotone() -> None:
    cond = C.get_condition("gamma_d1")
    params = cond.parameters()
    levels = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, None, None, :], (1, 3, 32, 256)).copy()
    for s in cond.grid:
        gammas = C.gamma_exponents(params["z"], s)
        assert np.all(gammas > 0)
        f = C.transform_float(levels / 255.0, "gamma", gammas)
        assert np.all(np.diff(f, axis=3) > 0)
        stored = C.apply_drawn(levels, "gamma", params, s)
        assert np.all(np.diff(stored.astype(np.int64), axis=3) >= 0)
        assert np.all(stored[..., 0] == 0) and np.all(stored[..., -1] == 255)


@pytest.mark.parametrize("name", DRAWN)
def test_zero_severity_is_the_identity(name: str) -> None:
    cond = C.get_condition(name)
    params = cond.parameters()
    x = np.concatenate([_extremes(), _images(5, 8)])
    v = x / 255.0
    assert np.array_equal(C.transform_float(v, cond.transform, C.drawn_operator(cond.transform, params, 0.0)), v)
    assert np.array_equal(C.apply_drawn(x, cond.transform, params, 0.0, chunk=3), x)


@pytest.mark.parametrize("name", DRAWN)
def test_chunking_changes_nothing(name: str) -> None:
    cond = C.get_condition(name)
    x = _images(11, 9)
    s = cond.grid[len(cond.grid) // 2]
    assert np.array_equal(
        C.apply_drawn(x, cond.transform, cond.parameters(), s, chunk=1),
        C.apply_drawn(x, cond.transform, cond.parameters(), s, chunk=1000),
    )


def test_parameters_are_determined_by_the_stream() -> None:
    streams = [C.get_condition(name).parameter_stream for name in DRAWN]
    assert len(set(streams)) == len(streams)
    used = set(colored.COLOR_STREAMS.values()) | set(C.CORRUPTION_CONDITIONS["gaussian_noise_s3"].streams.values())
    used.add(700)  # the ogbn-arxiv node permutation stream (weightcode.data.arxiv.PERMUTATION_STREAM)
    used.add(701)  # the ogbn-arxiv old holdouts (weightcode.data.arxiv.OLD_HOLDOUT_STREAM)
    used.add(702)  # the ogbn-arxiv class-id permutation (weightcode.data.arxiv.CLASS_PERMUTATION_STREAM)
    used.add(703)  # the ogbn-arxiv inductive node ids (weightcode.data.arxiv.INDUCTIVE_PERMUTATION_STREAM)
    used.add(704)  # the E6 separability draw (weightcode.analysis.arxiv_headroom.SEPARABILITY_STREAM)
    used.add(705)  # the E7 new-class draw (weightcode.data.arxiv_classes.CLASS_STREAM)
    used.add(706)  # the E7 replay draw (weightcode.evalkit.graphnew.REPLAY_STREAM)
    used.add(707)  # the E9 base-class holdouts (weightcode.data.cifar100_fscil.BASE_HOLDOUT_STREAM)
    used.add(708)  # the E9 new-class split (weightcode.data.cifar100_fscil.NEW_CLASS_STREAM)
    used.add(709)  # the E9 replay draw (weightcode.evalkit.fscil.REPLAY_STREAM)
    from weightcode.data import cifar100_fscil

    assert (cifar100_fscil.BASE_HOLDOUT_STREAM, cifar100_fscil.NEW_CLASS_STREAM) == (707, 708)
    assert not set(streams) & used
    for name in DRAWN:
        cond = C.get_condition(name)
        first, second = cond.parameters(), cond.parameters()
        expected = C.draw_parameters(cond.transform, np.random.default_rng([SPLIT_SEED, cond.parameter_stream]))
        assert first.keys() == second.keys() == expected.keys()
        for key in first:
            assert np.array_equal(first[key], second[key]) and np.array_equal(first[key], expected[key])
        other = C.draw_parameters(cond.transform, np.random.default_rng([SPLIT_SEED, cond.parameter_stream + 1]))
        assert not all(np.array_equal(first[key], other[key]) for key in first)


def test_corruption_builder_refuses_a_drawn_condition() -> None:
    with pytest.raises(TypeError, match="CIFAR-10-C"):
        C.build_condition("gamma_d1", {}, {}, {}, "0" * 64)


# ---------------------------------------------------------------------------------------------------------------
# Severity rule


def _curve(drops: list[int], eligible: list[bool] | None = None) -> list[dict]:
    eligible = eligible or [True] * len(drops)
    return [{"severity": (k + 1) / 20, "eligible": e, "drop_count": d} for k, (d, e) in enumerate(zip(drops, eligible))]


def test_select_severity_rule() -> None:
    # Reference: 4 pairs dropping 40 in total, so D* = 10 per pair; with 2 models s reaches at drop_count >= 20.
    assert K.select_severity(_curve([5, 19, 20, 30]), 40, 4, 2) == {"severity": 0.15, "status": "reached_target"}
    assert K.select_severity(_curve([5, 19, 20, 30], [True, True, False, True]), 40, 4, 2)["severity"] == 0.2
    below = K.select_severity(_curve([5, 19, 18, 30], [True, True, True, False]), 40, 4, 2)
    assert below == {"severity": 0.15, "status": "below_target"}
    with pytest.raises(ValueError, match="eligible"):
        K.select_severity(_curve([30], [False]), 40, 4, 2)
    with pytest.raises(ValueError, match="increasing"):
        K.select_severity(list(reversed(_curve([1, 2]))), 40, 4, 2)


@pytest.mark.parametrize("split", ["V_select", "V_evidence", "V_check", "test", "train_uniform"])
def test_calibration_refuses_every_split_but_train(split: str) -> None:
    with pytest.raises(ValueError, match="only the train split"):
        K.load_calibration_split("cifar10", split)
    with pytest.raises(ValueError, match="only the train split"):
        K.load_calibration_split("cifar10", split, "contrast_s3")


# ---------------------------------------------------------------------------------------------------------------
# Calibrate, build, write, verify on a tiny store

SIZES = {"train": 64, "V_select": 32, "V_evidence": 16, "V_check": 16}


def _train_base_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_base_drawn_test", REPO / "scripts" / "train_base.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tiny_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """A clean cifar10 store with contrast_s3 and gaussian_noise_s3 and dev models 100 and 101; returns raw test."""
    for name in ("private", "sealed"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(tmp_path / "sealed"))
    rng = np.random.default_rng(5)
    splits_meta = {}
    for offset, (split, n) in enumerate(SIZES.items()):
        y = np.arange(n, dtype=np.int64) % 10
        x = rng.integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)
        x[:, y % 3] = np.minimum(x[:, y % 3], 60)
        arrays = {"x": x, "y": y, "idx": np.arange(100 * offset, 100 * offset + n, dtype=np.int64)}
        splits_meta[split] = {"path": f"splits/{split}.npz", "n": n, "sha256": store.save_split(store.split_path("cifar10", split), arrays)}
    test = {
        "x": rng.integers(0, 256, size=(20, 3, 32, 32), dtype=np.uint8),
        "y": np.arange(20, dtype=np.int64) % 10,
        "idx": np.arange(20, dtype=np.int64),
    }
    mean, std = store.channel_stats(store.load_split("cifar10", "train")["x"])
    forward_sha = write_forward(store.forward_path("cifar10"), render_forward((3, 32, 32), mean, std, 10))
    store.write_json(
        store.dataset_json_path("cifar10"),
        {
            "schema": "weightcode.dataset.v0",
            "dataset": "cifar10",
            "source": "CIFAR10",
            "class_names": list(CIFAR10_CLASSES),
            "image_shape": [3, 32, 32],
            "norm": {"mean": mean, "std": std, "computed_on": "train"},
            "split_seed": SPLIT_SEED,
            "splits": splits_meta,
            "test": {"location": "$WEIGHTCODE_SEALED/cifar10/test.npz", "n": 20, "sha256": store.split_sha256(test)},
            "forward_sha256": forward_sha,
        },
    )
    for ref in K.REFERENCE_CONDITIONS:
        C.write_condition(C.build_from_store(ref, test), {"note": "test"})
    git_state = tmp_path / "git_state.json"
    git_state.write_text(json.dumps({"commit": "0" * 40, "dirty": True}))
    trainer = _train_base_module().Trainer("cifar10", "cpu", git_state_file=git_state)
    for seed in (100, 101):
        trainer.train_one("dev", seed, "train")
    return test


def _with_severity(monkeypatch: pytest.MonkeyPatch, name: str, severity: float | None) -> None:
    cond = C.DRAWN_CONDITIONS[name]
    replaced = C.DrawnCondition(cond.name, cond.dataset, cond.transform, cond.parameter_stream, severity)
    monkeypatch.setattr(C, "CONDITIONS", MappingProxyType({**C.CONDITIONS, name: replaced}))


def test_calibrate_build_write_verify(tiny_store: dict[str, np.ndarray], monkeypatch: pytest.MonkeyPatch) -> None:
    name = "colormix_d1"
    read: list[tuple[str, str | None]] = []
    original = store.load_split

    def recording_load(dataset: str, split: str, condition: str | None = None) -> dict[str, np.ndarray]:
        read.append((split, condition))
        return original(dataset, split, condition)

    monkeypatch.setattr(store, "load_split", recording_load)
    log = K.calibrate(name, {"note": "test"}, log_fn=lambda line: None)
    monkeypatch.setattr(store, "load_split", original)
    assert {split for split, _ in read} == {"train"}
    assert sorted({c for _, c in read}, key=str) == sorted([None, *K.REFERENCE_CONDITIONS], key=str)

    cond = C.get_condition(name)
    assert log["split"] == "train" and [row["severity"] for row in log["curve"]] == list(cond.grid)
    assert [m["seed"] for m in log["models"]] == [100, 101]
    assert log["parameters"]["values"]["Q"] == cond.parameters()["Q"].tolist()
    assert all("cond_2" in row and len(row["per_model"]) == 2 for row in log["curve"])
    choice = K.select_severity(log["curve"], log["target_drop"]["drop_count"], 4, 2)
    assert log["chosen"]["severity"] == choice["severity"] and log["chosen"]["status"] == choice["status"]
    path = K.write_calibration(log)
    assert path == store.dataset_dir("cifar10", name) / "calibration.json"
    with pytest.raises(FileExistsError):
        K.write_calibration(log)
    assert store.list_conditions("cifar10") == sorted(K.REFERENCE_CONDITIONS)

    # The registered severities are entered after calibration; the refusal is tested on an uncalibrated copy.
    _with_severity(monkeypatch, name, None)
    with pytest.raises(ValueError, match="no calibrated severity"):
        C.build_from_store(name, tiny_store)
    other = next(s for s in cond.grid if s != log["chosen"]["severity"])
    _with_severity(monkeypatch, name, other)
    with pytest.raises(ValueError, match="chosen severity"):
        C.build_from_store(name, tiny_store)

    chosen = log["chosen"]["severity"]
    _with_severity(monkeypatch, name, chosen)
    built = C.build_from_store(name, tiny_store)
    manifest = C.write_condition(built, {"note": "test"})
    recorded = store.read_dataset_json("cifar10", name)
    assert recorded == manifest
    block = recorded["condition"]
    assert block["kind"] == "drawn_transform" and block["transform"] == "colormix" and block["severity"] == chosen
    assert block["formula"] == C.DRAWN_FORMULAS["colormix"] and block["storage"] == C.STORAGE_RULE
    assert block["parameters"]["values"]["Q"] == cond.parameters()["Q"].tolist()
    assert block["parameters"]["stream"] == cond.parameter_stream and block["parameters"]["split_seed"] == SPLIT_SEED
    assert block["calibration"]["path"] == "calibration.json"
    assert block["calibration"]["sha256"] == C._file_sha256(path)
    assert block["calibration"]["status"] == log["chosen"]["status"]
    assert block["operator_at_severity"]["A"] == C.colormix_matrix(cond.parameters()["Q"], chosen).tolist()
    assert block["severity_rule"]["grid"] == list(cond.grid)
    assert store.list_conditions("cifar10") == sorted([*K.REFERENCE_CONDITIONS, name])
    for split in SPLIT_NAMES:
        on_disk = store.load_split("cifar10", split, name)
        clean = store.load_split("cifar10", split)
        assert np.array_equal(on_disk["y"], clean["y"]) and np.array_equal(on_disk["idx"], clean["idx"])
        assert recorded["splits"][split]["clean_sha256"] == store.read_dataset_json("cifar10")["splits"][split]["sha256"]
    assert store.sealed_test_path("cifar10", name).is_file()

    report = C.verify_condition(C.build_from_store(name, tiny_store))
    assert all(entry["match"] for entry in report.values())
    with pytest.raises(FileExistsError, match="precedes the build"):
        K.calibrate(name, {}, log_fn=lambda line: None)

    tampered = json.loads(path.read_text())
    tampered["parameters"]["values"]["Q"][0][0] += 1e-9
    path.write_text(json.dumps(tampered, indent=2) + "\n")
    with pytest.raises(ValueError, match="parameters"):
        C.build_from_store(name, tiny_store)
    store.write_json(path, log)
    assert C._file_sha256(path) == block["calibration"]["sha256"]
    assert all(entry["match"] for entry in C.verify_condition(C.build_from_store(name, tiny_store)).values())
