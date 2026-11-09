"""End-to-end tests of the fitting references and random controls on tiny synthetic instances.

Each toy instance has the real layout (model.npz, model_spec.json, forward.py, evidence.npz, task.json, private.json
with correct hashes, a fake V_check). Labels come from a random teacher network; the base model is the teacher with
the final bias of class b raised, so images of class a near the a/b boundary become (a -> b) failures. The defect sits
in the final layer, so a working final-layer fit must repair part of it on V_check. Nothing is written to $WEIGHTCODE_*
locations except through the CLI test, which points the variables at a temporary directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from weightcode.baselines.fit.controls import N_DRAWS, baseline_session_id, control_run_seed, run_control
from weightcode.baselines.fit.edits import MAX_REL_FROB, PATCH_APPLIER, REL_FROB_MIN_NDIM, apply_patch_path
from weightcode.baselines.fit.grad import candidate_grid, run_grad
from weightcode.baselines.fit.ridge import RIDGE_SCALES, RIDGE_SPECS, run_ridge
from weightcode.baselines.fit.rules import run_prior_shift, run_tie
from weightcode.baselines.fit.runner import RUN_SCHEMA, SplitRoleError, check_split_role, read_run_record
from weightcode.baselines.fit.torch_forward import architecture_from, assert_parity, max_logit_diff
from weightcode.baselines.instance_io import Instance, load_instance, write_delta_patch
from weightcode.common.identity import params_sha256, save_params
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.evalkit.patch_check import load_forward

REPO = Path(__file__).resolve().parents[1]
N_CLASSES = 4

TOY_FORWARD = '''"""Toy numpy reference forward for tests: same signature as an instance forward.py."""
import numpy as np

SHAPE = __SHAPE__
MEAN = np.array(__MEAN__, dtype=np.float32)
STD = np.array(__STD__, dtype=np.float32)


def _conv_same(x, w, b):
    out_c, in_c, k, _ = w.shape
    p = (k - 1) // 2
    n, _, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    cols = np.stack([xp[:, :, i:i + h, j:j + wd] for i in range(k) for j in range(k)], axis=2)
    return np.einsum("nckhw,ock->nohw", cols, w.reshape(out_c, in_c, k * k)) + b[None, :, None, None]


def _pool2(x):
    n, c, h, w = x.shape
    x = x[:, :, : h - h % 2, : w - w % 2]
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def forward(params, x_uint8):
    x = np.asarray(x_uint8).reshape(len(x_uint8), *SHAPE).astype(np.float32) / 255.0
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    n_conv = sum(1 for k in params if k.startswith("conv") and k.endswith(".weight"))
    n_fc = sum(1 for k in params if k.startswith("fc") and k.endswith(".weight"))
    for i in range(1, n_conv + 1):
        x = _pool2(np.maximum(_conv_same(x, params[f"conv{i}.weight"], params[f"conv{i}.bias"]), 0.0))
    x = x.reshape(len(x), -1)
    for i in range(1, n_fc + 1):
        x = x @ params[f"fc{i}.weight"].T + params[f"fc{i}.bias"]
        if i < n_fc:
            x = np.maximum(x, 0.0)
    return x
'''


@dataclass(frozen=True)
class Toy:
    """A built toy instance and the teacher it was derived from."""

    instance: Instance
    teacher: dict[str, np.ndarray]


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def _teacher(
    family: str, rng: np.random.Generator, channels: int = 1
) -> tuple[dict[str, np.ndarray], tuple[int, int, int]]:
    params: dict[str, np.ndarray] = {}
    if family == "mlp":
        shape, dims = (channels, 8, 8), [channels * 64, 24, 24, N_CLASSES]
        for i in range(3):
            params[f"fc{i + 1}.weight"] = _he(rng, (dims[i + 1], dims[i]), dims[i])
            params[f"fc{i + 1}.bias"] = (0.1 * rng.standard_normal(dims[i + 1])).astype(np.float32)
        return params, shape
    shape, channels = (3, 16, 16), [3, 4, 6, 8]
    for i in range(3):
        params[f"conv{i + 1}.weight"] = _he(rng, (channels[i + 1], channels[i], 3, 3), channels[i] * 9)
        params[f"conv{i + 1}.bias"] = (0.1 * rng.standard_normal(channels[i + 1])).astype(np.float32)
    params["fc1.weight"] = _he(rng, (16, 8 * 2 * 2), 32)
    params["fc1.bias"] = (0.1 * rng.standard_normal(16)).astype(np.float32)
    params["fc2.weight"] = _he(rng, (N_CLASSES, 16), 16)
    params["fc2.bias"] = (0.1 * rng.standard_normal(N_CLASSES)).astype(np.float32)
    return params, shape


def build_toy_instance(
    private_root: Path,
    family: str,
    *,
    seed: int = 0,
    split_role: str = "dev",
    rule: str = "evidence-v0",
    channels: int = 1,
) -> Toy:
    """Write a toy instance under ``private_root/instances/<id>`` and load it with the shared reader.

    evidence-v0: 16 target-pair failures (y = a predicted b). evidence-v1: 16 random base errors of any class and no
    target pair. The MLP input has ``channels`` channels; the CNN input always has 3.
    """
    rng = np.random.default_rng(seed)
    teacher, shape = _teacher(family, rng, channels)
    v1 = rule == "evidence-v1"
    instance_id = f"toy_{family}{'' if channels == 1 else f'c{channels}'}{'_v1' if v1 else ''}_{split_role}_{seed}"
    root = private_root / "instances" / instance_id
    root.mkdir(parents=True)
    mean, std = [0.5] * shape[0], [0.25] * shape[0]
    source = TOY_FORWARD.replace("__SHAPE__", repr(shape)).replace("__MEAN__", repr(mean))
    (root / "forward.py").write_text(source.replace("__STD__", repr(std)))
    forward, forward_sha = load_forward(root / "forward.py")
    images = rng.integers(0, 256, size=(3000, *shape), dtype=np.uint8)
    teacher_logits = forward(teacher, images)
    y = np.argmax(teacher_logits, axis=1).astype(np.int64)
    a = int(np.argmax(np.bincount(y, minlength=N_CLASSES)))
    others = teacher_logits.copy()
    others[:, a] = -np.inf
    in_a = y == a
    b = int(np.argmax(np.bincount(np.argmax(others[in_a], axis=1), minlength=N_CLASSES)))
    margin = teacher_logits[in_a, a] - teacher_logits[in_a, b]
    base = {key: value.copy() for key, value in teacher.items()}
    last_bias = "fc3.bias" if family == "mlp" else "fc2.bias"
    base[last_bias][b] += np.float32(np.quantile(margin, 0.4))
    base_pred = np.argmax(forward(base, images), axis=1)
    pool = np.arange(1000)
    if v1:
        errors = pool[base_pred[pool] != y[pool]]
        assert len(errors) >= 16, "toy pool has too few base errors"
        fail_rows = np.sort(rng.choice(errors, size=16, replace=False))
    else:
        fail_rows = pool[(y[pool] == a) & (base_pred[pool] == b)][:16]
    assert len(fail_rows) == 16, "toy pool has too few target failures"
    rand_rows = rng.choice(np.setdiff1d(pool, fail_rows), size=48, replace=False)
    np.savez(
        root / "evidence.npz",
        x_fail=images[fail_rows],
        y_fail=y[fail_rows],
        pred_fail=base_pred[fail_rows],
        x_rand=images[rand_rows],
        y_rand=y[rand_rows],
        pred_rand=base_pred[rand_rows],
        split_ids=np.array([f"e{i:012x}" for i in range(64)]),
    )
    vcheck = private_root / f"vcheck_{instance_id}.npz"
    np.savez(vcheck, x=images[1000:], y=y[1000:], idx=np.arange(1000, 3000, dtype=np.int64))
    model_sha = save_params(root / "model.npz", base)
    spec = {
        "schema": "weightcode.model_spec.v0",
        "model_id": "toy",
        "input": {"shape": list(shape), "dtype": "uint8", "preprocess": "x/255 then (x - mean) / std, per channel",
                  "mean": mean, "std": std, "layout": "NCHW, flatten row-major"},
        "classes": [f"c{i}" for i in range(N_CLASSES)],
        "output": "logits; prediction = argmax",
    }
    (root / "model_spec.json").write_text(json.dumps(spec))
    task = {"schema": "weightcode.task.v1"} if v1 else {"schema": "weightcode.task.v0", "target": {"a": a, "b": b}}
    (root / "task.json").write_text(json.dumps(task))
    private = {
        "schema": "weightcode.instance_private.v1" if v1 else "weightcode.instance_private.v0",
        "dataset": "toy_v1" if v1 else "toy",
        "split_role": split_role,
        "seed": seed,
        "arch": family,
        "evidence_rule": rule,
        "vcheck_path": str(vcheck),
        "model_sha256": model_sha,
        "forward_sha256": forward_sha,
        "k_fail": 16,
        "k_rand": 48,
    }
    if v1:
        private["condition"] = "toy"
    else:
        private["target"] = [a, b]
    (root / "private.json").write_text(json.dumps(private))
    return Toy(instance=load_instance(root), teacher=teacher)


@pytest.fixture(scope="module")
def toys(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Toy]:
    private_root = tmp_path_factory.mktemp("private")
    built = {
        "mlp": build_toy_instance(private_root, "mlp"),
        "cnn": build_toy_instance(private_root, "cnn", seed=1),
        "main": build_toy_instance(private_root, "mlp", seed=2, split_role="main"),
        "v1_mlp": build_toy_instance(private_root, "mlp", seed=3, rule="evidence-v1", channels=3),
        "v1_cnn": build_toy_instance(private_root, "cnn", seed=4, rule="evidence-v1"),
    }
    rows = [
        {
            "instance_id": toy.instance.instance_id,
            "dataset": toy.instance.private["dataset"],
            "split_role": toy.instance.private["split_role"],
        }
        for toy in built.values()
    ]
    (private_root / "instances" / "index.json").write_text(
        json.dumps({"schema": "weightcode.instances_index.v0", "instances": rows})
    )
    return built


def _spec(instance: Instance) -> dict:
    return json.loads((instance.instance_dir / "model_spec.json").read_text())


def _check_final_rule(record: dict) -> None:
    candidates = [
        Candidate(patch_id=e["patch_id"], score=e["score"], n_changed=e["n_changed"], query_index=e["query_index"])
        for e in record["oracle_log"]
    ]
    expected = choose_final(candidates)
    assert record["final"]["patch_id"] == expected.patch_id
    assert record["final"]["score"] == max([0, *(c.score for c in candidates)])


def _ops(run_dir: Path, slot: int) -> list[dict]:
    return json.loads((run_dir / f"q{slot}" / "patch.json").read_text())["ops"]


@pytest.mark.parametrize("family", ["mlp", "cnn"])
def test_torch_rebuild_matches_toy_forward(toys: dict[str, Toy], family: str) -> None:
    toy = toys[family]
    instance = toy.instance
    arch = architecture_from(instance.params, _spec(instance))
    assert arch.family == family
    x = np.random.default_rng(5).integers(0, 256, size=(64, *arch.input_shape), dtype=np.uint8)
    for params in (instance.params, toy.teacher):
        assert assert_parity(arch, params, instance.forward, x) <= 1e-4


@pytest.mark.parametrize("family", ["mlp", "cnn"])
def test_torch_rebuild_matches_dataset_forward_template(toys: dict[str, Toy], family: str, tmp_path: Path) -> None:
    forward_gen = pytest.importorskip("weightcode.models.forward_gen", reason="dataset forward template not available")
    instance = toys[family].instance
    spec = _spec(instance)
    source = forward_gen.render_forward(spec["input"]["shape"], spec["input"]["mean"], spec["input"]["std"], N_CLASSES)
    (tmp_path / "forward.py").write_text(source)
    forward, _ = load_forward(tmp_path / "forward.py")
    arch = architecture_from(instance.params, spec)
    x = np.random.default_rng(6).integers(0, 256, size=(64, *arch.input_shape), dtype=np.uint8)
    assert max_logit_diff(arch, instance.params, forward, x) <= 1e-4


def test_parity_check_catches_a_wrong_rebuild(toys: dict[str, Toy]) -> None:
    instance = toys["mlp"].instance
    spec = _spec(instance)
    spec["input"]["mean"] = [0.4]
    arch = architecture_from(instance.params, spec)
    x = instance.evidence["x_rand"]
    with pytest.raises(RuntimeError, match="does not match"):
        assert_parity(arch, instance.params, instance.forward, x)


def test_grad_last_end_to_end(toys: dict[str, Toy], tmp_path: Path) -> None:
    instance = toys["mlp"].instance
    run_dir = tmp_path / "grad_last" / "0"
    record = run_grad(instance, "grad_last", run_dir)
    assert record == read_run_record(run_dir / "run.json") or record["status"] == "complete"
    on_disk = read_run_record(run_dir / "run.json")
    assert on_disk["schema"] == RUN_SCHEMA and on_disk["status"] == "complete"
    assert on_disk["queries_used"] == 8 and len(on_disk["oracle_log"]) == 8
    assert [c["config"] for c in on_disk["candidates"]] == [{"lr": lr, "steps": s} for lr, s in candidate_grid()]
    assert on_disk["flags"] == {"uses_labels_for_fit": True, "edit_limits_enforced": True, "respects_edit_limits": True}
    assert on_disk["generation"]["parity_max_abs_logit_diff"] <= 1e-4
    for entry in on_disk["candidates"]:
        assert entry["roundtrip_exact"]
        assert {op["tensor"] for op in _ops(run_dir, entry["slot"])} <= {"fc3.weight", "fc3.bias"}
        for key, value in entry["edit"]["per_tensor"].items():
            if np.ndim(instance.params[key]) >= REL_FROB_MIN_NDIM:
                assert value["rel_frob"] <= MAX_REL_FROB
            if not key.startswith("fc3."):
                assert value["changed"] == 0
    assert on_disk["generation"]["n_candidates_norm_scaled"] >= 1  # lr 1e-2 for 300 steps exceeds the limit
    _check_final_rule(on_disk)
    assert on_disk["final"]["score"] > 0, "a final-layer fit must repair part of a final-bias defect"
    assert not on_disk["final"]["is_base_noop"]


def test_grad_sparse_respects_mask_and_limits(toys: dict[str, Toy], tmp_path: Path) -> None:
    instance = toys["mlp"].instance
    record = run_grad(instance, "grad_sparse", tmp_path / "grad_sparse" / "0")
    mask = record["generation"]["mask"]
    n_params = sum(value.size for value in instance.params.values())
    assert mask["k"] == n_params * 5 // 100 and mask["n_params"] == n_params
    assert sum(mask["selected_per_tensor"].values()) == mask["k"]
    for entry in record["candidates"]:
        assert entry["edit"]["n_changed"] <= mask["k"]
        assert entry["edit"]["within_limits"]
    assert record["flags"]["respects_edit_limits"]
    _check_final_rule(record)


def test_grad_full_is_an_unconstrained_reference_on_a_cnn(toys: dict[str, Toy], tmp_path: Path) -> None:
    instance = toys["cnn"].instance
    record = run_grad(instance, "grad_full", tmp_path / "grad_full" / "0")
    assert record["flags"]["edit_limits_enforced"] is False
    assert record["flags"]["respects_edit_limits"] is False
    assert "unconstrained" in record["label"]
    assert record["queries_used"] == 8
    assert record["candidates"][-1]["edit"]["changed_frac"] > 0.05
    _check_final_rule(record)


def test_ridge_variants_end_to_end(toys: dict[str, Toy], tmp_path: Path) -> None:
    instance = toys["v1_mlp"].instance  # its two smallest-lam ridge solutions exceed the per-tensor norm limit
    records = {}
    for method in ("ridge_last", "ridge_last_free"):
        record = run_ridge(instance, tmp_path / method / "0", method=method)
        records[method] = record
        assert record["method"] == method and record["label"] == RIDGE_SPECS[method].label
        assert "not training-free" in record["label"]
        assert record["flags"]["uses_labels_for_fit"] is True
        assert record["flags"]["edit_limits_enforced"] is (method == "ridge_last")
        assert [c["config"]["lam_relative"] for c in record["candidates"]] == list(RIDGE_SCALES)
        residuals = [c["notes"]["target_residual_rel"] for c in record["candidates"]]
        assert residuals == sorted(residuals), "the fit residual must grow with lam"
        assert residuals[0] < 0.5 < residuals[-1]
        assert record["generation"]["float64_feature_parity"] <= 1e-4
        for entry in record["candidates"]:
            assert {op["tensor"] for op in _ops(tmp_path / method / "0", entry["slot"])} <= {"fc3.weight", "fc3.bias"}
        _check_final_rule(record)
    bounded, free = records["ridge_last"], records["ridge_last_free"]
    assert bounded["flags"]["respects_edit_limits"] is True
    assert free["flags"]["respects_edit_limits"] is False, "the toy's small-lam solutions exceed the norm limit"
    assert bounded["generation"]["n_candidates_norm_scaled"] == 2
    for scaled, unscaled in zip(bounded["candidates"], free["candidates"], strict=True):
        assert scaled["edit"]["per_tensor"]["fc3.weight"]["rel_frob"] <= MAX_REL_FROB
        if scaled["notes"]["norm_scaled_tensors"]:
            assert scaled["notes"]["norm_scaled_tensors"] == ["fc3.weight"]
            assert unscaled["edit"]["per_tensor"]["fc3.weight"]["rel_frob"] > MAX_REL_FROB
            assert scaled["edit"]["per_tensor"]["fc3.weight"]["rel_frob"] == pytest.approx(MAX_REL_FROB * (1 - 1e-4))
        else:
            assert scaled["check"]["child_sha256"] == unscaled["check"]["child_sha256"]
        assert scaled["edit"]["per_tensor"]["fc3.bias"] == unscaled["edit"]["per_tensor"]["fc3.bias"]


def test_controls_match_the_grad_last_reference(toys: dict[str, Toy], tmp_path: Path) -> None:
    instance = toys["mlp"].instance
    reference_dir = tmp_path / "grad_last" / "0"
    reference = run_grad(instance, "grad_last", reference_dir)
    assert not reference["final"]["is_base_noop"]
    ref_edit = next(c["edit"] for c in reference["candidates"] if c["patch_id"] == reference["final"]["patch_id"])
    ref_child = apply_patch_path(instance.params, reference_dir / reference["final"]["dir"] / "patch.json",
                                 instance.params_sha256, enforce_limits=True)
    session = baseline_session_id(instance.instance_id, "grad_last", 0)
    for kind in ("permuted", "gaussian"):
        run_dir = tmp_path / kind / "grad_last" / "r1"
        record = run_control(instance, kind, reference_dir, session_id=session, arm="grad_last", run_index=1,
                             run_dir=run_dir)
        assert record["status"] == "complete" and record["queries_used"] == N_DRAWS
        assert record["run_seed"] == control_run_seed(session, kind, 1)
        assert record["config"]["b1"]["session_id"] == session and record["config"]["b1"]["run_index"] == 1
        assert record["flags"]["uses_labels_for_fit"] is False and record["flags"]["edit_limits_enforced"] is False
        assert record["generation"]["reference"]["kind"] == "run_record"
        assert record["generation"]["reference"]["child_sha256"] == reference["final"]["child_sha256"]
        assert [entry["config"]["rng"] for entry in record["candidates"]] == [
            [record["run_seed"], j] for j in range(N_DRAWS)
        ]
        hashes = {entry["check"]["child_sha256"] for entry in record["candidates"]}
        assert len(hashes) == N_DRAWS
        for entry in record["candidates"]:
            for key, ref in ref_edit["per_tensor"].items():
                got = entry["edit"]["per_tensor"][key]
                assert got["changed"] == ref["changed"]
                assert got["rel_frob"] == pytest.approx(ref["rel_frob"], rel=1e-5)
            child = apply_patch_path(instance.params, run_dir / entry["patch_id"] / "patch.json",
                                     instance.params_sha256, enforce_limits=False)
            same_positions = all(
                np.array_equal(child[key] != instance.params[key], ref_child[key] != instance.params[key])
                for key in instance.params
            )
            if kind == "gaussian":
                assert same_positions
            assert any(not np.array_equal(child[key], ref_child[key]) for key in instance.params)
        _check_final_rule(record)


def test_control_of_a_noop_reference_writes_no_control(toys: dict[str, Toy], tmp_path: Path) -> None:
    instance = toys["mlp"].instance
    empty = tmp_path / "empty_patch"
    write_delta_patch(empty, instance.params_sha256, instance.params, instance.params, "no-op", "none")
    fake_run = tmp_path / "fake_run"
    fake_run.mkdir()
    (fake_run / "run.json").write_text(json.dumps({
        "schema": RUN_SCHEMA, "status": "complete", "instance": {"instance_id": instance.instance_id},
        "final": {"is_base_noop": True, "dir": None},
    }))
    for label, reference in (("empty", empty), ("noop_run", fake_run)):
        run_dir = tmp_path / "permuted" / label / "r0"
        record = run_control(instance, "permuted", reference, session_id=f"{label}-session", arm=label, run_index=0,
                             run_dir=run_dir)
        assert record["generation"]["reference_is_noop"] is True
        assert record["queries_used"] == 0 and record["final"]["is_base_noop"]
        assert "oracle_log" not in record
        assert sorted(path.name for path in run_dir.iterdir()) == ["run.json"]


def test_split_role_guard_and_no_overwrite(toys: dict[str, Toy], tmp_path: Path) -> None:
    main = toys["main"].instance
    run_dir = tmp_path / "ridge_last" / "0"
    with pytest.raises(SplitRoleError, match="main instance"):
        run_ridge(main, run_dir)
    assert not run_dir.exists()
    assert check_split_role(main, allow_main=True) == "main"
    unknown = Instance(**{**main.__dict__, "private": {**main.private, "split_role": None}})
    with pytest.raises(SplitRoleError, match="expected"):
        check_split_role(unknown, allow_main=True)
    dev = toys["mlp"].instance
    run_ridge(dev, run_dir)
    with pytest.raises(FileExistsError):
        run_ridge(dev, run_dir)


def test_written_patch_roundtrips_through_the_executor(toys: dict[str, Toy], tmp_path: Path) -> None:
    from weightcode.patch import Limits, apply_patch, load_patch

    instance = toys["mlp"].instance
    record = run_grad(instance, "grad_sparse", tmp_path / "grad_sparse" / "0")
    assert record["patch_applier"] == PATCH_APPLIER
    slot = record["final"]["patch_id"] or "q0"
    patch_path = tmp_path / "grad_sparse" / "0" / slot / "patch.json"
    result = apply_patch(instance.params, load_patch(patch_path), patch_path.parent, Limits(), instance.params_sha256)
    expected = next(c["check"]["child_sha256"] for c in record["candidates"] if c["patch_id"] == slot)
    assert result.child_sha256 == expected == params_sha256(result.child)
    via_helper = apply_patch_path(instance.params, patch_path, instance.params_sha256, enforce_limits=True)
    assert params_sha256(via_helper) == expected


def _cli_env(toys: dict[str, Toy], tmp_path: Path) -> tuple[dict[str, str], Path]:
    private_root = toys["mlp"].instance.instance_dir.parents[1]
    runs = tmp_path / "runs"
    sealed = tmp_path / "sealed"
    runs.mkdir()
    sealed.mkdir()
    return dict(os.environ, WEIGHTCODE_PRIVATE=str(private_root), WEIGHTCODE_RUNS=str(runs), WEIGHTCODE_SEALED=str(sealed)), runs


def _run_script(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args, "--threads", "2"], cwd=REPO, env=env, capture_output=True, text=True, timeout=600
    )


def test_cli_selects_dev_instances_from_the_index_by_default(toys: dict[str, Toy], tmp_path: Path) -> None:
    env, runs = _cli_env(toys, tmp_path)
    dev_ids = [toys["mlp"].instance.instance_id, toys["cnn"].instance.instance_id]
    main_id = toys["main"].instance.instance_id
    fit = _run_script(env, "scripts/run_fit_baselines.py", "--dataset", "toy", "--methods", "grad_last", "ridge_last")
    assert fit.returncode == 0, fit.stderr
    for instance_id in dev_ids:
        for method in ("grad_last", "ridge_last"):
            record = read_run_record(runs / "baselines" / instance_id / method / "0" / "run.json")
            assert record["status"] == "complete" and record["instance"]["split_role"] == "dev"
    assert not (runs / "baselines" / main_id).exists()
    template = str(runs / "baselines" / "{instance_id}" / "grad_last" / "0")
    controls = _run_script(
        env, "scripts/run_random_controls.py", "--dataset", "toy", "--reference-patch", template,
        "--ref-label", "grad_last",
    )
    assert controls.returncode == 0, controls.stderr
    for instance_id in dev_ids:
        session = f"{instance_id}/grad_last/0"
        for kind in ("permuted", "gaussian"):
            for run_index in range(3):
                path = runs / "baselines" / instance_id / kind / "grad_last" / f"r{run_index}" / "run.json"
                record = read_run_record(path)
                assert record["status"] == "complete"
                assert record["generation"]["reference"]["kind"] == "run_record"
                assert record["config"]["b1"]["session_id"] == session
                assert record["run_seed"] == control_run_seed(session, kind, run_index)
            assert f"B1 {instance_id} {kind}[grad_last] session {session}: final S per run" in controls.stdout
    assert not (runs / "baselines" / main_id).exists()
    again = _run_script(env, "scripts/run_random_controls.py", "--dataset", "toy", "--reference-patch", template)
    assert again.returncode == 2 and "already exist" in again.stderr


def test_cli_refuses_main_instances_without_allow_main(toys: dict[str, Toy], tmp_path: Path) -> None:
    env, runs = _cli_env(toys, tmp_path)
    main_id = toys["main"].instance.instance_id
    dev_id = toys["mlp"].instance.instance_id
    attempts = [
        ("scripts/run_fit_baselines.py", "--instance", main_id, "--methods", "ridge_last"),
        ("scripts/run_fit_baselines.py", "--instance", dev_id, "--instance", main_id, "--methods", "ridge_last"),
        ("scripts/run_fit_baselines.py", "--split-role", "main", "--methods", "ridge_last"),
        ("scripts/run_random_controls.py", "--instance", main_id, "--reference-patch", str(tmp_path),
         "--ref-label", "x"),
        ("scripts/run_random_controls.py", "--split-role", "main", "--reference-patch", "{instance_id}",
         "--ref-label", "x"),
    ]
    for args in attempts:
        refused = _run_script(env, *args)
        assert refused.returncode == 2, (args, refused.stderr)
        assert "refused:" in refused.stderr and "main instance" in refused.stderr and "--allow-main" in refused.stderr
        assert "Traceback" not in refused.stderr
    assert not (runs / "baselines").exists(), "a refused selection must not write anything"
    several = _run_script(env, "scripts/run_random_controls.py", "--reference-patch", str(tmp_path), "--ref-label", "x")
    assert several.returncode == 2 and "{instance_id}" in several.stderr
    allowed = _run_script(env, "scripts/run_fit_baselines.py", "--instance", main_id, "--methods", "ridge_last",
                          "--allow-main")
    assert allowed.returncode == 0, allowed.stderr
    record = read_run_record(runs / "baselines" / main_id / "ridge_last" / "0" / "run.json")
    assert record["instance"]["split_role"] == "main" and record["instance"]["allow_main"] is True


def test_v1_runs_record_fixed_and_broken_without_a_target(toys: dict[str, Toy], tmp_path: Path) -> None:
    from weightcode.baselines.fit.cli import summary_line

    instance = toys["v1_mlp"].instance
    assert instance.target is None
    record = run_grad(instance, "grad_last", tmp_path / "grad_last" / "0")
    assert record["status"] == "complete" and record["queries_used"] == 8
    assert record["instance"]["evidence_rule"] == "evidence-v1" and record["instance"]["target"] is None
    assert record["evidence"]["rule"] == "evidence-v1"
    for entry in record["oracle_log"]:
        assert entry["fixed_target"] is None and entry["score"] == entry["fixed"] - entry["broken"]
    final = record["final"]
    assert final["fixed_target"] is None
    assert final["score"] == final["fixed"] - final["broken"]
    assert final["delta_acc"] == pytest.approx(final["score"] / final["n_vcheck"])
    line = summary_line(record)
    assert f"fixed={final['fixed']}" in line and "fixed_target" not in line
    _check_final_rule(record)


def test_v1_evidence_checks_reject_inconsistent_instances(toys: dict[str, Toy]) -> None:
    from weightcode.baselines.fit.runner import load_evidence

    instance = toys["v1_mlp"].instance
    evidence = load_evidence(instance)
    assert evidence.rule == "evidence-v1" and evidence.n_fail == 16 and evidence.n_rand == 48
    assert np.all(evidence.base_pred[:16] != evidence.y[:16])
    data = instance.evidence
    right = int(np.flatnonzero(data["pred_rand"] == data["y_rand"])[0])
    swapped = dict(data)
    for key in ("x", "y", "pred"):
        swapped[f"{key}_fail"] = data[f"{key}_fail"].copy()
        swapped[f"{key}_fail"][0] = data[f"{key}_rand"][right]
    with pytest.raises(ValueError, match="classified correctly"):
        load_evidence(replace(instance, evidence=swapped))
    wrong_pred = dict(data, pred_rand=(data["pred_rand"] + 1) % N_CLASSES)
    with pytest.raises(ValueError, match="differ from the forward.py predictions"):
        load_evidence(replace(instance, evidence=wrong_pred))
    with pytest.raises(ValueError, match="has no target pair"):
        load_evidence(replace(instance, private={**instance.private, "target": [0, 1]}))
    with pytest.raises(ValueError, match="unknown evidence_rule"):
        load_evidence(replace(instance, private={**instance.private, "evidence_rule": "evidence-v9"}))


def test_prior_shift_moves_the_last_bias_along_the_label_shift_direction(toys: dict[str, Toy], tmp_path: Path) -> None:
    from weightcode.baselines.fit.rules import PRIOR_SHIFT_FRACTIONS, label_shift_direction

    instance = toys["v1_mlp"].instance
    run_dir = tmp_path / "prior_shift" / "0"
    record = run_prior_shift(instance, run_dir)
    assert record["status"] == "complete" and record["queries_used"] == 8 and record["n_invalid"] == 0
    assert record["flags"]["uses_labels_for_fit"] is True and record["flags"]["respects_edit_limits"] is True
    data = instance.evidence
    expected = label_shift_direction(instance.forward(dict(instance.params), data["x_rand"]), data["y_rand"])
    direction = np.asarray(record["generation"]["direction"])
    np.testing.assert_allclose(direction, expected["direction"], rtol=1e-12, atol=1e-12)
    assert abs(float(direction.sum())) < 1e-9
    base_bias = instance.params["fc3.bias"].astype(np.float64)
    for entry, fraction in zip(record["candidates"], PRIOR_SHIFT_FRACTIONS, strict=True):
        assert entry["config"]["fraction"] == fraction and entry["roundtrip_exact"]
        assert {op["tensor"] for op in _ops(run_dir, entry["slot"])} == {"fc3.bias"}
        assert entry["edit"]["edit_size"] <= N_CLASSES
        assert sum(stats["changed"] for stats in entry["edit"]["per_tensor"].values()) == entry["edit"]["n_changed"]
        child = apply_patch_path(instance.params, run_dir / entry["patch_id"] / "patch.json", instance.params_sha256,
                                 enforce_limits=True)
        np.testing.assert_allclose(child["fc3.bias"] - base_bias, fraction * direction, atol=1e-6)
    gaps = [entry["notes"]["marginal_gap_l1_after"] for entry in record["candidates"]]
    assert gaps[0] < record["generation"]["marginal_gap_l1_base"]
    _check_final_rule(record)
    # The toy's defect is a raised final bias of one class, a prior-like shift that this rule should partly undo.
    assert record["final"]["score"] > 0


@pytest.mark.parametrize(("key", "tensor", "n_positions", "group_period"), [
    ("v1_mlp", "fc1.weight", 3 * 64, 64),
    ("v1_cnn", "conv1.weight", 3 * 9, 9),
])
def test_tie_inputs_writes_one_tie_op_at_admissible_strengths(
    toys: dict[str, Toy], tmp_path: Path, key: str, tensor: str, n_positions: int, group_period: int
) -> None:
    instance = toys[key].instance
    run_dir = tmp_path / "tie_inputs" / "0"
    record = run_tie(instance, run_dir)
    generation = record["generation"]
    assert record["status"] == "complete" and record["n_invalid"] == 0
    assert record["flags"]["uses_labels_for_fit"] is False and record["flags"]["respects_edit_limits"] is True
    assert generation["plan"]["tensor"] == tensor and generation["plan"]["n_positions"] == n_positions
    assert generation["planned_strengths"] == [0.5, 1.0]
    full = generation["rel_frob_full_tie"]
    assert generation["s_max"] == pytest.approx(min(1.0, MAX_REL_FROB * (1 - 1e-4) / full), rel=1e-12)
    strengths = generation["strengths"]
    assert strengths == ([0.5, generation["s_max"]] if generation["s_max"] > 0.5 else [generation["s_max"]])
    assert record["queries_used"] == len(strengths)
    for entry, strength in zip(record["candidates"], strengths, strict=True):
        (op,) = _ops(run_dir, entry["slot"])
        assert op == {"op": "tie", "tensor": tensor, "axis": 1,
                      "groups": (np.arange(n_positions) % group_period).tolist(), "strength": strength}
        assert entry["patch_form"] == "ops" and entry["roundtrip_exact"]
        assert entry["edit"]["edit_size"] == n_positions and entry["edit"]["edit_size_source"] == "executor"
        assert entry["edit"]["per_tensor"][tensor]["rel_frob"] <= MAX_REL_FROB
        assert entry["edit"]["per_tensor"][tensor]["rel_frob"] == pytest.approx(strength * full, rel=1e-5)
        assert entry["edit"]["n_changed"] == entry["edit"]["per_tensor"][tensor]["changed"]
    _check_final_rule(record)


@pytest.mark.parametrize("key", ["v1_mlp", "v1_cnn"])
def test_full_tie_makes_the_first_layer_blind_to_channel_order(toys: dict[str, Toy], key: str) -> None:
    from weightcode.baselines.fit.rules import tie_plan
    from weightcode.patch import Limits, apply_patch

    instance = toys[key].instance
    spec = _spec(instance)
    plan = tie_plan(architecture_from(instance.params, spec), instance.params, spec)
    patch = {"schema": "weightcode.patch.v0", "parent_sha256": instance.params_sha256, "hypothesis": "h",
             "predicted_effect": "p",
             "ops": [{"op": "tie", "tensor": plan.tensor, "axis": 1, "groups": plan.groups.tolist(), "strength": 1.0}]}
    tied = apply_patch(instance.params, patch, None, Limits().without_size_limits(), instance.params_sha256).child
    x = instance.evidence["x_rand"]
    permuted = x[:, [2, 0, 1]]
    def channel_order_gap(params) -> float:
        return float(np.abs(instance.forward(dict(params), x) - instance.forward(dict(params), permuted)).max())

    tied_gap, base_gap = channel_order_gap(tied), channel_order_gap(instance.params)
    assert tied_gap < 1e-4 < base_gap


def test_tie_inputs_is_not_applicable_to_one_input_channel(toys: dict[str, Toy], tmp_path: Path) -> None:
    instance = toys["mlp"].instance
    run_dir = tmp_path / "tie_inputs" / "0"
    record = run_tie(instance, run_dir)
    assert record["status"] == "complete" and "one channel" in record["generation"]["not_applicable"]
    assert record["queries_used"] == 0 and record["final"]["is_base_noop"]
    assert sorted(path.name for path in run_dir.iterdir()) == ["run.json"]


def test_executor_rejection_is_recorded_as_an_invalid_patch(toys: dict[str, Toy], tmp_path: Path) -> None:
    from weightcode.baselines.fit.runner import CandidatePatch, MethodSpec, execute_run

    instance = toys["v1_mlp"].instance
    over = dict(instance.params)
    over["fc2.weight"] = (instance.params["fc2.weight"] * np.float32(2.0)).astype(np.float32)
    fine = dict(instance.params)
    fine["fc3.bias"] = (instance.params["fc3.bias"] + np.float32(0.01)).astype(np.float32)
    spec = MethodSpec(method="probe", label="test", uses_labels_for_fit=False, edit_limits_enforced=True)

    def generate(evidence):
        return [CandidatePatch(slot=0, child=over, config={}, hypothesis="h", predicted_effect="p"),
                CandidatePatch(slot=1, child=fine, config={}, hypothesis="h", predicted_effect="p")], {}

    record = execute_run(instance=instance, spec=spec, run_dir=tmp_path / "probe" / "0", run_seed=0, config={},
                         generate=generate)
    rejected, accepted = record["candidates"]
    assert rejected["invalid_stage"] == "executor" and "above the limit" in rejected["invalid"]
    assert rejected["invalid"] == "\n".join(rejected["executor_messages"]) and "check" not in rejected
    assert accepted["check"]["query_index"] == 0
    assert record["queries_used"] == 1 and record["n_invalid"] == 1


def test_cli_runs_the_fixed_rule_arms_on_v1_instances(toys: dict[str, Toy], tmp_path: Path) -> None:
    env, runs = _cli_env(toys, tmp_path)
    result = _run_script(env, "scripts/run_fit_baselines.py", "--dataset", "toy_v1", "--methods", "prior_shift",
                         "tie_inputs")
    assert result.returncode == 0, result.stderr
    for key in ("v1_mlp", "v1_cnn"):
        instance_id = toys[key].instance.instance_id
        for method in ("prior_shift", "tie_inputs"):
            record = read_run_record(runs / "baselines" / instance_id / method / "0" / "run.json")
            assert record["status"] == "complete" and record["instance"]["evidence_rule"] == "evidence-v1"
            assert record["final"]["fixed_target"] is None
    assert "fixed=" in result.stdout and "delta_acc=" in result.stdout


# ------------------------------------------------------------------------------------- harness-style references


def _small_edit(instance: Instance) -> dict[str, np.ndarray]:
    child = dict(instance.params)
    child["fc3.bias"] = (instance.params["fc3.bias"] + np.float32(0.25)).astype(np.float32)
    weight = instance.params["fc2.weight"].copy()
    weight[0, :5] += np.float32(0.05)
    child["fc2.weight"] = weight
    return child


def _harness_session(
    root: Path, instance: Instance, child: dict[str, np.ndarray] | None, *, parent: str, session_id: str
) -> Path:
    """A synthetic editor-session root holding run.json in the shared format, as weightcode.harness.finalize writes it."""
    session = root / session_id
    final_dir = session / "final"
    final_dir.mkdir(parents=True)
    noop = child is None
    chosen = dict(instance.params) if noop else child
    model_sha = save_params(final_dir / "model.npz", chosen)
    if not noop and parent == "instance":
        write_delta_patch(final_dir, instance.params_sha256, instance.params, chosen, "h", "p")
    final = {
        "is_base_noop": noop, "dir": None if noop else "final", "parent": parent, "model": "final/model.npz",
        "child_sha256": model_sha, "score": 0 if noop else 3, "fixed": 0 if noop else 5, "broken": 0 if noop else 2,
        "edit_size": 0, "edit_frac": 0.0, "n_changed": 0, "patch_id": None if noop else "p3", "query_index": None,
        "kind": "base" if noop else "checked", "editor_patch": None, "reason": "test",
    }
    run = {
        "schema": "weightcode.editor_run.v0", "method": "M2", "status": "complete", "session_id": session_id, "repeat": 0,
        "instance": {"instance_id": instance.instance_id},
        "flags": {"edit_limits_enforced": True, "infra": False, "audit_verdict": None, "audit_status": "skipped"},
        "final": final, "queries_used": 4, "budget": 8,
    }
    (session / "run.json").write_text(json.dumps(run))
    return session


@pytest.mark.parametrize("parent", ["instance", "decoy_readdressed"])
def test_controls_take_a_harness_session_run_record(toys: dict[str, Toy], tmp_path: Path, parent: str) -> None:
    from weightcode.baselines.fit.controls import FINAL_CHILD_REBUILDER, load_reference, resolve_reference

    instance = toys["mlp"].instance
    child = _small_edit(instance)
    session_id = "s0123456789abcdef"
    session = _harness_session(tmp_path / "sessions", instance, child, parent=parent, session_id=session_id)
    resolved = resolve_reference(session, instance)
    assert resolved.kind == "run_record" and resolved.arm == "M2"
    assert resolved.session_id(instance.instance_id) == session_id
    assert resolved.output_label() == f"M2_{session_id}"
    reference = load_reference(instance, session / "run.json")
    assert reference.child_sha256 == params_sha256(child) and reference.applier == FINAL_CHILD_REBUILDER
    for key in instance.params:
        np.testing.assert_array_equal(
            reference.delta[key], child[key].astype(np.float64) - instance.params[key].astype(np.float64)
        )
    run_dir = tmp_path / "gaussian" / resolved.output_label() / "r2"
    record = run_control(instance, "gaussian", session, session_id=session_id, arm="M2", run_index=2, run_dir=run_dir)
    assert record["status"] == "complete" and record["queries_used"] == N_DRAWS
    assert record["run_seed"] == control_run_seed(session_id, "gaussian", 2)
    described = record["generation"]["reference"]
    assert described["kind"] == "run_record" and described["child_sha256"] == params_sha256(child)
    assert described["run_record"]["schema"] == "weightcode.editor_run.v0" and described["run_record"]["method"] == "M2"
    for entry in record["candidates"]:
        drawn = apply_patch_path(instance.params, run_dir / entry["patch_id"] / "patch.json", instance.params_sha256,
                                 enforce_limits=False)
        for key in instance.params:
            assert np.array_equal(drawn[key] != instance.params[key], child[key] != instance.params[key])
    _check_final_rule(record)


def test_harness_references_are_checked(toys: dict[str, Toy], tmp_path: Path) -> None:
    from weightcode.baselines.fit.controls import load_reference
    from weightcode.evalkit.devreadout import DevReadoutError

    instance = toys["mlp"].instance
    other = toys["cnn"].instance
    noop = _harness_session(tmp_path, instance, None, parent="instance", session_id="s00000000000000aa")
    run_dir = tmp_path / "permuted" / "noop" / "r0"
    record = run_control(instance, "permuted", noop, session_id="s00000000000000aa", arm="M2", run_index=0,
                         run_dir=run_dir)
    assert record["generation"]["reference_is_noop"] is True and record["queries_used"] == 0
    assert sorted(path.name for path in run_dir.iterdir()) == ["run.json"]
    with pytest.raises(ValueError, match="belongs to instance"):
        load_reference(other, noop)
    tampered = _harness_session(tmp_path, instance, _small_edit(instance), parent="instance",
                                session_id="s00000000000000bb")
    run = json.loads((tampered / "run.json").read_text())
    run["final"]["child_sha256"] = instance.params_sha256
    (tampered / "run.json").write_text(json.dumps(run))
    with pytest.raises(DevReadoutError, match="recorded final child"):
        load_reference(instance, tampered)
    unfinished = tmp_path / "unfinished"
    unfinished.mkdir()
    (unfinished / "run.json").write_text(json.dumps({"instance": {"instance_id": instance.instance_id},
                                                     "status": "failed"}))
    with pytest.raises(ValueError, match="no final pick"):
        load_reference(instance, unfinished)


def test_cli_controls_of_a_harness_session(toys: dict[str, Toy], tmp_path: Path) -> None:
    env, runs = _cli_env(toys, tmp_path)
    instance = toys["mlp"].instance
    session_id = "s00000000000000cc"
    session = _harness_session(tmp_path / "sessions", instance, _small_edit(instance), parent="instance",
                               session_id=session_id)
    result = _run_script(env, "scripts/run_random_controls.py", "--instance", instance.instance_id,
                         "--reference-patch", str(session), "--kinds", "gaussian", "--runs", "2")
    assert result.returncode == 0, result.stderr
    label = f"M2_{session_id}"
    for run_index in range(2):
        record = read_run_record(runs / "baselines" / instance.instance_id / "gaussian" / label / f"r{run_index}"
                                 / "run.json")
        assert record["config"]["b1"]["arm"] == "M2" and record["config"]["b1"]["session_id"] == session_id
        assert record["run_seed"] == control_run_seed(session_id, "gaussian", run_index)
    assert f"B1 {instance.instance_id} gaussian[{label}] session {session_id}" in result.stdout
    mislabeled = _run_script(env, "scripts/run_random_controls.py", "--instance", instance.instance_id,
                             "--reference-patch", str(session), "--ref-label", "M2")
    assert mislabeled.returncode == 2 and "differs" in mislabeled.stderr


# --------------------------------------------------------------------------------------------------- input_stats


def _conv_same_padded(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: np.ndarray) -> np.ndarray:
    out_c, in_c, k, _ = w.shape
    p = (k - 1) // 2
    n, _, h, wd = x.shape
    xp = np.empty((n, in_c, h + 2 * p, wd + 2 * p))
    xp[:] = pad[None, :, None, None]
    xp[:, :, p:p + h, p:p + wd] = x
    cols = np.stack([xp[:, :, i:i + h, j:j + wd] for i in range(k) for j in range(k)], axis=2)
    return np.einsum("nckhw,ock->nohw", cols, w.reshape(out_c, in_c, k * k)) + b[None, :, None, None]


def _pool2(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    x = x[:, :, : h - h % 2, : w - w % 2]
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def _preprocessed(instance: Instance, x_uint8: np.ndarray) -> np.ndarray:
    spec = _spec(instance)
    shape = tuple(spec["input"]["shape"])
    mean, std = np.asarray(spec["input"]["mean"]), np.asarray(spec["input"]["std"])
    x = x_uint8.reshape(len(x_uint8), *shape).astype(np.float64) / 255.0
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


def _k_form_logits(
    instance: Instance, x_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray, t: float, *, shifted_padding: bool
) -> np.ndarray:
    """float64 toy forward of the base model on x'' = k (x' - t m), k = 1 + t (1/s - 1): the unfolded transform.

    ``shifted_padding``: the first conv pads with k_c (0 - t m_c), the value the folded first bias implies, instead
    of 0 (later layers always pad with 0).
    """
    k = 1.0 + t * (1.0 / std - 1.0)
    x = k[None, :, None, None] * (_preprocessed(instance, x_uint8) - t * mean[None, :, None, None])
    params = {key: value.astype(np.float64) for key, value in instance.params.items()}
    n_conv = sum(1 for key in params if key.startswith("conv") and key.endswith(".weight"))
    n_fc = sum(1 for key in params if key.startswith("fc") and key.endswith(".weight"))
    for i in range(1, n_conv + 1):
        weight = params[f"conv{i}.weight"]
        pad = k * (0.0 - t * mean) if (i == 1 and shifted_padding) else np.zeros(weight.shape[1])
        x = _pool2(np.maximum(_conv_same_padded(x, weight, params[f"conv{i}.bias"], pad), 0.0))
    x = x.reshape(len(x), -1)
    for i in range(1, n_fc + 1):
        x = x @ params[f"fc{i}.weight"].T + params[f"fc{i}.bias"]
        if i < n_fc:
            x = np.maximum(x, 0.0)
    return x


@pytest.mark.parametrize("key", ["v1_mlp", "v1_cnn", "mlp"])
def test_input_stats_fold_equals_the_direct_k_form(toys: dict[str, Toy], key: str) -> None:
    from weightcode.baselines.fit.rules import input_stats_fold
    from weightcode.patch import Limits, apply_patch

    instance = toys[key].instance
    arch = architecture_from(instance.params, _spec(instance))
    channels = arch.input_shape[0]
    # Statistics far from the identity, so that k, g and r all differ from 1 and between channels.
    mean = np.array([0.3, -0.6, 0.15][:channels])
    std = np.array([0.5, 2.0, 1.3][:channels])
    x = np.random.default_rng(11).integers(0, 256, size=(200, *arch.input_shape), dtype=np.uint8)
    for t in (0.5, 1.0):
        fold = input_stats_fold(arch, instance.params, mean, std, t)
        k = 1.0 + t * (1.0 / std - 1.0)
        np.testing.assert_allclose(fold.k, k, rtol=1e-15)
        assert fold.g == pytest.approx(float(np.prod(k) ** (1.0 / channels)), rel=1e-12)
        np.testing.assert_allclose(fold.r, k / fold.g, rtol=1e-15)
        patch = {"schema": "weightcode.patch.v0", "parent_sha256": instance.params_sha256, "hypothesis": "h",
                 "predicted_effect": "p", "ops": fold.ops}
        result = apply_patch(instance.params, patch, None, Limits().without_size_limits(), instance.params_sha256)
        assert result.child_sha256 == params_sha256(fold.child), "the expected child must match the executor"
        folded = fold.g * instance.forward(dict(result.child), x).astype(np.float64)
        direct = _k_form_logits(instance, x, mean, std, t, shifted_padding=True)
        scale = float(np.abs(direct).max())
        np.testing.assert_allclose(folded, direct, rtol=0, atol=2e-5 * scale)
        assert np.array_equal(np.argmax(folded, axis=1), np.argmax(direct, axis=1))
        unshifted = _k_form_logits(instance, x, mean, std, t, shifted_padding=False)
        if arch.family == "cnn":
            assert float(np.abs(unshifted - direct).max()) > 1e-3 * scale, "the padding deviation must be visible"
        else:
            np.testing.assert_array_equal(unshifted, direct)
        assert fold.rel_frob_admissible == pytest.approx(
            float(np.linalg.norm(fold.child[fold.first_weight].astype(np.float64)
                                 - instance.params[fold.first_weight].astype(np.float64))
                  / np.linalg.norm(instance.params[fold.first_weight].astype(np.float64))), rel=1e-5)
        if channels == 1:
            assert fold.r.tolist() == [1.0] and fold.rel_frob_admissible == 0.0
            assert all(op["op"] != "scale_cols" for op in fold.ops)


@pytest.mark.parametrize("key", ["v1_mlp", "v1_cnn", "mlp"])
def test_input_stats_end_to_end(toys: dict[str, Toy], tmp_path: Path, key: str) -> None:
    from weightcode.baselines.fit.rules import CONV_PADDING_DEVIATION, INPUT_STATS_STRENGTHS, run_input_stats

    instance = toys[key].instance
    arch = architecture_from(instance.params, _spec(instance))
    channels, height, width = arch.input_shape
    run_dir = tmp_path / "input_stats" / "0"
    record = run_input_stats(instance, run_dir)
    assert record["status"] == "complete" and record["n_invalid"] == 0 and record["queries_used"] == 2
    assert record["flags"] == {"uses_labels_for_fit": False, "edit_limits_enforced": True, "respects_edit_limits": True}
    generation = record["generation"]
    pre = _preprocessed(instance, instance.evidence["x_rand"])
    np.testing.assert_allclose(generation["channel_mean"], pre.mean(axis=(0, 2, 3)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(generation["channel_std"], pre.std(axis=(0, 2, 3)), rtol=1e-12)
    assert generation["n_images"] == 48
    expected_deviations = [CONV_PADDING_DEVIATION] if arch.family == "cnn" else []
    assert generation["deviations"] == expected_deviations and record["deviations"] == expected_deviations
    first = "fc1" if arch.family == "mlp" else "conv1"
    later = [f"{layer}.bias" for layer in (*arch.conv_layers, *arch.linear_layers) if layer != first]
    std = np.asarray(generation["channel_std"])
    for entry, t in zip(record["candidates"], INPUT_STATS_STRENGTHS, strict=True):
        notes = entry["notes"]
        k = 1.0 + t * (1.0 / std - 1.0)
        np.testing.assert_allclose(notes["k"], k, rtol=1e-12)
        np.testing.assert_allclose(notes["r"], k / notes["g"], rtol=1e-12)
        assert notes["g"] == pytest.approx(float(np.exp(np.mean(np.log(k)))), rel=1e-12)
        ops = _ops(run_dir, entry["slot"])
        scale_cols = [op for op in ops if op["op"] == "scale_cols"]
        assert [op["factor"] for op in scale_cols] == [r for r in notes["r"] if r != 1.0]
        if channels == 1:
            assert scale_cols == [] and notes["r"] == [1.0]
        elif arch.family == "mlp":
            assert [op["cols"] for op in scale_cols] == [
                list(range(c * height * width, (c + 1) * height * width)) for c in range(channels)
            ]
        else:
            assert [op["cols"] for op in scale_cols] == [[c] for c in range(channels)]
        (bias_op,) = [op for op in ops if op["op"] == "set"]
        assert bias_op["tensor"] == f"{first}.bias"
        rows = [op for op in ops if op["op"] == "scale_rows"]
        assert [op["tensor"] for op in rows] == later and all(op["factor"] == 1.0 / notes["g"] for op in rows)
        size = sum(len(op["cols"]) for op in scale_cols) + len(bias_op["index"]) + sum(len(op["rows"]) for op in rows)
        assert entry["edit"]["edit_size"] == size and entry["roundtrip_exact"]
        assert entry["edit"]["per_tensor"][f"{first}.weight"]["rel_frob"] == pytest.approx(
            notes["rel_frob_first_weight_admissible"], rel=1e-5, abs=1e-12)
        assert notes["rel_frob_first_weight_k_form"] >= notes["rel_frob_first_weight_admissible"] - 1e-12
    _check_final_rule(record)


def test_a_control_run_whose_draws_are_all_rejected_fails(
    toys: dict[str, Toy], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import weightcode.baselines.fit.runner as runner
    from weightcode.baselines.fit.controls import b1_value
    from weightcode.baselines.fit.runner import NoValidCandidate
    from weightcode.patch import PatchError

    instance = toys["mlp"].instance
    session = _harness_session(tmp_path / "sessions", instance, _small_edit(instance), parent="instance",
                               session_id="s00000000000000dd")

    def reject(*args, **kwargs):
        raise PatchError(["patch.json is 2232320 bytes, above the 2097152-byte limit"])

    monkeypatch.setattr(runner, "apply_patch_result", reject)
    run_dir = tmp_path / "permuted" / "M2" / "r0"
    with pytest.raises(NoValidCandidate, match="2097152-byte limit"):
        run_control(instance, "permuted", session, session_id="s00000000000000dd", arm="M2", run_index=0,
                    run_dir=run_dir)
    record = read_run_record(run_dir / "run.json")
    assert record["status"] == "failed" and "all 8 candidates were invalid" in record["error"]
    assert "2097152-byte limit" in record["error"] and "final" not in record
    with pytest.raises(ValueError, match="incomplete control runs"):
        b1_value([record])
