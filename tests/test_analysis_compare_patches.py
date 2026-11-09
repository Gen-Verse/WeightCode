"""Tests for the dev-only run comparison against a channel tie (weightcode.analysis.compare_patches)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import ToyStore, build_v1, fixing_child, toy_store  # noqa: F401

import weightcode.common.paths
import weightcode.evalkit.sealed
from weightcode.analysis import compare_patches as cp
from weightcode.baselines.instance_io import Instance, load_instance, write_delta_patch
from weightcode.common.identity import params_sha256
from weightcode.evalkit import devreadout
from weightcode.evalkit.patch_check import predict
from weightcode.patch import SCHEMA, Limits, apply_patch

# ----------------------------------------------------------------------------------------------- synthetic 3-channel model

C, H, W = 3, 4, 4
SPEC = {
    "input": {"shape": [C, H, W], "layout": "NCHW, flatten row-major"},
    "layers": [
        {"op": "flatten", "order": "C, H, W (row-major)"},
        {"name": "fc1", "op": "linear", "weight": "fc1.weight", "bias": "fc1.bias", "shape_w": [6, C * H * W]},
        {"op": "relu"},
        {"name": "fc2", "op": "linear", "weight": "fc2.weight", "bias": "fc2.bias", "shape_w": [4, 6]},
    ],
}


def _forward(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x.reshape(len(x), -1).astype(np.float32) / np.float32(255.0)
    h = np.maximum(h @ params["fc1.weight"].T + params["fc1.bias"], np.float32(0.0))
    return (h @ params["fc2.weight"].T + params["fc2.bias"]).astype(np.float32)


def _params() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "fc1.weight": rng.normal(size=(6, C * H * W)).astype(np.float32),
        "fc1.bias": rng.normal(scale=0.1, size=6).astype(np.float32),
        "fc2.weight": rng.normal(size=(4, 6)).astype(np.float32),
        "fc2.bias": np.zeros(4, dtype=np.float32),
    }


def _tie_child(params: dict[str, np.ndarray], strength: float) -> dict[str, np.ndarray]:
    """The executor's channel tie: every input position is grouped with the same pixel of the other channels."""
    groups = [p % (H * W) for p in range(C * H * W)]
    patch = {"schema": SCHEMA, "parent_sha256": params_sha256(params), "hypothesis": "tie", "predicted_effect": "tie",
             "ops": [{"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": groups, "strength": strength}]}
    return apply_patch(params, patch, None, Limits().without_size_limits(), params_sha256(params)).child


def test_pure_tie_child_is_recovered_exactly() -> None:
    base = _params()
    child = _tie_child(base, 0.3)
    layer = cp.first_layer(SPEC, base)
    assert layer.view_shape == (6, C, H, W) and layer.tensor == "fc1.weight"
    tie = cp.tie_geometry(base["fc1.weight"], child["fc1.weight"], layer)
    assert tie["s_eff"] == pytest.approx(0.3, abs=1e-6)
    assert tie["tie_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert tie["channel_spread_ratio"] == pytest.approx(0.7, abs=1e-6)
    assert tie["cos_delta_minus_d"] == pytest.approx(1.0, abs=1e-6) and tie["flags"] == []
    space = cp.weight_space(base, child, layer)
    assert space["first_layer_share_of_squared_delta"] == 1.0

    tie_part, residual, s_eff = cp.decompose(base, child, layer)
    assert s_eff == tie["s_eff"]
    assert all(tie_part[k].dtype == np.float32 and residual[k].dtype == np.float32 for k in base)
    assert np.allclose(residual["fc1.weight"], base["fc1.weight"], rtol=0, atol=1e-6)
    for key in ("fc1.bias", "fc2.weight", "fc2.bias"):
        assert np.array_equal(tie_part[key], base[key]) and np.array_equal(residual[key], child[key])
    x = np.random.default_rng(8).integers(0, 256, size=(400, C, H, W), dtype=np.uint8)
    child_pred = predict(_forward, child, x, 64)
    assert np.array_equal(predict(_forward, tie_part, x, 64), child_pred)
    assert np.array_equal(predict(_forward, residual, x, 64), predict(_forward, base, x, 64))


def test_channel_common_delta_has_no_tie_component() -> None:
    base = _params()
    common = np.random.default_rng(9).normal(scale=0.1, size=(6, 1, H, W))
    delta = np.broadcast_to(common, (6, C, H, W)).reshape(6, -1)
    child = dict(base, **{"fc1.weight": (base["fc1.weight"].astype(np.float64) + delta).astype(np.float32)})
    tie = cp.tie_geometry(base["fc1.weight"], child["fc1.weight"], cp.first_layer(SPEC, base))
    assert tie["s_eff"] == pytest.approx(0.0, abs=1e-6)
    assert tie["tie_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert tie["channel_spread_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_zero_delta_gives_a_flagged_zero_tie_fraction() -> None:
    base = _params()
    tie = cp.tie_geometry(base["fc1.weight"], base["fc1.weight"], cp.first_layer(SPEC, base))
    assert tie["s_eff"] == 0.0 and tie["tie_fraction"] == 0.0 and tie["cos_delta_minus_d"] is None
    assert tie["flags"] == [cp.FLAG_DELTA_ZERO] and tie["channel_spread_ratio"] == 1.0


def _masked_tie_child(base: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """A full channel tie (s = 1) where ``mask`` holds, none elsewhere: all tie, but uneven."""
    layer = cp.first_layer(SPEC, base)
    w64 = base["fc1.weight"].astype(np.float64).reshape(layer.view_shape)
    d = cp.channel_difference(base["fc1.weight"], layer)
    return dict(base, **{"fc1.weight": (w64 - np.where(mask, 1.0, 0.0) * d).reshape(6, -1).astype(np.float32)})


@pytest.mark.parametrize("where", ["units", "pixels"])
def test_an_uneven_tie_reads_as_a_tie_in_the_local_fits(where: str) -> None:
    base = _params()
    layer = cp.first_layer(SPEC, base)
    mask = np.arange(6)[:, None, None, None] < 3 if where == "units" else np.arange(H)[None, None, :, None] < 2
    child = _masked_tie_child(base, mask)
    tie = cp.tie_geometry(base["fc1.weight"], child["fc1.weight"], layer)
    # The uniform global fit sees a partial tie; the local fits see all of it.
    assert tie["fit"] == "uniform global tie" and tie["tie_fraction"] < 0.9
    assert tie["channel_difference_share"] == pytest.approx(1.0, abs=1e-6)
    fit = "per_unit" if where == "units" else "per_spatial"
    for name in (fit, "per_unit_spatial"):
        local = tie["local"][name]
        assert local["explained_fraction"] == pytest.approx(1.0, abs=1e-6)
        assert local["explained_fraction_tie_sign"] == pytest.approx(1.0, abs=1e-6)
        assert local["s_quantiles"]["q05"] == pytest.approx(0.0, abs=1e-6)
        assert local["s_quantiles"]["q95"] == pytest.approx(1.0, abs=1e-6)
    assert tie["local"]["per_unit_spatial"]["isotropic_chance"] == pytest.approx(1 / C)
    tie_local, residual_local = cp.decompose_local(base, child, layer)
    assert np.allclose(tie_local["fc1.weight"], child["fc1.weight"], rtol=0, atol=1e-6)
    assert np.allclose(residual_local["fc1.weight"], base["fc1.weight"], rtol=0, atol=1e-6)
    x = np.random.default_rng(8).integers(0, 256, size=(400, C, H, W), dtype=np.uint8)
    assert np.array_equal(predict(_forward, tie_local, x, 64), predict(_forward, child, x, 64))
    assert np.array_equal(predict(_forward, residual_local, x, 64), predict(_forward, base, x, 64))


def test_an_anti_tie_is_flagged_and_signed() -> None:
    base = _params()
    layer = cp.first_layer(SPEC, base)
    d = cp.channel_difference(base["fc1.weight"], layer).reshape(6, -1)
    child = dict(base, **{"fc1.weight": (base["fc1.weight"].astype(np.float64) + 0.3 * d).astype(np.float32)})
    tie = cp.tie_geometry(base["fc1.weight"], child["fc1.weight"], layer)
    assert tie["s_eff"] == pytest.approx(-0.3, abs=1e-6) and tie["channel_spread_ratio"] == pytest.approx(1.3, abs=1e-6)
    assert tie["tie_fraction"] == pytest.approx(1.0, abs=1e-6)  # cos^2: sign-blind by definition
    assert tie["signed_tie_fraction"] == pytest.approx(-1.0, abs=1e-6) and cp.FLAG_ANTI_TIE in tie["flags"]
    local = tie["local"]["per_unit_spatial"]
    assert local["explained_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert local["explained_fraction_tie_sign"] == 0.0 and local["share_groups_negative"] == 1.0
    tie_local, _ = cp.decompose_local(base, child, layer)
    assert np.array_equal(tie_local["fc1.weight"], base["fc1.weight"])  # no tie-signed part to keep
    assert cp.FLAG_ANTI_TIE not in cp.tie_geometry(base["fc1.weight"], _tie_child(base, 0.3)["fc1.weight"], layer)["flags"]


def test_first_layer_is_resolved_from_the_spec_or_refused() -> None:
    conv_params = {"conv1.weight": np.zeros((5, 3, 3, 3), np.float32), "conv1.bias": np.zeros(5, np.float32)}
    conv_spec = {"input": {"shape": [3, 8, 8]}, "layers": [{"name": "conv1", "op": "conv2d", "weight": "conv1.weight",
                                                            "bias": "conv1.bias", "shape_w": [5, 3, 3, 3]}]}
    assert cp.first_layer(conv_spec, conv_params).view_shape == (5, 3, 3, 3)
    bad_channels = json.loads(json.dumps(conv_spec))
    bad_channels["input"]["shape"] = [1, 8, 8]
    with pytest.raises(cp.CompareError, match="input channels"):
        cp.first_layer(bad_channels, conv_params)
    wrong_flat = json.loads(json.dumps(SPEC))
    wrong_flat["input"]["shape"] = [3, 4, 5]
    with pytest.raises(cp.CompareError, match="flat input"):
        cp.first_layer(wrong_flat, _params())
    wrong_order = json.loads(json.dumps(SPEC))
    wrong_order["layers"][0]["order"] = "H, W, C"
    with pytest.raises(cp.CompareError, match="flatten"):
        cp.first_layer(wrong_order, _params())
    with pytest.raises(cp.CompareError, match="no layer with a weight"):
        cp.first_layer({"input": {"shape": [3, 4, 4]}, "layers": [{"op": "relu"}]}, _params())


# ----------------------------------------------------------------------------------------------- toy store runs


def _instances(store: ToyStore) -> dict[str, Instance]:
    root = store.private / "instances"
    return {f"{o.split_role}{o.seed}": load_instance(root / o.instance_id) for o in build_v1(store)}


def _write_run(run_dir: Path, instance_id: str, final: dict[str, object]) -> Path:
    """A run record; the toy fixing child is larger than the edit limits allow, so it is run as a label-fit reference."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run = {"instance": {"instance_id": instance_id}, "method": "toy", "flags": {"edit_limits_enforced": False},
           "final": final}
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir / "run.json"


def _patch_run(store: ToyStore, instance: Instance, name: str, child: dict[str, np.ndarray]) -> Path:
    run_dir = store.runs / "toy" / instance.instance_id / name
    write_delta_patch(run_dir / "q1", instance.params_sha256, instance.params, child, "toy", "toy")
    final = {"is_base_noop": False, "dir": "q1", "score": 1, "fixed": 1, "broken": 0, "edit_size": 1,
             "child_sha256": params_sha256(child)}
    return _write_run(run_dir, instance.instance_id, final)


def _noop_run(store: ToyStore, instance: Instance) -> Path:
    final = {"is_base_noop": True, "dir": None, "score": 0, "fixed": 0, "broken": 0, "edit_size": 0,
             "child_sha256": instance.params_sha256}
    return _write_run(store.runs / "toy" / instance.instance_id / "noop", instance.instance_id, final)


def test_a_run_compared_with_itself(toy_store: ToyStore, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    dev = _instances(toy_store)["dev100"]
    run = _patch_run(toy_store, dev, "fix", fixing_child(dict(dev.params)))
    record = cp.compare_runs(run.parent, run)
    fs = record["function_space"]
    assert fs["fixed"]["run"] > 0 and fs["fixed"]["jaccard"] == 1.0 and fs["fixed"]["intersection_over_run"] == 1.0
    assert fs["agreement_run_ref"] == {"all": 1.0, "base_errors": 1.0, "n_base_errors": fs["agreement_run_ref"]["n_base_errors"]}
    assert fs["paired_run_minus_ref"]["difference"] == 0.0
    expected = devreadout.dev_readout(dev, fixing_child(dict(dev.params)))["readouts"]
    assert fs["run"]["readouts"] == expected and fs["ref"]["readouts"] == expected
    cos = record["weight_space"]["cosine_run_ref"]
    assert cos["all_parameters"] == pytest.approx(1.0) and cos["first_layer"] == pytest.approx(1.0)
    # The toy model has one input channel, so D is zero and the tie direction is empty (flagged, not silent).
    tie = record["weight_space"]["run"]["tie"]
    assert cp.FLAG_CHANNEL_DIFFERENCE_ZERO in tie["flags"] and tie["s_eff"] == 0.0 and tie["tie_fraction"] == 0.0
    assert record["decomposition"]["agreement_residual_with_base"] < 1.0
    assert record["op_summary"]["applicable"] and record["op_summary"]["by_op"] == {"add": 2}

    out = toy_store.runs / "compare.json"
    assert cp.main(["--run", str(run), "--reference", str(run.parent), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["instance_id"] == dev.instance_id
    assert "s_eff" in capsys.readouterr().out


def test_main_instance_is_refused(toy_store: ToyStore) -> None:  # noqa: F811
    main = _instances(toy_store)["main0"]
    run = _noop_run(toy_store, main)
    with pytest.raises(devreadout.DevReadoutError, match="dev only"):
        cp.compare_runs(run, run)


def test_mismatched_instance_ids_are_refused(toy_store: ToyStore) -> None:  # noqa: F811
    instances = _instances(toy_store)
    with pytest.raises(cp.CompareError, match="same one"):
        cp.compare_runs(_noop_run(toy_store, instances["dev100"]), _noop_run(toy_store, instances["dev101"]))


_FORBIDDEN_ROOTS: list[str] = []


def _guard_sealed(event: str, args: tuple[object, ...]) -> None:
    if not _FORBIDDEN_ROOTS or event not in ("open", "os.listdir", "os.scandir"):
        return
    target = args[0] if args else None
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    path = os.path.realpath(os.fsdecode(target))
    for root in _FORBIDDEN_ROOTS:
        if path == root or path.startswith(root + os.sep):
            raise PermissionError(f"test guard: {event} on the sealed store {path}")


sys.addaudithook(_guard_sealed)


def _raise(*_: object, **__: object) -> None:
    raise AssertionError("a sealed reader was called")


def test_the_sealed_store_is_never_opened(toy_store: ToyStore, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    dev = _instances(toy_store)["dev100"]
    run = _patch_run(toy_store, dev, "fix", fixing_child(dict(dev.params)))
    ref = _noop_run(toy_store, dev)
    shutil.rmtree(toy_store.sealed / toy_store.dataset)
    monkeypatch.setattr(weightcode.evalkit.sealed, "_load_test_set", _raise)
    monkeypatch.setattr(weightcode.evalkit.sealed, "evaluate_manifest", _raise)
    monkeypatch.setattr(weightcode.common.paths, "sealed_dir", _raise)
    _FORBIDDEN_ROOTS.append(os.path.realpath(toy_store.sealed))
    try:
        with pytest.raises(PermissionError, match="test guard"):
            open(toy_store.sealed / "probe", "w")  # the guard is live
        record = cp.compare_runs(run, ref)
    finally:
        _FORBIDDEN_ROOTS.clear()
    assert record["function_space"]["run"]["readouts"]["fixed"] > 0


def test_base_noop_run_has_zero_deltas_and_a_flagged_tie_fraction(toy_store: ToyStore) -> None:  # noqa: F811
    dev = _instances(toy_store)["dev100"]
    noop = _noop_run(toy_store, dev)
    ref = _patch_run(toy_store, dev, "fix", fixing_child(dict(dev.params)))
    record = cp.compare_runs(noop, ref)
    space = record["weight_space"]["run"]
    assert space["total_delta_norm"] == 0.0 and space["first_layer_share_of_squared_delta"] is None
    assert all(t["delta_norm"] == 0.0 and t["relative_norm"] == 0.0 for t in space["tensors"].values())
    assert space["tie"]["tie_fraction"] == 0.0 and cp.FLAG_DELTA_ZERO in space["tie"]["flags"]
    assert record["weight_space"]["cosine_run_ref"] == {"all_parameters": None, "first_layer": None}
    assert record["op_summary"] == {"applicable": False, "reason": "not applicable: base no-op final"}
    fs = record["function_space"]
    assert fs["run"]["readouts"]["delta_acc"] == 0 and fs["fixed"]["run"] == 0
    assert fs["fixed"]["intersection_over_run"] is None and fs["fixed"]["jaccard"] == 0.0
    assert record["decomposition"]["agreement_tie_part_with_run"] == 1.0
    # The run is the base, so the paired difference is minus the ref's net gain, and the decomposition is of the run.
    paired, n = fs["paired_run_minus_ref"], fs["n"]
    assert fs["fixed"]["ref"] > 0
    assert paired["ref_wrong_run_right"] == fs["broken"]["ref"] and paired["ref_right_run_wrong"] == fs["fixed"]["ref"]
    assert paired["difference"] == pytest.approx((fs["broken"]["ref"] - fs["fixed"]["ref"]) / n) and paired["difference"] < 0
    assert record["decomposition"]["agreement_residual_with_base"] == 1.0
    assert record["decomposition"]["agreement_residual_local_with_base"] == 1.0


def _write_tie_run(store: ToyStore, instance: Instance, strength: float, groups: list[int]) -> tuple[Path, dict[str, np.ndarray]]:
    """A run whose final patch is one executor tie op on fc1.weight, with the child it rebuilds to."""
    run_dir = store.runs / "toy" / instance.instance_id / "tie"
    (run_dir / "q1").mkdir(parents=True, exist_ok=True)
    patch = {"schema": SCHEMA, "parent_sha256": instance.params_sha256, "hypothesis": "tie", "predicted_effect": "tie",
             "ops": [{"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": groups, "strength": strength}]}
    (run_dir / "q1" / "patch.json").write_text(json.dumps(patch))
    child = apply_patch(instance.params, patch, None, Limits().without_size_limits(), instance.params_sha256).child
    final = {"is_base_noop": False, "dir": "q1", "score": 0, "fixed": 0, "broken": 0, "edit_size": 1,
             "child_sha256": params_sha256(child)}
    return _write_run(run_dir, instance.instance_id, final), child


def test_end_to_end_tie_run_against_a_fixing_ref_on_a_two_channel_view(
    toy_store: ToyStore, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The toy fc1 reads 16 inputs; viewed as (2, 2, 4) they have a channel axis, so a tie run has a real D."""
    dev = _instances(toy_store)["dev100"]
    out = dev.params["fc1.weight"].shape[0]
    view = cp.FirstLayer("fc1", "linear", "fc1.weight", (out, 2, 2, 4), (2, 2, 4))
    monkeypatch.setattr(cp, "instance_first_layer", lambda instance: view)
    run, tie_child = _write_tie_run(toy_store, dev, 0.4, [p % 8 for p in range(16)])
    ref = _patch_run(toy_store, dev, "fix", fixing_child(dict(dev.params)))
    record = cp.compare_runs(run, ref)

    tie = record["weight_space"]["run"]["tie"]
    assert tie["s_eff"] == pytest.approx(0.4, abs=1e-6) and tie["tie_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert tie["local"]["per_unit_spatial"]["explained_fraction_tie_sign"] == pytest.approx(1.0, abs=1e-6)
    assert record["op_summary"]["by_op_tensor"] == {"tie:fc1.weight": 1}
    dc = record["decomposition"]
    assert dc["agreement_tie_part_with_run"] == 1.0 and dc["agreement_residual_with_base"] == 1.0
    assert dc["agreement_tie_part_local_with_run"] == 1.0 and dc["agreement_residual_local_with_base"] == 1.0

    data = devreadout.readout_split(dev)
    y = data["y"]
    run_pred = predict(dev.forward, tie_child, data["x"], 64)
    ref_pred = predict(dev.forward, fixing_child(dict(dev.params)), data["x"], 64)
    gain, loss = int(np.sum((run_pred == y) & (ref_pred != y))), int(np.sum((run_pred != y) & (ref_pred == y)))
    paired = record["function_space"]["paired_run_minus_ref"]
    assert (paired["ref_wrong_run_right"], paired["ref_right_run_wrong"]) == (gain, loss) and gain != loss
    assert paired["difference"] == pytest.approx((gain - loss) / len(y))
    swapped = cp.compare_runs(ref, run)["function_space"]["paired_run_minus_ref"]
    assert swapped["difference"] == pytest.approx(-paired["difference"])

