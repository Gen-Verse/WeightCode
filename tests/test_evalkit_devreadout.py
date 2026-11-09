"""Tests for the dev readout on V_select (weightcode.evalkit.devreadout)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import ToyStore, build_v1, toy_store  # noqa: F401

from weightcode.baselines.instance_io import load_instance, write_delta_patch
from weightcode.common.identity import params_sha256
from weightcode.evalkit import devreadout
from weightcode.evalkit.instances import StaleInstanceError, supersede_instances
from weightcode.evalkit.metrics import sealed_readout
from weightcode.evalkit.patch_check import predict


def _instances(store: ToyStore) -> dict[str, Path]:
    outcomes = build_v1(store)
    root = store.private / "instances"
    return {f"{o.split_role}{o.seed}": root / o.instance_id for o in outcomes}


def _shifted_child(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    child = {key: np.array(value, copy=True) for key, value in params.items()}
    last_bias = sorted(key for key in child if key.endswith(".bias"))[-1]
    child[last_bias][0] += np.float32(3.0)
    return child


def test_dev_readout_matches_a_direct_computation_on_v_select(toy_store: ToyStore) -> None:  # noqa: F811
    instance = load_instance(_instances(toy_store)["dev100"])
    child = _shifted_child(dict(instance.params))
    record = devreadout.dev_readout(instance, child)
    with np.load(toy_store.private / toy_store.dataset / "splits" / "V_select.npz") as archive:
        x, y = archive["x"], archive["y"]
    base_pred = predict(instance.forward, instance.params, x, 1024)
    child_pred = predict(instance.forward, child, x, 1024)
    expected = sealed_readout(y, base_pred, child_pred, n_classes=len(instance.task["classes"]))
    assert record["split"] == "V_select" and record["n"] == len(y)
    assert record["readouts"]["delta_acc"] == expected["delta_acc"]
    assert record["readouts"]["fixed"] == expected["fixed"] and record["readouts"]["broken"] == expected["broken"]
    noop = devreadout.dev_readout(instance, dict(instance.params))
    assert noop["readouts"]["delta_acc"] == 0 and noop["readouts"]["fixed"] == noop["readouts"]["broken"] == 0


def test_dev_readout_refuses_main_superseded_and_bad_children(toy_store: ToyStore) -> None:  # noqa: F811
    dirs = _instances(toy_store)
    main = load_instance(dirs["main0"])
    with pytest.raises(devreadout.DevReadoutError, match="dev only"):
        devreadout.dev_readout(main, dict(main.params))
    dev = load_instance(dirs["dev100"])
    wrong_shape = dict(dev.params)
    key = sorted(wrong_shape)[0]
    wrong_shape[key] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(devreadout.DevReadoutError, match="base"):
        devreadout.dev_readout(dev, wrong_shape)
    non_finite = {k: np.array(v, copy=True) for k, v in dev.params.items()}
    non_finite[key].flat[0] = np.nan
    with pytest.raises(devreadout.DevReadoutError, match="non-finite"):
        devreadout.dev_readout(dev, non_finite)
    supersede_instances([dev.instance_id], "test: retired")
    with pytest.raises(StaleInstanceError, match="superseded"):
        devreadout.dev_readout(dev, dict(dev.params))


def _write_run(run_dir: Path, instance_id: str, final: dict[str, object], enforced: bool = True) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    run = {"instance": {"instance_id": instance_id}, "method": "toy", "flags": {"edit_limits_enforced": enforced}, "final": final}
    path = run_dir / "run.json"
    path.write_text(json.dumps(run))
    return path


def test_readout_run_rebuilds_the_final_child_and_checks_its_hash(toy_store: ToyStore) -> None:  # noqa: F811
    dev = load_instance(_instances(toy_store)["dev100"])
    child = _shifted_child(dict(dev.params))
    run_dir = toy_store.runs / "toy" / dev.instance_id / "arm" / "0"
    write_delta_patch(run_dir / "q1", dev.params_sha256, dev.params, child, "toy", "toy")
    final = {"is_base_noop": False, "dir": "q1", "score": 5, "fixed": 6, "broken": 1, "edit_size": 1,
             "child_sha256": params_sha256(child)}
    record = devreadout.readout_run(_write_run(run_dir, dev.instance_id, final))
    assert record["readouts"]["delta_acc"] == devreadout.dev_readout(dev, child)["readouts"]["delta_acc"]
    assert record["vcheck"] == {"score": 5, "fixed": 6, "broken": 1}

    final["child_sha256"] = "0" * 64
    with pytest.raises(devreadout.DevReadoutError, match="recorded final child"):
        devreadout.readout_run(_write_run(run_dir, dev.instance_id, final))

    noop_dir = toy_store.runs / "toy" / dev.instance_id / "noop" / "0"
    noop = {"is_base_noop": True, "dir": None, "score": 0, "fixed": 0, "broken": 0, "edit_size": 0,
            "child_sha256": dev.params_sha256}
    record = devreadout.readout_run(_write_run(noop_dir, dev.instance_id, noop))
    assert record["readouts"]["delta_acc"] == 0 and record["is_base_noop"]
    noop["child_sha256"] = "1" * 64
    with pytest.raises(devreadout.DevReadoutError, match="base no-op"):
        devreadout.readout_run(_write_run(noop_dir, dev.instance_id, noop))

    found = devreadout.run_files([toy_store.runs / "toy"])
    assert found == sorted(found) and len(found) == 2
    out = toy_store.runs / "dev.jsonl"
    noop["child_sha256"] = dev.params_sha256
    _write_run(noop_dir, dev.instance_id, noop)
    final["child_sha256"] = params_sha256(child)
    _write_run(run_dir, dev.instance_id, final)
    assert devreadout.main([str(toy_store.runs / "toy"), "--out", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 2

    # A run.json written inside a run's directory (as an editor could under patches/) is refused by the search,
    # while naming the real run.json still works.
    planted = run_dir / "patches" / "x" / "run.json"
    planted.parent.mkdir(parents=True)
    planted.write_text((run_dir / "run.json").read_text())
    with pytest.raises(devreadout.NestedRunError, match="inside another run's directory"):
        devreadout.run_files([toy_store.runs / "toy"])
    with pytest.raises(devreadout.NestedRunError):
        devreadout.main([str(toy_store.runs / "toy")])
    assert devreadout.run_files([run_dir / "run.json"]) == [run_dir / "run.json"]


def test_saved_children_for_decoy_and_rewrite_are_loaded_and_hash_checked(toy_store: ToyStore) -> None:  # noqa: F811
    from weightcode.common.identity import save_params

    dev = load_instance(_instances(toy_store)["dev100"])
    child = _shifted_child(dict(dev.params))
    run_dir = toy_store.runs / "harness" / dev.instance_id / "rewrite"
    (run_dir / "final").mkdir(parents=True)
    digest = save_params(run_dir / "final" / "model.npz", child)
    final = {"is_base_noop": False, "dir": "final", "parent": "zeros", "model": "final/model.npz", "score": 4,
             "fixed": 5, "broken": 1, "edit_size": 9, "child_sha256": digest}
    record = devreadout.readout_run(_write_run(run_dir, dev.instance_id, final, enforced=False))
    assert record["readouts"]["delta_acc"] == devreadout.dev_readout(dev, child)["readouts"]["delta_acc"]
    final["parent"] = "decoy_readdressed"
    devreadout.readout_run(_write_run(run_dir, dev.instance_id, final))
    for bad, match in (
        ({"child_sha256": "2" * 64}, "saved child"),
        ({"model": "../outside.npz"}, "leaves the run directory"),
        ({"model": None}, "needs final.model"),
        ({"parent": "teacher"}, "unknown final.parent"),
    ):
        with pytest.raises(devreadout.DevReadoutError, match=match):
            devreadout.readout_run(_write_run(run_dir, dev.instance_id, dict(final, **bad)))


def test_a_harness_final_is_rebuilt_with_the_session_s_own_limits(toy_store: ToyStore) -> None:  # noqa: F811
    """A program-arm final (a dense delta on a whole tensor, about 19% of entries) exceeds the default edit-size
    limit but not the limits its session recorded; the readout rebuilds it with the recorded ones."""
    from dataclasses import asdict

    from weightcode.patch import Limits, PatchError

    dev = load_instance(_instances(toy_store)["dev100"])
    child = {key: np.array(value, copy=True) for key, value in dev.params.items()}
    child["fc2.weight"] += np.float32(0.01)
    run_dir = toy_store.runs / "sessions" / "s_learn"
    write_delta_patch(run_dir / "final", dev.params_sha256, dev.params, child, "toy", "toy")
    final = {"is_base_noop": False, "dir": "final", "parent": "instance", "score": 3, "fixed": 4, "broken": 1,
             "edit_size": 32, "child_sha256": params_sha256(child)}
    run_path = _write_run(run_dir, dev.instance_id, final)
    program_limits = asdict(Limits(max_changed_frac=0.2, max_edit_frac=None))
    (run_dir / "session.json").write_text(json.dumps({"arm": "LEARN", "limits": program_limits}))
    record = devreadout.readout_run(run_path)
    assert record["final_limits"] == "session.json"
    assert record["readouts"] == devreadout.dev_readout(dev, child)["readouts"]
    # The same patch under the default limits is refused by the executor.
    (run_dir / "session.json").unlink()
    with pytest.raises(PatchError):
        devreadout.readout_run(run_path)
    # A harness run.json (it names its session_result) without session.json is refused.
    run = json.loads(run_path.read_text())
    run["session_result"] = "session_result.json"
    run_path.write_text(json.dumps(run))
    with pytest.raises(devreadout.DevReadoutError, match="session.json"):
        devreadout.readout_run(run_path)
    (run_dir / "session.json").write_text(json.dumps({"arm": "LEARN"}))
    with pytest.raises(devreadout.DevReadoutError, match="records no limits"):
        devreadout.readout_run(run_path)


def test_an_e10_reference_final_is_rebuilt_with_the_e10_program_limits(toy_store: ToyStore) -> None:  # noqa: F811
    """An E10 reference final (a dense delta on a whole tensor) exceeds the default edit-size limit; the readout
    rebuilds it with the E10 program limits the run records, and refuses a run that records other limits."""
    from dataclasses import asdict

    from weightcode.baselines.fit.runner import E10_LIMITS_SOURCE
    from weightcode.evalkit.e10 import e10_program_limits
    from weightcode.patch import PatchError

    dev = load_instance(_instances(toy_store)["dev100"])
    child = {key: np.array(value, copy=True) for key, value in dev.params.items()}
    child["fc2.weight"] += np.float32(0.01)
    run_dir = toy_store.runs / "baselines" / dev.instance_id / "grad_sparse19_sal" / "e10" / "0"
    write_delta_patch(run_dir / "q0", dev.params_sha256, dev.params, child, "toy", "toy")
    final = {"is_base_noop": False, "dir": "q0", "score": 3, "fixed": 4, "broken": 1, "edit_size": 32,
             "child_sha256": params_sha256(child)}
    run_path = _write_run(run_dir, dev.instance_id, final)
    program = asdict(e10_program_limits(dev.task["limits"], len(dev.params)))

    def write(limits: object, source: object = E10_LIMITS_SOURCE) -> None:
        run = json.loads(run_path.read_text())
        run.update({"limits": limits, "limits_source": source})
        run_path.write_text(json.dumps(run))

    write(program)
    record = devreadout.readout_run(run_path)
    assert record["final_limits"] == devreadout.E10_BASELINE_LIMITS
    assert record["readouts"] == devreadout.dev_readout(dev, child)["readouts"]
    write(dev.task["limits"])
    with pytest.raises(devreadout.DevReadoutError, match="E10 program limits"):
        devreadout.readout_run(run_path)
    write(program, None)  # without the E10 source the default limits apply, and the executor refuses the patch
    with pytest.raises(PatchError):
        devreadout.readout_run(run_path)
