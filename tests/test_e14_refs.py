"""PREREG E14 instance build, reference fits, V_select readouts and the bar, end to end on a scratch store.

The student is an untrained film_policy recorded as a gated base, and the V_check and V_select ranges are cut to a few
episodes each (``SEED_RANGES`` patched), so the whole path runs in seconds. This tests the machinery only: instance
files and hashes, the rollout oracle, every reference shape's candidates and limits, the readout rebuild of every
candidate, the own and kept picks, the bar, the label-equivalent evidence and the wrapper.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("minigrid")

from weightcode.common.identity import params_sha256, save_params  # noqa: E402
from weightcode.data import e14 as D  # noqa: E402
from weightcode.data.store import write_json  # noqa: E402
from weightcode.models import e14_export as X  # noqa: E402
from weightcode.models import e14_policy as P  # noqa: E402

N_EPISODES = 6
SMALL_RANGES = ("V_check_old", "V_check_new", "V_select_old", "V_select_new")


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("pstore")  # no leak token in the session mounts
    patch = pytest.MonkeyPatch()
    for name, sub in (("WEIGHTCODE_PRIVATE", "private"), ("WEIGHTCODE_SEALED", "sealed"), ("WEIGHTCODE_RUNS", "runs")):
        (root / sub).mkdir()
        patch.setenv(name, str(root / sub))
    for name in SMALL_RANGES:
        patch.setitem(D.SEED_RANGES, name, dataclasses.replace(D.SEED_RANGES[name], n=N_EPISODES))
    X.write_store_forward()
    sets = {"train": ("old", 0, 2_000), "V_evidence_new": ("new", D.SEED_RANGES["V_evidence_new"].start, 40)}
    manifest = {"smoke": True, "versions": D.environment_versions(), "sets": {}}
    for name, (mission, start, n) in sets.items():
        arrays = D.merge_demo_chunks([D.demo_chunk((mission, start, start + n))])
        manifest["sets"][name] = {"file": D.DEMO_FILES[name], "sha256": D.save_demos(name, arrays)}
    write_json(D.demos_dir() / D.DEMOS_MANIFEST, manifest)
    torch.manual_seed(100)
    params = P.export_params(P.FilmPolicy(P.SIZES["base"]))
    model_id = "m00e14fix"
    forward, _ = X.load_store_forward()
    readouts = X.readout(forward, params)
    readouts["gate"]["passed"] = readouts["headroom"]["passed"] = True  # recorded as a base: machinery only
    model_dir = X.models_dir() / model_id
    model_dir.mkdir(parents=True)
    digest = save_params(model_dir / "model.npz", params)
    write_json(model_dir / "model_spec.json", P.build_spec(model_id, params))
    write_json(model_dir / "private.json", {"model_id": model_id, "reserved_rows_unchanged": True,
                                            "readouts": readouts, "split_role": "dev"})
    write_json(X.models_dir() / "index.json", [{
        "model_id": model_id, "seed": 100, "arch": P.ARCH, "size": "base", "n_episodes": 2_000, "role": "dev",
        "params_sha256": digest, "gate_passed": True}])
    yield {"root": root, "model_id": model_id, "params": params}
    patch.undo()


@pytest.fixture(scope="module")
def instance_id(store) -> str:
    from weightcode.evalkit import e14 as E

    monkey = pytest.MonkeyPatch()
    monkey.setattr(D, "TRAIN_SIZES", (*D.TRAIN_SIZES, 2_000))
    row = E.build_e14_instance(store["model_id"])
    again = E.build_e14_instance(store["model_id"])
    monkey.undo()
    assert row["status"] == "created" and again["status"] == "reused"
    assert again["instance_id"] == row["instance_id"]
    return str(row["instance_id"])


def test_instance_files(store, instance_id: str) -> None:
    from weightcode.evalkit import e14 as E

    instance = E.load_e14_instance(instance_id)
    E.check_e14_current(instance.instance_dir)
    ev, private = instance.evidence, instance.private
    assert sorted(ev) == sorted(E.EVIDENCE_KEYS)
    assert sorted(set(ev["episode_new"].tolist())) == list(range(E.K_NEW))
    assert int(np.count_nonzero(ev["y_new"] == D.PICKUP)) == E.K_NEW
    assert len(ev["y_old"]) == E.N_REPLAY and np.isin(ev["y_old"], D.OLD_ACTIONS).all()
    assert private["evidence_episodes"] == list(range(E.K_NEW))  # all 40 smoke episodes are usable
    assert np.array_equal(E.policy_logits(instance.forward, instance.params, ev["x_new"]), ev["logits_new"])
    assert private["n_vcheck_new"] == private["n_vcheck_old"] == N_EPISODES
    text = (Path(instance.instance_dir) / "task.json").read_text()
    assert not any(token in text.lower() for token in E.e14_leak_tokens())


@pytest.mark.parametrize("method", ["bc_new_rows", "bc_sparse19_sal", "f1_last", "lora_r4", "ft_full_lwf",
                                    "ft_full_noreplay", "labeleq_k16"])
def test_reference_run_and_readout(store, instance_id: str, method: str) -> None:
    from weightcode.baselines.fit.e14refs import E14_REF_LABEL, method_configs, run_e14_ref
    from weightcode.baselines.fit.runner import baseline_run_dir
    from weightcode.evalkit import e14 as E
    from weightcode.evalkit.e14read import read_run

    torch.set_num_threads(1)
    instance = E.load_e14_instance(instance_id)
    run_dir = baseline_run_dir(instance_id, method, 0, E14_REF_LABEL)
    record = run_e14_ref(instance, method, run_dir, run_seed=0)
    assert [c["config"] for c in record["candidates"]] == method_configs(method)
    assert record["joint"]["oracle"] == "RolloutJointOracle"
    assert record["joint"]["replay_in_fit"] is (method != "ft_full_noreplay")
    if method == "labeleq_k16":
        assert record["evidence"]["n_rand"] > instance.private["n_new_rows"]
    log = record["oracle_log"]
    assert log and all(e["n_new"] == e["n_old"] == N_EPISODES for e in log)
    read = read_run(run_dir / "run.json")
    assert len(read["candidates"]) == len(log)
    for c in read["candidates"]:
        ro = c["readout"]
        assert ro["new"]["n"] == ro["old"]["n"] == N_EPISODES
        assert ro["joint"] == pytest.approx(ro["gain"] + ro["clean"])
    assert read["kept"]["selection"]["s_old_floor"] == -3
    json.dumps(read, allow_nan=False)


def test_bar_and_wrapper(store, instance_id: str) -> None:
    from weightcode.baselines.fit.e14refs import E14_REF_LABEL, run_e14_ref
    from weightcode.baselines.fit.runner import baseline_run_dir
    from weightcode.evalkit import e14 as E
    from weightcode.evalkit.e14read import e14_bar, read_run

    instance = E.load_e14_instance(instance_id)
    records = {}
    for method in ("bc_full", "f1_full"):
        run_dir = baseline_run_dir(instance_id, method, 0, E14_REF_LABEL)
        run_e14_ref(instance, method, run_dir, run_seed=0)
        records[method] = read_run(run_dir / "run.json")
    wrapper = E.e14_wrapper_readout(instance_id)
    assert wrapper["old"]["n"] == N_EPISODES
    bar = e14_bar(records, ["bc_full"], ["f1_full"], wrapper={"instance_id": instance_id, "readout": wrapper})
    best = max(records, key=lambda m: records[m]["kept"]["joint"])
    assert bar["bar_joint"] == pytest.approx(records[best]["kept"]["joint"] + 1.0)
    assert bar["labeleq_curve"][0]["k"] == 8
    assert bar["instance_id"] == instance_id
    for pick in (*bar["kept"].values(), bar["wrapper"]):
        split = pick["success_new_split"]
        assert sorted(split) == ["demonstrated", "not_demonstrated"]
        assert split["demonstrated"]["n"] + split["not_demonstrated"]["n"] == N_EPISODES
    assert bar["wrapper"]["success_new"] == wrapper["new"]["success_child"]
    json.dumps(bar, allow_nan=False)
    with pytest.raises(Exception):
        e14_bar(records, ["bc_full", "bc_emb_head"], ["f1_full"])
    with pytest.raises(Exception):
        e14_bar(records, ["bc_full"], ["f1_full"], wrapper={"instance_id": "p00000000", "readout": wrapper})


def test_fn_readout_equals_the_readouts_of_record(store, instance_id: str) -> None:
    """PREREG E14 T11: ``e14_fn_readout`` gives ``e14_wrapper_readout`` on the wrapper and ``e14_readout`` (without its
    descriptor list) on plain weights, and a trace of it splits into the two parts' episodes."""
    from weightcode.evalkit import e14 as E

    instance = E.load_e14_instance(instance_id)
    wrapper, _ = E.e14_fn_readout(instance_id, E.wrapper_logits_fn(instance.forward, instance.params))
    assert wrapper == E.e14_wrapper_readout(instance_id)
    trace = E.RolloutTrace(E.logits_fn(instance.forward, store["params"]))
    plain, children = E.e14_fn_readout(instance_id, trace)
    record = E.e14_readout(instance_id, store["params"])
    assert {key: value for key, value in record.items() if key != "demonstrated_descriptors"} == plain
    steps = trace.steps([children["new"].length, children["old"].length])
    assert [len(s["length"]) for s in steps] == [N_EPISODES, N_EPISODES]
    assert params_sha256(instance.params) == params_sha256(store["params"])
