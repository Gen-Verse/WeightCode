"""Tests for evidence-v1 instances, supersession, decoy maps and edit-limit definitions (weightcode.evalkit.instances)."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    BASE_ERRORS,
    DEV_PLAN,
    FIVE_SEED_DECOYS,
    FIVE_SEEDS,
    PLAN,
    ToyStore,
    build_v0,
    build_v1,
    toy_store,
)

import weightcode.evalkit.instances as instances
from weightcode.baselines.instance_io import load_instance
from weightcode.evalkit.evidence import (
    EVIDENCE_KEYS,
    K_RAND,
    draw_evidence,
    draw_evidence_v1,
    evidence_seed_v1,
    load_evidence,
)
from weightcode.evalkit.instances import (
    DEFAULT_DECOYS,
    EDITOR_VISIBLE_FILES,
    INDEX_FIELDS_V1,
    LIMIT_DEFINITIONS,
    InstanceConflict,
    active_instance_ids,
    build_deployment_instances,
    check_decoys,
    format_decoys,
    leak_tokens,
    limit_definitions,
    parse_decoys,
    read_index,
    supersede_instances,
)
from weightcode.evalkit.patch_check import PatchCheckOracle
from weightcode.evalkit.targets import K_FAIL, SeedSelection
from weightcode.patch import task_limits


def _errors_input(n: int = 300, n_errors: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    x = rng.integers(0, 256, size=(n, 1, 2, 2), dtype=np.uint8)
    y = rng.integers(0, 4, size=n).astype(np.int64)
    pred = y.copy()
    rows = rng.choice(n, size=n_errors, replace=False)
    pred[rows] = (y[rows] + 1 + rng.integers(0, 3, size=n_errors)) % 4
    return x, y, pred


def test_evidence_v1_seed_and_draw() -> None:
    sha = "cd" * 32
    expected = int.from_bytes(hashlib.sha256(f"{sha}|deploy|evidence-v1".encode()).digest()[:8], "big")
    assert evidence_seed_v1(sha, "deploy") == expected
    assert evidence_seed_v1(sha, "gaussian_noise_3") != expected
    for bad_sha, condition in (("not-a-hash", "deploy"), (sha, "Deploy"), (sha, ""), (sha, "a|b")):
        with pytest.raises(ValueError):
            evidence_seed_v1(bad_sha, condition)
    x, y, pred = _errors_input()
    evidence = draw_evidence_v1(x, y, pred, seed=11)
    rows = evidence.row_ids
    fail_rows, rand_rows = rows[:K_FAIL], rows[K_FAIL:]
    assert set(evidence.arrays) == set(EVIDENCE_KEYS) and len(set(rows.tolist())) == K_FAIL + K_RAND
    assert np.all(pred[fail_rows] != y[fail_rows])
    assert np.array_equal(evidence.arrays["pred_fail"], pred[fail_rows])
    assert np.array_equal(evidence.arrays["y_rand"], y[rand_rows])
    errors = set(np.flatnonzero(pred != y).tolist())
    assert {int(row) for row in fail_rows} <= errors
    again = draw_evidence_v1(x, y, pred, seed=11)
    assert np.array_equal(again.row_ids, rows) and again.split_ids == evidence.split_ids
    assert not np.array_equal(draw_evidence_v1(x, y, pred, seed=12).row_ids, rows)
    with pytest.raises(ValueError, match="k_fail"):
        draw_evidence_v1(*_errors_input(n_errors=K_FAIL - 1), seed=11)
    draw_evidence_v1(*_errors_input(n_errors=K_FAIL), seed=11)


def test_v0_and_v1_draw_the_same_way_from_the_same_pool() -> None:
    """With every base error inside one (a, b) cell, the two rules draw identical rows from the same seed."""
    x, y, pred = _errors_input(n_errors=0)
    cell = np.flatnonzero(y == 0)[:30]
    pred[cell] = 1
    v0 = draw_evidence(x, y, pred, a=0, b=1, seed=21)
    v1 = draw_evidence_v1(x, y, pred, seed=21)
    assert np.array_equal(v0.row_ids, v1.row_ids) and v0.split_ids == v1.split_ids


def test_v1_layout_task_and_private(toy_store: ToyStore) -> None:  # noqa: F811
    outcomes = build_v1(toy_store)
    assert [(o.split_role, o.seed, o.status) for o in outcomes] == [
        ("dev", 100, "created"), ("dev", 101, "created"), ("main", 0, "created"), ("main", 1, "created")
    ]
    ids = {**toy_store.ids("base"), **toy_store.ids("dev")}
    root = toy_store.private / "instances"
    with np.load(toy_store.private / toy_store.dataset / "splits" / "V_evidence.npz") as archive:
        v_evidence = {key: archive[key] for key in archive.files}
    n_ev = len(v_evidence["y"])
    n_check = PLAN["V_check"][0] * 4
    forbidden = [*leak_tokens(toy_store.dataset), *(m.lower() for m in ids.values())]
    for outcome in outcomes:
        folder = root / outcome.instance_id
        assert sorted(p.name for p in folder.iterdir()) == sorted([*EDITOR_VISIBLE_FILES, "private.json"])
        task_text = (folder / "task.json").read_text()
        assert not [token for token in forbidden if token in task_text.lower()]
        task = json.loads(task_text)
        assert task["schema"] == "weightcode.task.v1" and "target" not in task and "selection_counts" not in task
        assert "gets wrong" in task["statement"] and "random examples" in task["statement"]
        assert task["evidence"]["fail"]["count"] == K_FAIL and task["evidence"]["random"]["count"] == K_RAND
        assert "S = fixed - broken" in task["patch_check"]["readout"]
        assert task["limits"] == task_limits()
        assert task["limit_definitions"] == limit_definitions(task["limits"])
        private = json.loads((folder / "private.json").read_text())
        assert private["schema"] == "weightcode.instance_private.v1" and private["evidence_rule"] == "evidence-v1"
        assert private["condition"] == "deploy" and private["split_role"] == outcome.split_role
        assert not [key for key in private if key.startswith("target")]
        decoy_seed = DEFAULT_DECOYS[outcome.seed]
        assert private["decoy_seed"] == decoy_seed and private["decoy_model_id"] == ids[decoy_seed]
        assert private["decoy_model_path"].endswith(f"{ids[decoy_seed]}/model.npz")
        assert (private["n_vevidence"], private["n_base_errors_vevidence"]) == (n_ev, BASE_ERRORS["V_evidence"])
        assert (private["n_vcheck"], private["n_base_errors_vcheck"]) == (n_check, BASE_ERRORS["V_check"])
        meta = json.loads((toy_store.private / toy_store.dataset / "dataset.json").read_text())
        assert private["vcheck_content_sha256"] == meta["splits"]["V_check"]["sha256"]
        assert private["vevidence_content_sha256"] == meta["splits"]["V_evidence"]["sha256"]
        assert private["vcheck_path"].endswith("splits/V_check.npz") and len(private["vcheck_sha256"]) == 64
        assert private["evidence_seed"] == evidence_seed_v1(private["model_sha256"], "deploy")
        evidence = load_evidence(folder / "evidence.npz")
        rows = np.asarray(private["evidence_row_ids"])
        assert np.array_equal(evidence["x_fail"], v_evidence["x"][rows[:K_FAIL]])
        assert np.all(evidence["pred_fail"] != evidence["y_fail"])
        assert np.array_equal(evidence["y_rand"], v_evidence["y"][rows[K_FAIL:]])
        assert private["evidence_source_idx"] == v_evidence["idx"][rows].tolist()
        instance = load_instance(folder)
        with np.load(private["vcheck_path"]) as archive:
            oracle = PatchCheckOracle(archive["x"], archive["y"], instance.forward, instance.params, None, 8)
        check = oracle.query(instance.params, "noop")
        assert check["n_base_errors"] == private["n_base_errors_vcheck"] and check["score"] == 0
    index = read_index()
    assert [row["instance_id"] for row in index["instances"]] == [o.instance_id for o in outcomes]
    assert all(set(row) == set(INDEX_FIELDS_V1) for row in index["instances"])
    assert index["superseded"] == []


def test_v1_rebuild_reuses_and_refuses_changes(toy_store: ToyStore) -> None:  # noqa: F811
    first = build_v1(toy_store)
    second = build_v1(toy_store)
    assert [o.instance_id for o in second] == [o.instance_id for o in first]
    assert all(o.status == "reused" for o in second)
    task_path = toy_store.private / "instances" / first[0].instance_id / "task.json"
    task = json.loads(task_path.read_text())
    task["statement"] += " Edited."
    task_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n")
    with pytest.raises(InstanceConflict, match="task.json"):
        build_v1(toy_store)


def test_a_changed_decoy_map_is_refused_not_rewritten(toy_store: ToyStore) -> None:  # noqa: F811
    build_v1(toy_store, roles=("main",), selection=SeedSelection(main=(0, 1, 2), dev=(100, 101)),
             decoys={0: 1, 1: 2, 2: 0, 100: 101, 101: 100})
    with pytest.raises(InstanceConflict, match="decoy"):
        build_v1(toy_store, roles=("main",), selection=SeedSelection(main=(0, 1, 2), dev=(100, 101)),
                 decoys={0: 1, 1: 0, 2: 0, 100: 101, 101: 100})


def test_stray_instances_are_refused_until_superseded(toy_store: ToyStore) -> None:  # noqa: F811
    five = build_v1(toy_store, selection=FIVE_SEEDS, decoys=FIVE_SEED_DECOYS)
    assert {o.seed: o.decoy_seed for o in five} == {100: 101, 101: 100, 0: 1, 1: 0, 2: 3, 3: 4, 4: 2}
    with pytest.raises(InstanceConflict, match="outside the seed selection"):
        build_v1(toy_store)
    extra = [o.instance_id for o in five if o.seed in (2, 3, 4)]
    with pytest.raises(ValueError, match="reason"):
        supersede_instances(extra, " ")
    rows = supersede_instances(extra, "development uses main seeds 0 and 1")
    assert [row["instance_id"] for row in rows] == extra
    assert all(row["superseded_reason"] == "development uses main seeds 0 and 1" for row in rows)
    root = toy_store.private / "instances"
    assert all((root / instance_id / "private.json").is_file() for instance_id in extra)
    index = read_index()
    assert sorted(row["instance_id"] for row in index["superseded"]) == sorted(extra)
    assert not {row["instance_id"] for row in index["instances"]} & set(extra)
    events = [json.loads(line) for line in (root / "changes.jsonl").read_text().splitlines()]
    assert [(e["event"], e["instance_id"]) for e in events] == [("superseded", i) for i in extra]
    again = build_v1(toy_store)
    kept = {o.seed: o.instance_id for o in five if o.seed not in (2, 3, 4)}
    assert {o.seed: o.instance_id for o in again} == kept and all(o.status == "reused" for o in again)
    with pytest.raises(InstanceConflict, match="superseded"):
        supersede_instances(extra[:1], "again")
    with pytest.raises(InstanceConflict, match="unknown"):
        supersede_instances(["i00000000"], "no such instance")
    assert active_instance_ids(dataset=toy_store.dataset) == [o.instance_id for o in again]


def test_five_seed_extension_keeps_first_pass_instances(toy_store: ToyStore) -> None:  # noqa: F811
    first = build_v1(toy_store)
    extended = build_v1(toy_store, selection=FIVE_SEEDS, decoys=FIVE_SEED_DECOYS)
    by_seed = {o.seed: o for o in extended}
    for outcome in first:
        assert by_seed[outcome.seed].instance_id == outcome.instance_id
        assert by_seed[outcome.seed].status == "reused"
    assert [by_seed[s].status for s in (2, 3, 4)] == ["created"] * 3
    assert [by_seed[s].decoy_seed for s in (2, 3, 4)] == [3, 4, 2]


def test_superseding_v0_leaves_v1_active(toy_store: ToyStore) -> None:  # noqa: F811
    v0 = build_v0(toy_store)
    v1 = build_v1(toy_store)
    assert active_instance_ids(dataset=toy_store.dataset, evidence_rule="evidence-v0") == [o.instance_id for o in v0]
    supersede_instances(active_instance_ids(dataset=toy_store.dataset, evidence_rule="evidence-v0"), "v1 replaces v0")
    index = read_index()
    assert [row["instance_id"] for row in index["instances"]] == [o.instance_id for o in v1]
    assert sorted(row["instance_id"] for row in index["superseded"]) == sorted(o.instance_id for o in v0)
    assert all("target" in row for row in index["superseded"])
    assert all(o.status == "reused" for o in build_v1(toy_store))


def test_decoy_map_parsing_and_validation() -> None:
    assert parse_decoys("0:1,1:0,100:101,101:100") == DEFAULT_DECOYS
    assert parse_decoys(" 2 : 3 ") == {2: 3}
    assert format_decoys(FIVE_SEED_DECOYS) == "0:1,1:0,2:3,3:4,4:2,100:101,101:100"
    assert parse_decoys(format_decoys(FIVE_SEED_DECOYS)) == FIVE_SEED_DECOYS
    for bad in ("0-1", "0:1,0:2", "", "0:1,", "a:b"):
        with pytest.raises(ValueError):
            parse_decoys(bad)
    assert check_decoys(DEV_PLAN, DEFAULT_DECOYS) == DEFAULT_DECOYS
    assert check_decoys(FIVE_SEEDS, FIVE_SEED_DECOYS) == FIVE_SEED_DECOYS
    cases = [
        ({0: 1, 1: 0, 100: 101}, "no entry for \\[101\\]"),
        ({**DEFAULT_DECOYS, 2: 3}, "unselected seeds \\[2\\]"),
        ({0: 100, 1: 0, 100: 101, 101: 100}, "not a selected main seed"),
        ({0: 0, 1: 0, 100: 101, 101: 100}, "own decoy"),
        ({0: 1, 1: 0, 100: 0, 101: 100}, "not a selected dev seed"),
        ({0: 2, 1: 0, 100: 101, 101: 100}, "not a selected main seed"),
    ]
    for decoys, message in cases:
        with pytest.raises(ValueError, match=message):
            check_decoys(DEV_PLAN, decoys)
    with pytest.raises(TypeError):
        check_decoys(DEV_PLAN, {0: 1, 1: True, 100: 101, 101: 100})


def test_limit_definitions_follow_the_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    current = task_limits()
    assert set(limit_definitions(current)) == set(current)
    if "max_edit_frac" not in current and "rel_frob_min_ndim" not in current:
        assert limit_definitions(current)["max_rel_frob"] == "per tensor, ||child - parent||_F / ||parent||_F"
    rank_2 = "per tensor of rank >= 2, ||child - parent||_F / ||parent||_F; 1-D tensors are exempt"
    switched = {"max_edit_frac": 0.05, "max_rel_frob": 0.5, "max_ops": 64, "max_rank": 8, "max_file_bytes": 1}
    assert limit_definitions(switched)["max_rel_frob"] == rank_2
    assert limit_definitions(switched)["max_edit_frac"] == LIMIT_DEFINITIONS["max_edit_frac"]
    assert limit_definitions(dict(switched, rel_frob_min_ndim=2))["max_rel_frob"] == rank_2
    assert "rank >= 3" in limit_definitions(dict(switched, rel_frob_min_ndim=3))["max_rel_frob"]
    assert set(limit_definitions(dict(switched, rel_frob_min_ndim=2))) == set(switched) | {"rel_frob_min_ndim"}
    exempt = limit_definitions(dict(switched, rel_frob_min_ndim=2, rel_frob_exempt=["fc2.weight", "fc2.bias"]))
    assert exempt["max_rel_frob"] == rank_2 + "; the tensors named in rel_frob_exempt are exempt whatever their rank"
    assert exempt["rel_frob_exempt"] == LIMIT_DEFINITIONS["rel_frob_exempt"]
    assert limit_definitions(dict(switched, rel_frob_exempt=[]))["max_rel_frob"] == rank_2

    @dataclasses.dataclass(frozen=True)
    class SwitchedLimits:
        max_edit_frac: float | None = 0.02
        max_rel_frob: float | None = 0.5
        rel_frob_min_ndim: int = 2
        max_ops: int = 64
        max_rank: int = 8
        max_file_bytes: int = 64 * 2**20

    monkeypatch.setattr(instances, "task_limits", lambda: dataclasses.asdict(SwitchedLimits()))
    task = instances.build_task_v1(["a", "b"])
    assert task["limits"] == dataclasses.asdict(SwitchedLimits())
    assert task["limit_definitions"]["max_rel_frob"] == rank_2
    assert "max_changed_frac" not in task["limit_definitions"]

    @dataclasses.dataclass(frozen=True)
    class FlippedLimits:
        """The switch as announced: max_edit_frac 0.05, max_changed_frac None, rel_frob_min_ndim 2."""

        max_changed_frac: float | None = None
        max_edit_frac: float | None = 0.05
        max_rel_frob: float | None = 0.5
        rel_frob_min_ndim: int = 2
        max_ops: int = 64
        max_rank: int = 8
        max_file_bytes: int = 64 * 2**20

    monkeypatch.setattr(instances, "task_limits", lambda: dataclasses.asdict(FlippedLimits()))
    task = instances.build_task_v1(["a", "b"])
    assert task["limits"] == dataclasses.asdict(FlippedLimits()) and task["limits"]["max_changed_frac"] is None
    definitions = task["limit_definitions"]
    assert set(definitions) == set(task["limits"])
    assert definitions["max_changed_frac"].endswith("; null: this limit is not applied")
    assert definitions["max_edit_frac"] == LIMIT_DEFINITIONS["max_edit_frac"]
    assert definitions["max_rel_frob"] == rank_2
    assert json.loads(json.dumps(task)) == task

    @dataclasses.dataclass(frozen=True)
    class UnknownLimits:
        max_mystery: float = 1.0

    monkeypatch.setattr(instances, "task_limits", lambda: dataclasses.asdict(UnknownLimits()))
    with pytest.raises(RuntimeError, match="max_mystery"):
        instances.patch_limits()


def test_unknown_condition_is_refused(toy_store: ToyStore) -> None:  # noqa: F811
    with pytest.raises(LookupError, match=r"'gaussian_noise_3' is not in the data store; known: \['deploy'\]"):
        build_deployment_instances(toy_store.dataset, "gaussian_noise_3", selection=DEV_PLAN, decoys=DEFAULT_DECOYS)


def test_check_task_current_refuses_stale_superseded_and_v0(toy_store: ToyStore, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    outcomes = build_v1(toy_store)
    root = toy_store.private / "instances"
    first = root / outcomes[0].instance_id
    task = instances.check_task_current(first)
    assert task == json.loads((first / "task.json").read_text())

    fields = task_limits()
    fields["max_rel_frob"] = 0.25
    tighter = dataclasses.make_dataclass(
        "TighterLimits", [(name, object, dataclasses.field(default=value)) for name, value in fields.items()], frozen=True
    )
    with monkeypatch.context() as patched:
        patched.setattr(instances, "task_limits", lambda: dataclasses.asdict(tighter()))
        with pytest.raises(instances.StaleInstanceError, match="limits"):
            instances.check_task_current(first)
    with monkeypatch.context() as patched:
        patched.setattr(instances, "OP_NAMES", tuple(instances.OP_NAMES)[:-1])
        with pytest.raises(instances.StaleInstanceError, match="patch_format"):
            instances.check_task_current(first)

    task_path = first / "task.json"
    original = task_path.read_text()
    task_path.write_text(original.replace("Improve", "Improve  ", 1))
    with pytest.raises(instances.StaleInstanceError, match="task_sha256"):
        instances.check_task_current(first)
    task_path.write_text(original)
    assert instances.check_task_current(first) == task

    supersede_instances([outcomes[0].instance_id], "test: retired")
    with pytest.raises(instances.StaleInstanceError, match="superseded"):
        instances.check_task_current(first)

    v0 = build_v0(toy_store)
    with pytest.raises(instances.StaleInstanceError, match="evidence-v0"):
        instances.check_task_current(root / v0[0].instance_id)


def test_decoy_evidence_is_the_decoy_models_own_draw(toy_store: ToyStore) -> None:  # noqa: F811
    outcomes = build_v1(toy_store)
    root = toy_store.private / "instances"
    by_seed = {o.seed: o.instance_id for o in outcomes}
    with np.load(toy_store.private / toy_store.dataset / "splits" / "V_evidence.npz") as archive:
        v_evidence = {key: archive[key] for key in archive.files}
    for seed, sibling_seed in DEFAULT_DECOYS.items():
        if seed not in by_seed:
            continue
        path, digest = instances.decoy_evidence(root / by_seed[seed])
        assert path == root / by_seed[sibling_seed] / "evidence.npz"
        decoy = load_instance(root / by_seed[sibling_seed])
        pred = instances.predict(decoy.forward, decoy.params, v_evidence["x"], 1024)
        redrawn = draw_evidence_v1(
            v_evidence["x"], v_evidence["y"], pred, seed=evidence_seed_v1(decoy.params_sha256, "deploy")
        )
        stored = load_evidence(path)
        for key in EVIDENCE_KEYS:
            np.testing.assert_array_equal(stored[key], redrawn.arrays[key])
        assert np.all(stored["pred_fail"] != stored["y_fail"])
        assert digest == json.loads((path.parent / "private.json").read_text())["evidence_content_sha256"]
    supersede_instances([by_seed[101]], "test: decoy instance retired")
    with pytest.raises(InstanceConflict, match="exactly one active instance"):
        instances.decoy_evidence(root / by_seed[100])
