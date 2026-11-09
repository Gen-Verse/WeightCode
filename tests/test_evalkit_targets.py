"""Tests for the pre-registered target rule targets-v0 (weightcode.evalkit.targets)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    DEV_PLAN,
    EXPECTED_CHOSEN,
    FIVE_SEEDS,
    ToyStore,
    build_toy_store,
    toy_store,
)

from weightcode.evalkit.targets import (
    K_FAIL,
    MIN_VCHECK,
    RULE_TEXT,
    SeedSelection,
    TargetsConflict,
    confusion_counts,
    parse_seeds,
    rank_pairs,
    read_json,
    read_model_index,
    run_target_selection,
    select_entries,
    select_pairs,
    targets_path,
)


def test_rank_pairs_orders_by_count_then_pair() -> None:
    confusion = np.array([[5, 3, 3], [0, 5, 7], [3, 0, 5]])
    assert rank_pairs(confusion) == [(1, 2, 7), (0, 1, 3), (0, 2, 3), (2, 0, 3), (1, 0, 0), (2, 1, 0)]


def test_confusion_counts_by_hand_and_range_check() -> None:
    y = np.array([0, 0, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 0, 0, 2])
    assert confusion_counts(y, pred, 3).tolist() == [[1, 1, 0], [0, 1, 0], [2, 0, 1]]
    with pytest.raises(ValueError):
        confusion_counts(y, np.array([0, 1, 1, 0, 0, 3]), 3)


def test_select_pairs_feasibility_reasons_and_first_two_accepted() -> None:
    vselect = np.zeros((4, 4), dtype=np.int64)
    vselect[0, 1], vselect[1, 0], vselect[2, 3], vselect[3, 2], vselect[3, 0] = 40, 30, 30, 20, 10
    np.fill_diagonal(vselect, 100)
    vevidence = np.zeros_like(vselect)
    vevidence[0, 1], vevidence[1, 0], vevidence[2, 3], vevidence[3, 2], vevidence[3, 0] = 16, 30, 15, 20, 20
    vcheck = np.zeros_like(vselect)
    vcheck[0, 1], vcheck[1, 0], vcheck[2, 3], vcheck[3, 2], vcheck[3, 0] = 10, 9, 30, 10, 10
    decisions = select_pairs(vselect, vevidence, vcheck)
    assert [(d.a, d.b) for d in decisions[:5]] == [(0, 1), (1, 0), (2, 3), (3, 2), (3, 0)]
    assert [d.accepted for d in decisions[:5]] == [True, False, False, True, True]
    assert [d.chosen for d in decisions] == [True, False, False, True] + [False] * 8
    assert decisions[1].reasons == (f"V_check count 9 < {MIN_VCHECK}",)
    assert decisions[2].reasons == (f"V_evidence count 15 < k_fail {K_FAIL}",)
    assert "not chosen" in decisions[4].reasons[-1]
    assert decisions[0].vselect_n_a == 140
    assert all(d.vselect_count == 0 and not d.accepted for d in decisions[5:])


def test_select_pairs_records_a_shortfall_without_relaxing() -> None:
    vselect = np.array([[50, 30], [10, 50]])
    vevidence = np.array([[20, 16], [5, 20]])
    vcheck = np.array([[40, 12], [20, 40]])
    decisions = select_pairs(vselect, vevidence, vcheck)
    assert [d.chosen for d in decisions] == [True, False]


def test_targets_on_toy_store(toy_store: ToyStore) -> None:  # noqa: F811
    outcomes = run_target_selection(toy_store.dataset, selection=DEV_PLAN)
    assert [(o.split_role, o.seed) for o in outcomes] == [("dev", 100), ("dev", 101), ("main", 0), ("main", 1)]
    for seed in (2, 3, 4):
        assert not targets_path(toy_store.dataset, toy_store.ids("base")[seed]).exists()
    for outcome in outcomes:
        assert outcome.written and outcome.shortfall == 0
        assert [(p["a"], p["b"]) for p in outcome.chosen] == EXPECTED_CHOSEN
        record = read_json(outcome.path)
        assert record["rule"] == RULE_TEXT and record["rule_id"] == "targets-v0"
        assert record["split_role"] == outcome.split_role
        assert len(record["ranked_pairs"]) == 12
        assert record["vselect_acc_forward"] == pytest.approx(0.75)
        assert record["vselect_confusion_check"] == "equal to model private.json"
        first, second = record["chosen"]
        assert (first["rank"], first["vselect_count"], first["vselect_n_a"]) == (1, 30, 100)
        assert (first["vevidence_count"], first["vcheck_count"]) == (20, 12)
        assert (second["rank"], second["vselect_count"], second["vevidence_count"], second["vcheck_count"]) == (
            4, 20, 17, 11)
        by_pair = {(r["a"], r["b"]): r for r in record["ranked_pairs"]}
        assert by_pair[(1, 0)]["reasons"] == ["V_check count 5 < 10"]
        assert by_pair[(2, 3)]["reasons"] == ["V_evidence count 10 < k_fail 16"]
        assert by_pair[(1, 0)]["rank"] == 2 and by_pair[(2, 3)]["rank"] == 3


def test_split_role_filter_and_ceiling_models_get_no_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_toy_store(tmp_path / "store", ceiling_seeds=(0, 1))
    store.activate(monkeypatch)
    dev_only = run_target_selection(store.dataset, selection=DEV_PLAN, split_roles=("dev",))
    assert sorted(o.seed for o in dev_only) == [100, 101]
    assert not targets_path(store.dataset, store.ids("base")[0]).exists()
    run_target_selection(store.dataset, selection=DEV_PLAN)
    for model_id in store.ids("ceiling").values():
        assert not targets_path(store.dataset, model_id).exists()
    with pytest.raises(ValueError):
        run_target_selection(store.dataset, selection=DEV_PLAN, split_roles=("ceiling",))


def test_rerun_keeps_identical_targets_and_refuses_a_different_one(toy_store: ToyStore) -> None:  # noqa: F811
    run_target_selection(toy_store.dataset, selection=DEV_PLAN)
    path = targets_path(toy_store.dataset, toy_store.ids("dev")[100])
    before = path.read_text()
    assert not any(o.written for o in run_target_selection(toy_store.dataset, selection=DEV_PLAN))
    assert path.read_text() == before
    record = json.loads(before)
    record["chosen"] = record["chosen"][:1]
    path.write_text(json.dumps(record))
    with pytest.raises(TargetsConflict):
        run_target_selection(toy_store.dataset, selection=DEV_PLAN)


def test_forward_and_checkpoint_disagreement_is_refused(toy_store: ToyStore) -> None:  # noqa: F811
    index_path = toy_store.private / toy_store.dataset / "models" / "index.json"
    index = json.loads(index_path.read_text())
    assert index[0]["role"] == "base"
    index[0]["vselect_acc"] = 0.74
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="V_select accuracy"):
        run_target_selection(toy_store.dataset, selection=DEV_PLAN, split_roles=("main",))


def test_confusion_recorded_at_export_must_match(toy_store: ToyStore) -> None:  # noqa: F811
    model_id = toy_store.ids("dev")[100]
    private_path = toy_store.private / toy_store.dataset / "models" / model_id / "private.json"
    private = json.loads(private_path.read_text())
    private["vselect_confusion"][0][1] += 1
    private["vselect_confusion"][0][0] -= 1
    private_path.write_text(json.dumps(private))
    with pytest.raises(ValueError, match="recorded at export"):
        run_target_selection(toy_store.dataset, selection=DEV_PLAN, split_roles=("dev",))


def test_forward_hash_mismatch_is_refused(toy_store: ToyStore) -> None:  # noqa: F811
    forward_path = toy_store.private / toy_store.dataset / "forward.py"
    forward_path.write_text(forward_path.read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="forward_sha256"):
        run_target_selection(toy_store.dataset, selection=DEV_PLAN)


def test_seed_selection_parsing_and_validation(toy_store: ToyStore) -> None:  # noqa: F811
    assert parse_seeds("0,1") == (0, 1) and parse_seeds(" 2 , 3 ") == (2, 3) and parse_seeds("") == ()
    for bad in ("0,x", "0,,1", "-1", "1.5"):
        with pytest.raises(ValueError):
            parse_seeds(bad)
    with pytest.raises(ValueError, match="disjoint"):
        SeedSelection(main=(0, 1), dev=(1, 100))
    with pytest.raises(ValueError, match="repeat"):
        SeedSelection(main=(0, 0), dev=())
    with pytest.raises(TypeError):
        SeedSelection(main=[0, 1], dev=())  # type: ignore[arg-type]
    entries = read_model_index(toy_store.dataset)
    chosen = select_entries(entries, DEV_PLAN, ("dev", "main"))
    assert [(e.role, e.seed) for e in chosen] == [("dev", 100), ("dev", 101), ("base", 0), ("base", 1)]
    assert [e.seed for e in select_entries(entries, FIVE_SEEDS, ("main",))] == [0, 1, 2, 3, 4]
    with pytest.raises(LookupError, match="main seed 7"):
        select_entries(entries, SeedSelection(main=(0, 7), dev=()), ("main",))
    assert select_entries(entries, SeedSelection(main=(0, 7), dev=()), ("dev",)) == []
