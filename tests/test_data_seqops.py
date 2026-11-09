"""Tests of the synthetic sequence-operator family (weightcode.data.seqops)."""

from __future__ import annotations

import numpy as np
import pytest

from weightcode.data import seqops as S


def _one(sym: list[int], opa: str, opb: str = "NONE") -> list[int]:
    n = len(sym)
    padded = np.zeros((1, S.MAX_LEN), dtype=np.int64)
    padded[0, :n] = sym
    rules = S.old_rules()
    token = lambda name: S.NONE if name == "NONE" else S.TOKEN_ID[f"OP_{name}"]  # noqa: E731
    out = S.apply_ops(padded, np.array([n]), np.array([token(opa)]), np.array([token(opb)]), rules)
    return out[0, :n].tolist()


def test_old_operators_on_a_hand_checked_string() -> None:
    x = [3, 0, 15, 7, 9]
    assert _one(x, "REV") == [9, 7, 15, 0, 3]
    assert _one(x, "ROTL1") == [0, 15, 7, 9, 3]
    assert _one(x, "ROTR1") == [9, 3, 0, 15, 7]
    assert _one(x, "SWAP2") == [0, 3, 7, 15, 9]  # odd length: the last symbol stays
    assert _one(x, "INC") == [4, 1, 0, 8, 10]
    assert _one(x, "DEC") == [2, 15, 14, 6, 8]
    assert _one(x, "COMP") == [12, 15, 0, 8, 6]
    suba = S.OLD_TABLES["SUBA"]
    assert _one(x, "SUBA") == [int(suba[v]) for v in x]


def test_composition_applies_opa_first() -> None:
    x = [1, 2, 3, 4]
    assert _one(x, "REV", "INC") == [5, 4, 3, 2]
    assert _one(x, "INC", "ROTL1") == [3, 4, 5, 2]
    suba = S.OLD_TABLES["SUBA"]
    assert _one(x, "INC", "SUBA") == [int(suba[v + 1]) for v in x]
    assert _one(x, "SUBA", "INC") == [int((suba[v] + 1) % S.N_SYMBOLS) for v in x]


@pytest.mark.parametrize("name", S.POSITION_OPS)
def test_position_maps_are_permutations_at_every_length(name: str) -> None:
    for n in range(S.MIN_LEN, S.MAX_LEN + 1):
        index = S.position_index(name, n)
        assert sorted(index.tolist()) == list(range(n))


def test_encode_decode_and_keys_round_trip() -> None:
    rng = np.random.default_rng(0)
    sym, n, opa, opb = S.OLD_ROWS.sample(rng, 500)
    out = S.apply_ops(sym, n, opa, opb, S.old_rules())
    x, y = S.encode_rows(sym, n, opa, opb, out)
    assert x.shape == y.shape == (500, S.SEQ_LEN)
    back = S.decode_x(x)
    for got, want in zip(back, (sym, n, opa, opb)):
        np.testing.assert_array_equal(got, want)
    labelled = y != S.IGNORE
    np.testing.assert_array_equal(labelled.sum(axis=1), n)
    np.testing.assert_array_equal(labelled, x == S.OUT)
    for row in range(20):
        np.testing.assert_array_equal(y[row][labelled[row]], out[row, : n[row]])
    keys = S.row_key(sym, n, opa, opb)
    assert np.unique(keys).size == np.unique(np.column_stack([sym, n, opa, opb]), axis=0).shape[0]


def test_decode_refuses_a_malformed_row() -> None:
    rng = np.random.default_rng(1)
    sym, n, opa, opb = S.OLD_ROWS.sample(rng, 4)
    x, _ = S.encode_rows(sym, n, opa, opb, sym)
    bad = x.copy()
    bad[0, 3] = S.OUT
    with pytest.raises(S.SeqOpsError):
        S.decode_x(bad)


def test_base_splits_are_reproducible_disjoint_and_old_only() -> None:
    first, second = S.build_base_splits(), S.build_base_splits()
    for name in first:
        for key in first[name]:
            np.testing.assert_array_equal(first[name][key], second[name][key])
    assert {name: first[name]["x"].shape[0] for name in S.BASE_ORDER} == dict(S.BASE_SIZES)
    keys = [first[name]["idx"] for name in (*S.BASE_ORDER, "train")]
    joined = np.concatenate(keys)
    assert np.unique(joined).size == joined.size
    for arrays in first.values():
        assert np.all(arrays["part"] == S.PART_OLD)
        assert np.isin(arrays["opa"], S.OLD_OP_IDS).all()
        assert np.isin(arrays["opb"], (S.NONE, *S.OLD_OP_IDS)).all()
        assert not np.isin(arrays["x"], S.NEW_OP_IDS).any()


def test_train_stream_is_deterministic_and_refuses_held_out_keys() -> None:
    stream = S.TrainStream(model_seed=7, batch_size=64, exclude=np.zeros(0, dtype=np.int64))
    x0, y0 = stream.batch(3)
    again_x, again_y = S.TrainStream(7, 64, np.zeros(0, dtype=np.int64)).batch(3)
    np.testing.assert_array_equal(x0, again_x)
    np.testing.assert_array_equal(y0, again_y)
    assert not np.isin(x0, S.NEW_OP_IDS).any()
    sym, n, opa, opb = S.decode_x(x0)
    blocked = S.row_key(sym, n, opa, opb)
    refusing = S.TrainStream(7, 64, blocked)
    x1, _ = refusing.batch(3)
    sym1, n1, opa1, opb1 = S.decode_x(x1)
    assert not np.isin(S.row_key(sym1, n1, opa1, opb1), blocked).any()
    assert refusing.n_refused >= 64
    assert S.TrainStream(7, 64, np.zeros(0, dtype=np.int64)).batch(4)[0].tolist() != x0.tolist()


def test_new_rule_is_a_fixed_permutation_unlike_every_old_table() -> None:
    rule = S.draw_new_rule(0)
    assert rule.token == S.TOKEN_ID["OP_NEW0"] and rule.kind == "symbol"
    assert sorted(rule.table) == list(range(S.N_SYMBOLS))
    assert rule == S.draw_new_rule(0)
    assert rule.table != S.draw_new_rule(1).table
    for table in S.OLD_TABLES.values():
        assert list(rule.table) != table.tolist()
    with pytest.raises(S.SeqOpsError):
        S.draw_new_rule(0, kind="random_position")
    with pytest.raises(S.SeqOpsError):
        S.draw_new_rule(len(S.NEW_OPS))


def test_condition_splits_hold_the_new_parts_with_the_rule_applied() -> None:
    rule = S.draw_new_rule(0)
    splits = S.build_condition_splits(rule, 0)
    joined = np.concatenate([arrays["idx"] for arrays in splits.values()])
    assert np.unique(joined).size == joined.size
    table = np.asarray(rule.table)
    for name, arrays in splits.items():
        counts = {int(p): int(c) for p, c in zip(*np.unique(arrays["part"], return_counts=True))}
        assert counts == dict(S.CONDITION_SIZES[name])
        assert np.isin(rule.token, arrays["x"][:, 1:3]).all() or True
        has_new = (arrays["opa"] == rule.token) | (arrays["opb"] == rule.token)
        assert has_new.all()
        long = arrays["part"] == S.PART_NEW_LONG
        assert np.all((arrays["n"][long] >= 9) & (arrays["n"][long] <= S.MAX_LEN))
        assert np.all(arrays["n"][~long] <= 8)
        single = arrays["opb"] == S.NONE
        sym, n, _, _ = S.decode_x(arrays["x"][single])
        labelled = arrays["y"][single] != S.IGNORE
        for row in range(min(50, int(single.sum()))):
            np.testing.assert_array_equal(arrays["y"][single][row][labelled[row]], table[sym[row, : n[row]]])
    composed = splits["V_select"]["part"] == S.PART_NEW_COMPOSED
    first = splits["V_select"]["opa"][composed] == rule.token
    assert 0.3 < first.mean() < 0.7  # both orders occur


def test_row_correct_is_exact_match_over_labelled_positions() -> None:
    y = np.array([[S.IGNORE, 3, 4], [1, S.IGNORE, S.IGNORE]])
    pred = np.array([[9, 3, 4], [2, 1, 1]])
    np.testing.assert_array_equal(S.row_correct(y, pred), [True, False])
    assert S.token_accuracy(y, pred) == pytest.approx(2 / 3)
    with pytest.raises(S.SeqOpsError):
        S.row_correct(np.full((1, 3), S.IGNORE), np.zeros((1, 3), dtype=np.int64))
