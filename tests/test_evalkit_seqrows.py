"""The sequence-row path of evalkit: per-position predictions, exact-match rows, joint patch checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.data import seqops as S
from weightcode.evalkit.metrics import IGNORE, rows_right, sequence_readout, summarize, token_accuracy
from weightcode.evalkit.patch_check import JointPatchCheckOracle, load_forward, predict
from weightcode.models import transformer as T
from weightcode.models.forward_gen import render_seq_forward, write_forward


def test_rows_right_is_exact_match_and_matches_labels_for_one_dimensional_targets() -> None:
    y = np.array([[IGNORE, 3, 4], [1, IGNORE, IGNORE], [2, 2, IGNORE]])
    pred = np.array([[0, 3, 4], [1, 7, 7], [2, 3, 9]])
    np.testing.assert_array_equal(rows_right(y, pred), [True, True, False])
    np.testing.assert_array_equal(rows_right(np.array([1, 2]), np.array([1, 3])), [True, False])
    assert S.IGNORE == IGNORE
    with pytest.raises(ValueError):
        rows_right(np.full((1, 2), IGNORE), np.zeros((1, 2), dtype=np.int64))
    assert token_accuracy(y, pred) == pytest.approx(4 / 5)  # 5 labelled positions, 4 right


def test_summarize_counts_rows_of_sequences() -> None:
    y = np.array([[1, IGNORE], [2, IGNORE], [3, 3]])
    base = np.array([[1, 0], [0, 0], [3, 3]])  # right, wrong, right
    child = np.array([[0, 0], [2, 9], [3, 3]])  # wrong, right, right
    summary = summarize(y, base, child)
    assert (summary.n, summary.fixed, summary.broken, summary.score, summary.n_base_errors) == (3, 1, 1, 0, 1)
    with pytest.raises(ValueError):
        summarize(y, base, child, a=1, b=2)


def _student(tmp_path: Path):
    torch.manual_seed(0)
    config = T.TfmConfig(S.VOCAB_SIZE, S.SEQ_LEN, S.N_SYMBOLS, S.PAD, d_model=32, n_layers=1, n_heads=2, d_mlp=48)
    params = T.export_seq_params(T.build_seq_model(config))
    source = render_seq_forward(config.seq_len, config.vocab_size, config.n_out, config.n_heads, config.pad_id, T.LN_EPS)
    write_forward(tmp_path / "forward.py", source)
    forward, _ = load_forward(tmp_path / "forward.py")
    return params, forward


def test_predict_returns_one_prediction_per_position(tmp_path: Path) -> None:
    params, forward = _student(tmp_path)
    x = S.build_base_splits()["V_check"]["x"][:50]
    pred = predict(forward, params, x, batch_size=16)
    assert pred.shape == x.shape and pred.dtype == np.int64
    np.testing.assert_array_equal(pred, forward(params, x).argmax(axis=-1))


def test_joint_patch_check_on_new_and_old_sequence_sets(tmp_path: Path) -> None:
    params, forward = _student(tmp_path)
    old = S.build_base_splits()["V_check"]
    new = S.build_condition_splits(S.draw_new_rule(0), 0)["V_check"]
    oracle = JointPatchCheckOracle(new["x"][:200], new["y"][:200], old["x"][:200], old["y"][:200], forward, params, budget=2)
    same = oracle.query(params, "noop")
    assert same["score"] == 0 and same["n"] == 400 and same["n_new"] == 200 and same["n_old"] == 200
    child = {key: value.copy() for key, value in params.items()}
    child["head.bias"] = child["head.bias"] + np.linspace(-3, 3, S.N_SYMBOLS, dtype=np.float32)
    changed = oracle.query(child, "bias")
    assert changed["fixed"] - changed["broken"] == changed["score"]
    assert oracle.queries_remaining == 0


def test_sequence_readout_reports_each_part() -> None:
    y = np.array([[1, IGNORE], [2, IGNORE], [3, 3], [4, 4]])
    base = np.array([[1, 0], [0, 0], [3, 3], [4, 0]])
    child = np.array([[1, 0], [2, 0], [3, 0], [4, 4]])
    readout = sequence_readout(y, base, child, part=np.array([1, 1, 4, 4]))
    assert readout["delta_acc"] == pytest.approx(1 / 4)
    assert readout["parts"]["1"]["fixed"] == 1 and readout["parts"]["4"]["broken"] == 1
    assert readout["parts"]["4"]["fixed"] == 1
