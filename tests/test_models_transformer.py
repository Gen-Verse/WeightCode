"""Tests of the Transformer student (weightcode.models.transformer) and its numpy reference forward."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.data import seqops as S
from weightcode.evalkit.patch_check import load_forward
from weightcode.models import transformer as T
from weightcode.models.forward_gen import render_seq_forward, write_forward


def _config(**overrides: int) -> T.TfmConfig:
    base = dict(vocab_size=S.VOCAB_SIZE, seq_len=S.SEQ_LEN, n_out=S.N_SYMBOLS, pad_id=S.PAD)
    base.update(overrides)
    return T.TfmConfig(**base)  # type: ignore[arg-type]


def _forward(tmp_path: Path, config: T.TfmConfig):
    source = render_seq_forward(config.seq_len, config.vocab_size, config.n_out, config.n_heads, config.pad_id, T.LN_EPS)
    write_forward(tmp_path / "forward.py", source)
    forward, _ = load_forward(tmp_path / "forward.py")
    return forward


def _rows(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sym, length, opa, opb = S.OLD_ROWS.sample(rng, n)
    x, _ = S.encode_rows(sym, length, opa, opb, S.apply_ops(sym, length, opa, opb, S.old_rules()))
    return x


def _perturbed(model: T.SeqTransformer, seed: int) -> T.SeqTransformer:
    """Move every parameter off its initial value (LayerNorm gains and biases included) so parity covers them."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.3 * torch.randn(param.shape, generator=generator))
    return model


@pytest.mark.parametrize("overrides", [{}, {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 48}])
def test_numpy_forward_matches_torch(tmp_path: Path, overrides: dict[str, int]) -> None:
    torch.manual_seed(0)
    config = _config(**overrides)
    model = _perturbed(T.build_seq_model(config), 1).eval()
    params = T.export_seq_params(model)
    x = _rows(600)
    with torch.no_grad():
        reference = model(torch.from_numpy(x)).numpy().astype(np.float64)
    logits = _forward(tmp_path, config)(params, x)
    assert logits.dtype == np.float32 and logits.shape == (600, config.seq_len, config.n_out)
    assert np.max(np.abs(logits.astype(np.float64) - reference)) < 1e-4


def test_pad_keys_do_not_influence_other_positions(tmp_path: Path) -> None:
    torch.manual_seed(1)
    config = _config(d_model=32, n_layers=2, n_heads=2, d_mlp=48)
    params = T.export_seq_params(_perturbed(T.build_seq_model(config), 2))
    forward = _forward(tmp_path, config)
    x = _rows(8, seed=3)
    pad = x == S.PAD
    x_other = x.copy()
    x_other[pad] = S.PAD  # unchanged, then change the embedding row of PAD: non-PAD outputs must not move
    shifted = dict(params)
    shifted["tok_emb.weight"] = params["tok_emb.weight"].copy()
    shifted["tok_emb.weight"][S.PAD] += 5.0
    a, b = forward(params, x), forward(shifted, x_other)
    np.testing.assert_allclose(a[~pad], b[~pad], atol=1e-5)


def test_parameter_names_and_layer_list(tmp_path: Path) -> None:
    config = _config()
    params = T.export_seq_params(T.build_seq_model(config))
    expected = {"tok_emb.weight", "pos_emb.weight", "ln_f.weight", "ln_f.bias", "head.weight", "head.bias"}
    for i in range(1, config.n_layers + 1):
        for ln in ("ln1", "ln2"):
            expected |= {f"b{i}.{ln}.weight", f"b{i}.{ln}.bias"}
        expected |= {f"b{i}.attn.{p}.{k}" for p in "qkvo" for k in ("weight", "bias")}
        expected |= {f"b{i}.mlp.{p}.{k}" for p in ("fc1", "fc2") for k in ("weight", "bias")}
    assert set(params) == expected
    assert params["b1.attn.q.weight"].shape == (64, 64) and params["b2.mlp.fc1.weight"].shape == (256, 64)
    assert params["head.weight"].shape == (S.N_SYMBOLS, 64)
    layers = T.seq_layer_list(config, params)
    assert [layer["op"] for layer in layers[:6]] == [
        "embedding", "position_embedding", "layernorm", "attention", "layernorm", "mlp"
    ]
    assert layers[-1]["op"] == "linear" and layers[-2]["name"] == "ln_f"
    assert T.config_from_params(params, config.n_heads, config.pad_id) == config
    n_params = sum(v.size for v in params.values())
    assert 150_000 < n_params < 250_000
    json.dumps(layers)


def test_decay_groups_leave_embeddings_layernorm_and_biases_undecayed() -> None:
    model = T.build_seq_model(_config(d_model=32, n_layers=1, n_heads=2, d_mlp=48))
    decay, no_decay = T.decay_groups(model, 0.01)
    names = {id(p): n for n, p in model.named_parameters()}
    decayed = {names[id(p)] for p in decay["params"]}
    assert decayed == {n for n in names.values() if n.endswith(".weight") and ("attn" in n or "mlp" in n or n == "head.weight")}
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


def test_absent_token_rows_stay_bit_identical_under_adamw() -> None:
    torch.manual_seed(2)
    model = T.build_seq_model(_config(d_model=32, n_layers=1, n_heads=2, d_mlp=48))
    before = model.tok_emb.weight.detach().clone()
    optimizer = torch.optim.AdamW(T.decay_groups(model, 0.01), lr=1e-2)
    stream = S.TrainStream(0, 32, np.zeros(0, dtype=np.int64))
    for step in range(5):
        x, y = stream.batch(step)
        logits = model(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, S.N_SYMBOLS), torch.from_numpy(y).reshape(-1), ignore_index=S.IGNORE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = model.tok_emb.weight.detach()
    reserved = list(S.NEW_OP_IDS)
    assert torch.equal(after[reserved], before[reserved])
    assert not torch.equal(after[S.TOKEN_ID["OP_REV"]], before[S.TOKEN_ID["OP_REV"]])


def test_forward_refuses_bad_inputs(tmp_path: Path) -> None:
    config = _config(d_model=32, n_layers=1, n_heads=2, d_mlp=48)
    params = T.export_seq_params(T.build_seq_model(config))
    forward = _forward(tmp_path, config)
    x = _rows(2)
    with pytest.raises(TypeError):
        forward(params, x.astype(np.float32))
    with pytest.raises(ValueError):
        forward(params, x[:, :-1])
    bad = x.copy()
    bad[0, 1] = S.VOCAB_SIZE
    with pytest.raises(ValueError):
        forward(params, bad)
    assert forward(params, x[:0]).shape == (0, config.seq_len, config.n_out)
