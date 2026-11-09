"""PREREG E14 student: layout, numpy forward parity, model_spec.json and the training recipe on a small set."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("minigrid")

from weightcode.data import e14 as D  # noqa: E402
from weightcode.evalkit.patch_check import load_forward  # noqa: E402
from weightcode.models import e14_policy as P  # noqa: E402
from weightcode.models.parity import check_logit_parity  # noqa: E402


@pytest.fixture(scope="module")
def forward(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("fwd") / "forward.py"
    path.write_text(P.forward_source(), encoding="utf-8")
    return load_forward(path)[0]


@pytest.fixture(scope="module")
def rows() -> np.ndarray:
    demos = D.demo_arrays(D.bot_episode("old", s) for s in range(40))
    new = D.demo_arrays(D.bot_episode("new", s) for s in range(2_000_000, 2_000_010))
    return np.concatenate([demos["x"], new["x"]]).astype(np.int64)


def test_layout_and_sizes() -> None:
    torch.manual_seed(0)
    model = P.FilmPolicy(P.SIZES["base"])
    params = P.export_params(model)
    assert {k: v.shape for k, v in params.items()} == P.param_shapes(P.SIZES["base"])
    assert sum(v.size for v in params.values()) == 1_151_799
    assert P.config_from_params(params) == P.SIZES["base"]
    nxt = P.export_params(P.FilmPolicy(P.SIZES["next"]))
    assert P.config_from_params(nxt) == P.SIZES["next"]


@pytest.mark.parametrize("size", ["base", "next"])
def test_numpy_forward_parity(forward, rows: np.ndarray, size: str) -> None:
    torch.manual_seed(1)
    model = P.FilmPolicy(P.SIZES[size])
    with torch.no_grad():  # move FiLM and LayerNorm off their identity initialisation so every path is exercised
        for name, tensor in model.named_parameters():
            if name.startswith(("film", "ln.")):
                tensor.add_(0.3 * torch.randn_like(tensor))
    params = P.export_params(model)
    ours = np.asarray(forward(params, rows), dtype=np.float64)
    ref64 = P.eval_logits(P.build_policy(P.SIZES[size], params, torch.float64), rows)
    ref32 = P.eval_logits(P.build_policy(P.SIZES[size], params, torch.float32), rows)
    result = check_logit_parity(ours, ref64, ref32)
    assert result["float64_max_rel_diff"] <= result["float64_tolerance"]


def test_forward_refuses_bad_rows(forward, rows: np.ndarray) -> None:
    params = P.export_params(P.FilmPolicy(P.SIZES["base"]))
    bad = rows[:2].copy()
    bad[0, 0] = 11
    with pytest.raises(ValueError):
        forward(params, bad)
    with pytest.raises(ValueError):
        forward(params, rows[:2, :-1])
    with pytest.raises(ValueError):
        forward({k: v for k, v in params.items() if k != "ln.bias"}, rows[:2])


def test_spec_is_public_and_marks_reserved_words() -> None:
    params = P.export_params(P.FilmPolicy(P.SIZES["base"]))
    spec = P.build_spec("m00000000", params)
    words = {w["word"]: w["meaning"] for w in spec["input"]["instruction"]["words"]}
    assert words["pick"].startswith("reserved word") and words["up"].startswith("reserved word")
    assert [a["name"] for a in spec["actions"]] == list(D.ACTIONS)
    text = str(spec).lower()
    assert not any(word in text for word in D.LEAK_WORDS)


def test_training_recipe_small(rows: np.ndarray) -> None:
    demos = D.demo_arrays(D.bot_episode("old", s) for s in range(200))
    x, y = demos["x"].astype(np.int64), demos["y"]
    config = P.E14TrainConfig(epochs=2, warmup_steps=2)
    torch.use_deterministic_algorithms(True)
    a = P.train_policy(config, P.SIZES["base"], x, y, seed=100)
    b = P.train_policy(config, P.SIZES["base"], x, y, seed=100)
    pa, pb = P.export_params(a.model), P.export_params(b.model)
    assert all(np.array_equal(pa[k], pb[k]) for k in pa)
    assert a.reserved_rows_unchanged
    torch.manual_seed(100)
    init = P.export_params(P.FilmPolicy(P.SIZES["base"]))
    assert np.array_equal(init["word_emb.weight"][list(D.RESERVED_IDS)], pa["word_emb.weight"][list(D.RESERVED_IDS)])
    with pytest.raises(ValueError):
        P.train_policy(config, P.SIZES["base"], rows, np.zeros(len(rows), np.int64), seed=0)
    assert P.lr_at(config, 0, 100) == pytest.approx(config.lr / 2)
    assert P.lr_at(config, 99, 100) == pytest.approx(config.lr * config.min_lr_ratio)
    with pytest.raises(ValueError):
        P.train_policy(config, P.SIZES["base"], x, y, seed=100, device="mps")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_recipe_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """PREREG E14 device amendment: CUDA training under the deterministic settings is repeatable, starts from the
    CPU initialisation, keeps the reserved rows and returns a CPU model."""
    from weightcode.models.train import configure_determinism

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    demos = D.demo_arrays(D.bot_episode("old", s) for s in range(200))
    x, y = demos["x"].astype(np.int64), demos["y"]
    config = P.E14TrainConfig(epochs=2, warmup_steps=2)
    configure_determinism(100)
    a = P.train_policy(config, P.SIZES["base"], x, y, seed=100, device="cuda")
    b = P.train_policy(config, P.SIZES["base"], x, y, seed=100, device="cuda")
    pa, pb = P.export_params(a.model), P.export_params(b.model)
    assert all(np.array_equal(pa[k], pb[k]) for k in pa)
    assert a.reserved_rows_unchanged
    assert all(p.device.type == "cpu" for p in a.model.parameters())
    torch.manual_seed(100)
    init = P.export_params(P.FilmPolicy(P.SIZES["base"]))
    assert np.array_equal(init["word_emb.weight"][list(D.RESERVED_IDS)], pa["word_emb.weight"][list(D.RESERVED_IDS)])
    cpu = P.export_params(P.train_policy(config, P.SIZES["base"], x, y, seed=100).model)
    assert max(float(np.abs(pa[k] - cpu[k]).max()) for k in pa) < 1e-2  # same math, other kernels
