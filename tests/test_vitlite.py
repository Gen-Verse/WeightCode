"""PREREG E11 student: ViT-Lite-7/4 (weightcode.models.vitlite) and its CIFAR recipe."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from weightcode.common.identity import params_sha256
from weightcode.models import vitlite as V

MEAN, STD = [0.49, 0.48, 0.45], [0.25, 0.24, 0.26]


def _images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)


def test_parameters_have_the_published_checkpoint_layout() -> None:
    shapes = V.param_shapes(10)
    assert len(shapes) == 85 and sum(math.prod(s) for s in shapes.values()) == 3_716_874
    model = V.ViTLite(V.ViTLiteConfig(), MEAN, STD)
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == shapes
    assert [round(getattr(b.drop_path, "drop_prob", 0.0), 6) for b in model.classifier.blocks] == [
        round(0.1 * i / 6, 6) for i in range(7)]
    assert model.classifier.dropout.p == 0.1 and model.classifier.blocks[0].self_attn.attn_drop.p == 0.0
    assert model.classifier.blocks[3].self_attn.proj_drop.p == 0.1 and model.classifier.blocks[3].dropout2.p == 0.1
    assert torch.equal(model.classifier.class_emb, torch.zeros(1, 1, 256))
    assert model.classifier.blocks[0].self_attn.qkv.bias is None


def test_raw_images_are_normalised_inside_and_export_round_trips() -> None:
    torch.manual_seed(0)
    model = V.ViTLite(V.ViTLiteConfig(), MEAN, STD).double().eval()
    x = _images(8)
    z = (torch.from_numpy(x).double() / 255.0 - torch.tensor(MEAN, dtype=torch.float64).view(1, 3, 1, 1)) / (
        torch.tensor(STD, dtype=torch.float64).view(1, 3, 1, 1))
    with torch.no_grad():
        assert torch.allclose(model(torch.from_numpy(x)), model.forward_normalized(z), rtol=0, atol=1e-12)
        feats = model.features(torch.from_numpy(x))
        assert torch.allclose(model.classifier.fc(feats), model(torch.from_numpy(x)), rtol=0, atol=1e-12)
    params = V.export_params(model)
    rebuilt = V.build_vitlite(V.ViTLiteConfig(), MEAN, STD, params, torch.float64)
    a, b = V.eval_logits(rebuilt, x), V.eval_logits(model, x)
    assert np.max(np.abs(a - b)) < 1e-5  # float32 export of float64 weights
    with pytest.raises(KeyError):
        V.check_params({k: v for k, v in params.items() if k != V.HEAD_BIAS}, 10)


def test_epoch_lr_is_the_scripts_schedule() -> None:
    c = V.CCTTrainConfig()
    assert [round(V.epoch_lr(c, e), 10) for e in range(6)] == [1e-4, 1.25e-4, round(5e-4 / 3, 10), 2.5e-4, 5e-4,
                                                               5e-4]
    assert math.isclose(V.epoch_lr(c, 199), 5e-4 * 0.5 * (1 + math.cos(math.pi * 194 / 195)))
    assert V.epoch_lr(c, 102) < V.epoch_lr(c, 101)


def test_label_smoothing_loss() -> None:
    logits = torch.randn(5, 10, dtype=torch.float64)
    target = torch.tensor([0, 3, 9, 2, 2])
    logp = torch.log_softmax(logits, -1)
    expected = (0.9 * -logp[torch.arange(5), target] + 0.1 * -logp.mean(-1)).mean()
    assert torch.allclose(V.label_smoothing_cross_entropy(logits, target, 0.1), expected)


def test_augmentation_yields_normalised_tensors() -> None:
    ds = V.AugmentedImages(_images(4), np.arange(4, dtype=np.int64), V.train_transform(MEAN, STD, 4))
    image, label = ds[2]
    assert image.dtype == torch.float32 and tuple(image.shape) == (3, 32, 32) and label == 2


@pytest.mark.parametrize("workers", [0, 2])
def test_training_is_reproducible(workers: int) -> None:
    x = _images(96, seed=1)
    y = np.random.default_rng(2).integers(0, 10, size=96).astype(np.int64)
    config = V.CCTTrainConfig(epochs=2, batch_size=32, num_workers=workers)
    digests = []
    for _ in range(2):
        result = V.train_vitlite(config, V.ViTLiteConfig(), 100, x, y, MEAN, STD, torch.device("cpu"))
        assert len(result.log) == 2 and all(np.isfinite(e["train_loss"]) for e in result.log)
        assert result.log[0]["lr"] == pytest.approx(1e-4) and result.log[1]["lr"] == pytest.approx(1.25e-4)
        digests.append(params_sha256(V.export_params(result.model)))
    assert digests[0] == digests[1]
    other = V.train_vitlite(config, V.ViTLiteConfig(), 101, x, y, MEAN, STD, torch.device("cpu"))
    assert params_sha256(V.export_params(other.model)) != digests[0]


def _load_forward(text: str):
    namespace: dict = {}
    exec(compile(text, "forward.py", "exec"), namespace)
    return namespace


def test_numpy_forward_matches_torch_float64() -> None:
    from weightcode.models.forward_gen import render_vitlite_forward

    text = render_vitlite_forward(MEAN, STD, 10, V.LN_EPS)
    for word in ("gaussian", "cifar", "vl7"):
        assert word not in text.lower()
    fwd = _load_forward(text)
    torch.manual_seed(3)
    model = V.ViTLite(V.ViTLiteConfig(), MEAN, STD).double().eval()
    with torch.no_grad():  # move away from the near-uniform init so that every path matters
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    params = V.export_params(model)
    ref = V.eval_logits(V.build_vitlite(V.ViTLiteConfig(), MEAN, STD, params, torch.float64), _images(300, 5))
    ours = np.asarray(fwd["forward"](params, _images(300, 5)), dtype=np.float64)
    assert ours.dtype == np.float64 and ours.shape == (300, 10)
    rel = np.max(np.abs(ours - ref)) / np.max(np.abs(ref))
    assert rel < 1e-6  # forward.py casts its float64 logits to float32 once
    torch_model = V.build_vitlite(V.ViTLiteConfig(), MEAN, STD, params, torch.float64)
    with torch.no_grad():
        f_ref = torch_model.features(torch.from_numpy(_images(7, 6))).numpy()
    assert np.allclose(fwd["features"](params, _images(7, 6)), f_ref, rtol=0, atol=1e-10)
    assert set(fwd["expected_shapes"]()) == set(V.param_shapes(10))
    with pytest.raises(KeyError):
        fwd["forward"]({k: v for k, v in params.items() if k != V.HEAD_BIAS}, _images(2))
