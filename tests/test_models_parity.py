"""Numpy reference forward vs PyTorch: parity on small random models of both architectures, strictness, spec."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.evalkit.patch_check import load_forward
from weightcode.models.arch import build_model, export_params, head_layer, load_params_into
from weightcode.models.forward_gen import render_forward, write_forward
from weightcode.models.spec import FORBIDDEN_TERMS, build_model_spec, check_public
from weightcode.models.train import augment_crop_flip

TOL = 1e-4
REL_TOL = 1e-5
"""The torch reference runs in float32, so its rounding grows with the logits; past |logit| 10 the bound is
relative (about sqrt(n) float32 epsilons for the longest dot products). The production gate stays absolute:
export_model refuses a trained model whose numpy logits differ from float32 torch by more than 1e-4."""
CASES = {
    "mlp_gray": ("mlp", (1, 28, 28), [0.286], [0.353]),
    "mlp_color": ("mlp", (3, 28, 28), [0.11, 0.12, 0.09], [0.29, 0.30, 0.27]),
    "cnn_color": ("cnn", (3, 32, 32), [0.49, 0.48, 0.45], [0.25, 0.24, 0.26]),
    "cnn4_color": ("cnn4", (3, 32, 32), [0.49, 0.48, 0.45], [0.25, 0.24, 0.26]),
}


def _forward_file(tmp_path: Path, shape: tuple[int, int, int], mean: list[float], std: list[float]):
    path = tmp_path / "forward.py"
    write_forward(path, render_forward(shape, mean, std, 10))
    forward, _ = load_forward(path)
    return forward


def _model(arch: str, shape, mean, std, seed: int, scale: float) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = build_model(arch, shape, 10, mean, std)
    if scale != 1.0:
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(scale * torch.randn(param.shape, generator=generator) * param.std())
    return model.eval()


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_numpy_forward_matches_torch(tmp_path: Path, case: str, scale: float) -> None:
    arch, shape, mean, std = CASES[case]
    forward = _forward_file(tmp_path, shape, mean, std)
    model = _model(arch, shape, mean, std, seed=7, scale=scale)
    params = export_params(model)
    x = np.random.default_rng(3).integers(0, 256, size=(37, *shape), dtype=np.uint8)
    with torch.no_grad():
        reference = model(torch.from_numpy(x)).numpy()
    logits = forward(params, x)
    assert logits.dtype == np.float32 and logits.shape == (37, 10)
    tolerance = max(TOL, REL_TOL * float(np.max(np.abs(reference))))
    assert np.max(np.abs(logits.astype(np.float64) - reference)) <= tolerance
    assert np.array_equal(np.argmax(logits, axis=1), np.argmax(reference, axis=1))


def test_parameter_names_and_shapes() -> None:
    mlp = build_model("mlp", (1, 28, 28), 10, [0.5], [0.5])
    assert {k: tuple(v.shape) for k, v in mlp.state_dict().items()} == {
        "fc1.weight": (128, 784), "fc1.bias": (128,), "fc2.weight": (128, 128), "fc2.bias": (128,),
        "fc3.weight": (10, 128), "fc3.bias": (10,),
    }
    assert build_model("mlp", (3, 28, 28), 10, [0.5] * 3, [0.5] * 3).state_dict()["fc1.weight"].shape == (128, 2352)
    cnn = build_model("cnn", (3, 32, 32), 10, [0.5] * 3, [0.5] * 3)
    assert {k: tuple(v.shape) for k, v in cnn.state_dict().items()} == {
        "conv1.weight": (32, 3, 3, 3), "conv1.bias": (32,), "conv2.weight": (64, 32, 3, 3), "conv2.bias": (64,),
        "conv3.weight": (64, 64, 3, 3), "conv3.bias": (64,), "fc1.weight": (128, 1024), "fc1.bias": (128,),
        "fc2.weight": (10, 128), "fc2.bias": (10,),
    }
    cnn4 = build_model("cnn4", (3, 32, 32), 10, [0.5] * 3, [0.5] * 3)
    assert {k: tuple(v.shape) for k, v in cnn4.state_dict().items()} == {
        "conv1.weight": (32, 3, 3, 3), "conv1.bias": (32,), "conv2.weight": (64, 32, 3, 3), "conv2.bias": (64,),
        "conv3.weight": (128, 64, 3, 3), "conv3.bias": (128,), "conv4.weight": (128, 128, 3, 3),
        "conv4.bias": (128,), "fc1.weight": (10, 512), "fc1.bias": (10,),
    }
    assert [head_layer(arch) for arch in ("mlp", "cnn", "cnn4")] == ["fc3", "fc2", "fc1"]
    with pytest.raises(ValueError, match="too small"):
        build_model("cnn4", (3, 8, 8), 10, [0.5] * 3, [0.5] * 3)


def test_export_reload_roundtrip_is_bitwise() -> None:
    model = _model("cnn", (3, 32, 32), [0.5] * 3, [0.25] * 3, seed=1, scale=1.0)
    params = export_params(model)
    clone = load_params_into(build_model("cnn", (3, 32, 32), 10, [0.5] * 3, [0.25] * 3), params)
    for key, value in export_params(clone).items():
        assert value.dtype == np.float32 and np.array_equal(value, params[key])


def test_forward_chunking_does_not_change_results(tmp_path: Path) -> None:
    arch, shape, mean, std = CASES["mlp_gray"]
    forward = _forward_file(tmp_path, shape, mean, std)
    params = export_params(_model(arch, shape, mean, std, seed=2, scale=1.0))
    x = np.random.default_rng(0).integers(0, 256, size=(1500, *shape), dtype=np.uint8)
    whole = forward(params, x)
    parts = np.concatenate([forward(params, x[:700]), forward(params, x[700:])])
    np.testing.assert_allclose(whole, parts, rtol=0, atol=1e-6)
    assert forward(params, x[:0]).shape == (0, 10)


def test_forward_rejects_malformed_inputs(tmp_path: Path) -> None:
    arch, shape, mean, std = CASES["mlp_gray"]
    forward = _forward_file(tmp_path, shape, mean, std)
    params = export_params(_model(arch, shape, mean, std, seed=0, scale=1.0))
    x = np.zeros((2, *shape), dtype=np.uint8)
    with pytest.raises(TypeError):
        forward(params, x.astype(np.float32))
    with pytest.raises(ValueError):
        forward(params, np.zeros((2, 3, 28, 28), dtype=np.uint8))
    with pytest.raises(KeyError):
        forward({**params, "extra.weight": params["fc1.weight"]}, x)
    with pytest.raises(KeyError):
        forward({k: v for k, v in params.items() if k != "fc2.bias"}, x)
    renumbered = {k.replace("fc3", "fc4"): v for k, v in params.items()}
    with pytest.raises(KeyError):
        forward(renumbered, x)
    with pytest.raises(ValueError):
        forward({**params, "fc2.weight": params["fc2.weight"][:, :64]}, x)


def test_rendered_forward_imports_only_numpy() -> None:
    for arch, shape, mean, std in CASES.values():
        tree = ast.parse(render_forward(shape, mean, std, 10))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add((node.module or "").split(".")[0])
        assert modules <= {"numpy", "__future__"}


def test_write_forward_refuses_to_replace_different_content(tmp_path: Path) -> None:
    path = tmp_path / "forward.py"
    first = write_forward(path, render_forward((1, 28, 28), [0.2], [0.3], 10))
    assert write_forward(path, render_forward((1, 28, 28), [0.2], [0.3], 10)) == first
    with pytest.raises(FileExistsError):
        write_forward(path, render_forward((1, 28, 28), [0.2], [0.31], 10))


@pytest.mark.parametrize("case", sorted(CASES))
def test_model_spec_is_public_and_consistent(case: str) -> None:
    arch, shape, mean, std = CASES[case]
    params = export_params(_model(arch, shape, mean, std, seed=0, scale=1.0))
    spec = build_model_spec("m0123abcd", arch, shape, mean, std, [str(i) for i in range(10)], params)
    assert spec["schema"] == "weightcode.model_spec.v0"
    named = [layer for layer in spec["layers"] if "name" in layer]
    assert {layer["weight"] for layer in named} | {layer["bias"] for layer in named} == set(params)
    for layer in named:
        assert layer["shape_w"] == list(params[layer["weight"]].shape)
        assert layer["shape_b"] == list(params[layer["bias"]].shape)
    if arch == "cnn":
        convs = [layer for layer in spec["layers"] if layer["op"] == "conv2d"]
        pools = [layer for layer in spec["layers"] if layer["op"] == "maxpool2d"]
        assert [c["out_shape"] for c in convs] == [[32, 32, 32], [64, 16, 16], [64, 8, 8]]
        assert all(c["stride"] == [1, 1] and c["padding"] == [1, 1] and c["kernel"] == [3, 3] for c in convs)
        assert [p["out_shape"] for p in pools] == [[32, 16, 16], [64, 8, 8], [64, 4, 4]]
    if arch == "cnn4":
        convs = [layer for layer in spec["layers"] if layer["op"] == "conv2d"]
        pools = [layer for layer in spec["layers"] if layer["op"] == "maxpool2d"]
        assert [c["out_shape"] for c in convs] == [[32, 32, 32], [64, 16, 16], [128, 8, 8], [128, 4, 4]]
        assert [p["out_shape"] for p in pools] == [[32, 16, 16], [64, 8, 8], [128, 4, 4], [128, 2, 2]]
        linears = [layer for layer in spec["layers"] if layer["op"] == "linear"]
        assert [layer["name"] for layer in linears] == ["fc1"] and linears[0]["shape_w"] == [10, 512]
        assert spec["layers"][-1]["op"] == "linear"
    flatten = next(layer for layer in spec["layers"] if layer["op"] == "flatten")
    assert flatten["out_features"] == params["fc1.weight"].shape[1]
    with pytest.raises(ValueError):
        check_public({**spec, "note": f"trained with {FORBIDDEN_TERMS[0]} 3"})


def test_augmentation_crop_and_flip_geometry() -> None:
    x = torch.arange(2 * 1 * 4 * 4, dtype=torch.uint8).reshape(2, 1, 4, 4)
    generator = torch.Generator().manual_seed(0)
    out = augment_crop_flip(x, pad=0, generator=generator)
    for i in range(2):
        assert torch.equal(out[i], x[i]) or torch.equal(out[i], x[i].flip(-1))
    padded = augment_crop_flip(x, pad=2, generator=torch.Generator().manual_seed(1))
    assert padded.shape == x.shape and padded.dtype == torch.uint8
    again = augment_crop_flip(x, pad=2, generator=torch.Generator().manual_seed(1))
    assert torch.equal(padded, again)


@pytest.mark.skipif(not os.environ.get("WEIGHTCODE_PRIVATE"), reason="exported models need $WEIGHTCODE_PRIVATE")
def test_exported_models_have_consistent_records() -> None:
    """If models were exported, their files, hashes and index entries agree (no data is read)."""
    import json

    from weightcode.common.identity import load_params, params_sha256
    from weightcode.data.registry import DATASETS
    from weightcode.data.store import models_dir
    from weightcode.models.export import read_index

    checked = 0
    for dataset in DATASETS:
        if not models_dir(dataset).exists():
            continue
        for entry in read_index(dataset):
            folder = models_dir(dataset) / entry["model_id"]
            private = json.loads((folder / "private.json").read_text())
            spec = json.loads((folder / "model_spec.json").read_text())
            assert params_sha256(load_params(folder / "model.npz")) == entry["params_sha256"] == private["params_sha256"]
            assert private["parity"]["max_abs_diff"] <= TOL
            assert spec["model_id"] == entry["model_id"]
            check_public(spec)
            checked += 1
    if checked == 0:
        pytest.skip("no exported models yet")
