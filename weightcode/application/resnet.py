"""Explicit torchvision ResNet-18 adapter; no arbitrary checkpoint unpickling."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from weightcode.application.bundles import INPUT, bind, compatibility, new_dir, sha, write
from weightcode.common.identity import params_sha256, save_params


def forward_module(n_classes: int) -> types.ModuleType:
    from weightcode.models.resnet18native import render_forward_u8
    mod = types.ModuleType("weightcode_resnet18_forward")
    exec(compile(render_forward_u8(n_classes), "<weightcode-resnet18-adapter>", "exec"), mod.__dict__)
    return mod


def reference(params: dict, buffers: dict):
    import torch
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=len(params["fc.bias"]))
    state = {k: torch.from_numpy(v.copy()) for k, v in {**params, **buffers}.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def reference_logits(model, x: np.ndarray, *, dtype=None) -> np.ndarray:
    import torch
    dtype = dtype or torch.float32
    model = model.to(dtype).eval()
    mean = torch.tensor(INPUT["mean"], dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(INPUT["std"], dtype=dtype).view(1, 3, 1, 1)
    outputs = []
    with torch.no_grad():
        for image in x:
            value = (torch.from_numpy(image.copy()).unsqueeze(0).to(dtype) / 255.0 - mean) / std
            outputs.append(model(value).to(torch.float64).numpy())
    return np.concatenate(outputs)


def parity(params: dict, buffers: dict, x: np.ndarray) -> dict:
    import torch
    from weightcode.models.parity import check_logit_parity

    mine = forward_module(len(params["fc.bias"])).forward(params, x)
    model = reference(params, buffers)
    fp32 = reference_logits(model, x)
    exact = reference_logits(model, x, dtype=torch.float64)
    result = check_logit_parity(mine, exact, fp32)
    result["n_images"] = len(x)
    result["argmax_agrees_float32"] = bool(np.array_equal(mine.argmax(1), fp32.argmax(1)))
    return result


def import_checkpoint(checkpoint: Path, labels: list[str], out: Path) -> dict:
    import torch
    from torchvision.models import resnet18
    from weightcode.models.resnet18native import render_forward_u8
    from weightcode.models.resnet18up import param_shapes

    if not labels or any(not isinstance(s, str) or not s for s in labels) or len(set(labels)) != len(labels):
        raise ValueError("labels must be a nonempty ordered list of distinct strings")
    source_sha = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = resnet18(weights=None, num_classes=len(labels)).state_dict()
    missing = set(expected) - set(state) if isinstance(state, dict) else set()
    # Official legacy ImageNet weights predate BatchNorm's training counters. PyTorch also initializes
    # these absent counters to zero on load; they do not enter evaluation-mode inference.
    if not isinstance(state, dict) or set(state) - set(expected) \
            or any(not key.endswith("num_batches_tracked") for key in missing):
        raise ValueError("checkpoint must be a plain torchvision ResNet-18 state_dict with matching labels/head")
    state = {**state, **{key: expected[key].clone() for key in missing}}
    for key, template in expected.items():
        v = state[key]
        if not isinstance(v, torch.Tensor) or v.layout != torch.strided or v.shape != template.shape \
                or v.dtype != template.dtype or not bool(torch.isfinite(v).all()):
            raise ValueError(f"unsupported tensor shape, dtype or value: {key}")
        if key.endswith("running_var") and bool((v < 0).any()):
            raise ValueError(f"negative BatchNorm variance: {key}")
    params = {k: state[k].detach().contiguous().numpy().copy() for k in param_shapes(len(labels))}
    buffers = {k: v.detach().contiguous().numpy().copy() for k, v in state.items() if k not in params}
    # Fixed synthetic probe checks implementation equivalence, never task performance.
    x = np.random.default_rng(0).integers(0, 256, (1, 3, 224, 224), dtype=np.uint8)
    result = parity(params, buffers, x)
    if sha(checkpoint) != source_sha:
        raise ValueError("checkpoint changed during import")
    out = new_dir(out)
    save_params(out / "params.npz", params)
    np.savez(out / "buffers.npz", **buffers)
    (out / "forward.py").write_text(render_forward_u8(len(labels)))
    m = {"schema": "weightcode.model_bundle.v1", "compatibility": compatibility(params, labels), "labels": labels,
         "input": INPUT, "params_sha256": params_sha256(params), "source_checkpoint_sha256": source_sha,
         "parity": result, "parity_probe": {"kind": "synthetic", "seed": 0},
         "initialized_missing_counters": sorted(missing),
         "files": bind(out, ["params.npz", "buffers.npz", "forward.py"])}
    write(out / "manifest.json", m)
    return m


def load_buffers(root: Path) -> dict:
    with np.load(root / "buffers.npz", allow_pickle=False) as z:
        buffers = {k: z[k] for k in z.files}
    # Loading into the strict reference model also checks buffer names and shapes.
    if any(not k.endswith("num_batches_tracked") or v.dtype != np.int64 or v.shape != () for k, v in buffers.items()):
        raise ValueError("unsupported non-floating buffers")
    return buffers


def export_checkpoint(params: dict, buffers: dict, path: Path, probe: np.ndarray) -> dict:
    import torch

    if path.exists():
        raise FileExistsError(path)
    model = reference(params, buffers)
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    reloaded = {k: state[k].numpy().copy() for k in params}
    if params_sha256(reloaded) != params_sha256(params) \
            or any(not torch.equal(state[k], torch.from_numpy(v)) for k, v in buffers.items()):
        raise ValueError("export reload changed the model")
    return {"checkpoint_sha256": sha(path), "params_sha256": params_sha256(reloaded),
            "parity": parity(reloaded, buffers, probe)}
