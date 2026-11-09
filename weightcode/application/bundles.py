"""Content-bound model, task and executable program bundles."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from weightcode.common.identity import load_params, params_sha256
from weightcode.harness.program import MAX_PROGRAM_BYTES

ADAPTER = "torchvision-resnet18-u8-v1"
INPUT = {"dtype": "uint8", "shape": [3, 224, 224], "layout": "NCHW", "color": "RGB",
         "resize_shorter": 256, "crop": 224, "interpolation": "bilinear", "antialias": True,
         "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
EXPERIENCE = "weightcode.classification_experience.v1"


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: dict) -> None:
    # Application records are write-once, including failure records.
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def new_dir(path: Path) -> Path:
    path = path.absolute()
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    return path


def bind(root: Path, names: list[str]) -> dict:
    return {name: sha(root / name) for name in names}


def copy_model(source: Path, dest: Path) -> None:
    dest.mkdir()
    for name in ("manifest.json", "params.npz", "buffers.npz", "forward.py"):
        shutil.copyfile(source / name, dest / name)


def copy_task(source: Path, dest: Path) -> None:
    dest.mkdir()
    for name in ("manifest.json", "experience.npz", "selection.npz", "final.npz"):
        shutil.copyfile(source / name, dest / name)
    copy_model(source / "model", dest / "model")


def verify(root: Path, schema: str, names: set[str]) -> dict:
    manifest = read(root / "manifest.json")
    if manifest.get("schema") != schema or set(manifest.get("files", {})) != names:
        raise ValueError(f"invalid {schema} manifest: {root}")
    for name, expected in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError(f"missing or changed bundle file: {path}")
    return manifest


def model_bundle(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    from weightcode.models.resnet18native import render_forward_u8
    from weightcode.models.resnet18up import check_params

    m = verify(root, "weightcode.model_bundle.v1", {"params.npz", "buffers.npz", "forward.py"})
    labels = m.get("labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(s, str) or not s for s in labels) \
            or len(set(labels)) != len(labels):
        raise ValueError("labels must be a nonempty ordered list of distinct strings")
    params = load_params(root / "params.npz")
    check_params(params, len(labels))
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in params.values()):
        raise ValueError("model tensors must be finite float32")
    if any(np.any(v < 0) for k, v in params.items() if k.endswith("running_var")):
        raise ValueError("negative BatchNorm variance")
    if m.get("params_sha256") != params_sha256(params):
        raise ValueError("model parameter identity mismatch")
    expected = compatibility(params, labels)
    if m.get("compatibility") != expected or m.get("input") != INPUT:
        raise ValueError("unsupported model adapter or input contract")
    if (root / "forward.py").read_text() != render_forward_u8(len(labels)):
        raise ValueError("forward.py differs from the supported adapter")
    from weightcode.application.resnet import load_buffers
    buffers = load_buffers(root)
    expected_buffers = {k.removesuffix("running_mean") + "num_batches_tracked"
                        for k in params if k.endswith("running_mean")}
    if set(buffers) != expected_buffers:
        raise ValueError("BatchNorm counter layout mismatch")
    return m, params


def compatibility(params: dict, labels: list[str]) -> dict:
    return {"adapter": ADAPTER, "input_sha256": digest(INPUT), "labels_sha256": digest(labels),
            "tensors_sha256": digest({k: [str(v.dtype), list(v.shape)] for k, v in params.items()})}


def program_source(path: Path) -> bytes:
    data = path.read_bytes()
    if not 0 < len(data) <= MAX_PROGRAM_BYTES:
        raise ValueError("program must be nonempty and at most 1 MiB")
    tree = ast.parse(data, filename=str(path))
    if not any(isinstance(n, ast.FunctionDef) and n.name == "update" for n in tree.body):
        raise ValueError("program must define update(student, experience)")
    return data


def pack_program(source: Path, model: Path, out: Path, *, provenance: dict | None = None) -> dict:
    m, _ = model_bundle(model)
    data = program_source(source)
    out = new_dir(out)
    (out / "program.py").write_bytes(data)
    manifest = {"schema": "weightcode.program_bundle.v1", "compatibility": m["compatibility"],
                "experience_schema": EXPERIENCE, "engine": "weightcode", "engine_version": "0.0.1",
                "clock": "virtual-v1", "requirements": ["numpy"], "files": bind(out, ["program.py"]),
                "provenance": provenance or {"kind": "user-supplied", "validated_gain": False}}
    write(out / "manifest.json", manifest)
    return manifest


def program_bundle(root: Path, model: dict) -> dict:
    p = verify(root, "weightcode.program_bundle.v1", {"program.py"})
    if p.get("compatibility") != model["compatibility"] or p.get("experience_schema") != EXPERIENCE \
            or p.get("clock") != "virtual-v1" or p.get("engine_version") != "0.0.1":
        raise ValueError("program and model/runtime are incompatible")
    program_source(root / "program.py")
    return p


def dataset(path: Path, n_classes: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"x_new", "y_new", "x_old", "y_old"}:
            raise ValueError(f"{path}: expected x_new, y_new, x_old, y_old")
        data = {k: z[k] for k in z.files}
    for role in ("new", "old"):
        x, y = data[f"x_{role}"], data[f"y_{role}"]
        if x.dtype != np.uint8 or x.ndim != 4 or tuple(x.shape[1:]) != (3, 224, 224) or not len(x):
            raise ValueError(f"{path}: x_{role} must be nonempty uint8 [N,3,224,224]")
        if y.dtype != np.int64 or y.shape != (len(x),) or np.any(y < 0) or np.any(y >= n_classes):
            raise ValueError(f"{path}: y_{role} must be int64 [N] in the model's class order")
    return data


def _row_hashes(data: dict) -> set[str]:
    return {hashlib.sha256(x.tobytes()).hexdigest() for role in ("new", "old") for x in data[f"x_{role}"]}


def prepare_task(model: Path, experience: Path, selection: Path, final: Path, out: Path, *, goal: str,
                 retention_drop: float, min_new_gain: float) -> dict:
    m, _ = model_bundle(model)
    if not goal.strip() or not 0 <= retention_drop <= 1 or not 0 <= min_new_gain <= 1:
        raise ValueError("goal required; retention_drop and min_new_gain must be fractions in [0,1]")
    files = {"experience.npz": experience, "selection.npz": selection, "final.npz": final}
    seen: set[str] = set()
    for path in files.values():
        rows = _row_hashes(dataset(path, len(m["labels"])))
        if seen & rows:
            raise ValueError("experience, selection and final images must be disjoint")
        seen |= rows
    out = new_dir(out)
    copy_model(model, out / "model")
    for name, path in files.items():
        shutil.copyfile(path, out / name)
    task = {"schema": "weightcode.task_bundle.v1", "goal": goal, "experience_schema": EXPERIENCE,
            "retention_drop": retention_drop, "min_new_gain": min_new_gain,
            "selection_rule": "highest new+old accuracy sum among eligible checks; earliest tie; else base",
            "model_manifest_sha256": sha(out / "model" / "manifest.json"), "files": bind(out, list(files))}
    write(out / "manifest.json", task)
    # Detect source changes during snapshotting instead of binding accidental mixed versions.
    if task["model_manifest_sha256"] != sha(model / "manifest.json") \
            or any(task["files"][name] != sha(source) for name, source in files.items()):
        raise ValueError("input changed while preparing task; incomplete output retained")
    model_bundle(out / "model")
    return task


def task_bundle(root: Path) -> tuple[dict, dict, dict]:
    t = verify(root, "weightcode.task_bundle.v1", {"experience.npz", "selection.npz", "final.npz"})
    if sha(root / "model" / "manifest.json") != t["model_manifest_sha256"]:
        raise ValueError("task model manifest changed")
    m, params = model_bundle(root / "model")
    if t.get("experience_schema") != EXPERIENCE or not isinstance(t.get("goal"), str) or not t["goal"].strip():
        raise ValueError("invalid task contract")
    for key in ("retention_drop", "min_new_gain"):
        if type(t.get(key)) not in (int, float) or not 0 <= t[key] <= 1:
            raise ValueError(f"invalid {key}")
    return t, m, params
