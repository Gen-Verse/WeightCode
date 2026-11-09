"""Read one repair instance and write baseline patches in the shared format (contract sections 3 and 5.0).

Baselines see exactly what an editor arm sees (model, forward.py, evidence, task) plus a counted patch-check
oracle. The private record is read only to locate V_check for the oracle and to verify hashes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.evalkit.patch_check import ForwardFn, PatchCheckOracle, load_forward
from weightcode.patch import MAX_PATCH_JSON_BYTES

PATCH_SCHEMA = "weightcode.patch.v0"
DEFAULT_QUERY_BUDGET = 8


@dataclass(frozen=True)
class Instance:
    """Editor-visible files of one instance plus its private record."""

    instance_dir: Path
    instance_id: str
    params: Params
    params_sha256: str
    forward: ForwardFn
    forward_sha256: str
    evidence: dict[str, np.ndarray]
    task: dict[str, Any]
    private: dict[str, Any]

    @property
    def target(self) -> tuple[int, int] | None:
        """The (a, b) pair of an evidence-v0 instance; None for evidence-v1 instances, which have no pair."""
        pair = self.private.get("target")
        if pair is None:
            return None
        a, b = pair
        return int(a), int(b)


def load_instance(instance_dir: str | Path) -> Instance:
    """Load and hash-check an instance directory built by ``weightcode.evalkit``."""
    root = Path(instance_dir)
    params = load_params(root / "model.npz")
    forward, forward_sha = load_forward(root / "forward.py")
    with np.load(root / "evidence.npz", allow_pickle=False) as archive:
        evidence = {key: archive[key] for key in archive.files}
    task = json.loads((root / "task.json").read_text())
    private = json.loads((root / "private.json").read_text())
    digest = params_sha256(params)
    if private.get("model_sha256") != digest:
        raise ValueError(f"{root}: model.npz hash {digest} != private.json model_sha256 {private.get('model_sha256')}")
    if private.get("forward_sha256") != forward_sha:
        raise ValueError(f"{root}: forward.py hash {forward_sha} != private.json forward_sha256")
    return Instance(
        instance_dir=root,
        instance_id=root.name,
        params=params,
        params_sha256=digest,
        forward=forward,
        forward_sha256=forward_sha,
        evidence=evidence,
        task=task,
        private=private,
    )


def make_oracle(instance: Instance, budget: int = DEFAULT_QUERY_BUDGET, batch_size: int = 1024) -> PatchCheckOracle:
    """Counted patch-check oracle on the instance's V_check, with the same forward file the editor sees.

    V_check is checked against the canonical content hash the instance recorded (``vcheck_content_sha256``), so a
    changed or swapped file is refused rather than scored.
    """
    with np.load(Path(instance.private["vcheck_path"]), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    recorded = instance.private.get("vcheck_content_sha256")
    if recorded is not None and params_sha256(arrays) != recorded:
        raise ValueError(f"{instance.private['vcheck_path']}: V_check content hash differs from the instance record")
    return PatchCheckOracle(arrays["x"], arrays["y"], instance.forward, instance.params, instance.target, budget, batch_size)


INDEX_FORM_MAX_ENTRIES = 40_000
"""Largest changed-entry count of one tensor written inline as an index-form ``add`` op; a tensor with more changed
entries, or with every entry changed, is written as ``add_dense``. Compact JSON costs about 30 to 35 bytes per
entry, so one such tensor stays well under the executor's patch JSON cap; ``write_delta_patch`` checks the whole file
against that cap and moves tensors to ``add_dense`` when several index-form tensors together would exceed it."""


def delta_encoding(delta: np.ndarray) -> tuple[str, int]:
    """Per-tensor op kind for one delta and its edit size as the executor counts it (contract 5.1).

    A delta that changes only part of a tensor is written in index form (``add``, edit size = changed entries);
    a delta that changes every entry, or more than ``INDEX_FORM_MAX_ENTRIES`` entries, as ``add_dense`` (edit
    size = tensor size). An all-zero delta gets no op (edit size 0). ``write_delta_patch`` can still move an
    index-form tensor to ``add_dense`` when the patch file as a whole would exceed the executor's JSON cap, so the
    authoritative edit size of a written patch is the one the executor reports when it applies the file.
    """
    nnz = int(np.count_nonzero(delta))
    if nnz == 0:
        return "none", 0
    if nnz < delta.size and nnz <= INDEX_FORM_MAX_ENTRIES:
        return "add", nnz
    return "add_dense", int(delta.size)


def _dense_name(key: str) -> str:
    return "d_" + key.replace(".", "_") + ".npy"


def _patch_text(
    parent_sha256: str,
    deltas: Mapping[str, np.ndarray],
    kinds: Mapping[str, str],
    hypothesis: str,
    predicted_effect: str,
) -> str:
    ops: list[dict[str, Any]] = []
    for key, delta in deltas.items():
        kind = kinds[key]
        if kind == "add":
            positions = np.argwhere(delta != 0.0)
            ops.append(
                {
                    "op": "add",
                    "tensor": key,
                    "index": positions.tolist(),
                    "values": [float(v) for v in delta[tuple(positions.T)]],
                }
            )
        elif kind == "add_dense":
            ops.append({"op": "add_dense", "tensor": key, "delta": _dense_name(key)})
    patch = {
        "schema": PATCH_SCHEMA,
        "parent_sha256": parent_sha256,
        "ops": ops,
        "hypothesis": hypothesis,
        "predicted_effect": predicted_effect,
    }
    return json.dumps(patch, separators=(",", ":"))


def write_delta_patch(
    out_dir: str | Path,
    parent_sha256: str,
    base: Mapping[str, np.ndarray],
    child: Mapping[str, np.ndarray],
    hypothesis: str,
    predicted_effect: str,
) -> Path:
    """Write ``child - base`` as one op per changed tensor, in the encoding chosen by ``delta_encoding``.

    The JSON is compact. If the file would still exceed the executor's cap (``weightcode.patch.MAX_PATCH_JSON_BYTES``),
    the index-form tensor with the most changed entries is moved to ``add_dense``, repeatedly, until it fits, so a
    valid delta is never written as a file the executor refuses. Deltas are float64 (inline JSON floats for ``add``, a
    bare-named .npy for ``add_dense``) so that applying them in the executor's float64 arithmetic and casting once to
    float32 reproduces ``child`` exactly or to within one float32 rounding step, which callers should verify by
    applying the patch with ``weightcode.patch``; the edit size to record is the one the executor reports.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    deltas = {
        key: np.asarray(child[key], dtype=np.float64) - np.asarray(base[key], dtype=np.float64) for key in sorted(base)
    }
    kinds = {key: delta_encoding(delta)[0] for key, delta in deltas.items()}
    text = _patch_text(parent_sha256, deltas, kinds, hypothesis, predicted_effect)
    while len(text.encode("utf-8")) > MAX_PATCH_JSON_BYTES:
        index_form = [key for key, kind in kinds.items() if kind == "add"]
        if not index_form:
            raise ValueError(f"patch JSON is {len(text.encode('utf-8'))} bytes with every changed tensor dense")
        largest = max(index_form, key=lambda key: (int(np.count_nonzero(deltas[key])), key))
        kinds[largest] = "add_dense"
        text = _patch_text(parent_sha256, deltas, kinds, hypothesis, predicted_effect)
    for key, kind in kinds.items():
        if kind == "add_dense":
            np.save(root / _dense_name(key), deltas[key], allow_pickle=False)
    path = root / "patch.json"
    path.write_text(text)
    return path
