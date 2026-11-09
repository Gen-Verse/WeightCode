"""PREREG T03 (native ImageNet ResNet-18 precheck) on tiny synthetic inputs: no ImageNet image, no weights file.

The model has random weights with BatchNorm statistics moved away from 0 and 1, so that the eval-form BatchNorm
matters. Two or three 224 x 224 inputs keep memory small.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.common.identity import params_sha256
from weightcode.data import imagenet_val as I
from weightcode.evalkit import t03_native as T
from weightcode.evalkit.patch_check import load_forward
from weightcode.models import resnet18native as N
from weightcode.models import resnet18up as M
from weightcode.models.forward_gen import write_forward
from weightcode.models.parity import check_logit_parity

HF_ROW_COUNTS = [3572] * 6 + [3571] * 8


@pytest.fixture(scope="module")
def params() -> dict[str, np.ndarray]:
    torch.manual_seed(0)
    model = N.ResNet18Native()
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("running_mean"):
                buffer.copy_(0.1 * torch.randn_like(buffer))
            elif name.endswith("running_var"):
                buffer.copy_(0.5 + torch.rand_like(buffer))
        for name, p in model.named_parameters():
            if name.endswith("bn1.weight") or name.endswith("bn2.weight"):
                p.copy_(0.8 + 0.4 * torch.rand_like(p))
    state = model.state_dict()
    return {name: state[name].detach().numpy().astype(np.float32).copy() for name in M.param_shapes(1000)}


@pytest.fixture(scope="module")
def forward(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("fwd") / "forward.py"
    write_forward(path, N.render_forward())
    return load_forward(path)[0]


def _inputs(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, *N.INPUT_SHAPE)).astype(np.float32)


def test_spec_builds_with_e12_layers_without_upsample(params: dict[str, np.ndarray]) -> None:
    classes = [f"class {i}" for i in range(1000)]
    spec = N.build_resnet18native_spec("t03native", classes, params)
    assert spec["arch"] == N.ARCH and spec["n_classes"] == 1000 and spec["input_shape"] == [3, 224, 224]
    assert spec["n_parameters"] == sum(v.size for v in params.values())
    assert set(spec["tensors"]) == set(M.param_shapes(1000))
    e12 = M.build_resnet18up_spec("t03native", classes, M.ResNet18UpConfig(n_classes=1000, resize=128), params)
    assert e12["layers"][0]["op"] == "resize_bilinear"
    assert spec["layers"] == e12["layers"][1:]
    assert all(layer["op"] != "resize_bilinear" for layer in spec["layers"])
    json.dumps(spec)


def test_native_module_is_torchvision_resnet18(params: dict[str, np.ndarray]) -> None:
    model = N.build_resnet18native(params, torch.float32)
    assert set(model.state_dict()) - {k for k in model.state_dict() if k.endswith("num_batches_tracked")} == set(
        M.param_shapes(1000))
    check = M.torchvision_reference_diff(model, _inputs(2, seed=3))
    if check["torchvision"] is not None:
        assert check["max_abs_diff"] < 1e-4


def test_forward_matches_torch_rebuilds(params: dict[str, np.ndarray], forward) -> None:
    x = _inputs(3)
    ours = np.asarray(forward(params, x))
    assert ours.dtype == np.float32 and ours.shape == (3, 1000)
    ref64 = N.eval_logits(N.build_resnet18native(params, torch.float64), x)
    ref32 = N.eval_logits(N.build_resnet18native(params, torch.float32), x)
    result = check_logit_parity(ours, ref64, ref32)
    assert result["float64_max_rel_diff"] <= 2.0**-23
    record = T.parity_record(ours, ref64, ref32)
    assert record["verdict"] == "pass" and record["rows"] == 3
    assert record["float32_argmax_disagreements"] == int(np.count_nonzero(ours.argmax(1) != ref32.argmax(1)))
    shifted = ours.astype(np.float64) + 0.5
    assert T.parity_record(shifted, ref64, ref32)["verdict"] == "fail"


def test_forward_refuses_wrong_inputs(params: dict[str, np.ndarray], forward) -> None:
    with pytest.raises(TypeError):
        forward(params, np.zeros((1, 3, 224, 224), dtype=np.uint8))
    with pytest.raises(ValueError):
        forward(params, np.zeros((1, 3, 32, 32), dtype=np.float32))
    bad = _inputs(1)
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        forward(params, bad)
    with pytest.raises(KeyError):
        forward({k: v for k, v in params.items() if k != "fc.bias"}, _inputs(1))
    assert forward(params, np.zeros((0, 3, 224, 224), dtype=np.float32)).shape == (0, 1000)


def test_d_rule_on_the_hf_shard_layout() -> None:
    rows = I.d_rows(HF_ROW_COUNTS, I.shard_names())
    assert len(rows) == I.N_D == 1000
    assert [r.global_index for r in rows] == list(range(0, 50_000, 50))
    offsets = np.concatenate([[0], np.cumsum(HF_ROW_COUNTS)])
    for r in rows:
        assert I.in_d(r.global_index)
        assert r.shard == I.shard_names()[r.shard_index]
        assert offsets[r.shard_index] + r.row == r.global_index and 0 <= r.row < HF_ROW_COUNTS[r.shard_index]
    first_of_shard1 = next(r for r in rows if r.shard_index == 1)
    assert (first_of_shard1.global_index, first_of_shard1.row) == (3600, 28)
    assert rows[-1].shard == "validation-00013-of-00014.parquet" and rows[-1].global_index == 49_950
    assert I.shard_names()[0] == "validation-00000-of-00014.parquet"
    assert not I.in_d(1) and I.in_d(0) and I.in_d(49_950)
    with pytest.raises(ValueError):
        I.d_rows(HF_ROW_COUNTS[:-1], I.shard_names()[:-1])


def test_save_reload_round_trip_and_fixed_edit(params: dict[str, np.ndarray], forward, tmp_path: Path) -> None:
    x = _inputs(2, seed=1)
    base = np.asarray(forward(params, x))
    reloaded, record = T.save_reload(params, tmp_path / "model.npz")
    assert record["same_sha256"] and record["arrays_bitwise_equal"]
    assert record["params_sha256"] == params_sha256(params) and record["file_bytes"] > 4 * 11_000_000
    assert T.bitwise_equal(np.asarray(forward(reloaded, x)), base)

    edited, edit = T.apply_fixed_edit(params, tmp_path / "edit")
    assert edit["parent_sha256"] == params_sha256(params) and edit["child_sha256"] == params_sha256(edited)
    assert edit["save_reload"]["same_sha256"]
    assert edit["changed_per_tensor"] == {"fc.weight": 8, "fc.bias": 2, "layer1.0.conv1.weight": 2,
                                          "layer4.1.conv2.weight": 2 * 512 * 9}
    assert json.loads((tmp_path / "edit" / T.PATCH_FILE).read_text())["ops"] == [dict(op) for op in T.FIXED_EDIT_OPS]
    assert edited["fc.weight"][1, 7] == np.float32(np.float64(params["fc.weight"][1, 7]) - 0.05)
    after = np.asarray(forward(edited, x))
    assert not T.bitwise_equal(after, base)
    ref64 = N.eval_logits(N.build_resnet18native(edited, torch.float64), x)
    ref32 = N.eval_logits(N.build_resnet18native(edited, torch.float32), x)
    assert T.parity_record(after, ref64, ref32)["verdict"] == "pass"


def test_accuracy_record_band() -> None:
    labels = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    logits = np.zeros((10, 4))
    logits[np.arange(10), labels] = 1.0
    logits[7:, 0] = 2.0
    record = T.accuracy_record(logits, labels)
    assert (record["n_correct"], record["top1_percent"], record["verdict"]) == (7, 70.0, "pass")
    assert record["band"] == [T.ACCURACY_REFERENCE - 5.0, T.ACCURACY_REFERENCE + 5.0]
    assert record["top5_percent"] == 100.0 and T.top_k_accuracy(logits, labels, 1) == 70.0
    logits[5:, 0] = 2.0
    assert T.accuracy_record(logits, labels)["verdict"] == "fail"


def test_manifest_and_classes(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.parquet").write_bytes(b"abc")
    text = ('from collections import OrderedDict\n\nIMAGENET2012_CLASSES = OrderedDict(\n    {\n'
            '        "n01440764": "tench, Tinca tinca",\n        "n01443537": "goldfish, Carassius auratus",\n'
            '    }\n)\n')
    (tmp_path / "classes.py").write_text(text)
    lines = [f"{hashlib.sha256((tmp_path / p).read_bytes()).hexdigest()}  {p}" for p in ("classes.py", "data/a.parquet")]
    (tmp_path / "manifest.sha256").write_text("\n".join(lines) + "\n")
    listed = I.verify_manifest(tmp_path, I.file_sha256(tmp_path / "manifest.sha256"))
    assert set(listed) == {"classes.py", "data/a.parquet"}
    with pytest.raises(ValueError):
        I.verify_manifest(tmp_path, "0" * 64)
    (tmp_path / "data" / "a.parquet").write_bytes(b"abd")
    with pytest.raises(ValueError):
        I.verify_manifest(tmp_path, None)
    classes = I.parse_classes(tmp_path / "classes.py")
    assert classes == [("n01440764", "tench, Tinca tinca"), ("n01443537", "goldfish, Carassius auratus")]
    assert I.classes_wnid_order(classes)["sorted"] is True


def test_preprocess_is_torchvision_eval_transform() -> None:
    pytest.importorskip("torchvision")
    from PIL import Image

    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8), "RGB")
    gray = Image.fromarray(rng.integers(0, 256, (260, 240), dtype=np.uint8), "L")
    for image in (rgb, gray):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        out = I.preprocess(buffer.getvalue())
        assert out.shape == (3, 224, 224) and out.dtype == np.float32
        with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
            expected = I.eval_transform()(decoded.convert("RGB")).numpy()
        assert np.array_equal(out, expected)


def test_read_d_records_takes_only_d_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    counts = [70, 60, 70]
    names = I.shard_names(3)
    labels = iter(range(10_000))
    for name, count in zip(names, counts):
        images = [{"bytes": f"img-{name}-{r}".encode(), "path": f"{name}-{r}.JPEG"} for r in range(count)]
        table = pa.table({"image": images, "label": [next(labels) % 1000 for _ in range(count)]})
        pq.write_table(table, tmp_path / name, row_group_size=25)
    rows = I.d_rows(I.shard_row_counts(tmp_path, names), names, expected_rows=200)
    records = list(I.read_d_records(tmp_path, rows))
    assert [r.global_index for r in records] == [0, 50, 100, 150]
    assert [(r.shard, r.row) for r in records] == [(names[0], 0), (names[0], 50), (names[1], 30), (names[2], 20)]
    for record in records:
        assert record.image_bytes == f"img-{record.shard}-{record.row}".encode()
        assert record.image_sha256 == hashlib.sha256(record.image_bytes).hexdigest()
        assert record.label == record.global_index % 1000
        assert "image_bytes" not in record.index_entry()
