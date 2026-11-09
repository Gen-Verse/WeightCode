"""Synthetic tests of the local sealed read of in1k_t04 (``weightcode.evalkit.sealed_t04``): 14 tiny parquet shards of 70
validation rows, a linear forward on mean colours, children that differ from the base, a stand-in official file with
the official function bodies. Nothing real is read; every expectation is recomputed here row by row."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from weightcode.common.identity import params_sha256, save_params  # noqa: E402
from weightcode.data import imagenet_val as V  # noqa: E402
from weightcode.data import store  # noqa: E402
from weightcode.data import t04 as T  # noqa: E402
from weightcode.evalkit import sealed_finals as SF  # noqa: E402
from weightcode.evalkit import sealed_t04 as ST  # noqa: E402
from weightcode.evalkit.sealed import manifest_sha256, read_registry  # noqa: E402
from weightcode.evalkit.sealed_range import encode_bits  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
N_ROWS = 70  # 5 rows in each of the 14 shards
N_CLASSES = 4
OFFICIAL = '''import os

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255
'''
FORWARD = '''import numpy as np


def forward(params, x):
    m = np.asarray(x).reshape(len(x), 3, -1).astype(np.float64).mean(axis=2) / 255.0
    return m @ params["w"] + params["b"]
'''
TAGS = {"gaussian": ("gaussian_noise_s3", "i4c8c63bd"), "contrast": ("contrast_s4", "i62cbf956")}
SPLIT_SIZES = {"V_evidence": 4, "V_check": 6, "V_select": 4, "train": 4}


def _jpeg(row: int) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(row)
    colour = rng.integers(20, 236, size=3)
    pixels = np.clip(colour + rng.integers(-15, 16, size=(230 + row % 5, 260, 3)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _net(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"w": rng.standard_normal((3, N_CLASSES)).astype(np.float32),
            "b": (0.1 * rng.standard_normal(N_CLASSES)).astype(np.float32)}


def _predict(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    m = x.reshape(len(x), 3, -1).astype(np.float64).mean(axis=2) / 255.0
    return (m @ params["w"] + params["b"]).argmax(axis=1)


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    for name in ("private", "sealed", "runs"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(f"WEIGHTCODE_{name.upper()}", str(tmp_path / name))
    monkeypatch.setattr(V, "N_VAL_ROWS", N_ROWS)
    rows = np.arange(N_ROWS)
    dev = rows[np.isin(rows % T.ROW_MODULUS, T.DEV_RESIDUES)]
    monkeypatch.setattr(T, "N_TEST", int(N_ROWS - len(dev)))
    order = np.random.default_rng(0).permutation(dev)
    splits, start = {}, 0
    for split, size in SPLIT_SIZES.items():
        splits[split] = np.sort(order[start:start + size]).astype(np.int64)
        start += size
    monkeypatch.setattr(T, "split_rows", lambda labels: {k: v.copy() for k, v in splits.items()})
    labels = {int(r): int(r) % N_CLASSES for r in rows}
    images = {int(r): _jpeg(int(r)) for r in rows}

    # the shards, classes.py and the manifest, where the read finds them
    root = tmp_path / "sealed" / "in1k_t04" / "hf_49e2ee26"
    (root / T.SHARD_DIR).mkdir(parents=True)
    per = N_ROWS // V.N_SHARDS
    for k, name in enumerate(V.shard_names()):
        mine = range(k * per, (k + 1) * per)
        table = pa.table({"image": [{"bytes": images[r], "path": f"img{r}.JPEG"} for r in mine],
                          "label": pa.array([labels[r] for r in mine], type=pa.int64())})
        pq.write_table(table, root / T.SHARD_DIR / name)
    (root / V.CLASSES_FILE).write_text("IMAGENET2012_CLASSES = {}\n", encoding="utf-8")
    listed = [V.CLASSES_FILE] + [f"{T.SHARD_DIR}/{name}" for name in V.shard_names()]
    manifest_text = "".join(f"{hashlib.sha256((root / rel).read_bytes()).hexdigest()}  {rel}\n" for rel in listed)

    kit = tmp_path / "runs" / "t04" / "sealed_kit_v6"
    (kit / "frozen").mkdir(parents=True)
    (kit / V.MANIFEST).write_text(manifest_text, encoding="utf-8")
    monkeypatch.setattr(V, "MANIFEST_SHA256", hashlib.sha256(manifest_text.encode()).hexdigest())
    (root / V.MANIFEST).write_text(manifest_text, encoding="utf-8")
    (kit / "make_imagenet_c.py").write_text(OFFICIAL, encoding="utf-8")
    monkeypatch.setattr(T, "OFFICIAL_SHA256", hashlib.sha256(OFFICIAL.encode()).hexdigest())
    official = T.load_official(kit / "make_imagenet_c.py", T.OFFICIAL_SHA256)

    # dataset.json copies (the NSCC store's records), computed with the store's own functions
    test_rows = T.test_rows()
    rows_sha = T.rows_sha256(test_rows)
    check = splits["V_check"]
    crops = np.stack([T.center_crop_u8(images[int(r)]) for r in check])
    clean_arrays = {"x": crops, "y": np.array([labels[int(r)] for r in check], dtype=np.int64), "idx": check}
    clean_meta = {
        "splits": {s: {"rows_sha256": T.rows_sha256(r), "sha256": store.split_sha256(clean_arrays) if s == "V_check"
                       else "0" * 64} for s, r in splits.items()},
        "test": {"location": "not materialised: rows of the validation shards", "sha256": rows_sha, "n": T.N_TEST,
                 "rows": "row % 25 not in 0..5"},
        "source_provenance": {"revision": V.REVISION, "manifest_sha256": V.MANIFEST_SHA256,
                              "classes_py_sha256": hashlib.sha256((root / V.CLASSES_FILE).read_bytes()).hexdigest()},
    }
    (kit / "in1k_t04").mkdir()
    (kit / "in1k_t04" / "dataset.json").write_text(json.dumps(clean_meta), encoding="utf-8")
    for condition, _ in TAGS.values():
        corruption, severity = ST.condition_recipe(condition)
        x = T.corrupt_split(official, corruption, severity, crops, check)
        meta = {"test": {"clean_sha256": rows_sha, "n": T.N_TEST,
                         "sha256": hashlib.sha256(f"{rows_sha}:{condition}".encode()).hexdigest()},
                "condition": {"name": condition, "corruption": corruption, "severity": severity, "source": "test",
                              "rng": "test"},
                "splits": {"V_check": {"sha256": store.split_sha256({**clean_arrays, "x": x})}}}
        path = kit / "in1k_t04" / "conditions" / condition / "dataset.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(meta), encoding="utf-8")

    # two instances on one base and forward.py, in the kit (installed by install_instances)
    base = _net(1)
    forward_sha = hashlib.sha256(FORWARD.encode()).hexdigest()
    for condition, instance_id in TAGS.values():
        directory = kit / "instances" / instance_id
        directory.mkdir(parents=True)
        save_params(directory / "model.npz", base)
        (directory / "forward.py").write_text(FORWARD, encoding="utf-8")
        np.savez(directory / "evidence.npz", x=np.zeros((1, 3), dtype=np.uint8))
        (directory / "task.json").write_text(json.dumps({"classes": [f"c{i}" for i in range(N_CLASSES)]}))
        (directory / "private.json").write_text(json.dumps({
            "dataset": T.DATASET, "condition": condition, "split_role": "dev", "seed": 100,
            "model_sha256": params_sha256(base), "forward_sha256": forward_sha}))

    # the list: the real v6 sources and names, with synthetic children
    real = json.loads((REPO / "scripts" / "sealed" / "finals_v6.json").read_text(encoding="utf-8"))
    keep = ("LEARN_r1", "LEARN_r2", "main_", "base", "arm3_lora_r4")
    nets: dict[str, dict[str, np.ndarray]] = {}
    export_rows = []
    blocks = {}
    seeds = iter(range(10, 100))
    for block in ("finals", "references"):
        blocks[block] = []
        for entry, seed in zip(real[block], seeds):
            if not any(entry["name"].split("_", 2)[2].startswith(k) for k in keep):
                continue
            params = base if entry["name"].endswith("_base") else _net(seed)
            entry = {**entry, "child_sha256": params_sha256(params)}
            blocks[block].append(entry)
            nets[entry["name"]] = params
            path = kit / "frozen" / f"{entry['name']}.npz"
            save_params(path, params)
            export_rows.append({"name": entry["name"], "instance_id": entry["instance_id"],
                                "params_sha256": entry["child_sha256"], "file_sha256": store_sha(path),
                                "child": {"kind": entry["source"]["kind"]}})
    listing = {"schema": SF.LIST_SCHEMA, "status": SF.STATUS_APPROVED, "approval": "synthetic test", "note": "",
               **blocks}
    list_path = tmp_path / "finals_test.json"
    list_path.write_text(json.dumps(listing, indent=1), encoding="utf-8")
    parsed, _ = SF.load_list(list_path)
    (kit / "frozen" / "export_test.json").write_text(json.dumps({
        "schema": "weightcode.t04.frozen_export.v1", "list_sha256": manifest_sha256(parsed),
        "code": {"commit": "c" * 40}, "entries": export_rows}), encoding="utf-8")
    ST.write_kit(kit, "synthetic")
    installed = ST.install_instances(kit)
    plan_path = tmp_path / "finals_test_secondary.json"
    plan_path.write_text(json.dumps(ST.make_secondary_plan(list_path)), encoding="utf-8")

    monkeypatch.setattr(SF, "code_state", lambda *a, **k: {"commit": "a" * 40, "dirty": False, "repo": str(tmp_path),
                                                           "source": "test", "files": {}})
    monkeypatch.setattr(SF, "_check_list_committed", lambda path, state: Path(path).name)
    return {"tmp": tmp_path, "kit": kit, "list": list_path, "plan": plan_path, "nets": nets, "base": base,
            "images": images, "labels": labels, "official": official, "root": root, "installed": installed,
            "parsed": parsed, "test_rows": test_rows}


def store_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected(world: dict) -> dict:
    """Per entry, the correctness bits of base and child on every set, recomputed row by row."""
    rows = world["test_rows"]
    crops = np.stack([T.center_crop_u8(world["images"][int(r)]) for r in rows])
    y = np.array([world["labels"][int(r)] for r in rows])
    secondary = np.isin(rows % T.ROW_MODULUS, ST.SECONDARY_RESIDUES)
    sets = {"clean": crops}
    for condition in {"gaussian_noise_s3", "contrast_s4", *[s for c, _ in TAGS.values() for s in ST.secondary_sets(c)]}:
        corruption, severity = ST.condition_recipe(condition)
        sets[condition] = T.corrupt_split(world["official"], corruption, severity, crops, rows)
    return {"y": y, "sets": sets, "secondary": secondary}


def _check_records(world: dict, results: Path) -> None:
    expected = _expected(world)
    records = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert [r["name"] for r in records] == [e["name"] for b in ("finals", "references") for e in world["parsed"][b]]
    plan = json.loads(world["plan"].read_text(encoding="utf-8"))
    in_plan = {name: group["sets"] for group in plan["groups"] for name in group["entries"]}
    y, sec = expected["y"], expected["secondary"]
    for record in records:
        child = world["nets"][record["name"]]
        condition = record["condition"]
        for key, set_name in (("per_image", condition), ("per_image_clean", "clean")):
            x = expected["sets"][set_name]
            assert record[key]["base_correct"] == encode_bits(_predict(world["base"], x) == y)
            assert record[key]["child_correct"] == encode_bits(_predict(child, x) == y)
        base_acc = float(np.mean(_predict(world["base"], expected["sets"][condition]) == y))
        child_acc = float(np.mean(_predict(child, expected["sets"][condition]) == y))
        assert record["gain"] == pytest.approx(100 * (child_acc - base_acc))
        assert record["joint"] == pytest.approx(record["gain"] + record["clean"])
        assert record["rows"]["condition"]["n"] == len(y) == T.N_TEST
        assert sorted(record["secondary"]) == sorted(in_plan.get(record["name"], []))
        for set_name, part in record["secondary"].items():
            x = expected["sets"][set_name][sec]
            assert part["per_image"]["base_correct"] == encode_bits(_predict(world["base"], x) == y[sec])
            assert part["per_image"]["child_correct"] == encode_bits(_predict(child, x) == y[sec])
        if record["name"].endswith("_base"):
            assert record["gain"] == record["clean"] == 0.0


def test_plan_names_the_base_learn_finals_and_main_control(world: dict) -> None:
    plan = json.loads(world["plan"].read_text(encoding="utf-8"))
    assert [g["entries"] for g in plan["groups"]] == [
        ["T04_gaussian_base", "T04_gaussian_LEARN_r1", "T04_gaussian_LEARN_r2", "T04_gaussian_main_ft_bn"],
        ["T04_contrast_base", "T04_contrast_LEARN_r1", "T04_contrast_LEARN_r2", "T04_contrast_main_grad_full"]]
    assert plan["groups"][0]["sets"][-1] == "contrast_s4" and plan["groups"][1]["sets"][-1] == "gaussian_noise_s3"
    draft = {**world["parsed"], "status": SF.STATUS_DRAFT, "approval": None}
    assert ST.entries_sha256(draft) == plan["entries_sha256"]  # approval leaves the plan valid
    broken = {**world["parsed"], "references": [e for e in world["parsed"]["references"] if "_main_" not in e["name"]]}
    path = world["tmp"] / "no_main.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SF.SealedFinalsError, match="main role"):
        ST.make_secondary_plan(path)


def test_the_real_v6_plan_is_the_committed_one() -> None:
    committed = json.loads((REPO / "scripts" / "sealed" / "finals_v6_secondary.json").read_text(encoding="utf-8"))
    made = ST.make_secondary_plan(REPO / "scripts" / "sealed" / "finals_v6.json")
    assert made["entries_sha256"] == committed["entries_sha256"] and made["groups"] == committed["groups"]


def test_fidelity_passes_and_a_changed_record_stops_it(world: dict) -> None:
    out = world["tmp"] / "runs" / "fid.json"
    record = ST.fidelity(world["kit"], world["root"], ["gaussian_noise_s3", "contrast_s4"], out)
    assert record["passed"] and all(v["match"] for v in record["v_check_content"].values())
    assert json.loads(out.read_text())["passed"]
    path = world["kit"] / "in1k_t04" / "conditions" / "contrast_s4" / "dataset.json"
    meta = json.loads(path.read_text())
    meta["splits"]["V_check"]["sha256"] = "f" * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(SF.SealedFinalsError, match="kit.json"):  # the kit check sees the edit first
        ST.fidelity(world["kit"], world["root"], ["contrast_s4"], None)
    (world["kit"] / ST.KIT_FILE).unlink()
    ST.write_kit(world["kit"], "edited")
    with pytest.raises(SF.SealedFinalsError, match="contrast_s4 V_check content"):
        ST.fidelity(world["kit"], world["root"], ["contrast_s4"], world["tmp"] / "runs" / "fid2.json")
    assert not json.loads((world["tmp"] / "runs" / "fid2.json").read_text())["passed"]


def test_test_row_reader_refuses_dev_rows_and_shards_outside_sealed(world: dict) -> None:
    plan = {"row_modulus": T.ROW_MODULUS, "dev_residues": list(T.DEV_RESIDUES)}
    data_dir = world["root"] / T.SHARD_DIR
    name = V.shard_names()[1]  # rows 5 .. 9; row 5 is a dev row
    with pytest.raises(SF.SealedFinalsError, match="dev row"):
        ST.read_test_records(data_dir, name, np.array([0, 1]), np.array([5, 6]), plan)
    outside = world["tmp"] / "copy"
    shutil.copytree(data_dir, outside)
    with pytest.raises(SF.SealedFinalsError, match="not under"):
        ST.read_test_records(outside, name, np.array([2]), np.array([7]), plan)
    data, labels = ST.read_test_records(data_dir, name, np.array([1, 2]), np.array([6, 7]), plan)
    assert data == [world["images"][6], world["images"][7]] and labels.tolist() == [world["labels"][6],
                                                                                     world["labels"][7]]


def test_generic_pipeline_refuses_vision_t04(world: dict) -> None:
    with pytest.raises(SF.SealedFinalsError, match="own module"):
        SF.run_sealed(world["list"])
    assert not (world["tmp"] / "sealed" / "registry.jsonl").exists()
    with pytest.raises(SF.SealedFinalsError, match="streaming pass"):
        ST.FAMILY.sealed_rows(None, {}, {})


def test_sealed_read_matches_a_row_by_row_recomputation(world: dict) -> None:
    draft = {**json.loads(world["list"].read_text()), "status": SF.STATUS_DRAFT, "approval": None}
    draft_path = world["tmp"] / "draft.json"
    draft_path.write_text(json.dumps(draft))
    with pytest.raises(SF.SealedFinalsError, match="only an approved list"):
        ST.run_sealed(draft_path, world["plan"], world["kit"])
    results, count = ST.run_sealed(world["list"], world["plan"], world["kit"], batch_size=7, workers=2)
    assert count == len(world["nets"])
    _check_records(world, results)
    events = [r["event"] for r in read_registry(world["tmp"] / "sealed" / "registry.jsonl")]
    assert events[0] == "evaluation_started" and events[-1] == "evaluation_finished"
    assert events.count(ST.UNIT_FINISHED) == V.N_SHARDS and events.count("entry_finished") == count
    with pytest.raises(SF.SealedFinalsError, match="already read"):
        ST.run_sealed(world["list"], world["plan"], world["kit"])


def test_interrupted_read_resumes_with_the_finished_units(world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    real = ST.read_unit

    def failing(plan, unit):
        if unit == 3:
            raise MemoryError("synthetic worker failure")
        return real(plan, unit)

    monkeypatch.setattr(ST, "read_unit", failing)
    with pytest.raises(MemoryError):
        ST.run_sealed(world["list"], world["plan"], world["kit"], batch_size=4)
    registry = world["tmp"] / "sealed" / "registry.jsonl"
    events = [r["event"] for r in read_registry(registry)]
    assert events[-1] == ST.EVALUATION_INTERRUPTED and events.count(ST.UNIT_FINISHED) == 3
    with pytest.raises(SF.SealedFinalsError, match="already read or attempted"):
        ST.run_sealed(world["list"], world["plan"], world["kit"], batch_size=4)
    with pytest.raises(SF.SealedFinalsError, match="plan_sha256"):  # another batch size is another plan
        ST.run_sealed(world["list"], world["plan"], world["kit"], batch_size=5, resume=True)
    monkeypatch.setattr(ST, "read_unit", real)
    results, _ = ST.run_sealed(world["list"], world["plan"], world["kit"], batch_size=4, resume=True)
    _check_records(world, results)
    records = read_registry(registry)
    units = [r["unit"] for r in records if r["event"] == ST.UNIT_FINISHED]
    assert sorted(units) == list(range(V.N_SHARDS)) and len(units) == V.N_SHARDS
    assert [r["event"] for r in records].count(ST.EVALUATION_RESUMED) == 1
