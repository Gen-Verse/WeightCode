"""PREREG T04 (dataset ``in1k_t04``): the row rule and its recorded hashes, the class-name rule, the uint8 crop against
T03's preprocessing, the official ImageNet-C functions taken from their file, the corruption pipeline, the severity
rule and the gain bar, the uint8 native forward, the harness module, the reference CLI dispatch, and a tiny store built
end to end through the real instance builder (random-weight networks; no ImageNet image, no weights file, no shard).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weightcode.baselines.fit import cli as C
from weightcode.baselines.fit import ftcompare as FT
from weightcode.baselines.fit import torch_forward as TF
from weightcode.data import imagenet_val as V
from weightcode.data import t04 as T
from weightcode.evalkit.instances import leak_tokens, scan_for_tokens
from weightcode.harness import t04 as H
from weightcode.models import resnet18native as N
from weightcode.models import resnet18up as M

OFFICIAL_FUNCTIONS = '''import os
import wand  # never executed by load_official

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

for name in []:
    raise SystemExit("the loop is never executed")
'''
"""The two functions as the official file at OFFICIAL_COMMIT writes them, inside a file whose other lines must not run."""


def _official(tmp_path: Path) -> dict:
    path = tmp_path / "make_imagenet_c.py"
    path.write_text(OFFICIAL_FUNCTIONS, encoding="utf-8")
    import hashlib

    return T.load_official(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _jpeg(rng: np.random.Generator, height: int, width: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


READOUT_CLASSES = 8


def _params(seed: int) -> dict[str, np.ndarray]:
    from weightcode.data.t04_build import decoy_params

    params = decoy_params(seed)
    rng = np.random.default_rng(seed)
    for name in params:
        if name.endswith("running_mean"):
            params[name] = (0.1 * rng.standard_normal(params[name].shape)).astype(np.float32)
        elif name.endswith("running_var"):
            params[name] = (0.5 + rng.random(params[name].shape)).astype(np.float32)
    return params


# --- rows ---------------------------------------------------------------------------------------------------------


def test_row_rule_partitions_the_validation_rows_and_matches_the_prereg_hashes() -> None:
    dev, test = T.dev_rows(), T.test_rows()
    assert len(dev) == T.N_DEV == 12_000 and len(test) == T.N_TEST == 38_000
    assert not set(dev.tolist()) & set(test.tolist()) and len(set(dev.tolist()) | set(test.tolist())) == 50_000
    assert set(range(0, 50_000, 50)) <= set(dev.tolist())
    record = T.row_record()
    assert record["dev"]["sha256"] == "8abf52d92ec992161fc084d42fd5d4b8a3a85cafb58f743ffad78fe639d999b9"
    assert record["test"]["sha256"] == "471d25563a658afe65dd22b770a1c4358856d9820bd3b80e00705d606d164844"
    assert record["d_in_dev"] and "splits" not in record


def test_split_rule_covers_every_class_in_v_select_and_v_check() -> None:
    dev = T.dev_rows()
    rng = np.random.default_rng(5)
    labels = {int(r): int(c) for r, c in zip(dev, rng.permutation(np.arange(len(dev)) % T.N_CLASSES))}
    splits = T.split_rows(labels)
    assert [len(splits[s]) for s, _ in T.SPLIT_SIZES] == [n for _, n in T.SPLIT_SIZES]
    assert sorted(np.concatenate(list(splits.values())).tolist()) == dev.tolist()
    for split in T.COVERED_SPLITS:
        assert len({labels[int(r)] for r in splits[split]}) == T.N_CLASSES
    assert {k: v.tolist() for k, v in T.split_rows(labels).items()} == {k: v.tolist() for k, v in splits.items()}
    record = T.row_record(labels)
    assert set(record["splits"]) == {s for s, _ in T.SPLIT_SIZES} and sum(record["d_by_split"].values()) == 1000
    short = dict(labels)
    for row in [r for r, c in labels.items() if c == 7][1:]:
        short[row] = 8
    with pytest.raises(T.T04DataError, match="fewer than 2 dev rows"):
        T.split_rows(short)
    with pytest.raises(T.T04DataError, match="no label"):
        T.split_rows({r: c for r, c in labels.items() if r != int(dev[3])})


def test_test_rows_are_refused_before_any_shard_is_opened(tmp_path: Path) -> None:
    with pytest.raises(T.T04DataError, match="not dev rows"):
        list(T.read_dev_records(tmp_path / "no_such_dir", [0, 6]))
    assert T.check_dev_rows([0, 5, 25, 30]) == [0, 5, 25, 30]


def test_locate_follows_the_shard_order() -> None:
    counts = [3572] * 6 + [3571] * 8
    located = T.locate([0, 3571, 3572, 49_999], counts)
    assert [(r.shard_index, r.row) for r in located] == [(0, 0), (0, 3571), (1, 0), (13, 3570)]
    assert located[0].shard == V.shard_names()[0]


# --- class names --------------------------------------------------------------------------------------------------


def test_class_name_rule() -> None:
    classes = [(f"n{i:08d}", f"thing {i}") for i in range(1000)]
    classes[3] = ("n00000003", "maillot")
    classes[4] = ("n00000004", "maillot, tank suit")
    classes[5] = ("n00000005", "rapeseed")
    classes[6] = ("n00000006", "white noise box, hiss box")
    classes[7] = ("n00000007", "cougar, puma, Felis concolor")
    names, changes = T.class_names(classes, T.forbidden_name_tokens())
    assert names[3] == "maillot" and names[4] == "maillot, tank suit"  # shared first synonym: the full entry
    assert names[5] == "n00000005"  # no clean synonym: the WordNet id
    assert names[6] == "hiss box"  # first clean synonym
    assert names[7] == "cougar"  # only a later synonym holds a token
    assert {c["label"] for c in changes} == {4, 5, 6}
    assert len(set(names)) == 1000


def test_leak_words_are_forbidden_in_class_names() -> None:
    assert set(H.leak_words()) <= set(T.forbidden_name_tokens())
    assert not {"snow", "digital"} & set(H.leak_words())
    assert {"gaussian", "contrast", "noise", "severity", "imagenet-c", "hendrycks"} <= set(H.leak_words())


# --- images and the official functions ----------------------------------------------------------------------------


def test_center_crop_is_t03_preprocessing_before_the_scaling() -> None:
    rng = np.random.default_rng(3)
    for height, width in ((300, 400), (517, 260)):
        data = _jpeg(rng, height, width)
        crop = T.center_crop_u8(data)
        assert crop.dtype == np.uint8 and crop.shape == (3, 224, 224)
        mean = np.array(M.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(M.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        ours = (crop.astype(np.float32) / 255.0 - mean) / std
        np.testing.assert_allclose(ours, V.preprocess(data), atol=2e-6)


def test_official_functions_are_taken_from_the_file_and_checked(tmp_path: Path) -> None:
    official = _official(tmp_path)
    assert set(official["_sources"]) == {"gaussian_noise", "contrast"}
    record = T.check_official_constants(official)
    assert record["s5"] == {"contrast": 0.05, "gaussian_noise": 0.38}
    (tmp_path / "other.py").write_text(OFFICIAL_FUNCTIONS.replace("0.38", "0.39"), encoding="utf-8")
    with pytest.raises(T.T04DataError, match="sha256"):
        T.load_official(tmp_path / "other.py")


def test_official_file_of_record_when_present() -> None:
    import os

    path = os.environ.get("WEIGHTCODE_T04_OFFICIAL")
    if not path:
        pytest.skip("set WEIGHTCODE_T04_OFFICIAL to the official make_imagenet_c.py to check the file of record")
    official = T.load_official(Path(path))
    T.check_official_constants(official)


def test_corruption_is_seeded_per_row_and_paired_across_severities(tmp_path: Path) -> None:
    official = _official(tmp_path)
    crop = np.random.default_rng(1).integers(0, 256, size=(3, 224, 224), dtype=np.uint8)
    a = T.corrupt_crop(official, "gaussian_noise", 3, crop, 25)
    assert np.array_equal(a, T.corrupt_crop(official, "gaussian_noise", 3, crop, 25))
    assert not np.array_equal(a, T.corrupt_crop(official, "gaussian_noise", 3, crop, 26))
    np.random.seed(0)
    first = T.corrupt_crop(official, "contrast", 2, crop, 25)
    np.random.seed(1)
    assert np.array_equal(first, T.corrupt_crop(official, "contrast", 2, crop, 25))
    assert a.dtype == np.uint8 and a.shape == crop.shape
    with pytest.raises(ValueError):
        T.corrupt_crop(official, "snow", 3, crop, 25)
    x = np.stack([crop, crop])
    out = T.corrupt_split(official, "gaussian_noise", 1, x, np.array([0, 1]))
    assert not np.array_equal(out[0], out[1])


def test_severity_rule_and_gain_bar() -> None:
    choice = T.choose_severity(0.684, {1: 0.55, 2: 0.42, 3: 0.24, 4: 0.09, 5: 0.02})
    assert choice["chosen"] == 3 and not choice["by_severity"]["s1"]["qualifies"]  # s1 drops 13.4 points
    assert T.choose_severity(0.684, {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01})["chosen"] is None
    q_e10 = T.retained_share(0.23775, 0.91275, chance=0.1)
    assert abs(q_e10 - 0.1695) < 1e-3 and T.Q_MIN <= q_e10
    assert T.gain_bar(67.5) == 10.0 and T.gain_bar(66.625) == 10.0 and T.gain_bar(46.0) == 7.0


# --- network ------------------------------------------------------------------------------------------------------


def test_u8_forward_matches_torch_and_the_float_path() -> None:
    params = _params(4)
    namespace: dict = {}
    exec(compile(N.render_forward_u8(), "forward.py", "exec"), namespace)
    x = np.random.default_rng(5).integers(0, 256, size=(2, 3, 224, 224), dtype=np.uint8)
    ours = np.asarray(namespace["forward"](params, x), dtype=np.float64)
    ref64 = N.eval_logits_u8(N.build_resnet18native_u8(params, torch.float64), x)
    assert np.max(np.abs(ours - ref64) / np.maximum(1.0, np.abs(ref64))) < 1e-6
    mean = np.array(M.IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(M.IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    float_path = N.eval_logits(N.build_resnet18native(params, torch.float64),
                               ((x.astype(np.float32) / 255.0 - mean) / std).astype(np.float32))
    assert np.max(np.abs(float_path - ref64)) < 1e-4


def test_torch_rebuild_of_the_u8_family_matches_forward_py() -> None:
    params = _params(6)
    spec = N.build_resnet18native_u8_spec("m00000000", [f"c{i}" for i in range(1000)], params)
    arch = TF.architecture_from(params, spec)
    assert arch.family == TF.RESNET18NAT and TF.RESNET18NAT in TF.BN_FAMILIES
    namespace: dict = {}
    exec(compile(N.render_forward_u8(), "forward.py", "exec"), namespace)
    x = np.random.default_rng(7).integers(0, 256, size=(2, 3, 224, 224), dtype=np.uint8)
    TF.assert_parity(arch, params, namespace["forward"], x)


# --- harness and CLI ----------------------------------------------------------------------------------------------


def test_harness_student_check_timeouts_and_lookups() -> None:
    params = _params(8)
    summary = H.check_student(params)
    assert summary["n_tensors"] == 102 and summary["n_batchnorm_stat_entries"] == 9_600
    small = dict(params, **{"fc.weight": np.zeros((10, 512), np.float32), "fc.bias": np.zeros(10, np.float32)})
    with pytest.raises(H.T04HarnessError):
        H.check_student(small)
    private = {"dataset": H.DATASET, "evidence_rule": "evidence-v1"}
    assert H.is_t04(private) and not H.is_t04({**private, "dataset": "cifar10_r18"})
    assert H.mcp_timeouts(private, None, None, (1, 1)) == (H.MCP_TOOL_TIMEOUT_MS, H.MCP_CONNECT_TIMEOUT_MS)
    with pytest.raises(ValueError):
        H.mcp_timeouts(private, 600_000, None, (1, 1))
    assert len(H.CONDITIONS) == 10 and "gaussian_noise_s3" in H.CONDITIONS
    with pytest.raises(H.T04HarnessError):
        H.find("snow_s5")


def test_editor_visible_text_has_no_leak_token() -> None:
    tokens = set(leak_tokens(H.DATASET)) | set(H.leak_words())
    assert scan_for_tokens(N.render_forward_u8(), tokens) == []
    spec = N.build_resnet18native_u8_spec("m00000000", [f"c{i}" for i in range(1000)], _params(9))
    assert scan_for_tokens(json.dumps(spec), tokens) == []


def test_cli_dispatch_for_t04(tmp_path: Path) -> None:
    instance = SimpleNamespace(private={"dataset": C.T04_DATASET})
    assert C.T04_METHODS == C.E12_METHODS and C.T04_DATASET == T.DATASET == H.DATASET
    with pytest.raises(ValueError, match="runs with t04"):
        C.run_method(instance, "grad_last", tmp_path, run_seed=0, allow_main=False, joint=True, e10=True)
    with pytest.raises(ValueError, match="not two of them"):
        C.run_method(instance, "grad_last", tmp_path, run_seed=0, allow_main=False, joint=True, e10=True, e12=True,
                     t04=True)
    other = SimpleNamespace(private={"dataset": "cifar10_r20"})
    with pytest.raises(ValueError, match="need a in1k_t04"):
        C.run_method(other, "grad_last", tmp_path, run_seed=0, allow_main=False, joint=True, e10=True, t04=True)
    assert FT.F1_DATASET_METHODS[FT.F1_T04_DATASET] == FT.F1_DATASET_METHODS[FT.F1_E12_DATASET]
    assert FT.F1_DATASET_REUSED[FT.F1_T04_DATASET]["full"] == ("grad_full", "t04g2")


# --- a tiny store end to end --------------------------------------------------------------------------------------


@pytest.fixture()
def tiny_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A store with 72 V_evidence, 8 V_check, 8 V_select and 64 train rows of synthetic JPEGs, 1000 synthetic classes
    (with the duplicate and the forbidden-token cases), random-weight networks, one condition."""
    from weightcode.data import t04_build as B

    for name in ("private", "sealed", "runs"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(f"WEIGHTCODE_{name.upper()}", str(tmp_path / name))
    sizes = {"V_evidence": 72, "V_check": 8, "V_select": 8, "train": 64}
    dev = T.dev_rows()[: sum(sizes.values())]
    order = np.random.default_rng(0).permutation(dev)
    splits, start = {}, 0
    for split, n in sizes.items():
        splits[split] = np.sort(order[start:start + n])
        start += n
    monkeypatch.setattr(T, "dev_rows", lambda: dev)
    monkeypatch.setattr(T, "split_rows", lambda labels: splits)
    rng = np.random.default_rng(1)
    labels = {int(r): int(rng.integers(0, 1000)) for r in dev}
    for split in ("V_check", "V_select"):  # classes 0 .. 7 in each, so a readout over READOUT_CLASSES sees them all
        for k, row in enumerate(splits[split]):
            labels[int(row)] = k

    def records(data_dir: Path, rows):
        assert data_dir.name == T.SHARD_DIR and (data_dir.parent / "classes.py").is_file()  # the snapshot layout
        for row in T.check_dev_rows(rows):
            data = _jpeg(rng, 240 + row % 7, 300)
            import hashlib

            yield V.DRecord(row, "shard", row, labels[row], hashlib.sha256(data).hexdigest(), f"img{row}.JPEG", data)

    monkeypatch.setattr(T, "read_dev_records", records)
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    lines = [f'    "n{i:08d}": "thing {i}",' for i in range(1000)]
    lines[5] = '    "n00000005": "rapeseed",'
    (data_dir / "classes.py").write_text("IMAGENET2012_CLASSES = OrderedDict({\n" + "\n".join(lines) + "\n})\n")
    monkeypatch.setattr(N, "imagenet_params", lambda path: _params(11))
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"not a real file")
    return {"data_dir": data_dir, "weights": weights, "B": B, "official": _official(tmp_path), "tmp": tmp_path}


def test_tiny_store_end_to_end(tiny_store: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from weightcode.data.store import load_split, read_dataset_json
    from weightcode.evalkit.instances import build_deployment_instances
    from weightcode.evalkit.targets import SeedSelection, deployment_splits

    B = tiny_store["B"]
    info = {"note": "test"}
    manifest = B.build_clean(tiny_store["data_dir"], info, verify_shards=False)
    assert manifest["class_names"][5] == "n00000005" and manifest["test"]["n"] == 38_000
    with pytest.raises(T.T04DataError, match="exists"):
        B.build_clean(tiny_store["data_dir"], info, verify_shards=False)
    train = load_split(T.DATASET, "train")
    assert train["x"].dtype == np.uint8 and train["x"].shape[1:] == (3, 224, 224)
    assert all(T.is_dev_row(r) for r in train["idx"])
    models = B.build_models(tiny_store["weights"], info)
    assert [m["seed"] for m in models] == [100, 101]
    monkeypatch.setattr(T, "load_official", lambda path: tiny_store["official"])
    choice = {"chosen": 3}
    meta = B.build_condition("gaussian_noise", 3, Path("unused"), info, choice)
    assert meta["condition"]["name"] == "gaussian_noise_s3"
    with pytest.raises(T.T04DataError, match="chose"):
        B.build_condition("contrast", 2, Path("unused"), info, {"chosen": 4})
    splits = deployment_splits(T.DATASET, "gaussian_noise_s3")
    assert splits.v_check.exists()
    cond = load_split(T.DATASET, "V_check", "gaussian_noise_s3")
    clean = load_split(T.DATASET, "V_check")
    assert np.array_equal(cond["y"], clean["y"]) and not np.array_equal(cond["x"], clean["x"])
    outcomes = build_deployment_instances(
        T.DATASET, "gaussian_noise_s3", selection=SeedSelection(main=(), dev=(100, 101)), decoys={100: 101, 101: 100},
        split_roles=("dev",), batch_size=16)
    assert sorted(o.seed for o in outcomes) == [100, 101]
    assert all(o.status == "created" for o in outcomes)
    instance_dir = tiny_store["tmp"] / "private" / "instances" / outcomes[0].instance_id
    task = json.loads((instance_dir / "task.json").read_text())
    tokens = set(leak_tokens(T.DATASET)) | set(H.leak_words())
    assert scan_for_tokens(json.dumps(task), tokens) == []
    assert H.find("gaussian_noise_s3", 100) in {o.instance_id for o in outcomes}
    assert read_dataset_json(T.DATASET)["splits"]["V_check"]["n"] == 8
    # the timing of record, with the large-sample step timing that sizes R37's large-sample jobs
    import importlib.util

    spec = importlib.util.spec_from_file_location("t04_store", Path(__file__).parents[1] / "scripts/t04/t04_store.py")
    store_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(store_cli)
    timing_path = tiny_store["tmp"] / "timing.json"
    assert store_cli.main(["timing", "--out", str(timing_path), "--repeats", "1", "--train-threads", "2",
                           "--train-steps", "1"]) == 0
    timing = json.loads(timing_path.read_text())
    assert timing["rows_per_check"] == 16 and timing["session_wall_limit_min"] == 120
    step = timing["large_ft_step_s"]["2"]
    assert step["batch"] == 128 and step["torch_threads"] == 2 and len(step["steps_s"]) == 1
    # the LEARN dry run of R37: trial assembly, the leak scan and the session record, no editor
    from weightcode.harness import runner
    from weightcode.harness.finalize import STATUS_DRY_RUN

    result = runner.run_session(H.find("gaussian_noise_s3", 100), "LEARN", 1, effort="max", wall_timeout_s=8100.0,
                                dry_run=True, retention_floor_frac=0.25)  # 8 clean rows here, 4,000 on NSCC
    if "t04" in str(Path(__file__).resolve().parents[1]).lower():
        # this checkout's own path holds the leak token "t04" (the dataset id's part), and the sandbox self-test
        # rightly finds it in the mount table; NSCC mounts the code at WeightCode-nsess. That must be the only failure.
        selftest = json.loads((result.root / "logs" / "selftest.json").read_text())
        assert selftest["failed"] == ["forbidden_strings_absent"]
        assert json.loads(selftest["checks"]["forbidden_strings_absent"]["detail"]) == ["'t04' in mountinfo"]
    else:
        assert result.status == STATUS_DRY_RUN
    session = json.loads((result.root / "session.json").read_text())
    assert session["leak_scan"]["passed"]
    assert session["mcp"]["tool_timeout_ms"] >= 1_200_000 and session["mcp"]["connect_timeout_ms"] >= 900_000
    assert (session["limits"]["max_ops"], session["limits"]["max_rel_frob"], session["limits"]["max_changed_frac"]) \
        == (102, 0.5, 0.2)
    assert set(H.leak_words()) <= set(session["forbidden_strings"]["text"])
    shown = " ".join(p.read_text() for p in (result.root / "prompts").glob("*.md"))
    assert scan_for_tokens(shown, set(H.leak_words())) == []


def test_tiny_store_reference_arms(tiny_store: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """bn_adapt through the reference CLI with t04, and the large-sample fine-tuning diagnostic with small constants,
    both through execute_run's joint oracle under the E10 limits."""
    import importlib.util

    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import build_deployment_instances
    from weightcode.evalkit.targets import SeedSelection

    B = tiny_store["B"]
    B.build_clean(tiny_store["data_dir"], {}, verify_shards=False)
    B.build_models(tiny_store["weights"], {})
    monkeypatch.setattr(T, "load_official", lambda path: tiny_store["official"])
    B.build_condition("contrast", 4, Path("unused"), {}, {"chosen": 4})
    outcomes = build_deployment_instances(
        T.DATASET, "contrast_s4", selection=SeedSelection(main=(), dev=(100, 101)), decoys={100: 101, 101: 100},
        split_roles=("dev",), batch_size=16)
    from weightcode.evalkit.instances import instances_root

    instance = load_instance(instances_root() / next(o.instance_id for o in outcomes if o.seed == 100))
    record = C.run_method(instance, "bn_adapt", tiny_store["tmp"] / "runs" / "bn", run_seed=0, allow_main=False,
                          joint=True, e10=True, t04=True)
    assert record["status"] == "complete" and record["queries_used"] >= 1
    # the edge extension (PREREG T04 amendment, main 9f0baa9): the same reference on its extension grid
    edge = C.run_method(instance, "bn_adapt", tiny_store["tmp"] / "runs" / "bn_edge", run_seed=0, allow_main=False,
                        joint=True, e10=True, t04=True, edge=True)
    assert edge["status"] == "complete" and edge["config"]["grid"]["name"] == "t04_edge"
    assert [c["config"] for c in edge["candidates"]] == [{"N": n} for n in (0, 256, 512, 1024, 2048, 4096, 8192, 16384)]
    spec = importlib.util.spec_from_file_location("t04_large_ft", Path(__file__).parents[1] / "scripts/t04/large_ft.py")
    large = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(large)
    monkeypatch.setattr(large, "STEP_CHECKPOINTS", (1, 2))
    monkeypatch.setattr(large, "HALF_BATCH", 4)
    assert large.main(["--instance", instance.instance_id]) == 0
    run = json.loads((tiny_store["tmp"] / "runs" / "baselines" / instance.instance_id / "ft_large" / "t04" / "0" /
                      "run.json").read_text())
    assert run["evidence"]["rule"] == large.RULE and run["evidence"]["n_rand"] == 72 and run["evidence"]["n_replay"] == 64
    assert run["queries_used"] == 4 and run["limits_source"].startswith("e10_program_limits")
    # the kept-pick dev readout every R37 fitting job runs on its own runs (V_select and its clean companion)
    from weightcode.evalkit import devreadout

    # the tiny V_select holds classes 0 .. 7 only; the real one holds all 1,000 (T.split_rows), which the per-class
    # readout requires
    real = devreadout.split_readouts
    monkeypatch.setattr(devreadout, "split_readouts",
                        lambda data, base, child, target, n_classes: real(data, base, child, target, READOUT_CLASSES))
    out = tiny_store["tmp"] / "kept.jsonl"
    assert devreadout.main([str(tiny_store["tmp"] / "runs" / "bn"), str(tiny_store["tmp"] / "runs" / "bn_edge"),
                            str(tiny_store["tmp"] / "runs" / "baselines"),
                            "--retention-floor", "-20", "--label", "t04-kept", "--out", str(out)]) == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(r["method"] for r in records) == ["bn_adapt", "bn_adapt", "ft_large"]
    for r in records:
        assert r["selection"]["n_checks"] >= 1 and {"readouts", "readouts_clean"} <= set(r)
