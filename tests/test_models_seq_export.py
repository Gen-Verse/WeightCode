"""Export and import of seqops students, and batch prefetch in training (no effect on the trained parameters).

Nothing here reads the real stores: WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point to tmp dirs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.data import seqops as S
from weightcode.data import seqops_build as B
from weightcode.data.store import forward_path, load_split, models_dir, read_dataset_json, read_json
from weightcode.models import train_seq as TS
from weightcode.models import transformer as T
from weightcode.models.export import read_index
from weightcode.models.forward_gen import render_seq_forward, write_forward
from weightcode.models.seq_export import export_seq_model, import_seq_models

TINY = TS.SeqTrainConfig(d_model=16, n_layers=1, d_mlp=32, steps=4, warmup_steps=1, log_every=2)


def _make_store(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("private", "sealed"):
        (root / name).mkdir(parents=True)
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(root / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(root / "sealed"))


@pytest.fixture
def local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _make_store(tmp_path / "local", monkeypatch)
    sha = write_forward(forward_path(S.DATASET), render_seq_forward(**B.forward_constants()))
    B.write_built(B.build_base(), {"forward_path": "forward.py", "forward_sha256": sha})
    return tmp_path / "local"


def _train(seed: int = 100) -> TS.SeqTrainResult:
    torch.manual_seed(seed)
    exclude = S.held_out_keys(S.build_base_splits())
    return TS.train_seq(TINY, seed, exclude, torch.device("cpu"))


def test_prefetch_depth_does_not_change_the_trained_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TS, "PREFETCH", 1)
    one = T.export_seq_params(_train().model)
    monkeypatch.setattr(TS, "PREFETCH", 4)
    four = T.export_seq_params(_train().model)
    assert one.keys() == four.keys() and all(np.array_equal(one[k], four[k]) for k in one)


def _fake_run(local: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str = "base") -> Path:
    """An NSCC-like run dir: the job's private store (inputs copied from local), one exported student, run files."""
    run = tmp_path / "run"
    store = run / "private" / S.DATASET
    (store / "splits").mkdir(parents=True)
    source = local / "private" / S.DATASET
    for name in ("dataset.json", "forward.py", "splits/V_select.npz"):
        shutil.copy(source / name, store / name)
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(run / "private"))
    manifest = read_dataset_json(S.DATASET)
    result = _train()
    entry = export_seq_model(manifest=manifest, role=role, seed=100, config=TINY, result=result,
                             size_check={"em": 0.0, "passed": False}, determinism={},
                             run_info={"versions": {"torch": torch.__version__}, "device": "cpu"},
                             vselect=load_split(S.DATASET, "V_select"))
    (models_dir(S.DATASET) / "index.json").write_text(json.dumps([entry]))
    (run / "meta.json").write_text(json.dumps({"run_name": "r1", "request": "R-test", "select": "select=1:ngpus=1",
                                               "walltime": "00:12:00", "code_digest": "abc"}))
    (run / "exit.json").write_text(json.dumps({"exit_code": 0, "elapsed_s": 10}))
    (run / "environment.txt").write_text(f"{'0' * 64} /images/container.sif\n")
    (run / "job_id.txt").write_text("1.pbs\n")
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(local / "private"))
    return run


def test_export_then_import_with_local_rechecks(local: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _fake_run(local, tmp_path, monkeypatch)
    row = json.loads((run / "private" / S.DATASET / "models" / "index.json").read_text())[0]
    exported = read_json(run / "private" / S.DATASET / "models" / row["model_id"] / "private.json")
    assert exported["parity"]["max_abs_diff"] <= 1e-4 and exported["reserved_rows_unchanged"]
    assert exported["train_data"]["held_out"] == read_dataset_json(S.DATASET)["held_out"]
    spec = read_json(run / "private" / S.DATASET / "models" / row["model_id"] / "model_spec.json")
    assert spec["layers"][-1]["op"] == "linear" and spec["input"]["kind"] == "token_ids"

    tampered = tmp_path / "tampered"
    shutil.copytree(run, tampered)
    index_file = tampered / "private" / S.DATASET / "models" / "index.json"
    index_file.write_text(json.dumps([{**row, "vselect_acc": row["vselect_acc"] + 0.5}]))
    with pytest.raises(ValueError, match="exact match"):
        import_seq_models(tampered)
    assert read_index(S.DATASET) == []

    checks = import_seq_models(run, note="test import")
    assert checks[0]["model_id"] == row["model_id"] and checks[0]["local_parity_max_abs_diff"] <= 1e-4
    assert read_index(S.DATASET) == [row]
    imported = read_json(models_dir(S.DATASET) / row["model_id"] / "private.json")
    assert imported["import"]["note"] == "test import" and imported["import"]["request"] == "R-test"
    with pytest.raises(ValueError, match="already used"):
        import_seq_models(run)


def _as_4090_run(run: Path, reason: str = "NUS unreachable: VPN down") -> None:
    """Turn the NSCC-like run dir into a 4090 fallback run dir as run_4090.sh and job_4090.sh write it."""
    meta = json.loads((run / "meta.json").read_text())
    (run / "meta.json").write_text(json.dumps({"run_name": meta["run_name"], "host": "4090", "name": "seqops_base",
                                               "reason": reason, "venv": "/home/bob/weightcode/venv_seqops",
                                               "code_digest": "abc"}))
    (run / "job_id.txt").unlink(missing_ok=True)
    freeze = "numpy==2.5.4\ntorch==2.14.0+cu130\n"
    (run / "venv_freeze.txt").write_text(freeze)
    digest = hashlib.sha256(freeze.encode()).hexdigest()
    (run / "environment.txt").write_text(f"$ hostname\nbob\nvenv /home/bob/weightcode/venv_seqops\npython_version 3.12.4\n"
                                         f"venv_freeze_sha256 {digest}\n")


def test_import_of_a_4090_fallback_run(local: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _fake_run(local, tmp_path, monkeypatch)
    _as_4090_run(run)
    broken = tmp_path / "broken"
    shutil.copytree(run, broken)
    (broken / "venv_freeze.txt").write_text("numpy==2.1.0\n")
    with pytest.raises(ValueError, match="sha256 differs"):
        import_seq_models(broken)
    unnamed = tmp_path / "unnamed"
    shutil.copytree(run, unnamed)
    _as_4090_run(unnamed, reason=" ")
    with pytest.raises(ValueError, match="names no reason"):
        import_seq_models(unnamed)
    assert read_index(S.DATASET) == []

    (check,) = import_seq_models(run, note="4090 fallback")
    imported = read_json(models_dir(S.DATASET) / check["model_id"] / "private.json")["import"]
    assert imported["host"] == "4090" and imported["fallback_reason"] == "NUS unreachable: VPN down"
    assert imported["environment"]["packages"] == {"numpy": "2.5.4", "torch": "2.14.0+cu130"}
    assert imported["request"] is None and "container" not in imported


def test_import_refuses_a_smoke_model(local: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _fake_run(local, tmp_path, monkeypatch)
    row = json.loads((run / "private" / S.DATASET / "models" / "index.json").read_text())[0]
    private_file = run / "private" / S.DATASET / "models" / row["model_id"] / "private.json"
    private_file.write_text(json.dumps({**json.loads(private_file.read_text()), "smoke": True}))
    with pytest.raises(ValueError, match="smoke"):
        import_seq_models(run)


def test_export_refuses_a_config_the_forward_cannot_run(local: Path) -> None:
    from weightcode.models.seq_export import check_model_config

    check_model_config(TINY.model_config())
    with pytest.raises(ValueError):
        check_model_config(T.TfmConfig(S.VOCAB_SIZE, S.SEQ_LEN, S.N_SYMBOLS, S.PAD, d_model=16, n_heads=2))


def test_parity_bounds_scale_with_the_logits(local: Path) -> None:
    """A common logit offset leaves the forward exact; the float32 bound grows with it, and a forward off by a few
    float32 steps fails the float64 test even where the absolute difference is small."""
    from weightcode.models.export import PARITY_TOL, ParityError, load_forward_no_cache
    from weightcode.models.seq_export import FLOAT32_STEP, seq_parity

    forward, _ = load_forward_no_cache(forward_path(S.DATASET))
    config = TINY.model_config()
    torch.manual_seed(0)
    params = T.export_seq_params(T.build_seq_model(config))
    x = np.ascontiguousarray(load_split(S.DATASET, "V_select")["x"][:64])
    plain = seq_parity(config, params, x, forward)
    assert plain["float64_max_rel_diff"] <= FLOAT32_STEP and plain["max_abs_diff"] <= plain["tolerance"]

    shifted = {**params, "head.bias": params["head.bias"] - np.float32(3e4)}
    result = seq_parity(config, shifted, x, forward)
    assert result["logit_scale"] > 3e4 and result["tolerance"] == pytest.approx(PARITY_TOL * result["logit_scale"])
    assert result["float64_max_rel_diff"] <= FLOAT32_STEP

    def off_by_steps(p: dict, rows: np.ndarray) -> np.ndarray:
        logits = forward(p, rows).astype(np.float64)
        return (logits * (1.0 + 4 * FLOAT32_STEP) + 4 * FLOAT32_STEP).astype(np.float32)

    with pytest.raises(ParityError, match="float64 torch rebuild"):
        seq_parity(config, params, x, off_by_steps)
