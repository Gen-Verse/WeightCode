"""A SCAN-shaped store for the E13 tests, written from the grammar; never the real SCAN files or stores.

``fake_raw`` writes a checkout whose files follow the official splits' definitions: addprim_jump's train is every
command without jump plus "jump" 1467 times and its test every other command with jump; around_right's train is
every command without "around right" (15,225) and its test every command with it except the 1,209 with "turn around
right", which the official template split leaves out of both files (4,476).
``scan.RAW_SHA256`` is patched to these files' hashes, and WEIGHTCODE_PRIVATE, WEIGHTCODE_SEALED and WEIGHTCODE_RUNS point into the
test's tmp dir. ``built_store`` also installs the raw copy, forward.py and both conditions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weightcode.data import scan as S
from weightcode.data import scan_build as B


def _write(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"IN: {c} OUT: {' '.join(S.interpret(c))}\n" for c in rows))


def fake_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "checkout"
    commands = S.grammar_commands()
    rng = np.random.default_rng(7)
    shuffled = [commands[i] for i in rng.permutation(len(commands))]
    _write(raw / "tasks.txt", shuffled)
    no_jump = [c for c in shuffled if not S.has_primitive(c, "addprim_jump")]
    _write(raw / S.CONDITION_FILES["addprim_jump"]["train"], ["jump"] * 1467 + [c for c in no_jump if c != "jump"])
    _write(raw / S.CONDITION_FILES["addprim_jump"]["test"],
           [c for c in shuffled if S.has_primitive(c, "addprim_jump") and c != "jump"])
    no_ar = [c for c in shuffled if not S.has_primitive(c, "around_right")]
    _write(raw / S.CONDITION_FILES["around_right"]["train"], no_ar)
    _write(raw / S.CONDITION_FILES["around_right"]["test"],
           [c for c in shuffled if S.has_primitive(c, "around_right") and "turn around right" not in c])
    (raw / "README.md").write_text("readme\n")
    (raw / "LICENSE").write_text("license\n")
    monkeypatch.setattr(S, "RAW_SHA256", {rel: S.file_sha256(raw / rel) for rel in S.RAW_SHA256})
    for name in ("private", "sealed", "runs"):
        (tmp_path / name).mkdir(exist_ok=True)
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(tmp_path / "sealed"))
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(tmp_path / "runs"))
    return raw


def built_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = fake_raw(tmp_path, monkeypatch)
    B.install_raw(raw)
    B.write_base({"note": "test"})
    for condition in S.CONDITIONS:
        B.write_condition(B.build_condition(condition), {"note": "test"})
    return tmp_path


def store_with_students(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int = 100) -> Path:
    """``built_store`` plus one tiny tfm student per condition (d 16, 1 layer, 2 epochs), exported into the store
    and indexed as a base. A student this small cannot reach the 0.95 gate, so the gate is set to 0 in the exporter
    and in the instance builder: the tests exercise the instance and reference code, not the student's skill."""
    import dataclasses

    import torch

    from weightcode.data.scan_build import checked_split
    from weightcode.data.store import read_dataset_json
    from weightcode.evalkit import scanskill
    from weightcode.models import scan_export
    from weightcode.models import train_scan as TS
    from weightcode.models.export import write_index

    built_store(tmp_path, monkeypatch)
    monkeypatch.setattr(scan_export, "GATE_EM", 0.0)
    monkeypatch.setattr(scanskill, "GATE_EM", 0.0)
    tiny = dataclasses.replace(TS.SIZE_STEPS[0], d_model=16, n_layers=1, d_mlp=32, epochs=2, warmup_steps=2,
                               log_every_epochs=1, batch_size=512)
    for condition in S.CONDITIONS:
        torch.manual_seed(seed)
        result = TS.train_scan(tiny, seed, checked_split(condition, "train"), torch.device("cpu"))
        entry = scan_export.export_scan_model(
            condition=condition, manifest=read_dataset_json(S.DATASET, condition),
            base_manifest=read_dataset_json(S.DATASET), seed=seed, size_step=0, config=tiny, result=result,
            determinism={"torch_manual_seed": seed}, run_info={"versions": {"torch": torch.__version__},
                                                               "device": "cpu"},
            old_select=checked_split(condition, "old_select"))
        assert entry["role"] == "base"
        write_index(S.DATASET, [entry], condition)
    return tmp_path
