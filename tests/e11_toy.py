"""A toy E11 store for the harness tests: tiny synthetic ViT-Lite students under the E11 dataset name.

The store is ``weightcode.models.vit_synth.build_store`` with the ``TINY`` configuration (3 x 8 x 8 images, 4 x 4 patches,
width 16, 2 layers, 2 heads, 4 classes; the authors' tensor names and layer, 30 tensors): random images, a stand-in
transform under each E10 condition name, dev seeds 100 to 105 with E10's decoy pairs, one evidence-v1 instance per
(condition, seed) and the replay draw. Its labels are planted so that the LayerNorm edit ``vit_synth.plant_deltas`` of
the seed-100 student fixes gaussian_noise_s5 rows and breaks no clean row. Everything lives under a pytest tmp
directory with WEIGHTCODE_* monkeypatched; nothing touches the real store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weightcode.harness import e10, e11
from weightcode.models import vit_synth as V

CLASSES = ("alpha", "beta", "gamma", "delta")
DEV_SEEDS = (100, 101, 102, 103, 104, 105)
DECOYS = {100: 101, 101: 100, 102: 103, 103: 102, 104: 105, 105: 104}
PLANT_CONDITION = "gaussian_noise_s5"
CONDITION_CORRUPTIONS = dict(zip(e10.STREAM_CONDITIONS, e10.STREAM_CORRUPTIONS))
ROWS = {"train": 96, "V_select": 160, "V_evidence": 128, "V_check": 160, "test": 32}


def activate(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point WEIGHTCODE_PRIVATE, WEIGHTCODE_SEALED and WEIGHTCODE_RUNS at ``root``."""
    for variable, sub in (("WEIGHTCODE_PRIVATE", "private"), ("WEIGHTCODE_SEALED", "sealed"), ("WEIGHTCODE_RUNS", "runs")):
        monkeypatch.setenv(variable, str(root / sub))


def build_e11_store(root: Path, monkeypatch: pytest.MonkeyPatch, *,
                    instance_conditions: tuple[str, ...] | None = None) -> V.SyntheticStore:
    """The activated toy store (module docstring); ``instance_conditions`` limits which conditions get instances."""
    activate(root, monkeypatch)
    return V.build_store(root, V.TINY, dataset=e11.DATASET, conditions=CONDITION_CORRUPTIONS,
                         plant_condition=PLANT_CONDITION, class_names=CLASSES, seeds=DEV_SEEDS, decoys=DECOYS,
                         rows=ROWS, instance_conditions=instance_conditions)


@pytest.fixture
def e11_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> V.SyntheticStore:
    """Under a neutral directory name, as ``harness_store`` (the sandbox self-test sees the session's host paths)."""
    return build_e11_store(tmp_path_factory.mktemp("hs") / "store", monkeypatch)
