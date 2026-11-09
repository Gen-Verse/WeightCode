"""Clean companions and replay evidence for the learning experiments (E1: add a capability without replacing one).

A named deployment condition keeps the rows of the clean splits and replaces ``x`` (``weightcode.data.conditions``). The
clean companion of a condition split is the clean split with the same rows in the same order: the same ``y`` and
``idx``, with clean ``x``. Both files are resolved through ``deployment_splits`` and checked against the content
hashes their dataset.json records, and the rows are compared, so a companion cannot pair the wrong images.

Companions serve the joint patch-check (``weightcode.evalkit.patch_check.JointPatchCheckOracle``: fixed - broken on the
condition V_check plus fixed - broken on its clean companion) and the clean readout of ``weightcode.evalkit.devreadout``.
They are never editor-visible. The clean companions of an instance's own evidence rows, in particular, must never
reach an editor: a clean and a corrupted view of the same image identify the corruption exactly, and no deployment
offers such pairs.

Replay evidence (rule ``replay-v1``) is the editor-visible sample of old data. It holds ``REPLAY_N`` rows of the
dataset's clean train split, drawn once per dataset with ``default_rng([SPLIT_SEED, REPLAY_STREAM])`` and shared by
every seed and condition. The train split is disjoint from V_evidence, V_check, V_select and test, so a replay row
can never coincide with an evidence row of any instance, built now or later. A drawn file is never redrawn: a later
call re-derives the draw and refuses any difference.

Usage::

    python -m weightcode.evalkit.companions replay --dataset cifar10     # draw once, or verify the existing draw
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from weightcode.baselines.instance_io import DEFAULT_QUERY_BUDGET, Instance, load_instance
from weightcode.common.identity import params_sha256
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.store import dataset_dir, read_dataset_json, split_path, write_json
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import JointPatchCheckOracle
from weightcode.evalkit.targets import DEPLOY_CONDITION, deployment_splits, load_checked_split

COMPANION_SPLITS: tuple[str, ...] = ("V_evidence", "V_check", "V_select")
REPLAY_RULE = "replay-v1"
REPLAY_SCHEMA = "weightcode.replay.v1"
REPLAY_POOL = "train"
REPLAY_N = 64
REPLAY_STREAM = 9001
"""Generator stream of the replay draw. Streams in use elsewhere: Colored-MNIST colours in ``weightcode.data.colored`` and
the condition streams of ``weightcode.data.conditions`` (base 300 plus split offsets, and the drawn-parameter streams of the
dev-only conditions), all far below 9000."""
REPLAY_ID_HEX_DIGITS = 12


class CompanionError(RuntimeError):
    """A clean companion or replay file was refused: unknown split, no companion, misaligned rows or a changed draw."""


@dataclass(frozen=True)
class CleanRows:
    """Clean rows aligned with one split of a named condition: same order, same labels and dataset indices."""

    dataset: str
    condition: str
    split: str
    x: np.ndarray
    y: np.ndarray
    idx: np.ndarray
    clean_sha256: str
    condition_sha256: str

    @property
    def row_ids(self) -> np.ndarray:
        """Position of each row in the split, identical for the condition split and its companion."""
        return np.arange(len(self.y))

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a readout or an oracle consumes."""
        return {"x": self.x, "y": self.y, "idx": self.idx}


def _split_file(splits: object, split: str) -> tuple[Path, str]:
    attribute = {"V_evidence": "v_evidence", "V_check": "v_check", "V_select": "v_select"}[split]
    return getattr(splits, attribute), getattr(splits, f"{attribute}_sha256")


def clean_companion(dataset: str, condition: str, split: str) -> CleanRows:
    """The clean companion of ``split`` of a named condition, both files hash-checked and their rows compared."""
    if split not in COMPANION_SPLITS:
        raise CompanionError(f"no clean companion for split {split!r}; known: {COMPANION_SPLITS}")
    if condition == DEPLOY_CONDITION:
        raise CompanionError(f"{dataset}: condition {DEPLOY_CONDITION!r} is the clean data itself and has no companion")
    condition_path, condition_sha = _split_file(deployment_splits(dataset, condition), split)
    clean_path, clean_sha = _split_file(deployment_splits(dataset, DEPLOY_CONDITION), split)
    shifted = load_checked_split(condition_path, condition_sha)
    clean = load_checked_split(clean_path, clean_sha)
    for key in ("y", "idx"):
        if key not in shifted or key not in clean:
            raise CompanionError(f"{dataset}/{condition} {split}: both files need {key!r} to align rows")
        if not np.array_equal(shifted[key], clean[key]):
            raise CompanionError(f"{dataset}/{condition} {split}: {key} differs from the clean split; rows do not align")
    if shifted["x"].shape != clean["x"].shape or shifted["x"].dtype != clean["x"].dtype:
        raise CompanionError(
            f"{dataset}/{condition} {split}: x is {shifted['x'].dtype}{shifted['x'].shape}, clean "
            f"{clean['x'].dtype}{clean['x'].shape}"
        )
    return CleanRows(dataset, condition, split, clean["x"], clean["y"], clean["idx"], clean_sha, condition_sha)


def load_clean_rows(instance: Instance | str, split: str) -> CleanRows:
    """Clean companion of one split of an instance's condition (``instance`` is an Instance or an instance id)."""
    resolved = instance if isinstance(instance, Instance) else load_instance(instances_root() / instance)
    return clean_companion(str(resolved.private["dataset"]), str(resolved.private["condition"]), split)


def make_joint_oracle(
    instance: Instance, budget: int = DEFAULT_QUERY_BUDGET, batch_size: int = 1024
) -> JointPatchCheckOracle:
    """Joint counted oracle of an instance: its own V_check (as ``make_oracle`` loads it) plus the clean companion.

    The instance's V_check is checked against the content hash the instance recorded, and its rows must equal the
    companion's rows, so the new and old parts score the same images, corrupted and clean.
    """
    with np.load(Path(instance.private["vcheck_path"]), allow_pickle=False) as archive:
        shifted = {key: archive[key] for key in archive.files}
    recorded = instance.private.get("vcheck_content_sha256")
    if recorded is None or params_sha256(shifted) != recorded:
        raise CompanionError(f"{instance.instance_id}: V_check content hash differs from the instance record")
    clean = load_clean_rows(instance, "V_check")
    if not (np.array_equal(shifted["y"], clean.y) and np.array_equal(shifted["idx"], clean.idx)):
        raise CompanionError(f"{instance.instance_id}: instance V_check rows differ from the clean companion's rows")
    return JointPatchCheckOracle(
        shifted["x"], shifted["y"], clean.x, clean.y, instance.forward, instance.params, budget, batch_size
    )


def replay_paths(dataset: str) -> tuple[Path, Path]:
    """The replay arrays and their record: ``$WEIGHTCODE_PRIVATE/<dataset>/replay/replay_v1.{npz,json}``."""
    folder = dataset_dir(dataset) / "replay"
    return folder / "replay_v1.npz", folder / "replay_v1.json"


def _draw_replay(dataset: str, n: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    meta = read_dataset_json(dataset)
    pool_sha = meta["splits"][REPLAY_POOL]["sha256"]
    pool = load_checked_split(split_path(dataset, REPLAY_POOL), pool_sha)
    n_pool = len(pool["y"])
    if not 0 < n <= n_pool:
        raise CompanionError(f"{dataset}: replay size {n} must be in 1..{n_pool}")
    rng = np.random.default_rng([SPLIT_SEED, REPLAY_STREAM])
    rows = rng.choice(n_pool, size=n, replace=False)
    codes = rng.choice(16**REPLAY_ID_HEX_DIGITS, size=n, replace=False)
    arrays = {
        "x": pool["x"][rows],
        "y": pool["y"][rows].astype(np.int64),
        "idx": pool["idx"][rows],
        "row_ids": rows.astype(np.int64),
        "split_ids": np.array([f"r{int(code):0{REPLAY_ID_HEX_DIGITS}x}" for code in codes]),
    }
    record: dict[str, object] = {
        "schema": REPLAY_SCHEMA,
        "rule": REPLAY_RULE,
        "dataset": dataset,
        "pool_split": REPLAY_POOL,
        "pool_sha256": pool_sha,
        "split_seed": int(SPLIT_SEED),
        "stream": REPLAY_STREAM,
        "n": int(n),
        "row_ids": [int(r) for r in rows],
        "idx": [int(i) for i in arrays["idx"]],
        "class_counts": [int(c) for c in np.bincount(arrays["y"], minlength=len(meta["class_names"]))],
        "content_sha256": params_sha256(arrays),
        "numpy_version": np.__version__,
    }
    return arrays, record


def build_replay(dataset: str, n: int = REPLAY_N) -> dict[str, object]:
    """Draw the replay evidence once and write it; an existing draw is re-derived and must match exactly."""
    arrays, record = _draw_replay(dataset, n)
    npz_path, json_path = replay_paths(dataset)
    if npz_path.exists() or json_path.exists():
        stored = json.loads(json_path.read_text())
        for key in ("rule", "pool_sha256", "stream", "n", "row_ids", "idx", "content_sha256"):
            if stored.get(key) != record[key]:
                raise CompanionError(f"{json_path}: recorded {key} differs from the re-derived draw; never redrawn")
        load_replay(dataset)
        return stored
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    write_json(json_path, record)
    return record


def load_replay(dataset: str) -> dict[str, np.ndarray]:
    """The replay arrays (x, y, idx, row_ids, split_ids), checked against the recorded content and pool hashes."""
    npz_path, json_path = replay_paths(dataset)
    if not npz_path.exists() or not json_path.exists():
        raise CompanionError(f"{dataset}: no replay evidence at {npz_path}; draw it with build_replay")
    record = json.loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if params_sha256(arrays) != record["content_sha256"]:
        raise CompanionError(f"{npz_path}: content hash differs from {json_path.name}")
    current_pool = read_dataset_json(dataset)["splits"][record["pool_split"]]["sha256"]
    if current_pool != record["pool_sha256"]:
        raise CompanionError(f"{dataset}: the {record['pool_split']} split changed since the replay draw")
    return arrays


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``replay --dataset NAME`` draws or verifies a dataset's replay evidence."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="draw the replay evidence once, or verify the existing draw")
    replay.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    record = build_replay(args.dataset)
    npz_path, _ = replay_paths(args.dataset)
    print(f"{args.dataset}: {record['rule']} n={record['n']} from {record['pool_split']} stream {record['stream']} "
          f"class counts {record['class_counts']} content {str(record['content_sha256'])[:12]} -> {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
