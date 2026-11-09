"""Train, export and parity-check the models of one dataset (brief items 4 and 5).

Groups:
    dev   role "dev", seeds 100 and 101 by default, trained on the train split (Colored-MNIST: the final biased
          split, so this group runs only after the shortcut check is done).
    main  role "base" on the train split, seeds 0 and 1 by default (development rule of 2026-11-03); seeds 2, 3, 4
          are confirmation seeds added later with --seeds 2 3 4. Colored-MNIST adds role "ceiling" on train_uniform
          for the same seeds. Its pre-registered shortcut check is decided once, on the main seeds of the first main
          run (one rebuild at rho = 0.999 and base retraining if triggered); later main runs only add models.

Main and dev seeds are disjoint. Each model is exported to $WEIGHTCODE_PRIVATE/<dataset>/models/<model_id>/ and appended
to models/index.json. A (role, seed) pair that is already in the index is refused.

    fewshot  PREREG E2 students: role "dev", seeds 100 and 101 by default, trained with the dataset's recipe on the
             train-split rows of --train-classes only (default 0-4), with every output kept. --arch picks the
             architecture (default the dataset's; cnn4 is the second CIFAR architecture). Exported to
             $WEIGHTCODE_PRIVATE/<dataset>/fewshot/models/ with their own index.json, so the dataset index and its
             builders never see them. Main and confirmation seeds are refused. The V_select monitor of the epoch log
             reads the rows of the train classes.

Deployment conditions (--condition, weightcode.data.conditions):
    ceiling    role "ceiling" on the condition's corrupted train split, seeds 0 and 1 by default, exported to
               $WEIGHTCODE_PRIVATE/<dataset>/conditions/<condition>/models/ (reference points, never edited). Runs on NSCC.
    base_eval  V_select accuracy of the clean base (0, 1) and dev (100, 101) models on the condition's V_select,
               written to conditions/<condition>/models/base_eval.json. Local, seconds.
    import     check and copy models exported by an NSCC run (--run-dir <fetched run>) into the store.
    report     as for a dataset.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/train_base.py --dataset fashion_mnist --group dev
    .venv/bin/python scripts/train_base.py --dataset fashion_mnist --group main              # seeds 0 1
    .venv/bin/python scripts/train_base.py --dataset fashion_mnist --group main --seeds 2 3 4
    .venv/bin/python scripts/train_base.py --dataset fashion_mnist --group report
    .venv/bin/python scripts/train_base.py --dataset cifar10 --group fewshot --arch cnn4      # E2, seeds 100 101
    .venv/bin/python scripts/train_base.py --dataset cifar10 --condition contrast_s3 --group base_eval
    .venv/bin/python scripts/train_base.py --dataset cifar10 --condition contrast_s3 --group ceiling   # NSCC job
    .venv/bin/python scripts/train_base.py --dataset cifar10 --condition contrast_s3 --group import --run-dir DIR
"""

from __future__ import annotations

import os

if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.data import build as B
from weightcode.data import colored
from weightcode.data.conditions import CONDITIONS
from weightcode.data.registry import DATASETS
from weightcode.data.sources import download_and_load
from weightcode.data.store import (
    dataset_dir,
    dataset_json_path,
    load_split,
    models_dir,
    read_dataset_json,
    split_sha256,
    write_json,
)
from weightcode.models.arch import build_model
from weightcode.models.export import (
    export_model,
    fewshot_models_dir,
    read_fewshot_index,
    read_index,
    top_confusions,
    write_fewshot_index,
    write_index,
)
from weightcode.models.importer import evaluate_on_condition, import_models
from weightcode.models.runinfo import run_info
from weightcode.models.train import (
    CONFIRMATION_SEEDS,
    DEFAULT_MAIN_SEEDS,
    DEV_SEEDS,
    TRAIN_CONFIGS,
    configure_determinism,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class Trainer:
    """Holds the manifest, cached device tensors and provenance for one dataset or one condition of it."""

    def __init__(
        self, dataset: str, device: str, condition: str | None = None, git_state_file: Path | None = None
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        self.dataset = dataset
        self.condition = condition
        self.device = torch.device(device)
        self.config = TRAIN_CONFIGS[dataset]
        self.manifest = read_dataset_json(dataset, condition)
        self.vselect = load_split(dataset, "V_select", condition)
        if split_sha256(self.vselect) != self.manifest["splits"]["V_select"]["sha256"]:
            raise RuntimeError("V_select.npz does not match dataset.json")
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
        self.run = run_info(REPO_ROOT, git_state_file)
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        self.run["device"] = f"{device_name} ({self.device})"
        self.run["torch_threads"] = torch.get_num_threads()

    def reload_manifest(self) -> None:
        self.manifest = read_dataset_json(self.dataset, self.condition)
        self._cache.clear()

    def _train_tensors(self, split: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        if split not in self._cache:
            self._cache.clear()
            arrays = load_split(self.dataset, split, self.condition)
            recorded = self.manifest["splits"][split]["sha256"]
            if split_sha256(arrays) != recorded:
                raise RuntimeError(f"{split}.npz does not match dataset.json")
            x = torch.from_numpy(arrays["x"]).to(self.device)
            y = torch.from_numpy(arrays["y"]).to(self.device)
            self._cache[split] = (x, y, recorded)
        return self._cache[split]

    def train_one(self, role: str, seed: int, split: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Train, export and index one model."""
        index = read_index(self.dataset, self.condition)
        if any(e["role"] == role and e["seed"] == seed for e in index):
            raise RuntimeError(f"{self.dataset} {self.condition or ''} {role} seed {seed} is already in index.json")
        x_train, y_train, split_sha = self._train_tensors(split)
        determinism = configure_determinism(seed)
        m = self.manifest
        model = build_model(self.config.arch, m["image_shape"], len(m["class_names"]), m["norm"]["mean"], m["norm"]["std"])
        model = model.to(self.device)
        x_mon = torch.from_numpy(self.vselect["x"]).to(self.device)
        y_mon = torch.from_numpy(self.vselect["y"]).to(self.device)
        tag = f"{self.dataset}{'/' + self.condition if self.condition else ''} {role} seed {seed}"

        def log(entry: dict[str, float]) -> None:
            print(
                f"  [{tag}] epoch {entry['epoch']:2d} loss {entry['train_loss']:.4f} "
                f"V_select(monitor) {entry['vselect_acc_monitor']:.4f} ({entry['epoch_time_s']:.1f}s)",
                flush=True,
            )

        result = train_model(model, self.config, seed, x_train, y_train, x_mon, y_mon, log_fn=log)
        entry = export_model(
            dataset=self.dataset,
            manifest=m,
            role=role,
            seed=seed,
            config=self.config,
            train_split=split,
            train_split_sha256=split_sha,
            result=result,
            determinism=determinism,
            run_info=self.run,
            vselect=self.vselect,
            extra_private=extra,
            condition=self.condition,
        )
        index = read_index(self.dataset, self.condition)
        index.append(entry)
        write_index(self.dataset, index, self.condition)
        private = json.loads((models_dir(self.dataset, self.condition) / entry["model_id"] / "private.json").read_text())
        print(
            f"[{tag}] model {entry['model_id']} V_select acc {entry['vselect_acc']:.4f} "
            f"parity {private['parity']['max_abs_diff']:.2e} wall {private['wall_time_s']:.1f}s",
            flush=True,
        )
        return entry

    def train_fewshot(self, arch: str, seed: int, train_classes: tuple[int, ...]) -> dict[str, Any]:
        """Train one PREREG E2 student on the train-split rows of ``train_classes``; export it to the fewshot store.

        The recipe is the dataset's (``TRAIN_CONFIGS``) with ``arch`` in place of its architecture; the model keeps
        every output. Returns the fewshot index row.
        """
        if self.condition is not None:
            raise ValueError("fewshot students train on the clean data; this trainer holds a condition")
        n_classes = len(self.manifest["class_names"])
        classes = tuple(int(c) for c in train_classes)
        if not classes or list(classes) != sorted(set(classes)) or classes[0] < 0 or classes[-1] >= n_classes:
            raise ValueError(f"train classes must be sorted distinct labels in 0..{n_classes - 1}, got {classes}")
        if len(classes) == n_classes:
            raise ValueError("a fewshot student leaves at least one class out")
        index = read_fewshot_index(self.dataset)
        if any((e["arch"], tuple(e["train_classes"]), e["seed"]) == (arch, classes, seed) for e in index):
            raise RuntimeError(f"{self.dataset} fewshot {arch} classes {list(classes)} seed {seed} is already indexed")
        x_all, y_all, split_sha = self._train_tensors("train")
        keep = torch.isin(y_all, torch.as_tensor(classes, device=y_all.device))
        x_train, y_train = x_all[keep], y_all[keep]
        config = dataclasses.replace(self.config, arch=arch)
        determinism = configure_determinism(seed)
        m = self.manifest
        model = build_model(arch, m["image_shape"], n_classes, m["norm"]["mean"], m["norm"]["std"]).to(self.device)
        monitor = np.isin(self.vselect["y"], classes)
        x_mon = torch.from_numpy(self.vselect["x"][monitor]).to(self.device)
        y_mon = torch.from_numpy(self.vselect["y"][monitor]).to(self.device)
        tag = f"{self.dataset} fewshot {arch} classes {classes[0]}-{classes[-1]} seed {seed}"

        def log(entry: dict[str, float]) -> None:
            print(
                f"  [{tag}] epoch {entry['epoch']:2d} loss {entry['train_loss']:.4f} "
                f"V_select train classes (monitor) {entry['vselect_acc_monitor']:.4f} ({entry['epoch_time_s']:.1f}s)",
                flush=True,
            )

        result = train_model(model, config, seed, x_train, y_train, x_mon, y_mon, log_fn=log)
        extra = {
            "family": "fewshot",
            "train_classes": list(classes),
            "new_classes": [c for c in range(n_classes) if c not in classes],
            "n_train_rows": int(x_train.shape[0]),
            "train_rows_rule": "train-split rows whose label is in train_classes, in stored order",
            "monitor": "V_select rows of the train classes (logged only)",
        }
        entry = export_model(
            dataset=self.dataset,
            manifest=m,
            role="dev",
            seed=seed,
            config=config,
            train_split="train",
            train_split_sha256=split_sha,
            result=result,
            determinism=determinism,
            run_info=self.run,
            vselect=self.vselect,
            extra_private=extra,
            models_root=fewshot_models_dir(self.dataset),
        )
        private = json.loads((fewshot_models_dir(self.dataset) / entry["model_id"] / "private.json").read_text())
        confusion = np.asarray(private["vselect_confusion"])
        acc_train_classes = float(np.trace(confusion[np.ix_(classes, classes)]) / confusion[list(classes)].sum())
        row = {**entry, "train_classes": list(classes), "vselect_acc_train_classes": acc_train_classes}
        write_fewshot_index(self.dataset, [*read_fewshot_index(self.dataset), row])
        print(
            f"[{tag}] model {entry['model_id']} V_select acc {entry['vselect_acc']:.4f} (all rows), "
            f"{acc_train_classes:.4f} (train classes) parity {private['parity']['max_abs_diff']:.2e} "
            f"wall {private['wall_time_s']:.1f}s",
            flush=True,
        )
        return row


def parse_classes(text: str) -> tuple[int, ...]:
    """``"0-4"`` or ``"0,1,2,3,4"`` as a sorted tuple of distinct labels."""
    labels: list[int] = []
    for part in text.split(","):
        low, dash, high = part.strip().partition("-")
        labels += list(range(int(low), int(high) + 1)) if dash else [int(low)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"repeated labels in {text!r}")
    return tuple(sorted(labels))


def run_fewshot(args: argparse.Namespace) -> int:
    """--group fewshot: the PREREG E2 students of one dataset and architecture."""
    seeds = tuple(args.seeds) if args.seeds else DEV_SEEDS
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError(f"seeds must be distinct non-negative integers, got {seeds}")
    reserved = set(DEFAULT_MAIN_SEEDS) | set(CONFIRMATION_SEEDS)
    if set(seeds) & reserved:
        raise ValueError(f"seeds {sorted(set(seeds) & reserved)} are main seeds; fewshot students are dev models")
    arch = args.arch or TRAIN_CONFIGS[args.dataset].arch
    classes = parse_classes(args.train_classes)
    trainer = Trainer(args.dataset, args.device, git_state_file=args.git_state_file)
    for seed in seeds:
        trainer.train_fewshot(arch, seed, classes)
    return 0


def colored_extra(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    if role == "ceiling":
        return {"train_color_distribution": "uniform (deployment distribution)"}
    return {"train_color_distribution": "biased", "rho": manifest["colored"]["rho"]}


def run_shortcut_check(trainer: Trainer, seeds: tuple[int, ...]) -> None:
    """Pre-registered Colored-MNIST check on V_select over main ``seeds``, with at most one rebuild at rho 0.999."""
    dataset = trainer.dataset
    while True:
        manifest = read_dataset_json(dataset)
        check = manifest["shortcut_check"]
        index = read_index(dataset)
        base = {e["seed"]: e for e in index if e["role"] == "base" and e["seed"] in seeds}
        ceiling = {e["seed"]: e for e in index if e["role"] == "ceiling" and e["seed"] in seeds}
        if set(base) != set(seeds) or set(ceiling) != set(seeds):
            raise RuntimeError(f"shortcut check needs base and ceiling models for seeds {seeds}")
        decision = colored.shortcut_decision(
            [base[s]["vselect_acc"] for s in seeds], [ceiling[s]["vselect_acc"] for s in seeds]
        )
        attempt = {
            "attempt": len(check["attempts"]) + 1,
            "seeds": list(seeds),
            "rho": manifest["colored"]["rho"],
            "train_sha256": manifest["splits"]["train"]["sha256"],
            "base_vselect_acc": {str(s): base[s]["vselect_acc"] for s in seeds},
            "base_model_ids": {str(s): base[s]["model_id"] for s in seeds},
            "ceiling_vselect_acc": {str(s): ceiling[s]["vselect_acc"] for s in seeds},
            "ceiling_model_ids": {str(s): ceiling[s]["model_id"] for s in seeds},
            **decision,
        }
        check["attempts"].append(attempt)
        print(f"shortcut check attempt {attempt['attempt']}: {json.dumps(decision)}", flush=True)
        if not decision["triggered"] or attempt["attempt"] >= 2:
            check["status"] = "done"
            check["final_rho"] = manifest["colored"]["rho"]
            if decision["triggered"]:
                check["note"] = "gap still within the threshold after the single pre-registered rebuild; no further change"
            write_json(dataset_json_path(dataset), manifest)
            return
        tag = f"attempt1_rho{manifest['colored']['rho']}"
        superseded = dataset_dir(dataset) / "superseded" / tag / "models"
        superseded.mkdir(parents=True)
        for s in seeds:
            shutil.move(str(dataset_dir(dataset) / "models" / base[s]["model_id"]), str(superseded / base[s]["model_id"]))
        write_index(dataset, [e for e in index if not (e["role"] == "base" and e["seed"] in seeds)])
        rebuild = B.rebuild_colored_train(download_and_load("MNIST"), colored.RHO_FALLBACK, tag)
        attempt["rebuild"] = rebuild
        manifest["colored"]["rho"] = colored.RHO_FALLBACK
        manifest["colored"]["train_conflict_fraction"] = rebuild["new_conflict_fraction"]
        manifest["splits"]["train"]["sha256"] = rebuild["new_sha256"]
        manifest["norm"]["note"] = "computed on the attempt-1 train split (rho 0.99) and kept unchanged by the rule"
        check["status"] = "rebuilt"
        write_json(dataset_json_path(dataset), manifest)
        trainer.reload_manifest()
        for s in seeds:
            trainer.train_one("base", s, "train", colored_extra(trainer.manifest, "base"))


def report_condition(dataset: str, condition: str) -> None:
    """Print V_select accuracy of the condition's ceilings and of the clean models on the condition's V_select."""
    for e in read_index(dataset, condition):
        private = json.loads((models_dir(dataset, condition) / e["model_id"] / "private.json").read_text())
        print(f"{dataset}/{condition} {e['role']:7s} seed {e['seed']:3d} {e['model_id']} V_select {e['vselect_acc']:.4f} wall {private['wall_time_s']:.1f}s")
    path = models_dir(dataset, condition) / "base_eval.json"
    if path.exists():
        for row in json.loads(path.read_text())["models"]:
            print(
                f"{dataset}/{condition} {row['role']:7s} seed {row['seed']:3d} {row['model_id']} V_select "
                f"{row['vselect_acc']:.4f} (clean V_select {row['clean_vselect_acc']:.4f}; clean model)"
            )


def report(dataset: str) -> None:
    """Print per-model V_select accuracy and the top-5 V_select confusions of base seed 0."""
    index = read_index(dataset)
    for e in index:
        private = json.loads((dataset_dir(dataset) / "models" / e["model_id"] / "private.json").read_text())
        print(f"{dataset} {e['role']:7s} seed {e['seed']:3d} {e['model_id']} V_select {e['vselect_acc']:.4f} wall {private['wall_time_s']:.1f}s")
        if e["role"] == "base" and e["seed"] == 0:
            names = read_dataset_json(dataset)["class_names"]
            pairs = top_confusions(np.asarray(private["vselect_confusion"]), 5)
            print("  top-5 V_select confusions (true -> predicted: count): " + "; ".join(f"{names[a]} -> {names[b]}: {c}" for a, b, c in pairs))


def run_condition(args: argparse.Namespace) -> int:
    """The --condition groups: ceiling, base_eval, import, report."""
    dataset, condition = args.dataset, args.condition
    if condition not in CONDITIONS or CONDITIONS[condition].dataset != dataset:
        raise SystemExit(f"{condition!r} is not a condition of {dataset}")
    if args.group == "report":
        report_condition(dataset, condition)
        return 0
    if args.group == "import":
        if args.run_dir is None:
            raise SystemExit("--group import needs --run-dir <fetched run directory>")
        for check in import_models(dataset, condition, Path(args.run_dir), args.note):
            print(json.dumps(check), flush=True)
        report_condition(dataset, condition)
        return 0
    if args.group == "base_eval":
        if args.seeds:
            raise SystemExit("base_eval reads the clean main seeds 0 1 and dev seeds 100 101; --seeds is not used")
        selection = {"base": DEFAULT_MAIN_SEEDS, "dev": DEV_SEEDS}
        evaluate_on_condition(dataset, condition, selection, run_info(REPO_ROOT, args.git_state_file))
        report_condition(dataset, condition)
        return 0
    if args.group != "ceiling":
        raise SystemExit(f"--group {args.group} is not defined for a condition (ceiling, base_eval, import, report)")
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_MAIN_SEEDS
    if len(set(seeds)) != len(seeds) or min(seeds) < 0 or set(seeds) & set(DEV_SEEDS):
        raise ValueError(f"ceiling seeds must be distinct non-negative main seeds, got {seeds}")
    trainer = Trainer(dataset, args.device, condition, args.git_state_file)
    extra = {
        "condition_note": "ceiling reference: the same recipe trained on the condition's corrupted train split; "
        "never edited"
    }
    for seed in seeds:
        trainer.train_one("ceiling", seed, "train", extra)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--condition", help=f"a deployment condition ({sorted(CONDITIONS)})")
    parser.add_argument(
        "--group", choices=["main", "dev", "report", "ceiling", "base_eval", "import", "fewshot"], required=True
    )
    parser.add_argument("--arch", help="--group fewshot: architecture (default: the dataset's)")
    parser.add_argument("--train-classes", default="0-4", help="--group fewshot: labels to train on (default 0-4)")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help=f"seeds to train (default: main {list(DEFAULT_MAIN_SEEDS)}, dev {list(DEV_SEEDS)}; "
        f"confirmation main seeds {list(CONFIRMATION_SEEDS)} only on request)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, help="torch intra-op CPU threads (recorded); default: torch's own")
    parser.add_argument("--git-state-file", type=Path, help="git state recorded at submission (code copy without .git)")
    parser.add_argument("--run-dir", help="--group import: directory containing an exported training run")
    parser.add_argument("--note", help="--group import: a note stored in each imported model's import block")
    args = parser.parse_args()
    if args.threads is not None:
        if args.threads < 1:
            raise SystemExit("--threads must be positive")
        torch.set_num_threads(args.threads)
    if args.condition is not None:
        if args.group == "fewshot":
            raise SystemExit("--group fewshot trains on the clean data; drop --condition")
        return run_condition(args)
    if args.group == "fewshot":
        return run_fewshot(args)
    if args.arch is not None:
        raise SystemExit("--arch is for --group fewshot; the other groups use the dataset's architecture")
    if args.group in ("ceiling", "base_eval", "import"):
        raise SystemExit(f"--group {args.group} needs --condition")
    if args.group == "report":
        report(args.dataset)
        return 0
    seeds = tuple(args.seeds) if args.seeds else (DEFAULT_MAIN_SEEDS if args.group == "main" else DEV_SEEDS)
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError(f"seeds must be distinct non-negative integers, got {seeds}")
    reserved = set(DEV_SEEDS) if args.group == "main" else set(DEFAULT_MAIN_SEEDS) | set(CONFIRMATION_SEEDS)
    other_roles = {"dev"} if args.group == "main" else {"base", "ceiling"}
    used = {e["seed"] for e in read_index(args.dataset) if e["role"] in other_roles} | reserved
    if set(seeds) & used:
        raise ValueError(f"seeds {sorted(set(seeds) & used)} belong to the other group; main and dev seeds are disjoint")
    trainer = Trainer(args.dataset, args.device, git_state_file=args.git_state_file)
    is_colored = args.dataset == "colored_mnist"
    if args.group == "dev":
        if is_colored and trainer.manifest["shortcut_check"]["status"] != "done":
            raise RuntimeError("Colored-MNIST dev models wait until the shortcut check is done")
        for seed in seeds:
            extra = colored_extra(trainer.manifest, "dev") if is_colored else None
            trainer.train_one("dev", seed, "train", extra)
        return 0
    status = trainer.manifest["shortcut_check"]["status"] if is_colored else None
    if status not in (None, "pending", "done"):
        raise RuntimeError(f"Colored-MNIST shortcut check is in state {status!r}; resolve it before training more")
    for seed in seeds:
        trainer.train_one("base", seed, "train", colored_extra(trainer.manifest, "base") if is_colored else None)
    if is_colored:
        for seed in seeds:
            trainer.train_one("ceiling", seed, "train_uniform", colored_extra(trainer.manifest, "ceiling"))
        if status == "pending":
            run_shortcut_check(trainer, seeds)
        else:
            print("shortcut check was decided earlier (dataset.json); these seeds only add models", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
