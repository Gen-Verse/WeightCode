"""Train, export, import and report the ogbn-arxiv GCN students (``weightcode.models.gcn``, ``weightcode.models.gcn_export``).

Groups:

``--protocol`` picks the store dataset (``weightcode.data.arxiv.PROTOCOLS``): ``transductive`` (ogbn_arxiv) or
``inductive`` (ogbn_arxiv_ind).

* ``train``: for each seed, the OGB reference GCN under the fixed recipe (``GCNTrainConfig``), as ``--role base``
  (labels of the train split), ``--role ceiling`` (train plus valid_unused) or ``--role newclass`` (PREREG E7: the
  train nodes of the train classes of ``graphnew_classes.json``, ``weightcode.data.arxiv_classes``; inductive only), on
  the role's training graph (``gcn_export.training_graph``). The job needs dataset.json, forward.py, graph.npz
  (graph_train.npz too for an inductive base or newclass student; graphnew_classes.json for newclass) and the
  training and readout splits (train, V_select, old_select; valid_unused for a ceiling). Each student is exported
  with parity at every node of graph.npz and its V_select and old_select accuracy; the index (the fewshot index for
  newclass) is rewritten after every student.
* ``import``: the students of a fetched NSCC run (``import_gcn_models``; ``--role newclass`` imports the run's
  fewshot students).
* ``report``: the local index (and the fewshot index) with each student's accuracies, logit scale and per-layer
  weight norms.

``--smoke-epochs N`` trains N epochs into ``--models-root`` (required with it), marks every model smoke (the
importer refuses them) and never writes the store's index.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/train_arxiv.py --protocol inductive --group train --role base --seeds 100 --device cuda
    .venv/bin/python scripts/train_arxiv.py --protocol inductive --group import --run-dir $WEIGHTCODE_PRIVATE/nscc_runs/RUN --note "..."
    .venv/bin/python scripts/train_arxiv.py --protocol inductive --group report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from weightcode.data import arxiv as A
from weightcode.data.arxiv_classes import class_draw_block
from weightcode.data.store import read_dataset_json, write_json
from weightcode.models import gcn as G
from weightcode.models.export import read_fewshot_index, read_index, write_fewshot_index, write_index
from weightcode.models.gcn_export import (
    CLASS_SUBSET_ROLES,
    export_gcn_model,
    import_gcn_models,
    training_graph,
    training_nodes,
)
from weightcode.models.runinfo import run_info
from weightcode.models.train import configure_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]


def train_group(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    if args.smoke_epochs is not None and args.models_root is None:
        raise SystemExit("--smoke-epochs needs --models-root (smoke models never enter the store)")
    dataset = A.PROTOCOLS[args.protocol]
    manifest = read_dataset_json(dataset)
    if manifest.get("protocol") != args.protocol:
        raise SystemExit(f"{dataset}/dataset.json is not the {args.protocol} protocol")
    subset = args.role in CLASS_SUBSET_ROLES
    if subset and args.protocol != "inductive":
        raise SystemExit("--role newclass is defined on the inductive protocol only (PREREG E7)")
    draw = class_draw_block(dataset) if subset else None
    train_classes = draw["train_classes"] if draw else None
    graph, nodes, graph_block = training_graph(dataset, args.role, manifest)
    x, y, used = training_nodes(dataset, args.role, manifest, nodes, train_classes)
    info = run_info(REPO_ROOT, args.git_state_file)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    info["device"] = f"{name} ({device})"
    config = G.GCNTrainConfig()
    if args.smoke_epochs is not None:
        config = dataclasses.replace(config, epochs=args.smoke_epochs, log_every=max(1, args.smoke_epochs // 2))
    if args.models_root is not None:
        index_file = Path(args.models_root) / "index.json"
        previous = json.loads(index_file.read_text()) if index_file.exists() else []
    else:
        previous = read_fewshot_index(dataset) if subset else read_index(dataset)
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        determinism = configure_determinism(seed)
        result = G.train_gcn(config, seed, graph, x, y, A.N_CLASSES, device)
        extra: dict[str, Any] = {"smoke": True} if args.smoke_epochs is not None else {}
        if draw is not None:
            extra["class_draw"] = draw
        entry = export_gcn_model(dataset=dataset, manifest=manifest, role=args.role, seed=seed, config=config,
                                 result=result, training_splits=used, train_graph=graph_block,
                                 determinism=determinism, run_info=info, models_root=args.models_root,
                                 extra_private=extra, train_classes=train_classes)
        entries.append(entry)
        if args.models_root is not None:
            write_json(Path(args.models_root) / "index.json", previous + entries)
        elif subset:
            write_fewshot_index(dataset, previous + entries)
        else:
            write_index(dataset, previous + entries)
        print(json.dumps({"protocol": args.protocol, "seed": seed, "role": args.role, "model_id": entry["model_id"],
                          "vselect_acc": entry["vselect_acc"], "final_train_acc": result.log[-1]["train_acc"],
                          "final_loss": result.log[-1]["loss"], "wall_time_s": round(result.wall_time_s, 1)}),
              flush=True)
    return 0


def import_group(args: argparse.Namespace) -> int:
    fewshot = args.role in CLASS_SUBSET_ROLES
    print(json.dumps(import_gcn_models(A.PROTOCOLS[args.protocol], args.run_dir, args.note, fewshot=fewshot), indent=2))
    return 0


def report_group(args: argparse.Namespace) -> int:
    from weightcode.data.store import models_dir, read_json
    from weightcode.models.export import fewshot_models_dir

    dataset = A.PROTOCOLS[args.protocol]
    rows = []
    listed = [(models_dir(dataset), entry) for entry in read_index(dataset)]
    listed += [(fewshot_models_dir(dataset), entry) for entry in read_fewshot_index(dataset)]
    for root, entry in listed:
        private = read_json(root / entry["model_id"] / "private.json")
        rows.append({**entry, "old_select_acc": private["final_old_select_acc"],
                     "logit_scale": private["parity"]["logit_scale"], "layer_norms": private["layer_norms"],
                     "train_splits": sorted(private["train_data"]["splits"]),
                     "train_graph": private["train_data"]["graph"]["path"]})
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=tuple(A.PROTOCOLS), required=True)
    parser.add_argument("--group", choices=("train", "import", "report"), required=True)
    parser.add_argument("--role", choices=("base", "ceiling", "newclass"), default="base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--git-state-file", type=Path, help="git_state.json of a job's code copy (run_info)")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--note")
    args = parser.parse_args()
    if args.group == "import" and args.run_dir is None:
        raise SystemExit("--group import needs --run-dir")
    return {"train": train_group, "import": import_group, "report": report_group}[args.group](args)


if __name__ == "__main__":
    sys.exit(main())
