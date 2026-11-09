"""PREREG E10 store steps in the project environment (``weightcode.data.e10``, ``weightcode.models.e10_export``).

    source scripts/env.sh
    .venv/bin/python scripts/e10/e10_store.py clean
    .venv/bin/python scripts/e10/e10_store.py condition --staged DIR --fidelity FILE --install FILE CORRUPTION ...
    .venv/bin/python scripts/e10/e10_store.py import --run-dir $WEIGHTCODE_PRIVATE/nscc_runs/RUN --note "..."
    .venv/bin/python scripts/e10/e10_store.py headroom --seed 100 --out FILE

``headroom`` (the entry's check before any instance): the student's clean V_select accuracy and its accuracy on each
stored condition's V_select; a condition runs only if clean minus condition is at least ``HEADROOM_POINTS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from weightcode.data import e10 as D
from weightcode.data.store import list_conditions, load_split, read_dataset_json, split_sha256
from weightcode.models import e10_export as E
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADROOM_POINTS = 15.0


def _headroom(seed: int, out: Path) -> dict:
    from weightcode.common.identity import load_params, params_sha256
    from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta, read_model_index

    if out.exists():
        raise SystemExit(f"{out} exists; nothing written")
    meta = read_dataset_meta(D.DATASET)
    forward, forward_sha, _ = load_dataset_forward(D.DATASET, meta)
    (entry,) = [e for e in read_model_index(D.DATASET) if e.seed == seed and e.role == "dev"]
    from weightcode.data.store import models_dir

    params = load_params(models_dir(D.DATASET) / entry.model_id / "model.npz")
    if params_sha256(params) != entry.params_sha256:
        raise SystemExit("model.npz hash differs from the index")

    def accuracy(condition: str | None) -> tuple[float, str]:
        arrays = load_split(D.DATASET, "V_select", condition)
        recorded = read_dataset_json(D.DATASET, condition)["splits"]["V_select"]["sha256"]
        if split_sha256(arrays) != recorded:
            raise SystemExit(f"V_select of {condition or 'clean'} differs from its dataset.json")
        pred = np.asarray(forward(params, arrays["x"])).argmax(1)
        return float(np.mean(pred == arrays["y"])), recorded

    clean, clean_sha = accuracy(None)
    rows = {}
    for condition in list_conditions(D.DATASET):
        acc, sha = accuracy(condition)
        gap = 100.0 * (clean - acc)
        rows[condition] = {"vselect_acc": acc, "vselect_sha256": sha, "gap_points": gap,
                           "runs": gap >= HEADROOM_POINTS}
    record = {"schema": "weightcode.e10.headroom.v1", "rule": f"clean minus condition >= {HEADROOM_POINTS} points on "
              "V_select (PREREG E10, before any instance)", "model_id": entry.model_id, "seed": seed,
              "params_sha256": entry.params_sha256, "forward_sha256": forward_sha,
              "clean_vselect_acc": clean, "clean_vselect_sha256": clean_sha, "conditions": rows}
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"-> {out} {hashlib.sha256(text.encode()).hexdigest()}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="group", required=True)
    sub.add_parser("clean")
    cond = sub.add_parser("condition")
    cond.add_argument("--staged", type=Path, required=True)
    cond.add_argument("--fidelity", type=Path, required=True)
    cond.add_argument("--install", type=Path, required=True)
    cond.add_argument("corruptions", nargs="+")
    imp = sub.add_parser("import")
    imp.add_argument("--run-dir", type=Path, required=True)
    imp.add_argument("--note", required=True)
    head = sub.add_parser("headroom")
    head.add_argument("--seed", type=int, default=100)
    head.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    info = run_info(REPO_ROOT)
    if args.group == "clean":
        manifest = D.build_clean(info)
        print(json.dumps({"dataset": manifest["dataset"], "forward_sha256": manifest["forward_sha256"],
                          "splits": {k: v["sha256"] for k, v in manifest["splits"].items()}}, indent=1))
    elif args.group == "condition":
        for corruption in args.corruptions:
            manifest = D.build_condition(corruption, args.staged, args.staged / "dev_rows.json", args.fidelity,
                                         args.install, info)
            print(json.dumps({"condition": manifest["condition"]["name"],
                              "splits": {k: v["sha256"] for k, v in manifest["splits"].items()}}, indent=1))
    elif args.group == "import":
        print(json.dumps(E.import_e10_models(args.run_dir, args.note), indent=1))
    else:
        _headroom(args.seed, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
