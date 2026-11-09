"""Build and store the seqops dataset and its new-skill conditions (PREREG E5; ``weightcode.data.seqops_build``).

Base: $WEIGHTCODE_PRIVATE/seqops/splits/{train,V_select,V_evidence,V_check}.npz, the numpy forward.py, dataset.json and
$WEIGHTCODE_SEALED/seqops/test.npz. A condition new<k>_<kind> binds reserved operator NEW<k> to the rule drawn for it and
writes $WEIGHTCODE_PRIVATE/seqops/conditions/<condition>/{dataset.json,splits/*.npz} and its sealed test. Draw a condition
only after the base students are trained (PREREG E5); nothing in base training reads it.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/prepare_seqops.py                        # base dataset
    .venv/bin/python scripts/prepare_seqops.py --verify
    .venv/bin/python scripts/prepare_seqops.py --condition new0_substitution
    .venv/bin/python scripts/prepare_seqops.py --condition new0_substitution --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightcode.data import seqops_build as B
from weightcode.data.seqops import DATASET
from weightcode.data.store import dataset_json_path, forward_path, read_dataset_json
from weightcode.models.forward_gen import render_seq_forward, write_forward
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[1]


def prepare_base() -> dict:
    if dataset_json_path(DATASET).exists():
        raise FileExistsError(f"{dataset_json_path(DATASET)} exists; run with --verify to check it")
    built = B.build_base()
    constants = B.forward_constants()
    forward_sha = write_forward(forward_path(DATASET), render_seq_forward(**constants))
    manifest = B.write_built(built, {"forward_path": "forward.py", "forward_sha256": forward_sha,
                                     "forward_constants": constants, "created_by": run_info(REPO_ROOT)})
    return {"dataset": DATASET, "split_sizes": manifest["split_sizes"], "test_n": manifest["test"]["n"],
            "held_out_keys": manifest["held_out"]["n_keys"], "forward_sha256": forward_sha}


def prepare_condition(name: str) -> dict:
    base_manifest = read_dataset_json(DATASET)
    built = B.build_condition(name)
    manifest = B.write_built(built, {**B.condition_extra(base_manifest), "created_by": run_info(REPO_ROOT)})
    return {"condition": name, "split_sizes": manifest["split_sizes"], "test_n": manifest["test"]["n"],
            "part_counts": {split: record["part_counts"] for split, record in manifest["splits"].items()},
            "rule_agreement_max": manifest["condition"]["rule_agreement"]["max_overall"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", help="a condition new<k>_<kind>, e.g. new0_substitution")
    parser.add_argument("--verify", action="store_true", help="re-derive every split and compare hashes")
    args = parser.parse_args()
    if args.verify:
        report = B.verify_condition(args.condition) if args.condition else B.verify_base()
        print(json.dumps({key: value["match"] for key, value in report.items()}, indent=2))
        return 0 if all(value["match"] for value in report.values()) else 1
    summary = prepare_condition(args.condition) if args.condition else prepare_base()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
