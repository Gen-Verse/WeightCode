"""PREREG E7 instrument: Correct-and-Smooth and label propagation on a graphnew instance, read on V_select.

Not a reference, never gating, not a weight edit (``weightcode.analysis.graphnew_cs``). The seeds are the instance's 144
evidence nodes with their labels, the base model output is the softmax of forward.py's logits at every node, and the
settings are the official C&S code's ogbn-arxiv settings. Each variant's 40-way argmax is read on V_select against the
base (``fewshot.class_split_readout``: all, new-class and old-class rows), with class-balanced new-class accuracy;
a class with no V_select row has a null accuracy. Writes one JSON record and refuses to overwrite it::

    source scripts/env.sh
    .venv/bin/python scripts/dev/e7_cs_instrument.py --instance g06de3fa0 \\
        --out $WEIGHTCODE_RUNS/devreadouts/e7_cs_g06de3fa0_dev_20261104.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", required=True, help="graphnew instance id g<8 hex>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True
    if args.out.exists():
        print(f"{args.out} exists; nothing overwritten", file=sys.stderr)
        return 3
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.analysis import graphnew_cs as C
    from weightcode.common.identity import params_sha256
    from weightcode.evalkit.devreadout import json_safe
    from weightcode.evalkit import graphnew as GN
    from weightcode.evalkit.fewshot import class_split_readout

    started = time.perf_counter()
    instance = GN.load_graphnew_instance(args.instance)
    if instance.private.get("split_role") != "dev":
        print(f"refused: {args.instance} is not a dev instance", file=sys.stderr)
        return 2
    GN.check_graphnew_current(instance.instance_dir)
    private = instance.private
    graph = GN.instance_graph(instance)
    n_nodes = int(private["n_nodes"])
    logits = np.asarray(instance.forward(dict(instance.params), np.arange(n_nodes, dtype=np.int64)), dtype=np.float64)
    evidence = instance.evidence
    seeds = np.concatenate([evidence["x_new"], evidence["x_old"]]).astype(np.int64)
    seed_labels = np.concatenate([evidence["y_new"], evidence["y_old"]]).astype(np.int64)
    outputs = C.instrument_outputs(graph["indptr"], graph["indices"], logits, seeds, seed_labels)
    v_sel = GN._load_recorded(private, "V_select")
    y = v_sel["y"].astype(np.int64)
    base_pred = np.argmax(logits[v_sel["x"]], axis=1)
    (check,) = GN.predict_nodes(instance.forward, instance.params, v_sel["x"])
    if not np.array_equal(check, base_pred):
        raise RuntimeError("base predictions on V_select differ between the two forward calls")
    train_classes = [int(c) for c in private["train_classes"]]
    new_classes = [int(c) for c in private["new_classes"]]
    variants = {}
    for name, output in outputs.items():
        pred = output[torch.from_numpy(v_sel["x"])].numpy().argmax(axis=1)
        readout = class_split_readout(base_pred, pred, y, train_classes, new_classes)
        balanced = {side: float(np.mean([readout["per_class"][str(c)][f"acc_{side}"] for c in new_classes]))
                    for side in ("base", "child")}
        readout["acc_new_balanced_base"], readout["acc_new_balanced_child"] = balanced["base"], balanced["child"]
        readout["n_vselect_zero_mass_rows"] = C.zero_rows(output, v_sel["x"])
        variants[name] = {"setting": C.SETTINGS[name], "readout": json_safe(readout)}
    record = {
        "schema": "weightcode.e7_cs_instrument.v1",
        "prereg": "E7 instrument (label propagation / Correct-and-Smooth on the base softmax); not a reference, "
                  "never gating, not a weight edit",
        "instance_id": instance.instance_id,
        "params_sha256": instance.params_sha256,
        "graph_file_sha256": private["graph_file_sha256"],
        "evidence_content_sha256": private["evidence_content_sha256"],
        "vselect_content_sha256": private["vselect_content_sha256"],
        "seeds": {"n": int(seeds.size), "n_new": int(evidence["x_new"].size), "n_replay": int(evidence["x_old"].size),
                  "rule": "the 144 evidence nodes (LEARN evidence) with their labels; no other label is read"},
        "base_output": "softmax of forward.py logits at every node, float64",
        "source": C.CS_SOURCE,
        "split": "V_select",
        "variants": variants,
        "torch": torch.__version__,
        "wall_s": time.perf_counter() - started,
    }
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(text)
    for name, variant in variants.items():
        r = variant["readout"]
        print(f"{name}: V_select all {100 * r['acc_all_child']:.2f}% ({r['all_points']:+.2f}), new "
              f"{100 * r['acc_new_child']:.2f}% (balanced {100 * r['acc_new_balanced_child']:.2f}%), old "
              f"{100 * r['acc_old_child']:.2f}% ({100 * r['delta_old']:+.2f}); zero-mass rows "
              f"{r['n_vselect_zero_mass_rows']}")
    print(f"params {params_sha256(instance.params)[:12]} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
