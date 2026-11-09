"""Install, split and store ogbn-arxiv (``weightcode.data.arxiv``), with the GCN students' numpy forward.py.

Steps: ``--install ZIP`` copies a downloaded official arxiv.zip into $WEIGHTCODE_PRIVATE/raw/ogbn_arxiv/ (read-only; its
sha256 must be ``ARXIV_ZIP_SHA256``) and records its provenance. The build then parses the zip and writes both
protocol datasets (``weightcode.data.arxiv.PROTOCOLS``), each with permuted node and class ids, graph.npz,
permutation.npz, the splits (train, old_select, old_check, old_replay, V_select, V_evidence, V_check,
valid_unused), forward.py (``templates/forward_gcn.py.tmpl``, rendered for its graph's node count) and
dataset.json: ``ogbn_arxiv`` (transductive, every node; the official test nodes go to
$WEIGHTCODE_SEALED/ogbn_arxiv/test.npz only) and ``ogbn_arxiv_ind`` (inductive: graph.npz is the snapshot up to 2018,
graph_train.npz the snapshot up to 2017). ``--verify`` re-derives everything from the zip and compares hashes (the
sealed file is not read).

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/prepare_arxiv.py --install DOWNLOADED_ZIP --fetch-note "curl ... on DATE"
    .venv/bin/python scripts/prepare_arxiv.py
    .venv/bin/python scripts/prepare_arxiv.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightcode.data import arxiv as A
from weightcode.data.store import forward_path
from weightcode.models.forward_gen import render_gcn_forward, write_forward
from weightcode.models.gcn import GCNConfig
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[1]


def forward_source(n_nodes: int) -> str:
    """forward.py of the arxiv GCN students on a graph of ``n_nodes`` (BatchNorm eps of ``GCNConfig``)."""
    config = GCNConfig(n_nodes, A.N_FEATURES, A.N_CLASSES)
    return render_gcn_forward(n_nodes, A.N_FEATURES, A.N_CLASSES, config.bn_eps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install", type=Path, help="downloaded official arxiv.zip to install into the raw store")
    parser.add_argument("--fetch-note", help="how the zip was fetched (required with --install)")
    parser.add_argument("--verify", action="store_true", help="re-derive from the raw zip and compare hashes")
    args = parser.parse_args(argv)
    if args.install is not None:
        if not args.fetch_note:
            raise SystemExit("--install needs --fetch-note")
        print(json.dumps(A.install_raw_zip(args.install, args.fetch_note)["files"], indent=2))
        return 0
    raw = A.load_raw()
    builds = (A.build(raw), A.build_inductive(raw))
    if args.verify:
        ok = True
        for built in builds:
            report = A.verify(built)
            n_nodes = built.manifest["graph"]["n_nodes"]
            source_ok = forward_path(built.dataset).read_text(encoding="utf-8") == forward_source(n_nodes)
            ok = ok and source_ok and all(value["match"] for value in report.values())
            print(json.dumps({"dataset": built.dataset, **{key: value["match"] for key, value in report.items()},
                              "forward": source_ok}, indent=2))
        return 0 if ok else 1
    from weightcode.data.sources import provenance_path
    from weightcode.data.store import read_json

    provenance = read_json(provenance_path())["sources"][A.SOURCE]
    info = run_info(REPO_ROOT)
    for built in builds:
        forward_sha = write_forward(forward_path(built.dataset), forward_source(built.manifest["graph"]["n_nodes"]))
        manifest = A.write(built, forward_sha, provenance, info)
        graphs = {key: {k: manifest[key][k] for k in ("sha256", "n_nodes", "nnz", "n_undirected_edges")}
                  for key in ("graph", "train_graph") if key in manifest}
        print(json.dumps({"dataset": built.dataset, "split_sizes": manifest["split_sizes"], **graphs,
                          "test_n": manifest["test"]["n"], "forward_sha256": forward_sha}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
