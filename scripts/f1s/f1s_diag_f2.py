"""PREREG F1s diagnostic for F2 (decides nothing): F2's fine-tuning recipe on clean rows only.

F2's recipe (AdamW wd 3e-2, label smoothing 0.1, train mode with dropout and stochastic depth, AutoAugment, batch 128
of 64 + 64, constant lr) is run from the seed-100 student with the "corrupted" half replaced by clean train rows, so
every row is clean. N in {64, 1024} (each mixed 1:1 with N more clean rows), lr 1e-5, 100 steps. The two checkpoints
are checked once each by the joint oracle on the gaussian_noise_s5 instance (V_check S_old is the retention cost of
the recipe alone). If S_old stays at about -40 or worse, F2's gaussian retention failures are a recipe effect.
Output: ``$WEIGHTCODE_RUNS/baselines/<instance>/f1s_diag_f2/0/`` (run.json) and a summary line per N.

    python -B scripts/f1s/f1s_diag_f2.py --instance i00802425 --threads 8
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", default="i00802425")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True
    import dataclasses

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit import labeleq as L
    from weightcode.baselines.fit.cli import load_checked_instances, summary_line
    from weightcode.baselines.fit.runner import CandidatePatch, baseline_run_dir, execute_run
    from weightcode.models.train import configure_determinism

    instance = load_checked_instances([args.instance], allow_main=False)[0]
    condition = L.condition_of(instance)
    data = L.load_f2_data([condition])
    data = dataclasses.replace(data, x_corrupt={condition: data.x_clean})  # every "corrupted" row is clean
    configure_determinism(L.F2_RUN_SEED)
    out_root = Path(tempfile.mkdtemp(prefix="f1s_diag_f2_"))
    rows = []
    for n in (64, 1024):
        trajectory = L.Trajectory(condition, n, 1e-5, (100,), tag="clean")
        rows += L.train_trajectory(trajectory, data, device=torch.device("cpu"), seed=L.F2_RUN_SEED, num_workers=0,
                                   out_root=out_root, provenance={"diagnostic": "PREREG F1s, F2 recipe on clean rows"})
    candidates = []
    for slot, row in enumerate(rows):
        child = L.load_params(out_root / row["path"] / "model.npz")
        candidates.append(CandidatePatch(
            slot=slot, child=child, config={"n_clean_pairs": row["n"] if "n" in row else (64, 1024)[slot],
                                            "lr": 1e-5, "steps": 100},
            hypothesis="F2 recipe on clean rows only (diagnostic): retention cost of the recipe",
            predicted_effect="no condition gain expected; V_check S_old measures the recipe's damage to clean accuracy",
            notes={"train_loss_mean_last100": row["train_loss_mean_last100"]}))

    def generate(evidence):
        return candidates, {"prereg": "PREREG F1s diagnostic for F2", "rows": rows,
                            "fit_rows": "N + N clean train images (no condition rows)"}

    run_dir = baseline_run_dir(instance.instance_id, "f1s_diag_f2", 0, "f1s_diag")
    record = execute_run(instance=instance, spec=L._spec(), run_dir=run_dir, run_seed=0,
                         config={"diagnostic": "F2 recipe on clean rows only", "n": [64, 1024], "lr": 1e-5, "steps": 100},
                         generate=generate, allow_main=False, joint=True, replay_in_fit=True, e10=True)
    print(summary_line(record), flush=True)
    for entry in record["oracle_log"]:
        print(f"{entry['patch_id']}: S_new {entry.get('score_new')} S_old {entry.get('score_old')} S {entry['score']}",
              flush=True)
    print(f"  -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
