"""E5 post-hoc diagnostic: does a gradient fit on the E5 evidence find a solution inside the E5 program limits?

Not a PREREG reference. It answers one question about the E5 miss (session s92c8b990b661d23a ended at the base):
the only reference that learned the new operator, seq_grad_full, changed 99.6% of the entries, every tensor of rank 2
or more within the relative Frobenius norm limit 0.5, so the 20% changed-entry cap alone excluded it. The two
in-limits references (seq_grad_emb_head, seq_grad_new_row) train too few tensors to learn it. This script trains
every tensor but only a fixed 19% of each tensor's entries, so the 20% cap holds by construction, and scales each
checkpoint to the norm limit as the in-limits references do.

Two masks, fixed before any run, each the top 19% of a tensor's entries (stable argsort, ties by flat index):

* ``seq_grad_sparse19_sal``: by |d loss / d w| at the base weights on the 128 evidence rows.
* ``seq_grad_sparse19_traj``: by |w - w_base| after unconstrained full-batch Adam, lr 1e-3, 300 steps (the
  seq_grad_full trajectory whose checkpoint was that reference's final).

Everything else is seq_grad_full's recipe (``weightcode.baselines.fit.seqgrad``): the same 64 new-operator plus 64 replay
rows, mean token cross-entropy, Adam lr {1e-3, 1e-2} x steps {10, 30, 100, 300}, run seed 0, 8 candidates, each
checked once by the seqskill joint oracle. Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance>/<method>/e5diag/0/``; read
them with ``weightcode.evalkit.devreadout`` (own final, and ``--retention-floor -20`` for the kept rule).

    source scripts/env.sh
    .venv/bin/python scripts/dev/e5_diag_sparse.py --instance qa000f6cd
"""

from __future__ import annotations

import argparse
import os
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MASK_FRACTION = 0.19
"""Entries trained per tensor; below the 20% changed-entry cap with a margin, so every candidate meets it."""
TRAJ_LR = 1e-3
TRAJ_STEPS = 300
REF_LABEL = "e5diag"
METHODS = ("seq_grad_sparse19_sal", "seq_grad_sparse19_traj")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", required=True, help="seqskill instance id q<8 hex>")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0, help="run seed (torch seed; directory name)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit.edits import scale_to_norm_limit
    from weightcode.baselines.fit.grad import LEARNING_RATES, STEP_CHECKPOINTS
    from weightcode.baselines.fit.runner import CandidatePatch, MethodSpec, baseline_run_dir, execute_run
    from weightcode.baselines.fit.seqgrad import (adam_checkpoints, rebuild, seq_grad_config, seq_parity, token_loss)
    from weightcode.evalkit.seqskill import load_seqskill_instance, seqskill_program_limits

    instance = load_seqskill_instance(args.instance)
    params = instance.params
    limits = seqskill_program_limits(instance.task["limits"], len(params))
    trainable = sorted(params)

    def top_fraction(score: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        masks = {}
        for key in trainable:
            flat = np.abs(score[key]).ravel()
            k = int(np.floor(MASK_FRACTION * flat.size))
            if k < 1:
                raise RuntimeError(f"{key}: {flat.size} entries give an empty {MASK_FRACTION:g} mask")
            order = np.argsort(-flat, kind="stable")
            mask = np.zeros(flat.size, dtype=bool)
            mask[order[:k]] = True
            masks[key] = mask.reshape(score[key].shape)
        return masks

    def make_generator(method: str):
        def generate(evidence):
            model = rebuild(params)
            parity, parity_bound = seq_parity(model, params, instance.forward, evidence.x)
            x = torch.from_numpy(np.ascontiguousarray(evidence.x, dtype=np.int64))
            y = torch.from_numpy(np.ascontiguousarray(evidence.y, dtype=np.int64))
            if method == "seq_grad_sparse19_sal":
                for tensor in model.parameters():
                    tensor.requires_grad_(True)
                token_loss(model, x, y).backward()
                score = {name: p.grad.detach().numpy().copy() for name, p in model.named_parameters()}
            else:
                (_, traj, _), = adam_checkpoints(params, x, y, trainable=trainable, lr=TRAJ_LR,
                                                 checkpoints=(TRAJ_STEPS,), seed=args.seed)
                score = {key: traj[key].astype(np.float64) - params[key].astype(np.float64) for key in trainable}
            if set(score) != set(trainable):
                raise RuntimeError(f"score keys differ from the student's: {sorted(set(score) ^ set(trainable))}")
            masks = top_fraction(score)
            n_mask = int(sum(m.sum() for m in masks.values()))
            n_params = int(sum(v.size for v in params.values()))
            info = {"model_config": model.config.as_dict(), "parity_max_abs_logit_diff": parity,
                    "parity_bound": parity_bound, "mask_rule": method, "mask_fraction_per_tensor": MASK_FRACTION,
                    "n_trainable_entries": n_mask, "n_params": n_params, "trainable_frac": n_mask / n_params}
            with torch.no_grad():
                info["base_evidence_loss"] = float(token_loss(rebuild(params), x, y))
            candidates = []
            for lr in LEARNING_RATES:
                for steps, child, loss in adam_checkpoints(params, x, y, trainable=trainable, lr=lr,
                                                           checkpoints=STEP_CHECKPOINTS, seed=args.seed, masks=masks):
                    for key in trainable:
                        if not np.array_equal(child[key][~masks[key]], params[key][~masks[key]]):
                            raise RuntimeError(f"{key} changed outside its mask")
                    notes = {"evidence_loss_at_checkpoint": loss}
                    child, scaling = scale_to_norm_limit(params, child, trainable, limit=limits.max_rel_frob,
                                                         min_ndim=limits.rel_frob_min_ndim,
                                                         exempt=limits.rel_frob_exempt)
                    notes["norm_scaled_tensors"] = [key for key, entry in scaling.items() if entry["scaled"]]
                    if notes["norm_scaled_tensors"]:
                        with torch.no_grad():
                            notes["evidence_loss_after_scaling"] = float(token_loss(rebuild(child), x, y))
                    candidates.append(CandidatePatch(
                        slot=len(candidates), child=child, config={"lr": lr, "steps": steps},
                        hypothesis=(f"{method}: Adam lr={lr:g} for {steps} full-batch steps on the {len(evidence.y)} "
                                    f"evidence rows, every tensor, top {MASK_FRACTION:g} of each tensor's entries"),
                        predicted_effect="new-operator rows answered right if a sparse fit inside the limits exists",
                        notes=notes))
            info["n_candidates_norm_scaled"] = sum(1 for c in candidates if c.notes["norm_scaled_tensors"])
            return candidates, info
        return generate

    for method in args.methods:
        spec = MethodSpec(method=method,
                          label=("E5 post-hoc diagnostic, not a PREREG reference: Adam on a fixed 19% of every "
                                 "tensor's entries, E5 program limits enforced"),
                          uses_labels_for_fit=True, edit_limits_enforced=True)
        config = seq_grad_config("seq_grad_full", args.seed, instance)
        config.update({"trained": f"every tensor, top {MASK_FRACTION:g} of its entries by the {method} mask",
                       "norm_limit": limits.max_rel_frob, "mask_fraction_per_tensor": MASK_FRACTION,
                       "traj_mask": {"lr": TRAJ_LR, "steps": TRAJ_STEPS} if method.endswith("traj") else None})
        run_dir = baseline_run_dir(instance.instance_id, method, args.seed, REF_LABEL)
        record = execute_run(instance=instance, spec=spec, run_dir=run_dir, run_seed=args.seed, config=config,
                             generate=make_generator(method), require_valid_candidate=True, seqskill=True)
        print(method, run_dir / "run.json", record.get("final", {}).get("patch_id"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
