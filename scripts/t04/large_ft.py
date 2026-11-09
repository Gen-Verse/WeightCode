"""PREREG T04's large-sample fine-tuning diagnostic (no LLM): full fine-tuning on 2,000 condition rows and 2,000 clean
rows, checked and kept exactly as every T04 reference.

    source scripts/env.sh
    python scripts/t04/large_ft.py --instance <id> [--run-seed 0]

Fit rows: all 2,000 rows of the condition's V_evidence (the split the instance's 64 evidence rows are drawn from) and
all 2,000 rows of the clean train split (the pool of the 64 replay rows), with labels: about 31 times the
deployment rows and the replay rows an editor gets. Optimiser: Adam (betas 0.9, 0.999, eps 1e-8, no weight decay)
on mean cross-entropy, every trainable tensor (grad_full's set: BatchNorm running statistics excepted, BatchNorm in
eval form), float32 on the CPU. Minibatches of 64 condition rows and 64 clean rows, drawn without replacement within
each pass over a part by a torch generator seeded with the run seed (the same sequence for both learning rates).
Candidates: lr {1e-5, 1e-4} x steps {30, 100, 300, 1000}, eight, checked once each on the condition's V_check and
the clean V_check by the joint oracle, kept by joint-retain-v1 like every reference, read on V_select by the dev
readout. Edit limits are not enforced (as grad_full). The run goes to
``$WEIGHTCODE_RUNS/baselines/<instance>/ft_large/t04/<seed>/``. It is a diagnostic of what labelled data at scale buys, read
beside the bar and never part of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.fit.grad import ADAM_BETAS, ADAM_EPS
from weightcode.baselines.fit.runner import (
    CandidatePatch,
    Evidence,
    MethodSpec,
    baseline_run_dir,
    execute_run,
    read_model_spec,
)
from weightcode.baselines.fit.torch_forward import (
    architecture_from,
    assert_parity,
    images_tensor,
    logits,
    to_tensors,
    trainable_tensors,
)
from weightcode.baselines.instance_io import load_instance
from weightcode.data import t04 as T
from weightcode.data.store import load_split, read_dataset_json, split_sha256
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.targets import deployment_splits, load_checked_split

METHOD = "ft_large"
REF_LABEL = "t04"
LEARNING_RATES = (1e-5, 1e-4)
STEP_CHECKPOINTS = (30, 100, 300, 1000)
HALF_BATCH = 64
RULE = "t04-large-ft-v1"
POOL_RULE = "t04-clean-train-pool"
SPEC = MethodSpec(method=METHOD, label="full fine-tuning on 2,000 condition and 2,000 clean labelled rows (T04 "
                  "diagnostic)", uses_labels_for_fit=True, edit_limits_enforced=False)


def fit_rows(instance: Any) -> Evidence:
    """The condition's whole V_evidence, then the whole clean train split, with the base's forward.py predictions."""
    private = instance.private
    if private.get("dataset") != T.DATASET:
        raise SystemExit(f"{instance.instance_id} is not a {T.DATASET} instance")
    splits = deployment_splits(T.DATASET, str(private["condition"]))
    cond = load_checked_split(splits.v_evidence, splits.v_evidence_sha256)
    meta = read_dataset_json(T.DATASET)
    clean = load_split(T.DATASET, "train")
    if split_sha256(clean) != meta["splits"]["train"]["sha256"]:
        raise SystemExit("clean train differs from its dataset.json hash")
    x = np.concatenate([cond["x"], clean["x"]])
    y = np.concatenate([cond["y"], clean["y"]]).astype(np.int64)
    base_pred = predict(instance.forward, instance.params, x, 256)
    return Evidence(x=x, y=y, n_fail=0, n_rand=len(cond["y"]), sha256=splits.v_evidence_sha256, base_pred=base_pred,
                    rule=RULE, n_replay=len(clean["y"]), replay_sha256=meta["splits"]["train"]["sha256"],
                    replay_rule=POOL_RULE)


def generator(instance: Any, run_seed: int):
    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        parity = assert_parity(arch, params, instance.forward, evidence.x[:64])
        trainable = trainable_tensors(arch, params)
        k = evidence.n_deployment
        x_new, y_new = images_tensor(evidence.x[:k]), torch.from_numpy(evidence.y[:k])
        x_old, y_old = images_tensor(evidence.x[k:]), torch.from_numpy(evidence.y[k:])
        candidates: list[CandidatePatch] = []
        losses: dict[str, list[float]] = {}
        for lr in LEARNING_RATES:
            torch.manual_seed(run_seed)
            draw = torch.Generator().manual_seed(int(run_seed))
            tensors = to_tensors(params, trainable=trainable)
            optimizer = torch.optim.Adam([tensors[key] for key in trainable], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                                         weight_decay=0.0, foreach=False)
            queues = {"new": torch.empty(0, dtype=torch.long), "old": torch.empty(0, dtype=torch.long)}

            def take(part: str, n: int) -> torch.Tensor:
                size = k if part == "new" else len(evidence.y) - k
                while len(queues[part]) < n:
                    queues[part] = torch.cat([queues[part], torch.randperm(size, generator=draw)])
                head, queues[part] = queues[part][:n], queues[part][n:]
                return head

            trace = []
            for step in range(1, max(STEP_CHECKPOINTS) + 1):
                i_new, i_old = take("new", HALF_BATCH), take("old", HALF_BATCH)
                xb = torch.cat([x_new[i_new], x_old[i_old]])
                yb = torch.cat([y_new[i_new], y_old[i_old]])
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(logits(arch, tensors, xb), yb)
                loss.backward()
                optimizer.step()
                trace.append(float(loss.detach()))
                if step in STEP_CHECKPOINTS:
                    child = {key: value.detach().numpy().copy() for key, value in tensors.items()}
                    candidates.append(CandidatePatch(
                        slot=len(candidates), child=child, config={"lr": lr, "steps": step},
                        hypothesis=(f"full fine-tuning, Adam lr={lr:g}, {step} steps of {HALF_BATCH} condition and "
                                    f"{HALF_BATCH} clean labelled rows from {k} and {len(evidence.y) - k}; no edit "
                                    "limits"),
                        predicted_effect="more labelled rows than any arm gets: an upper reading of what fine-tuning "
                                         "buys on this condition",
                        notes={"minibatch_loss_last_10": trace[-10:]}))
            losses[f"{lr:g}"] = trace
        info = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity, "trainable": trainable,
                "n_fit_condition": k, "n_fit_clean": len(evidence.y) - k,
                "loss_every_50_steps": {lr: trace[::50] for lr, trace in losses.items()}}
        return candidates, info

    return generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instance", required=True)
    parser.add_argument("--run-seed", type=int, default=0)
    args = parser.parse_args(argv)
    instance = load_instance(instances_root() / args.instance)
    run_dir = baseline_run_dir(instance.instance_id, METHOD, args.run_seed, REF_LABEL)
    config = {"rule": RULE, "learning_rates": list(LEARNING_RATES), "step_checkpoints": list(STEP_CHECKPOINTS),
              "minibatch": {"condition": HALF_BATCH, "clean": HALF_BATCH, "sampling": "without replacement per pass, "
                            "torch.Generator seeded with the run seed, the same sequence for each lr"},
              "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
              "loss": "mean cross-entropy over the 128 rows of a step", "dtype": "float32", "seed": args.run_seed,
              "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": "cpu",
              "candidate_order": [{"lr": lr, "steps": s} for lr in LEARNING_RATES for s in STEP_CHECKPOINTS]}
    record = execute_run(instance=instance, spec=SPEC, run_dir=run_dir, run_seed=args.run_seed, config=config,
                         generate=generator(instance, args.run_seed), joint=True, e10=True,
                         evidence_override=fit_rows(instance))
    print(json.dumps({"run_dir": str(run_dir), "final": record.get("final")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
