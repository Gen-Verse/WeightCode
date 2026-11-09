"""PREREG E12 in the harness: E10 with a publicly pretrained ResNet-18 student (dataset ``cifar10_r18``).

An E12 instance is an ordinary evidence-v1 instance of the dataset ``cifar10_r18`` on E10's data. The conditions,
rows, trial, experience, limits, check and readouts are E10's (``weightcode.harness.trial``, ``experience``,
``arms.applied_limits``, ``clean``). Only the student differs: torchvision's ImageNet ResNet-18 fine-tuned on E1's
train rows, with ImageNet normalisation and bilinear upsampling to R x R in front of it (``weightcode.models.resnet18up``).
The limits are rank-based and need no change. The 20 convolution weights and the fc weight are rank 2 or more and
fall under the relative Frobenius limit. The 80 BatchNorm tensors (weights, biases and running statistics) and the
fc bias are 1-D, so they are exempt from it and counted in the 20% of changed entries. max_ops is max(64, tensor
count), 102 for this layout. What this module adds:

* ``DATASET``: the store's dataset id (``weightcode.data.e12.DATASET``). Its ``_``-separated parts become leak tokens
  (``weightcode.evalkit.instances.leak_tokens``); "r18" is a whole-word token that no description of the network spells.
* Leak words: E10's (``weightcode.harness.e10.leak_words``), because the conditions are E10's. Architecture and
  pretraining words are not leak words: the editor reads forward.py and model_spec.json, which describe the network.
  The text an editor sees never names the pretraining dataset with a hyphenated suffix, because one E10 leak word is
  the name of that dataset's corruption benchmark.
* Self-test tokens (``self_test_tokens``): E10's condition and corruption names with an underscore, every active
  E12, E11 and E10 instance id (the conditions are shared, so an id of any of them names one) and the given session
  ids.
* MCP timeouts (``mcp_timeouts``): E10's floors, 1,200 s per tool call and 900 s to connect, refused below. A check
  predicts 4,000 condition rows and 4,000 clean rows after the program run (up to 300 s), and the server predicts the
  base on the same 8,000 rows at connect. The time of forward.py per 8,000 rows at R is recorded in PREREG E12 before
  any instance; a launch that needs more passes larger values, which this module accepts.
* ``check_student``: before any session write, an E12 student must have exactly the resnet18_up tensors
  (``weightcode.models.resnet18up.param_shapes``), BatchNorm running statistics included as 1-D parameters.
* Lookups for the launch commands (``find``, ``condition_ids``; ``python -m weightcode.harness.e12``): instances are found
  by dataset, condition and seed in ``$WEIGHTCODE_PRIVATE/instances/index.json``, never by a remembered id. A condition
  that missed headroom has no instance, and ``ids`` reports it as null. PREREG E12 runs no C (one student), so there
  are no transfer targets.

Nothing here reads V_select, the sealed store or a condition's images.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.harness import e10, e11

DATASET = "cifar10_r18"
CONDITIONS = ("gaussian_noise_s5", "contrast_s5", "defocus_blur_s5", "snow_s5")
"""PREREG E12: gaussian_noise_s5 and contrast_s5, then the substitutes defocus_blur_s5 and snow_s5 in that order."""
SOURCE_SEED = e10.SOURCE_SEED
MCP_CONNECT_TIMEOUT_MS = e10.MCP_CONNECT_TIMEOUT_MS
MCP_TOOL_TIMEOUT_MS = e10.MCP_TOOL_TIMEOUT_MS
BATCHNORM_STATS = ("running_mean", "running_var")


class E12HarnessError(LookupError):
    """The E12 instances asked for do not exist as one active instance each, or a student is not an E12 student."""


def is_e12(private: Mapping[str, Any]) -> bool:
    """True for an evidence-v1 instance of the E12 dataset (its private.json)."""
    return private.get("dataset") == DATASET and private.get("evidence_rule") == EVIDENCE_RULE_V1


def leak_words() -> list[str]:
    """Lower-case words an E12 session's text scan forbids beyond its base list: E10's (module docstring)."""
    return e10.leak_words()


def instance_ids() -> list[str]:
    """Every active E12 instance id, any split role."""
    from weightcode.harness.instance import list_instances

    return sorted(str(row["instance_id"]) for row in list_instances(split_role=None, dataset=DATASET))


def self_test_tokens(session_ids: Iterable[str] = ()) -> list[str]:
    """Tokens an E12 session forbids like a condition name (self-test and text scan): E10's condition and corruption
    names with an underscore, every active E12, E11 and E10 instance id and ``session_ids``."""
    return sorted({*e10.self_test_tokens(session_ids), *e11.instance_ids(), *instance_ids()})


def mcp_timeouts(private: Mapping[str, Any], tool_ms: int | None, connect_ms: int | None,
                 defaults: tuple[int, int]) -> tuple[int, int]:
    """The (tool, connect) MCP timeouts of a session: the given values, else E10's floors on an E12 instance and
    ``defaults`` otherwise. An E12 instance refuses values below the floors (``ValueError``)."""
    return e10.floored_mcp_timeouts(is_e12(private), "E12", "the ResNet-18 check predictions at the upsampled size",
                                    tool_ms, connect_ms, defaults)


def check_student(params: Mapping[str, Any]) -> dict[str, Any]:
    """The student's tensor kinds under the E12 limits (module docstring); ``E12HarnessError`` unless it has exactly
    the resnet18_up tensors with their shapes."""
    from weightcode.models import resnet18up as M

    shapes = {str(name): tuple(int(s) for s in np.shape(value)) for name, value in params.items()}
    head = shapes.get(M.HEAD_WEIGHT)
    if head is None or len(head) != 2:
        raise E12HarnessError(f"not an E12 student: no 2-D {M.HEAD_WEIGHT}")
    expected = M.param_shapes(head[0])
    if shapes != expected:
        missing = sorted(set(expected) - set(shapes))
        extra = sorted(set(shapes) - set(expected))
        wrong = sorted(k for k in set(expected) & set(shapes) if shapes[k] != expected[k])
        raise E12HarnessError(f"not an E12 student: missing {missing[:4]}, unexpected {extra[:4]}, wrong shapes "
                              f"{wrong[:4]}")
    one_d = [name for name, shape in shapes.items() if len(shape) < 2]
    stats = [name for name in shapes if name.endswith(BATCHNORM_STATS)]
    return {"n_tensors": len(shapes), "n_entries": int(sum(int(np.prod(shape)) for shape in shapes.values())),
            "n_rel_frob_tensors": len(shapes) - len(one_d), "n_exempt_1d_tensors": len(one_d),
            "n_exempt_1d_entries": int(sum(int(np.prod(shapes[name])) for name in one_d)),
            "n_batchnorm_stat_tensors": len(stats),
            "n_batchnorm_stat_entries": int(sum(int(np.prod(shapes[name])) for name in stats))}


def find(condition: str, seed: int) -> str:
    """The one active E12 dev instance of ``condition`` and student ``seed``."""
    from weightcode.harness.instance import find_instance

    if condition not in CONDITIONS:
        raise E12HarnessError(f"{condition!r} is not an E12 condition; E12's are {list(CONDITIONS)}")
    try:
        return find_instance(dataset=DATASET, seed=int(seed), condition=condition)
    except LookupError as exc:
        raise E12HarnessError(str(exc)) from None


def condition_ids(seed: int = SOURCE_SEED) -> dict[str, str | None]:
    """Condition -> the one active E12 dev instance of student ``seed``, or None where there is none (a condition that
    missed headroom or was not built). More than one instance of a (condition, seed) is an error."""
    from weightcode.harness.instance import list_instances

    found: dict[str, str | None] = {}
    for condition in CONDITIONS:
        rows = list_instances(dataset=DATASET, seed=int(seed), condition=condition)
        if len(rows) > 1:
            raise E12HarnessError(f"{len(rows)} active E12 instances of {condition} seed {seed}: "
                                  f"{[row.get('instance_id') for row in rows]}")
        found[condition] = str(rows[0]["instance_id"]) if rows else None
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E12 instance lookups for the launch commands")
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("ids", help="one condition's instance id, or condition -> id (null: none) as JSON")
    ids.add_argument("--seed", type=int, default=SOURCE_SEED)
    ids.add_argument("--condition", help="print only this condition's id (an error unless exactly one exists)")
    args = parser.parse_args(argv)
    if args.condition is not None:
        print(find(args.condition, args.seed))
        return 0
    print(json.dumps(condition_ids(args.seed), indent=1))
    return 0


__all__ = ["BATCHNORM_STATS", "CONDITIONS", "DATASET", "MCP_CONNECT_TIMEOUT_MS", "MCP_TOOL_TIMEOUT_MS", "SOURCE_SEED",
           "E12HarnessError", "check_student", "condition_ids", "find", "instance_ids", "is_e12", "leak_words", "main",
           "mcp_timeouts", "self_test_tokens"]

if __name__ == "__main__":
    raise SystemExit(main())
