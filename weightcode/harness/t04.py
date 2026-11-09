"""PREREG T04 in the harness: native ImageNet installation with the IMAGENET1K_V1 ResNet-18 (dataset ``in1k_t04``).

A T04 instance is an ordinary evidence-v1 instance of the dataset ``in1k_t04`` (``weightcode.data.t04``): ImageNet-C's
gaussian_noise or contrast at the severity PREREG T04's rule chose on D, on the dev rows of the ImageNet validation
split, with the network's own 1000-class head. Trial, experience, limits, check and readouts are E10's
(``weightcode.harness.trial``, ``experience``, ``arms.applied_limits``, ``clean``): rel-Frob 0.5 on tensors of rank 2 or
more, at most 20% of the entries changed, BatchNorm running statistics as 1-D parameters, 8 checks with S_new on the
condition's V_check and S_old on the clean V_check (4,000 rows each). What this module adds:

* ``DATASET`` and the condition grid (``CONDITIONS``: both corruptions at severities 1 to 5; the instances exist only
  for the chosen severities).
* Leak words (``leak_words``), added to the session's text scan only, as for E10: the instance builder's condition
  leak tokens of every condition in the grid, the family word "noise" and the benchmark's names (``BENCHMARK_WORDS``).
  E10's list is not used: its words "snow" and "digital" are parts of ImageNet class names (snow leopard, snowmobile,
  snowplow, digital clock, digital watch), which every T04 task.json lists. The class names are built free of every
  word here (``weightcode.data.t04.class_names`` with ``weightcode.data.t04.forbidden_name_tokens``).
* Self-test tokens (``self_test_tokens``): the grid's condition and corruption names with an underscore, every active
  T04 instance id and the given session ids.
* MCP timeouts (``mcp_timeouts``): E10's floors, 1,200 s per tool call and 900 s to connect, refused below. A check
  predicts 4,000 condition rows and 4,000 clean rows at 224 x 224 after the program run (up to 300 s); T03 measured
  0.038 s per image at 4 threads on an ASPIRE2B node, about 304 s per check. The LEARN dry run of NSCC request R37
  measures it on the session's own settings before any session.
* ``check_student``: before any session write, a T04 student must have exactly the torchvision resnet18 tensors with a
  1000-class head (``weightcode.models.resnet18up.param_shapes(1000)``), BatchNorm running statistics as 1-D parameters.
* Lookups for the launch commands (``find``, ``condition_ids``; ``python -m weightcode.harness.t04``): instances are found
  by dataset, condition and seed in ``$WEIGHTCODE_PRIVATE/instances/index.json``, never by a remembered id.

Nothing here reads V_select, the sealed rows or a condition's images.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from weightcode.data import t04 as D
from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.evalkit.instances import condition_leak_tokens
from weightcode.harness import e10

DATASET = D.DATASET
CONDITIONS: tuple[str, ...] = tuple(D.condition_name(c, s) for c in D.CORRUPTIONS for s in D.SEVERITIES)
SOURCE_SEED = 100
N_CLASSES = D.N_CLASSES
BENCHMARK_WORDS = D.BENCHMARK_WORDS
MCP_CONNECT_TIMEOUT_MS = e10.MCP_CONNECT_TIMEOUT_MS
MCP_TOOL_TIMEOUT_MS = e10.MCP_TOOL_TIMEOUT_MS
BATCHNORM_STATS = ("running_mean", "running_var")


class T04HarnessError(LookupError):
    """The T04 instances asked for do not exist as one active instance each, or a student is not a T04 student."""


def is_t04(private: Mapping[str, Any]) -> bool:
    """True for an evidence-v1 instance of the T04 dataset (its private.json)."""
    return private.get("dataset") == DATASET and private.get("evidence_rule") == EVIDENCE_RULE_V1


def leak_words() -> list[str]:
    """Lower-case words a T04 session's text scan forbids beyond its base list (module docstring)."""
    words = {*BENCHMARK_WORDS, "noise"}
    for corruption in D.CORRUPTIONS:
        for severity in D.SEVERITIES:
            words.update(condition_leak_tokens(D.condition_name(corruption, severity),
                                               {"condition": {"corruption": corruption, "severity": severity}}))
    return sorted(words)


def instance_ids() -> list[str]:
    """Every active T04 instance id, any split role."""
    from weightcode.harness.instance import list_instances

    return sorted(str(row["instance_id"]) for row in list_instances(split_role=None, dataset=DATASET))


def self_test_tokens(session_ids: Iterable[str] = ()) -> list[str]:
    """Tokens a T04 session forbids like a condition name (self-test and text scan): the grid's condition and
    corruption names with an underscore, every active T04 instance id and ``session_ids``."""
    names = {name for name in (*CONDITIONS, *D.CORRUPTIONS) if "_" in name}
    return sorted({*names, *instance_ids(), *(str(item) for item in session_ids if item)})


def mcp_timeouts(private: Mapping[str, Any], tool_ms: int | None, connect_ms: int | None,
                 defaults: tuple[int, int]) -> tuple[int, int]:
    """The (tool, connect) MCP timeouts of a session: the given values, else E10's floors on a T04 instance and
    ``defaults`` otherwise. A T04 instance refuses values below the floors (``ValueError``)."""
    return e10.floored_mcp_timeouts(is_t04(private), "T04", "the native ResNet-18 check predictions at 224 x 224",
                                    tool_ms, connect_ms, defaults)


def check_student(params: Mapping[str, Any]) -> dict[str, Any]:
    """The student's tensor kinds under the E10 limits; ``T04HarnessError`` unless it has exactly the resnet18 tensors
    with a 1000-class head."""
    from weightcode.models import resnet18up as M

    shapes = {str(name): tuple(int(s) for s in np.shape(value)) for name, value in params.items()}
    expected = M.param_shapes(N_CLASSES)
    if shapes != expected:
        missing = sorted(set(expected) - set(shapes))
        extra = sorted(set(shapes) - set(expected))
        wrong = sorted(k for k in set(expected) & set(shapes) if shapes[k] != expected[k])
        raise T04HarnessError(f"not a T04 student: missing {missing[:4]}, unexpected {extra[:4]}, wrong shapes "
                              f"{wrong[:4]}")
    one_d = [name for name, shape in shapes.items() if len(shape) < 2]
    stats = [name for name in shapes if name.endswith(BATCHNORM_STATS)]
    return {"n_tensors": len(shapes), "n_entries": int(sum(int(np.prod(shape)) for shape in shapes.values())),
            "n_rel_frob_tensors": len(shapes) - len(one_d), "n_exempt_1d_tensors": len(one_d),
            "n_exempt_1d_entries": int(sum(int(np.prod(shapes[name])) for name in one_d)),
            "n_batchnorm_stat_tensors": len(stats),
            "n_batchnorm_stat_entries": int(sum(int(np.prod(shapes[name])) for name in stats))}


def find(condition: str, seed: int = SOURCE_SEED) -> str:
    """The one active T04 dev instance of ``condition`` and student ``seed``."""
    from weightcode.harness.instance import find_instance

    if condition not in CONDITIONS:
        raise T04HarnessError(f"{condition!r} is not a T04 condition; the grid is {list(CONDITIONS)}")
    try:
        return find_instance(dataset=DATASET, seed=int(seed), condition=condition)
    except LookupError as exc:
        raise T04HarnessError(str(exc)) from None


def condition_ids(seed: int = SOURCE_SEED) -> dict[str, str]:
    """Condition -> the one active T04 dev instance of student ``seed``, for every condition that has one."""
    from weightcode.harness.instance import list_instances

    found: dict[str, str] = {}
    for condition in CONDITIONS:
        rows = list_instances(dataset=DATASET, seed=int(seed), condition=condition)
        if len(rows) > 1:
            raise T04HarnessError(f"{len(rows)} active T04 instances of {condition} seed {seed}: "
                                  f"{[row.get('instance_id') for row in rows]}")
        if rows:
            found[condition] = str(rows[0]["instance_id"])
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T04 instance lookups for the launch commands")
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("ids", help="one condition's instance id, or condition -> id as JSON")
    ids.add_argument("--seed", type=int, default=SOURCE_SEED)
    ids.add_argument("--condition", help="print only this condition's id (an error unless exactly one exists)")
    args = parser.parse_args(argv)
    if args.condition is not None:
        print(find(args.condition, args.seed))
        return 0
    print(json.dumps(condition_ids(args.seed), indent=1))
    return 0


__all__ = ["BATCHNORM_STATS", "BENCHMARK_WORDS", "CONDITIONS", "DATASET", "MCP_CONNECT_TIMEOUT_MS",
           "MCP_TOOL_TIMEOUT_MS", "N_CLASSES", "SOURCE_SEED", "T04HarnessError", "check_student", "condition_ids", "find",
           "instance_ids", "is_t04", "leak_words", "main", "mcp_timeouts", "self_test_tokens"]

if __name__ == "__main__":
    raise SystemExit(main())
