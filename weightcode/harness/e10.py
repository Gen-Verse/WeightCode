"""PREREG E10 in the harness: E1's LEARN protocol on CIFAR-10-C at severity 5 with ResNet-20 (dataset ``cifar10_r20``).

An E10 instance is an ordinary evidence-v1 instance of the dataset ``cifar10_r20``: its trial, experience, limits,
check and readouts are E1's (``weightcode.harness.trial``, ``experience``, ``arms.applied_limits``, ``clean``), and the
student is a ResNet-20 with BatchNorm whose running statistics are 1-D parameters of model.npz (exempt from rel-Frob,
counted in the changed entries, one op each, so max_ops is max(64, 97)). What this module adds:

* Leak words (``leak_words``), added to the session's text scan only, as graphnew and fscil do: the condition leak
  tokens the instance builder scans for (``weightcode.evalkit.instances.condition_leak_tokens``) of every E10 corruption,
  fog included, the four family names and the benchmark's names ("CIFAR-10-C", the authors, "severity"). The
  registered condition list (``weightcode.data.conditions.CONDITIONS``) holds none of the E10 names, so without them a
  session would forbid only its own condition. The sandbox self-test, a case-variant substring match over mounts,
  environment and listing, gets the distinctive ones only (``self_test_tokens``: the condition and corruption names
  with an underscore, every E10 instance id and the given session ids), since words such as "fog" or "s5" can occur
  in host mount paths by chance. A loop round's library programs are scanned, and withheld, without the leak words,
  on E4-loop's standard (``weightcode.harness.leaks``: ``ForbiddenStrings.library_text``): the teacher wrote them itself,
  and an earlier round's comments ("contrast", "noise") name no condition of the current round (coordinator
  decision, 2026-11-04, after 10 of the 12 E1 and E4-loop kept programs would have been withheld on the full list).
* MCP timeouts (``MCP_CONNECT_TIMEOUT_MS``, ``MCP_TOOL_TIMEOUT_MS``). The session server computes the base
  predictions of every check part before it answers Claude Code's initialize, and a patch check runs the program
  (up to 300 s) and then predicts every check part. With the numpy ResNet-20 forward that is about 66 s for the 8,000
  rows of a B session and about 165 s for the 20,000 rows of a round-4 loop session at 4 threads on a loaded host
  (measured 2026-11-04: 1,024 images in 8.4 s), against E1's defaults of 120 s to connect and 600 s per tool call.
  An E10 session defaults to 900 s and 1,200 s and refuses smaller values (``weightcode.harness.runner``).
* Lookups for the launch commands (``stream_ids``, ``transfer_targets``; ``python -m weightcode.harness.e10``): instances
  are found by dataset, condition and seed, never by a remembered id.

Nothing here reads V_select, the sealed store or a condition's images.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.evalkit.instances import condition_leak_tokens

DATASET = "cifar10_r20"
SEVERITY = 5
CORRUPTIONS = ("gaussian_noise", "defocus_blur", "snow", "fog", "contrast")
FAMILIES = {"gaussian_noise": "noise", "defocus_blur": "blur", "snow": "weather", "fog": "weather",
            "contrast": "digital"}
"""``weightcode.data.e10``'s dataset, severity, corruptions (fog is snow's substitute) and families, restated so that the
runner and the session server do not import that module (it imports torch); a test holds the two equal."""
STREAM_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "snow", "contrast")
"""PREREG E10: the four conditions, in the B order and the stream order (fog replaces snow only if snow fails the
fidelity check, which it passed)."""
SOURCE_SEED = 100
TRANSFER_SEEDS = (101, 102, 103, 104, 105)
"""PREREG E10 C: the seed-100 final runs unchanged on the same condition's students of these seeds."""
BENCHMARK_WORDS = ("cifar-10-c", "cifar10-c", "cifar10c", "cifar-10c", "hendrycks", "dietterich", "imagenet-c",
                   "severity")
MCP_CONNECT_TIMEOUT_MS = 900_000
MCP_TOOL_TIMEOUT_MS = 1_200_000


def condition_name(corruption: str) -> str:
    """``weightcode.data.e10.condition_name``: the condition of one corruption at severity 5."""
    if corruption not in CORRUPTIONS:
        raise ValueError(f"unknown E10 corruption {corruption!r}")
    return f"{corruption}_s{SEVERITY}"


STREAM_CONDITIONS = tuple(condition_name(corruption) for corruption in STREAM_CORRUPTIONS)


class E10HarnessError(LookupError):
    """The E10 instances asked for do not exist as one active instance each."""


def is_e10(private: Mapping[str, Any]) -> bool:
    """True for an evidence-v1 instance of the E10 dataset (its private.json)."""
    return private.get("dataset") == DATASET and private.get("evidence_rule") == EVIDENCE_RULE_V1


def condition_tokens(corruption: str) -> list[str]:
    """The instance builder's leak tokens of one E10 corruption at severity 5 (name, parts, stems, generic words)."""
    return condition_leak_tokens(condition_name(corruption), {"condition": {"corruption": corruption,
                                                                            "severity": SEVERITY}})


def leak_words() -> list[str]:
    """Lower-case words an E10 session's text scan forbids beyond its base list (module docstring)."""
    words = {*BENCHMARK_WORDS, *FAMILIES.values()}
    for corruption in CORRUPTIONS:
        words.update(condition_tokens(corruption))
    return sorted(words)


def instance_ids() -> list[str]:
    """Every active E10 instance id, any split role."""
    from weightcode.harness.instance import list_instances

    return sorted(str(row["instance_id"]) for row in list_instances(split_role=None, dataset=DATASET))


def self_test_tokens(session_ids: Iterable[str] = ()) -> list[str]:
    """Tokens an E10 session forbids like a condition name (self-test and text scan): every E10 condition and
    corruption name with an underscore, every active E10 instance id and ``session_ids``."""
    names = {name for corruption in CORRUPTIONS for name in (corruption, condition_name(corruption)) if "_" in name}
    return sorted({*names, *instance_ids(), *(str(item) for item in session_ids if item)})


def library_session_ids(library: Sequence[Mapping[str, Any]]) -> list[str]:
    """The source session ids of a loop round's library (``weightcode.harness.loop`` context entries)."""
    found = []
    for item in library:
        source = item.get("source")
        if isinstance(source, Mapping) and source.get("session_id"):
            found.append(str(source["session_id"]))
    return sorted(set(found))


def floored_mcp_timeouts(applies: bool, line: str, reason: str, tool_ms: int | None, connect_ms: int | None,
                         defaults: tuple[int, int]) -> tuple[int, int]:
    """The (tool, connect) MCP timeouts: the given values, else this module's floors when ``applies`` (a session of
    PREREG ``line``, whose check predictions are ``reason``) and ``defaults`` otherwise. Where the floors apply, values
    below them are refused (``ValueError``). Shared by E10 and E11 (``weightcode.harness.e11``)."""
    floor_tool, floor_connect = (MCP_TOOL_TIMEOUT_MS, MCP_CONNECT_TIMEOUT_MS) if applies else defaults
    tool = floor_tool if tool_ms is None else int(tool_ms)
    connect = floor_connect if connect_ms is None else int(connect_ms)
    if applies and (tool < MCP_TOOL_TIMEOUT_MS or connect < MCP_CONNECT_TIMEOUT_MS):
        raise ValueError(f"an {line} session needs MCP timeouts of at least {MCP_TOOL_TIMEOUT_MS} ms per tool call and "
                         f"{MCP_CONNECT_TIMEOUT_MS} ms to connect ({reason}), got {tool} and {connect}")
    return tool, connect


def mcp_timeouts(private: Mapping[str, Any], tool_ms: int | None, connect_ms: int | None,
                 defaults: tuple[int, int]) -> tuple[int, int]:
    """The (tool, connect) MCP timeouts of a session: the given values, else E10's on an E10 instance and
    ``defaults`` otherwise. An E10 instance refuses values below E10's (``ValueError``)."""
    return floored_mcp_timeouts(is_e10(private), "E10", "the ResNet-20 check predictions", tool_ms, connect_ms,
                                defaults)


def find(condition: str, seed: int) -> str:
    """The one active E10 dev instance of ``condition`` and student ``seed``."""
    from weightcode.harness.instance import find_instance

    try:
        return find_instance(dataset=DATASET, seed=int(seed), condition=condition)
    except LookupError as exc:
        raise E10HarnessError(str(exc)) from None


def stream_ids(seed: int = SOURCE_SEED) -> list[str]:
    """The stream's instances on the student ``seed``, one per condition in ``STREAM_CONDITIONS`` order."""
    return [find(condition, seed) for condition in STREAM_CONDITIONS]


def transfer_targets(instance_id: str) -> list[str]:
    """PREREG E10 C: the instances of the same condition on the students of ``TRANSFER_SEEDS``."""
    from weightcode.harness.instance import read_instance_record

    private = read_instance_record(instance_id).private
    if not is_e10(private):
        raise E10HarnessError(f"{instance_id} is not an E10 instance")
    if int(private["seed"]) != SOURCE_SEED:
        raise E10HarnessError(f"{instance_id} is a seed-{private['seed']} instance; C transfers seed {SOURCE_SEED}")
    return [find(str(private["condition"]), seed) for seed in TRANSFER_SEEDS]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E10 instance lookups for the launch commands")
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("ids", help="the stream's instance ids, in condition order")
    ids.add_argument("--seed", type=int, default=SOURCE_SEED)
    ids.add_argument("--condition", help="print only this condition's id")
    ids.add_argument("--json", action="store_true", help="condition -> id as JSON")
    targets = sub.add_parser("targets", help="C: the same condition's instances of seeds 101 to 105")
    targets.add_argument("instance")
    args = parser.parse_args(argv)
    if args.command == "targets":
        print(" ".join(transfer_targets(args.instance)))
        return 0
    if args.condition is not None:
        print(find(args.condition, args.seed))
        return 0
    found = dict(zip(STREAM_CONDITIONS, stream_ids(args.seed)))
    print(json.dumps(found, indent=1) if args.json else " ".join(found.values()))
    return 0


__all__ = ["BENCHMARK_WORDS", "CORRUPTIONS", "DATASET", "FAMILIES", "MCP_CONNECT_TIMEOUT_MS", "MCP_TOOL_TIMEOUT_MS",
           "SEVERITY", "SOURCE_SEED", "STREAM_CONDITIONS", "STREAM_CORRUPTIONS", "TRANSFER_SEEDS", "E10HarnessError",
           "condition_name", "condition_tokens", "find", "floored_mcp_timeouts", "instance_ids", "is_e10", "leak_words",
           "library_session_ids", "main", "mcp_timeouts", "self_test_tokens", "stream_ids", "transfer_targets"]

if __name__ == "__main__":
    raise SystemExit(main())
