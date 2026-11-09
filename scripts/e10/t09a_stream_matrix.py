"""T09-A (Execution Plan v1.1): the per-round capability matrix of recorded E10 streams and its marginal readings.

Reads stream records (weightcode.harness.loop_stream.v0 or weightcode.baselines.checked_reference_stream.v0) and writes, for each,
the matrix A[t, j] = V_select gain on condition j after round t (W0 = the base, A[0, j] = 0), the clean change after
each round, and these readings (points, i.e. accuracy x 100):

- own_marginal[t]      A[t, j_t] - A[t-1, j_t], the gain of round t on its own condition, read against W(t-1);
- round_sum[t]         sum over all conditions of A[t, j] - A[t-1, j], so a broad update shows on the others too;
- seen_mean[t]         mean of A[t, j] over the conditions of rounds 1..t; all_mean[t] the mean over every condition,
                       future ones included;
- forgetting_signed[j] A[t_j, j] - A[T, j] (the frozen stream definition, weightcode.harness.program_stream.stream_summary);
- forgetting_best[j]   max over t >= t_j of A[t, j] minus A[T, j] (T09-A's definition, never negative);
- forward[j]           A[t_j - 1, j], the gain on condition j before its own round (improvement before installation);
- stop_after_2         the stream read at W2 against W4 on the same four conditions (mean, clean, mean + clean).

No model is run: every number is a recorded readout. Usage:
    python scripts/e10/t09a_stream_matrix.py --record NAME=PATH [--record ...] --out OUT.json [--markdown OUT.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POINTS = 100.0
SCHEMAS = ("weightcode.harness.loop_stream.v0", "weightcode.baselines.checked_reference_stream.v0")


class StreamRecordError(ValueError):
    """A record that this analysis cannot read as a complete stream."""


def _instance_id(item: Any) -> str:
    return item if isinstance(item, str) else str(item["instance_id"])


def stream_matrix(record: Mapping[str, Any]) -> dict[str, Any]:
    """The matrix and readings of one stream record (module docstring)."""
    if record.get("schema") not in SCHEMAS:
        raise StreamRecordError(f"schema {record.get('schema')!r} is not one of {SCHEMAS}")
    rounds = list(record["rounds"])
    instances = [_instance_id(item) for item in record["instances"]]
    if len(rounds) != len(instances):
        raise StreamRecordError(f"{len(rounds)} rounds for {len(instances)} instances")
    conditions = [str(row["condition"]) for row in rounds]
    by_instance = {row["instance_id"]: row["condition"] for row in rounds}
    if [by_instance[i] for i in instances] != conditions:
        raise StreamRecordError("round order does not match the instance order")
    gain = [[0.0] * len(instances)]
    clean = [0.0]
    for row in rounds:
        readouts = row["readouts"]
        missing = [i for i in instances if i not in readouts]
        if missing:
            raise StreamRecordError(f"round {row['round']} has no readout for {missing}")
        gain.append([readouts[i]["select"]["delta_acc"] * POINTS for i in instances])
        cleans = {round(readouts[i]["clean"]["delta_acc"], 12) for i in instances}
        if len(cleans) != 1:
            raise StreamRecordError(f"round {row['round']}: the clean change differs across instances: {cleans}")
        clean.append(cleans.pop() * POINTS)
    last = len(rounds)
    own = [gain[t][t - 1] - gain[t - 1][t - 1] for t in range(1, last + 1)]
    round_sum = [sum(gain[t][j] - gain[t - 1][j] for j in range(last)) for t in range(1, last + 1)]
    seen_mean = [sum(gain[t][:t]) / t for t in range(1, last + 1)]
    all_mean = [sum(gain[t]) / last for t in range(1, last + 1)]
    per_condition = {}
    for j, name in enumerate(conditions):
        acquired = j + 1
        since = [gain[t][j] for t in range(acquired, last + 1)]
        per_condition[name] = {
            "round": acquired,
            "gain_before_own_round": gain[acquired - 1][j],
            "gain_after_own_round": gain[acquired][j],
            "gain_after_last_round": gain[last][j],
            "forgetting_signed": gain[acquired][j] - gain[last][j],
            "forgetting_best": max(since) - gain[last][j],
        }
    earlier = [name for name in conditions[:-1]]
    stop = None
    if last >= 2:
        stop = {
            "w2_mean": all_mean[1], "w2_clean": clean[2], "w2_joint": all_mean[1] + clean[2],
            "wT_mean": all_mean[-1], "wT_clean": clean[last], "wT_joint": all_mean[-1] + clean[last],
            "rounds_after_2_mean_gain": all_mean[-1] - all_mean[1],
            "rounds_after_2_per_condition": {name: gain[last][j] - gain[2][j] for j, name in enumerate(conditions)},
        }
    return {
        "conditions": conditions,
        "instances": instances,
        "round_status": [row["status"] for row in rounds],
        "gain": gain,
        "clean": clean,
        "own_marginal": own,
        "round_sum": round_sum,
        "seen_mean": seen_mean,
        "all_mean": all_mean,
        "per_condition": per_condition,
        "max_forgetting_signed": max(per_condition[name]["forgetting_signed"] for name in earlier),
        "max_forgetting_best": max(per_condition[name]["forgetting_best"] for name in earlier),
        "stop_after_2": stop,
    }


def markdown(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Two tables: the matrices, then the readings, one row per stream (points, two decimals)."""
    first = next(iter(results.values()))
    names = first["conditions"]
    lines = ["| stream, after | " + " | ".join(names) + " | clean | seen mean | all mean |",
             "|---" * (len(names) + 4) + "|"]
    for stream, res in results.items():
        for t in range(1, len(res["gain"])):
            row = " | ".join(f"{value:+.2f}" for value in res["gain"][t])
            lines.append(f"| {stream} W{t} ({res['round_status'][t - 1]}) | {row} | {res['clean'][t]:+.2f} | "
                         f"{res['seen_mean'][t - 1]:+.2f} | {res['all_mean'][t - 1]:+.2f} |")
    lines += ["", "| stream | own-condition marginal by round | all-condition sum by round | forward (gain before "
              "own round) | forgetting signed / best-since | W2 mean, clean, joint | final mean, clean, joint |",
              "|---|---|---|---|---|---|---|"]
    for stream, res in results.items():
        own = ", ".join(f"{value:+.2f}" for value in res["own_marginal"])
        rsum = ", ".join(f"{value:+.2f}" for value in res["round_sum"])
        fwd = ", ".join(f"{name.split('_')[0]} {row['gain_before_own_round']:+.2f}"
                        for name, row in list(res["per_condition"].items())[1:])
        stop = res["stop_after_2"]
        lines.append(f"| {stream} | {own} | {rsum} | {fwd} | {res['max_forgetting_signed']:+.2f} / "
                     f"{res['max_forgetting_best']:+.2f} | {stop['w2_mean']:+.2f}, {stop['w2_clean']:+.2f}, "
                     f"{stop['w2_joint']:+.2f} | {stop['wT_mean']:+.2f}, {stop['wT_clean']:+.2f}, "
                     f"{stop['wT_joint']:+.2f} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="append", required=True, help="NAME=PATH of a stream record")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    results: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for item in args.record:
        name, _, path = item.partition("=")
        if not name or not path:
            parser.error(f"--record wants NAME=PATH, got {item!r}")
        raw = Path(path).read_bytes()
        results[name] = stream_matrix(json.loads(raw))
        sources[name] = {"path": str(Path(path).absolute()), "sha256": hashlib.sha256(raw).hexdigest()}
    conditions = {tuple(res["conditions"]) for res in results.values()}
    if len(conditions) != 1:
        raise StreamRecordError(f"the streams do not share one condition order: {conditions}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "weightcode.e10.t09a_stream_matrix.v0", "units": "points (accuracy x 100)",
                                    "readout": "V_select", "sources": sources, "streams": results},
                                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
