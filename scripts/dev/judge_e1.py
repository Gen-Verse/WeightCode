"""E1 and E3 numbers of one LEARN session against the bars of PREREG section 0 (dev, seed 100). Reads only the outputs of
scripts/dev/post_learn.sh: $WEIGHTCODE_RUNS/devreadouts/learn/<sid>/info.json (session_info) and
$WEIGHTCODE_RUNS/devreadouts/learn/<sid>/transfer/transfer_*.json.

Usage: judge_e1.py [--bar INSTANCE=JOINT ...] SESSION_ID [SESSION_ID ...]

``--bar`` gives the joint bar of an instance that ``BARS`` does not hold (PREREG E10: 1 point above the best
reference's joint score under the same kept rule, from the E10 references), with the 10-point gain floor; the
condition name is then read from the session's info.json.

Points are 100 x delta_acc on V_select (condition) and on its clean companion, each against the base model.
E1 (per condition): LEARN supported if joint = gain + clean >= the bar, gain >= 10 (no gain floor for colormix_d1)
and clean >= -1.0. E3: reuse ratio = mean gain on the five unseen students / gain on the session's student,
supported if >= 0.7 with clean >= -1.0 on at least 4 of the 5.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LEARN_ROOT = Path(os.environ["WEIGHTCODE_RUNS"]) / "devreadouts" / "learn"
# PREREG section 0, E1 bars (joint needed), fixed before any LEARN session ran.
BARS = {
    "ibb1f3fc6": ("gaussian_noise_s3", 10.55, True),
    "i7346ae21": ("contrast_s3", 13.25, True),
    "if9258f2a": ("blur5_d1", 14.93, True),
    "i4f9f5974": ("gamma_d1", 15.08, True),
    "i984a5043": ("colormix_d1", 2.20, False),
}


def points(readout: dict | None) -> float | None:
    if not readout or readout.get("delta_acc") is None:
        return None
    return 100.0 * float(readout["delta_acc"])


def fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:+6.2f}"


def judge(sid: str, bars: dict[str, tuple[str, float | None, bool]] | None = None) -> None:
    out = LEARN_ROOT / sid
    info = json.loads((out / "info.json").read_text())
    instance = info.get("instance_id") or info.get("session", {}).get("instance_id")
    final = info["final"]
    condition, bar, gain_floor = {**BARS, **(bars or {})}.get(str(instance), ("?", None, True))
    if condition == "?":
        condition = str(info.get("condition") or info.get("session", {}).get("condition") or "?")
    gain, clean = points(final.get("vselect")), points(final.get("vselect_clean"))
    joint = None if gain is None or clean is None else gain + clean
    print(f"== {sid}  instance {instance}  {condition}")
    print(f"   final: noop {final.get('is_base_noop')}  kind {final.get('kind')}  query {final.get('query_index')}  "
          f"program {final.get('patch_id')}")
    if final.get("vcheck"):
        vc = final["vcheck"]
        print(f"   V_check: score_new {vc.get('score_new')}  score_old {vc.get('score_old')}  S_joint {vc.get('score')}")
    print(f"   E1 V_select: gain {fmt(gain)}  clean {fmt(clean)}  joint {fmt(joint)}  bar {bar}")
    worst = (final.get("vselect_clean") or {}).get("worst_class_delta")
    if worst is not None:  # PREREG E1: reported next to the mean clean change, without gating
        print(f"   worst-class clean change {100.0 * worst:+.2f} (references: -6 to -30)")
    if bar is not None and joint is not None:
        checks = {"joint>=bar": joint >= bar, "clean>=-1.0": clean >= -1.0}
        if gain_floor:
            checks["gain>=10"] = gain >= 10.0
        print(f"   E1 {'SUPPORTED' if all(checks.values()) else 'not supported'}: {checks}")
    rows = []
    for path in sorted((out / "transfer").glob("transfer_*.json")):
        record = json.loads(path.read_text())
        target = record["target"]["instance_id"]
        rows.append((target, record.get("status"), record.get("within_limits"), points(record.get("readouts")),
                     points(record.get("readouts_clean"))))
    own = [row for row in rows if row[0] == instance]
    others = [row for row in rows if row[0] != instance]
    for target, status, within, g, c in rows:
        tag = "own " if target == instance else "E3  "
        print(f"   {tag}{target}  {status}  within_limits {within}  gain {fmt(g)}  clean {fmt(c)}")
    if own and others and own[0][3]:
        gains = [row[3] for row in others if row[3] is not None]
        if gains:
            ratio = (sum(gains) / len(gains)) / own[0][3]
            n_clean_ok = sum(1 for row in others if row[4] is not None and row[4] >= -1.0)
            ok = ratio >= 0.7 and n_clean_ok >= 4 and all(row[2] for row in others)
            print(f"   E3 reuse ratio {ratio:.3f} (mean unseen gain {sum(gains) / len(gains):+.2f} / own "
                  f"{own[0][3]:+.2f}); clean >= -1.0 on {n_clean_ok} of {len(others)}; all within limits "
                  f"{all(row[2] for row in others)} -> {'SUPPORTED' if ok else 'not supported'}")


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar", action="append", default=[], metavar="INSTANCE=JOINT",
                        help="the joint bar of an instance BARS does not hold (gain floor 10 applies)")
    parser.add_argument("sessions", nargs="+")
    args = parser.parse_args(argv)
    bars: dict[str, tuple[str, float | None, bool]] = {}
    for item in args.bar:
        instance, sep, value = item.partition("=")
        if not sep or not instance:
            parser.error(f"--bar takes INSTANCE=JOINT, not {item!r}")
        if instance in BARS:
            parser.error(f"{instance} has a PREREG bar already ({BARS[instance][1]})")
        bars[instance] = ("?", float(value), True)
    for session_id in args.sessions:
        judge(session_id, bars)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
