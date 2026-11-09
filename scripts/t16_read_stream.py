"""PREREG T16 reading of one E11 two-round stream record: the stream matrix, the criteria, the cost and the log overlap.

For a finished stream record (scripts/run_loop_stream.py, teacher mode, two rounds: gaussian then contrast), prints and
optionally writes JSON with:

- the stream matrix in V_select points against the base W0: gaussian, contrast and clean (and the clean worst class)
  after round 1 (W1) and round 2 (W2); W0 is 0 by definition;
- the marginal-gain condition (research plan, unchanged for T16): round 2's condition W1 to W2 >= 10 (contrast in T16,
  gaussian in follow-up R), round 1's condition W1 to W2 >= -3 (forgets <= 3), clean W0 to W2 >= -2.0;
- the formal stream criteria (PREREG E11 stream): mean gain after the last round >= 10, clean >= -2.0, forgetting
  <= 3; and the stream joint (mean gain plus clean change);
- the round-2 session: id, status, checks used, final program, its n_changed, audit verdict and the API cost from the
  transcript's result line (total_cost_usd), and the arm (session.json loop.edit_log);
- how many of the logged entries (round 1's changed entries, the edit log's content; computed the same way for an
  arm-N stream) the round-2 final writes, i.e. differ between W1 and W2.

Reads the stream record, its saved students and the round-2 session's records; no V_select is read again::

    .venv/bin/python scripts/t16_read_stream.py $WEIGHTCODE_RUNS/loop_streams/<stem>.json [--out reading.json]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import load_params, params_sha256
from weightcode.harness.loop import edit_log_document

GAUSSIAN, CONTRAST = "gaussian_noise_s5", "contrast_s5"
SHORT = {GAUSSIAN: "gaussian", CONTRAST: "contrast"}
MARGINAL_GAIN, FORGET_MAX, CLEAN_MIN, MEAN_GAIN_MIN = 10.0, 3.0, -2.0, 10.0


def _points(value: float) -> float:
    return round(100.0 * float(value), 3)


def _student(row: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    params = load_params(Path(str(row[key]["path"])))
    if params_sha256(params) != row[key]["sha256"]:
        raise SystemExit(f"{row[key]['path']} does not hash to the record")
    return params


def _cost(session_root: Path) -> float | None:
    transcript = session_root / "logs" / "transcript.jsonl"
    if not transcript.is_file():
        return None
    for line in reversed(transcript.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            last = json.loads(line)
            return float(last["total_cost_usd"]) if last.get("type") == "result" else None
    return None


def reading(record_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    rounds = record["rounds"]
    if len(rounds) != 2:
        raise SystemExit(f"{record_path}: a T16 stream has two rounds, not {len(rounds)}")
    by_condition = {row["condition"]: row["instance_id"] for row in record["instances"]}
    matrix: dict[str, dict[str, float]] = {"W0": {GAUSSIAN: 0.0, CONTRAST: 0.0, "clean": 0.0, "clean_worst_class": 0.0}}
    for name, row in zip(("W1", "W2"), rounds):
        readouts = row["readouts"]
        clean = readouts[by_condition[GAUSSIAN]]["clean"]
        if {readouts[i]["clean"]["delta_acc"] for i in readouts} != {clean["delta_acc"]}:
            raise SystemExit("the instances' clean readouts differ")
        matrix[name] = {GAUSSIAN: _points(readouts[by_condition[GAUSSIAN]]["select"]["delta_acc"]),
                        CONTRAST: _points(readouts[by_condition[CONTRAST]]["select"]["delta_acc"]),
                        "clean": _points(clean["delta_acc"]), "clean_worst_class": _points(clean["worst_class_delta"])}
    first, second = rounds[0]["condition"], rounds[1]["condition"]  # T16: gaussian then contrast; follow-up R reversed
    if {first, second} != {GAUSSIAN, CONTRAST}:
        raise SystemExit(f"{record_path}: rounds learn {first} and {second}")
    new_gain = round(matrix["W2"][second] - matrix["W1"][second], 3)
    old_change = round(matrix["W2"][first] - matrix["W1"][first], 3)
    marginal = {f"{SHORT[second]}_W1_to_W2": new_gain, f"{SHORT[second]}_met": new_gain >= MARGINAL_GAIN,
                f"{SHORT[first]}_W1_to_W2": old_change, f"{SHORT[first]}_met": old_change >= -FORGET_MAX,
                "clean_W0_to_W2": matrix["W2"]["clean"], "clean_met": matrix["W2"]["clean"] >= CLEAN_MIN,
                "round1_condition": first, "round2_condition": second}
    marginal["met"] = (marginal[f"{SHORT[second]}_met"] and marginal[f"{SHORT[first]}_met"]
                       and marginal["clean_met"])
    summary = record["summary"]
    mean_gain = _points(summary["mean_gain_after_last_round"])
    forgetting = _points(summary["max_forgetting"])
    formal = {"mean_gain": mean_gain, "mean_gain_met": mean_gain >= MEAN_GAIN_MIN, "clean": matrix["W2"]["clean"],
              "clean_met": matrix["W2"]["clean"] >= CLEAN_MIN, "max_forgetting": forgetting,
              "forgetting_met": forgetting <= FORGET_MAX}
    formal["met"] = formal["mean_gain_met"] and formal["clean_met"] and formal["forgetting_met"]
    two = rounds[1]
    root = Path(two["sessions"][-1]["root"])
    session = json.loads((root / "session.json").read_text(encoding="utf-8"))
    base, w1, w2 = _student(rounds[0], "student_before"), _student(rounds[0], "student_after"), \
        _student(two, "student_after")
    log = edit_log_document([(1, base, w1)])
    logged = sum(len(t["flat_indices"]) for t in log["rounds"][0]["tensors"])
    written = sum(int(np.count_nonzero(np.asarray(w2[t["tensor"]]).ravel()[t["flat_indices"]]
                                       != np.asarray(w1[t["tensor"]]).ravel()[t["flat_indices"]]))
                  for t in log["rounds"][0]["tensors"])
    per_tensor = {t["tensor"]: int(np.count_nonzero(np.asarray(w2[t["tensor"]]).ravel()[t["flat_indices"]]
                                                    != np.asarray(w1[t["tensor"]]).ravel()[t["flat_indices"]]))
                  for t in log["rounds"][0]["tensors"]}
    return {
        "record": str(record_path), "tag": record.get("tag"), "edit_log_arm": (record.get("edit_log") or {}).get(
            "enabled"), "billing": record.get("billing"), "round1_source": record["round1"]["source"]["kind"],
        "W1_child_sha256": rounds[0]["child_sha256"], "matrix_points": matrix, "marginal_gain": marginal,
        "formal": formal, "stream_joint": round(mean_gain + matrix["W2"]["clean"], 3),
        "session": {"session_id": root.name, "status": two["session_status"], "round_status": two["status"],
                    "final": two["final"], "queries_used": two["queries_used"], "audit_verdict": two["audit_verdict"],
                    "code_changed_during_session": two["code_changed_during_session"],
                    "edit_log": (session.get("loop") or {}).get("edit_log", {}).get("enabled"),
                    "claude_version": (session.get("claude") or {}).get("version"),
                    "billing": (session.get("claude") or {}).get("billing"), "cost_usd": _cost(root)},
        "logged_entries": logged, "logged_entries_written_by_final": written,
        "logged_entries_written_per_tensor": {k: v for k, v in per_tensor.items() if v},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = [reading(path) for path in args.records]
    text = json.dumps(result if len(result) > 1 else result[0], indent=1)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
