"""Example ``scan`` (Figure 1b, left): a rule becomes a capability in a SCAN sequence model, reproduced from public data.

The network of record (instance ``ncc5c260f``, a small transformer with 803,975 weights trained on SCAN's
``addprim_jump`` split) gets no command such as "jump twice" right: its training data hold "jump" only alone. A frozen language
model wrote the program ``centroid_meanrow_floor.py`` for it in session ``s9be29c2c302e4daa``. This example

1. downloads the network, the session record and the program from the Hugging Face dataset (pinned revision, every
   file checked by sha256) and SCAN's text files from its GitHub repository at the commit the paper used;
2. rebuilds the paper's splits from those files (``weightcode.data.scan.build_condition_splits``, a pure function of
   them) and checks that the held-out test set hashes to the one the paper's test reading used;
3. reruns the program on the network, with no language model, and checks that the edited network it builds is the
   published one bit for bit;
4. reads base and edited network on the 2,568 held-out new commands and 1,000 held-out old commands.

The test reading of record gave +99.81 points of exact match on the new commands (0% to 99.8%) and 0.00 on the old
ones, with 257 weights changed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256
from weightcode.data import scan as S
from weightcode.evalkit import scanskill as K
from weightcode.examples.fetch import checked_files, hf_files
from weightcode.examples.replay import ReplayError, recorded_experience_sha256, replay_session
from weightcode.harness.jsonio import utc_now
from weightcode.harness.seqskill import scan_experience
from weightcode.sandbox.program import ProgramRunner

NAME = "scan"
TITLE = "SCAN: a rule becomes a capability (Figure 1b, left)"
CONDITION = "addprim_jump"
INSTANCE_ID = "ncc5c260f"
SESSION_ID = "s9be29c2c302e4daa"
PROGRAM_SHA256 = "b003c6b9e7a03a3d367633b600cb2ca8cac11cde428ea12b55bdc60362fc5320"
CHILD_SHA256 = "6c9bc3bdaf347dfb6c868a71f17506bea58ab51f82707f236692604dc34198a0"
TEST_SHA256 = "32f4aea890601a2193db6b704c3475fe3408812b391f79a455f1e9a15d6c393a"
N_TEST_NEW, N_TEST_OLD = 2568, 1000
PAPER = {"gain_points": 99.80529595015575, "clean_points": 0.0, "n_changed": 257}
"""The test reading of record (sealed read of finals list v1, entry E13_addprim_jump_s100)."""
SCAN_RAW_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/{commit}/{path}"
SESSION_PREFIXES = (f"records/sessions/{SESSION_ID}/run.json", f"records/sessions/{SESSION_ID}/session.json",
                    f"records/sessions/{SESSION_ID}/final/")


def _stamp() -> str:
    return utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")


def test_rows(raw_dir: Path, private: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The held-out new and old commands, rebuilt from SCAN's files and checked against the paper's hashes."""
    built = S.build_condition_splits(raw_dir, CONDITION)
    for split, key in (("V_select", "vselect_content_sha256"), ("old_select", "oldselect_content_sha256")):
        if params_sha256(built.private[split]) != private[key]:
            raise ReplayError(f"the rebuilt {split} split differs from the one the network's instance was built with")
    if params_sha256(built.sealed) != TEST_SHA256:
        raise ReplayError(f"the rebuilt test set hashes to {params_sha256(built.sealed)}, not the paper's {TEST_SHA256}")
    part = np.asarray(built.sealed["part"])
    if not (np.all(part[:N_TEST_NEW] == S.PART_NEW) and np.all(part[N_TEST_NEW:] == S.PART_OLD)
            and len(part) == N_TEST_NEW + N_TEST_OLD):
        raise ReplayError("the rebuilt test set is not 2,568 new commands followed by 1,000 old commands")
    return ({key: value[:N_TEST_NEW] for key, value in built.sealed.items()},
            {key: value[N_TEST_NEW:] for key, value in built.sealed.items()})


def run(out: Path, runner: ProgramRunner, *, scan_dir: Path | None = None, batch_size: int = K.EVALUATION_BATCH,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    """Reproduce the example under ``out`` (module docstring) and return its summary, also written to result.json."""
    downloads = out / "downloads"
    log(f"[1/4] Network, session record and program from Hugging Face -> {downloads / 'hf'}")
    hf_files([f"weights/instances/{INSTANCE_ID}/", *SESSION_PREFIXES], downloads / "hf")
    source = "local copy " + str(scan_dir) if scan_dir else f"github.com/brendenlake/SCAN at {S.SCAN_COMMIT[:7]}"
    log(f"[2/4] SCAN text files from {source}; rebuilding the paper's test split")
    raw_dir = downloads / "scan"
    checked_files({path: SCAN_RAW_URL.format(commit=S.SCAN_COMMIT, path=path) for path in S.RAW_SHA256},
                  S.RAW_SHA256, raw_dir, scan_dir)
    instance = load_instance(downloads / "hf" / "weights" / "instances" / INSTANCE_ID)
    new, old = test_rows(raw_dir, instance.private)

    log(f"[3/4] Rerunning the program {SESSION_ID}/final/ on the network (no language model)")
    session_dir = downloads / "hf" / "records" / "sessions" / SESSION_ID
    experience = scan_experience(instance)
    if experience[1]["content_sha256"] != recorded_experience_sha256(session_dir):
        raise ReplayError("the assembled experience differs from the one the session recorded")
    run_dir = out / "runs" / _stamp()
    replay = replay_session(instance, session_dir, PROGRAM_SHA256, CHILD_SHA256, runner, run_dir / "replay",
                            experience=experience)

    log(f"[4/4] Reading both networks on the held-out test commands ({N_TEST_NEW:,} new, {N_TEST_OLD:,} old)")
    readout = K.scan_readout_arrays(new, old, instance.forward, instance.params, replay.child, batch_size)
    n_weights = int(sum(value.size for value in instance.params.values()))
    summary = {
        "example": NAME, "created_at": utc_now(), "instance_id": INSTANCE_ID, "session_id": SESSION_ID,
        "test_sha256": TEST_SHA256, "n_weights": n_weights, "replay": replay.record,
        "new": {key: readout["new"][key] for key in ("n", "em_base", "em_child", "fixed", "broken")},
        "old": {key: readout["old"][key] for key in ("n", "em_base", "em_child", "fixed", "broken")},
        "gain_points": readout["gain_points"], "clean_points": readout["clean_points"], "paper": PAPER,
        "matches_paper": bool(replay.exact and abs(readout["gain_points"] - PAPER["gain_points"]) < 1e-9
                              and abs(readout["clean_points"] - PAPER["clean_points"]) < 1e-9),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report(summary, log)
    log(f"Record: {run_dir / 'result.json'}")
    return summary


def report(summary: dict[str, Any], log: Callable[[str], None] = print) -> None:
    """The result table, with the paper's numbers beside it."""
    new, old = summary["new"], summary["old"]
    log("")
    log(f"  {TITLE}")
    log("  Exact match on held-out test commands      before    after")
    log(f"    new commands with 'jump' ({new['n']:,})       {100 * new['em_base']:6.1f}%  {100 * new['em_child']:6.1f}%")
    log(f"    old commands ({old['n']:,})                   {100 * old['em_base']:6.1f}%  {100 * old['em_child']:6.1f}%")
    replay = summary["replay"]
    log(f"  Weights changed: {replay['n_changed']:,} of {summary['n_weights']:,}")
    log(f"  Rebuilt network identical to the published one: {'yes' if replay['exact'] else 'NO'}"
        f" (sha256 {replay['child_sha256'][:12]})")
    log(f"  Paper: +{summary['paper']['gain_points']:.2f} pt on new commands, {summary['paper']['clean_points']:+.2f} pt"
        f" on old ones, {summary['paper']['n_changed']} weights changed -> "
        f"{'reproduced exactly' if summary['matches_paper'] else 'NOT reproduced'}")


__all__ = ["CHILD_SHA256", "INSTANCE_ID", "NAME", "PAPER", "PROGRAM_SHA256", "SESSION_ID", "TEST_SHA256", "TITLE",
           "report", "run", "test_rows"]
