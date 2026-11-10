"""WeightCode on a toy network, end to end, with no data download, no store and no LLM (about one minute).

The script builds a toy store under ``--root``: six tiny synthetic ViT-Lite students (seeds 100 to 105, 3 x 8 x 8
images, 4 classes, 30 tensors) and one repair instance per student for the condition ``gaussian_noise_s5``. The
store is the one the harness tests use (``tests/e11_toy.py``); its images and labels are synthetic and its condition is
a stand-in transform that only carries the benchmark's name. Then it runs the three steps of the method:

1. One LEARN editor session on the seed-100 student. The editor is a fixed script instead of Claude Code. It writes
   two update programs, checks each once on the counted check split, and declares one final. Everything else is the
   real harness: the MCP server the LLM would talk to, the bubblewrap sandbox, the edit limits, the retention floor
   and the final-pick rule.
2. The dev readout of the final network on the condition's selection split and on its clean companion.
3. The reuse check: the final program reruns, unchanged, on the five other students of the same condition. The toy
   labels are planted for the seed-100 student only, so this step shows the mechanics of reuse, not a reuse gain.

Records go under ``--root`` exactly as a real session's go under the store (``runs/sessions/<session id>``), and
``--root``/env.sh points the three store variables at the toy store, so the commands in README.md can be tried on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# The update programs the scripted editor writes. Each is a module with update(student, experience) -> {tensor: delta}.
# HYPOTHESIS and PREDICTED_EFFECT are the two strings every LEARN program declares.
HEADER = 'HYPOTHESIS = "{h}"\nPREDICTED_EFFECT = "{e}"\nimport numpy as np\n\n'
PROGRAMS = {
    # Flips the sign of the final LayerNorm weight on a quarter of its dimensions: the edit the toy labels reward.
    "norm_flip.py": HEADER.format(h="the final normalisation scale carries the wrong sign on the first dimensions",
                                  e="fixes condition rows, breaks no clean row") + """\
def update(student, experience):
    w = np.asarray(student["params"]["classifier.norm.weight"], dtype=np.float64)
    delta = np.zeros_like(w)
    cut = int(round(0.25 * w.size))
    delta[:cut] = -2.0 * w[:cut]
    return {"classifier.norm.weight": delta}
""",
    # Scales the query block of the first attention layer by 1.3: within the limits, but not the rewarded edit.
    "query_scale.py": HEADER.format(h="sharper first-layer queries separate the classes better under the condition",
                                    e="a small gain on condition rows") + """\
def update(student, experience):
    w = np.asarray(student["params"]["classifier.blocks.0.self_attn.qkv.weight"], dtype=np.float64)
    delta = np.zeros_like(w)
    d = w.shape[1]
    delta[:d] = 0.3 * w[:d]
    return {"classifier.blocks.0.self_attn.qkv.weight": delta}
""",
}
FINAL_PROGRAM = "norm_flip.py"

# The toy store of tests/e11_toy.py.
CONDITION = "gaussian_noise_s5"
CLASSES = ("alpha", "beta", "gamma", "delta")
SEEDS = (100, 101, 102, 103, 104, 105)
DECOYS = {100: 101, 101: 100, 102: 103, 103: 102, 104: 105, 105: 104}
ROWS = {"train": 96, "V_select": 160, "V_evidence": 128, "V_check": 160, "test": 32}
RETENTION_FLOOR_FRAC = 0.005  # the floor of the E11 sessions of record: S_old >= -round(0.005 x clean check rows)


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f} points"


def _check_prerequisites(repo: Path) -> None:
    from weightcode.harness.runner import SANDBOX_PYTHON
    from weightcode.sandbox import SandboxError, check_bwrap

    missing = []
    try:
        check_bwrap()  # $WEIGHTCODE_BWRAP or /usr/bin/bwrap, 0.8.0 or newer
    except SandboxError as error:
        missing.append(str(error))
    if not SANDBOX_PYTHON.exists():
        missing.append(f"the sandbox venv {SANDBOX_PYTHON} (run: bash scripts/setup_sandbox_venv.sh)")
    if missing:
        sys.exit("toy_walkthrough: missing " + "; ".join(missing) + f". See {repo / 'README.md'}.")


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=repo / "out" / "toy_store",
                        help="a new directory for the toy store and its records (default: out/toy_store)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        sys.exit(f"toy_walkthrough: {root} already exists and is not empty; remove it or pass another --root")
    _check_prerequisites(repo)

    # The store variables must point at the toy store before any weightcode module resolves them; the real store is never
    # touched (vit_synth.build_store refuses to write anywhere else).
    env = {"WEIGHTCODE_PRIVATE": root / "private", "WEIGHTCODE_SEALED": root / "sealed", "WEIGHTCODE_RUNS": root / "runs"}
    for name, path in env.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    from weightcode.evalkit.devreadout import readout_run
    from weightcode.harness import e10, e11, runner
    from weightcode.harness.jsonio import read_json
    from weightcode.harness.scripted import ScriptedLauncher, call_and_record
    from weightcode.harness.session import SessionPaths
    from weightcode.harness.transfer import run_transfer
    from weightcode.models import vit_synth
    from weightcode.sandbox.program import ProgramSandbox

    print(f"[1/4] Building the toy store under {root}")
    store = vit_synth.build_store(root, vit_synth.TINY, dataset=e11.DATASET,
                                  conditions=dict(zip(e10.STREAM_CONDITIONS, e10.STREAM_CORRUPTIONS)),
                                  plant_condition=CONDITION, class_names=CLASSES, seeds=SEEDS, decoys=DECOYS,
                                  rows=ROWS, instance_conditions=(CONDITION,))
    source = store.ids[(CONDITION, 100)]
    (root / "env.sh").write_text("".join(f"export {name}={path}\n" for name, path in env.items()), encoding="utf-8")
    print(f"      6 students, repair instance {source} on the seed-100 student; store variables in {root / 'env.sh'}")

    print("[2/4] Running one LEARN session with a scripted editor (the real MCP server, sandbox and limits)")
    cell = ("import json\n"
            f"programs = json.loads({json.dumps(json.dumps(PROGRAMS))})\n"
            "for name, source in programs.items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n"
            "print(sorted(programs))\n")

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": cell})
        for name in PROGRAMS:
            call_and_record(client, replies, "patch_check", {"program_file": name})
        call_and_record(client, replies, "declare_final", {"program_file": FINAL_PROGRAM,
                                                           "rationale": "highest checked score above the floor"})

    launcher = ScriptedLauncher(script)
    result = runner.run_session(source, "LEARN", 0, effort="max", wall_timeout_s=300.0, launcher=launcher,
                                skip_audit=True, retention_floor_frac=RETENTION_FLOOR_FRAC)
    if result.status != "complete":
        sys.exit(f"toy_walkthrough: the session ended with status {result.status}: "
                 f"{result.record.get('infra_reasons')}; records in {result.root}")
    paths = SessionPaths(result.root)
    session = read_json(result.root / "session.json")
    floor = int(session["kept_rule"]["s_old_floor"])
    print(f"      session {result.root.name}; retention floor S_old >= {floor}")
    for reply in launcher.replies:
        if reply.name == "patch_check":
            shown = reply.payload
            if reply.is_error:
                print(f"      check {reply.arguments['program_file']}: refused ({json.dumps(shown)[:160]})")
            else:
                print(f"      check {reply.arguments['program_file']}: score_new {shown['score_new']:+d}, "
                      f"score_old {shown['score_old']:+d}, score {shown['score']:+d}")
    final = read_json(paths.selection_json)["final"]
    print(f"      final program: {final.get('patch_id')} (kind {final.get('kind')})")

    print("[3/4] Dev readout of the final network on the selection split (never the sealed test split)")
    readout = readout_run(result.root / "run.json", clean=True)
    print(f"      condition rows: {readout['n']}, accuracy change {_points(readout['readouts']['delta_acc'])}")
    print(f"      clean rows:     {readout['n_clean']}, accuracy change "
          f"{_points(readout['readouts_clean']['delta_acc'])}")

    print("[4/4] Reuse: the same final program, unchanged, on the five other students")
    program = paths.final / FINAL_PROGRAM
    sandbox = ProgramSandbox(python=runner.SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)
    written = run_transfer(program, [source, *e11.transfer_targets(source)], root / "transfer", sandbox)
    seed_of = {instance_id: seed for (_, seed), instance_id in store.ids.items()}
    for path in written:
        record = json.loads(path.read_text(encoding="utf-8"))
        target = record["target"]["instance_id"]
        own = " (its own student)" if target == source else ""
        print(f"      seed {seed_of[target]} {target}{own}: {record['status']}, condition "
              f"{_points(record['readouts']['delta_acc'])}, clean {_points(record['readouts_clean']['delta_acc'])}")
    print("      The toy labels reward this edit on the seed-100 student only, so the other five show no systematic")
    print("      gain here. The same reuse workflow can be applied to saved experiment programs.")

    print("\nDone. Records:")
    print(f"  session:  {result.root} (session.json, logs/, final/, run.json)")
    print(f"  transfer: {root / 'transfer'}")
    print("Next: render the prompts a real LLM session on this toy instance would get (no API call):")
    print(f"  source {root / 'env.sh'}")
    print(f"  .venv/bin/python scripts/run_editor_session.py --instance {source} --arm LEARN --repeat 1 "
          f"--effort max --wall-timeout-min 120 --retention-floor-frac {RETENTION_FLOOR_FRAC} --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
