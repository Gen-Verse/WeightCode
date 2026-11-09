"""PREREG E11, mechanism of the gaussian final (sc918e11dceef637a, gate3_lower; seed 100): A1, A2 and A3 (T0 to T3).

The spec is the entry "Mechanism of the E11 gaussian final" of docs/PREREG_v0_DRAFT.md (d83f0fc). No LLM call. Dev only:
instance i00802425 (gaussian_noise_s5, the seed-100 ViT-Lite-7/4 student mae049764), its experience (x_new, x_old; no
variant reads a label), V_check with its clean companion through the joint oracle (T0 and T1 only) and V_select with
its clean companion for every readout. Nothing reads the sealed store.

Shared paths, reused, not reimplemented:

* Readouts: ``weightcode.harness.clean.dev_readout_joint`` (``weightcode.evalkit.devreadout.dev_readout_joint``), the readout of
  ``scripts/dev/post_learn.sh``, the transfer runs and the references. The base model's predictions on the two
  V_select sets are memoized within a process (``install_base_memo``: same function, same params hash, same rows hash),
  which changes no number and halves the cost.
* Program runs (repro, T0, T1, T2): ``weightcode.harness.transfer.stage_student`` and ``weightcode.harness.program.run_update`` in
  the LEARN program sandbox (``.venv-sandbox``, 4 threads, 300 s), the derived patch of ``write_derived_patch``, as
  ``scripts/run_program_transfer.py`` runs a frozen program.
* Checks (T0, T1): ``weightcode.baselines.fit.runner.execute_run`` with ``joint=True, e10=True`` (the E10 program limits,
  the joint oracle, at most 8 checks), as ``run_fit_baselines.py --e11`` runs the references; the pick is
  ``weightcode.evalkit.devreadout.readout_retention`` with floor -20 (joint-retain-v1, S_old >= -20 of 4000), read on
  V_select, as the references' floor20 records.
* Limits: ``weightcode.evalkit.e10.e10_program_limits`` (rel-Frob 0.5 on tensors of rank >= 2, at most 20% of the entries,
  max_ops 85), asserted equal to the session's recorded limits. An unchecked variant (A1, T2, T3) is applied within
  them; if refused, the refusal is recorded and the variant is read out without the size limits (as transfer does).

Subcommands, each writing ``$WEIGHTCODE_RUNS/devreadouts/e11_mech/<name>.jsonl`` (never overwritten) and its sha256:

* ``repro``: the final's deltas re-applied to the base (child hash e40c2078...) and the program rerun in the sandbox
  (deltas hash 5b9cdfac...), read on V_select; must give gain +31.625 and clean +0.125.
* ``a1``: for each non-empty subset of the three gates, (a) gate off: the gate units' linear2 columns set to 0, the
  bias fold kept; (b) revert: that block's four MLP tensors restored to the base.
* ``a2``: per gate, the fraction of tokens above the ramp threshold, the mean gate output norm, on the condition and
  clean V_select rows (paired by idx); the residual offset (condition minus clean, paired) at each gate block's output
  in the base and the final.
* ``t0``: a priori alignment, block {1, 2, 3, 4} x strength {0.5, 1.0}: linear2.bias -= s (mean block output on x_new
  minus on x_old, over images and all 65 tokens). Checked.
* ``t1``: the program's gate built from x_new against x_old (unpaired, no synthetic noise, no KL oracle), patch gate
  at block {1, 2, 3}, class-token gates at block + 2 and the last block on or off. Checked. ``--instance i3eb3fc78``
  runs the same six programs (built from the gaussian final, byte-identical) on E11 contrast_s5 instead (PREREG A3x,
  1010241): only the instance, its experience and limits checks, and the record labels change.
* ``t2``: the program with the two units of each gate drawn at random (draws 0 to 4). Not checked.
* ``t3``: the final with every gate always open: the gate units' columns zeroed (fold kept) and the program's own
  offsets subtracted through linear2.bias at every token. Not checked.
* ``summary``: the tables and the pre-registered reading from the records.

    source scripts/env.sh
    .venv/bin/python -B scripts/e11/e11_mech.py repro
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import itertools
import json
import math
import os
import sys
import time
import types
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

THREADS = 4
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = str(THREADS)  # before numpy is imported, or BLAS ignores it
sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE

import numpy as np  # noqa: E402

SESSION_ID = "sc918e11dceef637a"
INSTANCE_ID = "i00802425"
PROGRAM_FILE = "gate3_lower.py"
PROGRAM_SHA256 = "3c1cebc35ef05b3bc4552a556319509dfac240a4cce5bdc07acbda6c2f0d5946"
DELTAS_SHA256 = "5b9cdfac442ef41a36106864eefe047c3d349fd0998996c1fc771488ad5bf767"
CHILD_SHA256 = "e40c20786ae91c904cfd7a195addb7e03b6601b955d731518da1ff54bc01c251"
EXPERIENCE_CONTENT_SHA256 = "ae18db7953e50ed04c8411acba56ce76d43740d58775a1d1c95ce5de31c3365f"
SPEC = "docs/PREREG_v0_DRAFT.md d83f0fc, 'Mechanism of the E11 gaussian final'"
TARGETS = {
    INSTANCE_ID: {"condition": "gaussian_noise_s5", "session": SESSION_ID,
                  "experience_content_sha256": EXPERIENCE_CONTENT_SHA256, "spec": SPEC,
                  "run_spec": "PREREG E11 mechanism entry (d83f0fc)"},
    "i3eb3fc78": {"condition": "contrast_s5", "session": "s760cb52ce82fb937",
                  "experience_content_sha256": "316a3850839c89660edcbf185d6c71b020f958274f7808d634b6bbec0fe7a7eb",
                  "spec": "docs/PREREG_v0_DRAFT.md 1010241, 'A3x, the T1 gate as a fixed template on the other E11 "
                          "condition'", "run_spec": "PREREG A3x (1010241)"},
}
"""Instances a checked program run may target: the gaussian instance (every subcommand) and, for ``t1 --instance``,
the E11 contrast_s5 instance (A3x). Each names the E11 B session whose experience and limits the run must match;
the programs are always derived from the gaussian final (``final_source``)."""
_TARGET_ID = INSTANCE_ID
RECORDED_DELTA_ACC = {"gain": 0.31625, "clean": 0.00125}
"""The final's V_select delta_acc (condition, clean) of record: DEV_RESULTS E11 B (+31.62, +0.12, joint +31.75)."""
RETENTION_FLOOR = -20
"""joint-retain-v1 on 4000 clean V_check rows, frac 0.005 (PREREG E10 and E11 kept rule)."""
GATE_BLOCKS = (2, 4, 6)
GATE_KINDS = {2: "patch", 4: "cls", 6: "cls"}
EXPECTED_UNITS = {2: (503, 360), 4: (31, 34), 6: (182, 418)}
"""(j1, j2) per gate as the final program printed them (program_run.json stdout)."""
K_RAMP = 3.0
"""The program's ramp height K: unit j1 is GELU(u), unit j2 GELU(u - K), output (GELU(u) - GELU(u - K)) (-off / K)."""
T0_BLOCKS = (1, 2, 3, 4)
T0_STRENGTHS = (0.5, 1.0)
T1_PATCH_BLOCKS = (1, 2, 3)
T2_DRAWS = (0, 1, 2, 3, 4)
READING_FRACTION = 0.75
READING_CLEAN_FLOOR = -1.0
MAX_CHECKS = 8
MLP_PARTS = ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias")
SCHEMA = "weightcode.e11.mech.v1"
A2_CHUNK = 250


class MechError(RuntimeError):
    """An input does not match the record it must reproduce, or a derived program cannot be built as specified."""


# --------------------------------------------------------------------------------------------------------------------
# Paths, inputs and records


def runs_root() -> Path:
    from weightcode.common.paths import runs_dir

    return Path(runs_dir())


def session_dir() -> Path:
    return runs_root() / "sessions" / SESSION_ID


def target() -> dict[str, Any]:
    return {"instance_id": _TARGET_ID, **TARGETS[_TARGET_ID]}


def target_session_dir() -> Path:
    """The E11 B session of the instance a run targets (the gaussian session unless ``t1 --instance``)."""
    return runs_root() / "sessions" / TARGETS[_TARGET_ID]["session"]


def out_dir() -> Path:
    return runs_root() / "devreadouts" / "e11_mech"


def work_root() -> Path:
    return runs_root() / "e11" / "mech"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_INSTANCE: Any = None


def instance() -> Any:
    """The dev instance, loaded once; refuses anything but a dev instance of the expected student."""
    global _INSTANCE
    if _INSTANCE is None:
        from weightcode.baselines.instance_io import load_instance
        from weightcode.evalkit.instances import instances_root

        condition = TARGETS[_TARGET_ID]["condition"]
        inst = load_instance(instances_root() / _TARGET_ID)
        if inst.private.get("split_role") != "dev" or inst.private.get("condition") != condition:
            raise MechError(f"{_TARGET_ID} is not the dev {condition} instance")
        recorded = json.loads((target_session_dir() / "session.json").read_text())
        if recorded["instance_id"] != _TARGET_ID or inst.params_sha256 != recorded["real_model_sha256"]:
            raise MechError(f"{_TARGET_ID} is not the instance and student of session {recorded['instance_id']}")
        _INSTANCE = inst
    return _INSTANCE


def program_limits(inst: Any) -> Any:
    """E10 program limits of the instance, asserted equal to the limits the session applied."""
    from weightcode.evalkit.e10 import e10_program_limits
    from weightcode.patch import Limits

    limits = e10_program_limits(inst.task["limits"], len(inst.params))
    recorded = Limits(**json.loads((target_session_dir() / "session.json").read_text())["limits"])
    if limits != recorded:
        raise MechError(f"E10 program limits {limits} differ from the session's {recorded}")
    return limits


def final_source() -> str:
    path = session_dir() / "final" / PROGRAM_FILE
    if file_sha256(path) != PROGRAM_SHA256:
        raise MechError(f"{path} does not hash to the kept program")
    return path.read_text(encoding="utf-8")


def final_child() -> dict[str, np.ndarray]:
    """The session's final child, rebuilt by devreadout.final_child from its patch under the session's limits."""
    from weightcode.evalkit.devreadout import final_child as rebuild

    run = json.loads((session_dir() / "run.json").read_text())
    child = rebuild(run, session_dir(), instance())
    return {key: np.asarray(value) for key, value in child.items()}


def write_records(name: str, records: Sequence[Mapping[str, Any]]) -> Path:
    """One JSONL file per subcommand, never overwritten, with a .sha256 file beside it."""
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.jsonl"
    if path.exists():
        raise MechError(f"{path} exists; records are never overwritten")
    from weightcode.baselines.fit.runner import jsonable

    path.write_text("".join(json.dumps(jsonable(r), sort_keys=True, allow_nan=False) + "\n" for r in records),
                    encoding="utf-8")
    (folder / f"{name}.jsonl.sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="utf-8")
    print(f"wrote {path} sha256 {file_sha256(path)}", flush=True)
    return path


def provenance() -> dict[str, Any]:
    from weightcode.baselines.fit.runner import git_state

    return {"script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
            "git": git_state(), "threads": THREADS, "numpy": np.__version__, "created_at": utc_now(),
            "spec": TARGETS[_TARGET_ID]["spec"]}


# --------------------------------------------------------------------------------------------------------------------
# Readout, limits and edit records


_BASE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def install_base_memo(inst: Any) -> None:
    """Memoize the base model's predictions inside weightcode.evalkit.devreadout (same function, same inputs)."""
    import weightcode.evalkit.devreadout as devreadout
    from weightcode.common.identity import params_sha256

    if getattr(devreadout.predict, "_e11_mech_memo", False):
        return
    original = devreadout.predict
    base_sha = inst.params_sha256

    def predict(forward: Any, params: Mapping[str, np.ndarray], x: np.ndarray, batch_size: int) -> np.ndarray:
        if params_sha256(params) != base_sha:
            return original(forward, params, x, batch_size)
        key = (hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), int(batch_size))
        if key not in _BASE_CACHE:
            _BASE_CACHE[key] = original(forward, params, x, batch_size)
        return _BASE_CACHE[key].copy()

    predict._e11_mech_memo = True  # type: ignore[attr-defined]
    devreadout.predict = predict


def points(value: float) -> float:
    return 100.0 * float(value)


def readout(inst: Any, child: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """dev_readout_joint of one child: gain and clean in points, joint = gain + clean, the worst classes, per class."""
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
    from weightcode.harness import clean

    install_base_memo(inst)
    started = time.perf_counter()
    record = clean.dev_readout_joint(inst, child, DEFAULT_BATCH_SIZE)
    return summarize_readout(record, time.perf_counter() - started, clean.source_name())


def summarize_readout(record: Mapping[str, Any], seconds: float | None = None, source: str | None = None
                      ) -> dict[str, Any]:
    ro, rc = record["readouts"], record["readouts_clean"]
    gain, clean_points = points(ro["delta_acc"]), points(rc["delta_acc"])
    return {
        "gain": gain, "clean": clean_points, "joint": gain + clean_points,
        "worst_class_clean": points(rc["worst_class_delta"]), "worst_class_gain": points(ro["worst_class_delta"]),
        "per_class_gain": [points(v) for v in ro["per_class_acc_delta"]],
        "per_class_clean": [points(v) for v in rc["per_class_acc_delta"]],
        "fixed": ro["fixed"], "broken": ro["broken"], "fixed_clean": rc["fixed"], "broken_clean": rc["broken"],
        "base_acc": ro["base_acc"], "child_acc": ro["child_acc"], "base_acc_clean": rc["base_acc"],
        "child_acc_clean": rc["child_acc"], "n": record["n"], "n_clean": record["n_clean"],
        "clean_split_sha256": record.get("clean_split_sha256"), "readout_seconds": seconds,
        "readout_source": source or "weightcode.evalkit.devreadout.dev_readout_joint",
    }


def apply_within_limits(inst: Any, patch: Mapping[str, Any], patch_dir: Path) -> tuple[dict[str, np.ndarray],
                                                                                         dict[str, Any]]:
    """Apply a patch under the E10 program limits; on a refusal, record it and apply it without the size limits
    (``weightcode.harness.transfer._child``'s rule). Returns the child and its edit record."""
    from weightcode.baselines.fit.edits import edit_stats_under
    from weightcode.patch import PatchError, apply_patch, task_limits

    limits = program_limits(inst)
    try:
        result = apply_patch(inst.params, patch, patch_dir, limits, inst.params_sha256)
        within, messages, child_kind = True, [], "within_limits"
    except PatchError as refusal:
        within, messages, child_kind = False, list(refusal.messages), "unbounded"
        unbounded = dataclasses.replace(limits, max_rel_frob=None, max_changed_frac=None)
        result = apply_patch(inst.params, patch, patch_dir, unbounded, inst.params_sha256)
    child = {key: np.asarray(value) for key, value in result.child.items()}
    stats = edit_stats_under(inst.params, child, limits, edit_size=result.edit_size)
    changed = {key: value for key, value in stats["per_tensor"].items() if value["changed"]}
    record = {
        "limits": task_limits(limits), "within_limits": within and bool(stats["within_limits"]),
        "limit_messages": messages, "violations": stats["violations"], "readout_child": child_kind,
        "child_sha256": result.child_sha256, "tensors_changed": sorted(changed),
        "n_tensors_changed": len(changed), "n_changed": stats["n_changed"], "changed_frac": stats["changed_frac"],
        "edit_size": stats["edit_size"], "edit_size_source": "executor, the patch as written",
        "per_tensor_rel_frob": {key: value["rel_frob"] for key, value in changed.items()},
        "per_tensor_changed": {key: value["changed"] for key, value in changed.items()},
        "per_tensor_max_abs_delta": {key: value["max_abs_delta"] for key, value in changed.items()},
    }
    return child, record


def delta_patch_child(inst: Any, child: Mapping[str, np.ndarray], patch_dir: Path, hypothesis: str
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Write ``child - base`` as a delta patch (``write_delta_patch``), apply it within the limits, check the round
    trip, and return the applied child with its edit record."""
    from weightcode.baselines.instance_io import write_delta_patch
    from weightcode.common.identity import params_sha256

    patch_file = write_delta_patch(patch_dir, inst.params_sha256, inst.params, child, hypothesis,
                                   "mechanism analysis variant (PREREG E11 mechanism entry); not a teacher program")
    patch = json.loads(patch_file.read_text(encoding="utf-8"))
    applied, record = apply_within_limits(inst, patch, patch_file.parent)
    record["roundtrip_exact"] = record["child_sha256"] == params_sha256(child)
    if not record["roundtrip_exact"]:
        record["roundtrip_max_abs_diff"] = max(float(np.max(np.abs(applied[k].astype(np.float64)
                                                                   - np.asarray(child[k], np.float64))))
                                               for k in child)
    record["patch_sha256"] = file_sha256(patch_file)
    record["patch_form"] = "delta (write_delta_patch)"
    return applied, record


# --------------------------------------------------------------------------------------------------------------------
# Program runs in the LEARN sandbox


def sandbox() -> Any:
    from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, SANDBOX_PYTHON
    from weightcode.sandbox.program import ProgramSandbox

    return ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=DEFAULT_MEMORY_BYTES, threads=THREADS)


def run_program(inst: Any, program_file: Path, work: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Run one program file on the instance's own student and experience, as a transfer run does. Returns the update,
    its derived patch (dict, directory) and the run record."""
    from weightcode.harness.program import write_derived_patch
    from weightcode.harness.program import run_update
    from weightcode.harness.transfer import stage_student

    work.mkdir(parents=True, mode=0o700)
    student, experience_file, experience = stage_student(inst, work / "student")
    if experience["content_sha256"] != TARGETS[_TARGET_ID]["experience_content_sha256"]:
        raise MechError(f"the staged experience {experience['content_sha256']} is not the session's")
    started = time.perf_counter()
    update = run_update(student, experience_file, program_file, work / "run", sandbox())
    patch_dir = work / "patch"
    patch_dir.mkdir()
    patch = write_derived_patch(update, inst.params, inst.params_sha256, patch_dir)
    record = {
        "program": {"path": str(program_file), "sha256": file_sha256(program_file)},
        "deltas_sha256": update.record["deltas_sha256"], "tensors": list(update.tensors),
        "stdout": update.record["run"]["stdout"], "stderr": update.record["run"]["stderr"],
        "duration_s": update.record["run"]["duration_s"], "wall_s": time.perf_counter() - started,
        "experience_content_sha256": experience["content_sha256"], "sandbox": update.record["sandbox"],
        "inputs_sha256": update.record["inputs_sha256"], "work_dir": str(work),
    }
    return update, (patch, patch_dir), record


def write_program(name: str, text: str) -> Path:
    folder = work_root() / "programs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.py"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise MechError(f"{path} exists with other content")
        return path
    path.write_text(text, encoding="utf-8")
    return path


def program_diff(text: str, name: str) -> str:
    return "".join(difflib.unified_diff(final_source().splitlines(True), text.splitlines(True), fromfile=PROGRAM_FILE,
                                        tofile=name, n=1))


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise MechError(f"expected exactly one occurrence of {old[:80]!r}, found {text.count(old)}")
    return text.replace(old, new)


def _replace_strings(text: str, hypothesis: str, predicted: str) -> str:
    """Replace the module strings HYPOTHESIS and PREDICTED_EFFECT (the driver needs both non-empty)."""
    start = text.index("HYPOTHESIS = (")
    end_marker = "\nPREDICTED_EFFECT = "
    end = text.index("\n", text.index(end_marker) + 1)
    return text[:start] + f"HYPOTHESIS = {hypothesis!r}\nPREDICTED_EFFECT = {predicted!r}" + text[end:]


_CONST_ANCHOR = ("CAND_FRAC = (0.15, 0.3, 0.45)   # candidate gate blocks as fractions of depth (label-free oracle "
                 "picks one)\n")


def t0_program(block: int, strength: float) -> str:
    """T0: the final's helpers (forward, LayerNorm, GELU) and an update that moves one block's linear2 bias by
    -strength (mean block output on x_new - on x_old), over images and all 65 tokens; no gate, no unit, no label."""
    src = final_source()
    head = src[: src.index("\n\ndef _resid_var(")]
    head = _replace_strings(
        head,
        f"T0 (mechanism analysis, a priori feature alignment): subtract {strength:g} x (mean block-{block} output on "
        "x_new - mean block-{block} output on x_old) through that block's linear2 bias, at every token".replace(
            "{block}", str(block)),
        "an unconditional shift of the block output toward the clean mean")
    body = f'''

T0_BLOCK = {int(block)}
T0_STRENGTH = {float(strength)!r}


def update(student, experience):
    params = student["params"]; layout = student["layout"]
    x_old = np.asarray(experience["x_old"]); x_new = np.asarray(experience["x_new"])
    vit = _ViT(layout, params)
    ref = np.asarray(student["forward"](params, x_old[:8]), np.float64)
    if np.max(np.abs(ref - vit.run(params, x_old[:8])[0])) > 1e-3:
        raise RuntimeError("own forward does not match the reference")
    _, _, on = vit.run(params, x_new, want_out=True)
    _, _, oc = vit.run(params, x_old, want_out=True)
    off = on[T0_BLOCK].mean((0, 1)) - oc[T0_BLOCK].mean((0, 1))
    name = vit.blocks[T0_BLOCK]["linear2"]["bias"]
    print("T0 block %d strength %.2f: |offset| %.4f (all 65 tokens), patch-token |offset| %.4f, class-token |offset| %.4f"
          % (T0_BLOCK, T0_STRENGTH, np.linalg.norm(off),
             np.linalg.norm(on[T0_BLOCK][:, 1:].mean((0, 1)) - oc[T0_BLOCK][:, 1:].mean((0, 1))),
             np.linalg.norm(on[T0_BLOCK][:, 0].mean(0) - oc[T0_BLOCK][:, 0].mean(0))))
    return {{name: -T0_STRENGTH * off}}
'''
    return head + body


_T1_SIGMA_BLOCK = '''    nv = np.maximum((_resid_var(x_new) - _resid_var(x_old)) / 1.125, 0.0)
    print("estimated noise sigma per channel (uint8):", np.round(np.sqrt(nv), 2))
    if not np.any(nv > 4.0):
        print("no additive noise detected; no change"); return {}
    rng = np.random.default_rng(SEED)
    sig = np.sqrt(nv).reshape(1, -1, 1, 1)
    x_noisy = np.concatenate([np.clip(np.round(x_old.astype(np.float64) + rng.normal(0.0, 1.0, x_old.shape) * sig), 0, 255)
                              for _ in range(N_DRAWS)]).astype(np.uint8)
    x_pair = np.concatenate([x_old] * N_DRAWS)
'''
_T1_KL_BLOCK = '''    l_pair0 = vit.run(params, x_old)[0]
    l_pair = np.concatenate([l_pair0] * N_DRAWS)
    l_noisy0 = vit.run(params, x_noisy)[0]
    lg_c0 = vit.run(params, x_clean)[0]
    print("noisy KL to clean before edit: %.3f" % _kl(l_pair, l_noisy0))
'''
_T1_IMG = "img_c = np.tile(np.arange(n0), len(x_clean) // n0); img_n = np.tile(np.arange(n0), N_DRAWS)"
_T1_PRINT = '''        lg_n = vit.run(new, x_noisy)[0]; lg_c = vit.run(new, x_clean)[0]
        print("gate %-5s block %d: |offset| %.2f, held-out/clean sd %.3f, units %d,%d, kappa %.2f (ramp %.2f sd), thr %.2f sd"
              " -> noisy KL %.3f | clean KL %.5f | clean changes %d/%d"
              % (kind, b, np.linalg.norm(off), sd, j1, j2, kappa, K_RAMP / kappa / sd, t_sd, _kl(l_pair, lg_n),
                 _kl(lg_c0, lg_c), int(np.sum(lg_c.argmax(1) != lg_c0.argmax(1))), len(lg_c)))
'''
_T1_PRINT_NEW = '''        lg_c = vit.run(new, x_clean)[0]
        print("gate %-5s block %d: |offset| %.2f, held-out/clean sd %.3f, units %d,%d, kappa %.2f (ramp %.2f sd), thr %.2f sd"
              " -> clean KL %.5f | clean changes %d/%d"
              % (kind, b, np.linalg.norm(off), sd, j1, j2, kappa, K_RAMP / kappa / sd, t_sd,
                 _kl(lg_c0, lg_c), int(np.sum(lg_c.argmax(1) != lg_c0.argmax(1))), len(lg_c)))
'''
_T1_PLACEMENT_START = "    # gate 1: patch tokens after the early block where removing their mean offset helps most (label-free oracle)\n"
_T1_PLACEMENT_END = "    for k_ in params:\n"
_T1_PLACEMENT_NEW = '''    # T1: fixed placement, no KL oracle. Gate 1 on the patch tokens after PATCH_BLOCK; with CLS_GATES, a class-token gate
    # two blocks later and one in the last block, each built on the updated model as in the final program.
    oc, on = offsets(patched)
    b_patch = PATCH_BLOCK
    patched = add_gate(patched, b_patch, "patch", on[b_patch][:, 1:].mean((0, 1)) - oc[b_patch][:, 1:].mean((0, 1)), T_SD)
    if CLS_GATES:
        oc, on = offsets(patched)
        b_mid = b_patch + 2
        patched = add_gate(patched, b_mid, "cls", on[b_mid][:, 0].mean(0) - oc[b_mid][:, 0].mean(0), T_SD_MID)
        oc, on = offsets(patched)
        bl = depth - 1
        patched = add_gate(patched, bl, "cls", ALPHA_LAST * (on[bl][:, 0].mean(0) - oc[bl][:, 0].mean(0)), T_SD_CLS)

'''


def t1_program(patch_block: int, cls_gates: bool) -> str:
    """T1: the final program with x_noisy := x_new (the condition's own 64 inputs, unpaired with x_old), without the
    noise estimate, the synthetic copies and the KL oracle, and with a fixed gate placement. add_gate (direction,
    clean statistics on x_old and its flips and shifts, thresholds, units, kappa budget, fold) is unchanged."""
    text = final_source()
    text = _replace_strings(
        text,
        (f"T1 (mechanism analysis): the final program's GELU ramp gates built from x_new against x_old (unpaired), "
         f"without the noise estimate, the synthetic noisy copies and the KL oracle; patch gate after block "
         f"{patch_block}" + (f", class-token gates after blocks {patch_block + 2} and the last" if cls_gates else
                             ", no class-token gate")),
        "the gate's correction without the program's corruption model")
    text = _replace_once(text, _CONST_ANCHOR, _CONST_ANCHOR + f"PATCH_BLOCK = {int(patch_block)}\n"
                         f"CLS_GATES = {bool(cls_gates)}\n")
    text = _replace_once(text, _T1_SIGMA_BLOCK,
                         "    x_noisy = x_new   # T1: the condition's own inputs, unpaired; no noise estimate, no copies\n")
    text = _replace_once(text, _T1_KL_BLOCK, "    lg_c0 = vit.run(params, x_clean)[0]\n")
    text = _replace_once(text, _T1_IMG, "img_c = np.tile(np.arange(n0), len(x_clean) // n0); "
                         "img_n = np.arange(len(x_noisy))")
    text = _replace_once(text, _T1_PRINT, _T1_PRINT_NEW)
    start, end = text.index(_T1_PLACEMENT_START), text.index(_T1_PLACEMENT_END)
    if text.count(_T1_PLACEMENT_START) != 1 or text.count(_T1_PLACEMENT_END) != 1 or end < start:
        raise MechError("the final program's placement section is not where T1 expects it")
    return text[:start] + _T1_PLACEMENT_NEW + text[end:]


_T2_UNITS = "        j1, j2 = free[0], free[1]\n"


def t2_program(draw: int) -> str:
    """T2: the final program unchanged except that each gate's two units are drawn uniformly at random, without
    replacement, from the block's units not yet used, by numpy default_rng(draw) (separate from the noise rng)."""
    text = final_source()
    text = _replace_once(text, _CONST_ANCHOR, _CONST_ANCHOR + f"UNIT_DRAW = {int(draw)}\n"
                         "_UNIT_RNG = np.random.default_rng(UNIT_DRAW)   # T2: gate units drawn at random\n")
    return _replace_once(text, _T2_UNITS, "        j1, j2 = (int(j) for j in _UNIT_RNG.choice(np.array(sorted(free)), "
                         "size=2, replace=False))\n        print(\"T2 draw %d: units %d,%d at block %d\" % "
                         "(UNIT_DRAW, j1, j2, b))\n")


# --------------------------------------------------------------------------------------------------------------------
# The final's gates


def block_tensor(block: int, part: str) -> str:
    return f"classifier.blocks.{block}.{part}"


def gate_table(base: Mapping[str, np.ndarray], final: Mapping[str, np.ndarray]) -> dict[int, dict[str, Any]]:
    """Per gate block: the two units (from the changed linear1 rows), kappa, threshold t (in d.z units), the offset
    vector off = K * W2[:, j2] (the program writes W2[:, j1] = -off / K, W2[:, j2] = off / K), checked for form."""
    table: dict[int, dict[str, Any]] = {}
    for b in GATE_BLOCKS:
        w1, b1 = (np.asarray(final[block_tensor(b, p)], np.float64) for p in ("linear1.weight", "linear1.bias"))
        w2 = np.asarray(final[block_tensor(b, "linear2.weight")], np.float64)
        w1_0 = np.asarray(base[block_tensor(b, "linear1.weight")], np.float64)
        rows = sorted(int(r) for r in np.nonzero(np.any(w1 != w1_0, axis=1))[0])
        expected = sorted(EXPECTED_UNITS[b])
        if rows != expected:
            raise MechError(f"block {b}: changed linear1 rows {rows}, the program printed units {expected}")
        j1, j2 = EXPECTED_UNITS[b]
        if not np.allclose(w1[j1], w1[j2], rtol=0, atol=1e-6) or not np.isclose(b1[j1] - b1[j2], K_RAMP, atol=1e-4):
            raise MechError(f"block {b}: units {j1}, {j2} are not a ramp pair (same row, bias gap K)")
        if not np.allclose(w2[:, j1], -w2[:, j2], rtol=0, atol=1e-6):
            raise MechError(f"block {b}: W2 columns {j1}, {j2} are not opposite")
        kappa = float(np.linalg.norm(w1[j1]))
        off = K_RAMP * w2[:, j2]
        table[b] = {"block": b, "kind": GATE_KINDS[b], "j1": j1, "j2": j2, "kappa": kappa,
                    "threshold_dz": float(-b1[j1] / kappa), "offset": off, "offset_norm": float(np.linalg.norm(off))}
    return table


def a1_child(base: Mapping[str, np.ndarray], final: Mapping[str, np.ndarray], gates: dict[int, dict[str, Any]],
             subset: Sequence[int], variant: str) -> dict[str, np.ndarray]:
    """A1 child: ``final`` with the gates of ``subset`` off (variant "off") or their blocks reverted ("revert")."""
    child = {key: np.array(value, copy=True) for key, value in final.items()}
    for b in subset:
        if variant == "off":
            name = block_tensor(b, "linear2.weight")
            w2 = child[name]
            w2[:, gates[b]["j1"]] = 0.0
            w2[:, gates[b]["j2"]] = 0.0
        elif variant == "revert":
            for part in MLP_PARTS:
                name = block_tensor(b, part)
                child[name] = np.array(base[name], copy=True)
        else:
            raise ValueError(variant)
    return child


def t3_child(base: Mapping[str, np.ndarray], final: Mapping[str, np.ndarray], gates: dict[int, dict[str, Any]]
             ) -> dict[str, np.ndarray]:
    """T3: the final with every gate always open: the A1 all-off child (the fold kept) with linear2.bias -= off at
    each gate block, the gate's fully open output (GELU(u) - GELU(u - K) = K) applied at every token."""
    child = a1_child(base, final, gates, GATE_BLOCKS, "off")
    for b in GATE_BLOCKS:
        name = block_tensor(b, "linear2.bias")
        child[name] = (np.asarray(child[name], np.float64) - gates[b]["offset"]).astype(child[name].dtype)
    return child


# --------------------------------------------------------------------------------------------------------------------
# Subcommands


def cmd_repro(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The final's deltas re-applied (child hash) and read on V_select; the program rerun in the sandbox (deltas)."""
    inst = instance()
    from weightcode.common.identity import params_sha256, load_params

    records: list[dict[str, Any]] = []
    child = final_child()
    sha = params_sha256(child)
    saved = load_params(session_dir() / "final" / "model.npz")
    deltas = np.load(session_dir() / "final" / "deltas.npz")
    if file_sha256(session_dir() / "final" / "deltas.npz") != DELTAS_SHA256:
        raise MechError("final/deltas.npz does not hash to the program's deltas")
    rebuilt = {}
    for key, value in inst.params.items():
        if key in deltas.files:  # the derived patch's rule: float32(float64(parent) + float64(delta))
            rebuilt[key] = (np.asarray(value, np.float64) + np.asarray(deltas[key], np.float64)).astype(value.dtype)
        else:
            rebuilt[key] = np.asarray(value)
    ro = readout(inst, child)
    ok = (sha == CHILD_SHA256 and params_sha256(saved) == CHILD_SHA256 and params_sha256(rebuilt) == CHILD_SHA256
          and math.isclose(ro["gain"], points(RECORDED_DELTA_ACC["gain"]), abs_tol=1e-9)
          and math.isclose(ro["clean"], points(RECORDED_DELTA_ACC["clean"]), abs_tol=1e-9))
    records.append({"schema": SCHEMA, "analysis": "repro", "name": "final_reapplied",
                    "config": {"source": "session run.json final, devreadout.final_child (patch.json under the "
                               "session's limits); also float32(float64(base) + float64(final/deltas.npz)) and "
                               "final/model.npz"},
                    "child_sha256": sha, "model_npz_sha256": params_sha256(saved),
                    "deltas_reapplied_sha256": params_sha256(rebuilt), "expected_child_sha256": CHILD_SHA256,
                    "recorded": {"gain": points(RECORDED_DELTA_ACC["gain"]),
                                 "clean": points(RECORDED_DELTA_ACC["clean"])},
                    "reproduced": ok, **ro})
    print(f"repro reapplied: child {sha[:12]} gain {ro['gain']:+.3f} clean {ro['clean']:+.3f} joint {ro['joint']:+.3f}"
          f" reproduced {ok}", flush=True)
    if not args.skip_rerun:
        program = session_dir() / "final" / PROGRAM_FILE
        update, (patch, patch_dir), run = run_program(inst, program, work_root() / "repro" / f"rerun_{int(time.time())}")
        applied, edit = apply_within_limits(inst, patch, patch_dir)
        rerun_ok = run["deltas_sha256"] == DELTAS_SHA256 and edit["child_sha256"] == CHILD_SHA256 and edit[
            "within_limits"]
        records.append({"schema": SCHEMA, "analysis": "repro", "name": "final_program_rerun",
                        "config": {"program": str(program), "sandbox": "LEARN program sandbox, 4 threads"},
                        "run": run, "edit": edit, "expected_deltas_sha256": DELTAS_SHA256,
                        "expected_child_sha256": CHILD_SHA256, "reproduced": rerun_ok})
        print(f"repro rerun: deltas {run['deltas_sha256'][:12]} child {edit['child_sha256'][:12]} within limits "
              f"{edit['within_limits']} reproduced {rerun_ok}", flush=True)
        ok = ok and rerun_ok
    records[0]["provenance"] = provenance()
    if not ok:
        write_records("repro_FAILED" if not args.name else args.name, records)
        raise MechError("the final did not reproduce; stop")
    write_records(args.name or "repro", records)
    return records


def cmd_a1(args: argparse.Namespace) -> list[dict[str, Any]]:
    inst = instance()
    base = {k: np.asarray(v) for k, v in inst.params.items()}
    final = final_child()
    gates = gate_table(base, final)
    final_ro = readout(inst, final)
    records: list[dict[str, Any]] = [{
        "schema": SCHEMA, "analysis": "A1", "name": "all_on", "config": {"subset": [], "variant": None},
        "gates": {str(b): {k: v for k, v in g.items() if k != "offset"} for b, g in gates.items()},
        "edit": None, **final_ro, "provenance": provenance()}]
    run_work = work_root() / "a1" / f"run_{int(time.time())}"
    for variant in ("off", "revert"):
        for size in (1, 2, 3):
            for subset in itertools.combinations(GATE_BLOCKS, size):
                name = f"{variant}_{'_'.join(f'g{b}' for b in subset)}"
                child = a1_child(base, final, gates, subset, variant)
                applied, edit = delta_patch_child(inst, child, run_work / name,
                                                  f"A1 {variant} gates {list(subset)} of the E11 gaussian final")
                ro = readout(inst, applied)
                records.append({"schema": SCHEMA, "analysis": "A1", "name": name,
                                "config": {"subset": list(subset), "variant": variant,
                                           "definition": ("gate units' linear2 columns set to 0, bias fold kept"
                                                          if variant == "off" else
                                                          "the block's four MLP tensors restored to the base")},
                                "edit": edit, **ro})
                print(f"A1 {name}: gain {ro['gain']:+.3f} clean {ro['clean']:+.3f} joint {ro['joint']:+.3f} worst "
                      f"clean {ro['worst_class_clean']:+.3f} within {edit['within_limits']}", flush=True)
    write_records(args.name or "a1", records)
    return records


def _vit_module() -> types.ModuleType:
    module = types.ModuleType("e11_mech_final_program")
    exec(compile(final_source(), PROGRAM_FILE, "exec"), module.__dict__)
    return module


def cmd_a2(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Selectivity on V_select (condition rows and their clean companions, paired by idx) and residual offsets."""
    inst = instance()
    from weightcode.evalkit.companions import load_clean_rows
    from weightcode.evalkit.devreadout import readout_split

    base = {k: np.asarray(v) for k, v in inst.params.items()}
    final = final_child()
    gates = gate_table(base, final)
    cond = readout_split(inst)
    clean_rows = load_clean_rows(inst, "V_select")
    if not (np.array_equal(cond["idx"], clean_rows.idx) and np.array_equal(cond["y"], clean_rows.y)):
        raise MechError("V_select rows and their clean companion are not paired by idx")
    mod = _vit_module()
    layout = {k: v for k, v in json.loads((inst.instance_dir / "model_spec.json").read_text()).items()
              if k != "model_id"}
    vit = mod._ViT(layout, base)
    for name, params in (("base", base), ("final", final)):
        ref = np.asarray(inst.forward(dict(params), cond["x"][:64]), np.float64)
        mine = vit.run(params, cond["x"][:64])[0]
        if np.max(np.abs(ref - mine)) > 1e-6:
            raise MechError(f"the program's forward differs from forward.py on the {name} model")
    ln, gelu = mod._ln, mod._gelu
    acc: dict[str, Any] = {}

    def add(key: str, value: np.ndarray) -> None:
        acc[key] = acc.get(key, 0.0) + value

    n = len(cond["y"])
    for s0 in range(0, n, A2_CHUNK):
        xs = {"cond": cond["x"][s0:s0 + A2_CHUNK], "clean": clean_rows.x[s0:s0 + A2_CHUNK]}
        outs: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        for model_name, params in (("base", base), ("final", final)):
            for split, x in xs.items():
                h = vit.tokens(params, x)
                block_out: dict[int, np.ndarray] = {}
                for i, blk in enumerate(vit.blocks):
                    h = h + vit.attn(params, blk, ln(h, vit.T(params, blk["pre_norm"]["weight"]),
                                                     vit.T(params, blk["pre_norm"]["bias"]), vit.eps))
                    h = ln(h, vit.T(params, blk["norm1"]["weight"]), vit.T(params, blk["norm1"]["bias"]), vit.eps)
                    if model_name == "final" and i in gates:
                        g = gates[i]
                        w1 = vit.T(params, blk["linear1"]["weight"]); b1 = vit.T(params, blk["linear1"]["bias"])
                        w2 = vit.T(params, blk["linear2"]["weight"])
                        u1 = h @ w1[g["j1"]] + b1[g["j1"]]
                        u2 = h @ w1[g["j2"]] + b1[g["j2"]]
                        gate_out = gelu(u1)[..., None] * w2[:, g["j1"]] + gelu(u2)[..., None] * w2[:, g["j2"]]
                        out_norm = np.linalg.norm(gate_out, axis=-1)
                        for tok, sl in (("cls", np.s_[:, 0]), ("patch", np.s_[:, 1:])):
                            key = f"{split}_g{i}_{tok}"
                            add(key + "_n", np.asarray(np.size(u1[sl]), np.float64))
                            add(key + "_above", np.asarray(np.sum(u1[sl] > 0.0), np.float64))
                            add(key + "_full", np.asarray(np.sum(u2[sl] > 0.0), np.float64))
                            add(key + "_out_norm", np.asarray(np.sum(out_norm[sl]), np.float64))
                            add(key + "_opening", np.asarray(np.sum((gelu(u1[sl]) - gelu(u2[sl])) / K_RAMP), np.float64))
                        # per image: any patch token above threshold
                        add(f"{split}_g{i}_img_any_patch", np.asarray(np.sum(np.any(u1[:, 1:] > 0.0, axis=1)),
                                                                      np.float64))
                        add(f"{split}_g{i}_img_patch_frac_sum", np.asarray(np.sum(np.mean(u1[:, 1:] > 0.0, axis=1)),
                                                                           np.float64))
                    hid = gelu(h @ vit.T(params, blk["linear1"]["weight"]).T + vit.T(params, blk["linear1"]["bias"]))
                    h = h + hid @ vit.T(params, blk["linear2"]["weight"]).T + vit.T(params, blk["linear2"]["bias"])
                    if i in gates:
                        block_out[i] = h.copy()
                outs[(model_name, split)] = block_out
        for model_name in ("base", "final"):
            for b in GATE_BLOCKS:
                diff = outs[(model_name, "cond")][b] - outs[(model_name, "clean")][b]
                add(f"{model_name}_b{b}_cls_sum", diff[:, 0].sum(0))
                add(f"{model_name}_b{b}_patch_sum", diff[:, 1:].sum((0, 1)))
                add(f"{model_name}_b{b}_cls_pairnorm", np.asarray(np.linalg.norm(diff[:, 0], axis=-1).sum()))
                add(f"{model_name}_b{b}_patch_pairnorm", np.asarray(np.linalg.norm(diff[:, 1:], axis=-1).sum()))
        for b in GATE_BLOCKS:  # what the edit does to each split at the gate block's output
            for split in ("cond", "clean"):
                moved = outs[("final", split)][b] - outs[("base", split)][b]
                add(f"edit_{split}_b{b}_cls_norm", np.asarray(np.linalg.norm(moved[:, 0], axis=-1).sum()))
                add(f"edit_{split}_b{b}_patch_norm", np.asarray(np.linalg.norm(moved[:, 1:], axis=-1).sum()))
        print(f"A2 rows {s0 + len(xs['cond'])}/{n}", flush=True)
    n_patch = n * 64
    records: list[dict[str, Any]] = []
    for b in GATE_BLOCKS:
        g = gates[b]
        gated = "patch" if g["kind"] == "patch" else "cls"
        sel: dict[str, Any] = {}
        for split in ("cond", "clean"):
            for tok in ("cls", "patch"):
                key = f"{split}_g{b}_{tok}"
                count = float(acc[key + "_n"])
                sel[f"{split}_{tok}"] = {"n_tokens": int(count), "frac_above_threshold": float(acc[key + "_above"]) / count,
                                         "frac_fully_open": float(acc[key + "_full"]) / count,
                                         "mean_opening": float(acc[key + "_opening"]) / count,
                                         "mean_gate_output_norm": float(acc[key + "_out_norm"]) / count}
            sel[f"{split}_images_any_patch_above"] = float(acc[f"{split}_g{b}_img_any_patch"]) / n
        offsets: dict[str, Any] = {}
        for model_name in ("base", "final"):
            mean_cls = acc[f"{model_name}_b{b}_cls_sum"] / n
            mean_patch = acc[f"{model_name}_b{b}_patch_sum"] / n_patch
            offsets[model_name] = {
                "cls_mean_offset_norm": float(np.linalg.norm(mean_cls)),
                "patch_mean_offset_norm": float(np.linalg.norm(mean_patch)),
                "cls_mean_paired_norm": float(acc[f"{model_name}_b{b}_cls_pairnorm"]) / n,
                "patch_mean_paired_norm": float(acc[f"{model_name}_b{b}_patch_pairnorm"]) / n_patch,
                "cos_program_offset_" + gated: float(np.dot(mean_cls if gated == "cls" else mean_patch, g["offset"])
                                                     / (np.linalg.norm(mean_cls if gated == "cls" else mean_patch)
                                                        * g["offset_norm"])),
            }
        edit_move = {split: {"cls_mean_norm": float(acc[f"edit_{split}_b{b}_cls_norm"]) / n,
                             "patch_mean_norm": float(acc[f"edit_{split}_b{b}_patch_norm"]) / n_patch}
                     for split in ("cond", "clean")}
        records.append({"schema": SCHEMA, "analysis": "A2", "name": f"gate_b{b}",
                        "config": {"block": b, "kind": g["kind"], "gated_tokens": gated, "units": [g["j1"], g["j2"]],
                                   "threshold": "u = z . W1[j1] + b1[j1] > 0, i.e. d.z > t; fully open u > K = 3",
                                   "rows": "V_select condition rows and their clean companions, paired by idx",
                                   "offset": "block output (after the MLP), condition minus clean, paired per image",
                                   "model_for_selectivity": "final (deployed; later gates see earlier gates' output)"},
                        "gate": {k: v for k, v in g.items() if k != "offset"},
                        "selectivity": sel, "offsets": offsets, "edit_moves_block_output": edit_move,
                        "n_images": n})
        print(json.dumps(records[-1]["selectivity"], indent=None)[:600], flush=True)
    records[0]["provenance"] = provenance()
    write_records(args.name or "a2", records)
    return records


def _checked_run(method: str, label: str, variants: Sequence[tuple[str, dict[str, Any], Path]],
                 hypothesis: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """execute_run over program variants (generated before any check), then the joint-retain-v1 pick on V_select."""
    from weightcode.baselines.fit.runner import CandidatePatch, MethodSpec, execute_run
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE, readout_retention

    inst = instance()
    install_base_memo(inst)
    if len(variants) > MAX_CHECKS:
        raise MechError(f"{len(variants)} variants exceed {MAX_CHECKS} checks")
    stamp = int(time.time())
    run_dir = work_root() / "runs" / f"{method}_{stamp}"
    gen_work = work_root() / "gen" / f"{method}_{stamp}"
    with np.load(target_session_dir() / "trial" / "experience.npz", allow_pickle=False) as archive:
        x_new, x_old = archive["x_new"], archive["x_old"]

    def generate(evidence: Any) -> tuple[list[Any], dict[str, Any]]:
        k = evidence.n_deployment
        if not (np.array_equal(evidence.x[:k], x_new) and np.array_equal(evidence.x[k:], x_old)):
            raise MechError("the run's evidence rows are not the session's x_new and x_old")
        candidates = []
        runs = {}
        for slot, (name, config, program) in enumerate(variants):
            update, (patch, patch_dir), run = run_program(inst, program, gen_work / name)
            from weightcode.patch import apply_patch

            unbounded = program_limits(inst).without_edit_limits(len(inst.params))
            child = {key: np.asarray(v) for key, v in apply_patch(inst.params, patch, patch_dir, unbounded,
                                                                     inst.params_sha256).child.items()}
            runs[name] = run
            candidates.append(CandidatePatch(slot=slot, child=child, config={"name": name, **config},
                                             hypothesis=hypothesis(config),
                                             predicted_effect="mechanism analysis variant; label-free",
                                             notes={"program_sha256": run["program"]["sha256"],
                                                    "deltas_sha256": run["deltas_sha256"], "stdout": run["stdout"],
                                                    "program_diff": program_diff(program.read_text(), program.name)}))
            print(f"{method} generated {name}: {run['stdout'].strip().splitlines()[-1][:200]}", flush=True)
        return candidates, {"variants": [v[0] for v in variants], "runs": runs, "labels_read": False,
                            "inputs": "x_new and x_old of the session's experience (the program reads no label)"}

    spec = MethodSpec(method=method, label=label, uses_labels_for_fit=False, edit_limits_enforced=True)
    record = execute_run(instance=inst, spec=spec, run_dir=run_dir, run_seed=0,
                         config={"variants": [{"name": v[0], **v[1]} for v in variants],
                                 "spec": TARGETS[_TARGET_ID]["run_spec"], "kept_rule":
                                 f"joint-retain-v1, S_old >= {RETENTION_FLOOR} of 4000"},
                         generate=generate, joint=True, replay_in_fit=True, e10=True)
    checks = []
    for entry in record["candidates"]:
        item = {"patch_id": entry["patch_id"], "config": entry["config"], "invalid": entry.get("invalid"),
                "invalid_stage": entry.get("invalid_stage"), "executor_messages": entry.get("executor_messages")}
        if "check" in entry:
            c = entry["check"]
            item.update({key: c.get(key) for key in ("query_index", "score", "score_new", "score_old", "fixed_new",
                                                     "broken_new", "fixed_old", "broken_old", "n_changed")})
            item["eligible"] = int(c["score_old"]) >= RETENTION_FLOOR
        if "edit" in entry:
            edit = entry["edit"]
            item["edit"] = {"within_limits": edit["within_limits"], "violations": edit["violations"],
                            "n_changed": edit["n_changed"], "changed_frac": edit["changed_frac"],
                            "edit_size": edit["edit_size"],
                            "per_tensor_rel_frob": {k: v["rel_frob"] for k, v in edit["per_tensor"].items()
                                                    if v["changed"]},
                            "tensors_changed": sorted(k for k, v in edit["per_tensor"].items() if v["changed"])}
        checks.append(item)
    if record.get("oracle_log"):
        pick = readout_retention(run_dir / "run.json", RETENTION_FLOOR, DEFAULT_BATCH_SIZE,
                                 label="PREREG E11 mechanism: joint-retain-v1 pick (floor -20) read on V_select")
        ro = summarize_readout(pick)
        selection = pick["selection"]
        vcheck = pick["vcheck"]
    else:  # nothing checked: the base no-op
        ro = summarize_readout(readout_base_record(inst))
        selection = {"patch_id": None, "n_checks": 0, "n_eligible": 0, "rule": "base (no valid candidate)"}
        vcheck = {"score": 0, "score_new": 0, "score_old": 0}
    return {"schema": SCHEMA, "analysis": method.upper().replace("MECH_", ""), "name": f"{method}_pick",
            "target": target(),
            "config": {"variants": [v[0] for v in variants], "run_dir": str(run_dir),
                       "run_json_sha256": file_sha256(run_dir / "run.json")},
            "checks": checks, "queries_used": record["queries_used"], "n_invalid": record["n_invalid"],
            "selection": selection, "pick_vcheck": vcheck, **ro, "provenance": provenance()}


def readout_base_record(inst: Any) -> dict[str, Any]:
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
    from weightcode.harness import clean

    install_base_memo(inst)
    return clean.dev_readout_joint(inst, dict(inst.params), DEFAULT_BATCH_SIZE)


def cmd_t0(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants = []
    for block in T0_BLOCKS:
        for strength in T0_STRENGTHS:
            name = f"t0_b{block}_s{str(strength).replace('.', 'p')}"
            variants.append((name, {"block": block, "strength": strength}, write_program(name, t0_program(block,
                                                                                                         strength))))
    record = _checked_run("mech_t0", "PREREG E11 mechanism T0: a priori feature alignment through one block's "
                          "linear2 bias (mean block output on x_new minus on x_old); label-free", variants,
                          lambda c: f"T0 block {c['block']} strength {c['strength']}")
    record["analysis"] = "T0"
    write_records(args.name or "t0", [record])
    return [record]


def cmd_t1(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants = []
    for block in T1_PATCH_BLOCKS:
        for cls_gates in (False, True):
            name = f"t1_p{block}_cls{int(cls_gates)}"
            variants.append((name, {"patch_block": block, "cls_gates": cls_gates,
                                    "cls_blocks": [block + 2, 6] if cls_gates else []},
                             write_program(name, t1_program(block, cls_gates))))
    method = "mech_t1" if _TARGET_ID == INSTANCE_ID else f"mech_t1_{_TARGET_ID}"
    record = _checked_run(method, "PREREG E11 mechanism T1: the final program's gates from x_new against x_old "
                          "(unpaired), no synthetic noise, no KL oracle; label-free", variants,
                          lambda c: f"T1 patch gate at block {c['patch_block']}, class-token gates {c['cls_blocks']}")
    record["analysis"] = "T1"
    write_records(args.name or "t1", [record])
    return [record]


def cmd_t2(args: argparse.Namespace) -> list[dict[str, Any]]:
    inst = instance()
    records = []
    stamp = int(time.time())
    for draw in T2_DRAWS:
        name = f"t2_draw{draw}"
        text = t2_program(draw)
        program = write_program(name, text)
        update, (patch, patch_dir), run = run_program(inst, program, work_root() / "t2" / f"{name}_{stamp}")
        child, edit = apply_within_limits(inst, patch, patch_dir)
        ro = readout(inst, child)
        records.append({"schema": SCHEMA, "analysis": "T2", "name": name,
                        "config": {"draw": draw, "rng": f"numpy default_rng({draw}), choice of 2 among the block's "
                                   "unused units (sorted), without replacement", "checked": False},
                        "program_diff": program_diff(text, program.name), "run": run, "edit": edit, **ro})
        print(f"T2 {name}: gain {ro['gain']:+.3f} clean {ro['clean']:+.3f} joint {ro['joint']:+.3f} within "
              f"{edit['within_limits']}", flush=True)
    joints = [r["joint"] for r in records]
    records.append({"schema": SCHEMA, "analysis": "T2", "name": "t2_mean",
                    "config": {"draws": list(T2_DRAWS)},
                    "gain": float(np.mean([r["gain"] for r in records])),
                    "clean": float(np.mean([r["clean"] for r in records])), "joint": float(np.mean(joints)),
                    "joint_min": float(np.min(joints)), "joint_max": float(np.max(joints)),
                    "worst_class_clean": float(np.mean([r["worst_class_clean"] for r in records])),
                    "provenance": provenance()})
    write_records(args.name or "t2", records)
    return records


def cmd_t3(args: argparse.Namespace) -> list[dict[str, Any]]:
    inst = instance()
    base = {k: np.asarray(v) for k, v in inst.params.items()}
    final = final_child()
    gates = gate_table(base, final)
    child = t3_child(base, final, gates)
    applied, edit = delta_patch_child(inst, child, work_root() / "t3" / f"run_{int(time.time())}",
                                      "T3: the E11 gaussian final with every gate always open")
    ro = readout(inst, applied)
    record = {"schema": SCHEMA, "analysis": "T3", "name": "t3_always_open",
              "config": {"definition": "final with each gate's linear2 columns zeroed (fold kept) and linear2.bias -= "
                         "off (off = K W2[:, j2], the program's own offset) at blocks 2, 4, 6, every token",
                         "offset_norms": {str(b): g["offset_norm"] for b, g in gates.items()}, "checked": False},
              "edit": edit, **ro, "provenance": provenance()}
    print(f"T3: gain {ro['gain']:+.3f} clean {ro['clean']:+.3f} joint {ro['joint']:+.3f}", flush=True)
    write_records(args.name or "t3", [record])
    return [record]


# --------------------------------------------------------------------------------------------------------------------
# Summary and the pre-registered reading


def _load(name: str) -> list[dict[str, Any]]:
    path = out_dir() / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def cmd_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    fmt = lambda v: f"{v:+.3f}"  # noqa: E731
    repro = _load("repro")
    final_joint = None
    for r in repro:
        if r["name"] == "final_reapplied":
            final_joint = r["joint"]
            print(f"repro: gain {fmt(r['gain'])} clean {fmt(r['clean'])} joint {fmt(r['joint'])} worst clean "
                  f"{fmt(r['worst_class_clean'])} reproduced {r['reproduced']}")
        else:
            print(f"repro rerun: deltas {r['run']['deltas_sha256'][:12]} reproduced {r['reproduced']}")
    a1 = {r["name"]: r for r in _load("a1")}
    if a1:
        print("\nA1 | subset ablated | variant | gain | clean | joint | worst-class clean | within limits")
        for name, r in a1.items():
            within = None if r.get("edit") is None else r["edit"]["within_limits"]
            print(f"{name} | {r['config']['subset']} | {r['config']['variant']} | {fmt(r['gain'])} | {fmt(r['clean'])} "
                  f"| {fmt(r['joint'])} | {fmt(r['worst_class_clean'])} | {within}")
        on = a1["all_on"]
        for variant in ("off", "revert"):
            for b in GATE_BLOCKS:
                r = a1[f"{variant}_g{b}"]
                print(f"marginal g{b} ({variant}): gain {fmt(on['gain'] - r['gain'])} clean {fmt(on['clean'] - r['clean'])}"
                      f" joint {fmt(on['joint'] - r['joint'])}")
    reading = None
    t0, t1 = _load("t0"), _load("t1")
    if final_joint is not None and t0 and t1:
        bar = READING_FRACTION * final_joint
        reach = {name: (rec[0]["joint"] >= bar - 1e-9 and rec[0]["clean"] >= READING_CLEAN_FLOOR - 1e-9)
                 for name, rec in (("T0", t0), ("T1", t1))}
        if reach["T0"]:
            branch = "T0 reaches it: the gain is generic feature alignment; the teacher's claim narrows to finding it"
        elif reach["T1"]:
            branch = "T1 reaches it, T0 does not: the gate carries the gain; the corruption model is not needed"
        else:
            branch = "neither reaches it: both the inferred corruption model and the gate matter"
        reading = {"final_joint": final_joint, "bar": bar, "clean_floor": READING_CLEAN_FLOOR,
                   "T0_pick": {k: t0[0][k] for k in ("gain", "clean", "joint")},
                   "T1_pick": {k: t1[0][k] for k in ("gain", "clean", "joint")}, "reaches": reach, "branch": branch}
        print("\nreading:", json.dumps(reading, indent=1))
    return [reading] if reading else []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("repro", "a1", "a2", "t0", "t1", "t2", "t3", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--name", help="record file stem (default: the subcommand)")
        if name == "repro":
            p.add_argument("--skip-rerun", action="store_true", help="skip the sandbox rerun of the program")
        if name == "t1":
            p.add_argument("--instance", choices=sorted(TARGETS), default=INSTANCE_ID,
                           help="the instance the six T1 programs run and are checked on (A3x: i3eb3fc78)")
    args = parser.parse_args(argv)
    global _TARGET_ID
    _TARGET_ID = getattr(args, "instance", INSTANCE_ID)
    handlers = {"repro": cmd_repro, "a1": cmd_a1, "a2": cmd_a2, "t0": cmd_t0, "t1": cmd_t1, "t2": cmd_t2,
                "t3": cmd_t3, "summary": cmd_summary}
    started = time.perf_counter()
    handlers[args.command](args)
    print(f"{args.command} done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
