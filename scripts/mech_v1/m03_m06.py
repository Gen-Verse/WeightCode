"""PREREG "M03 and M06": fixed-component interventions on the E11 contrast program (seed 100, V_select, no LLM).

Spec: docs/PREREG_v0_DRAFT.md (71d1602, entry "M03 and M06, fixed-component interventions on the E11 contrast
program"); NSCC request R30-mech-m03-m06 (docs/NSCC_REQUESTS.md); background docs/MECHANISM_EVIDENCE_v1.md sections 3
and 5. Every network is built from deltas already on disk; no check is spent and nothing is selected. Labels enter only
the shared dev readout (``weightcode.harness.clean.dev_readout_joint``) and the per-image correctness that the paired
bootstrap resamples, computed from the same predictions and labels.

The contrast final s760cb52ce82fb937 (k145_gated_b2_r6_orig_targets.py) and the control (a) candidate (the same
program run unchanged on W1) each write three parts, read from their deltas:

- B: ``tokenizer.conv_layers.0.0.weight`` set to the child's value (the ungated AC boost);
- F: for the 8 block-2 units U (the rows of the child's ``linear1.weight`` delta), their ``linear2.weight`` columns set
  to 0, and ``linear2.bias`` set to the child's value (the mean fold, its only writer);
- R: on top of F, the units' ``linear1`` rows, ``linear1`` bias entries and ``linear2`` columns set to the child's
  values.

The child is the derived patch applied by the patch executor under the LEARN program limits (as control (a) applied
its candidate); every part copies the child's float32 entries, so the full combination equals the child bit for bit.

Subcommands (each writes its records into ``--out``, never overwriting a file):

- ``m03``: N0, N_B, N_F, N_FB, N_FR, N_FBR on W0; contrast_s5 V_select and its clean companion against N0; internal
  statistics; the image-level paired bootstrap.
- ``m06``: W1, W1_B', W1_F', W1_F'R', W1_B'F'R' on W1; gaussian_noise_s5, contrast_s5 and clean against W0 and W1;
  internal statistics; the fixed M06 reading.
- ``m06b``: only if the m06 reading says collision: control (a) reproduced in the program sandbox (its child must hash
  to the candidate's), then the program with units 360 and 503 removed from its candidate units, read as in m06.
- ``summary``: metrics.jsonl (one line per network), internal_stats.json, bootstrap.json, summary.json, summary.md.

Run inside the store and code layout of the NSCC job (scripts/mech_v1/job.pbs), with ``source scripts/env.sh``.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import math
import os
import re
import sys
import time
import types
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

THREADS = 4
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = str(THREADS)  # before numpy is imported, or BLAS ignores it
sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE

import numpy as np  # noqa: E402

SCHEMA = "weightcode.mech.m03_m06.v1"
SPEC = ("docs/PREREG_v0_DRAFT.md 71d1602, 'M03 and M06, fixed-component interventions on the E11 contrast program'; "
        "NSCC R30-mech-m03-m06")

CONTRAST_ID, CONTRAST = "i3eb3fc78", "contrast_s5"
GAUSSIAN_ID, GAUSSIAN = "i00802425", "gaussian_noise_s5"
W0_SHA256 = "fd7f6d4b80145c98bc7c6a9cbc2c8e440e3f45e328168df2a8b55bec3f7c5b6a"

SESSION_ID = "s760cb52ce82fb937"
PROGRAM_FILE = "k145_gated_b2_r6_orig_targets.py"
PROGRAM_SHA256 = "207b312d4295895e5fed63699f908c9560015e1a8b9ad362935181148dd8495b"
FINAL_DELTAS_SHA256 = "ae46f7551ca4c60f700622eb9c3424c655b202d04a57320f92d64815587f5e49"
FINAL_PATCH_SHA256 = "d010a09aa33d95b94d194494155d90da3541320a1074c18df79ea7d4ed9ad857"
FINAL_CHILD_SHA256 = "07943b6df1f9424be9b7c76cc32630f4c194b79e44b8ad00d9a89749aa0308e7"
FINAL_INFO_SHA256 = "c4028fbc78d527749c2eb4e0391e8838e0a10c7d53af0cce353a4f1470791816"
"""runs/devreadouts/learn/s760cb52ce82fb937/info.json: the final's V_select readouts of record."""

LOOP_STEM = "loop_checked_selection_2faae3ff4b3c_fd7f6d4b8014"
LOOP_JSON_SHA256 = "005073bdeeab8d1497322644fc8fc7f4aad49f068b3bb1a729a028a12467feec"
W1_SHA256 = "e40c20786ae91c904cfd7a195addb7e03b6601b955d731518da1ff54bc01c251"
CANDIDATE_PATCH_SHA256 = "bd821781a79004389e9490b662f138174a3f3e36e97e5a58bb9b1bdc2901e357"
CANDIDATE_CHILD_SHA256 = "c42c4e0e36e82cc8e920ba320c4291dcff0f95f2d34c6ac3868792061713185a"
CANDIDATE_DELTAS_SHA256 = "8733a3002117176777b188d3b98a83ecee505c5ee9c8e53208518c2bee1faec4"
"""round_2/candidate_1/run/out/deltas.npz of control (a)."""
ROUND2_EXPERIENCE_SHA256 = "cc3d2a729fd8635202fc89eabc328f6480745c52c940b6f44595779145aed035"
ROUND2_EVIDENCE_SHA256 = "2013de292c2a91cc67780986422b69d3851eb729eba112df8415127de97ca884"

EXPECTED_UNITS = (57, 74, 360, 368, 401, 469, 491, 503)
"""U0 = U1 as the PREREG states them; the deltas and both stdout lists must give this set."""
RANK = 6
KAPPA = 2.5
"""The program's rank and the bias offset of unit units[rank] (``head, kappa = 2.5, 2.5``)."""
G2_UNITS = (503, 360)
G2_K_RAMP = 3.0
"""The gaussian final's block-2 gate on W1 (j1, j2; MECHANISM_EVIDENCE_v1 section 1, e11_mech.EXPECTED_UNITS[2])."""
M06B_REMOVED_UNITS = (360, 503)

BOOT_SEED = 20261106
BOOT_RESAMPLES = 1000
PARITY_ATOL = 1e-5
"""Largest allowed |logit difference| between the program's float64 forward and forward.py (float32 logits)."""
ROW_ATOL = 1e-6
BIAS_GAP_ATOL = 1e-4
STDOUT_ROUND = 5e-4 + 1e-5
"""The program prints alpha and theta to 3 decimals."""
CHUNK = 250

BLOCK = 2
TOK = "tokenizer.conv_layers.0.0.weight"
L1W, L1B, L2W, L2B = (f"classifier.blocks.{BLOCK}.{part}"
                      for part in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"))
COMPONENT_TENSORS = (TOK, L1W, L1B, L2W, L2B)

M03_NETWORKS = (("N0", ""), ("N_B", "B"), ("N_F", "F"), ("N_FB", "BF"), ("N_FR", "FR"), ("N_FBR", "BFR"))
M06_NETWORKS = (("W1", ""), ("W1_B'", "B"), ("W1_F'", "F"), ("W1_F'R'", "FR"), ("W1_B'F'R'", "BFR"))
M03_CONTRASTS = {
    "N_FBR - N_FB": {"N_FBR": 1, "N_FB": -1},
    "N_FBR - N_FR": {"N_FBR": 1, "N_FR": -1},
    "I = N_FBR - N_FB - N_FR + N_F": {"N_FBR": 1, "N_FB": -1, "N_FR": -1, "N_F": 1},
}
READING_COLLISION = "collision is the main cause"
READING_BOOST = "boost is the main cause"
READING_BOTH = "both"

M06B_OLD_LINE = '    units = np.argsort(imp, kind="stable")[:rank + 2]\n'
M06B_NEW_LINES = ('    order = np.argsort(imp, kind="stable")\n'
                  '    units = order[~np.isin(order, (360, 503))][:rank + 2]\n')
M06B_PROGRAM_FILE = "k145_gated_b2_r6_orig_targets_m06b.py"


class MechError(RuntimeError):
    """An input or a rebuilt network does not match its record, or a component cannot be built as specified."""


# --------------------------------------------------------------------------------------------------------------------
# Pure parts (unit-tested in tests/test_mech_m03_m06.py)


def units_from_delta(delta_l1w: np.ndarray) -> tuple[int, ...]:
    """The freed units: the rows of the ``linear1.weight`` delta with any non-zero entry, ascending."""
    rows = np.nonzero(np.any(np.asarray(delta_l1w) != 0, axis=1))[0]
    return tuple(int(row) for row in rows)


def stdout_units(stdout: str) -> list[int]:
    """The ``units=[...]`` list the program printed, in program order (exactly one such list)."""
    found = re.findall(r"units=\[([0-9, ]*)\]", stdout)
    if len(found) != 1:
        raise MechError(f"expected one units=[...] list in the program's stdout, found {len(found)}")
    return [int(value) for value in found[0].split(",") if value.strip()]


def stdout_gate(stdout: str) -> dict[str, float]:
    """alpha and theta as the program printed them."""
    found = re.findall(r"alpha=([-0-9.]+) theta=([-0-9.]+)", stdout)
    if len(found) != 1:
        raise MechError(f"expected one 'alpha=... theta=...' in the program's stdout, found {len(found)}")
    return {"alpha": float(found[0][0]), "theta": float(found[0][1])}


def check_support(parent: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], units: Sequence[int]
                  ) -> dict[str, int]:
    """The child differs from the parent only where B, F and R write: the tokenizer weight, the units' linear1 rows
    and bias entries, their linear2 columns, and linear2.bias. Returns the changed-entry count per tensor."""
    if set(parent) != set(child):
        raise MechError(f"parent and child keys differ: {sorted(set(parent) ^ set(child))}")
    unit_set = set(int(u) for u in units)
    changed: dict[str, int] = {}
    for key in parent:
        p, c = np.asarray(parent[key]), np.asarray(child[key])
        if p.shape != c.shape or p.dtype != c.dtype:
            raise MechError(f"{key}: parent {p.dtype}{p.shape}, child {c.dtype}{c.shape}")
        diff = p != c
        if not diff.any():
            continue
        changed[key] = int(diff.sum())
        if key not in COMPONENT_TENSORS:
            raise MechError(f"the child changes {key}, which no component writes")
        if key == L1W:
            outside = sorted(set(np.nonzero(diff.any(axis=1))[0].tolist()) - unit_set)
        elif key == L1B:
            outside = sorted(set(np.nonzero(diff)[0].tolist()) - unit_set)
        elif key == L2W:
            outside = sorted(set(np.nonzero(diff.any(axis=0))[0].tolist()) - unit_set)
        else:
            outside = []
        if outside:
            raise MechError(f"{key} changes outside the units {sorted(unit_set)}: {outside[:10]}")
    return changed


def build_network(parent: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], units: Sequence[int],
                  parts: str) -> dict[str, np.ndarray]:
    """The parent with the components named in ``parts`` (a subset of "BFR"; R only on top of F) taken from the child."""
    if not set(parts) <= {"B", "F", "R"} or len(set(parts)) != len(parts):
        raise ValueError(f"parts must be a subset of 'BFR', got {parts!r}")
    if "R" in parts and "F" not in parts:
        raise ValueError("R is defined on top of F")
    index = np.asarray(sorted(int(u) for u in units), dtype=np.int64)
    net = {key: np.array(value, copy=True) for key, value in parent.items()}
    if "B" in parts:
        net[TOK] = np.array(child[TOK], copy=True)
    if "F" in parts:
        net[L2W][:, index] = 0
        net[L2B] = np.array(child[L2B], copy=True)
    if "R" in parts:
        net[L1W][index] = np.asarray(child[L1W])[index]
        net[L1B][index] = np.asarray(child[L1B])[index]
        net[L2W][:, index] = np.asarray(child[L2W])[:, index]
    return net


def identify_gate(l1w: np.ndarray, l1b: np.ndarray, units: Sequence[int], kappa: float = KAPPA,
                  row_atol: float = ROW_ATOL, gap_atol: float = BIAS_GAP_ATOL) -> dict[str, Any]:
    """The gate unit from the child's weights. In the program, units[rank] and units[rank + 1] carry the same linear1
    row alpha * g; the gate unit has bias -alpha * theta and units[rank] has bias -alpha * theta + kappa. Exactly one
    pair of the units may share a row; the gate is its member with the lower bias, and the gap must be kappa."""
    rows = {int(u): np.asarray(l1w[int(u)], dtype=np.float64) for u in units}
    ordered = sorted(rows)
    pairs = [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]
             if np.allclose(rows[a], rows[b], rtol=0.0, atol=row_atol)]
    if len(pairs) != 1:
        raise MechError(f"expected exactly one pair of units with the same linear1 row, found {pairs}")
    a, b = pairs[0]
    gate, kappa_unit = (a, b) if float(l1b[a]) < float(l1b[b]) else (b, a)
    gap = float(l1b[kappa_unit]) - float(l1b[gate])
    if not math.isclose(gap, kappa, rel_tol=0.0, abs_tol=gap_atol):
        raise MechError(f"units {gate}, {kappa_unit}: bias gap {gap}, expected kappa {kappa}")
    alpha = float(np.linalg.norm(rows[gate]))
    return {"gate_unit": gate, "kappa_unit": kappa_unit, "bias_gap": gap,
            "row_max_abs_diff": float(np.max(np.abs(rows[gate] - rows[kappa_unit]))),
            "alpha": alpha, "theta": -float(l1b[gate]) / alpha, "gate_bias": float(l1b[gate])}


def check_gate_against_stdout(gate: Mapping[str, Any], printed_units: Sequence[int], stdout: str,
                              rank: int = RANK) -> dict[str, Any]:
    """The gate found in the weights is units[rank + 1] of the printed list, the kappa unit units[rank], and alpha and
    theta match the printed values to their rounding."""
    printed = stdout_gate(stdout)
    checks = {
        "gate_is_units[rank+1]": printed_units[rank + 1] == gate["gate_unit"],
        "kappa_unit_is_units[rank]": printed_units[rank] == gate["kappa_unit"],
        "alpha_matches_print": abs(gate["alpha"] - printed["alpha"]) <= STDOUT_ROUND,
        "theta_matches_print": abs(gate["theta"] - printed["theta"]) <= STDOUT_ROUND,
    }
    if not all(checks.values()):
        raise MechError(f"the gate from the weights {dict(gate)} does not match the printed program state "
                        f"{printed} units {list(printed_units)}: {checks}")
    return {"printed": printed, "printed_units": list(printed_units), "checks": checks}


def m06_reading(loss_f: float, loss_b: float, loss_star: float) -> dict[str, Any]:
    """The PREREG's fixed M06 reading. L(X) is the gaussian loss of X against W1 in points of gaussian V_select gain;
    loss_star = L(W1_B'F'R'). Collision when L(F') >= 0.5 L* and L(F') >= L(B'); boost when L(B') >= 0.5 L* and
    L(B') >= L(F'); otherwise both. Both conditions can hold only at L(F') = L(B'); the record flags that tie, and the
    collision label (stated first in the PREREG) is kept, since its condition holds as written."""
    collision = loss_f >= 0.5 * loss_star and loss_f >= loss_b
    boost = loss_b >= 0.5 * loss_star and loss_b >= loss_f
    if collision:
        reading = READING_COLLISION
    elif boost:
        reading = READING_BOOST
    else:
        reading = READING_BOTH
    shares = None if loss_star == 0 else {"F'": loss_f / loss_star, "B'": loss_b / loss_star}
    return {"reading": reading, "collision_condition": bool(collision), "boost_condition": bool(boost),
            "tie_both_conditions_hold": bool(collision and boost), "L": {"W1_F'": loss_f, "W1_B'": loss_b,
                                                                         "W1_B'F'R' (L*)": loss_star},
            "shares_of_L*": shares, "L*_positive": bool(loss_star > 0),
            "rule": ("collision if L(F') >= 0.5 L* and L(F') >= L(B'); boost if L(B') >= 0.5 L* and L(B') >= "
                     "L(F'); otherwise both; shares need not add to 1")}


def paired_bootstrap(correct: Mapping[str, Mapping[str, np.ndarray]], contrasts: Mapping[str, Mapping[str, int]],
                     metrics: Sequence[str], seed: int = BOOT_SEED, n_resamples: int = BOOT_RESAMPLES
                     ) -> dict[str, Any]:
    """Image-level paired bootstrap of linear contrasts of per-image correctness, in points.

    ``correct[network][metric]`` is a 0/1 vector over the same images in the same order for every network and metric
    (the condition row and its clean companion of one image share an index). One index matrix of shape
    (n_resamples, n) is drawn once from ``default_rng(seed)`` and serves every contrast and metric. Each contrast's
    coefficients sum to 0, so the parent's accuracy cancels and the statistic is the contrast of the scores."""
    lengths = {len(np.asarray(v)) for per in correct.values() for v in per.values()}
    if len(lengths) != 1:
        raise MechError(f"correctness vectors of unequal length: {sorted(lengths)}")
    n = lengths.pop()
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(n_resamples, n))
    result: dict[str, Any] = {"seed": seed, "generator": "numpy default_rng", "n_resamples": n_resamples,
                              "n_images": n, "interval": "percentile 2.5 and 97.5 (numpy linear)",
                              "resample_unit": "image; one index matrix for every contrast and metric",
                              "covers": "V_select sampling only, not program generation", "contrasts": {}}
    for name, coefs in contrasts.items():
        if sum(coefs.values()) != 0:
            raise MechError(f"contrast {name}: coefficients sum to {sum(coefs.values())}, not 0")
        per_metric: dict[str, Any] = {}
        for metric in metrics:
            values = sum(int(coef) * np.asarray(correct[net][metric]).astype(np.int64) for net, coef in coefs.items())
            boot = 100.0 * values[index].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5])
            per_metric[metric] = {"point": 100.0 * float(values.mean()), "boot_mean": float(boot.mean()),
                                  "boot_sd": float(boot.std(ddof=1)), "ci95": [float(low), float(high)],
                                  "share_of_resamples_le_0": float(np.mean(boot <= 0.0))}
        result["contrasts"][name] = {"coefficients": dict(coefs), **per_metric}
    return result


def token_spread(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per image, from the tokenizer output t [n, tokens, dim]: (a) the token std, the RMS over channels of the std
    over the image's tokens (the spread of its tokens around its mean token); (b) the std over all its entries."""
    t = np.asarray(t, dtype=np.float64)
    spread = np.sqrt(t.var(axis=1).mean(axis=1))
    std_all = t.reshape(len(t), -1).std(axis=1)
    return spread, std_all


def ratio_summary(cond: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """Condition over clean, paired per image."""
    ratio = np.asarray(cond, np.float64) / np.asarray(clean, np.float64)
    return {"mean_of_image_ratios": float(ratio.mean()), "median_of_image_ratios": float(np.median(ratio)),
            "ratio_of_means": float(np.mean(cond) / np.mean(clean)), "mean_condition": float(np.mean(cond)),
            "mean_clean": float(np.mean(clean))}


def loss_against_w1(readout_vs_w1: Mapping[str, Any]) -> float:
    """L(X): the gaussian loss of X against W1, in points of gaussian V_select gain (minus the change against W1)."""
    return -float(readout_vs_w1[GAUSSIAN]["delta"])


def loss_rows_against_w1(readout_vs_w1: Mapping[str, Any]) -> int:
    """L(X) in rows: broken minus fixed on gaussian V_select against W1 (exact; points = 100 * rows / n)."""
    return int(readout_vs_w1[GAUSSIAN]["broken"]) - int(readout_vs_w1[GAUSSIAN]["fixed"])


# --------------------------------------------------------------------------------------------------------------------
# Inputs and identity


def runs_root() -> Path:
    from weightcode.common.paths import runs_dir

    return Path(runs_dir())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise MechError(f"{path} hashes to {actual}, the record says {expected}")
    return actual


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    from weightcode.baselines.fit.runner import jsonable as convert

    return convert(value)


def write_json(path: Path, data: Any) -> Path:
    if path.exists():
        raise MechError(f"{path} exists; records are never overwritten")
    path.write_text(json.dumps(jsonable(data), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {path} sha256 {file_sha256(path)}", flush=True)
    return path


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if path.exists():
        raise MechError(f"{path} exists; records are never overwritten")
    path.write_text("".join(json.dumps(jsonable(r), sort_keys=True, allow_nan=False) + "\n" for r in rows),
                    encoding="utf-8")
    print(f"wrote {path} sha256 {file_sha256(path)}", flush=True)
    return path


def provenance() -> dict[str, Any]:
    from weightcode.baselines.fit.runner import git_state

    return {"script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
            "git": git_state(), "threads": THREADS, "numpy": np.__version__, "python": sys.version.split()[0],
            "created_at": utc_now(), "spec": SPEC, "host": os.uname().nodename,
            "pbs_jobid": os.environ.get("PBS_JOBID")}


def load_dev_instance(instance_id: str, condition: str) -> Any:
    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import instances_root

    inst = load_instance(instances_root() / instance_id)
    if inst.private.get("split_role") != "dev" or inst.private.get("condition") != condition:
        raise MechError(f"{instance_id} is not the dev {condition} instance")
    if inst.params_sha256 != W0_SHA256:
        raise MechError(f"{instance_id}: base {inst.params_sha256} is not the seed-100 student W0 {W0_SHA256}")
    return inst


def program_limits(inst: Any, params: Mapping[str, np.ndarray]) -> Any:
    """The LEARN program limits a derived patch is applied under (``loop_stream.select_round``)."""
    from weightcode.harness.arms import applied_limits
    from weightcode.harness.transfer import LEARN_ARM
    from weightcode.patch.limits import dense_delta_bytes

    return applied_limits(inst.task["limits"], LEARN_ARM, len(params),
                          dense_delta_bytes(np.asarray(v).shape for v in params.values()))


def e10_limits_equal(inst: Any, params: Mapping[str, np.ndarray], limits: Any) -> bool:
    """Whether the LEARN program limits equal the E10 program limits the E11 session applied (informational)."""
    from weightcode.evalkit.e10 import e10_program_limits
    from weightcode.patch.limits import dense_delta_bytes

    return e10_program_limits(inst.task["limits"], len(params),
                              dense_delta_bytes(np.asarray(v).shape for v in params.values())) == limits


def apply_derived_patch(parent: Mapping[str, np.ndarray], patch_file: Path, limits: Any, parent_sha: str
                        ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply a derived patch with the patch executor; returns a writable copy of the child and the edit record."""
    from weightcode.patch import apply_patch, task_limits

    patch = json.loads(patch_file.read_text(encoding="utf-8"))
    if patch.get("parent_sha256") != parent_sha:
        raise MechError(f"{patch_file}: parent {patch.get('parent_sha256')}, expected {parent_sha}")
    result = apply_patch(parent, patch, patch_file.parent, limits, parent_sha)
    child = {key: np.array(value, copy=True) for key, value in result.child.items()}
    return child, {"child_sha256": result.child_sha256, "n_changed": result.n_changed,
                   "changed_frac": result.changed_frac, "limits": task_limits(limits),
                   "per_tensor_changed": {k: v["changed"] for k, v in result.per_tensor.items() if v["changed"]},
                   "per_tensor_rel_frob": {k: v["rel_frob"] for k, v in result.per_tensor.items() if v["changed"]}}


def patch_deltas(patch_file: Path) -> dict[str, np.ndarray]:
    patch = json.loads(patch_file.read_text(encoding="utf-8"))
    deltas: dict[str, np.ndarray] = {}
    for op in patch["ops"]:
        if op["op"] != "add_dense":
            raise MechError(f"{patch_file}: unexpected op {op['op']}")
        deltas[op["tensor"]] = np.load(patch_file.parent / op["delta"], allow_pickle=False)
    if tuple(sorted(deltas)) != tuple(sorted(COMPONENT_TENSORS)):
        raise MechError(f"{patch_file}: tensors {sorted(deltas)}, expected {sorted(COMPONENT_TENSORS)}")
    return deltas


def final_inputs() -> dict[str, Any]:
    """The contrast final's files, each checked against its record."""
    folder = runs_root() / "sessions" / SESSION_ID / "final"
    require_sha(folder / PROGRAM_FILE, PROGRAM_SHA256)
    require_sha(folder / "deltas.npz", FINAL_DELTAS_SHA256)
    require_sha(folder / "patch.json", FINAL_PATCH_SHA256)
    deltas = patch_deltas(folder / "patch.json")
    with np.load(folder / "deltas.npz", allow_pickle=False) as archive:
        npz = {key: archive[key] for key in archive.files}
    if set(npz) != set(deltas) or any(not np.array_equal(npz[k], deltas[k]) or npz[k].dtype != deltas[k].dtype
                                      for k in deltas):
        raise MechError("final/deltas.npz and the patch's delta files differ")
    run = json.loads((folder / "program_run.json").read_text(encoding="utf-8"))
    if run["deltas_sha256"] != FINAL_DELTAS_SHA256 or run["program_sha256"] != PROGRAM_SHA256:
        raise MechError("final/program_run.json names other deltas or another program")
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    if selection["final"]["child_sha256"] != FINAL_CHILD_SHA256:
        raise MechError("final/selection.json names another child")
    info_path = runs_root() / "devreadouts" / "learn" / SESSION_ID / "info.json"
    require_sha(info_path, FINAL_INFO_SHA256)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["final"]["child_sha256"] != FINAL_CHILD_SHA256:
        raise MechError("info.json's final is another child")
    from weightcode.common.identity import load_params, params_sha256

    return {"folder": folder, "deltas": deltas, "stdout": run["run"]["stdout"],
            "program_text": (folder / PROGRAM_FILE).read_text(encoding="utf-8"),
            "model_npz_sha256": params_sha256(load_params(folder / "model.npz")),
            "record_select": info["final"]["vselect"], "record_clean": info["final"]["vselect_clean"]}


def loop_paths() -> dict[str, Path]:
    root = runs_root() / "loop_streams"
    work = root / LOOP_STEM
    cand = work / "round_2" / "candidate_1"
    return {"json": root / f"{LOOP_STEM}.json", "w1": work / "student_after_round_1.npz",
            "evidence2": work / "evidence_round_2.npz", "patch": cand / "patch" / "derived_patch.json",
            "stdout": cand / "run" / "stdout.txt", "deltas": cand / "run" / "out" / "deltas.npz",
            "experience": cand / "student" / "experience.npz"}


def candidate_inputs() -> dict[str, Any]:
    """Control (a)'s round-2 candidate (the contrast program run unchanged on W1), checked against its record."""
    from weightcode.common.identity import load_params, params_sha256

    paths = loop_paths()
    require_sha(paths["json"], LOOP_JSON_SHA256)
    require_sha(paths["patch"], CANDIDATE_PATCH_SHA256)
    record = json.loads(paths["json"].read_text(encoding="utf-8"))
    round1, round2 = record["rounds"][0], record["rounds"][1]
    candidates = round2["candidates"]
    if len(candidates) != 1 or candidates[0]["name"] != "candidate_1" or candidates[0]["status"] != "checked":
        raise MechError("control (a) round 2 does not hold the one checked candidate_1")
    candidate = candidates[0]
    if candidate["child_sha256"] != CANDIDATE_CHILD_SHA256 or candidate["program"]["sha256"] != PROGRAM_SHA256:
        raise MechError("candidate_1 is not the recorded child of the contrast program")
    if not (round1["student_after"]["sha256"] == round2["parent_sha256"] == W1_SHA256):
        raise MechError("round 2's parent is not W1")
    w1 = load_params(paths["w1"])
    if params_sha256(w1) != W1_SHA256:
        raise MechError(f"{paths['w1']} does not hash to W1")
    return {"record": record, "round1": round1, "round2": round2, "candidate": candidate, "w1": w1,
            "deltas": patch_deltas(paths["patch"]), "stdout": paths["stdout"].read_text(encoding="utf-8"),
            "paths": paths}


def unit_checks(deltas: Mapping[str, np.ndarray], stdout: str) -> dict[str, Any]:
    """U from the linear1.weight delta, checked against the printed list and the PREREG's set."""
    units = units_from_delta(deltas[L1W])
    printed = stdout_units(stdout)
    if tuple(sorted(printed)) != units or units != EXPECTED_UNITS:
        raise MechError(f"units from the delta {units}, printed {printed}, PREREG {EXPECTED_UNITS}")
    return {"units": list(units), "printed_units_program_order": printed, "prereg_units": list(EXPECTED_UNITS)}


def holds_w1_g2(params: Mapping[str, np.ndarray], w1: Mapping[str, np.ndarray]) -> bool:
    """Units 503 and 360 of ``params`` hold W1's values (linear1 rows and bias entries, linear2 columns)."""
    return all(np.array_equal(np.asarray(params[key])[..., u] if key == L2W else np.asarray(params[key])[u],
                              np.asarray(w1[key])[..., u] if key == L2W else np.asarray(w1[key])[u])
               for key in (L1W, L1B, L2W) for u in G2_UNITS)


def g2_structure(params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The gaussian block-2 gate on W1: units j1 = 503, j2 = 360 share a linear1 row, their biases differ by K = 3 and
    their linear2 columns are opposite (``e11_mech.gate_table``)."""
    j1, j2 = G2_UNITS
    w1 = np.asarray(params[L1W], np.float64)
    b1 = np.asarray(params[L1B], np.float64)
    w2 = np.asarray(params[L2W], np.float64)
    ok = (np.allclose(w1[j1], w1[j2], rtol=0, atol=ROW_ATOL) and math.isclose(b1[j1] - b1[j2], G2_K_RAMP, abs_tol=1e-4)
          and np.allclose(w2[:, j1], -w2[:, j2], rtol=0, atol=ROW_ATOL))
    return {"j1": j1, "j2": j2, "is_ramp_pair": bool(ok), "bias_gap": float(b1[j1] - b1[j2])}


# --------------------------------------------------------------------------------------------------------------------
# Readouts


_PRED: dict[tuple[str, str, int], np.ndarray] = {}


def install_memo() -> None:
    """Memoize ``weightcode.evalkit.devreadout.predict`` on (params hash, rows hash, batch size): the same function on the
    same inputs, so no number changes. All instances read here share one forward.py (checked by the callers)."""
    import weightcode.evalkit.devreadout as devreadout
    from weightcode.common.identity import params_sha256

    if getattr(devreadout.predict, "_mech_memo", False):
        return
    original = devreadout.predict

    def predict(forward: Any, params: Mapping[str, np.ndarray], x: np.ndarray, batch_size: int) -> np.ndarray:
        key = (params_sha256(params), hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), int(batch_size))
        if key not in _PRED:
            _PRED[key] = original(forward, params, x, batch_size)
        return _PRED[key].copy()

    predict._mech_memo = True  # type: ignore[attr-defined]
    devreadout.predict = predict


def _points(value: float) -> float:
    return 100.0 * float(value)


def summarize_split(readout: Mapping[str, Any]) -> dict[str, Any]:
    return {"delta": _points(readout["delta_acc"]), "fixed": int(readout["fixed"]), "broken": int(readout["broken"]),
            "worst_class": _points(readout["worst_class_delta"]), "base_acc": float(readout["base_acc"]),
            "child_acc": float(readout["child_acc"]), "n": int(readout["n"]),
            "per_class": [_points(v) for v in readout["per_class_acc_delta"]]}


def readout(inst: Any, child: Mapping[str, np.ndarray], parent: Mapping[str, np.ndarray] | None
            ) -> tuple[dict[str, Any], dict[str, Any], str]:
    """``dev_readout_joint`` of ``child`` on the instance's V_select and its clean companion, against the instance's
    base (``parent`` None) or against ``parent`` (the same function with the base replaced)."""
    from weightcode.common.identity import params_sha256
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
    from weightcode.harness import clean

    install_memo()
    ref = inst if parent is None else dataclasses.replace(inst, params=dict(parent),
                                                          params_sha256=params_sha256(parent))
    record = clean.dev_readout_joint(ref, child, DEFAULT_BATCH_SIZE)
    return (summarize_split(record["readouts"]), summarize_split(record["readouts_clean"]),
            str(record["clean_split_sha256"]))


def per_image_correct(inst: Any, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """0/1 correctness per V_select row and per clean companion row, from the memoized predictions of the readout."""
    import weightcode.evalkit.devreadout as devreadout
    from weightcode.evalkit.companions import load_clean_rows

    install_memo()
    cond = devreadout.readout_split(inst)
    clean_rows = load_clean_rows(inst, devreadout.READOUT_SPLIT)
    if not (np.array_equal(cond["idx"], clean_rows.idx) and np.array_equal(cond["y"], clean_rows.y)):
        raise MechError(f"{inst.instance_id}: V_select rows and their clean companion are not paired by idx")
    batch = devreadout.DEFAULT_BATCH_SIZE
    return {"gain": devreadout.predict(inst.forward, params, cond["x"], batch) == cond["y"],
            "clean": devreadout.predict(inst.forward, params, clean_rows.x, batch) == clean_rows.y}


def compare_record(name: str, ours: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Our readout fields against a record's (delta_acc, fixed, broken, worst_class_delta, base_acc, child_acc)."""
    pairs = {"delta_acc": (ours["delta"] / 100.0, record["delta_acc"]), "fixed": (ours["fixed"], record["fixed"]),
             "broken": (ours["broken"], record["broken"]),
             "worst_class_delta": (ours["worst_class"] / 100.0, record["worst_class_delta"]),
             "base_acc": (ours["base_acc"], record["base_acc"]), "child_acc": (ours["child_acc"], record["child_acc"])}
    diffs = {key: [a, b] for key, (a, b) in pairs.items() if not math.isclose(float(a), float(b), abs_tol=1e-12)}
    return {"name": name, "equal": not diffs, "differences": diffs}


def m03_record_checks(name: str, cond: Mapping[str, Any], clean_ro: Mapping[str, Any], fin: Mapping[str, Any]
                      ) -> list[dict[str, Any]]:
    """N0: the base accuracies of record (contrast V_select and clean); N_FBR: the final's readouts of record."""
    if name == "N0":
        return [{"name": "N0 base accuracies", "equal": (
            math.isclose(cond["base_acc"], fin["record_select"]["base_acc"], abs_tol=1e-12)
            and math.isclose(clean_ro["base_acc"], fin["record_clean"]["base_acc"], abs_tol=1e-12)
            and cond["delta"] == 0.0 and clean_ro["delta"] == 0.0),
            "ours": [cond["base_acc"], clean_ro["base_acc"]],
            "record": [fin["record_select"]["base_acc"], fin["record_clean"]["base_acc"]]}]
    if name == "N_FBR":
        return [compare_record("N_FBR = final, contrast V_select", cond, fin["record_select"]),
                compare_record("N_FBR = final, clean V_select", clean_ro, fin["record_clean"])]
    return []


def w1_record_checks(line: Mapping[str, Any], insts: Mapping[str, Any], recorded: Mapping[str, Any]
                     ) -> list[dict[str, Any]]:
    """W1 against W0 on each condition and clean: the control (a) record's round-1 readouts."""
    checks = []
    for condition, inst in insts.items():
        rec = recorded[inst.instance_id]
        checks.append(compare_record(f"W1, {condition} V_select", line["readouts"]["vs_W0"][condition], rec["select"]))
        checks.append(compare_record(f"W1, clean V_select ({inst.instance_id})", line["readouts"]["vs_W0"]["clean"],
                                     rec["clean"]))
        checks.append({"name": f"clean split sha ({inst.instance_id})",
                       "equal": rec["clean_split_sha256"] == line["clean_split_sha256"]})
    return checks


# --------------------------------------------------------------------------------------------------------------------
# Internal statistics (the program's own float64 forward, checked against forward.py)


def program_module(text: str) -> types.ModuleType:
    module = types.ModuleType("mech_contrast_program")
    exec(compile(text, PROGRAM_FILE, "exec"), module.__dict__)
    return module


def layout_of(inst: Any) -> dict[str, Any]:
    spec = json.loads((inst.instance_dir / "model_spec.json").read_text(encoding="utf-8"))
    return {key: value for key, value in spec.items() if key != "model_id"}


class Internal:
    """Block-2 statistics of one network, computed with the contrast program's own forward (``_ViT``)."""

    def __init__(self, mod: types.ModuleType, layout: Mapping[str, Any], params: Mapping[str, np.ndarray]) -> None:
        self.mod = mod
        self.arch = mod._parse(layout, params)
        if self.arch is None:
            raise MechError("the program's layout parser does not recognise the student")
        self.vit = mod._ViT(params, layout, self.arch)

    def logits(self, x: np.ndarray) -> np.ndarray:
        vit, arch, mod = self.vit, self.arch, self.mod
        h = vit.tokens(x)
        for blk in arch["blocks"]:
            _, h = vit.block_parts(h, blk)
        f = mod._ln(h, vit.p[arch["norm"]["weight"]], vit.p[arch["norm"]["bias"]], vit.eps)[:, 0]
        return f @ vit.p[arch["head"]["weight"]].T + vit.p[arch["head"]["bias"]]

    def block2(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The tokenizer output t [n, 64, D] (patch projection, before the class token and positions) and the block-2
        MLP input x2 [n, 65, D] (the norm1 output), float64."""
        vit, arch, mod = self.vit, self.arch, self.mod
        z = vit.norm_img(x)
        n, c, height, width = z.shape
        p = vit.P
        g = z.reshape(n, c, height // p, p, width // p, p).transpose(0, 2, 4, 1, 3, 5).reshape(
            n, (height // p) * (width // p), c * p * p)
        weight, bias = vit.p[arch["patch"]["weight"]], vit.p[arch["patch"]["bias"]]
        t = g @ weight.reshape(weight.shape[0], -1).T + bias
        h = vit.tokens(x)
        for i in range(BLOCK):
            _, h = vit.block_parts(h, arch["blocks"][i])
        blk = arch["blocks"][BLOCK]
        h = h + vit.attn(mod._ln(h, vit.p[blk["pre_norm"]["weight"]], vit.p[blk["pre_norm"]["bias"]], vit.eps), blk)
        x2 = mod._ln(h, vit.p[blk["norm1"]["weight"]], vit.p[blk["norm1"]["bias"]], vit.eps)
        return t, x2


def forward_parity(internal: Internal, inst: Any, params: Mapping[str, np.ndarray], x: np.ndarray) -> float:
    ref = np.asarray(inst.forward(dict(params), x), np.float64)
    diff = float(np.max(np.abs(internal.logits(x) - ref)))
    if diff > PARITY_ATOL:
        raise MechError(f"the program's forward differs from forward.py by {diff} (> {PARITY_ATOL})")
    return diff


def internal_stats(internal: Internal, x: np.ndarray, units: Sequence[int], gate_unit: int | None,
                   g2_unit: int | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """On one image set: the gate opening (pre-activation of the gate unit > 0) on patch tokens (denominator tokens)
    with per-image shares and on the class token; the mean norm of the 8 units' summed output; per-image token spreads
    of the tokenizer output (returned for the paired ratio); and, where given, the gaussian G2 gate opening (unit j1's
    pre-activation > 0, the A2 definition)."""
    vit, mod = internal.vit, internal.mod
    index = np.asarray(sorted(int(u) for u in units), dtype=np.int64)
    w1, b1, w2 = vit.p[L1W], vit.p[L1B], vit.p[L2W]
    n = len(x)
    acc = {"gate_open_patch": 0, "gate_open_cls": 0, "gate_img_any": 0, "gate_img_majority": 0,
           "g2_open_patch": 0, "g2_open_cls": 0, "g2_img_any": 0, "norm_patch": 0.0, "norm_cls": 0.0}
    spreads, stds = [], []
    n_patch_tokens = None
    for start in range(0, n, CHUNK):
        t, x2 = internal.block2(x[start:start + CHUNK])
        spread, std_all = token_spread(t)
        spreads.append(spread)
        stds.append(std_all)
        n_patch_tokens = x2.shape[1] - 1
        pre = x2 @ w1[index].T + b1[index]
        out = mod._gelu(pre) @ w2[:, index].T
        norm = np.linalg.norm(out, axis=-1)
        acc["norm_patch"] += float(norm[:, 1:].sum())
        acc["norm_cls"] += float(norm[:, 0].sum())
        for key, unit in (("gate", gate_unit), ("g2", g2_unit)):
            if unit is None:
                continue
            opened = (x2 @ w1[unit] + b1[unit]) > 0.0
            acc[f"{key}_open_patch"] += int(opened[:, 1:].sum())
            acc[f"{key}_open_cls"] += int(opened[:, 0].sum())
            acc[f"{key}_img_any"] += int(np.any(opened[:, 1:], axis=1).sum())
            if key == "gate":
                acc["gate_img_majority"] += int((opened[:, 1:].mean(axis=1) > 0.5).sum())
    total_patch = n * int(n_patch_tokens)
    stats: dict[str, Any] = {
        "n_images": n, "n_patch_tokens": total_patch,
        "units_output_norm": {"patch_token_mean": acc["norm_patch"] / total_patch, "cls_token_mean": acc["norm_cls"] / n,
                              "all_token_mean": (acc["norm_patch"] + acc["norm_cls"]) / (total_patch + n)},
        "gate": None if gate_unit is None else {
            "unit": gate_unit, "open_patch_tokens": acc["gate_open_patch"],
            "rate_patch_tokens": acc["gate_open_patch"] / total_patch,
            "share_images_any_patch_token_open": acc["gate_img_any"] / n,
            "share_images_majority_patch_tokens_open": acc["gate_img_majority"] / n,
            "rate_cls_token": acc["gate_open_cls"] / n},
        "g2_gate_extra": None if g2_unit is None else {
            "unit": g2_unit, "rate_patch_tokens": acc["g2_open_patch"] / total_patch,
            "share_images_any_patch_token_open": acc["g2_img_any"] / n, "rate_cls_token": acc["g2_open_cls"] / n},
    }
    return stats, {"spread": np.concatenate(spreads), "std_all": np.concatenate(stds)}


def network_internal(mod: types.ModuleType, layout: Mapping[str, Any], inst: Any, params: Mapping[str, np.ndarray],
                     sets: Mapping[str, np.ndarray], units: Sequence[int], gate_unit: int | None,
                     g2_unit: int | None) -> dict[str, Any]:
    """Internal statistics of one network on each image set; the tokenizer token spread as condition over clean."""
    internal = Internal(mod, layout, params)
    parity = forward_parity(internal, inst, params, sets["clean"][:64])
    per_set: dict[str, Any] = {}
    spreads: dict[str, dict[str, np.ndarray]] = {}
    for name, x in sets.items():
        per_set[name], spreads[name] = internal_stats(internal, x, units, gate_unit, g2_unit)
    tokenizer = {name: {"token_spread": ratio_summary(spreads[name]["spread"], spreads["clean"]["spread"]),
                        "std_all_entries": ratio_summary(spreads[name]["std_all"], spreads["clean"]["std_all"])}
                 for name in sets if name != "clean"}
    return {"forward_parity_max_abs_logit_diff": parity, "sets": per_set,
            "tokenizer_condition_over_clean": tokenizer,
            "definitions": {
                "gate_open": "block-2 gate unit pre-activation (norm1 output . linear1 row + bias) > 0",
                "units_output_norm": "|| sum over the 8 units of GELU(pre-activation) * linear2 column ||, per token",
                "token_spread": ("per image: sqrt(mean over channels of the variance over the image's 64 tokenizer "
                                 "output tokens); std_all_entries: std over all 64 x 256 entries"),
                "pairing": "condition row i and clean companion row i are one image (idx checked)"}}


def image_sets(instances: Sequence[Any]) -> dict[str, np.ndarray]:
    """V_select inputs of each instance's condition, plus their shared clean companion (no labels kept)."""
    import weightcode.evalkit.devreadout as devreadout
    from weightcode.evalkit.companions import load_clean_rows

    sets: dict[str, np.ndarray] = {}
    clean_x = None
    for inst in instances:
        cond = devreadout.readout_split(inst)
        rows = load_clean_rows(inst, devreadout.READOUT_SPLIT)
        if not np.array_equal(cond["idx"], rows.idx):
            raise MechError(f"{inst.instance_id}: V_select rows and their clean companion are not paired")
        if clean_x is not None and not np.array_equal(clean_x, rows.x):
            raise MechError("the instances' clean companions differ")
        clean_x = rows.x
        sets[str(inst.private["condition"])] = cond["x"]
    sets["clean"] = clean_x
    return sets


# --------------------------------------------------------------------------------------------------------------------
# Subcommands


def out_dir(args: argparse.Namespace) -> Path:
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    return path


def network_line(experiment: str, name: str, parts: str, net: Mapping[str, np.ndarray],
                 parent: Mapping[str, np.ndarray], parent_label: str) -> dict[str, Any]:
    from weightcode.common.identity import count_changed, params_sha256

    return {"schema": SCHEMA, "experiment": experiment, "network": name, "parts": parts or "none",
            "params_sha256": params_sha256(net), "parent": parent_label,
            "n_changed_vs_parent": count_changed(parent, net)}


def cmd_m03(args: argparse.Namespace) -> int:
    from weightcode.common.identity import params_sha256

    out = out_dir(args)
    started = time.perf_counter()
    inst = load_dev_instance(CONTRAST_ID, CONTRAST)
    fin = final_inputs()
    w0 = {key: np.array(value) for key, value in inst.params.items()}
    limits = program_limits(inst, w0)
    child, edit = apply_derived_patch(w0, fin["folder"] / "patch.json", limits, W0_SHA256)
    identity: dict[str, Any] = {"schema": SCHEMA, "experiment": "M03", "provenance": provenance(),
                                "final_child_sha256": edit["child_sha256"], "model_npz_sha256": fin["model_npz_sha256"],
                                "expected_child_sha256": FINAL_CHILD_SHA256, "edit": edit,
                                "limits_equal_e10_program_limits": e10_limits_equal(inst, w0, limits)}
    if edit["child_sha256"] != FINAL_CHILD_SHA256 or fin["model_npz_sha256"] != FINAL_CHILD_SHA256:
        write_json(out / "m03_identity_FAILED.json", identity)
        raise MechError("the final's derived patch does not rebuild the final child")
    identity["units"] = unit_checks(fin["deltas"], fin["stdout"])
    units = tuple(identity["units"]["units"])
    identity["support_changed_entries"] = check_support(w0, child, units)
    gate = identify_gate(child[L1W], child[L1B], units)
    identity["gate"] = gate | check_gate_against_stdout(gate, identity["units"]["printed_units_program_order"],
                                                        fin["stdout"])
    nets = {name: build_network(w0, child, units, parts) for name, parts in M03_NETWORKS}
    identity["N0_sha256"], identity["N_FBR_sha256"] = params_sha256(nets["N0"]), params_sha256(nets["N_FBR"])
    identity["N0_is_W0"] = identity["N0_sha256"] == W0_SHA256
    identity["N_FBR_is_final_child"] = identity["N_FBR_sha256"] == FINAL_CHILD_SHA256
    if not (identity["N0_is_W0"] and identity["N_FBR_is_final_child"]):
        write_json(out / "m03_identity_FAILED.json", identity)
        raise MechError("N0 or N_FBR does not hash to its record")
    lines: dict[str, dict[str, Any]] = {}
    correct: dict[str, dict[str, np.ndarray]] = {}
    checks: list[dict[str, Any]] = []
    for name in ("N0", "N_FBR", "N_B", "N_F", "N_FB", "N_FR"):  # the networks with records first
        parts = dict(M03_NETWORKS)[name]
        line = network_line("M03", name, parts, nets[name], w0, "N0")
        t0 = time.perf_counter()
        cond, clean_ro, clean_sha = readout(inst, nets[name], None)
        line["readouts"] = {"vs_N0": {CONTRAST: cond, "clean": clean_ro, "joint": cond["delta"] + clean_ro["delta"]}}
        line["clean_split_sha256"] = clean_sha
        line["readout_seconds"] = time.perf_counter() - t0
        checks.extend(m03_record_checks(name, cond, clean_ro, fin))
        if not all(c["equal"] for c in checks):
            identity["readout_checks"] = checks
            write_json(out / "m03_identity_FAILED.json", identity)
            raise MechError(f"a readout differs from its record: {checks}")
        if name == "N_FBR":
            identity["readout_checks"] = checks
            write_json(out / "m03_identity.json", identity)
        correct[name] = per_image_correct(inst, nets[name])
        lines[name] = line
        print(f"M03 {name}: gain {cond['delta']:+.3f} clean {clean_ro['delta']:+.3f} joint "
              f"{cond['delta'] + clean_ro['delta']:+.3f} fixed {cond['fixed']} broken {cond['broken']} worst clean "
              f"{clean_ro['worst_class']:+.3f} ({line['readout_seconds']:.0f} s)", flush=True)
    boot = paired_bootstrap(correct, M03_CONTRASTS, ("gain", "clean"))
    boot["experiment"] = "M03"
    boot["condition"] = CONTRAST
    write_json(out / "m03_bootstrap.json", boot)
    mod = program_module(fin["program_text"])
    layout = layout_of(inst)
    sets = image_sets([inst])
    internal: dict[str, Any] = {}
    for name, parts in M03_NETWORKS:
        t0 = time.perf_counter()
        gate_unit = gate["gate_unit"] if "R" in parts else None
        internal[name] = network_internal(mod, layout, inst, nets[name], sets, units, gate_unit, None)
        internal[name]["gate_defined"] = gate_unit is not None
        print(f"M03 internal {name} ({time.perf_counter() - t0:.0f} s): "
              f"{json.dumps(jsonable(internal[name]['sets'][CONTRAST]['gate']))}", flush=True)
    write_json(out / "m03_internal.json", {"schema": SCHEMA, "experiment": "M03", "units": list(units),
                                           "gate_unit": gate["gate_unit"], "networks": internal})
    write_jsonl(out / "m03_metrics.jsonl", [lines[name] for name, _ in M03_NETWORKS])
    print(f"M03 done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


def m06_readouts(insts: Mapping[str, Any], net: Mapping[str, np.ndarray], w1: Mapping[str, np.ndarray]
                 ) -> tuple[dict[str, Any], str]:
    """A network on gaussian, contrast and clean V_select against W0 (the instances' base) and against W1."""
    result: dict[str, Any] = {}
    shas = set()
    for parent_label, parent in (("vs_W0", None), ("vs_W1", w1)):
        per: dict[str, Any] = {}
        cleans = []
        for condition, inst in insts.items():
            cond, clean_ro, clean_sha = readout(inst, net, parent)
            per[condition] = cond
            cleans.append(clean_ro)
            shas.add(clean_sha)
        if any(c != cleans[0] for c in cleans[1:]):
            raise MechError("the clean readouts of the two instances differ")
        per["clean"] = cleans[0]
        result[parent_label] = per
    if len(shas) != 1:
        raise MechError(f"the instances' clean companions differ: {sorted(shas)}")
    return result, shas.pop()


def cmd_m06(args: argparse.Namespace) -> int:
    from weightcode.common.identity import params_sha256

    out = out_dir(args)
    started = time.perf_counter()
    inst_g = load_dev_instance(GAUSSIAN_ID, GAUSSIAN)
    inst_c = load_dev_instance(CONTRAST_ID, CONTRAST)
    if inst_g.forward_sha256 != inst_c.forward_sha256:
        raise MechError("the two instances have different forward.py files")
    insts = {GAUSSIAN: inst_g, CONTRAST: inst_c}
    cand = candidate_inputs()
    w1 = {key: np.array(value) for key, value in cand["w1"].items()}
    limits = program_limits(inst_c, w1)
    from weightcode.patch import task_limits

    identity: dict[str, Any] = {"schema": SCHEMA, "experiment": "M06", "provenance": provenance(),
                                "w1_sha256": params_sha256(w1), "limits_match_control_a":
                                task_limits(limits) == cand["round2"]["limits"]}
    if not identity["limits_match_control_a"]:
        write_json(out / "m06_identity_FAILED.json", identity)
        raise MechError(f"limits {task_limits(limits)} differ from control (a)'s {cand['round2']['limits']}")
    child, edit = apply_derived_patch(w1, cand["paths"]["patch"], limits, W1_SHA256)
    identity.update(candidate_child_sha256=edit["child_sha256"], expected_child_sha256=CANDIDATE_CHILD_SHA256,
                    edit=edit)
    if edit["child_sha256"] != CANDIDATE_CHILD_SHA256:
        write_json(out / "m06_identity_FAILED.json", identity)
        raise MechError("the candidate's derived patch does not rebuild the candidate child")
    identity["units"] = unit_checks(cand["deltas"], cand["stdout"])
    units = tuple(identity["units"]["units"])
    identity["support_changed_entries"] = check_support(w1, child, units)
    gate = identify_gate(child[L1W], child[L1B], units)
    identity["gate"] = gate | check_gate_against_stdout(gate, identity["units"]["printed_units_program_order"],
                                                        cand["stdout"])
    identity["g2_on_w1"] = g2_structure(w1)
    if not identity["g2_on_w1"]["is_ramp_pair"]:
        write_json(out / "m06_identity_FAILED.json", identity)
        raise MechError("units 503 and 360 of W1 are not the gaussian block-2 ramp pair")
    nets = {name: build_network(w1, child, units, parts) for name, parts in M06_NETWORKS}
    identity["W1_net_sha256"], identity["W1_B'F'R'_sha256"] = params_sha256(nets["W1"]), params_sha256(nets["W1_B'F'R'"])
    identity["W1_is_W1"] = identity["W1_net_sha256"] == W1_SHA256
    identity["W1_B'F'R'_is_candidate_child"] = identity["W1_B'F'R'_sha256"] == CANDIDATE_CHILD_SHA256
    if not (identity["W1_is_W1"] and identity["W1_B'F'R'_is_candidate_child"]):
        write_json(out / "m06_identity_FAILED.json", identity)
        raise MechError("W1 or W1_B'F'R' does not hash to its record")
    lines: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    recorded = cand["round1"]["readouts"]
    for name, parts in M06_NETWORKS:  # W1 first: its readouts have a record
        line = network_line("M06", name, parts, nets[name], w1, "W1")
        line["n_changed_vs_W0"] = network_line("M06", name, parts, nets[name], inst_g.params, "W0")[
            "n_changed_vs_parent"]
        t0 = time.perf_counter()
        line["readouts"], line["clean_split_sha256"] = m06_readouts(insts, nets[name], w1)
        line["readout_seconds"] = time.perf_counter() - t0
        if name == "W1":
            checks.extend(w1_record_checks(line, insts, recorded))
            if not all(c["equal"] for c in checks):
                identity["readout_checks"] = checks
                write_json(out / "m06_identity_FAILED.json", identity)
                raise MechError(f"W1's readouts differ from the control (a) record: {checks}")
            identity["readout_checks"] = checks
            write_json(out / "m06_identity.json", identity)
        lines[name] = line
        r0, r1 = line["readouts"]["vs_W0"], line["readouts"]["vs_W1"]
        print(f"M06 {name}: vs W0 gaussian {r0[GAUSSIAN]['delta']:+.3f} contrast {r0[CONTRAST]['delta']:+.3f} clean "
              f"{r0['clean']['delta']:+.3f} | vs W1 gaussian {r1[GAUSSIAN]['delta']:+.3f} contrast "
              f"{r1[CONTRAST]['delta']:+.3f} clean {r1['clean']['delta']:+.3f} ({line['readout_seconds']:.0f} s)",
              flush=True)
    named = ("W1_F'", "W1_B'", "W1_F'R'", "W1_B'F'R'")
    rows = {name: loss_rows_against_w1(lines[name]["readouts"]["vs_W1"]) for name in named}
    n_gauss = lines["W1"]["readouts"]["vs_W1"][GAUSSIAN]["n"]
    losses = {name: 100.0 * rows[name] / n_gauss for name in named}
    if any(not math.isclose(losses[name], loss_against_w1(lines[name]["readouts"]["vs_W1"]), abs_tol=1e-9)
           for name in named):
        raise MechError("L in rows and L in points disagree")
    reading = m06_reading(rows["W1_F'"], rows["W1_B'"], rows["W1_B'F'R'"])  # exact, in rows
    reading["L"] = {"W1_F'": losses["W1_F'"], "W1_B'": losses["W1_B'"], "W1_B'F'R' (L*)": losses["W1_B'F'R'"]}
    reading.update(schema=SCHEMA, experiment="M06", L_all=losses, L_rows=rows, n_gaussian_rows=n_gauss,
                   comparison="in rows of gaussian V_select (exact integers); L in points = 100 * rows / n",
                   scope="decides the cause in DEV_RESULTS (E11 two-round stream) and the paper; says nothing about "
                         "other programs or other orders")
    write_json(out / "m06_reading.json", reading)
    print(f"M06 reading: {reading['reading']} L {losses}", flush=True)
    mod = program_module(final_inputs()["program_text"])
    layout = layout_of(inst_c)
    sets = image_sets([inst_g, inst_c])
    internal: dict[str, Any] = {}
    for name, parts in M06_NETWORKS:
        t0 = time.perf_counter()
        gate_unit = gate["gate_unit"] if "R" in parts else None
        holds_g2 = holds_w1_g2(nets[name], w1)
        internal[name] = network_internal(mod, layout, inst_c, nets[name], sets, units, gate_unit,
                                          G2_UNITS[0] if holds_g2 else None)
        internal[name]["gate_defined"] = gate_unit is not None
        internal[name]["g2_extra_defined"] = holds_g2
        print(f"M06 internal {name} ({time.perf_counter() - t0:.0f} s)", flush=True)
    write_json(out / "m06_internal.json", {"schema": SCHEMA, "experiment": "M06", "units": list(units),
                                           "gate_unit": gate["gate_unit"], "networks": internal,
                                           "g2_gate_extra_note": ("not in the PREREG list: the gaussian G2 gate "
                                                                  "(unit 503 pre-activation > 0, the A2 definition), "
                                                                  "only where units 503 and 360 hold W1's values")})
    write_jsonl(out / "m06_metrics.jsonl", [lines[name] for name, _ in M06_NETWORKS])
    print(f"M06 done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


def run_program_on_w1(inst: Any, w1: Mapping[str, np.ndarray], experience: Mapping[str, np.ndarray],
                      program: Path, work: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run one program file on W1 with control (a)'s round-2 experience in the LEARN program sandbox (4 threads,
    8 GiB, 300 s), write its derived patch and apply it under control (a)'s limits. Returns the update, the run
    record and the edit record (with the child, or the refusal)."""
    from weightcode.harness.program import run_update, write_derived_patch
    from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, SANDBOX_PYTHON
    from weightcode.harness.transfer import stage_student
    from weightcode.patch import PatchError
    from weightcode.sandbox.program import PROGRAM_WALL_S, ProgramSandbox

    work.mkdir(parents=True, mode=0o700)
    record_exp = {"content_sha256": ROUND2_EXPERIENCE_SHA256}
    student, experience_file, _ = stage_student(inst, work / "student", params=dict(w1),
                                                experience=(dict(experience), record_exp))
    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=DEFAULT_MEMORY_BYTES, threads=THREADS)
    t0 = time.perf_counter()
    update = run_update(student, experience_file, program, work / "run", sandbox, PROGRAM_WALL_S)
    patch_dir = work / "patch"
    patch_dir.mkdir()
    write_derived_patch(update, w1, W1_SHA256, patch_dir)
    run = {"program": {"path": str(program), "sha256": file_sha256(program)},
           "deltas_sha256": update.record["deltas_sha256"], "tensors": list(update.tensors),
           "stdout": update.record["run"]["stdout"], "stderr": update.record["run"]["stderr"],
           "duration_s": update.record["run"]["duration_s"], "wall_s": time.perf_counter() - t0,
           "sandbox": update.record["sandbox"], "inputs_sha256": update.record["inputs_sha256"],
           "wall_limit_s": PROGRAM_WALL_S, "work_dir": str(work)}
    limits = program_limits(inst, w1)
    try:
        child, edit = apply_derived_patch(w1, patch_dir / "derived_patch.json", limits, W1_SHA256)
        edit["within_limits"] = True
    except PatchError as refusal:
        edit = {"within_limits": False, "limit_messages": list(refusal.messages)}
        child = None
    edit["child"] = child
    return update, run, edit


def cmd_m06b(args: argparse.Namespace) -> int:
    out = out_dir(args)
    reading = json.loads((out / "m06_reading.json").read_text(encoding="utf-8"))
    if reading["reading"] != READING_COLLISION:
        write_json(out / "m06b.json", {"schema": SCHEMA, "experiment": "M06b", "ran": False,
                                       "reason": f"the M06 reading is {reading['reading']!r}, not "
                                                 f"{READING_COLLISION!r} (PREREG: M06b runs only on collision)",
                                       "provenance": provenance()})
        print("M06b not run: the reading is not collision", flush=True)
        return 0
    from weightcode.common.identity import params_sha256

    started = time.perf_counter()
    inst_g = load_dev_instance(GAUSSIAN_ID, GAUSSIAN)
    inst_c = load_dev_instance(CONTRAST_ID, CONTRAST)
    cand = candidate_inputs()
    w1 = {key: np.array(value) for key, value in cand["w1"].items()}
    paths = cand["paths"]
    require_sha(paths["evidence2"], ROUND2_EVIDENCE_SHA256)
    with np.load(paths["experience"], allow_pickle=False) as archive:
        experience = {key: archive[key] for key in archive.files}
    with np.load(paths["evidence2"], allow_pickle=False) as archive:
        evidence = {key: archive[key] for key in archive.files}
    record: dict[str, Any] = {"schema": SCHEMA, "experiment": "M06b", "ran": True, "provenance": provenance(),
                              "experience_content_sha256": params_sha256(experience),
                              "experience_x_new_is_round2_evidence": bool(
                                  np.array_equal(experience["x_new"], evidence["x"])
                                  and np.array_equal(experience["split_ids_new"], evidence["split_ids"]))}
    if record["experience_content_sha256"] != ROUND2_EXPERIENCE_SHA256 or not record[
            "experience_x_new_is_round2_evidence"]:
        write_json(out / "m06b_FAILED.json", record)
        raise MechError("the staged experience is not control (a)'s round-2 experience")
    final_program = runs_root() / "sessions" / SESSION_ID / "final" / PROGRAM_FILE
    require_sha(final_program, PROGRAM_SHA256)
    work = out / "m06b_work"
    _, repro_run, repro_edit = run_program_on_w1(inst_c, w1, experience, final_program, work / "repro")
    repro_child = repro_edit.pop("child")
    record["repro"] = {"run": repro_run, "edit": repro_edit, "expected_child_sha256": CANDIDATE_CHILD_SHA256,
                       "expected_deltas_sha256": CANDIDATE_DELTAS_SHA256,
                       "deltas_equal_control_a": repro_run["deltas_sha256"] == CANDIDATE_DELTAS_SHA256,
                       "stdout_equal_control_a": repro_run["stdout"] == cand["stdout"]}
    record["repro"]["reproduced"] = bool(repro_child is not None
                                         and repro_edit["child_sha256"] == CANDIDATE_CHILD_SHA256)
    if not record["repro"]["reproduced"]:
        if repro_child is not None:
            ref, _ = apply_derived_patch(w1, paths["patch"], program_limits(inst_c, w1), W1_SHA256)
            record["repro"]["max_abs_diff_vs_candidate"] = {
                key: float(np.max(np.abs(repro_child[key].astype(np.float64) - ref[key].astype(np.float64))))
                for key in COMPONENT_TENSORS}
        write_json(out / "m06b.json", record)
        raise MechError("control (a) did not reproduce in the sandbox; M06b stops before the changed program")
    print(f"M06b repro: child {repro_edit['child_sha256'][:12]} reproduced", flush=True)
    original = final_program.read_text(encoding="utf-8")
    if original.count(M06B_OLD_LINE) != 1:
        raise MechError("the unit-selection line does not occur exactly once in the program")
    changed = original.replace(M06B_OLD_LINE, M06B_NEW_LINES)
    program_dir = work / "program"
    program_dir.mkdir(parents=True)
    changed_path = program_dir / M06B_PROGRAM_FILE
    changed_path.write_text(changed, encoding="utf-8")
    diff = "".join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile=PROGRAM_FILE,
                                        tofile=M06B_PROGRAM_FILE, n=2))
    (out / "m06b_program.diff").write_text(diff, encoding="utf-8")
    record["change"] = {"description": ("units 360 and 503 (those the gaussian program edits in block 2) removed from "
                                        "the candidate units before the 8 least used are taken; nothing else changed"),
                        "diff": diff, "program_sha256": file_sha256(changed_path), "program_file": M06B_PROGRAM_FILE,
                        "written_by": "M06b implementation (R30 job), as the PREREG names the change"}
    _, run, edit = run_program_on_w1(inst_c, w1, experience, changed_path, work / "changed")
    child = edit.pop("child")
    record["changed"] = {"run": run, "edit": edit}
    if child is None:  # refused under control (a)'s limits: recorded and read without the size limits
        from weightcode.patch import apply_patch

        unbounded = dataclasses.replace(program_limits(inst_c, w1), max_rel_frob=None, max_changed_frac=None)
        patch_file = work / "changed" / "patch" / "derived_patch.json"
        result = apply_patch(w1, json.loads(patch_file.read_text(encoding="utf-8")), patch_file.parent, unbounded,
                             W1_SHA256)
        child = {key: np.array(value, copy=True) for key, value in result.child.items()}
        record["changed"]["edit"]["unbounded_child_sha256"] = result.child_sha256
    printed = stdout_units(run["stdout"])
    units = units_from_delta(np.asarray(child[L1W], np.float64) - np.asarray(w1[L1W], np.float64))
    if tuple(sorted(printed)) != units or set(units) & set(M06B_REMOVED_UNITS):
        raise MechError(f"M06b units: from the child {units}, printed {printed}")
    gate = identify_gate(child[L1W], child[L1B], units)
    record["changed"]["units"] = {"units": list(units), "printed_units_program_order": printed}
    record["changed"]["gate"] = gate | check_gate_against_stdout(gate, printed, run["stdout"])
    record["changed"]["support_changed_entries"] = check_support(w1, child, units)
    insts = {GAUSSIAN: inst_g, CONTRAST: inst_c}
    line = network_line("M06b", "W1_M06b", "program rerun, units 360 and 503 excluded", child, w1, "W1")
    line["n_changed_vs_W0"] = network_line("M06b", "W1_M06b", "", child, inst_g.params, "W0")["n_changed_vs_parent"]
    line["readouts"], line["clean_split_sha256"] = m06_readouts(insts, child, w1)
    line["within_limits"] = edit["within_limits"]
    record["changed"]["L_gaussian_vs_W1"] = loss_against_w1(line["readouts"]["vs_W1"])
    mod = program_module(original)
    keeps_g2 = holds_w1_g2(child, w1)
    internal = network_internal(mod, layout_of(inst_c), inst_c, child, image_sets([inst_g, inst_c]), units,
                                gate["gate_unit"], G2_UNITS[0] if keeps_g2 else None)
    internal["gate_defined"] = True
    internal["g2_extra_defined"] = keeps_g2
    write_json(out / "m06b_internal.json", {"schema": SCHEMA, "experiment": "M06b", "units": list(units),
                                            "gate_unit": gate["gate_unit"], "networks": {"W1_M06b": internal}})
    write_jsonl(out / "m06b_metrics.jsonl", [line])
    record["seconds"] = time.perf_counter() - started
    write_json(out / "m06b.json", record)
    r0, r1 = line["readouts"]["vs_W0"], line["readouts"]["vs_W1"]
    print(f"M06b: vs W0 gaussian {r0[GAUSSIAN]['delta']:+.3f} contrast {r0[CONTRAST]['delta']:+.3f} clean "
          f"{r0['clean']['delta']:+.3f} | vs W1 gaussian {r1[GAUSSIAN]['delta']:+.3f} contrast "
          f"{r1[CONTRAST]['delta']:+.3f} clean {r1['clean']['delta']:+.3f}", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# Summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _f(value: float) -> str:
    return f"{value:+.3f}"


def cmd_summary(args: argparse.Namespace) -> int:
    out = out_dir(args)
    parts: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    for name in ("m03", "m06", "m06b"):
        path = out / f"{name}_metrics.jsonl"
        parts[name] = {"metrics": path.exists(), "failed": sorted(p.name for p in out.glob(f"{name}*FAILED*"))}
        if path.exists():
            metrics.extend(_read_jsonl(path))
    write_jsonl(out / "metrics.jsonl", metrics)
    internal = {name.upper(): json.loads((out / f"{name}_internal.json").read_text(encoding="utf-8"))
                for name in ("m03", "m06", "m06b") if (out / f"{name}_internal.json").exists()}
    write_json(out / "internal_stats.json", internal)
    boot = json.loads((out / "m03_bootstrap.json").read_text(encoding="utf-8")) if (
        out / "m03_bootstrap.json").exists() else None
    write_json(out / "bootstrap.json", {"M03": boot})
    reading = json.loads((out / "m06_reading.json").read_text(encoding="utf-8")) if (
        out / "m06_reading.json").exists() else None
    m06b = json.loads((out / "m06b.json").read_text(encoding="utf-8")) if (out / "m06b.json").exists() else None
    m03_rows = [m for m in metrics if m["experiment"] == "M03"]
    m06_rows = [m for m in metrics if m["experiment"] in ("M06", "M06b")]
    md = ["# M03 and M06, fixed-component interventions on the E11 contrast program", "",
          f"Spec: {SPEC}. V_select development readouts in points, seed 100, one program; no check spent, nothing "
          "selected.", ""]
    if m03_rows:
        md += ["## M03 (W0, contrast_s5; changes against N0)", "",
               "| network | parts | gain | clean | joint | fixed | broken | worst class clean | gate open, cond / clean "
               "patch tokens | units' output norm, cond / clean | token spread cond/clean |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        m03_int = internal.get("M03", {}).get("networks", {})
        for row in m03_rows:
            r = row["readouts"]["vs_N0"]
            st = m03_int.get(row["network"], {})
            gate_c = st.get("sets", {}).get(CONTRAST, {}).get("gate")
            gate_k = st.get("sets", {}).get("clean", {}).get("gate")
            gate_txt = "n/a" if not gate_c else f"{gate_c['rate_patch_tokens']:.4f} / {gate_k['rate_patch_tokens']:.4f}"
            norm_txt = (f"{st['sets'][CONTRAST]['units_output_norm']['patch_token_mean']:.4f} / "
                        f"{st['sets']['clean']['units_output_norm']['patch_token_mean']:.4f}") if st else ""
            tok_txt = (f"{st['tokenizer_condition_over_clean'][CONTRAST]['token_spread']['mean_of_image_ratios']:.4f}"
                       if st else "")
            md.append(f"| {row['network']} | {row['parts']} | {_f(r[CONTRAST]['delta'])} | {_f(r['clean']['delta'])} | "
                      f"{_f(r['joint'])} | {r[CONTRAST]['fixed']} | {r[CONTRAST]['broken']} | "
                      f"{_f(r['clean']['worst_class'])} | {gate_txt} | {norm_txt} | {tok_txt} |")
        if boot:
            md += ["", "Paired bootstrap (image level, 1,000 resamples, default_rng(20261106); 95% percentile "
                   "intervals; V_select sampling only):", "",
                   "| contrast | gain point | gain 95% CI | clean point | clean 95% CI |", "|---|---|---|---|---|"]
            for name, c in boot["contrasts"].items():
                md.append(f"| {name} | {_f(c['gain']['point'])} | [{c['gain']['ci95'][0]:+.3f}, "
                          f"{c['gain']['ci95'][1]:+.3f}] | {_f(c['clean']['point'])} | [{c['clean']['ci95'][0]:+.3f}, "
                          f"{c['clean']['ci95'][1]:+.3f}] |")
    if m06_rows:
        md += ["", "## M06 (W1; gaussian, contrast, clean; against W0 and against W1)", "",
               "| network | gaussian vs W0 | contrast vs W0 | clean vs W0 | gaussian vs W1 (fixed/broken) | "
               "contrast vs W1 (fixed/broken) | clean vs W1 (fixed/broken) | worst class clean vs W1 |",
               "|---|---|---|---|---|---|---|---|"]
        for row in m06_rows:
            r0, r1 = row["readouts"]["vs_W0"], row["readouts"]["vs_W1"]
            md.append(f"| {row['network']} | {_f(r0[GAUSSIAN]['delta'])} | {_f(r0[CONTRAST]['delta'])} | "
                      f"{_f(r0['clean']['delta'])} | {_f(r1[GAUSSIAN]['delta'])} ({r1[GAUSSIAN]['fixed']}/"
                      f"{r1[GAUSSIAN]['broken']}) | {_f(r1[CONTRAST]['delta'])} ({r1[CONTRAST]['fixed']}/"
                      f"{r1[CONTRAST]['broken']}) | {_f(r1['clean']['delta'])} ({r1['clean']['fixed']}/"
                      f"{r1['clean']['broken']}) | {_f(r1['clean']['worst_class'])} |")
    if reading:
        md += ["", f"M06 reading (fixed rule): **{reading['reading']}**. L (points of gaussian V_select gain lost "
               f"against W1): {json.dumps(reading['L_all'])}; shares of L*: {json.dumps(reading['shares_of_L*'])}."]
    if m06b is not None:
        md += ["", "M06b: " + ("not run: " + m06b["reason"] if not m06b["ran"] else
                               f"ran; control (a) reproduced: {m06b['repro']['reproduced']}; units "
                               f"{m06b.get('changed', {}).get('units', {}).get('units')}")]
    md += ["", f"Parts: {json.dumps(parts)}", ""]
    summary = {"schema": SCHEMA, "spec": SPEC, "created_at": utc_now(), "parts": parts,
               "m03": {"metrics": m03_rows, "bootstrap": boot}, "m06": {"metrics": m06_rows, "reading": reading},
               "m06b": m06b and {k: v for k, v in m06b.items() if k != "provenance"}}
    write_json(out / "summary.json", summary)
    path = out / "summary.md"
    if path.exists():
        raise MechError(f"{path} exists")
    path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("m03", "m06", "m06b", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="record directory (created; files are never overwritten)")
    args = parser.parse_args(argv)
    handlers = {"m03": cmd_m03, "m06": cmd_m06, "m06b": cmd_m06b, "summary": cmd_summary}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
