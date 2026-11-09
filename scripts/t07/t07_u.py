"""PREREG T07-U (M05 phase 2): is the always-on part U of each contrast final the cause of its off-target effects?

The E11 ViT contrast final (s760cb52ce82fb937) and the E12 ResNet-18 contrast final (sccd12677dcdd125b) harm the noise
family, pixelate and spatter (and on ViT snow and frost) and help defocus, motion and zoom blur and fog, while the E11
gate is shut there (T07, T07-G). U is the gate-independent first-layer gain on the zero-mean (AC) part of the first
convolution in each program. Each final is split at the tensor level into U and its gated rest G; every network is
rebuilt from files on disk and hash-checked; nothing is trained and no program is run.

Arms per architecture (``DESIGN``): ``base_plus_U`` (the base with U's tensors set to the final's), ``final_minus_U``
(the base with G's tensors set to the final's) and, for E11 only, ``carrier_F`` (M03's N_F, the always-on carrier
preparation of the gated block, ``m03_m06.build_network(base, final, units, "F")``). E12 has no carrier arm: its gate
units are repurposed without a mean fold, so its carrier is not a separate program component.

Subcommands (CPU, numpy; run on NSCC by scripts/store/t07u_main.sh):

* ``build --arch A``: rebuild the final (``t07_range.network(A, "llm_contrast")``), check that U and G partition the
  tensors it changes and that the base with both replaced is the final, build every arm and check its hash (E11: M03's
  networks; E12: the hashes fixed before the run); writes ``build/<A>.json``.
* ``predict --arch A --arm M --set S``: predictions of arm M on T07 set S by T07's predict path and batch; writes
  ``preds/<A>/<M>/<S>.npy`` and a JSON beside it, never overwriting. Needs ``build/<A>.json``.
* ``table``: checks (R38's base and final predictions against R38's outputs.sha256 lines and t07_table.json, the E11
  arms against M03's readouts), every cell, the paired differences and additivity residuals, and the readings computed
  mechanically from the fixed rules (``readings``); writes ``t07u_table.json``. A failed check writes
  ``t07u_checks_FAILED.json`` and no reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t07_range as R  # noqa: E402

C = R.C
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_THREAD_ENV = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
sys.path.insert(0, str(HERE.parent / "mech_v1"))
import m03_m06 as M3  # noqa: E402  (its import sets the BLAS thread variables to 4; numpy is already loaded)

for _name, _value in _THREAD_ENV.items():  # restore them, so provenance records the threads BLAS was started with
    if _value is None:
        os.environ.pop(_name, None)
    else:
        os.environ[_name] = _value

SCHEMA = "weightcode.t07.u.v1"
SPEC = ("docs/PREREG_v0_DRAFT.md section 0.1, 'T07-U, the always-on part of the two contrast finals (M05 phase 2)'; "
        "NSCC R41-t07u")
FINAL = "llm_contrast"
"""T07's network name of the contrast final in both architectures."""
ARMS = ("base_plus_U", "final_minus_U", "carrier_F")

R38_RUN_REL = "t07/r38_20261107_124729_t07_r38_main_e4e638836a15"
"""T07's run of record under $WEIGHTCODE_RUNS: its base and final predictions are read, not recomputed."""
R38_TABLE_SHA256 = "55982f0da04b35140b494022768e64e4f0aa2115f7aff4084fbd2453265fa556"
R38_OUTPUTS_SHA256 = "d6d3bbb5cf4fad612e80ad2bfc2f30e1602b19c4e5b22a454cbbe6f78856c9e6"
"""private/nscc_runs/20261107_124729_t07_r38_main_e4e638836a15/outputs.sha256 (sha256sum lines, paths relative to the
store root). submit_store.sh refuses private/nscc_runs as a job input, so the job checks the lines below instead."""
R38_PREDS_LINES_SHA256 = "24fe72a26e47e022f57b6b4c1ceee7aaf4f91dc2b96ee90b7a05479a12625116"
"""sha256 of the 208 lines of R38's outputs.sha256 for preds/{vit,r18}/{base,llm_contrast}/ (the .npy and .json of 26
sets each), sorted bytewise, newline-terminated, i.e. with RUN = runs/ + R38_RUN_REL:
``grep -E "^[0-9a-f]{64}  $RUN/preds/(vit|r18)/(base|llm_contrast)/[^/]+$" outputs.sha256 | LC_ALL=C sort |
sha256sum``."""
R38_PREDS_LINES = 208
R38_NETWORKS = ("base", FINAL)

M03_METRICS_REL = "mech_v1/r30_20261106_205510_85bbf4d14aaf/m03_metrics.jsonl"
M03_METRICS_SHA256 = "9e3899ee1a885b2c8a9d1a2da8ad98ae591d91702ac87f71a5a58e123c373eca"
M03_FIELDS_INT = ("fixed", "broken", "n")
M03_FIELDS_FLOAT = ("base_acc", "child_acc", "delta")
REPRO_SETS = ("contrast_s5", C.CLEAN)
"""The T07 sets whose rows are M03's (contrast_s5 V_select and its clean companion)."""
FLOAT_ATOL = 1e-12

DESIGN: dict[str, dict[str, Any]] = {
    "vit": {
        "line": "E11 ViT-Lite-7/4, contrast final s760cb52ce82fb937 (instance i3eb3fc78)",
        "program": "k145_gated_b2_r6_orig_targets.py", "program_sha256": M3.PROGRAM_SHA256,
        "patch_sha256": M3.FINAL_PATCH_SHA256,
        "u_tensors": (M3.TOK,),
        "g_tensors": (M3.L1W, M3.L1B, M3.L2W, M3.L2B),
        "u_meaning": "M03's B: the ungated AC boost of the tokenizer's first convolution, K = 1.45",
        "g_meaning": ("M03's F and R: the 8 block-2 units' linear1 rows and bias entries and linear2 columns, and "
                      "linear2.bias (every block-2 tensor the final changes)"),
        "arms": ("base_plus_U", "final_minus_U", "carrier_F"),
        "expected": {"base_plus_U": "3410c8956990ffb933ca2476a94fc8762cc1f49e6096c9c969c8e066ce4cbe03",
                     "final_minus_U": "b58bcd92577a0551acd47ed653db698c4d7508ca2e2c3116a9c03130ddd3cbaa",
                     "carrier_F": "ed850a7c0f9baa2d9d7bbaa0f862290932110cbbb55349eea18358778bf9da8e"},
        "m03": {"base_plus_U": ("N_B", "B"), "final_minus_U": ("N_FR", "FR"), "carrier_F": ("N_F", "F")},
    },
    "r18": {
        "line": "E12 ResNet-18, contrast final sccd12677dcdd125b (instance ie3840b28)",
        "program": "g10_gain135_m025.py",
        "program_sha256": "985ba1fd5ee0362495309096fdf2a47ea1c72102e3e3270b914384488b87c62b",
        "patch_sha256": "2697ed87b1e6bde6b233fba16d1c92df923052df1b22c4dc032be191eb53a874",
        "u_tensors": ("conv1.weight", "bn1.weight", "bn1.running_mean"),
        "g_tensors": tuple(f"layer{block}.1.{tensor}" for block in (3, 4)
                           for tensor in ("conv1.weight", "bn1.weight", "bn1.bias", "conv2.weight")),
        "u_meaning": "the first-conv detail gain K_GAIN = 1.35, applied exactly through bn1",
        "g_meaning": "the two bounded-ReLU gates of layer3.1 and layer4.1 (two repurposed hidden units each)",
        "arms": ("base_plus_U", "final_minus_U"),
        # No earlier record: computed on the workstation on 2026-11-07 from the rebuilt final, before any prediction.
        "expected": {"base_plus_U": "6bb2c2d49137c088e920c84b3780372620b64edb5f076b958b035c9ea43376bf",
                     "final_minus_U": "9b2b69d45277d6b1e56dd7038098ab085bf20bd1aaa5206b8ef354485ecdb74a"},
        "m03": {},
    },
}

S_SETS: dict[str, dict[str, tuple[str, ...]]] = {
    "vit": {"S_harm": ("gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3", "gaussian_noise_s4",
                       "gaussian_noise_s5", "snow_s5", "shot_noise_s5", "impulse_noise_s5", "frost_s5", "pixelate_s5",
                       "speckle_noise_s5", "spatter_s5"),
            "S_help": ("defocus_blur_s5", "motion_blur_s5", "zoom_blur_s5", "fog_s5")},
    "r18": {"S_harm": ("gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3", "gaussian_noise_s4",
                       "gaussian_noise_s5", "shot_noise_s5", "impulse_noise_s5", "pixelate_s5", "speckle_noise_s5",
                       "spatter_s5"),
            "S_help": ("defocus_blur_s5", "motion_blur_s5", "zoom_blur_s5", "fog_s5")},
}
"""Fixed from the T07 record (t07_table.json 55982f0d..., the contrast final's labels): the sets other than clean and
contrast_s1..s5 where the final is harm (S_harm) or help (S_help). ``table`` re-derives them from the record and stops
if they differ."""
OWN = tuple(f"contrast_s{s}" for s in range(1, 6))
OWN_MAIN = "contrast_s5"
EXCLUDED_FROM_S = (C.CLEAN,) + OWN

SUPPORT, REFUTE, MIXED = "support", "refutation", "mixed"
GAIN_COSTS, GAIN_SEPARABLE, GAIN_PARTLY = "removing U costs the gain", "U is separable", "partly separable"
OWN_COSTS_BELOW = 0.50
OWN_SEPARABLE_AT_LEAST = 0.75

STATEMENTS = {
    SUPPORT: ("U causes the off-target {effect}: base_plus_U carries the final's label on at least 75% of the sets and "
              "final_minus_U on at most 25%. This rules out that the off-target {effect} needs the gate to open, and "
              "names the first-layer boost as its source. It does not show that a smaller or gated boost would keep "
              "the own gain without the cost (untested), and it is seed-100 dev evidence on two finals."),
    REFUTE: ("The off-target {effect} comes with the gated part, not U: base_plus_U carries the final's label on at "
             "most 25% of the sets and final_minus_U on at least 75%. This rules out the first-layer boost as its "
             "source and points to the gated block (gate leak or carrier). {carrier_note}"),
    MIXED: ("Mixed: both parts carry some of the off-target {effect}; both counts and the per-set shares "
            "delta(arm) / delta(final) are reported. Neither the boost nor the gated part is named alone."),
}
CARRIER_NOTE = {
    "vit": "On E11, carrier_F's labels tell the carrier preparation F from the gated correction R and its leak.",
    "r18": "It does not separate those two on E12, which has no carrier arm.",
}


class UError(RuntimeError):
    """An input, a rebuilt network or a check does not match its record or the fixed design."""


# ----------------------------------------------------------------------------------------------------------------
# Pure parts (tests/test_t07_u.py)


def changed_tensors(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Changed-entry count of every tensor in which ``child`` differs from ``base`` (same keys, shapes, dtypes)."""
    if set(base) != set(child):
        raise UError(f"base and child keys differ: {sorted(set(base) ^ set(child))}")
    changed: dict[str, int] = {}
    for key in sorted(base):
        b, c = np.asarray(base[key]), np.asarray(child[key])
        if b.shape != c.shape or b.dtype != c.dtype:
            raise UError(f"{key}: base {b.dtype}{b.shape}, child {c.dtype}{c.shape}")
        n = int(np.count_nonzero(b != c))
        if n:
            changed[key] = n
    return changed


def check_partition(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], u_tensors: Sequence[str],
                    g_tensors: Sequence[str]) -> dict[str, Any]:
    """U and G partition the tensors the child changes: disjoint, every changed tensor in exactly one, every named
    tensor changed, nothing else changed. Returns the changed-entry counts."""
    u, g = list(u_tensors), list(g_tensors)
    for name, names in (("U", u), ("G", g)):
        if len(set(names)) != len(names):
            raise UError(f"{name} names a tensor twice: {names}")
        unknown = sorted(set(names) - set(base))
        if unknown:
            raise UError(f"{name} names tensors the network does not have: {unknown}")
    overlap = sorted(set(u) & set(g))
    if overlap:
        raise UError(f"U and G share tensors: {overlap}")
    changed = changed_tensors(base, child)
    neither = sorted(set(changed) - set(u) - set(g))
    if neither:
        raise UError(f"the child changes tensors in neither U nor G: {neither}")
    unchanged = sorted((set(u) | set(g)) - set(changed))
    if unchanged:
        raise UError(f"U or G names tensors the child does not change: {unchanged}")
    return {"changed_entries": changed, "U": u, "G": g,
            "U_entries": sum(changed[k] for k in u), "G_entries": sum(changed[k] for k in g)}


def replace_tensors(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], names: Sequence[str]
                    ) -> dict[str, np.ndarray]:
    """A copy of ``base`` with the named tensors copied from ``child``."""
    missing = sorted(set(names) - (set(base) & set(child)))
    if missing:
        raise UError(f"cannot replace tensors absent from base or child: {missing}")
    out = {key: np.array(value, copy=True) for key, value in base.items()}
    for key in names:
        out[key] = np.array(child[key], copy=True)
    return out


def at_least_75(count: int, n: int) -> bool:
    """count >= 75% of n (integer arithmetic: 9 of 12, 8 of 10, 3 of 4)."""
    return 4 * count >= 3 * n


def at_most_25(count: int, n: int) -> bool:
    """count <= 25% of n (3 of 12, 2 of 10, 1 of 4)."""
    return 4 * count <= n


def at_least_25(count: int, n: int) -> bool:
    """count >= 25% of n (3 of 12, 1 of 4)."""
    return 4 * count >= n


def carries(labels: Mapping[str, str], sets: Sequence[str], label: str) -> int:
    """How many of ``sets`` an arm labels ``label`` (every set must have a label)."""
    missing = [s for s in sets if s not in labels]
    if missing:
        raise UError(f"no label for sets {missing}")
    return sum(labels[s] == label for s in sets)


def off_target_reading(n: int, plus_count: int, minus_count: int) -> str:
    """support, refutation or mixed, from how many of the n sets base_plus_U and final_minus_U carry the label."""
    if n <= 0:
        raise UError("a reading needs at least one set")
    if at_least_75(plus_count, n) and at_most_25(minus_count, n):
        return SUPPORT
    if at_most_25(plus_count, n) and at_least_75(minus_count, n):
        return REFUTE
    return MIXED


def share(delta_arm: float, delta_final: float) -> float | None:
    """delta(arm) / delta(final); None where the final's change is exactly 0."""
    return None if delta_final == 0 else float(delta_arm) / float(delta_final)


def own_gain_reading(share_s5: float, minus_harm_count: int, n_harm: int) -> str:
    """The own-gain question: final_minus_U's contrast_s5 change as a share of the final's."""
    if share_s5 < OWN_COSTS_BELOW:
        return GAIN_COSTS
    if share_s5 >= OWN_SEPARABLE_AT_LEAST and at_most_25(minus_harm_count, n_harm):
        return GAIN_SEPARABLE
    return GAIN_PARTLY


def carrier_named(harm_count: int, n_harm: int, help_count: int, n_help: int) -> bool:
    """E11: F is named as a contributor when carrier_F is harm on >= 25% of S_harm or help on >= 25% of S_help."""
    return at_least_25(harm_count, n_harm) or at_least_25(help_count, n_help)


def additivity_values(c_final: np.ndarray, c_plus: np.ndarray, c_minus: np.ndarray, c_base: np.ndarray
                      ) -> np.ndarray:
    """Per image, (final - base) - (base_plus_U - base) - (final_minus_U - base) in 0/1 correctness; its mean in
    points is delta(final) - delta(base_plus_U) - delta(final_minus_U)."""
    as_int = [np.asarray(c, dtype=np.int64) for c in (c_final, c_plus, c_minus, c_base)]
    if len({c.shape for c in as_int}) != 1:
        raise UError("correctness vectors differ in shape")
    f, p, m, b = as_int
    return f - p - m + b


def readings(arch: str, labels: Mapping[str, Mapping[str, str]], deltas: Mapping[str, Mapping[str, float]],
             s_harm: Sequence[str], s_help: Sequence[str]) -> dict[str, Any]:
    """Every fixed reading of one architecture from the cells' labels and changes (points); ``labels`` and ``deltas``
    are keyed by "final" and the arm names, then by set."""
    out: dict[str, Any] = {"off_target": {}}
    for effect, sets, label in (("harm", s_harm, "harm"), ("help", s_help, "help")):
        n = len(sets)
        plus = carries(labels["base_plus_U"], sets, label)
        minus = carries(labels["final_minus_U"], sets, label)
        final = carries(labels["final"], sets, label)
        if final != n:
            raise UError(f"{arch}: the final is {label} on {final} of the {n} S_{effect} sets")
        reading = off_target_reading(n, plus, minus)
        per_set = [{"set": s, "final_delta_points": deltas["final"][s],
                    **{f"{arm}_label": labels[arm][s] for arm in labels if arm != "final"},
                    **{f"{arm}_delta_points": deltas[arm][s] for arm in deltas if arm != "final"},
                    **{f"{arm}_share": share(deltas[arm][s], deltas["final"][s]) for arm in deltas if arm != "final"}}
                   for s in sets]
        statement = STATEMENTS[reading].format(effect=effect, carrier_note=CARRIER_NOTE[arch])
        out["off_target"][effect] = {
            "label": label, "n_sets": n, "sets": list(sets), "base_plus_U_carries": plus,
            "final_minus_U_carries": minus,
            "rule": {"support": f"base_plus_U >= 75% (>= {math.ceil(0.75 * n)} of {n}) and final_minus_U <= 25% "
                                f"(<= {math.floor(0.25 * n)} of {n})",
                     "refutation": f"base_plus_U <= 25% (<= {math.floor(0.25 * n)}) and final_minus_U >= 75% "
                                   f"(>= {math.ceil(0.75 * n)})", "mixed": "anything else"},
            "reading": reading, "statement": statement, "per_set": per_set}
    n_harm = len(s_harm)
    minus_harm = out["off_target"]["harm"]["final_minus_U_carries"]
    if not deltas["final"][OWN_MAIN] > 0:
        raise UError(f"{arch}: the final's {OWN_MAIN} change {deltas['final'][OWN_MAIN]} is not positive")
    share_s5 = share(deltas["final_minus_U"][OWN_MAIN], deltas["final"][OWN_MAIN])
    assert share_s5 is not None
    out["own_gain"] = {
        "share_s5": share_s5, "final_delta_points_s5": deltas["final"][OWN_MAIN],
        "final_minus_U_delta_points_s5": deltas["final_minus_U"][OWN_MAIN],
        "final_minus_U_harm_on_S_harm": minus_harm, "n_S_harm": n_harm,
        "rule": {GAIN_COSTS: f"share < {OWN_COSTS_BELOW}",
                 GAIN_SEPARABLE: f"share >= {OWN_SEPARABLE_AT_LEAST} and final_minus_U harm on <= 25% of S_harm "
                                 f"(<= {math.floor(0.25 * n_harm)} of {n_harm})",
                 GAIN_PARTLY: "otherwise"},
        "reading": own_gain_reading(share_s5, minus_harm, n_harm),
        "own_sets": [{"set": s, "final_delta_points": deltas["final"][s],
                      **{f"{arm}_delta_points": deltas[arm][s] for arm in deltas if arm != "final"},
                      **{f"{arm}_share": share(deltas[arm][s], deltas["final"][s]) for arm in deltas if arm != "final"}}
                     for s in OWN]}
    if "carrier_F" in labels:
        harm_f = carries(labels["carrier_F"], s_harm, "harm")
        help_f = carries(labels["carrier_F"], s_help, "help")
        named = carrier_named(harm_f, n_harm, help_f, len(s_help))
        out["carrier_F"] = {
            "harm_on_S_harm": harm_f, "help_on_S_help": help_f, "n_S_harm": n_harm, "n_S_help": len(s_help),
            "rule": f"F is named when carrier_F is harm on >= 25% of S_harm (>= {math.ceil(0.25 * n_harm)}) or help on "
                    f">= 25% of S_help (>= {math.ceil(0.25 * len(s_help))})",
            "F_named_as_contributor": named,
            "statement": ("F is named as a contributor: U alone is not the whole always-on part." if named else
                          "F is not named as a contributor.")}
    return out


def preds_lines_digest(lines: Sequence[str]) -> str:
    """sha256 of sha256sum-format lines sorted bytewise and newline-terminated (``LC_ALL=C sort | sha256sum``)."""
    text = "".join(f"{line}\n" for line in sorted(lines, key=lambda s: s.encode("utf-8")))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def same_fields(ours: Mapping[str, Any], record: Mapping[str, Any], ints: Sequence[str], floats: Sequence[str]
                ) -> dict[str, list[Any]]:
    """Fields that differ: integers exactly, floats within ``FLOAT_ATOL`` (the T07 and M03 record tolerance)."""
    diffs: dict[str, list[Any]] = {}
    for key in ints:
        if ours.get(key) is None or record.get(key) is None or int(ours[key]) != int(record[key]):
            diffs[key] = [ours.get(key), record.get(key)]
    for key in floats:
        a, b = ours.get(key), record.get(key)
        if a is None or b is None or not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=FLOAT_ATOL):
            diffs[key] = [a, b]
    return diffs


# ----------------------------------------------------------------------------------------------------------------
# Networks


def provenance() -> dict[str, Any]:
    out = R.provenance()
    out.pop("vit_internal_sha256", None)
    out.update({"script": str(Path(__file__).resolve()), "script_sha256": R.file_sha256(Path(__file__)),
                "t07_range_sha256": R.file_sha256(Path(R.__file__)), "m03_m06_sha256": R.file_sha256(Path(M3.__file__)),
                "spec": SPEC})
    return out


def final_folder(arch: str) -> Path:
    return R.runs_root() / "sessions" / R.ARCHS[arch]["finals"]["contrast"]["session"] / "final"


def program_checks(arch: str) -> dict[str, Any]:
    """The final's program file and patch.json against their records; the patch's tensors are exactly U and G."""
    spec = DESIGN[arch]
    folder = final_folder(arch)
    R.require_sha(folder / spec["program"], spec["program_sha256"])
    R.require_sha(folder / "patch.json", spec["patch_sha256"])
    patch = json.loads((folder / "patch.json").read_text(encoding="utf-8"))
    tensors = [op["tensor"] for op in patch["ops"]]
    if any(op["op"] != "add_dense" for op in patch["ops"]) or sorted(tensors) != sorted(spec["u_tensors"]
                                                                                      + spec["g_tensors"]):
        raise UError(f"{folder / 'patch.json'}: ops on {tensors}, expected add_dense on U and G")
    session = R.ARCHS[arch]["finals"]["contrast"]["session"]
    return {"program": f"runs/sessions/{session}/final/{spec['program']}",
            "program_sha256": spec["program_sha256"], "patch_sha256": spec["patch_sha256"], "patch_tensors": tensors}


def build_arms(arch: str, only: str | None = None) -> tuple[Any, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """(the contrast instance, {arm: params}, build record); every check of ``build`` runs on every call."""
    from weightcode.common.identity import count_changed, params_sha256

    spec = DESIGN[arch]
    arms = spec["arms"] if only is None else (only,)
    if not set(arms) <= set(spec["arms"]):
        raise UError(f"{arch} has no arm {only}; its arms are {spec['arms']}")
    inst, child, child_info = R.network(arch, FINAL)
    base = {key: np.asarray(value) for key, value in inst.params.items()}
    base_sha = params_sha256(base)
    if base_sha != R.ARCHS[arch]["base_sha256"]:
        raise UError(f"{arch}: base {base_sha}, expected {R.ARCHS[arch]['base_sha256']}")
    child_sha = R.ARCHS[arch]["finals"]["contrast"]["child_sha256"]
    record: dict[str, Any] = {"arch": arch, "line": spec["line"], "instance_id": inst.instance_id,
                              "base_sha256": base_sha, "final": child_info, "program": program_checks(arch),
                              "U_meaning": spec["u_meaning"], "G_meaning": spec["g_meaning"]}
    record["partition"] = check_partition(base, child, spec["u_tensors"], spec["g_tensors"])
    both = params_sha256(replace_tensors(base, child, spec["u_tensors"] + spec["g_tensors"]))
    if both != child_sha:
        raise UError(f"{arch}: the base with U and G replaced hashes to {both}, the final is {child_sha}")
    record["base_with_U_and_G_is_final"] = True
    units: tuple[int, ...] | None = None
    if arch == "vit":
        units = M3.units_from_delta(np.asarray(child[M3.L1W], np.float64) - np.asarray(base[M3.L1W], np.float64))
        if units != M3.EXPECTED_UNITS:
            raise UError(f"vit: block-2 units {units}, M03's {M3.EXPECTED_UNITS}")
        record["units"] = list(units)
        record["m03_support_changed_entries"] = M3.check_support(base, child, units)
    nets: dict[str, dict[str, np.ndarray]] = {}
    record["arms"] = {}
    for arm in arms:
        if arm == "base_plus_U":
            net = replace_tensors(base, child, spec["u_tensors"])
            how = "base with U's tensors set to the final's"
        elif arm == "final_minus_U":
            net = replace_tensors(base, child, spec["g_tensors"])
            how = "base with G's tensors set to the final's"
        else:
            assert units is not None
            net = M3.build_network(base, child, units, "F")
            how = 'm03_m06.build_network(base, final, units, "F")'
        sha = params_sha256(net)
        if sha != spec["expected"][arm]:
            raise UError(f"{arch} {arm}: {sha}, expected {spec['expected'][arm]}")
        row: dict[str, Any] = {"params_sha256": sha, "construction": how, "n_changed_vs_base": count_changed(base, net),
                               "n_changed_vs_final": count_changed(child, net)}
        if arch == "vit" and arm in spec["m03"]:  # the tensor-level arm is M03's component network
            assert units is not None
            name, parts = spec["m03"][arm]
            m03_sha = params_sha256(M3.build_network(base, child, units, parts))
            if m03_sha != sha:
                raise UError(f"vit {arm}: M03's {name} builds {m03_sha}, the arm is {sha}")
            row["m03_network"] = name
        record["arms"][arm] = row
        nets[arm] = net
    return inst, nets, record


# ----------------------------------------------------------------------------------------------------------------
# Subcommands


def cmd_build(args: argparse.Namespace) -> int:
    started = time.time()
    _, _, record = build_arms(args.arch)
    out = {"schema": SCHEMA, "kind": "build", **record, "seconds": round(time.time() - started, 1),
           "provenance": provenance()}
    sha = R.write_json(Path(args.out) / "build" / f"{args.arch}.json", out)
    print(f"build {args.arch}: " + ", ".join(f"{arm} {row['params_sha256'][:16]}" for arm, row in
                                              record["arms"].items()) + f"; sha256 {sha}", flush=True)
    return 0


def build_record(out_dir: Path, arch: str) -> tuple[dict[str, Any], str]:
    path = out_dir / "build" / f"{arch}.json"
    if not path.exists():
        raise UError(f"{path} is missing: run build first")
    record = json.loads(path.read_text(encoding="utf-8"))
    for arm in DESIGN[arch]["arms"]:
        if record["arms"][arm]["params_sha256"] != DESIGN[arch]["expected"][arm]:
            raise UError(f"{path}: {arm} {record['arms'][arm]['params_sha256']}")
    return record, R.file_sha256(path)


def cmd_predict(args: argparse.Namespace) -> int:
    from weightcode.evalkit.devreadout import predict

    started = time.time()
    out_dir = Path(args.out)
    if args.arm not in DESIGN[args.arch]["arms"]:
        raise UError(f"{args.arch} has no arm {args.arm}")
    folder = out_dir / "preds" / args.arch / args.arm
    target = folder / f"{args.set}.npy"
    if target.exists() or target.with_suffix(".json").exists():
        raise UError(f"{target} exists")
    record, record_sha = build_record(out_dir, args.arch)
    inst, nets, _ = build_arms(args.arch, args.arm)
    params = nets[args.arm]
    data, set_info = R.load_set(args.arch, args.set, args.conditions_sha256)
    pred = np.asarray(predict(inst.forward, params, data["x"], R.BATCH), dtype=np.int64)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(target, pred)
    meta = {"schema": SCHEMA, "kind": "predict", "arch": args.arch, "arm": args.arm, "set": args.set,
            "network_info": {"params_sha256": record["arms"][args.arm]["params_sha256"],
                             "build_json_sha256": record_sha},
            "set_info": set_info, "forward_sha256": inst.forward_sha256, "batch": R.BATCH, "n": int(len(pred)),
            "preds_sha256": R.file_sha256(target), "seconds": round(time.time() - started, 1),
            "provenance": provenance()}
    R.write_json(folder / f"{args.set}.json", meta)
    print(f"predict {args.arch} {args.arm} {args.set}: n {len(pred)} {meta['seconds']} s", flush=True)
    return 0


def r38_inputs() -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str, str], dict[str, Any]],
                          dict[str, Any], list[dict[str, Any]]]:
    """R38's base and final predictions (checked against its outputs.sha256 lines) and t07_table.json."""
    run = R.runs_root() / R38_RUN_REL
    R.require_sha(run / "t07_table.json", R38_TABLE_SHA256)
    table = json.loads((run / "t07_table.json").read_text(encoding="utf-8"))
    lines = []
    preds: dict[tuple[str, str, str], np.ndarray] = {}
    meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arch in DESIGN:
        for name in R38_NETWORKS:
            for s in C.SETS:
                npy, info = (run / "preds" / arch / name / f"{s}{suffix}" for suffix in (".npy", ".json"))
                npy_sha = R.file_sha256(npy)
                lines.append(f"{npy_sha}  runs/{R38_RUN_REL}/preds/{arch}/{name}/{s}.npy")
                lines.append(f"{R.file_sha256(info)}  runs/{R38_RUN_REL}/preds/{arch}/{name}/{s}.json")
                meta[(arch, name, s)] = json.loads(info.read_text(encoding="utf-8"))
                preds[(arch, name, s)] = np.load(npy)
                if (meta[(arch, name, s)]["preds_sha256"] != npy_sha or preds[(arch, name, s)].dtype != np.int64
                        or preds[(arch, name, s)].shape != (meta[(arch, name, s)]["n"],)):
                    raise UError(f"{npy}: the array does not match its JSON")
    digest = preds_lines_digest(lines)
    equal = len(lines) == R38_PREDS_LINES and digest == R38_PREDS_LINES_SHA256
    checks = [{"check": f"R38 preds: {len(lines)} files equal R38's outputs.sha256 lines (digest of the sorted lines)",
               "equal": equal, "differences": {} if equal else {"digest": [digest, R38_PREDS_LINES_SHA256],
                                                                 "n_lines": [len(lines), R38_PREDS_LINES]}}]
    return preds, meta, table, checks


def cell(y: np.ndarray, base_pred: np.ndarray, pred: np.ndarray, index: np.ndarray) -> dict[str, Any]:
    """T07's cell: the change against the base in points with its paired interval, the label and the readout."""
    from weightcode.evalkit.sealed import split_readouts

    ro = split_readouts({"y": y}, base_pred, pred, None, R.N_CLASSES)
    diff = (pred == y).astype(np.int64) - (base_pred == y).astype(np.int64)
    boot = R.boot_contrast(diff, index)
    return {"delta_points": boot["point"], "ci95": boot["ci95"], "boot_sd": boot["boot_sd"],
            "label": R.label(boot["point"], boot["ci95"]), "fixed": int(ro["fixed"]), "broken": int(ro["broken"]),
            "worst_class_delta_points": 100.0 * float(ro["worst_class_delta"]), "base_acc": float(ro["base_acc"]),
            "child_acc": float(ro["child_acc"]), "n": int(ro["n"]), "delta": 100.0 * float(ro["delta_acc"])}


def m03_lines() -> dict[str, dict[str, Any]]:
    path = R.runs_root() / M03_METRICS_REL
    R.require_sha(path, M03_METRICS_SHA256)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {line["network"]: line for line in lines if line.get("experiment") == "M03"}


def t07_record_checks(final_cells: Mapping[tuple[str, str], Mapping[str, Any]], t07_table: Mapping[str, Any]
                      ) -> list[dict[str, Any]]:
    """The final's cells, from R38's predictions, equal t07_table.json's rows; S_harm and S_help re-derived from the
    record equal the fixed lists."""
    t07_rows = {(r["arch"], r["set"]): r for r in t07_table["rows"] if r["network"] == FINAL}
    checks: list[dict[str, Any]] = []
    for arch in DESIGN:
        for s in C.SETS:
            ours, rec = final_cells[(arch, s)], t07_rows[(arch, s)]
            diffs = same_fields(ours, rec, ("fixed", "broken", "n"),
                                ("delta_points", "boot_sd", "worst_class_delta_points", "base_acc", "child_acc"))
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=FLOAT_ATOL) for a, b in zip(ours["ci95"], rec["ci95"])):
                diffs["ci95"] = [ours["ci95"], rec["ci95"]]
            if ours["label"] != rec["label"]:
                diffs["label"] = [ours["label"], rec["label"]]
            checks.append({"check": f"final {arch} {s} = t07_table.json row", "equal": not diffs,
                           "differences": diffs})
        derived = {key: tuple(s for s in C.SETS if s not in EXCLUDED_FROM_S and t07_rows[(arch, s)]["label"] == lab)
                   for key, lab in (("S_harm", "harm"), ("S_help", "help"))}
        checks.append({"check": f"{arch} S_harm and S_help from t07_table.json = the fixed lists",
                       "equal": derived == S_SETS[arch],
                       "differences": {} if derived == S_SETS[arch] else {"derived": derived}})
    return checks


def m03_checks(vit_cells: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each E11 arm's cells on contrast_s5 and clean equal M03's vs_N0 readouts of its network (fixed, broken and n
    exactly; base_acc, child_acc and delta within ``FLOAT_ATOL``)."""
    m03 = m03_lines()
    checks: list[dict[str, Any]] = []
    for arm, (name, _) in DESIGN["vit"]["m03"].items():
        line = m03[name]
        if line["params_sha256"] != DESIGN["vit"]["expected"][arm]:
            checks.append({"check": f"M03 {name} is vit {arm}", "equal": False,
                           "differences": {"params_sha256": [line["params_sha256"], DESIGN["vit"]["expected"][arm]]}})
            continue
        for s in REPRO_SETS:
            record = line["readouts"]["vs_N0"]["contrast_s5" if s == "contrast_s5" else "clean"]
            diffs = same_fields(vit_cells[(arm, s)], record, M03_FIELDS_INT, M03_FIELDS_FLOAT)
            checks.append({"check": f"vit {arm} {s} = M03 {name} vs_N0", "equal": not diffs, "differences": diffs})
    return checks


def cmd_table(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    y = {arch: R.clean_rows(arch).y for arch in DESIGN}
    if not np.array_equal(y["vit"], y["r18"]):
        raise UError("the two architectures' clean V_select labels differ")
    n = len(y["vit"])
    index = R.boot_index(n)
    r38_preds, r38_meta, t07_table, checks = r38_inputs()
    builds = {arch: build_record(out_dir, arch) for arch in DESIGN}
    preds: dict[tuple[str, str, str], np.ndarray] = {}
    for arch in DESIGN:
        for arm in DESIGN[arch]["arms"]:
            for s in C.SETS:
                path = out_dir / "preds" / arch / arm / f"{s}.npy"
                info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                R.require_sha(path, info["preds_sha256"])
                ref = r38_meta[(arch, "base", s)]
                same = (info["network_info"]["params_sha256"] == DESIGN[arch]["expected"][arm]
                        and info["network_info"]["build_json_sha256"] == builds[arch][1]
                        and info["set_info"]["x_sha256"] == ref["set_info"]["x_sha256"]
                        and info["forward_sha256"] == ref["forward_sha256"] and info["batch"] == ref["batch"]
                        and info["n"] == n)
                if not same:
                    raise UError(f"{path}: not the built arm, or other images, forward or batch than R38's base")
                preds[(arch, arm, s)] = np.load(path)
        for name in R38_NETWORKS:
            for s in C.SETS:
                preds[(arch, "final" if name == FINAL else "base", s)] = r38_preds[(arch, name, s)]
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arch in DESIGN:
        for arm in ("final",) + DESIGN[arch]["arms"]:
            for s in C.SETS:
                cells[(arch, arm, s)] = cell(y[arch], preds[(arch, "base", s)], preds[(arch, arm, s)], index)
    # Checks: T07's rows of the final, the fixed S sets, and M03's readouts of the E11 arms.
    checks += t07_record_checks({(arch, s): cells[(arch, "final", s)] for arch in DESIGN for s in C.SETS}, t07_table)
    checks += m03_checks({(arm, s): cells[("vit", arm, s)] for arm in DESIGN["vit"]["m03"] for s in REPRO_SETS})
    ok = all(c["equal"] for c in checks)
    base_out = {"schema": SCHEMA, "kind": "table", "checks": checks, "all_checks_equal": ok,
                "builds": {arch: {"build_json_sha256": sha, "arms": rec["arms"], "partition": rec["partition"]}
                           for arch, (rec, sha) in builds.items()},
                "r38": {"run": f"runs/{R38_RUN_REL}", "t07_table_sha256": R38_TABLE_SHA256,
                        "outputs_sha256_file_sha256": R38_OUTPUTS_SHA256,
                        "preds_lines_sha256": R38_PREDS_LINES_SHA256},
                "m03_metrics": {"path": f"runs/{M03_METRICS_REL}", "sha256": M03_METRICS_SHA256}}
    if not ok:
        R.write_json(out_dir / "t07u_checks_FAILED.json", {**base_out, "provenance": provenance()})
        print("checks failed: " + json.dumps([c for c in checks if not c["equal"]]), flush=True)
        return 1
    rows: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    out_readings: dict[str, Any] = {}
    for arch in DESIGN:
        for s in C.SETS:
            c = {arm: (preds[(arch, arm, s)] == y[arch]).astype(np.int64)
                 for arm in ("base", "final") + DESIGN[arch]["arms"]}
            for arm in ("final",) + DESIGN[arch]["arms"]:
                row = {"arch": arch, "arm": arm, "set": s, "group": C.group_of(s), "severity": C.severity_of(s),
                       **{k: v for k, v in cells[(arch, arm, s)].items() if k != "delta"},
                       "share_of_final": share(cells[(arch, arm, s)]["delta_points"],
                                               cells[(arch, "final", s)]["delta_points"])}
                if arm != "final":
                    boot = R.boot_contrast(c["final"] - c[arm], index)
                    row["final_minus_arm_points"], row["final_minus_arm_ci95"] = boot["point"], boot["ci95"]
                rows.append(row)
            boot = R.boot_contrast(additivity_values(c["final"], c["base_plus_U"], c["final_minus_U"], c["base"]),
                                   index)
            residuals.append({"arch": arch, "set": s, "residual_points": boot["point"], "ci95": boot["ci95"],
                              "final_delta_points": cells[(arch, "final", s)]["delta_points"],
                              "base_plus_U_delta_points": cells[(arch, "base_plus_U", s)]["delta_points"],
                              "final_minus_U_delta_points": cells[(arch, "final_minus_U", s)]["delta_points"],
                              "reading": "descriptive"})
        labels = {arm: {s: cells[(arch, arm, s)]["label"] for s in C.SETS} for arm in ("final",) + DESIGN[arch]["arms"]}
        deltas = {arm: {s: cells[(arch, arm, s)]["delta_points"] for s in C.SETS}
                  for arm in ("final",) + DESIGN[arch]["arms"]}
        out_readings[arch] = readings(arch, labels, deltas, S_SETS[arch]["S_harm"], S_SETS[arch]["S_help"])
    out = {**base_out, "bootstrap": {"seed": R.BOOT_SEED, "generator": "numpy default_rng",
                                     "n_resamples": R.BOOT_RESAMPLES, "n_images": n,
                                     "interval": "percentile 2.5 and 97.5 (numpy linear)",
                                     "resample_unit": "image; one index matrix for every contrast"},
           "rules": {"help": f">= +{R.HELP_POINTS} point and CI above 0", "harm": f"<= -{R.HELP_POINTS} and CI below 0",
                     "share": "delta(arm) / delta(final), None where delta(final) = 0",
                     "additivity_residual": "delta(final) - delta(base_plus_U) - delta(final_minus_U), descriptive"},
           "sets": list(C.SETS), "S_sets": {a: {k: list(v) for k, v in d.items()} for a, d in S_SETS.items()},
           "arms": {arch: list(DESIGN[arch]["arms"]) for arch in DESIGN}, "rows": rows,
           "additivity": residuals, "readings": out_readings,
           "evidence": "development evidence (V_select rows, seed 100), descriptive", "provenance": provenance()}
    sha = R.write_json(out_dir / "t07u_table.json", out)
    for arch in DESIGN:
        arms = ("final",) + DESIGN[arch]["arms"]
        print(f"\n{arch}: change in points vs base (95% CI) [label]; residual = final - plus_U - minus_U; n = {n}")
        print("set".ljust(22) + "".join(a.rjust(30) for a in arms) + "residual".rjust(12))
        for s in C.SETS:
            line = "".join(f"{cells[(arch, a, s)]['delta_points']:+.2f} ({cells[(arch, a, s)]['ci95'][0]:+.2f},"
                           f"{cells[(arch, a, s)]['ci95'][1]:+.2f}) {cells[(arch, a, s)]['label'][:4]}".rjust(30)
                           for a in arms)
            (res,) = [r for r in residuals if r["arch"] == arch and r["set"] == s]
            print(s.ljust(22) + line + f"{res['residual_points']:+.2f}".rjust(12))
        rd = out_readings[arch]
        for effect in ("harm", "help"):
            o = rd["off_target"][effect]
            print(f"{arch} {effect}: base_plus_U {o['base_plus_U_carries']} of {o['n_sets']}, final_minus_U "
                  f"{o['final_minus_U_carries']} of {o['n_sets']} -> {o['reading']}")
        print(f"{arch} own gain: share {rd['own_gain']['share_s5']:.3f} -> {rd['own_gain']['reading']}")
        if "carrier_F" in rd:
            cf = rd["carrier_F"]
            print(f"{arch} carrier_F: harm {cf['harm_on_S_harm']} of {cf['n_S_harm']}, help {cf['help_on_S_help']} of "
                  f"{cf['n_S_help']} -> named {cf['F_named_as_contributor']}")
    print(f"\nchecks all equal: {ok}; t07u_table.json sha256 {sha}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "predict", "table"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the output directory of this T07-U run")
        if name in ("build", "predict"):
            p.add_argument("--arch", choices=sorted(DESIGN), required=True)
        if name == "predict":
            p.add_argument("--arm", choices=ARMS, required=True)
            p.add_argument("--set", choices=C.SETS, required=True)
            p.add_argument("--conditions-sha256", required=True, help="sha256 of t07_conditions.json")
    args = parser.parse_args(argv)
    return {"build": cmd_build, "predict": cmd_predict, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
