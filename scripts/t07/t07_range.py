"""PREREG T07 (applicability range): the E11 and E12 B finals of record (seed 100) and their matched fine-tuning
references, read on the 26 T07 image sets against their base, with no rerun of any program or training.

Subcommands (all CPU, numpy; run on NSCC by scripts/store/t07_main.sh):

* ``identity --arch A --network N``: N's readout on its own condition's V_select and on the clean companion, by
  ``weightcode.harness.clean.dev_readout_joint`` as the records were made, must equal the record (the final's info.json, or
  the F1 floor -20 readout of the fine-tuning pick) on every field; writes ``identity/<A>_<N>.json``.
* ``predict --arch A --network N --set S``: predictions of N on set S (``weightcode.evalkit.devreadout.predict``, batch 1024,
  as every dev readout); writes ``preds/<A>/<N>/<S>.npy`` and a JSON beside it.
* ``gates --final F --set S``: T07-G, the trigger rates and output norms of the E11 final F's gates on set S
  (``weightcode.analysis.vit_internal``; the gates are the ones its T08 diagonal cell printed and built); writes
  ``gates/<F>_<S>.json``.
* ``table``: every readout of the T07 bullet from the predictions, with paired image bootstrap intervals, checked
  against the identity records; writes ``t07_table.json`` and prints the tables.

Networks per architecture (``ARCHS``): ``base`` (the seed-100 student both instances share), ``llm_<c>`` (the B final of
condition c, rebuilt by ``devreadout.final_child`` from its session) and ``ft_<c>`` (F1's best kept pick at floor -20
for c, rebuilt from its saved candidate patch as ``devreadout.readout_retention`` rebuilds it: the E10 program limits
without size limits). Sets (``t07_conditions.SETS``): the clean V_select rows, and the same rows under each
corruption; every set is paired with the clean rows' labels and dataset indices (row i is one image throughout).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t07_conditions as C  # noqa: E402

SCHEMA = "weightcode.t07.range.v1"
SPEC = "docs/PREREG_v0_DRAFT.md section 0.1, 'T07 and M05 phase 1'; NSCC R38-t07-m05"
BATCH = 1024
N_CLASSES = 10
BOOT_SEED = 20261107
BOOT_RESAMPLES = 1000
HELP_POINTS = 1.0
"""A change counts as help when it is at least +1.0 point and its 95% interval lies above 0; harm when at most -1.0
and the interval lies below 0; neutral otherwise."""
HEADROOM_MIN_POINTS = 2.0
"""The recovered share (change over the base's loss from clean) is reported only where that loss is >= 2 points."""
FLOOR = -20
DEV_ROWS_REL = "e10_staging/dev_rows_20261104/dev_rows.json"
DEV_ROWS_SHA256 = "b4dd5b083d47ca7cd591d55908d58306526eb1de34be2e46b5b8fba70312ad7d"
STORED_DATASET = "cifar10_r20"
"""The stored severity-5 sets are read from E10's dataset (all four conditions; E11 and E12 hold byte copies of two)
and checked against E10's dev_rows.json x_sha256."""
CLEAN_SPLIT_SHA256 = "75b0a7c4eea96a3ea9a27f4bac08f0de529a2d313c347c569bc8e9debde756cb"
CONDITIONS_DIR_REL = "t07_conditions/v1"

ARCHS: dict[str, dict[str, Any]] = {
    "vit": {
        "line": "E11 (ViT-Lite-7/4)", "dataset": "cifar10_vl7",
        "base_sha256": "fd7f6d4b80145c98bc7c6a9cbc2c8e440e3f45e328168df2a8b55bec3f7c5b6a",
        "instances": {"gaussian_noise": "i00802425", "contrast": "i3eb3fc78"},
        "finals": {
            "gaussian_noise": {"session": "sc918e11dceef637a",
                               "run_sha256": "8561fe2e5fdf95b354c7c87de4f1ae3b17efd5aa37118bc63d998229e935329d",
                               "child_sha256": "e40c20786ae91c904cfd7a195addb7e03b6601b955d731518da1ff54bc01c251",
                               "info_sha256": "a762cc3813c980578410a264ac4a30d062f24fb5bdeda5699aff179def2570d8"},
            "contrast": {"session": "s760cb52ce82fb937",
                         "run_sha256": "7c647d6d099f2cffcd61e5cc8743cc7321f0300ef228b8d0d5b134c6941a4910",
                         "child_sha256": "07943b6df1f9424be9b7c76cc32630f4c194b79e44b8ad00d9a89749aa0308e7",
                         "info_sha256": "c4028fbc78d527749c2eb4e0391e8838e0a10c7d53af0cce353a4f1470791816"}},
        "ft": {
            "gaussian_noise": {"run": "baselines/i00802425/lora_r4/f1/0/run.json", "pick": "q4",
                               "run_sha256": "e6a7c1f0fc8ffd46f4a0150bad6ddc49cdb36a32c2080dc4c4c1ef5c79e31f26",
                               "child_sha256": "bed61accac2f72db1f40149aead73e8a8074ec6f81305583f836d5ec0878e310",
                               "readout": "devreadouts/f1_i00802425_gaussian_noise_s5_floor20_dev_nscc_20261105.jsonl",
                               "readout_sha256": "4b89dc1698261cfaf12cf26f0713209f6fd4d605b4354cd2ee6583e3cb6716c1"},
            "contrast": {"run": "baselines/i3eb3fc78/ft_full_l2sp/f1/0/run.json", "pick": "q5",
                         "run_sha256": "9c1ba12a9ba268f4346a69edc64bed370b95df0b8e1121f4b336336c29d64682",
                         "child_sha256": "f5bd4352477c7cffcafde3ed52092ca96fd593e5fd31231c42a412593e95da90",
                         "readout": "devreadouts/f1_i3eb3fc78_contrast_s5_floor20_dev_nscc_20261105.jsonl",
                         "readout_sha256": "24221959161b43756e008acc5b0d422420a81ce01b5ccc3d2c8dc6ecf2fe4e06"}},
    },
    "r18": {
        "line": "E12 (ResNet-18, RESIZE 128)", "dataset": "cifar10_r18",
        "base_sha256": "6bf3fd0b11caa8d001f6db30dd34b694e02c5255c51898592c6e7f0369c44cb0",
        "instances": {"gaussian_noise": "i26d23d3e", "contrast": "ie3840b28"},
        "finals": {
            "gaussian_noise": {"session": "s2abaefab30d20b42",
                               "run_sha256": "15195a608214a4bba59c8d90c0981dde89300d2009b8796667c1331697efdda5",
                               "child_sha256": "ab961aa19217aabbf6b3298ff5546a1cffa10c3406f6f9af26f6dbf240712713",
                               "info_sha256": "80ff4a11a1ba2dc58e9f87f38522bfd47e924bf62fd7b2ee5bc3b0708109fef5"},
            "contrast": {"session": "sccd12677dcdd125b",
                         "run_sha256": "5f17da0b7a90dbeee5dd7f9b3d522768090fc6eb404fcd6fc67ea4d9cafe8da4",
                         "child_sha256": "9792a0d3d26ee935322c26c5df99cb32ceb083a46f930dc083732ca2ec1f50f5",
                         "info_sha256": "1c6855920266485dfd428cc0515b9b811001f7882c7d759e6a549c40b2dfec13"}},
        "ft": {
            "gaussian_noise": {"run": "baselines/i26d23d3e/lora_r16/f1/0/run.json", "pick": "q0",
                               "run_sha256": "d5b34c4fa9c48d0639f356ed4462f94ecdac49bc1ea3fbd13bbacb422c6d1125",
                               "child_sha256": "c00a51488bf06b0d015df58c73be0f6defbc814cab6228f603bcde79a963f5a8",
                               "readout": "devreadouts/f1e12_i26d23d3e_gaussian_noise_s5_floor20_dev_nscc_20261105.jsonl",
                               "readout_sha256": "c2fb079593df741bf05cfd74c94102fc41618991455cd82b619b195e242ef779"},
            "contrast": {"run": "baselines/ie3840b28/lora_r16/f1/0/run.json", "pick": "q0",
                         "run_sha256": "cf22419514a058c61b0b343fbf0ab92ce59b85d3965a1289ad271a6305b23c5c",
                         "child_sha256": "b932ef1dcbb1b19b13159d21899b69fe09fa40ff00050dce4ae0ba6eef4e2915",
                         "readout": "devreadouts/f1e12_ie3840b28_contrast_s5_floor20_dev_nscc_20261105.jsonl",
                         "readout_sha256": "dd7cca2248e77e6a020802d7654c48ce22b92ee5a3f90c2342abaeb6ebb8c647"}},
    },
}
NETWORKS = ("base",) + tuple(f"{kind}_{c}" for kind in ("llm", "ft") for c in C.OWN)
GATE_FINALS = {
    "gaussian_noise": {"cell": "t08/program/sc918e11dceef637a/transfer_3c1cebc35ef0_i00802425",
                       "cell_sha256": "741e42ec20a887990182d91accf61d426f13c9a320876adb41ffe30db31f9735"},
    "contrast": {"cell": "t08/program/s760cb52ce82fb937/transfer_207b312d4295_i3eb3fc78",
                 "cell_sha256": "cb230e17a7c6e8b89e7f6c666ee126e605e769246789d65acb6f58072ccdd2d8"},
}
"""T07-G: the E11 finals' gates, read from their T08 diagonal cells (child sha256 = the final's, checked)."""
RECORD_FIELDS = ("delta_acc", "fixed", "broken", "worst_class_delta", "base_acc", "child_acc", "n")


class RangeError(RuntimeError):
    """An input or a rebuilt network does not match its record."""


# ----------------------------------------------------------------------------------------------------------------
# Pure parts (tests/test_t07_range.py)


def label(delta_points: float, ci95: Sequence[float]) -> str:
    """help, harm or neutral (``HELP_POINTS`` and the interval's side of 0)."""
    if delta_points >= HELP_POINTS and ci95[0] > 0.0:
        return "help"
    if delta_points <= -HELP_POINTS and ci95[1] < 0.0:
        return "harm"
    return "neutral"


def boot_index(n: int, seed: int = BOOT_SEED, n_resamples: int = BOOT_RESAMPLES) -> np.ndarray:
    """One index matrix (n_resamples, n) from ``default_rng(seed)``; it serves every contrast of the table."""
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def boot_contrast(values: np.ndarray, index: np.ndarray) -> dict[str, Any]:
    """Point and percentile 95% interval (numpy linear) of mean(values) in points, resampling images by ``index``."""
    values = np.asarray(values, np.float64)
    boot = 100.0 * values[index].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"point": 100.0 * float(values.mean()), "ci95": [float(low), float(high)],
            "boot_sd": float(boot.std(ddof=1))}


def same_readout(ours: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Fields of ``RECORD_FIELDS`` that differ (abs 1e-12) between two readout dicts."""
    diffs = {}
    for key in RECORD_FIELDS:
        a, b = ours.get(key), record.get(key)
        if a is None or b is None or not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
            diffs[key] = [a, b]
    return diffs


def recovered_share(delta_points: float, base_clean_acc: float, base_set_acc: float) -> tuple[float, float | None]:
    """The base's loss from clean in points and, where it is >= ``HEADROOM_MIN_POINTS``, the change over it."""
    headroom = 100.0 * (float(base_clean_acc) - float(base_set_acc))
    return headroom, (float(delta_points) / headroom if headroom >= HEADROOM_MIN_POINTS else None)


# ----------------------------------------------------------------------------------------------------------------
# Files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise RangeError(f"{path} hashes to {actual}, the record says {expected}")
    return actual


def runs_root() -> Path:
    from weightcode.common.paths import runs_dir

    return Path(runs_dir())


def private_root() -> Path:
    from weightcode.common.paths import private_dir

    return Path(private_dir())


def jsonable(value: Any) -> Any:
    from weightcode.baselines.fit.runner import jsonable as convert

    return convert(value)


def write_json(path: Path, data: Any) -> str:
    if path.exists():
        raise RangeError(f"{path} exists; records are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(data), indent=1, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(path)
    return hashlib.sha256(text.encode()).hexdigest()


def provenance() -> dict[str, Any]:
    from weightcode.baselines.fit.runner import git_state

    return {"script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
            "vit_internal_sha256": file_sha256(Path(sys.modules["weightcode.analysis.vit_internal"].__file__))
            if "weightcode.analysis.vit_internal" in sys.modules else None,
            "git": git_state(), "numpy": np.__version__, "python": sys.version.split()[0],
            "threads": os.environ.get("OPENBLAS_NUM_THREADS"), "created_at": datetime.now(UTC).isoformat(),
            "host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"), "spec": SPEC}


# ----------------------------------------------------------------------------------------------------------------
# Networks


@functools.lru_cache(maxsize=None)
def instance(arch: str, condition: str) -> Any:
    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import instances_root

    spec = ARCHS[arch]
    inst = load_instance(instances_root() / spec["instances"][condition])
    if (inst.private.get("split_role") != "dev" or inst.private.get("condition") != f"{condition}_s5"
            or inst.private.get("dataset") != spec["dataset"]):
        raise RangeError(f"{inst.instance_id} is not the dev {spec['dataset']} {condition}_s5 instance")
    if inst.params_sha256 != spec["base_sha256"]:
        raise RangeError(f"{inst.instance_id}: base {inst.params_sha256}, expected {spec['base_sha256']}")
    return inst


def network(arch: str, name: str) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    """(the instance the network belongs to, its parameters, provenance), each rebuilt network hash-checked."""
    import weightcode.evalkit.devreadout as D
    from weightcode.common.identity import params_sha256
    from weightcode.evalkit.e10 import e10_program_limits
    from weightcode.patch.limits import dense_delta_bytes

    if name == "base":
        insts = [instance(arch, c) for c in C.OWN]
        return insts[0], dict(insts[0].params), {"params_sha256": insts[0].params_sha256,
                                                  "source": "the instances' model.npz (shared base)"}
    kind, condition = name.split("_", 1)
    inst = instance(arch, condition)
    if kind == "llm":
        spec = ARCHS[arch]["finals"][condition]
        session = runs_root() / "sessions" / spec["session"]
        require_sha(session / "run.json", spec["run_sha256"])
        run = json.loads((session / "run.json").read_text(encoding="utf-8"))
        child = D.final_child(run, session, inst)
        source = f"runs/sessions/{spec['session']} (devreadout.final_child)"
    elif kind == "ft":
        spec = ARCHS[arch]["ft"][condition]
        run_path = runs_root() / spec["run"]
        require_sha(run_path, spec["run_sha256"])
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if not str(run.get("limits_source", "")).startswith("e10_program_limits"):
            raise RangeError(f"{run_path}: limits_source {run.get('limits_source')!r}, expected the E10 limits")
        picked = D.retention_final(run, FLOOR)
        if picked["final"]["patch_id"] != spec["pick"]:
            raise RangeError(f"{run_path}: the floor {FLOOR} pick is {picked['final']['patch_id']}, not {spec['pick']}")
        swapped = {**run, "final": picked["final"]}
        expected = e10_program_limits(inst.task["limits"], len(inst.params),
                                      dense_delta_bytes(v.shape for v in inst.params.values()))
        limits = D._program_baseline_limits(swapped, run_path.parent, expected, "e10", "E10")
        child = D.final_child(swapped, run_path.parent, inst, limits)
        source = f"runs/{spec['run']} pick {spec['pick']} (retention_final floor {FLOOR}, E10 program limits)"
    else:
        raise RangeError(f"unknown network {name}")
    sha = params_sha256(child)
    if sha != spec["child_sha256"]:
        raise RangeError(f"{arch} {name}: rebuilt child {sha}, expected {spec['child_sha256']}")
    return inst, child, {"params_sha256": sha, "source": source}


# ----------------------------------------------------------------------------------------------------------------
# Sets


@functools.lru_cache(maxsize=None)
def clean_rows(arch: str) -> Any:
    from weightcode.evalkit.companions import load_clean_rows

    rows = load_clean_rows(instance(arch, C.OWN[0]), "V_select")
    if rows.clean_sha256 != CLEAN_SPLIT_SHA256:
        raise RangeError(f"{arch}: clean V_select {rows.clean_sha256}, expected {CLEAN_SPLIT_SHA256}")
    return rows


def conditions_record(expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    folder = private_root() / CONDITIONS_DIR_REL
    require_sha(folder / C.OUT_JSON, expected_sha256)
    record = json.loads((folder / C.OUT_JSON).read_text(encoding="utf-8"))
    if sorted(record["sets"]) != sorted(C.NEW) or not all(r["equal"] for r in record["reproduction"].values()):
        raise RangeError(f"{folder / C.OUT_JSON}: not the 21 new sets with a passed reproduction check")
    if record["clean"]["x_sha256"] != C.x_sha256(clean_rows("vit").x):
        raise RangeError("the new sets were made from other clean rows than the readout's clean V_select")
    for name, row in record["sets"].items():
        if row["seed"] != C.seed_of(row["corruption"], row["severity"]) or row["n"] != len(clean_rows("vit").y):
            raise RangeError(f"{name}: seed or row count differs from the T07 rule")
    return folder, record


def load_set(arch: str, name: str, conditions_sha256: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The set's arrays (x of the set, y and idx of the clean rows) and where x came from, hash-checked."""
    rows = clean_rows(arch)
    if name == C.CLEAN:
        return rows.arrays(), {"source": f"{ARCHS[arch]['dataset']} clean V_select", "x_sha256": C.x_sha256(rows.x),
                               "split_sha256": rows.clean_sha256}
    if name in C.STORED:
        from weightcode.evalkit.targets import deployment_splits, load_checked_split

        splits = deployment_splits(STORED_DATASET, name)
        arrays = load_checked_split(splits.v_select, splits.v_select_sha256)
        require_sha(private_root() / DEV_ROWS_REL, DEV_ROWS_SHA256)
        dev_rows = json.loads((private_root() / DEV_ROWS_REL).read_text(encoding="utf-8"))
        corruption = name.rsplit("_s", 1)[0]
        if C.x_sha256(arrays["x"]) != dev_rows["rows"][f"{corruption}/V_select"]["x_sha256"]:
            raise RangeError(f"{name}: {STORED_DATASET} V_select x differs from E10's dev rows")
        if not (np.array_equal(arrays["y"], rows.y) and np.array_equal(arrays["idx"], rows.idx)):
            raise RangeError(f"{name}: rows are not the clean V_select rows")
        x, info = arrays["x"], {"source": f"{STORED_DATASET}/conditions/{name} V_select (E10 dev rows)",
                                 "split_sha256": splits.v_select_sha256}
    elif name in C.NEW:
        folder, record = conditions_record(conditions_sha256)
        path = folder / f"{name}.npz"
        require_sha(path, record["sets"][name]["file_sha256"])
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["x"]:
                raise RangeError(f"{path}: arrays {archive.files}, expected x only")
            x = np.ascontiguousarray(archive["x"])
        if C.x_sha256(x) != record["sets"][name]["x_sha256"]:
            raise RangeError(f"{path}: x does not hash as recorded")
        info = {"source": f"{CONDITIONS_DIR_REL}/{name}.npz", "file_sha256": record["sets"][name]["file_sha256"],
                "seed": record["sets"][name]["seed"], "conditions_json_sha256": conditions_sha256}
    else:
        raise RangeError(f"{name} is not a T07 set")
    if x.shape != rows.x.shape or x.dtype != np.uint8:
        raise RangeError(f"{name}: x is {x.dtype}{x.shape}, the clean rows {rows.x.dtype}{rows.x.shape}")
    return {"x": x, "y": rows.y, "idx": rows.idx}, {**info, "x_sha256": C.x_sha256(x)}


# ----------------------------------------------------------------------------------------------------------------
# Subcommands


def record_readouts(arch: str, name: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """The readouts of record of a final or fine-tuning pick: (condition V_select, clean V_select, source)."""
    kind, condition = name.split("_", 1)
    if kind == "llm":
        spec = ARCHS[arch]["finals"][condition]
        path = runs_root() / "devreadouts" / "learn" / spec["session"] / "info.json"
        require_sha(path, spec["info_sha256"])
        final = json.loads(path.read_text(encoding="utf-8"))["final"]
        if final["child_sha256"] != spec["child_sha256"]:
            raise RangeError(f"{path}: final child {final['child_sha256']}")
        return final["vselect"], final["vselect_clean"], str(path)
    spec = ARCHS[arch]["ft"][condition]
    path = runs_root() / spec["readout"]
    require_sha(path, spec["readout_sha256"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = [r for r in rows if Path(r["run"]).as_posix().endswith(spec["run"]) and
            r["selection"]["patch_id"] == spec["pick"]]
    if len(hits) != 1:
        raise RangeError(f"{path}: {len(hits)} rows for {spec['run']} pick {spec['pick']}")
    return hits[0]["readouts"], hits[0]["readouts_clean"], str(path)


def cmd_identity(args: argparse.Namespace) -> int:
    from weightcode.harness.clean import dev_readout_joint

    started = time.time()
    inst, child, info = network(args.arch, args.network)
    record = dev_readout_joint(inst, child, BATCH)
    rec_select, rec_clean, source = record_readouts(args.arch, args.network)
    diffs = {"select": same_readout(record["readouts"], rec_select),
             "clean": same_readout(record["readouts_clean"], rec_clean)}
    equal = not diffs["select"] and not diffs["clean"] and record["clean_split_sha256"] == CLEAN_SPLIT_SHA256
    out = {"schema": SCHEMA, "kind": "identity", "arch": args.arch, "network": args.network, "network_info": info,
           "instance_id": inst.instance_id, "record_source": source, "equal": equal, "differences": diffs,
           "readouts": record["readouts"], "readouts_clean": record["readouts_clean"],
           "clean_split_sha256": record["clean_split_sha256"], "seconds": round(time.time() - started, 1),
           "provenance": provenance()}
    sha = write_json(Path(args.out) / "identity" / f"{args.arch}_{args.network}.json", out)
    print(f"identity {args.arch} {args.network}: equal {equal} {diffs if not equal else ''} sha256 {sha}", flush=True)
    return 0 if equal else 1


def cmd_predict(args: argparse.Namespace) -> int:
    from weightcode.evalkit.devreadout import predict

    started = time.time()
    inst, params, info = network(args.arch, args.network)
    data, set_info = load_set(args.arch, args.set, args.conditions_sha256)
    pred = np.asarray(predict(inst.forward, params, data["x"], BATCH), dtype=np.int64)
    folder = Path(args.out) / "preds" / args.arch / args.network
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{args.set}.npy"
    if target.exists():
        raise RangeError(f"{target} exists")
    np.save(target, pred)
    meta = {"schema": SCHEMA, "kind": "predict", "arch": args.arch, "network": args.network, "set": args.set,
            "network_info": info, "set_info": set_info, "forward_sha256": inst.forward_sha256, "batch": BATCH,
            "n": int(len(pred)), "preds_sha256": file_sha256(target), "accuracy": float(np.mean(pred == data["y"])),
            "seconds": round(time.time() - started, 1), "provenance": provenance()}
    write_json(folder / f"{args.set}.json", meta)
    print(f"predict {args.arch} {args.network} {args.set}: acc {meta['accuracy']:.4f} n {len(pred)} "
          f"{meta['seconds']} s", flush=True)
    return 0


def gate_spec(final: str) -> tuple[dict[str, Any], str]:
    spec = GATE_FINALS[final]
    cell = runs_root() / spec["cell"]
    require_sha(cell.with_suffix(".json"), spec["cell_sha256"])
    record = json.loads(cell.with_suffix(".json").read_text(encoding="utf-8"))
    if record["child_sha256"] != ARCHS["vit"]["finals"][final]["child_sha256"]:
        raise RangeError(f"{cell}: child {record['child_sha256']} is not the {final} final")
    return record, (cell / "run" / "stdout.txt").read_text(encoding="utf-8")


def cmd_gates(args: argparse.Namespace) -> int:
    import weightcode.analysis.vit_internal as V

    started = time.time()
    base_inst = instance("vit", args.final)
    _, child, info = network("vit", f"llm_{args.final}")
    record, stdout = gate_spec(args.final)
    data, set_info = load_set("vit", args.set, args.conditions_sha256)
    ns = V.forward_namespace(base_inst.forward)
    sums = V.Sums()
    if args.final == "gaussian_noise":
        gates = V.parse_gaussian_gates(stdout)
        gate_info: Any = V.check_ramp_gates(base_inst.params, child, gates)
        blocks = sorted({g.block for g in gates})
    else:
        gate_info = V.contrast_gate(base_inst.params, child, stdout)
        blocks = [gate_info["block"]]
    chunk = int(ns["CHUNK"])
    parity = None
    for start in range(0, len(data["y"]), chunk):
        x, y = data["x"][start:start + chunk], data["y"][start:start + chunk]
        tr = V.trace(ns, child, x, blocks)
        if parity is None:
            parity = V.parity(ns, child, x, tr.logits)
        if args.final == "gaussian_noise":
            for g in gates:
                V.ramp_gate_sums(sums, f"g{g.block}_", tr.mlp_in[g.block], child, g, ns["_gelu"], y, N_CLASSES)
        else:
            V.contrast_gate_sums(sums, "gate_", tr.mlp_in[gate_info["block"]], child, gate_info["units"],
                                 gate_info["gate_unit"], ns["_gelu"], y, N_CLASSES)
    if args.final == "gaussian_noise":
        stats = {f"{g.kind}_block{g.block}": V.ramp_gate_finish(sums, f"g{g.block}_", with_classes=True)
                 for g in gates}
    else:
        stats = {"gate": V.contrast_gate_finish(sums, "gate_", with_classes=True)}
    out = {"schema": SCHEMA, "kind": "gates", "final": args.final, "set": args.set, "network_info": info,
           "set_info": set_info, "gates": gate_info,
           "stats": stats, "parity_max_abs_logit_diff": parity, "n": int(len(data["y"])),
           "seconds": round(time.time() - started, 1), "provenance": provenance()}
    sha = write_json(Path(args.out) / "gates" / f"{args.final}_{args.set}.json", out)
    print(f"gates {args.final} {args.set}: {round(time.time() - started, 1)} s sha256 {sha}", flush=True)
    return 0


def main_gate_rate(final: str, stats: Mapping[str, Any]) -> float:
    """T07-G's main rate: the gaussian final's patch gate above threshold on patch tokens; the contrast final's gate
    open on patch tokens."""
    if final == "gaussian_noise":
        (key,) = [k for k in stats if k.startswith("patch_block")]
        return float(stats[key]["patch"]["frac_above_threshold"])
    return float(stats["gate"]["rate_patch_tokens"])


def cmd_table(args: argparse.Namespace) -> int:
    from weightcode.analysis.vit_internal import spearman
    from weightcode.evalkit.sealed import split_readouts

    out_dir = Path(args.out)
    y = {arch: clean_rows(arch).y for arch in ARCHS}
    if not np.array_equal(y["vit"], y["r18"]):
        raise RangeError("the two architectures' clean V_select labels differ")
    n = len(y["vit"])
    index = boot_index(n)
    checks: list[dict[str, Any]] = []
    for arch in ARCHS:
        for name in NETWORKS[1:]:
            path = out_dir / "identity" / f"{arch}_{name}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            checks.append({"check": f"identity {arch} {name}", "equal": bool(record["equal"])})
    preds: dict[tuple[str, str, str], np.ndarray] = {}
    meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arch in ARCHS:
        for name in NETWORKS:
            for s in C.SETS:
                path = out_dir / "preds" / arch / name / f"{s}.npy"
                info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                require_sha(path, info["preds_sha256"])
                preds[(arch, name, s)] = np.load(path)
                meta[(arch, name, s)] = info
    for s in C.SETS:  # both architectures read the same image bytes on every set
        if len({meta[(arch, nm, s)]["set_info"]["x_sha256"] for arch in ARCHS for nm in NETWORKS}) != 1:
            raise RangeError(f"set {s}: networks read different images")
    rows: list[dict[str, Any]] = []
    for arch in ARCHS:
        base_acc = {s: float(np.mean(preds[(arch, "base", s)] == y[arch])) for s in C.SETS}
        for name in NETWORKS[1:]:
            for s in C.SETS:
                ro = split_readouts({"y": y[arch]}, preds[(arch, "base", s)], preds[(arch, name, s)], None, N_CLASSES)
                correct_diff = ((preds[(arch, name, s)] == y[arch]).astype(np.int64)
                                - (preds[(arch, "base", s)] == y[arch]).astype(np.int64))
                boot = boot_contrast(correct_diff, index)
                headroom, share = recovered_share(boot["point"], base_acc[C.CLEAN], base_acc[s])
                rows.append({"arch": arch, "network": name, "set": s, "group": C.group_of(s),
                             "severity": C.severity_of(s), "delta_points": boot["point"], "ci95": boot["ci95"],
                             "boot_sd": boot["boot_sd"], "label": label(boot["point"], boot["ci95"]),
                             "fixed": int(ro["fixed"]), "broken": int(ro["broken"]),
                             "worst_class_delta_points": 100.0 * float(ro["worst_class_delta"]),
                             "base_acc": float(ro["base_acc"]), "child_acc": float(ro["child_acc"]),
                             "n": int(ro["n"]), "base_loss_from_clean_points": headroom,
                             "recovered_share": share})
                kind, condition = name.split("_", 1)
                if s in (C.CLEAN, f"{condition}_s5"):  # the per-set predictions reproduce the record as well
                    rec_select, rec_clean, _ = record_readouts(arch, name)
                    diffs = same_readout(ro, rec_clean if s == C.CLEAN else rec_select)
                    checks.append({"check": f"table readout {arch} {name} {s} = record", "equal": not diffs,
                                   "differences": diffs})
    llm_vs_ft: list[dict[str, Any]] = []
    for arch in ARCHS:
        for condition in C.OWN:
            for s in C.SETS:
                a = (preds[(arch, f"llm_{condition}", s)] == y[arch]).astype(np.int64)
                b = (preds[(arch, f"ft_{condition}", s)] == y[arch]).astype(np.int64)
                boot = boot_contrast(a - b, index)
                llm_vs_ft.append({"arch": arch, "condition": condition, "set": s, "group": C.group_of(s),
                                  "llm_minus_ft_points": boot["point"], "ci95": boot["ci95"], "n": n})
    gate_rows: list[dict[str, Any]] = []
    associations: dict[str, Any] = {}
    for final in GATE_FINALS:
        rates, deltas = [], []
        for s in C.SETS:
            g = json.loads((out_dir / "gates" / f"{final}_{s}.json").read_text(encoding="utf-8"))
            if g["network_info"]["params_sha256"] != ARCHS["vit"]["finals"][final]["child_sha256"]:
                raise RangeError(f"gates {final} {s}: not the final")
            (row,) = [r for r in rows if r["arch"] == "vit" and r["network"] == f"llm_{final}" and r["set"] == s]
            rate = main_gate_rate(final, g["stats"])
            gate_rows.append({"final": final, "set": s, "group": C.group_of(s), "main_gate_rate": rate,
                              "stats": g["stats"], "delta_points": row["delta_points"], "label": row["label"]})
            if s != C.CLEAN:
                rates.append(rate)
                deltas.append(row["delta_points"])
        associations[final] = {"spearman_delta_vs_main_gate_rate_25_sets": spearman(deltas, rates),
                               "n_sets": len(rates), "reading": "descriptive only"}
    ok = all(c["equal"] for c in checks)
    out = {"schema": SCHEMA, "kind": "table", "checks": checks, "all_checks_equal": ok,
           "bootstrap": {"seed": BOOT_SEED, "generator": "numpy default_rng", "n_resamples": BOOT_RESAMPLES,
                         "n_images": n, "interval": "percentile 2.5 and 97.5 (numpy linear)",
                         "resample_unit": "image; one index matrix for every contrast"},
           "rules": {"help": f">= +{HELP_POINTS} point and CI above 0", "harm": f"<= -{HELP_POINTS} and CI below 0",
                     "recovered_share": f"change / (base clean acc - base set acc), loss >= {HEADROOM_MIN_POINTS}"},
           "sets": list(C.SETS), "networks": list(NETWORKS), "rows": rows, "llm_minus_ft": llm_vs_ft,
           "gates": gate_rows, "gate_associations": associations, "provenance": provenance()}
    sha = write_json(out_dir / "t07_table.json", out)
    for arch in ARCHS:
        print(f"\n{arch}: change in points vs base (95% CI) [label]; n = {n} per set")
        print("set".ljust(26) + "".join(name.rjust(30) for name in NETWORKS[1:]))
        for s in C.SETS:
            cells = []
            for name in NETWORKS[1:]:
                (r,) = [r for r in rows if r["arch"] == arch and r["network"] == name and r["set"] == s]
                cells.append(f"{r['delta_points']:+.2f} ({r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}) {r['label'][:4]}")
            print(s.ljust(26) + "".join(c.rjust(30) for c in cells))
    print(f"\nchecks all equal: {ok}; t07_table.json sha256 {sha}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("identity", "predict", "gates", "table"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the output directory of this T07 run")
        if name in ("identity", "predict"):
            p.add_argument("--arch", choices=sorted(ARCHS), required=True)
            p.add_argument("--network", choices=NETWORKS if name == "predict" else NETWORKS[1:], required=True)
        if name == "gates":
            p.add_argument("--final", choices=sorted(GATE_FINALS), required=True)
        if name in ("predict", "gates"):
            p.add_argument("--set", choices=C.SETS, required=True)
            p.add_argument("--conditions-sha256", required=True, help="sha256 of t07_conditions.json")
    args = parser.parse_args(argv)
    return {"identity": cmd_identity, "predict": cmd_predict, "gates": cmd_gates, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
