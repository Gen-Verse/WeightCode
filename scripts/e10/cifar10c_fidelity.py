"""PREREG E10 fidelity check, pixels only: the official code on the clean test images against the official files.

    <cifarc venv>/bin/python scripts/e10/cifar10c_fidelity.py --official MAKE_CIFAR_C_PY --test TEST_NPZ \\
        --cifar10c DIR --manifest INSTALL_JSON --out OUT_JSON

The user allowed this one sealed read (2026-11-04 ~16:50 Beijing, "允许只读像素"): the ``x`` array of the clean test
set and the severity-5 rows (40,000 .. 49,999) of the official ``<name>.npy`` files. No label is read (``y`` of the
test file and ``labels.npy`` are never accessed) and no model runs. Only the statistics below are written.

The clean test file is not checked against cifar10's dataset.json: that hash covers x, y and idx, so recomputing it
would read the labels. The record states the hash of the ``x`` array alone (the store's canonical form,
``params_sha256({"x": x})``); the deterministic corruptions' match to the official files is what ties these pixels,
in this order, to the official test set.

For each of gaussian_noise, defocus_blur, snow and contrast, and fog (snow's substitute, computed in the same read so
that the sealed files are opened once):
- deterministic (defocus_blur, contrast): the mean absolute pixel difference between the code's output and the
  official file over all 10,000 x 3,072 values. Pass: at most 0.5 grey levels.
- random (gaussian_noise, snow, fog): per image, the mean absolute difference from the clean image; the median over
  the 10,000 images for the code's output and for the official file. Pass: the code's median within 5% of the
  official median.
The snow decision follows the entry: snow if it passes, else fog if fog passes, else no weather condition.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar10c_official as O  # noqa: E402

FIDELITY_SEEDS = {"gaussian_noise": 2026110401, "defocus_blur": 2026110402, "snow": 2026110403,
                  "contrast": 2026110404, "fog": 2026110405}
DETERMINISTIC_MAX_MAD = 0.5
RANDOM_MEDIAN_REL_TOL = 0.05


def _official_rows(path: Path) -> np.ndarray:
    """Severity-5 rows of an official file (N, H, W, C) as NCHW uint8."""
    array = np.load(path, mmap_mode="r")
    if array.shape != (5 * O.N_PER_SEVERITY, 32, 32, 3) or array.dtype != np.uint8:
        raise O.OfficialCodeError(f"{path}: shape {array.shape} {array.dtype}, expected (50000, 32, 32, 3) uint8")
    start = (O.SEVERITY - 1) * O.N_PER_SEVERITY
    return np.ascontiguousarray(np.asarray(array[start:start + O.N_PER_SEVERITY]).transpose(0, 3, 1, 2))


def _per_image_mad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).reshape(a.shape[0], -1).mean(axis=1)


def _x_sha256(x: np.ndarray) -> str:
    """The store's canonical hash (``weightcode.common.identity.params_sha256``) of the single array ``x``."""
    import hashlib

    digest = hashlib.sha256()
    for part in (b"x", x.dtype.str.encode("ascii"), ",".join(str(d) for d in x.shape).encode("ascii")):
        digest.update(part)
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(x).tobytes(order="C"))
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--cifar10c", type=Path, required=True, help="the installed CIFAR-10-C directory")
    parser.add_argument("--manifest", type=Path, required=True, help="the install record with each file's sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; nothing written")
    started = time.time()
    official = O.load_official(args.official)
    with np.load(args.test, allow_pickle=False) as test:
        members = list(test.files)  # the zip directory only
        clean = np.ascontiguousarray(test["x"])  # the pixels only; the labels are never accessed
    if clean.shape != (O.N_PER_SEVERITY, 3, 32, 32) or clean.dtype != np.uint8:
        raise SystemExit(f"{args.test}: x is {clean.shape} {clean.dtype}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = list(O.CONDITIONS) + list(O.SUBSTITUTE.values())
    results: dict[str, dict] = {}
    for name in names:
        path = args.cifar10c / f"{name}.npy"
        sha = O.file_sha256(path)
        if sha != manifest["files"][f"CIFAR-10-C/{name}.npy"]["sha256"]:
            raise SystemExit(f"{path}: sha256 differs from the install record")
        reference = _official_rows(path)
        t0 = time.time()
        ours = O.corrupt(official, name, clean, FIDELITY_SEEDS[name],
                         progress=lambda i, n=name: print(f"{name} {i}", file=sys.stderr, flush=True))
        entry: dict = {"official_file_sha256": sha, "seed": FIDELITY_SEEDS[name], "seconds": time.time() - t0,
                       "n_images": int(clean.shape[0])}
        if name in O.DETERMINISTIC:
            diff = np.abs(ours.astype(np.int16) - reference.astype(np.int16))
            entry.update({
                "kind": "deterministic",
                "mean_abs_diff": float(diff.mean()), "max_abs_diff": int(diff.max()),
                "frac_values_differ": float(np.mean(diff > 0)),
                "per_image_mean_abs_diff_max": float(diff.reshape(diff.shape[0], -1).mean(axis=1).max()),
                "threshold": DETERMINISTIC_MAX_MAD,
            })
            entry["pass"] = entry["mean_abs_diff"] <= DETERMINISTIC_MAX_MAD
        else:
            mine, theirs = _per_image_mad(ours, clean), _per_image_mad(reference, clean)
            q = lambda v: [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75])]  # noqa: E731
            entry.update({
                "kind": "random",
                "code_mad_from_clean_q1_median_q3": q(mine), "official_mad_from_clean_q1_median_q3": q(theirs),
                "code_mean_mad": float(mine.mean()), "official_mean_mad": float(theirs.mean()),
                "median_rel_diff": float(np.median(mine) / np.median(theirs) - 1.0),
                "threshold": RANDOM_MEDIAN_REL_TOL,
            })
            entry["pass"] = abs(entry["median_rel_diff"]) <= RANDOM_MEDIAN_REL_TOL
        results[name] = entry
        print(json.dumps({name: entry}), file=sys.stderr, flush=True)
    weather = "snow" if results["snow"]["pass"] else ("fog" if results["fog"]["pass"] else None)
    record = {
        "schema": "weightcode.e10.cifar10c_fidelity.v1",
        "rule": "PREREG E10 fidelity check, pixels only (no label, no model)",
        "official": {"repository": O.OFFICIAL_REPOSITORY, "file": O.OFFICIAL_FILE, "commit": O.OFFICIAL_COMMIT,
                     "file_last_changed": O.OFFICIAL_FILE_LAST_CHANGED, "sha256": O.OFFICIAL_SHA256},
        "severity": O.SEVERITY, "libraries": O.library_versions(),
        "inputs": {"test": str(args.test), "test_members": members, "test_x_sha256": _x_sha256(clean),
                   "manifest": str(args.manifest),
                   "manifest_sha256": O.file_sha256(args.manifest)},
        "results": results,
        "conditions_pass": {name: results[name]["pass"] for name in names},
        "weather_condition": weather,
        "e10_conditions": [c if c != "snow" else weather for c in O.CONDITIONS
                           if (c != "snow" or weather) and (c == "snow" or results[c]["pass"])],
        "seconds": time.time() - started,
    }
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(text)
    import hashlib

    print(f"-> {args.out} {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
