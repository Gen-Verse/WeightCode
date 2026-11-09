"""PREREG T07 (applicability range): the image sets, and the dev rows of the conditions no dev row exists for yet.

    <cifarc venv>/bin/python -I scripts/t07/t07_conditions.py make --official MAKE_CIFAR_C_PY --frost-dir FROST_DIR \\
        --clean-dir $WEIGHTCODE_PRIVATE/cifar10/splits --dev-rows $WEIGHTCODE_PRIVATE/e10_staging/dev_rows_20261104 \\
        --out-dir $WEIGHTCODE_PRIVATE/t07_conditions/v1

The T07 sets are the 4000 clean V_select rows that every E10, E11 and E12 instance reads (split sha256 75b0a7c4...,
the same rows in cifar10, cifar10_r20, cifar10_vl7 and cifar10_r18), each with one corruption applied: ``SETS``,
26 in all. Four exist (``STORED``: the E10 dev rows at severity 5, of which E11 and E12 hold byte copies); the other 21
(``NEW``: gaussian_noise and contrast at severities 1 to 4, and the 13 corruptions of the official file that no line
of this project used, at severity 5) are made here with the code that made the stored four: the official
make_cifar_c.py of hendrycks/robustness at the pinned commit, run as written by ``scripts/e10/cifar10c_official``
(``corrupt``: numpy's global RandomState seeded once per set, images in row order) in the pinned environment of E10
(Python 3.11, numpy 1.26.4, scikit-image 0.22.0, Wand 0.6.13 on ImageMagick 7; the project environment has none of
these libraries). Each set's seed is ``seed_of``: 70000 + 100 k + severity, k the corruption's 1-based place in the
official file's list (``REGISTRY``), fixed before any set exists.

Before writing anything, ``make`` regenerates the four stored V_select arrays with E10's seeds (1003, 2003, 3003,
5003) and requires their bytes to hash to the x_sha256 of E10's dev_rows.json: the code, the environment and the
clean rows are the ones that made the dev rows of record. ``frost`` reads the official repository's frost images from
its working directory; ``--frost-dir`` holds the six files of the pinned commit, each checked against its git blob
SHA1 (the commit's tree) and sha256 (``FROST_FILES``). glass_blur and gaussian_blur are not made: as written they call
skimage's ``gaussian(..., multichannel=True)``, which scikit-image 0.22 removed, so they raise; porting them would
change the official code (``NOT_RUNNABLE``).

Output: ``<out-dir>/<set>.npz`` with the single array ``x`` (NCHW uint8, the clean rows' order) and
``t07_conditions.json`` (per set its seed, file and array sha256, median MAD from clean; the reproduction check; the
official file, the frost files and the libraries). Labels and row ids are not copied: the readout pairs each array with
the clean V_select's y and idx. Nothing sealed is read, no official CIFAR-10-C test file is opened. Refuses to
overwrite. The constants here are imported by ``scripts/t07/t07_range.py`` (the project environment), so this module
imports only numpy and the standard library at its top.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "weightcode.t07.conditions.v1"
REGISTRY = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
            "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
            "speckle_noise", "gaussian_blur", "spatter", "saturate")
"""The official file's corruptions in the order of its list ``d`` (lines 421 to 441: the 15 of CIFAR-10-C, then the
four extra ones)."""
NOT_RUNNABLE = {
    "glass_blur": "calls skimage.filters.gaussian(..., multichannel=True); scikit-image 0.22 removed multichannel",
    "gaussian_blur": "calls skimage.filters.gaussian(..., multichannel=True); scikit-image 0.22 removed multichannel",
}
DEV_USED = ("gaussian_noise", "defocus_blur", "snow", "contrast")
"""Official corruptions some line of the project read dev rows of (all at severity 5): E10 all four (and its stream,
control (a), F4, T09-A); E11, E12, the E11 stream, T16, F1, F1s, F2 and K1 gaussian_noise and contrast. E1 to E4 used
their own gaussian_noise_s3 and contrast_s3 transcriptions (weightcode/data/conditions.py), not the official code."""
OWN = ("gaussian_noise", "contrast")
"""The corruptions of the T07 finals (E11 and E12 B, seed 100)."""
OWN_NEW_SEVERITIES = (1, 2, 3, 4)
SEVERITY = 5
UNSEEN = tuple(name for name in REGISTRY if name not in DEV_USED and name not in NOT_RUNNABLE)
"""The 13 corruptions of the official file no line used, made at severity 5."""
SEED_BASE = 70000
E10_SELECT_SEEDS = {"gaussian_noise": 1003, "defocus_blur": 2003, "snow": 3003, "contrast": 5003}
"""scripts/e10/cifar10c_make_dev.py DEV_SEEDS for V_select (the reproduction check)."""
STORED = tuple(f"{name}_s{SEVERITY}" for name in DEV_USED)
FROST_FILES = {
    "frost1.png": ("c7b78a17938df51894497b4782decbd82532dd5f",
                   "ff9f907860bd7a835d459e32f9d588062b7f61ee267343cc7222b56753a14755"),
    "frost2.png": ("48f7a861ffa41b6d7496b701fef96d5edf739282",
                   "fe211a89b336999c207a852ce05818d4545d0b57c5beadd824b4cc9d9a9b6137"),
    "frost3.png": ("d47f9d25f41251ee9a66b294c0bbfad6053017c9",
                   "2d0d50b4a9bb213f38b024ef7768731bb83cc08d2f26b5766bbc167cdfa0e504"),
    "frost4.jpg": ("f8b0c413176d70150b593e029d84b4a88c21dd4b",
                   "3f8b91ca1a9fa7167b09e773da53f5ae60d0a1fd88f02a783f6e328a72887f6e"),
    "frost5.jpg": ("95dc9056926d8201df760535f9bb9112f012e862",
                   "5fc6a19df4a429ba68abdcc8f8a4278d4c9f81c9ccafd2c92ab0c8cf8992ebd2"),
    "frost6.jpg": ("14e5d58e762a5d0808df9fa6494fd6d78ee4409b",
                   "1f92b2f48408748085b68dd81d816ef239f42cef3029c25d041fcb6760fb4f25"),
}
"""ImageNet-C/create_c/frost{1..6} at hendrycks/robustness 8190fe3 (git blob SHA1 from the commit's tree, sha256 of the
downloaded bytes). ``frost`` draws np.random.randint(5), so frost6.jpg is never read; all six are checked."""
CLEAN = "clean"
OUT_JSON = "t07_conditions.json"


class ConditionsError(RuntimeError):
    """An input, a check or an output does not match what T07 fixes."""


def set_name(name: str, severity: int) -> str:
    return f"{name}_s{int(severity)}"


def seed_of(name: str, severity: int) -> int:
    """The generator seed of a new set: 70000 + 100 k + severity, k = 1 + REGISTRY.index(name)."""
    if name not in REGISTRY or not 1 <= int(severity) <= 5:
        raise ConditionsError(f"no seed for {name} at severity {severity}")
    return SEED_BASE + 100 * (REGISTRY.index(name) + 1) + int(severity)


def new_sets() -> list[tuple[str, str, int]]:
    """(set, corruption, severity) of the 21 sets made here, in the order they are made."""
    rows = [(set_name(name, s), name, s) for name in OWN for s in OWN_NEW_SEVERITIES]
    rows += [(set_name(name, SEVERITY), name, SEVERITY) for name in UNSEEN]
    return rows


NEW = tuple(row[0] for row in new_sets())
SETS = ((CLEAN,) + tuple(set_name("gaussian_noise", s) for s in range(1, 6))
        + tuple(set_name("contrast", s) for s in range(1, 6))
        + tuple(set_name(name, SEVERITY) for name in DEV_USED if name not in OWN)
        + tuple(set_name(name, SEVERITY) for name in UNSEEN))
"""The 26 T07 sets in table order: clean; each own corruption at severities 1 to 5; the other two E10 conditions;
the 13 unseen corruptions."""


def group_of(name: str) -> str:
    """clean, own_gaussian_noise, own_contrast, e10_other (defocus_blur, snow) or unseen."""
    if name == CLEAN:
        return CLEAN
    corruption = name.rsplit("_s", 1)[0]
    if corruption in OWN:
        return f"own_{corruption}"
    if corruption in DEV_USED:
        return "e10_other"
    if corruption in UNSEEN:
        return "unseen"
    raise ConditionsError(f"{name} is not a T07 set")


def severity_of(name: str) -> int | None:
    return None if name == CLEAN else int(name.rsplit("_s", 1)[1])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def x_sha256(x: np.ndarray) -> str:
    """sha256 of the C-order bytes of an image array (E10's dev_rows.json x_sha256)."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check_frost_dir(frost_dir: Path) -> dict[str, Any]:
    out = {}
    for name, (blob, sha) in FROST_FILES.items():
        path = Path(frost_dir) / name
        if not path.is_file():
            raise ConditionsError(f"{path} is missing")
        got = (git_blob_sha1(path), file_sha256(path))
        if got != (blob, sha):
            raise ConditionsError(f"{path}: blob {got[0]} sha256 {got[1]}, expected {blob} {sha}")
        out[name] = {"git_blob_sha1": blob, "sha256": sha}
    return out


def corrupt_set(official: Mapping[str, Any], name: str, x: np.ndarray, seed: int, severity: int, frost_dir: Path,
                corrupt: Any) -> np.ndarray:
    """``corrupt`` of the official code; frost runs with the frost directory as working directory."""
    if name != "frost":
        return corrupt(official, name, x, seed, severity=severity)
    here = os.getcwd()
    os.chdir(frost_dir)
    try:
        return corrupt(official, name, x, seed, severity=severity)
    finally:
        os.chdir(here)


def median_mad(out: np.ndarray, x: np.ndarray) -> float:
    return float(np.median(np.abs(out.astype(np.int16) - x.astype(np.int16)).reshape(len(x), -1).mean(axis=1)))


def cmd_make(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e10"))
    import cifar10c_official as O  # noqa: PLC0415, the E10 wrapper of the official file

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise ConditionsError(f"{out_dir} exists; nothing written")
    frost = check_frost_dir(Path(args.frost_dir))
    official = O.load_official(Path(args.official))
    missing = [name for name in REGISTRY if not callable(official.get(name))]
    if missing:
        raise ConditionsError(f"the official file does not define {missing}")
    clean_path = Path(args.clean_dir) / "V_select.npz"
    with np.load(clean_path) as arrays:
        x = np.ascontiguousarray(arrays["x"])
    dev_rows_path = Path(args.dev_rows) / "dev_rows.json"
    dev_rows = json.loads(dev_rows_path.read_text(encoding="utf-8"))
    clean_files = {row["clean_file_sha256"] for key, row in dev_rows["rows"].items() if key.endswith("/V_select")}
    if clean_files != {file_sha256(clean_path)}:
        raise ConditionsError(f"{clean_path} is not the clean V_select file E10's dev rows were made from")
    record: dict[str, Any] = {
        "schema": SCHEMA, "spec": args.spec,
        "official": {"commit": O.OFFICIAL_COMMIT, "file": O.OFFICIAL_FILE, "sha256": O.OFFICIAL_SHA256},
        "libraries": O.library_versions(), "frost_files": frost, "not_runnable": NOT_RUNNABLE,
        "clean": {"path": str(clean_path), "file_sha256": file_sha256(clean_path), "x_sha256": x_sha256(x),
                  "n": int(len(x))},
        "dev_rows": {"path": str(dev_rows_path), "sha256": file_sha256(dev_rows_path)},
        "seed_rule": "70000 + 100 * (1 + REGISTRY.index(corruption)) + severity", "registry": list(REGISTRY),
        "reproduction": {}, "sets": {}}
    for name, seed in E10_SELECT_SEEDS.items():  # the stored dev rows, regenerated before anything is written
        started = time.time()
        out = corrupt_set(official, name, x, seed, SEVERITY, Path(args.frost_dir), O.corrupt)
        expected = dev_rows["rows"][f"{name}/V_select"]
        ok = x_sha256(out) == expected["x_sha256"] and seed == expected["seed"]
        record["reproduction"][set_name(name, SEVERITY)] = {
            "seed": seed, "x_sha256": x_sha256(out), "expected_x_sha256": expected["x_sha256"], "equal": ok,
            "seconds": round(time.time() - started, 1)}
        print(f"reproduction {name} s{SEVERITY} seed {seed}: equal {ok}", flush=True)
        if not ok:
            raise ConditionsError(f"{name}: the regenerated V_select does not hash to E10's dev rows; nothing written")
    staging = out_dir.with_name(out_dir.name + ".partial")
    staging.mkdir(parents=True)
    only = set(args.only.split(",")) if args.only else None
    for name, corruption, severity in new_sets():
        if only is not None and name not in only:
            continue
        seed = seed_of(corruption, severity)
        started = time.time()
        out = corrupt_set(official, corruption, x, seed, severity, Path(args.frost_dir), O.corrupt)
        target = staging / f"{name}.npz"
        np.savez(target, x=out)
        record["sets"][name] = {"corruption": corruption, "severity": severity, "seed": seed, "n": int(len(out)),
                                "file_sha256": file_sha256(target), "x_sha256": x_sha256(out),
                                "median_mad_from_clean": median_mad(out, x),
                                "seconds": round(time.time() - started, 1)}
        print(f"{name}: seed {seed}, median MAD {record['sets'][name]['median_mad_from_clean']:.2f}, "
              f"{record['sets'][name]['seconds']} s", flush=True)
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    (staging / OUT_JSON).write_text(text, encoding="utf-8")
    staging.rename(out_dir)
    print(f"-> {out_dir / OUT_JSON} sha256 {hashlib.sha256(text.encode()).hexdigest()}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make", help="the reproduction check, then the 21 new sets")
    make.add_argument("--official", required=True, help="local copy of make_cifar_c.py (sha256 checked)")
    make.add_argument("--frost-dir", required=True, help="the six frost files of the pinned commit")
    make.add_argument("--clean-dir", required=True, help="$WEIGHTCODE_PRIVATE/cifar10/splits")
    make.add_argument("--dev-rows", required=True, help="E10's dev rows directory (dev_rows.json)")
    make.add_argument("--out-dir", required=True)
    make.add_argument("--spec", required=True, help="the PREREG commit the sets are made under")
    make.add_argument("--only", default="", help="comma-separated set names (a test run); default all 21")
    args = parser.parse_args(argv)
    return cmd_make(args)


if __name__ == "__main__":
    raise SystemExit(main())
