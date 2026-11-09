"""PREREG E10 dev rows: the official corruption code at severity 5 on E1's held-out splits of the CIFAR-10 train set.

    <cifarc venv>/bin/python scripts/e10/cifar10c_make_dev.py --official MAKE_CIFAR_C_PY --clean-dir CIFAR10_SPLITS \\
        --fidelity FIDELITY_JSON --out-dir STAGING_DIR

For every E10 condition that passed the fidelity check (the record's ``e10_conditions``; snow or its substitute) and
every split of ``SPLITS``, the clean split's images are corrupted by ``cifar10c_official.corrupt`` with that
(condition, split)'s seed from ``DEV_SEEDS`` (one seed each, fixed here before any dev row exists) and written to
``<out-dir>/<condition>/<split>.npz`` with the single array ``x`` (NCHW uint8, the clean rows' order). A JSON record
beside them states the inputs' sha256, the seeds, the official code and the libraries. Labels and row ids are not
copied here: the project's store builder (``weightcode.data.e10``) pairs each array with the clean split's y and idx and
checks the row count. Nothing sealed is read. Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar10c_official as O  # noqa: E402

SPLITS = ("V_evidence", "V_check", "V_select")
# One recorded generator seed per (condition, split), fixed before any dev row was generated.
DEV_SEEDS = {
    ("gaussian_noise", "V_evidence"): 1001, ("gaussian_noise", "V_check"): 1002, ("gaussian_noise", "V_select"): 1003,
    ("defocus_blur", "V_evidence"): 2001, ("defocus_blur", "V_check"): 2002, ("defocus_blur", "V_select"): 2003,
    ("snow", "V_evidence"): 3001, ("snow", "V_check"): 3002, ("snow", "V_select"): 3003,
    ("fog", "V_evidence"): 4001, ("fog", "V_check"): 4002, ("fog", "V_select"): 4003,
    ("contrast", "V_evidence"): 5001, ("contrast", "V_check"): 5002, ("contrast", "V_select"): 5003,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True, help="$WEIGHTCODE_PRIVATE/cifar10/splits")
    parser.add_argument("--fidelity", type=Path, required=True, help="the fidelity record (cifar10c_fidelity.py)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; nothing written")
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    conditions = list(fidelity["e10_conditions"])
    official = O.load_official(args.official)
    staging = args.out_dir.with_name(args.out_dir.name + ".partial")
    staging.mkdir(parents=True)
    record = {"schema": "weightcode.e10.cifar10c_dev_rows.v1", "severity": O.SEVERITY,
              "official": {"commit": O.OFFICIAL_COMMIT, "file": O.OFFICIAL_FILE, "sha256": O.OFFICIAL_SHA256},
              "libraries": O.library_versions(),
              "fidelity": {"path": str(args.fidelity), "sha256": O.file_sha256(args.fidelity)},
              "conditions": conditions, "splits": list(SPLITS), "rows": {}}
    for split in SPLITS:
        clean_path = args.clean_dir / f"{split}.npz"
        with np.load(clean_path) as arrays:
            x = np.ascontiguousarray(arrays["x"])
        clean_sha = O.file_sha256(clean_path)
        for condition in conditions:
            seed = DEV_SEEDS[(condition, split)]
            started = time.time()
            out = O.corrupt(official, condition, x, seed)
            (staging / condition).mkdir(exist_ok=True)
            target = staging / condition / f"{split}.npz"
            np.savez(target, x=out)
            mad = np.abs(out.astype(np.int16) - x.astype(np.int16)).reshape(len(x), -1).mean(axis=1)
            record["rows"][f"{condition}/{split}"] = {
                "seed": seed, "n": int(len(x)), "clean_file_sha256": clean_sha,
                "out_file_sha256": O.file_sha256(target),
                "x_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
                "median_mad_from_clean": float(np.median(mad)), "seconds": round(time.time() - started, 1),
            }
            print(f"{condition}/{split}: {len(x)} rows, median MAD {np.median(mad):.2f}", flush=True)
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    (staging / "dev_rows.json").write_text(text, encoding="utf-8")
    staging.rename(args.out_dir)
    print(f"-> {args.out_dir}/dev_rows.json {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
