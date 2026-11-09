"""PREREG F1s s5 input: the replay rows corrupted by the pinned official CIFAR-10-C code at severity 5.

Runs in E10's pinned environment (``scripts/e10/cifar10c_official.py``), not the project's. Reads the 64 replay
images (clean train rows, ``replay_v1.npz``, x only), applies the official gaussian_noise and contrast functions at
severity 5 (seeds fixed here before any s5 run: gaussian 1005, contrast 5005; contrast draws nothing), and writes
``<out-dir>/<function>_s5_old.npz`` (array ``x``, NCHW-flattened as the evidence rows) plus ``s5_rows.json``. The
record holds the replay content hash (``replay_v1.json``), so a run refuses rows made from other replay images.
Refuses to overwrite.

    .venv/bin/python -I scripts/f1s/f1s_s5_rows.py \\
        --official $WEIGHTCODE_PRIVATE/external/robustness_8190fe3/make_cifar_c.py \\
        --replay-dir $WEIGHTCODE_PRIVATE/cifar10_vl7/replay --out-dir $WEIGHTCODE_PRIVATE/cifar10_vl7/f1s_s5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e10"))
import cifar10c_official as O  # noqa: E402

SEEDS = {"gaussian_noise": 1005, "contrast": 5005}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; nothing written")
    official = O.load_official(args.official)
    replay_json = json.loads((args.replay_dir / "replay_v1.json").read_text())
    with np.load(args.replay_dir / "replay_v1.npz", allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["x"])
    shape = x.shape
    x4 = x.reshape(len(x), 3, 32, 32)
    args.out_dir.mkdir(parents=True)
    record = {"schema": "weightcode.f1s.s5_rows.v1", "severity": O.SEVERITY,
              "official": {"commit": O.OFFICIAL_COMMIT, "file": O.OFFICIAL_FILE, "sha256": O.OFFICIAL_SHA256},
              "libraries": O.library_versions(),
              "replay": {"content_sha256": replay_json["content_sha256"], "n": int(len(x)),
                         "file_sha256": O.file_sha256(args.replay_dir / "replay_v1.npz")},
              "rows": {}}
    for name, seed in SEEDS.items():
        out = O.corrupt(official, name, x4, seed).reshape(shape)
        target = args.out_dir / f"{name}_s5_old.npz"
        np.savez(target, x=out)
        record["rows"][name] = {"file": target.name, "file_sha256": O.file_sha256(target), "seed": seed,
                                "x_sha256": hashlib.sha256(out.tobytes()).hexdigest()}
        print(f"{name}: {len(out)} rows, seed {seed}", flush=True)
    (args.out_dir / "s5_rows.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
