"""PREREG K1 control input: eight rule draws of the replay rows from the pinned official CIFAR-10-C code.

Runs in E10's pinned environment (``scripts/e10/cifar10c_official.py``), not the project's. Reads the 64 replay images
(clean train rows, ``replay_v1.npz``, x only) and applies the official ``gaussian_noise`` at severity 5 eight times,
draw k with seed ``1005 + k`` (k = 0 to 7, fixed in PREREG K1 before this script ran): ``O.corrupt`` seeds numpy's
generator once per draw and corrupts the 64 images in row order, exactly as ``scripts/f1s/f1s_s5_rows.py`` does for its
one draw, so draw 0 (seed 1005) equals that file's gaussian rows (checked when ``--f1s-rows`` is given). Writes
``<out-dir>/gaussian_noise_s5_rule_draws.npz`` (array ``x``: uint8 [8, 64, ...] in the replay rows' layout) and
``k1_rule_rows.json`` (schema weightcode.k1.rule_rows.v1; the replay content hash, each draw's seed and x sha256, the file's
sha256). No condition row and no label is read. Refuses to overwrite.

    .venv/bin/python -I scripts/k1/k1_rule_rows.py \\
        --official $WEIGHTCODE_PRIVATE/external/robustness_8190fe3/make_cifar_c.py \\
        --replay-dir $WEIGHTCODE_PRIVATE/cifar10_vl7/replay --out-dir $WEIGHTCODE_PRIVATE/cifar10_vl7/k1_rule \\
        --f1s-rows $WEIGHTCODE_PRIVATE/cifar10_vl7/f1s_s5/s5_rows.json
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

FUNCTION = "gaussian_noise"
SEED0 = 1005
DRAWS = 8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--f1s-rows", type=Path, help="f1s_s5/s5_rows.json: draw 0 must equal its gaussian_noise rows")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; nothing written")
    official = O.load_official(args.official)
    replay_json = json.loads((args.replay_dir / "replay_v1.json").read_text())
    with np.load(args.replay_dir / "replay_v1.npz", allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["x"])
    shape = x.shape
    x4 = x.reshape(len(x), 3, 32, 32)
    draws, entries = [], []
    for k in range(DRAWS):
        out = O.corrupt(official, FUNCTION, x4, SEED0 + k).reshape(shape)
        draws.append(out)
        entries.append({"k": k, "seed": SEED0 + k, "x_sha256": hashlib.sha256(out.tobytes()).hexdigest()})
        print(f"draw {k}: seed {SEED0 + k}, x sha256 {entries[-1]['x_sha256'][:12]}", flush=True)
    stacked = np.stack(draws)
    same_as_f1s = None
    if args.f1s_rows is not None:
        f1s = json.loads(args.f1s_rows.read_text())
        same_as_f1s = entries[0]["x_sha256"] == f1s["rows"][FUNCTION]["x_sha256"]
        print(f"draw 0 equals the F1s s5 gaussian rows: {same_as_f1s}", flush=True)
        if not same_as_f1s:
            raise SystemExit("draw 0 differs from the F1s s5 rows; nothing written")
    args.out_dir.mkdir(parents=True)
    target = args.out_dir / f"{FUNCTION}_s5_rule_draws.npz"
    np.savez(target, x=stacked)
    record = {"schema": "weightcode.k1.rule_rows.v1", "severity": O.SEVERITY,
              "official": {"commit": O.OFFICIAL_COMMIT, "file": O.OFFICIAL_FILE, "sha256": O.OFFICIAL_SHA256},
              "libraries": O.library_versions(),
              "replay": {"content_sha256": replay_json["content_sha256"], "n": int(len(x)),
                         "file_sha256": O.file_sha256(args.replay_dir / "replay_v1.npz")},
              "rows": {FUNCTION: {"file": target.name, "file_sha256": O.file_sha256(target),
                                  "seeds": [e["seed"] for e in entries], "draws": entries,
                                  "layout": "uint8 [draw, replay row, ...], replay rows' own layout",
                                  "draw0_equals_f1s_s5_rows": same_as_f1s}}}
    (args.out_dir / "k1_rule_rows.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
