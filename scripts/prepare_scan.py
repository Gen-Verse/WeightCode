"""Install the SCAN raw copy and build the stored E13 data (PREREG E13; ``weightcode.data.scan_build``).

Steps, each refusing to overwrite:

* ``--install-raw CHECKOUT``: copy the recorded files of a SCAN git checkout at the recorded commit to
  $WEIGHTCODE_PRIVATE/scan/raw (read-only, with provenance.json);
* ``--base``: forward.py and $WEIGHTCODE_PRIVATE/scan/dataset.json;
* ``--condition NAME``: the private splits, the sealed part ($WEIGHTCODE_SEALED, written once) and dataset.json of one
  condition (addprim_jump or around_right);
* ``--verify --condition NAME``: re-derive every split from the raw files and compare hashes (the sealed file is not
  read).

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/prepare_scan.py --install-raw /path/to/SCAN
    .venv/bin/python scripts/prepare_scan.py --base
    .venv/bin/python scripts/prepare_scan.py --condition addprim_jump
    .venv/bin/python scripts/prepare_scan.py --condition addprim_jump --verify
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from weightcode.data import scan as S
from weightcode.data import scan_build as B
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[1]


def install(checkout: Path) -> dict:
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    if head != S.SCAN_COMMIT:
        raise SystemExit(f"{checkout} is at {head}, not the recorded commit {S.SCAN_COMMIT}")
    provenance = B.install_raw(checkout)
    return {"raw": str(B.raw_dir()), "commit": provenance["commit"], "files": len(provenance["files"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-raw", type=Path, metavar="CHECKOUT")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--condition", choices=S.CONDITIONS)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.install_raw, args.base, args.condition)) != 1:
        raise SystemExit("give exactly one of --install-raw, --base and --condition")
    if args.install_raw:
        summary = install(args.install_raw)
    elif args.base:
        manifest = B.write_base(run_info(REPO_ROOT))
        summary = {"forward_sha256": manifest["forward_sha256"], "forward_constants": manifest["forward_constants"]}
    elif args.verify:
        report = B.verify_condition(args.condition)
        print(json.dumps({key: value["match"] for key, value in report.items()}, indent=2))
        return 0 if all(value["match"] for value in report.values()) else 1
    else:
        manifest = B.write_condition(B.build_condition(args.condition), run_info(REPO_ROOT))
        summary = {"condition": args.condition, "split_sizes": manifest["split_sizes"],
                   "split_sha256": {name: record["sha256"] for name, record in manifest["splits"].items()},
                   "sealed": {key: manifest["test"][key] for key in ("n", "sha256", "part_counts")},
                   "draw": manifest["draw"]}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
