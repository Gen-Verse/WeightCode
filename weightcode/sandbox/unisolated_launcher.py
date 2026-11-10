"""Run the update-program driver outside bubblewrap (``weightcode.sandbox.program.UnisolatedRunner``).

Usage::

    python -I -u unisolated_launcher.py --driver PATH --in-dir DIR --out-dir DIR -- <driver arguments>

Inside the sandbox the driver (``program_driver.py``) reads ``/work/in`` and writes ``/work/out``. This launcher loads
the unchanged driver file as a module, points its two directory constants at ``--in-dir`` and ``--out-dir`` and calls
its ``main`` with the remaining arguments. The program therefore sees the same inputs, the same virtual clock and the
same output format as in the sandbox; only the isolation is missing. The launcher imports the standard library only.
"""

from __future__ import annotations

import argparse
import sys
import types


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("driver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    driver_args = args.driver_args[1:] if args.driver_args[:1] == ["--"] else args.driver_args
    with open(args.driver, "rb") as handle:
        source = handle.read()
    driver = types.ModuleType("weightcode_program_driver")
    driver.__file__ = args.driver
    exec(compile(source, args.driver, "exec"), driver.__dict__)
    driver.IN_DIR, driver.OUT_DIR = args.in_dir, args.out_dir  # read by the driver's functions at call time
    return int(driver.main(driver_args))


if __name__ == "__main__":
    sys.exit(main())
