#!/usr/bin/env bash
# Create the sandbox venv: the interpreter that runs the editor's kernel and every update program inside bubblewrap
# (weightcode.harness.runner.SANDBOX_PYTHON = <repo>/.venv-sandbox/bin/python). It holds numpy only, pinned by
# scripts/sandbox_requirements.lock (numpy 2.5.4, as in every session of record).
#
#   bash scripts/setup_sandbox_venv.sh [--python /usr/bin/python3.14] [--target DIR]
#
# The base interpreter must resolve to a file under /usr: the sandbox binds only /usr, /proc, /dev, two tmpfs mounts
# and the venv itself (weightcode.sandbox.host.isolation_argv), so a venv whose python links outside /usr cannot start
# inside it. The sessions of record used the workstation's /usr/bin/python3.14 (CPython 3.14.4).
#
# The target must not exist yet; this script never modifies or replaces an existing venv. Defaults: --python
# /usr/bin/python3.14, --target <repo>/.venv-sandbox. Needs uv on PATH.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=/usr/bin/python3.14
TARGET=$REPO/.venv-sandbox
LOCK=$REPO/scripts/sandbox_requirements.lock
EXPECT_NUMPY=2.5.4

while (( $# )); do
  case $1 in
    --python) BASE=$2; shift 2 ;;
    --target) TARGET=$2; shift 2 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v uv > /dev/null || { echo "uv is not on PATH" >&2; exit 2; }
[[ -x $BASE ]] || { echo "base interpreter $BASE is not an executable file" >&2; exit 2; }
RESOLVED=$(realpath "$BASE")
[[ $RESOLVED == /usr/* ]] || { echo "base interpreter $BASE resolves to $RESOLVED, outside /usr; the sandbox binds only /usr" >&2; exit 2; }
[[ -e $TARGET || -L $TARGET ]] && { echo "$TARGET exists; this script does not modify an existing venv" >&2; exit 2; }
grep -qx "numpy==$EXPECT_NUMPY" "$LOCK" || { echo "$LOCK does not pin numpy==$EXPECT_NUMPY" >&2; exit 2; }

uv venv --no-project --no-config --no-managed-python --no-python-downloads --python "$BASE" "$TARGET"
uv pip install --no-config --python "$TARGET/bin/python" --only-binary :all: --no-deps -r "$LOCK"

# Checks: the venv's python links to the /usr interpreter, the system site is excluded, numpy is the pinned version
# and is the only package.
LINKED=$(realpath "$TARGET/bin/python")
[[ $LINKED == /usr/* ]] || { echo "$TARGET/bin/python resolves to $LINKED, outside /usr" >&2; exit 1; }
grep -qx "include-system-site-packages = false" "$TARGET/pyvenv.cfg" || { echo "$TARGET/pyvenv.cfg includes the system site" >&2; exit 1; }
"$TARGET/bin/python" -I -c "import numpy, sys; v = numpy.__version__; assert v == '$EXPECT_NUMPY', v; print('python', sys.version.split()[0], 'numpy', v)"
PACKAGES=$(uv pip list --no-config --python "$TARGET/bin/python" --format freeze)
[[ $PACKAGES == "numpy==$EXPECT_NUMPY" ]] || { echo "unexpected packages in $TARGET: $PACKAGES" >&2; exit 1; }
echo "sandbox venv ready at $TARGET (base $RESOLVED)"
