#!/usr/bin/env bash
# Create the sandbox venv: the interpreter that runs the editor's kernel and every update program inside bubblewrap
# (weightcode.harness.runner.SANDBOX_PYTHON = <repo>/.venv-sandbox/bin/python). It holds numpy only, pinned by
# scripts/sandbox_requirements.lock (numpy 2.5.4, as in every session of record).
#
#   bash scripts/setup_sandbox_venv.sh [--python PATH_OR_VERSION] [--target DIR]
#
# Base interpreter (--python): the distribution's /usr/bin/python3.14 when it exists, else a uv-managed CPython 3.14,
# which uv downloads if needed. A path or any uv version request is accepted. The sandbox binds /usr and the venv; a
# base interpreter outside /usr is bound file by file (its executable, standard library and libpython), see
# weightcode.sandbox.host.interpreter_mounts. The sessions of record used /usr/bin/python3.14 (CPython 3.14.4); the
# Python patch version does not enter the programs' numbers, numpy does.
#
# The target must not exist yet; this script never modifies or replaces an existing venv. Default --target
# <repo>/.venv-sandbox. Needs uv on PATH.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=
TARGET=$REPO/.venv-sandbox
LOCK=$REPO/scripts/sandbox_requirements.lock
EXPECT_NUMPY=2.5.4
MANAGED_REQUEST=3.14

while (( $# )); do
  case $1 in
    --python) BASE=$2; shift 2 ;;
    --target) TARGET=$2; shift 2 ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v uv > /dev/null || { echo "uv is not on PATH" >&2; exit 2; }
[[ -e $TARGET || -L $TARGET ]] && { echo "$TARGET exists; this script does not modify an existing venv" >&2; exit 2; }
grep -qx "numpy==$EXPECT_NUMPY" "$LOCK" || { echo "$LOCK does not pin numpy==$EXPECT_NUMPY" >&2; exit 2; }

if [[ -z $BASE ]]; then
  if [[ -x /usr/bin/python3.14 ]]; then
    BASE=/usr/bin/python3.14
  else
    echo "no /usr/bin/python3.14: using a uv-managed CPython $MANAGED_REQUEST"
    uv python install --no-config "$MANAGED_REQUEST"
    BASE=$(uv python find --no-config --managed-python "$MANAGED_REQUEST")
  fi
elif [[ $BASE != */* ]]; then
  uv python install --no-config "$BASE"
  BASE=$(uv python find --no-config "$BASE")
fi
[[ -x $BASE ]] || { echo "base interpreter $BASE is not an executable file" >&2; exit 2; }

uv venv --no-project --no-config --python "$BASE" "$TARGET"
uv pip install --no-config --python "$TARGET/bin/python" --only-binary :all: --no-deps -r "$LOCK"

# Checks: the system site is excluded, numpy is the pinned version and is the only package.
grep -qx "include-system-site-packages = false" "$TARGET/pyvenv.cfg" || { echo "$TARGET/pyvenv.cfg includes the system site" >&2; exit 1; }
"$TARGET/bin/python" -I -c "import numpy, sys; v = numpy.__version__; assert v == '$EXPECT_NUMPY', v; print('python', sys.version.split()[0], 'numpy', v)"
PACKAGES=$(uv pip list --no-config --python "$TARGET/bin/python" --format freeze)
[[ $PACKAGES == "numpy==$EXPECT_NUMPY" ]] || { echo "unexpected packages in $TARGET: $PACKAGES" >&2; exit 1; }
echo "sandbox venv ready at $TARGET (base $(realpath "$TARGET/bin/python"))"
echo "next: uv run weightcode doctor --sandbox-python $TARGET/bin/python"
