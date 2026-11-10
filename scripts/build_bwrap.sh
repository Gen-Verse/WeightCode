#!/usr/bin/env bash
# Build bubblewrap without root, for hosts whose distribution ships one older than 0.8.0 (Ubuntu 22.04: 0.6.1).
#
#   bash scripts/build_bwrap.sh [--version 0.11.1|0.8.0] [--prefix DIR] [--work DIR]
#
# Downloads the bubblewrap release and libcap 2.73 (both checked by sha256), builds libcap as a static library and
# bubblewrap against it, and installs <prefix>/bin/bwrap (default prefix ~/.local/opt/bubblewrap-<version>). Needs a C
# compiler, make, curl, xz and uv (meson, ninja and pkgconf come from PyPI through uv). Then pass the binary with
# --bwrap, or export WEIGHTCODE_BWRAP=<prefix>/bin/bwrap. Like any bubblewrap without setuid, it needs the host to allow
# unprivileged user namespaces; the script's last check says so if it does not. Built and checked on Ubuntu 22.04
# (GCC 11) and 26.04 (GCC 15).
set -euo pipefail

VERSION=0.11.1
PREFIX=
WORK=
LIBCAP=2.73
LIBCAP_SHA256=6405f6089cf4cdd8c271540cd990654d78dd0b1989b2d9bda20f933a75a795a5
declare -A BWRAP_SHA256=(
  [0.8.0]=957ad1149db9033db88e988b12bcebe349a445e1efc8a9b59ad2939a113d333a
  [0.11.1]=c1b7455a1283b1295879a46d5f001dfd088c0bb0f238abb5e128b3583a246f71
)

while (( $# )); do
  case $1 in
    --version) VERSION=$2; shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n ${BWRAP_SHA256[$VERSION]:-} ]] || { echo "no recorded sha256 for bubblewrap $VERSION (have: ${!BWRAP_SHA256[*]})" >&2; exit 2; }
PREFIX=${PREFIX:-$HOME/.local/opt/bubblewrap-$VERSION}
[[ -e $PREFIX ]] && { echo "$PREFIX exists; this script does not modify an existing installation" >&2; exit 2; }
for tool in cc make curl xz uv; do
  command -v "$tool" > /dev/null || { echo "$tool is not on PATH" >&2; exit 2; }
done
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/weightcode-bwrap-XXXXXX")}
mkdir -p "$WORK"

fetch() {  # url file sha256
  curl -fsSL -o "$WORK/$2" "$1"
  echo "$3  $WORK/$2" | sha256sum -c --quiet - || { echo "$2: sha256 mismatch" >&2; exit 1; }
}
fetch "https://mirrors.edge.kernel.org/pub/linux/libs/security/linux-privs/libcap2/libcap-$LIBCAP.tar.xz" \
  "libcap-$LIBCAP.tar.xz" "$LIBCAP_SHA256"
fetch "https://github.com/containers/bubblewrap/releases/download/v$VERSION/bubblewrap-$VERSION.tar.xz" \
  "bubblewrap-$VERSION.tar.xz" "${BWRAP_SHA256[$VERSION]}"

tar -C "$WORK" -xf "$WORK/libcap-$LIBCAP.tar.xz"
make -s -C "$WORK/libcap-$LIBCAP/libcap" prefix="$WORK/libcap" lib=lib SHARED=no PAM_CAP=no GOLANG=no install-static-cap

tar -C "$WORK" -xf "$WORK/bubblewrap-$VERSION.tar.xz"
MESON=(uv tool run --quiet --with ninja==1.11.1.4 --with pkgconf==2.4.3.post1 --from meson==1.6.1 meson)
PKG_CONFIG=pkgconf PKG_CONFIG_PATH="$WORK/libcap/lib/pkgconfig" "${MESON[@]}" setup "$WORK/build" \
  "$WORK/bubblewrap-$VERSION" --prefix="$PREFIX" -Dman=disabled -Dselinux=disabled -Dbash_completion=disabled \
  -Dzsh_completion=disabled -Dtests=false -Dc_std=gnu17  # gnu17: GCC 15 defaults to C23, where 0.8.0 does not build
"${MESON[@]}" compile -C "$WORK/build"
"${MESON[@]}" install -C "$WORK/build" --quiet

"$PREFIX/bin/bwrap" --version
"$PREFIX/bin/bwrap" --unshare-all --unshare-user --disable-userns --ro-bind /usr /usr --symlink usr/bin /bin \
  --symlink usr/lib /lib --symlink usr/lib64 /lib64 --proc /proc --dev /dev /usr/bin/true \
  || { echo "$PREFIX/bin/bwrap was built but cannot create the sandbox here (user namespaces disabled?)" >&2; exit 1; }
echo "bubblewrap $VERSION ready: $PREFIX/bin/bwrap"
echo "next: export WEIGHTCODE_BWRAP=$PREFIX/bin/bwrap   (or pass --bwrap $PREFIX/bin/bwrap)"
