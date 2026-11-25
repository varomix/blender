#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 varomix
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Build, install and package this Blender fork as a portable archive, with the Toast add-on
# bundled and enabled and the C++ node plugins included. Users extract the archive anywhere and
# run `./blender` inside it. It keeps its preferences in its own `<version>/portable` folder, so
# it never touches an installed Blender or its settings.
#
# Usage: TOAST_ADDON_DIR=<toast repo> build_files/toast/package_portable.sh [BUILD_DIR] [OUT_DIR]
#   FORMAT=zip  ...   # .zip instead of .tar.xz
# TOAST_ADDON_DIR is the Toast frontend repository (it contains `toast/` and `docs/`).

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$(realpath "${1:-$SRC_DIR/../build_linux_5.3}")"
OUT_DIR="$(realpath -m "${2:-$SRC_DIR/../packages}")"
FORMAT="${FORMAT:-tar.xz}"
TOAST_ADDON_DIR="$(realpath "${TOAST_ADDON_DIR:?set TOAST_ADDON_DIR to the Toast add-on repository}")"
if [ ! -f "$TOAST_ADDON_DIR/toast/__init__.py" ]; then
  echo "TOAST_ADDON_DIR=$TOAST_ADDON_DIR does not contain toast/__init__.py" >&2
  exit 1
fi

echo "Building and installing in $BUILD_DIR"
cmake --build "$BUILD_DIR" --target install -j"$(nproc)"

BLENDER_VERSION="$("$BUILD_DIR/bin/blender" --factory-startup --version 2>/dev/null | awk 'NR==1 {print $2}')"
GIT_HASH="$(git -C "$SRC_DIR" rev-parse --short=10 HEAD)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m | sed 's/x86_64/x64/')"
NAME="blender-${BLENDER_VERSION}-toast-${GIT_HASH}-${OS}-${ARCH}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# Everything Blender needs at run time, without the build tools CMake also writes to bin/.
rsync -a --exclude '/datatoc' --exclude '/makesdna' --exclude '/makesrna' --exclude '/msgfmt' \
  --exclude '/shader_tool' "$BUILD_DIR/bin/" "$STAGE/$NAME/"
VERSION_DIR="$STAGE/$NAME/$(echo "$BLENDER_VERSION" | cut -d. -f1-2)"

# The Toast add-on as a core add-on (enabled by default in factory preferences).
mkdir -p "$VERSION_DIR/scripts/addons_core"
rsync -a --exclude '__pycache__' "$TOAST_ADDON_DIR/toast/" "$VERSION_DIR/scripts/addons_core/toast/"
cp "$TOAST_ADDON_DIR/docs/guide.md" "$STAGE/$NAME/TOAST_GUIDE.md"
# Keep preferences and user scripts inside the package.
mkdir -p "$VERSION_DIR/portable"

ADDON_HASH="$(git -C "$TOAST_ADDON_DIR" rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
{
  git -C "$SRC_DIR" log -1 --format='Blender fork: %H %cd%n  %s'
  echo "Toast add-on: $ADDON_HASH"
} > "$STAGE/$NAME/BUILD_INFO.txt"
cat > "$STAGE/$NAME/README_TOAST.txt" <<'README'
Blender with Toast

Run ./blender in this folder. Settings are kept in the portable folder next to it, so this
build does not change an installed Blender.

Toast needs a C++ compiler to turn code into kernels: install g++ or clang++ (on Ubuntu:
sudo apt install g++), or set TOAST_CXX to one. Compiled kernels are cached in ~/.cache/toast.

Allow Preferences > Save & Load > Auto Run Python Scripts (or trust the file) for Toast nodes
to run. Start with Add > Toast > Toast in a Geometry Nodes tree; TOAST_GUIDE.md explains the
language and has a cookbook.
README

mkdir -p "$OUT_DIR"
case "$FORMAT" in
  zip)
    ARCHIVE="$OUT_DIR/$NAME.zip"
    rm -f "$ARCHIVE"
    (cd "$STAGE" && zip -qry "$ARCHIVE" "$NAME")
    ;;
  *)
    ARCHIVE="$OUT_DIR/$NAME.tar.xz"
    tar -C "$STAGE" -cf - "$NAME" | xz -T0 -6 > "$ARCHIVE"
    ;;
esac
echo "Created $ARCHIVE"
