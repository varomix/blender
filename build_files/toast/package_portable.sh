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
# Usage: build_files/toast/package_portable.sh [BUILD_DIR] [OUT_DIR]
#   FORMAT=zip  ...   # .zip instead of .tar.xz
# TOAST_ADDON_DIR is the Toast add-on repository (default: the `toast/addon` submodule).

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$(realpath "${1:-$SRC_DIR/../build_linux_5.3}")"
OUT_DIR="$(realpath -m "${2:-$SRC_DIR/../packages}")"
FORMAT="${FORMAT:-tar.xz}"
TOAST_ADDON_DIR="$(realpath "${TOAST_ADDON_DIR:-$SRC_DIR/toast/addon}")"
if [ ! -f "$TOAST_ADDON_DIR/toast/__init__.py" ]; then
  echo "TOAST_ADDON_DIR=$TOAST_ADDON_DIR does not contain toast/__init__.py (git submodule update --init toast/addon)" >&2
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
rm -rf "$VERSION_DIR/scripts/addons_core/toast"  # A development install may be a symbolic link.
rsync -a --exclude '__pycache__' "$TOAST_ADDON_DIR/toast/" "$VERSION_DIR/scripts/addons_core/toast/"
cp "$TOAST_ADDON_DIR/docs/guide.md" "$STAGE/$NAME/TOAST_GUIDE.md"
if [ -f "$TOAST_ADDON_DIR/docs/guide.html" ]; then
  cp "$TOAST_ADDON_DIR/docs/guide.html" "$STAGE/$NAME/TOAST_GUIDE.html"
fi
# The built-in kernel compiler carries clang and LLVM: drop its symbol tables.
if [ -f "$STAGE/$NAME/lib/libtoast_jit.so" ]; then
  strip --strip-unneeded "$STAGE/$NAME/lib/libtoast_jit.so"
fi
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

Toast turns code into native kernels with its built-in compiler: nothing else to install.
Compiled kernels are cached in ~/.cache/toast.

Allow Preferences > Save & Load > Auto Run Python Scripts (or trust the file) for Toast nodes
to run. Start with Add > Toast > Toast in a Geometry Nodes tree; TOAST_GUIDE.html (open it in a
browser, or read TOAST_GUIDE.md) explains the language and has a cookbook.
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
