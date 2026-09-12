#!/bin/bash
# Apply the PackFlow Crystal_Math edits on top of the pinned submodule.
#
# The overlay in crystal_math_overlay/ contains ONLY the files we modified
# relative to upstream nigalanakis/Crystal_Math @ 8808429 (the commit the
# external/Crystal_Math submodule is pinned to). This script copies those files
# into the submodule working tree so the data-extraction pipeline runs with our
# changes, without vendoring the whole upstream repository.
set -e

REPO_ROOT="${PACKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="$REPO_ROOT/preprocessing/crystal_math_overlay"
TARGET="$REPO_ROOT/external/Crystal_Math"

if [ ! -d "$TARGET/source_code" ]; then
    echo "ERROR: $TARGET not initialized. Run: git submodule update --init external/Crystal_Math" >&2
    exit 1
fi

echo "Applying Crystal_Math overlay -> $TARGET"
cp -v "$OVERLAY/source_code/"*.py   "$TARGET/source_code/"
cp -v "$OVERLAY/input_files/"*.txt  "$TARGET/input_files/"
cp -v "$OVERLAY/source_data/"*.json "$TARGET/source_data/"
echo "Overlay applied. The submodule now carries the PackFlow edits (uncommitted)."
