#!/bin/sh
# Copy the runtime modules into another project, pinned to this commit:
#   scripts/vendor.sh ../stemplayer/modal/stemsep
# The repo is private, so its users vendor it rather than pip install it.
set -eu
dest=${1:?usage: scripts/vendor.sh <dest dir>}
root=$(cd "$(dirname "$0")/.." && pwd)
sha=$(git -C "$root" rev-parse --short HEAD)
dirty=$(git -C "$root" status --porcelain -- src/stemsep | head -1)
[ -n "$dirty" ] && sha="$sha-dirty"
mkdir -p "$dest"
rm -f "$dest"/*.py
for f in __init__ engine model audio profile onnx_backend; do
  [ -f "$root/src/stemsep/$f.py" ] && cp "$root/src/stemsep/$f.py" "$dest/"
done
printf 'Vendored from github.com/dawsonxiong/stemsep at %s.\nDo not edit here: change stemsep, then rerun its scripts/vendor.sh.\n' "$sha" > "$dest/VENDORED"
echo "vendored stemsep@$sha into $dest"
