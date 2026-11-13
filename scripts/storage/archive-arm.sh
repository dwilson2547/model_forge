#!/bin/bash
# Archive one trained arm to the cold share (docs/storage.md), with its card as README.md.
#   scripts/storage/archive-arm.sh <family> <experiment> <arm> [--delete-local]
# Source: $FORGE_ROOT/runs/<family>/<experiment>/<arm>/ on this machine. Card: cards/arms/<family>/<arm>.md.
# Copies the weights (hf-recovered/ -> hf/, ckpt/final/repair -> adapter/, *.gguf) and the run's
# json, writes SHA256SUMS, then re-hashes on the share side. --delete-local removes the local
# weights only when every hash matches. Logs and json stay local either way.
set -euo pipefail
FAM=$1 EXP=$2 ARM=$3 DEL=${4:-}
ROOT=${FORGE_ROOT:-~/forge}
HOST=${FORGE_ARCHIVE_HOST:-ampere}
SHARE=${FORGE_ARCHIVE:-/mnt/models/forge}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$ROOT/runs/$FAM/$EXP/$ARM
DST=$SHARE/$FAM/$EXP/$ARM
CARD=$REPO/cards/arms/$FAM/$ARM.md
[ -d "$SRC" ] || { echo "no $SRC"; exit 1; }
[ -f "$CARD" ] || { echo "no card $CARD: write it first"; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
# staging = symlinks in the archive layout; rsync -L copies the real files
[ -d "$SRC/hf-recovered" ] && ln -s "$SRC/hf-recovered" "$STAGE/hf"
[ -d "$SRC/ckpt/final/repair" ] && ln -s "$SRC/ckpt/final/repair" "$STAGE/adapter"
for f in "$SRC"/*.gguf "$SRC"/*.json; do [ -e "$f" ] && ln -s "$f" "$STAGE/"; done
[ -e "$STAGE/hf" ] || ls "$STAGE"/*.gguf > /dev/null 2>&1 || { echo "$ARM: no weights to archive"; exit 1; }

echo "=== $ARM: hashing $(date -Is)"
(cd "$STAGE" && find -L . -type f ! -name SHA256SUMS | sort | xargs sha256sum) > "$STAGE/SHA256SUMS"
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
{ cat "$CARD"; echo; echo "## Archive"; echo
  echo "Archived $(date +%F) from \`$SRC\` at model-forge commit \`$COMMIT\`. Check with \`sha256sum -c SHA256SUMS\`."
  echo; echo '```'; cat "$STAGE/SHA256SUMS"; echo '```'; } > "$STAGE/README.md"

echo "=== $ARM: copying to $HOST:$DST $(date -Is)"
ssh "$HOST" "mkdir -p '$DST'"
rsync -rtL --no-perms --partial "$STAGE/" "$HOST:$DST/"

echo "=== $ARM: verifying on the share $(date -Is)"
if ssh "$HOST" "cd '$DST' && sha256sum --quiet -c SHA256SUMS"; then
  echo "=== $ARM: VERIFIED"
else
  echo "=== $ARM: VERIFY FAILED, local copy kept"; exit 1
fi
if [ "$DEL" = "--delete-local" ]; then
  rm -rf "$SRC/hf-recovered" "$SRC/ckpt" "$SRC"/*.gguf
  echo "=== $ARM: local weights removed; card + json + logs kept in $SRC"
fi
