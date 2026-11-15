#!/bin/bash
# Mirror ~/forge to the share and back (docs/storage.md). One tree, the same relative path on
# both machines and on the share, so a pull is the push with the ends swapped.
#   share-sync.sh push  [subpath]   copy ~/forge[/subpath] to the share (never deletes there)
#   share-sync.sh pull  <subpath>   copy it back from the share
#   share-sync.sh check [subpath]   list local files the share lacks or holds differently
#   share-sync.sh clean <subpath>   check, and only if nothing differs, delete the local copy
# subpath is relative to ~/forge, e.g. runs/qwen35-4b/drop2-repair/2529-distill.
# check compares size and mtime; CHECKSUM=1 compares content (slow: the share is spinning disk).
# Where the share isn't writable from this machine it goes through $FORGE_SHARE_HOST (ampere).
set -euo pipefail
ROOT=${FORGE_ROOT:-~/forge}
SHARE=${FORGE_SHARE:-/mnt/models/forge}
HOST=${FORGE_SHARE_HOST:-ampere}
CMD=${1:-}; SUB=${2:-}; SUB=${SUB%/}
case "$SUB" in /*|*..*) echo "subpath must be relative to $ROOT"; exit 1;; esac
if [ -w "$SHARE" ]; then FAR=$SHARE; mk() { mkdir -p "$1"; }
else FAR=$HOST:$SHARE; mk() { ssh "$HOST" "mkdir -p '$1'"; }; fi
SRC=$ROOT${SUB:+/$SUB}
[ "$CMD" = pull ] || [ -e "$SRC" ] || { echo "no $SRC"; exit 1; }
# -L: the share has no symlinks. modify-window: CIFS timestamps are coarser than ext4's.
# Excluded: the damage scan's scratch (rebuilt on resume) and half-written files.
OPTS=(-rtL --no-perms --no-owner --no-group --partial --modify-window=2
      --exclude '*.part' --exclude '*.kld' --exclude 'runs/damage-scan/*/slice.gguf')
# No trailing slash on the source: rsync copies the named file or directory into the parent.
if [ -n "$SUB" ]; then P=$(dirname "$SUB"); P=${P#.}; P=${P:+/$P}
  L=$SRC LP=$ROOT$P/ F=$FAR/$SUB FP=$FAR$P/ D=$SHARE$P
else L=$ROOT/ LP=$ROOT/ F=$FAR/ FP=$FAR/ D=$SHARE; fi

differs() { rsync "${OPTS[@]}" ${CHECKSUM:+--checksum} -n --out-format='%n' "$L" "$FP" | grep -v '/$' || true; }

case "$CMD" in
  push)  mk "$D"; rsync "${OPTS[@]}" --info=progress2,stats1 "$L" "$FP";;
  pull)  [ -n "$SUB" ] || { echo "pull needs a subpath"; exit 1; }
         mkdir -p "$LP"; rsync "${OPTS[@]}" --info=progress2,stats1 "$F" "$LP";;
  check) d=$(differs); [ -z "$d" ] && echo "share matches $SRC" || { echo "$d"; exit 1; };;
  clean) [ -n "$SUB" ] || { echo "clean needs a subpath"; exit 1; }
         d=$(differs)
         [ -z "$d" ] || { echo "not on the share yet, nothing deleted:"; echo "$d" | head -20; exit 1; }
         du -sh "$SRC"; rm -rf "$SRC"; echo "removed $SRC (share copy: $F)";;
  *) sed -n 2,10p "$0"; exit 1;;
esac
