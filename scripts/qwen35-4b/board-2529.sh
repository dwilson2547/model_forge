#!/bin/bash
# Chat-mode task board for the [25,29] repairs on the desktop GPU (design §0j.12): does the
# distill arm's closeness to the base (KLD) buy behaviour? Base and the unrepaired wound run
# on the same surface, since GPU and CPU boards are not row-comparable.
# Streams one f16 GGUF at a time (convert -> board -> delete); peak ~16 GB (docs/storage.md).
#   bash scripts/qwen35-4b/board-2529.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${FORGE_ROOT:-~/forge}
G=$ROOT/runs/qwen35-4b/board-2529
PY=${FORGE_PY:-~/forge-gpu-env/bin/python}
export FORGE_LLAMA=${FORGE_LLAMA:-~/src/llama.cpp} FORGE_NGL=${FORGE_NGL:-99}
SHARE=/mnt/models/forge/qwen35-4b
mkdir -p $G
free=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
[ "$free" -ge 200 ] || { echo "C: has ${free}G free (<200G): stopping"; exit 1; }

board() {  # label hf_dir
  $PY $FORGE_LLAMA/convert_hf_to_gguf.py "$2" --outtype f16 --outfile $G/$1-f16.gguf 2>&1 | tail -1
  (cd "$REPO" && $PY -m bench.runner --chat "$1@5080=$G/$1-f16.gguf")
  rm -f $G/$1-f16.gguf
}
board base $ROOT/base/Qwen/Qwen3.5-4B
board unrep-2529 $ROOT/runs/qwen35-4b/slice/drop2-explicit-2529/hf-edited
board 2529-distill $ROOT/runs/qwen35-4b/drop2-repair/2529-distill/hf-recovered
rsync -rt ampere:$SHARE/drop2-repair/2529-fresh1e/hf/ $G/fresh1e-hf/
board 2529-fresh1e $G/fresh1e-hf
rm -rf $G/fresh1e-hf
echo "=== BOARD DONE $(date -Is)"
