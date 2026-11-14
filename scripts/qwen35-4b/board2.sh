#!/bin/bash
# Board v2 (bench/board2.py, 664 items) on the desktop GPU for the [25,29] line: base twice
# (the second run is the board's own noise floor), the unrepaired cut, and the two repairs
# the v1 board could not separate (distill vs fresh1e, both 0.867 = base).
# Streams one f16 GGUF at a time (convert -> board -> delete); peak ~16 GB (docs/storage.md).
# Raw responses are cached per label, so a rerun after a crash resumes.
#   bash scripts/qwen35-4b/board2.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${FORGE_ROOT:-~/forge}
G=$ROOT/runs/qwen35-4b/board2
PY=${FORGE_PY:-~/forge-gpu-env/bin/python}
export FORGE_LLAMA=${FORGE_LLAMA:-~/src/llama.cpp} FORGE_NGL=${FORGE_NGL:-99}
SHARE=/mnt/models/forge/qwen35-4b
mkdir -p $G
free=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
[ "$free" -ge 200 ] || { echo "C: has ${free}G free (<200G): stopping"; exit 1; }

conv() { [ -f $G/$1-f16.gguf ] || $PY $FORGE_LLAMA/convert_hf_to_gguf.py "$2" --outtype f16 \
           --outfile $G/$1-f16.gguf 2>&1 | tail -1; }
board() { (cd "$REPO" && $PY -m bench.board2 run "$1@5080" "$2"); }

conv base $ROOT/base/Qwen/Qwen3.5-4B
board base $G/base-f16.gguf
board base-r2 $G/base-f16.gguf
rm -f $G/base-f16.gguf
conv unrep-2529 $ROOT/runs/qwen35-4b/slice/drop2-explicit-2529/hf-edited
board unrep-2529 $G/unrep-2529-f16.gguf; rm -f $G/unrep-2529-f16.gguf
conv 2529-distill $ROOT/runs/qwen35-4b/drop2-repair/2529-distill/hf-recovered
board 2529-distill $G/2529-distill-f16.gguf; rm -f $G/2529-distill-f16.gguf
rsync -rt ampere:$SHARE/drop2-repair/2529-fresh1e/hf/ $G/fresh1e-hf/
conv 2529-fresh1e $G/fresh1e-hf; rm -rf $G/fresh1e-hf
board 2529-fresh1e $G/2529-fresh1e-f16.gguf; rm -f $G/2529-fresh1e-f16.gguf
echo "=== BOARD2 DONE $(date -Is)"
