#!/bin/bash
# Board v2 for the measured-damage picks (design §0j.15): the scan's drop-2 [12,16] and drop-4
# [12,13,15,16], unrepaired, against the existing unrep-2529@5080 row. Desktop 5080.
# Builds each HF slice from its recipe (CPU), then convert -> board -> delete, one f16 GGUF at
# a time. Raw responses are cached per label, so a rerun resumes.
#   bash scripts/qwen35-4b/board-scan.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export FORGE_ROOT=${FORGE_ROOT:-~/forge}
ROOT=$FORGE_ROOT
G=$ROOT/runs/qwen35-4b/board2
PY=${FORGE_PY:-~/forge-gpu-env/bin/python}
export FORGE_LLAMA=${FORGE_LLAMA:-~/src/llama.cpp} FORGE_NGL=${FORGE_NGL:-99}
mkdir -p $G
cd "$REPO"
for s in drop2-scan-1216 drop4-scan-12131516; do
  HF=$ROOT/runs/qwen35-4b/slice/$s/hf-edited
  [ -f $HF/config.json ] || $PY -m surgery.pipeline recipes/qwen35-4b-$s.yaml --no-build
  L=unrep-${s##*-}
  $PY $FORGE_LLAMA/convert_hf_to_gguf.py $HF --outtype f16 --outfile $G/$L-f16.gguf 2>&1 | tail -1
  $PY -m bench.board2 run "$L@5080" $G/$L-f16.gguf
  rm -f $G/$L-f16.gguf
done
$PY -m bench.board2 compare unrep-2529@5080 unrep-1216@5080
$PY -m bench.board2 compare base@5080 unrep-1216@5080
$PY -m bench.board2 compare unrep-2529@5080 unrep-12131516@5080
echo "=== BOARD-SCAN DONE $(date -Is)"
