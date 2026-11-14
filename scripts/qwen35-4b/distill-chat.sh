#!/bin/bash
# Chat-distill the [25,29] slice (design §0j.14): the unsliced base answers the distill/
# prompt mix (thinking kept), then the slice trains on KL(base || slice) over those answers
# in the chat template, instead of over raw mixed.train.txt text like 2529-distill.
#   bash scripts/qwen35-4b/distill-chat.sh            # STAGE=data|train|all (default all)
# Desktop 5080. Peak disk ~9 GB (one f16 GGUF, deleted after generation).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${FORGE_ROOT:-~/forge}
PY=${FORGE_PY:-~/forge-gpu-env/bin/python}
export FORGE_LLAMA=${FORGE_LLAMA:-~/src/llama.cpp} FORGE_NGL=${FORGE_NGL:-99}
BASE=$ROOT/base/Qwen/Qwen3.5-4B
SLICE=$ROOT/runs/qwen35-4b/slice/drop2-explicit-2529/hf-edited
OUT=$ROOT/runs/qwen35-4b/drop2-repair/2529-chatdistill
MIX=${MIX:-default}
TEACHER=qwen35-4b-base
STAGE=${STAGE:-all}
mkdir -p $OUT
cd "$REPO"

if [ "$STAGE" != train ]; then
  G=$OUT/teacher-f16.gguf
  [ -f $G ] || $PY $FORGE_LLAMA/convert_hf_to_gguf.py $BASE --outtype f16 --outfile $G 2>&1 | tail -1
  $PY -m distill.datagen $TEACHER $G --mix $MIX --slots 4
  rm -f $G
fi

if [ "$STAGE" != data ]; then
  $PY surgery/recover_gpu.py $SLICE --sft-dataset $ROOT/data/distill/$TEACHER.$MIX.jsonl \
      --distill $BASE --out $OUT --ctx 8192 --batch 1 --accum 4 --epochs 1 \
      --lora-r 16 --lr 1e-4 --seed 1234 --val-frac 0.02 --stop-on-val-rise 3
fi
echo "=== CHATDISTILL DONE $(date -Is)"
