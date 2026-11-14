#!/bin/bash
# Repair at the cut, for the [25,29] slice (design §0j.15). Desktop 5080.
#   STAGE=prep bash scripts/qwen35-4b/repair-local.sh    # ~30 min, no training
#   STAGE=hid  bash scripts/qwen35-4b/repair-local.sh    # one training arm, ~2.5 h
# prep: the closed-form bridge (surgery/bridge.py), its raw-text KLD floor, the held-out chat
#       KLD reference + rows for the cut, 2529-distill and the bridge, and a short --calibrate
#       of the hidden-state trainer (throughput + peak VRAM before committing hours).
# hid:  2529-distill's exact recipe plus --hidden-loss 1, from the plain slice (ARM=hid) or
#       from the bridged one (ARM=bridge-hid). Only the objective / the init changes.
# DATASET is the teacher-answer file the chat KLD reads its holdout from.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${FORGE_ROOT:-~/forge}
PY=${FORGE_PY:-~/forge-gpu-env/bin/python}
export FORGE_LLAMA=${FORGE_LLAMA:-~/src/llama.cpp} FORGE_NGL=${FORGE_NGL:-99}
BASE=$ROOT/base/Qwen/Qwen3.5-4B
SLICE=$ROOT/runs/qwen35-4b/slice/drop2-explicit-2529/hf-edited
R=$ROOT/runs/qwen35-4b/drop2-repair
BR=$R/2529-bridge/hf-bridged
DATASET=${DATASET:-$ROOT/data/distill/qwen35-4b-q4.wide.partial.jsonl}
STAGE=${STAGE:-prep}
ARM=${ARM:-hid}
cd "$REPO"

if [ "$STAGE" = prep ]; then
  [ -f $BR/bridge-results.json ] || $PY -m surgery.bridge $BASE $SLICE $BR --dropped 25 29
  bash scripts/qwen35-4b/kld-arm.sh 2529-bridge $BR
  $PY -m distill.chatkld ref $DATASET
  $PY -m distill.chatkld score unrep-2529 $SLICE $DATASET
  $PY -m distill.chatkld score 2529-bridge $BR $DATASET
  $PY -m distill.chatkld score 2529-distill $R/2529-distill/hf-recovered $DATASET
  $PY surgery/recover_gpu.py $SLICE $ROOT/data/mixed.train.txt --distill $BASE --hidden-loss 1 \
      --dropped 25 29 --calibrate --max-chars 600000 --epochs 1 --batch 2 --accum 2 --seed 1234 \
      --out $R/calib-hid
fi

if [ "$STAGE" = hid ]; then
  SRC=$SLICE; [ "$ARM" = bridge-hid ] && SRC=$BR
  OUT=$R/2529-$ARM
  $PY surgery/recover_gpu.py $SRC $ROOT/data/mixed.train.txt --distill $BASE --hidden-loss 1 \
      --dropped 25 29 --max-chars 41000000 --epochs 1 --batch 2 --accum 2 --lora-r 16 --lr 1e-4 \
      --seed 1234 --val-frac 0.01 --out $OUT 2>&1 | tee $OUT.train.log
  bash scripts/qwen35-4b/kld-arm.sh 2529-$ARM $OUT/hf-recovered
  $PY -m distill.chatkld score 2529-$ARM $OUT/hf-recovered $DATASET
fi
echo "=== REPAIR-LOCAL $STAGE DONE $(date -Is)"
