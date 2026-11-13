#!/bin/bash
# KLD floor for one freshly trained arm on the desktop (design §0j.11/§0j.12).
#   bash scripts/qwen35-4b/kld-arm.sh <label> <arm hf-recovered dir>
# Rebuilds the base reference if it is missing, converts the arm to f16, scores it, then
# deletes the arm GGUF. Peak ~33 GB (base f16 + 3 refs + arm f16); the vhdx never shrinks,
# so --cleanup also removes the base GGUF and refs afterwards (docs/storage.md).
set -euo pipefail
LABEL=$1 HF=$2 CLEAN=${3:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${FORGE_ROOT:-~/forge}
K=$ROOT/runs/qwen35-4b/kld
G=$K/gguf
PY=${FORGE_PY:-~/forge-gpu-env/bin/python}
export FORGE_LLAMA=${FORGE_LLAMA:-~/src/llama.cpp} FORGE_NGL=${FORGE_NGL:-99} FORGE_KLD_REF=$K
mkdir -p "$G"
free=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
[ "$free" -ge 200 ] || { echo "C: has ${free}G free (<200G): stopping"; exit 1; }

if ! { [ -f $K/ref-wikitext.kld ] && [ -f $K/ref-fineweb.kld ] && [ -f $K/ref-code.kld ]; }; then
  [ -f $G/base-f16.gguf ] || $PY $FORGE_LLAMA/convert_hf_to_gguf.py $ROOT/base/Qwen/Qwen3.5-4B \
      --outtype f16 --outfile $G/base-f16.gguf 2>&1 | tail -1
  (cd "$REPO" && $PY -m surgery.kld ref $G/base-f16.gguf)
fi
$PY $FORGE_LLAMA/convert_hf_to_gguf.py "$HF" --outtype f16 --outfile $G/$LABEL-f16.gguf 2>&1 | tail -1
(cd "$REPO" && $PY -m surgery.kld score "$LABEL" $G/$LABEL-f16.gguf)
rm -f $G/$LABEL-f16.gguf
if [ "$CLEAN" = "--cleanup" ]; then rm -f $G/base-f16.gguf $K/ref-*.kld; fi
echo "=== KLD $LABEL DONE $(date -Is)"
