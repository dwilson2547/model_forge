#!/bin/bash
# The no-surgery context arms for design §0j.10, queued behind 27b-drop-context.sh:
#   nohup bash ~/model-forge/scripts/27b-context-arms.sh > ~/m0/runs/drop27/arms.log 2>&1 &
# A. IQ3_XS from the BF16 source. The §0j.4 attempt failed on the imatrix gap: the
#    imatrix pass never runs the MTP block, so blk.64 has no importance data at all,
#    and IQ3_XS's iq3_xxs tensors refuse to quantize without it. The whole block is
#    pinned to iq3_s, which needs no imatrix and is what IQ3_M gives most of it anyway.
# B. Unsliced IQ3_M with a q8_0 KV cache (flash attention on): the floor cost of halving KV.
# C. The same with f16 KV and flash attention on, the control for B (the §0j.4 row ran with
#    the default -fa auto).
# Same resume rule as the driver: finished labels are skipped.
set -euo pipefail
trap 'echo "=== FAILED at line $LINENO: $BASH_COMMAND $(date -Is) ==="' ERR

while pgrep -f 27b-drop-context.sh > /dev/null; do sleep 120; done

OUT=~/m0/runs/drop27
LLAMA=~/llama.cpp/build/bin
SRC=~/m0/baseline-27b/src-bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf
IMAT=~/m0/baseline-27b/imatrix-27b-bf16.dat
RESULTS=~/m0/runs/multicorpus-results.jsonl
source ~/forge_env/bin/activate
cd ~/model-forge
log() { echo "=== $* $(date -Is) ==="; }
done_row() { grep -q "\"label\": \"$1\"" "$RESULTS" 2> /dev/null; }

q=$OUT/srcbf16-IQ3_XS-mtp3s.gguf
if ! done_row 27b-IQ3_XS-mtp3s; then
  if [ ! -s "$q" ]; then
    log "[A] IQ3_XS quant"
    $LLAMA/llama-quantize --imatrix "$IMAT" --tensor-type "blk\.64\..*=iq3_s" \
      "$SRC" "$q.tmp" IQ3_XS > "$OUT/quant-IQ3_XS-mtp3s.log" 2>&1
    mv "$q.tmp" "$q"
  fi
  ls -l "$q" | awk '{print $9, $5}'
  log "[A] IQ3_XS floor"
  python surgery/multicorpus.py 27b-IQ3_XS-mtp3s "$q"
fi

if ! done_row 27b-IQ3_M-srcbf16-kvq8; then
  log "[B] IQ3_M + q8_0 KV floor"
  FORGE_PPL_EXTRA="-fa on -ctk q8_0 -ctv q8_0" \
    python surgery/multicorpus.py 27b-IQ3_M-srcbf16-kvq8 ~/m0/baseline-27b/srcbf16-IQ3_M.gguf
fi
if ! done_row 27b-IQ3_M-srcbf16-fa; then
  log "[C] IQ3_M + flash attention, f16 KV (control for B)"
  FORGE_PPL_EXTRA="-fa on" \
    python surgery/multicorpus.py 27b-IQ3_M-srcbf16-fa ~/m0/baseline-27b/srcbf16-IQ3_M.gguf
fi
log "ARMS DONE"
