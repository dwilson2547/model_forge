#!/bin/bash
# Resume the 27B bf16 chain from step 3: quants -> sizes -> floors.
# Fix: this llama.cpp build has no llama-quantize --merge; split input is read
# transparently and output is a single (merged) GGUF. Full output logged, no tail -4.
set -euo pipefail
cd ~/m0/baseline-27b
LLAMA=~/llama.cpp/build/bin
B1=src-bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf

echo "=== waiting for 2529-eval to release cores ==="
while pgrep -f "2529-eval.sh" > /dev/null; do sleep 300; done
echo "=== starting resume chain $(date) ==="

[ -f imatrix-27b-bf16.dat ] || { echo "FATAL: imatrix missing"; exit 1; }

echo "=== [3/5] quantize from BF16 (split input, no --merge) ==="
for q in IQ3_M IQ3_XS Q3_K_M; do
  echo "--- quant $q $(date) ---"
  $LLAMA/llama-quantize --imatrix imatrix-27b-bf16.dat "$B1" "srcbf16-$q.gguf" "$q" 2>&1 | tee "quant-$q.log" | tail -4
done

echo "=== [4/5] sizes ==="
ls -l srcbf16-*.gguf | awk "{print \$9, \$5}"

echo "=== [5/5] 3-corpus floors ==="
source ~/forge_env/bin/activate
cd ~/model-forge
for q in IQ3_M IQ3_XS Q3_K_M; do
  python3 surgery/multicorpus.py "27b-$q-srcbf16" "/home/daniel/m0/baseline-27b/srcbf16-$q.gguf"
done
echo "=== BASELINE-27B BF16-CHAIN DONE $(date) ==="
