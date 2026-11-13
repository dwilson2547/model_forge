#!/bin/bash
# Steps 3-5 remainder: Q3_K_M quant (no imatrix) + floors for IQ3_M & Q3_K_M.
# IQ3_XS failed on tail-tensor imatrix coverage (design 0j.4 gotcha 1) - dropped here.
set -euo pipefail
cd ~/m0/baseline-27b
LLAMA=~/llama.cpp/build/bin
B1=src-bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf

rm -f srcbf16-IQ3_XS.gguf
echo "=== quant Q3_K_M $(date) ==="
$LLAMA/llama-quantize "$B1" srcbf16-Q3_K_M.gguf Q3_K_M 2>&1 | tee quant-Q3_K_M.log | tail -4

echo "=== sizes ==="
ls -l srcbf16-*.gguf | awk "{print \$9, \$5}"

echo "=== floors ==="
source ~/forge_env/bin/activate
cd ~/model-forge
python3 surgery/multicorpus.py 27b-IQ3_M-srcbf16 /home/daniel/m0/baseline-27b/srcbf16-IQ3_M.gguf
python3 surgery/multicorpus.py 27b-Q3_K_M-srcbf16 /home/daniel/m0/baseline-27b/srcbf16-Q3_K_M.gguf
echo "=== RESUME2 DONE $(date) ==="
