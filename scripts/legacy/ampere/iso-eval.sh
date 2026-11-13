#!/bin/bash
# Floor-eval the two iso arms at BOTH precisions (f16 + Q4_K_M)
set -e
LLAMA=~/llama.cpp
source ~/forge_env/bin/activate
for mode in classic all; do
  d=/home/daniel/m0/runs/drop2-repair-iso-$mode
  echo "=== $mode: convert + quantize ==="
  python3 $LLAMA/convert_hf_to_gguf.py $d/hf-recovered --outtype f16 --outfile $d/iso-$mode-f16.gguf 2>&1 | tail -2
  $LLAMA/build/bin/llama-quantize $d/iso-$mode-f16.gguf $d/iso-$mode-Q4_K_M.gguf Q4_K_M 2>&1 | tail -2
  ls -l $d/*.gguf | awk '{print $9, $5}'
  cd /home/daniel/model-forge
  echo "=== $mode: Q4 floor ==="
  python3 surgery/multicorpus.py drop2-iso-$mode-q4 $d/iso-$mode-Q4_K_M.gguf
  echo "=== $mode: f16 floor ==="
  python3 surgery/multicorpus.py drop2-iso-$mode-f16 $d/iso-$mode-f16.gguf
done
echo "ISO EVAL DONE"
