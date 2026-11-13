#!/bin/bash
# Pass-1 [25,29] arm: f16 + Q4_K_M floors. Waits for the 27B bf16 chain to release the cores.
set -euo pipefail
LLAMA=~/llama.cpp
d=/home/daniel/m0/runs/drop2-repair-2529
echo "=== waiting for 27B bf16 chain (27b-bf16-driver) to exit ==="
while pgrep -f "27b-bf16-driver" > /dev/null; do sleep 300; done
echo "=== cores free, starting 2529 eval $(date) ==="
source ~/forge_env/bin/activate
echo "=== convert f16 ==="
python3 $LLAMA/convert_hf_to_gguf.py $d/hf-recovered --outtype f16 --outfile $d/2529-f16.gguf 2>&1 | tail -2
echo "=== quantize Q4_K_M ==="
$LLAMA/build/bin/llama-quantize $d/2529-f16.gguf $d/2529-Q4_K_M.gguf Q4_K_M 2>&1 | tail -2
ls -l $d/*.gguf | awk "{print \$9, \$5}"
cd /home/daniel/model-forge
echo "=== f16 floor ==="
python3 surgery/multicorpus.py drop2-2529-repair-f16 $d/2529-f16.gguf
echo "=== Q4_K_M floor ==="
python3 surgery/multicorpus.py drop2-2529-repair-q4 $d/2529-Q4_K_M.gguf
echo "2529 EVAL DONE"
