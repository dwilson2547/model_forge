#!/bin/bash
# Convert + quantize + floor-eval the r2fix plateau arm (post-training pipeline)
set -e
RUN=~/m0/runs/drop2-repair-r2fix-lr1e4
LLAMA=~/llama.cpp
mkdir -p $RUN
source ~/forge_env/bin/activate
echo "=== convert f16 ==="
python3 $LLAMA/convert_hf_to_gguf.py $RUN/hf-recovered --outtype f16 --outfile $RUN/r2fix-lr1e4-f16.gguf 2>&1 | tail -3
echo "=== quantize Q4_K_M ==="
$LLAMA/build/bin/llama-quantize $RUN/r2fix-lr1e4-f16.gguf $RUN/r2fix-lr1e4-Q4_K_M.gguf Q4_K_M 2>&1 | tail -3
ls -l $RUN/*.gguf | awk '{print $9, $5}'
echo "=== 3-corpus floor ==="
cd ~/model-forge
python3 surgery/multicorpus.py drop2-r2fix-lr1e4 $RUN/r2fix-lr1e4-Q4_K_M.gguf
echo "=== R2FIX FLOOR DONE ==="
