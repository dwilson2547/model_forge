#!/bin/bash
# 27B baseline, pure chain: BF16 -> bf16-imatrix -> IQ3_M/IQ3_XS/Q3_K_M -> floors.
# No quant-from-quant: source is unsloth's BF16 gguf; --allow-requantize NOT used.
# (The Q8_0-derived quants from driver2 stay for the quant-from-quant penalty comparison.)
set -e
set -o pipefail
cd ~/m0/baseline-27b
LLAMA=~/llama.cpp/build/bin
REG=~/m0
mkdir -p src-bf16
B1=src-bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf
B2=src-bf16/Qwen3.8-27B-BF16-00002-of-00002.gguf

echo "=== [1/5] download BF16 (2 parts, ~54 GB) ==="
for i in 1 2; do
  f=src-bf16/Qwen3.8-27B-BF16-0000$i-of-00002.gguf
  if [ ! -f "$f.done" ]; then
    wget -c --tries=10 --timeout=60 -q --show-progress -O "$f" \
      "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/BF16/Qwen3.8-27B-BF16-0000$i-of-00002.gguf" 2>&1 | tr '\r' '\n' | tail -1
    touch "$f.done"
  fi
done
ls -l src-bf16/ | awk '{print $9, $5}'

echo "=== [2/5] imatrix on BF16 ==="
if [ ! -f imatrix-27b-bf16.dat ]; then
  $LLAMA/llama-imatrix -m $B1 -f imatrix-mix.txt -o imatrix-27b-bf16.dat 2>&1 | tail -4
fi
cp imatrix-27b-bf16.dat imatrix-27b-bf16.dat.copy  # guard: partial writes

echo "=== [3/5] quantize from BF16 (split input, --merge) ==="
for q in IQ3_M IQ3_XS Q3_K_M; do
  $LLAMA/llama-quantize --merge --imatrix imatrix-27b-bf16.dat "$B1" "srcbf16-$q.gguf" "$q" 2>&1 | tail -4
done

echo "=== [4/5] sizes ==="
ls -l srcbf16-*.gguf | awk '{print $9, $5}'

echo "=== [5/5] 3-corpus floors ==="
source ~/forge_env/bin/activate
cd ~/model-forge
for q in IQ3_M IQ3_XS Q3_K_M; do
  python3 surgery/multicorpus.py 27b-$q-srcbf16 /home/daniel/m0/baseline-27b/srcbf16-$q.gguf
done
echo "=== BASELINE-27B BF16-CHAIN DONE ==="
