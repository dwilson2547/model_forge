#!/bin/bash
# 27B do-nothing baseline (review r2 finding 10): can an unsliced Qwen3.8-27B at
# 3-bit already sit in the 5080-comfort band (~12.5 GiB @ Q4-equivalent)? If yes,
# every sliced+repaired 27B candidate has to beat these rows.
# Pipeline: imatrix (from near-lossless Q8_0) -> IQ3_M / IQ3_XS / Q3_K_M quants ->
# size + 3-corpus PPL floor + Q8/Q4 reference rows.
set -e
cd ~/m0/baseline-27b
LLAMA=~/llama.cpp/build/bin
REG=~/m0/registry
echo "=== [1/4] imatrix calibration file (500k wiki + 500k fineweb + 200k code) ==="
{ head -c 500000 ~/m0/data/wiki.train.txt;
  head -c 500000 ~/m0/data/fineweb.test.txt;
  head -c 200000 ~/m0/data/code.test.txt; } > imatrix-mix.txt
if [ ! -f imatrix-27b.dat ]; then
  $LLAMA/llama-imatrix -m $REG/Qwen3.8-27B-Q8_0.gguf -f imatrix-mix.txt -o imatrix-27b.dat 2>&1 | tail -5
fi
echo "=== [2/4] quantize (source: Q8_0, near-lossless) ==="
for q in IQ3_M IQ3_XS Q3_K_M; do
  if [ ! -f Qwen3.8-27B-$q.gguf ]; then
    $LLAMA/llama-quantize --allow-requantize --imatrix imatrix-27b.dat $REG/Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-$q.gguf $q 2>&1 | tail -3
  fi
done
echo "=== [3/4] sizes ==="
ls -l Qwen3.8-27B-*.gguf | awk '{print $9, $5}'
echo "=== [4/4] 3-corpus PPL floor ==="
source ~/forge_env/bin/activate
cd ~/model-forge
python3 surgery/multicorpus.py 27b-q8-base $REG/Qwen3.8-27B-Q8_0.gguf
python3 surgery/multicorpus.py 27b-ud-q4km $REG/Qwen3.8-27B-UD-Q4_K_M.gguf
for q in IQ3_M IQ3_XS Q3_K_M; do
  python3 surgery/multicorpus.py 27b-$q /home/daniel/m0/baseline-27b/Qwen3.8-27B-$q.gguf
done
echo "=== BASELINE-27B DONE ==="
