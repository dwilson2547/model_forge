#!/usr/bin/env bash
set -uo pipefail
REG=~/m0/registry
BENCH=~/llama.cpp/build/bin/llama-bench
CLI=~/llama.cpp/build/bin/llama-cli
HF=~/llama_env/bin/hf
cd ~/m0

dl() { repo=$1; file=$2
  if [ -f "$REG/$file" ]; then echo "HAVE $file"; else
    echo "DOWNLOAD $file"; $HF download "$repo" "$file" --local-dir "$REG" 2>&1 | tail -1
  fi
}

dl unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
dl unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf
dl unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf
dl unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q8_0.gguf
dl unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf
dl unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf

echo; echo "=== FILES ==="; ls -l $REG

echo; echo "=== THREAD SCALING (9B Q4_K_M, pp512 tg128, r2) ==="
$BENCH -m $REG/Qwen3.5-9B-Q4_K_M.gguf -t 16,32,48,64,80,96,128 -p 512 -n 128 -r 2 2>/dev/null | tee bench_threads_9b_q4.txt

echo; echo "=== MAIN GRID (all models, t32+t96, pp512 tg128, r2) ==="
for m in Qwen3.5-4B-Q4_K_M Qwen3.5-4B-Q8_0 Qwen3.5-9B-Q4_K_M Qwen3.5-9B-Q8_0 Qwen3.8-27B-UD-Q4_K_M Qwen3.8-27B-Q8_0; do
  for t in 32 96; do
    echo "--- $m t=$t ---"
    $BENCH -m "$REG/$m.gguf" -t $t -p 512 -n 128 -r 2 2>/dev/null | tee -a bench_grid.txt
    rm -rf ~/.cache/llama 2>/dev/null
  done
done

echo; echo "=== PEAK RSS (8 tok gen, no warmup, t96) ==="
PROMPT="Write a short paragraph about the sea."
for m in Qwen3.5-4B-Q4_K_M Qwen3.5-4B-Q8_0 Qwen3.5-9B-Q4_K_M Qwen3.5-9B-Q8_0 Qwen3.8-27B-UD-Q4_K_M Qwen3.8-27B-Q8_0; do
  echo "--- $m ---"
  /usr/bin/time -v $CLI -m "$REG/$m.gguf" -t 96 -p "$PROMPT" -n 8 --no-warmup -no-cnv > /dev/null 2> rss_tmp.txt
  grep -E "Maximum resident|Elapsed" rss_tmp.txt | tee -a rss_results.txt
done
echo "M0_BASELINE_DONE"
