#!/usr/bin/env bash
set -uo pipefail
REG=~/m0/registry
CLI=~/llama.cpp/build/bin/llama-cli
mkdir -p ~/m0/sweep2
cd ~/m0
for m in Qwen3.5-4B-Q4_K_M Qwen3.5-4B-Q8_0 Qwen3.5-9B-Q4_K_M Qwen3.5-9B-Q8_0 Qwen3.8-27B-UD-Q4_K_M Qwen3.8-27B-Q8_0; do
  echo "=== $m ==="
  for T in 8 12 16 24 32; do
    log=~/m0/sweep2/${m}_t${T}.txt
    $CLI -m "$REG/$m.gguf" --file ~/m0/prompt.txt -n 64 -t $T -tb 96 --no-warmup < /dev/null > "$log" 2>&1
    tg=$(grep "^llama_print_timings:        eval time" "$log" | grep -oE "[0-9.]+ tokens per second")
    pp=$(grep "prompt eval time" "$log" | grep -oE "[0-9.]+ tokens per second")
    pptok=$(grep "prompt eval time" "$log" | grep -oE "/ +[0-9]+ tokens" | grep -oE "[0-9]+")
    echo "RESULT $m t=$T tb=96 tg=$tg pp=$pp pp_tokens=$pptok"
  done
done
echo SWEEP2_DONE
