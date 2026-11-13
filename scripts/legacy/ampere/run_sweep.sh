#!/usr/bin/env bash
set -uo pipefail
REG=~/m0/registry
CLI=~/llama.cpp/build/bin/llama-cli
mkdir -p ~/m0/sweep
cd ~/m0
for m in Qwen3.5-4B-Q4_K_M Qwen3.5-9B-Q4_K_M Qwen3.5-9B-Q8_0 Qwen3.8-27B-UD-Q4_K_M Qwen3.8-27B-Q8_0; do
  for T in 8 12 16 24 32; do
    log=~/m0/sweep/${m}_t${T}.txt
    /usr/bin/time -v $CLI -m "$REG/$m.gguf" --file ~/m0/prompt.txt -n 64 -t $T -tb 96 --no-warmup < /dev/null > /dev/null 2> "$log"
    tg=$(grep "eval time" "$log" | tail -1 | grep -oE "[0-9.]+ tokens per second")
    pp=$(grep "prompt eval time" "$log" | grep -oE "[0-9.]+ tokens per second")
    pptok=$(grep "prompt eval time" "$log" | grep -oE "/ +[0-9]+ tokens" | grep -oE "[0-9]+")
    rss=$(grep "Maximum resident" "$log" | grep -oE "[0-9]+$")
    echo "RESULT $m t=$T tb=96 tg=$tg pp=$pp pp_tokens=$pptok rss_kb=$rss"
  done
done
echo SWEEP_DONE
