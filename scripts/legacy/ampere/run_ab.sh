#!/usr/bin/env bash
set -uo pipefail
REG=/models
docker run --rm --entrypoint sh -v ~/m0/registry:/models -v ~/m0/prompt.txt:/prompt.txt amperecomputingai/llama.cpp:latest -c '
for m in Qwen3.5-9B-Q4_K_M Qwen3.8-27B-UD-Q4_K_M; do
  echo "=== AMPERE-BIN plain $m ==="
  /llm/llama-cli -m /models/$m.gguf --file /prompt.txt -n 64 -t 32 -tb 96 --no-warmup < /dev/null 2>/dev/null | grep -oE "\[ Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s \]"
  echo "=== AMPERE-BIN +AIO-init $m ==="
  . /.aio_init/init.sh >/dev/null 2>&1
  /llm/llama-cli -m /models/$m.gguf --file /prompt.txt -n 64 -t 32 -tb 96 --no-warmup < /dev/null 2>/dev/null | grep -oE "\[ Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s \]"
done'
echo AB_DONE
