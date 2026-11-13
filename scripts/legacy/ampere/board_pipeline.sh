#!/usr/bin/env bash
set -e
C=~/llama.cpp/build/bin
P=~/forge_env/bin/python
mkdir -p ~/m0/lmstudio-out/sftrep
echo "=== sftrep: convert f16"
$P ~/llama.cpp/convert_hf_to_gguf.py ~/m0/runs/drop2-repair-sft-hf --outfile ~/m0/lmstudio-out/sftrep/sftrep-f16-mtp.gguf --outtype f16 2>&1 | tail -1
echo "=== sftrep: quantize Q4_K_M"
$C/llama-quantize --allow-requantize --leave-output-tensor ~/m0/lmstudio-out/sftrep/sftrep-f16-mtp.gguf ~/m0/lmstudio-out/sftrep/sftrep-Q4_K_M-mtp.gguf Q4_K_M 2>&1 | tail -1
$C/llama-cli -m ~/m0/lmstudio-out/sftrep/sftrep-f16-mtp.gguf -p "The sea was" -n 8 -t 32 --no-warmup < /dev/null 2>/dev/null | grep -m1 -oE "\[ Prompt:.*\]" || echo "gen sanity: check manually"
echo "=== chat-mode board: 6 arms"
cd ~/model-forge
$P -m bench.runner --chat \
  base=$HOME/m0/runs/qwen35-4b-base/qwen35-4b-base-f16.gguf \
  drop2=$HOME/m0/lmstudio-out/unrepaired/unrepaired-f16-mtp.gguf \
  wikirep=$HOME/m0/lmstudio-out/wikirep/wikirep-f16-mtp.gguf \
  codefw=$HOME/m0/lmstudio-out/codefw/codefw-f16-mtp.gguf \
  mixfull=$HOME/m0/lmstudio-out/mixfull/mixfull-f16-mtp.gguf \
  sftrep=$HOME/m0/lmstudio-out/sftrep/sftrep-f16-mtp.gguf
echo BOARD_DONE
