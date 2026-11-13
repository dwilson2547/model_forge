#!/usr/bin/env bash
set -e
C=~/llama.cpp/build/bin
P=~/forge_env/bin/python
mkdir -p ~/m0/lmstudio-out/sftrep3
echo "=== sftrep3: graft MTP"
sed -e "s/\"mixfull-recovered\",/\"drop2-repair-sft3-hf\",/" -e "/\"fw-recovered\",/d" -e "/\"codefw-recovered\",/d" -e "/\"qwen35-4b-drop2-ranked\/recovered-hf\",/d" -e "/\"qwen35-4b-drop2-ranked\/hf-edited\",/d" -e "/\"drop2-repair-sft-hf\",/d" -e "/\"drop2-repair-sft2-hf\",/d" ~/m0/graft_mtp.py > ~/m0/graft_sft3.py
$P ~/m0/graft_sft3.py
echo "=== sftrep3: convert f16"
$P ~/llama.cpp/convert_hf_to_gguf.py ~/m0/runs/drop2-repair-sft3-hf --outfile ~/m0/lmstudio-out/sftrep3/sftrep3-f16-mtp.gguf --outtype f16 2>&1 | tail -1
echo "=== sftrep3: quantize Q4_K_M"
$C/llama-quantize --allow-requantize --leave-output-tensor ~/m0/lmstudio-out/sftrep3/sftrep3-f16-mtp.gguf ~/m0/lmstudio-out/sftrep3/sftrep3-Q4_K_M-mtp.gguf Q4_K_M 2>&1 | tail -1
echo "=== sftrep3: chat suite"
cd ~/model-forge
$P -m bench.runner --chat sftrep3=$HOME/m0/lmstudio-out/sftrep3/sftrep3-f16-mtp.gguf
echo SFT3_DONE
