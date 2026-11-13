#!/usr/bin/env bash
set -e
C=~/llama.cpp/build/bin
P=~/forge_env/bin/python
mkdir -p ~/m0/lmstudio-out/sftrep2
echo "=== sftrep2: graft MTP"
sed -e "s/\"mixfull-recovered\",/\"drop2-repair-sft2-hf\",/" -e "/\"fw-recovered\",/d" -e "/\"codefw-recovered\",/d" -e "/\"qwen35-4b-drop2-ranked\/recovered-hf\",/d" -e "/\"qwen35-4b-drop2-ranked\/hf-edited\",/d" -e "/\"drop2-repair-sft-hf\",/d" ~/m0/graft_mtp.py > ~/m0/graft_sft2.py
$P ~/m0/graft_sft2.py
echo "=== sftrep2: convert f16"
$P ~/llama.cpp/convert_hf_to_gguf.py ~/m0/runs/drop2-repair-sft2-hf --outfile ~/m0/lmstudio-out/sftrep2/sftrep2-f16-mtp.gguf --outtype f16 2>&1 | tail -1
echo "=== sftrep2: quantize Q4_K_M"
$C/llama-quantize --allow-requantize --leave-output-tensor ~/m0/lmstudio-out/sftrep2/sftrep2-f16-mtp.gguf ~/m0/lmstudio-out/sftrep2/sftrep2-Q4_K_M-mtp.gguf Q4_K_M 2>&1 | tail -1
echo "=== sftrep2: full chat suite"
cd ~/model-forge
$P -m bench.runner --chat sftrep2=$HOME/m0/lmstudio-out/sftrep2/sftrep2-f16-mtp.gguf
echo SFT2_DONE
