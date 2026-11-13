#!/usr/bin/env bash
set -e
C=~/llama.cpp/build/bin
P=~/forge_env/bin/python
jobs=(
  "mixfull-recovered:mixfull"
  "codefw-recovered:codefw"
  "qwen35-4b-drop2-ranked/recovered-hf:wikirep"
  "qwen35-4b-drop2-ranked/hf-edited:unrepaired"
)
for hfout in mixfull codefw wikirep unrepaired; do mkdir -p ~/m0/lmstudio-out/$hfout; done
for job in "${jobs[@]}"; do
  hf="${job%%:*}"; name="${job##*:}"
  echo "=== $name: convert"
  $P ~/llama.cpp/convert_hf_to_gguf.py ~/m0/runs/$hf --outfile ~/m0/lmstudio-out/$name/$name-f16-mtp.gguf --outtype f16 2>&1 | tail -1
  echo "=== $name: quantize"
  $C/llama-quantize --allow-requantize --leave-output-tensor ~/m0/lmstudio-out/$name/$name-f16-mtp.gguf ~/m0/lmstudio-out/$name/$name-Q4_K_M-mtp.gguf Q4_K_M 2>&1 | tail -1
  echo "=== $name: verify nextn tensors + load sanity"
  $P - <<PY
import gguf
r = gguf.GGUFReader("/home/daniel/m0/lmstudio-out/$name/$name-Q4_K_M-mtp.gguf")
n = [t.name for t in r.tensors if "nextn" in t.name]
print("$name nextn tensors:", n)
PY
  $C/llama-cli -m ~/m0/lmstudio-out/$name/$name-Q4_K_M-mtp.gguf -p "The sea was" -n 8 -t 32 --no-warmup < /dev/null 2>/dev/null | grep -m1 -oE "\[ Prompt:.*\]" || echo "load/gen: check manually"
done
echo REBUILD_DONE
