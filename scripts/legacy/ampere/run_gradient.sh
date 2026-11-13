#!/usr/bin/env bash
set -o pipefail
cd ~/model-forge
P=~/forge_env/bin/python
W=/home/daniel/m0/data/wiki.test.txt
for r in qwen35-4b-drop2-ranked qwen35-4b-drop2-ranked-fused \
         qwen35-4b-drop4-ranked qwen35-4b-drop4-ranked-fused \
         qwen35-4b-drop6-ranked qwen35-4b-drop6-ranked-fused; do
  echo "=========== RECIPE $r ==========="
  $P -m surgery.pipeline recipes/$r.yaml --ppl-file $W --prompt-file ~/m0/prompt.txt --chunks 24 2>&1 | tail -25
done
echo GRADIENT_DONE
