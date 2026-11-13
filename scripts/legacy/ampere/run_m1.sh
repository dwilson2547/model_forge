#!/usr/bin/env bash
set -o pipefail
cd ~/model-forge
P=~/forge_env/bin/python
W=/home/daniel/m0/data/wiki.test.txt
for r in qwen35-4b-base qwen35-4b-drop8-ranked qwen35-4b-drop8-slice85; do
  echo "=========== RECIPE $r ==========="
  $P -m surgery.pipeline recipes/$r.yaml --ppl-file $W --prompt-file ~/m0/prompt.txt --chunks 24 2>&1 | tail -40
done
echo M1_RUN_DONE
