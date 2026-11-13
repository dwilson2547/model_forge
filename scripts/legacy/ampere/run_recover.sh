#!/usr/bin/env bash
cd ~/model-forge
~/forge_env/bin/python -m surgery.recover /home/daniel/m0/runs/qwen35-4b-drop2-ranked/hf-edited /home/daniel/m0/data/wiki.train.small.txt --steps 100 --ctx 256 --batch 2 --out /home/daniel/m0/runs/qwen35-4b-drop2-ranked/recovered
echo RECOVER_DONE_RC=$?
