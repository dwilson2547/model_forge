#!/bin/bash
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH
pkill -f "vllm serve"; sleep 5
~/forge-env/bin/hf download google/gemma-4-26B-A4B-it --local-dir ~/forge-runs/gemma-hf > ~/dl-gemma-hf.log 2>&1 || { echo "download failed"; tail -3 ~/dl-gemma-hf.log; exit 1; }
du -sh ~/forge-runs/gemma-hf
echo "=== vllm serve $(date -Is)"
nohup vllm serve ~/forge-runs/gemma-hf --served-model-name gem --dtype bfloat16 --max-model-len 20480 \
  --gpu-memory-utilization 0.95 --port 8001 --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' > ~/vllm-serve-gemma.log 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8001/health > /dev/null && break; kill -0 $SP 2>/dev/null || { echo "serve died"; exit 1; }; sleep 10; done
curl -sf http://127.0.0.1:8001/health > /dev/null || { echo "serve not healthy"; exit 1; }
echo "=== ready $(date -Is) vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
sed -e 's/"temperature": 0.6, "top_p": 0.95, "top_k": 20/"temperature": 1.0, "top_p": 0.95, "top_k": 64/' ~/vgen.py > ~/vgen-gem.py
~/forge-env/bin/python ~/vgen-gem.py gem-vllm-c160 tsamp gem 160
(source ~/forge.env; cd ~/forge; python -m distill.datagen export gem-vllm-c160 --mix tsamp)
echo "=== GEM DONE $(date -Is)"
