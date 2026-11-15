#!/bin/bash
while pgrep -f "/gen.sh" > /dev/null; do sleep 10; done
pkill -x llama-server; sleep 5
echo "=== vllm serve $(date -Is)"
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH
nohup ~/vllm-env/bin/vllm serve ~/forge-runs/27b-hf --served-model-name q27 --dtype bfloat16 \
  --max-model-len 20480 --gpu-memory-utilization 0.95 --port 8001 --limit-mm-per-prompt '{"image":0,"video":0}' \
  > ~/vllm-serve.log 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8001/health > /dev/null && break; kill -0 $SP 2>/dev/null || { echo "serve died"; tail -30 ~/vllm-serve.log; exit 1; }; sleep 10; done
curl -sf http://127.0.0.1:8001/health > /dev/null || { echo "serve not healthy"; tail -30 ~/vllm-serve.log; exit 1; }
echo "=== ready $(date -Is) vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for c in 64; do
  echo "=== q27-vllm-c$c tsamp $(date -Is)"
  ~/forge-env/bin/python ~/vgen.py q27-vllm-c$c tsamp q27 $c
  (source ~/forge.env; cd ~/forge; python -m distill.datagen export q27-vllm-c$c --mix tsamp)
done
echo "=== VLLM DONE $(date -Is)"
