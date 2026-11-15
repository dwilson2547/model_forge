#!/bin/bash
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH
mk() { sed -e 's/"enable_thinking": False/"enable_thinking": True/' -e 's/"max_tokens": it\["max_tokens"\]/"max_tokens": 8192/' $1 > $2; }
mk ~/vgen-gem.py ~/vgen-gem-think.py; mk ~/vgen.py ~/vgen-think.py
grep -c '"enable_thinking": True' ~/vgen-gem-think.py ~/vgen-think.py
echo "=== gem think $(date -Is)"
~/forge-env/bin/python ~/vgen-gem-think.py gem-vllm-think tsamp gem 160
pkill -f "vllm serve"; sleep 8
echo "=== q27 serve $(date -Is)"
nohup vllm serve ~/forge-runs/27b-hf --served-model-name q27 --dtype bfloat16 --max-model-len 20480 \
  --gpu-memory-utilization 0.95 --port 8001 --limit-mm-per-prompt '{"image":0,"video":0}' > ~/vllm-serve2.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8001/health > /dev/null && break; sleep 10; done
echo "=== q27 think $(date -Is)"
~/forge-env/bin/python ~/vgen-think.py q27-vllm-think tsamp q27 160
echo "=== THINK DONE $(date -Is)"
