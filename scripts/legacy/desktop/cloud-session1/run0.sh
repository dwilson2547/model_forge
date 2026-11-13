set -euo pipefail
source ~/forge.env
cd ~/forge-runs
( python ~/llama.cpp/convert_hf_to_gguf.py 2529-hf-edited --outfile unrep-f16.gguf --outtype f16 > conv0a.log 2>&1 \
  && ~/llama.cpp/build/bin/llama-quantize unrep-f16.gguf unrep-Q4_K_M.gguf Q4_K_M > quant0a.log 2>&1 \
  && python ~/forge/surgery/multicorpus.py cloud-unrep-f16 $PWD/unrep-f16.gguf cloud-unrep-q4 $PWD/unrep-Q4_K_M.gguf > floor0a.log 2>&1 \
  && echo RUN0A-DONE >> floor0a.log || echo RUN0A-FAILED >> floor0a.log ) &
cd ~/forge
python -m surgery.recover_gpu ~/forge-runs/2529-hf-edited ~/forge-data/mixed.train.txt \
  --seed 1234 --epochs 2 --max-chars 12000000 --lr 1e-4 --lora-targets all --ctx 512 --batch 4 \
  --out ~/forge-runs/run0-parity > ~/forge-runs/run0b.log 2>&1 && echo RUN0B-DONE >> ~/forge-runs/run0b.log || echo RUN0B-FAILED >> ~/forge-runs/run0b.log
wait
