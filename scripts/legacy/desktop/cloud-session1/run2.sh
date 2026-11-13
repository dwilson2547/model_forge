set -uo pipefail
source ~/forge.env
R=~/forge-runs
floor() {  # $1 = run dir, $2 = label
  cd $1 && python ~/llama.cpp/convert_hf_to_gguf.py hf-recovered --outfile $2-f16.gguf --outtype f16 > conv.log 2>&1 \
  && ~/llama.cpp/build/bin/llama-quantize $2-f16.gguf $2-Q4_K_M.gguf Q4_K_M > quant.log 2>&1 \
  && python ~/forge/surgery/multicorpus.py $2-f16 $PWD/$2-f16.gguf $2-q4 $PWD/$2-Q4_K_M.gguf > floor.log 2>&1 \
  && echo "FLOOR-DONE $2" >> floor.log || echo "FLOOR-FAILED $2" >> $1/floor.log
}
cd ~/forge
for LR in 1e-6; do
  python -m surgery.recover_gpu $R/2529-hf-edited ~/forge-data/mixed.train.txt \
    --full --lr $LR --seed 1234 --epochs 1 --max-chars 41000000 --val-frac 0.01 \
    --stop-on-val-rise 2 --save-every 0 --ctx 512 --batch 4 \
    --out $R/run1-full-lr$LR > $R/run1-lr$LR.log 2>&1 \
    && { echo "RUN1-DONE lr$LR" >> $R/run1-lr$LR.log; floor $R/run1-full-lr$LR cloud-run1-lr$LR & } \
    || echo "RUN1-FAILED lr$LR" >> $R/run1-lr$LR.log
  cd ~/forge
done
wait
echo ALL-DONE > $R/all2.done
