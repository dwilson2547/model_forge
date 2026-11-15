#!/bin/bash
source ~/forge.env; cd ~/forge
Q=~/forge-runs/27b/Qwen3.8-27B-Q8_0.gguf
G=~/forge-runs/gemma/gemma-4-26B-A4B-it-Q8_0.gguf
smi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
run() { # label gguf mix slots seconds extra...
  l=$1 g=$2 m=$3 s=$4 t=$5; shift 5
  echo "=== $l $m slots=$s $(date -Is)"
  ( sleep 150; echo "  vram $(smi)" ) &
  timeout -s INT $t python -m distill.datagen $l $g --mix $m --slots $s "$@" 2>&1 | grep -v "^\s*$" | tail -60
  pkill -f llama-server; sleep 5
}
run q27-samp16 $Q tsamp 16 2400
run gem-samp16 $G tsamp 16 2400 --temp 1.0 --top-k 64 --top-p 0.95
run gem-s16 $G gpilot 16 900 --temp 1.0 --top-k 64 --top-p 0.95
echo "=== ALL DONE $(date -Is)"
