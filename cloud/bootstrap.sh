#!/bin/bash
# forge-cloud bootstrap: fresh Brev H100 instance -> ready to train + quantize + floor.
# Safe to re-run (idempotent-ish). ~5-10 min; llama.cpp CUDA build is the long pole.
set -euo pipefail

echo "=== [1/4] system deps ==="
if command -v apt-get > /dev/null; then
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential cmake curl git python3-venv python3-pip ccache libcurl4-openssl-dev > /dev/null
fi

nvidia-smi || { echo "FATAL: no GPU visible"; exit 1; }

echo "=== [2/4] python env -> forge-env ==="
python3 -m venv forge-env
source forge-env/bin/activate
python3 -m pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q -r "$(dirname "$0")/requirements.txt"

echo "=== [3/4] llama.cpp with CUDA ==="
if [ ! -x ~/llama.cpp/build/bin/llama-perplexity ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
  cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_BUILD_PARALLEL_LEVEL=$(nproc) > /dev/null
  cmake --build ~/llama.cpp/build --config Release -j $(nproc)
fi

echo "=== [4/4] sanity ==="
python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch"
print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
EOF
~/llama.cpp/build/bin/llama-perplexity --version | head -1
echo "BOOTSTRAP DONE"
