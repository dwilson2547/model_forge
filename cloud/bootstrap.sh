#!/bin/bash
# forge-cloud bootstrap: fresh rented CUDA instance -> ready to train + quantize + floor.
# Idempotent: re-running skips what is already in place. Run from anywhere:
#   bash ~/forge/cloud/bootstrap.sh
# Produces ~/forge-env (venv), ~/llama.cpp (pinned, CUDA), ~/forge.env (paths for the
# floor scripts). The llama.cpp CUDA build is the long pole.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Same commit as the Altra build that produced every floor row (TODO M0 pin).
LLAMA_SHA=e95426f16982054c379ca7bdc8186996dd22f0fe
CUDA_ARCH="${CUDA_ARCH:-90}"            # 90 = H100; 89 = L40S/4090; 80 = A100

echo "=== [1/6] system deps ==="
if command -v apt-get > /dev/null; then
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential cmake curl git python3-venv python3-pip python3-dev ccache \
    libcurl4-openssl-dev rsync > /dev/null
fi
nvidia-smi > /dev/null || { echo "FATAL: no GPU visible"; exit 1; }
# torch wheel: the desktop runs 2.15.0+cu132. Use it when the driver allows; older
# drivers get the nearest wheel and a loud parity warning (Run 0 then measures it).
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
if ver_ge "$DRV_CUDA" 13.2; then TORCH="torch==2.15.0"; IDX=cu132
elif ver_ge "$DRV_CUDA" 13.0; then TORCH="torch==2.14.1"; IDX=cu130
else TORCH="torch==2.14.1"; IDX=cu126; fi
echo "driver CUDA $DRV_CUDA -> $TORCH ($IDX)"
[ "$IDX" = cu132 ] || echo "WARNING: not the desktop's torch 2.15.0+cu132 — Run 0 parity is now also testing this torch"

# nvcc for the llama.cpp CUDA build (and causal-conv1d). Many GPU VM images ship only
# the driver: install the toolkit matching the torch wheel from NVIDIA's apt repo.
export PATH="/usr/local/cuda/bin:$PATH"
if ! command -v nvcc > /dev/null; then
  TK="cuda-toolkit-${IDX:2:2}-${IDX:4}"            # cu132 -> cuda-toolkit-13-2
  . /etc/os-release
  KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "no nvcc: installing $TK from $KEYRING_URL"
  curl -fsSLo /tmp/cuda-keyring.deb "$KEYRING_URL"
  sudo dpkg -i /tmp/cuda-keyring.deb > /dev/null
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$TK" > /dev/null
  [ -d /usr/local/cuda ] || sudo ln -s "/usr/local/cuda-${IDX:2:2}.${IDX:4}" /usr/local/cuda
fi
command -v nvcc > /dev/null || { echo "FATAL: still no nvcc after the toolkit install"; exit 1; }
nvcc --version | tail -1

echo "=== [2/6] python env -> ~/forge-env ==="
[ -d ~/forge-env ] || python3 -m venv ~/forge-env
source ~/forge-env/bin/activate
python3 -m pip install -q --upgrade pip wheel setuptools
pip install -q "$TORCH" --index-url "https://download.pytorch.org/whl/$IDX"
pip install -q -r "$REPO/cloud/requirements.txt"

echo "=== [3/6] causal-conv1d (optional fast conv kernel; source build) ==="
# Not installed on the desktop either (transformers warns and uses the torch reference
# conv). Worth having on paid time; a failed build is not fatal.
if ! python3 -c "import causal_conv1d" 2> /dev/null; then
  TORCH_CUDA_ARCH_LIST="${CUDA_ARCH:0:1}.${CUDA_ARCH:1}" MAX_JOBS="$(nproc)" \
    pip install -q --no-build-isolation causal-conv1d==1.7.0 \
    || echo "WARNING: causal-conv1d build failed — continuing on the torch reference conv"
fi

echo "=== [4/6] llama.cpp @ ${LLAMA_SHA:0:7} with CUDA ==="
if [ ! -d ~/llama.cpp/.git ]; then
  git init -q ~/llama.cpp
  git -C ~/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp
fi
if [ "$(git -C ~/llama.cpp rev-parse HEAD 2> /dev/null || true)" != "$LLAMA_SHA" ]; then
  git -C ~/llama.cpp fetch -q --depth 1 origin "$LLAMA_SHA"
  git -C ~/llama.cpp checkout -q "$LLAMA_SHA"
  rm -rf ~/llama.cpp/build
fi
if [ ! -x ~/llama.cpp/build/bin/llama-perplexity ]; then
  cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build ~/llama.cpp/build --config Release -j "$(nproc)" \
    --target llama-perplexity llama-quantize llama-imatrix llama-cli > /dev/null
fi

echo "=== [5/6] ~/forge.env ==="
mkdir -p ~/forge-data ~/forge-runs
cat > ~/forge.env <<EOF
# sourced by every cloud session: floor scripts read these (surgery/multicorpus.py)
export FORGE_LLAMA=\$HOME/llama.cpp
export FORGE_DATA=\$HOME/forge-data
export FORGE_RESULTS=\$HOME/forge-runs/multicorpus-results.jsonl
export FORGE_NGL=99
export FORGE_THREADS=$(nproc)
export PATH=/usr/local/cuda/bin:\$PATH
source \$HOME/forge-env/bin/activate
EOF

echo "=== [6/6] sanity ==="
python3 - <<'EOF'
import importlib.util, torch
assert torch.cuda.is_available(), "CUDA not available to torch"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}, GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB")
for mod in ("fla", "causal_conv1d", "bitsandbytes", "peft"):
    print(f"  {mod:14s} {'ok' if importlib.util.find_spec(mod) else 'MISSING'}")
assert importlib.util.find_spec("fla"), "flash-linear-attention missing: slow GatedDeltaNet path"
EOF
~/llama.cpp/build/bin/llama-perplexity --version 2>&1 | head -1
echo "BOOTSTRAP DONE — next: source ~/forge.env"
