#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Lucy 2.0 PoC — RunPod Environment Setup
# Run this on a fresh RunPod pod (1xH100 SXM or 4xH100 SXM NVLink)
#
# Usage:
#   bash scripts/setup/runpod_setup.sh [--skip-repos] [--skip-models]
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SKIP_REPOS=false
SKIP_MODELS=false
for arg in "$@"; do
  case $arg in
    --skip-repos) SKIP_REPOS=true ;;
    --skip-models) SKIP_MODELS=true ;;
  esac
done

# Auto-detect lucy repo root (parent of scripts/setup/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUCY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKSPACE="/workspace"
REPOS_DIR="$WORKSPACE/repos"
CKPT_DIR="$WORKSPACE/checkpoints"
NETWORK_VOL="$WORKSPACE/network-volume"
RESULTS_DIR="$WORKSPACE/results"

echo "=== Lucy 2.0 PoC — Environment Setup ==="
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── 1. System packages ───────────────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq && apt-get install -y -qq git git-lfs ffmpeg htop nvtop 2>/dev/null || true

# ── 2. Clone repos ────────────────────────────────────────────
if [ "$SKIP_REPOS" = false ]; then
  echo "[2/5] Cloning repositories..."
  mkdir -p "$REPOS_DIR"

  # Check network volume for cached repos first
  if [ -d "$NETWORK_VOL/repos/StreamDiffusionV2" ]; then
    echo "  Using cached repos from network volume..."
    ln -sf "$NETWORK_VOL/repos/StreamDiffusionV2" "$REPOS_DIR/StreamDiffusionV2"
    ln -sf "$NETWORK_VOL/repos/stable-fast" "$REPOS_DIR/stable-fast" 2>/dev/null || true
    ln -sf "$NETWORK_VOL/repos/TurboDiffusion" "$REPOS_DIR/TurboDiffusion" 2>/dev/null || true
    ln -sf "$NETWORK_VOL/repos/rcm" "$REPOS_DIR/rcm" 2>/dev/null || true
  else
    clone_if_missing() {
      local url=$1 dir=$2
      if [ ! -d "$dir" ]; then
        echo "  Cloning $url..."
        git clone --depth 1 "$url" "$dir"
      else
        echo "  $dir already exists, skipping."
      fi
    }
    clone_if_missing "https://github.com/chenfengxu714/StreamDiffusionV2.git" "$REPOS_DIR/StreamDiffusionV2"
    clone_if_missing "https://github.com/chengzeyi/stable-fast.git" "$REPOS_DIR/stable-fast"
    clone_if_missing "https://github.com/thu-ml/TurboDiffusion.git" "$REPOS_DIR/TurboDiffusion"
    clone_if_missing "https://github.com/NVlabs/rcm.git" "$REPOS_DIR/rcm"
  fi
else
  echo "[2/5] Skipping repo clone (--skip-repos)"
fi

# ── 3. Python dependencies ────────────────────────────────────
echo "[3/5] Installing Python dependencies..."

# Only install what's missing — skip torch/flash-attn/xformers (already in RunPod image)
echo "  Checking pre-installed packages..."
python3 -c "import torch; print(f'  torch {torch.__version__} ✓')" 2>/dev/null || true
python3 -c "import flash_attn; print(f'  flash-attn {flash_attn.__version__} ✓')" 2>/dev/null || true
python3 -c "import xformers; print(f'  xformers {xformers.__version__} ✓')" 2>/dev/null || true

echo "  Installing missing packages..."
pip install -q -r "$LUCY_ROOT/scripts/setup/requirements_poc.txt" 2>/dev/null || true

# Only build flash-attn if NOT already installed
python3 -c "import flash_attn" 2>/dev/null || {
  echo "  flash-attn not found, building from source (this takes ~15min)..."
  pip install -q flash-attn --no-build-isolation 2>/dev/null || \
    echo "  WARNING: flash-attn build failed, will use flex_attention fallback"
}

# StreamDiffusionV2 — add __init__.py where missing so causvid/streamv2v are importable
if [ -d "$REPOS_DIR/StreamDiffusionV2" ]; then
  echo "  Making StreamDiffusionV2 packages importable..."
  touch "$REPOS_DIR/StreamDiffusionV2/causvid/__init__.py" 2>/dev/null || true
  touch "$REPOS_DIR/StreamDiffusionV2/streamv2v/__init__.py" 2>/dev/null || true
  # Write a persistent PYTHONPATH config
  echo "$REPOS_DIR/StreamDiffusionV2" > /etc/python_paths.pth 2>/dev/null || true
  # Also write to site-packages for immediate effect
  SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
  if [ -n "$SITE_DIR" ]; then
    echo "$REPOS_DIR/StreamDiffusionV2" > "$SITE_DIR/streamdiffv2.pth" 2>/dev/null || true
    echo "  Added .pth file to $SITE_DIR"
  fi
fi

# ── 4. Download models ────────────────────────────────────────
if [ "$SKIP_MODELS" = false ]; then
  echo "[4/5] Downloading model weights..."
  mkdir -p "$CKPT_DIR"

  # Check network volume first
  if [ -d "$NETWORK_VOL/checkpoints/Wan2.1-T2V-1.3B" ]; then
    echo "  Using cached checkpoints from network volume..."
    ln -sf "$NETWORK_VOL/checkpoints/Wan2.1-T2V-1.3B" "$CKPT_DIR/Wan2.1-T2V-1.3B"
  else
    python3 -c "
from huggingface_hub import snapshot_download
import os

ckpt_dir = '$CKPT_DIR'

# Wan2.1 T2V 1.3B pretrained
print('  Downloading Wan2.1-T2V-1.3B...')
snapshot_download(
    'Wan-AI/Wan2.1-T2V-1.3B',
    local_dir=os.path.join(ckpt_dir, 'Wan2.1-T2V-1.3B'),
)
print('  Done.')
" || echo "  WARNING: Model download failed. Download manually:
  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $CKPT_DIR/Wan2.1-T2V-1.3B"
  fi

  # CausVid checkpoint — only need autoregressive_checkpoint/model.pt (~5GB), NOT the full 122GB repo
  if [ -f "$CKPT_DIR/causvid/model.pt" ]; then
    echo "  CausVid checkpoint already exists."
  elif [ -d "$NETWORK_VOL/checkpoints/causvid" ]; then
    ln -sf "$NETWORK_VOL/checkpoints/causvid" "$CKPT_DIR/causvid"
    echo "  Using cached CausVid checkpoint from network volume."
  else
    echo "  Downloading CausVid autoregressive checkpoint only..."
    mkdir -p "$CKPT_DIR/causvid"
    python3 -c "
from huggingface_hub import hf_hub_download
import os

ckpt_dir = os.path.join('$CKPT_DIR', 'causvid')

# Only download the autoregressive checkpoint (model.pt), not the full 122GB repo
print('  Downloading autoregressive_checkpoint/model.pt...')
hf_hub_download(
    'tianweiy/CausVid',
    filename='autoregressive_checkpoint/model.pt',
    local_dir=ckpt_dir,
)
# Move model.pt to expected location
src = os.path.join(ckpt_dir, 'autoregressive_checkpoint', 'model.pt')
dst = os.path.join(ckpt_dir, 'model.pt')
if os.path.exists(src) and not os.path.exists(dst):
    os.rename(src, dst)
print('  Done.')
" || echo "  WARNING: CausVid download failed. Download manually:
  huggingface-cli download tianweiy/CausVid autoregressive_checkpoint/model.pt --local-dir $CKPT_DIR/causvid"
  fi
else
  echo "[4/5] Skipping model download (--skip-models)"
fi

# ── 5. Create results dir & verify ────────────────────────────
echo "[5/5] Final setup..."
mkdir -p "$RESULTS_DIR"/{phase1,phase2,phase3,phase4}

# Verify environment
echo ""
echo "=== Environment Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'CUDA version: {torch.version.cuda}')
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
    print(f'GPU memory: {mem:.1f} GB')
    print(f'SM capability: {torch.cuda.get_device_capability()}')

try:
    import flash_attn
    print(f'flash-attn: {flash_attn.__version__}')
except ImportError:
    print('flash-attn: NOT INSTALLED')

try:
    import diffusers
    print(f'diffusers: {diffusers.__version__}')
except ImportError:
    print('diffusers: NOT INSTALLED')

try:
    import xformers
    print(f'xformers: {xformers.__version__}')
except ImportError:
    print('xformers: NOT INSTALLED')
"
echo ""
echo "=== Setup Complete ==="
echo "Next: Run Phase 1 baseline with:"
echo "  python scripts/phase1/run_baseline.py --config configs/experiment_configs.yaml"
