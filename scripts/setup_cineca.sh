#!/bin/bash
# ============================================================================
# Setup script for CINECA Leonardo — run this ONCE on the login node.
# Uses python/3.11.7 + venv (no conda available on this allocation).
#
# Usage:  bash $WORK/hyperbolic_CLIP/scripts/setup_cineca.sh
# ============================================================================

set -e

echo "=== Attribution-CLIP setup on CINECA Leonardo ==="
echo "WORK: $WORK"

# ── Load modules ──────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
echo "Python: $(python --version)"
echo "CUDA module: cuda/12.6"

# ── Create venv ───────────────────────────────────────────────────────────────
VENV="$WORK/hyp_fine_tuning"

if [ -d "$VENV" ]; then
    echo "Venv already exists at $VENV, skipping creation."
else
    echo "Creating venv at $VENV ..."
    python -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --upgrade pip --quiet

# ── Install packages ──────────────────────────────────────────────────────────
echo "Installing PyTorch 2.5.1 + CUDA 12.1 wheels (compatible with CUDA 12.6)..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install \
    transformers==5.8.0 \
    peft==0.19.1 \
    accelerate==1.13.0 \
    huggingface_hub==1.14.0 \
    Pillow \
    pandas \
    numpy \
    tqdm \
    requests \
    scikit-learn

# ── Persistent env vars ───────────────────────────────────────────────────────
grep -qxF 'export HF_HOME=$WORK/hf_cache' ~/.bashrc || \
    echo 'export HF_HOME=$WORK/hf_cache' >> ~/.bashrc
grep -qxF 'export TOKENIZERS_PARALLELISM=false' ~/.bashrc || \
    echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
export HF_HOME=$WORK/hf_cache
export TOKENIZERS_PARALLELISM=false
echo "HF_HOME set to $HF_HOME"

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying PyTorch + CUDA ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ── Pre-download ViT-L/14 ─────────────────────────────────────────────────────
echo ""
echo "=== Pre-caching ViT-L/14 ==="
python -c "
import os
cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
vitl = os.path.join(cache, 'hub', 'models--openai--clip-vit-large-patch14')
if os.path.exists(vitl):
    print(f'ViT-L/14 already cached: {vitl}')
else:
    print('Downloading ViT-L/14 (first time, ~1.7 GB)...')
    from transformers import CLIPModel, CLIPProcessor
    CLIPModel.from_pretrained('openai/clip-vit-large-patch14', use_safetensors=True)
    CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    print('Done.')
"

echo ""
echo "=== Setup complete ==="
echo "Venv: $VENV"
echo "Submit job with: sbatch \$WORK/hyperbolic_CLIP/scripts/slurm_cineca.sh"
