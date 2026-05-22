#!/bin/bash
# ============================================================================
# Setup script for CINECA Leonardo — run this ONCE on the login node.
# Creates the conda environment and verifies the installation.
#
# Usage:  bash $WORK/hyperbolic_CLIP/scripts/setup_cineca.sh
# ============================================================================

set -e

echo "=== Attribution-CLIP setup on CINECA Leonardo ==="
echo "WORK: $WORK"

# ── Load modules ──────────────────────────────────────────────────────────────
# Find the right anaconda/miniconda module name (varies by allocation/year)
_CONDA_MOD=""
for _m in anaconda3/2023.03-2 anaconda3/2023.09 anaconda3/2024.02 \
           miniconda3/24.1.2-0 miniconda3/23.5.2-0 miniconda3/4.12.0; do
    if module load "$_m" 2>/dev/null; then
        _CONDA_MOD="$_m"
        echo "Loaded conda module: $_CONDA_MOD"
        break
    fi
done
if [ -z "$_CONDA_MOD" ]; then
    echo "ERROR: no anaconda/miniconda module found. Run 'module avail anaconda' and set the name manually."
    exit 1
fi

# Find the right CUDA module
_CUDA_MOD=""
for _m in cuda/12.1 cuda/12.3 cuda/12.4 cuda/11.8; do
    if module load "$_m" 2>/dev/null; then
        _CUDA_MOD="$_m"
        echo "Loaded CUDA module: $_CUDA_MOD"
        break
    fi
done
if [ -z "$_CUDA_MOD" ]; then
    echo "WARNING: no CUDA module loaded. PyTorch GPU support may not work."
fi

# ── Create conda environment ──────────────────────────────────────────────────
ENV_NAME="deepfake-hyp"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists, skipping creation."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.11 -y
fi

conda activate $ENV_NAME

# ── Install packages ──────────────────────────────────────────────────────────
echo "Installing PyTorch 2.5.1 + CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install \
    transformers==5.8.0 \
    peft==0.19.1 \
    accelerate==1.13.0 \
    huggingface_hub==1.14.0 \
    Pillow \
    pandas==3.0.3 \
    numpy==2.4.4 \
    tqdm==4.67.3 \
    requests==2.34.0 \
    scikit-learn==1.8.0

# ── HuggingFace cache ─────────────────────────────────────────────────────────
echo "Setting HF_HOME to \$WORK/hf_cache (add this to your ~/.bashrc)..."
echo 'export HF_HOME=$WORK/hf_cache' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
export HF_HOME=$WORK/hf_cache

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying GPU access ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ── Verify model cache ────────────────────────────────────────────────────────
echo ""
echo "=== Verifying ViT-L/14 cache ==="
python -c "
import os
cache = os.environ.get('HF_HOME', '~/.cache/huggingface')
vitl = os.path.join(cache, 'hub', 'models--openai--clip-vit-large-patch14')
if os.path.exists(vitl):
    print(f'ViT-L/14 found in cache: {vitl}')
else:
    print('ViT-L/14 NOT in cache — downloading now...')
    from transformers import CLIPModel
    CLIPModel.from_pretrained('openai/clip-vit-large-patch14', use_safetensors=True)
    print('Done.')
"

echo ""
echo "=== Setup complete. Submit with: sbatch \$WORK/hyperbolic_CLIP/scripts/slurm_cineca.sh ==="
