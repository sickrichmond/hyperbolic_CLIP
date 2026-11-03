#!/bin/bash
# Run once on a login node with network access before submitting training.
set -euo pipefail
module load python/3.11.7
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TORCH_HOME="$WORK/hyp_fine_tuning/torch_cache"
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
cd "$REPO"
PYDEPS="${IAB_PYDEPS:-$HOME/iab_pydeps}"
mkdir -p "$PYDEPS"
# Reuse the baseline environment's torch/numpy dependencies; never shadow them.
python -m pip install --no-deps --target "$PYDEPS" -r comparison/requirements_omnidfa.txt
export PYTHONPATH="$REPO:$PYDEPS:${PYTHONPATH:-}"
python -c "import timm, tensorboard"
python -c "from timm import create_model; create_model('convnext_small', pretrained=True, num_classes=0)"
echo 'OmniDFA dependencies and ImageNet weights are ready.'
