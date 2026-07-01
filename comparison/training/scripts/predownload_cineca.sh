#!/bin/bash
# ============================================================================
# Run ONCE on a CINECA Leonardo LOGIN NODE (which has internet). Compute nodes
# are offline, so every pretrained weight / package a job needs must be fetched
# here first. Idempotent: re-running just re-checks the caches.
#
#   bash comparison/training/scripts/predownload_cineca.sh
# ============================================================================

set -euo pipefail

module load python/3.11.7
source $WORK/hyp_fine_tuning/bin/activate

export TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache   # MUST match the training scripts

echo "== 1/3 pip deps (yacs for HiFi-Net, openai-CLIP for DEFL) =="
pip install --quiet yacs
python -c "import clip" 2>/dev/null || pip install --quiet git+https://github.com/openai/CLIP.git

echo "== 2/3 torchvision ResNet-50 ImageNet weights -> $TORCH_HOME (resnet50 + defl) =="
python -c "import torchvision.models as m; m.resnet50(pretrained=True); print('  resnet50 weights cached')"

echo "== 3/3 openai-CLIP RN50x16 -> \$HOME/.cache/clip (defl) =="
python -c "import clip; clip.load('RN50x16', device='cpu'); print('  RN50x16 cached')"

echo "== done. You can now sbatch the training jobs. =="
