#!/bin/bash
# One-time setup for the four new baselines (DNA-Det, RepMix, PatchForensics, UCF).
# Run this on a CINECA LOGIN NODE — compute nodes have no network access.
#
#   bash comparison/training/scripts/prefetch_new_methods.sh
set -euo pipefail

module load python/3.11.7
source $WORK/hyp_fine_tuning/bin/activate
export TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache
REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo

# DNA-Det builds its 170 transformation classes with albumentations. Pin < 1.4:
# A.JpegCompression was renamed ImageCompression and always_apply / GaussNoise's
# var_limit changed in later versions — on 2.x the transform list fails to build.
pip install 'albumentations<1.4'

# UCF initialises both Xception encoders from ImageNet weights; without them the
# original authors note the model does not converge to anything useful.
mkdir -p "$REPO/pretrained"
curl -fL -o "$REPO/pretrained/xception-b5690688.pth" \
  http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth

# RepMix's backbone is a torchvision resnet50(pretrained=True) -> warm TORCH_HOME.
python -c "import torchvision; torchvision.models.resnet50(pretrained=True)"

# Sanity check: every method must be registered, with no defensive-import warning.
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python -m comparison.training.tests_new_methods

echo "Setup done. Launch the long jobs first:"
echo "  sbatch comparison/training/scripts/cineca_defl_train_22cls.sh      # re-train after the label_mapping fix"
echo "  sbatch comparison/training/scripts/cineca_dna_pretrain_22cls.sh    # DNA stage 1"
echo "  sbatch comparison/training/scripts/cineca_{repmix,patch,ucf}_train_22cls.sh"
