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
#
# The shared venv site-packages is READ-ONLY for us (owned by whoever created the
# venv), so a plain `pip install` dies with "Permission denied". Install into a
# personal overlay dir and put it on PYTHONPATH instead. --target does NOT touch
# deps already satisfied by the venv (numpy/scipy/pillow stay the venv's), so only
# the missing packages land in PYDEPS — no risk of shadowing torch's numpy.
PYDEPS="${IAB_PYDEPS:-$HOME/iab_pydeps}"
mkdir -p "$PYDEPS"
pip install --no-cache-dir --target "$PYDEPS" 'albumentations<1.4'
export PYTHONPATH="$PYDEPS:${PYTHONPATH:-}"
echo "DNA deps installed into $PYDEPS (add it to PYTHONPATH in the DNA slurm scripts)"

# UCF initialises both Xception encoders from ImageNet weights; without them the
# original authors note the model does not converge to anything useful.
mkdir -p "$REPO/pretrained"
# Original host (data.lip6.fr/cadene/...) is dead: expired TLS cert + 503. Use the
# Hugging Face mirror (valid cert) and VERIFY the sha256 — the file is authentic
# regardless of transport. Expected hash below is also the source of the -b5690688
# filename suffix (pretrainedmodels convention: sha256[:8]).
XCEPTION_SHA=b56906886cbaf573fc8819216d2dcc753fb564fa81a7733a0f2620b2a1973f7a
XCEPTION_PATH="$REPO/pretrained/xception-b5690688.pth"
curl -fL -o "$XCEPTION_PATH" \
  https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth
echo "$XCEPTION_SHA  $XCEPTION_PATH" | sha256sum -c - \
  || { echo "ERROR: xception weights failed sha256 check — not using them." >&2; exit 1; }

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
