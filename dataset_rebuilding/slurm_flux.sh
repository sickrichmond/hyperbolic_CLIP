#!/bin/bash
# ============================================================================
# CINECA Leonardo — regenerate FAKE images with FLUX from the new captions.
#
# Prereqs (run once on the LOGIN node):
#   bash dataset_rebuilding/setup_flux_cineca.sh        # diffusers + FLUX weights
#   (and produce the cleaned captions:
#    python dataset_rebuilding/check_captions.py ... --write_clean $WORK/iab_captions_detailed_clean)
#
# Submit:  sbatch dataset_rebuilding/slurm_flux.sh
#
# This is the PILOT config: --max_per_class 100 (~1000 images). Remove that flag
# to generate all 2000/class. Resumable: existing PNGs are skipped.
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=flux_recap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1            # FLUX (~12B) fits on one A100 64GB in bf16
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail

module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

export HF_HOME="$WORK/hf_cache"
export TRANSFORMERS_OFFLINE=1        # compute nodes have no internet
export HF_HUB_OFFLINE=1

REPO="$WORK/hyp_fine_tuning/hyperbolic_CLIP"
CAPS="$WORK/iab_captions_detailed_clean"
DATA="$WORK/iab_dataset"
OUT="$WORK/iab_recap_dataset"
MODEL="black-forest-labs/FLUX.1-schnell"   # what IAB used; ungated, fast (4 steps)

cd "$REPO"
nvidia-smi || true

python dataset_rebuilding/generate_flux_fakes.py \
    --captions_dir   "$CAPS" \
    --dataset_path   "$DATA" \
    --out_root       "$OUT" \
    --model          "$MODEL" \
    --steps          4 \
    --guidance       0 \
    --height         1024 \
    --width          1024 \
    --max_seq_len    256 \
    --max_per_class  100

echo "FLUX fakes under: $OUT/FLUX/"
