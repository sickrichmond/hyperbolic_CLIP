#!/bin/bash
# ============================================================================
# CINECA Leonardo — regenerate fakes for ONE diffusion generator from the new
# captions. Launch one job per generator (each gets its own GPU) to run them in
# parallel:
#
#   sbatch dataset_rebuilding/slurm_gen.sh SD3
#   sbatch dataset_rebuilding/slurm_gen.sh SD3_5
#   sbatch dataset_rebuilding/slurm_gen.sh SDXL
#   sbatch dataset_rebuilding/slurm_gen.sh FLUX
#
# Prereqs (run once on the LOGIN node):
#   bash dataset_rebuilding/setup_diffusion_cineca.sh     # diffusers + weights
#   (SD3 / SD3.5 are GATED — accept their licenses on HF first; see that script)
#   and the cleaned captions in $WORK/iab_captions_detailed_clean.
#
# PILOT config: --max_per_class 100 (~1000 imgs). Remove it for all 2000/class.
# Resumable: existing PNGs are skipped.
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=recap_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1            # one A100 64GB holds SD3/SD3.5/SDXL/FLUX in bf16
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail

GEN="${1:?usage: sbatch slurm_gen.sh <FLUX|SD3|SD3_5|SDXL>}"

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

cd "$REPO"
echo "Generator: $GEN"
nvidia-smi || true

python dataset_rebuilding/generate_fakes.py \
    --generator      "$GEN" \
    --captions_dir   "$CAPS" \
    --dataset_path   "$DATA" \
    --out_root       "$OUT" \
    --height         1024 \
    --width          1024 \
    --max_per_class  100

echo "Fakes under: $OUT/$GEN/"
