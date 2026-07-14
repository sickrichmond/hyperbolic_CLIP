#!/bin/bash
# ============================================================================
# CINECA Leonardo — Adebayo sanity check for AttributionCLIP heatmaps.
#
# Runs the model-parameter randomization test (Adebayo et al., NeurIPS 2018):
# progressively randomizes the ViT weights from the output back to the input
# and measures how fast the heatmap decorrelates from the trained-model one.
# A faithful method PASSES (correlation → 0); an input-only method FAILS.
#
# Heavier than the gallery: one forward+backward per randomization stage
# (~27 stages for ViT-L/14), still on a single image → 1 GPU, short walltime.
#
# Submit (positional arg DIM picks the checkpoint, like slurm_explain.sh);
# IMAGE is required (the test runs on one image):
#   IMAGE=$FAST/datasets/iab_dataset/FLUX/FFHQ/00000.png sbatch slurm/slurm_sanity.sh 16
#
# Override method / class / test mode on the CLI:
#   METHOD=chefer TARGET=FLUX INDEPENDENT=1 \
#     IMAGE=/path/to/img.png sbatch slurm/slurm_sanity.sh 16
#
# NOTE: run as a module (python -m explanation.sanity_checks), NOT as a file
# path — the package imports need the repo root on sys.path.
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=sanity_attribution
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache   # avoid filling home quota
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo

# ── Parameters ────────────────────────────────────────────────────────────────
DIM=${1:-16}                  # arg 1: embedding dim; selects the checkpoint below.
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_all_no_dalle_d${DIM}.pt}
METHOD=${METHOD:-chefer}      # agcam | guided | chefer
SCORE_MODE=${SCORE_MODE:-margin}
OUT=${OUT:-$WORK/outputs/sanity/d${DIM}_${METHOD}}

if [ -z "$IMAGE" ]; then
    echo "ERROR: set IMAGE=/path/to/image.png (the sanity check runs on one image)." >&2
    exit 1
fi

# Optional: TARGET=<class name> to check a class other than the predicted one.
TARGET_ARG=""; [ -n "$TARGET" ] && TARGET_ARG="--target $TARGET"
# Optional: INDEPENDENT=1 randomizes one stage at a time (default: cascading).
INDEP_ARG="";  [ -n "$INDEPENDENT" ] && INDEP_ARG="--independent"

mkdir -p $OUT

# ── Run ───────────────────────────────────────────────────────────────────────
python -m explanation.sanity_checks \
    --image       $IMAGE \
    --checkpoint  $CKPT \
    --method      $METHOD \
    --score_mode  $SCORE_MODE \
    $TARGET_ARG \
    $INDEP_ARG \
    --output_dir  $OUT

echo "Sanity check outputs → $OUT"
