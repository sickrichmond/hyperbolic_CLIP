#!/bin/bash
# ============================================================================
# CINECA Leonardo — per-class explanation gallery for AttributionCLIP.
#
# Picks one representative image PER CLASS (a real FLUX sample for FLUX, a real
# SD3 sample for SD3, …) and, for each, runs all three explanation methods
# (AGCAM, Guided and Chefer by default), laying them out next to the original:
#
#     class │ Original │ AGCAM │ GUIDED │ CHEFER
#
# Single-image inference + backprop through the ViT-L/14 attention stack:
# lightweight, so 1 GPU / short walltime is plenty.
#
# Submit (positional args: DIM picks the checkpoint, SEMANTIC the content):
#   sbatch slurm/slurm_explain.sh 16            # d16, COCO (default semantic)
#   sbatch slurm/slurm_explain.sh 16 FFHQ       # d16, faces
#
# Override the rest on the CLI, e.g. a different sample / method set:
#   IMAGE_INDEX=3 sbatch slurm/slurm_explain.sh 16 bedroom
#   METHODS="chefer" sbatch slurm/slurm_explain.sh 16 FFHQ   # only Chefer
#   METHODS="agcam guided chefer" sbatch slurm/slurm_explain.sh 16
#
# For the Adebayo sanity check see slurm/slurm_sanity.sh (separate job).
#
# NOTE: run as a module (python -m explanation.explain_gallery), NOT as a file
# path — the package imports (models., losses., explanation., data., geometry.)
# need the repo root on sys.path, which only the -m form provides.
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=explain_attribution
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hf_cache          # avoid filling home quota
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo

# ── Parameters ────────────────────────────────────────────────────────────────
DIM=${1:-16}                  # arg 1: embedding dim; selects the checkpoint below.
SEMANTIC=${2:-${SEMANTIC:-COCO}}  # arg 2 (or env): one semantic, shown for every class
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_all_no_dalle_d${DIM}.pt}
DATA=${DATA:-$WORK/iab_dataset}
IMAGE_INDEX=${IMAGE_INDEX:-0} # which sample per class (sorted order)
METHODS=${METHODS:-agcam guided chefer}  # space-separated: agcam | guided | chefer
SCORE_MODE=${SCORE_MODE:-margin}
OUT=${OUT:-$WORK/outputs/gallery/d${DIM}_${SEMANTIC}}

mkdir -p $OUT

# ── Run ───────────────────────────────────────────────────────────────────────
python -m explanation.explain_gallery \
    --checkpoint    $CKPT \
    --dataset_path  $DATA \
    --semantic      $SEMANTIC \
    --image_index   $IMAGE_INDEX \
    --methods       $METHODS \
    --score_mode    $SCORE_MODE \
    --output_dir    $OUT

echo "Gallery outputs → $OUT"
