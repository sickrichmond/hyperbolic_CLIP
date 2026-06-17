#!/bin/bash
# ============================================================================
# CINECA Leonardo — per-class explanation gallery for AttributionCLIP.
#
# Picks one representative image PER CLASS (a real FLUX sample for FLUX, a real
# SD3 sample for SD3, …), explains each with its own class heatmap, and builds a
# side-by-side comparison grid. Single-image inference + backprop through the
# ViT-L/14 attention stack: lightweight, so 1 GPU / short walltime is plenty.
#
# Submit (positional args: DIM picks the checkpoint, SEMANTIC the content):
#   sbatch slurm/slurm_explain.sh 16            # d16, COCO (default semantic)
#   sbatch slurm/slurm_explain.sh 16 FFHQ       # d16, faces
#
# Override the rest on the CLI, e.g. a different sample / method:
#   IMAGE_INDEX=3 METHOD=guided sbatch slurm/slurm_explain.sh 16 bedroom
#
# NOTE: run as a module (python -m explanation.explain_gallery), NOT as a file
# path — the package imports (models., losses., explanation., data., geometry.)
# need the repo root on sys.path, which only the -m form provides.
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=explain_attribution
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
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

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

# ── Parameters ────────────────────────────────────────────────────────────────
DIM=${1:-16}                  # arg 1: embedding dim; selects the checkpoint below.
SEMANTIC=${2:-${SEMANTIC:-COCO}}  # arg 2 (or env): one semantic, shown for every class
CKPT=${CKPT:-$WORK/checkpoints/attribution_all_no_dalle_d${DIM}.pt}
DATA=${DATA:-$WORK/iab_dataset}
IMAGE_INDEX=${IMAGE_INDEX:-0} # which sample per class (sorted order)
OUT=${OUT:-$WORK/outputs/gallery/d${DIM}_${SEMANTIC}}
METHOD=${METHOD:-agcam}       # agcam | guided

mkdir -p $OUT

# ── Run ───────────────────────────────────────────────────────────────────────
python -m explanation.explain_gallery \
    --checkpoint    $CKPT \
    --dataset_path  $DATA \
    --semantic      $SEMANTIC \
    --image_index   $IMAGE_INDEX \
    --method        $METHOD \
    --score_mode    margin \
    --output_dir    $OUT

echo "Gallery outputs → $OUT"
