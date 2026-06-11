#!/bin/bash
# ============================================================================
# CINECA Leonardo — AGCAM / Guided explainability for AttributionCLIP.
#
# Single-image inference + backprop through the ViT-L/14 attention stack:
# lightweight, so 1 GPU / short walltime is plenty.
#
# Submit (DIM picks the checkpoint, mirrors slurm_cineca_all.sh):
#   sbatch slurm/slurm_explain.sh 16
#
# Override any path on the CLI, e.g.:
#   IMAGE=$WORK/iab_dataset/FLUX/COCO/xxx.jpg OUT=$WORK/outputs/foo \
#       sbatch slurm/slurm_explain.sh 16
#
# NOTE: run as a module (python -m explanation.explain_image), NOT as a file
# path — the package imports (models., losses., explanation., geometry.) need
# the repo root on sys.path, which only the -m form provides.
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
DIM=${1:-16}               # embedding dim; selects the checkpoint name below.
CKPT=${CKPT:-$WORK/checkpoints/attribution_all_no_dalle_d${DIM}.pt}

# Image to explain — MUST already exist on CINECA. Point at a dataset sample or
# rsync one over from local. Override with IMAGE=... on the sbatch line.
IMAGE=${IMAGE:-$WORK/iab_dataset/FLUX/COCO/COCO-new_p0_i0.png}

OUT=${OUT:-$WORK/outputs/explanation/d${DIM}}
METHOD=${METHOD:-agcam}    # agcam | guided

mkdir -p $OUT

# ── Run ───────────────────────────────────────────────────────────────────────
python -m explanation.explain_image \
    --image       $IMAGE \
    --checkpoint  $CKPT \
    --method      $METHOD \
    --score_mode  margin \
    --output_dir  $OUT \
    --all_classes

echo "Explanation outputs → $OUT"
