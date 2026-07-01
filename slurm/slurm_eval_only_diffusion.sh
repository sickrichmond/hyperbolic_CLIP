#!/bin/bash
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_only_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

# ── Configuration ────────────────────────────────────────────────────────────
# Positional arguments (pass them explicitly at launch so you can never silently
# eval the wrong images / split):
#   $1 = dataset root   (default: original IAB dataset)
#   $2 = split          (default: val)   — all | val | train
# This is the "only diffusion" eval, so the generator set is FIXED to the 4
# diffusion models (no real). CKPT / CAPS / VAL_FRAC stay env-overridable.
#
#   # OLD diffusion checkpoint on the NEW regenerated fakes (recommended:
#   # split=all → every new fake evaluated once, no celebahq val-imbalance):
#   sbatch slurm/slurm_eval_only_diffusion.sh $WORK/hyp_fine_tuning/iab_recap_dataset_v2 all
#
#   # original behaviour (old IAB dataset, val split):
#   sbatch slurm/slurm_eval_only_diffusion.sh
DATA="${1:-$WORK/hyp_fine_tuning/iab_dataset}"
SPLIT="${2:-val}"
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_diffusion.pt}
CAPS=${CAPS:-$WORK/hyp_fine_tuning/iab_captions}
VAL_FRAC=${VAL_FRAC:-0.2}
GENERATORS="SD3 SD3_5 SDXL FLUX"

echo "=== eval_only_diffusion config ==="
echo "  DATASET:    $DATA"
echo "  CHECKPOINT: $CKPT"
echo "  CAPTIONS:   $CAPS"
echo "  GENERATORS: $GENERATORS"
echo "  SPLIT:      $SPLIT (val_frac=$VAL_FRAC)"
echo "=================================="

python -m tests.eval_attribution \
    --checkpoint   "$CKPT" \
    --dataset_path "$DATA" \
    --captions_dir "$CAPS" \
    --generators   $GENERATORS \
    --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split        "$SPLIT" \
    --val_frac     "$VAL_FRAC" \
    --batch_size   256 \
    --num_workers  4
