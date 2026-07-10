#!/bin/bash
# ============================================================================
# CINECA Leonardo — Evaluate a trained DEFL attributor checkpoint.
# Tests degraded levels [level_start, level_end). Use 0..7 for all 7 levels.
#
# Submit:  sbatch comparison/training/scripts/cineca_defl_test.sh
# Pre-fetch torchvision ResNet-50 + openai-CLIP RN50x16 once on a login node
# (see cineca_defl_train.sh).
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_defl_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail

# ── Environment ─────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP
DATA=$WORK/hyp_fine_tuning/iab_dataset
cd $REPO
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Checkpoint: pass an explicit path as the first arg (sbatch <script> <ckpt>),
# else auto-pick the most recent run's ckpt_best.pth for this method.
CKPT="${1:-}"
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t comparison/training/logs/default_split/defl/*/ckpt_best.pth 2>/dev/null | head -1 || true)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: no checkpoint found. Pass one: sbatch $0 <path/to/ckpt_best.pth>" >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

CONFIG=comparison/training/config/model/defl.yaml

# ── STANDARD SPLIT, all 7 degraded levels ───────────────────────────────────
python -m comparison.training.test \
  --config "$CONFIG" \
  --resume_checkpoint "$CKPT" \
  --root_dir "$DATA" \
  --batch_size 8 \
  --num_workers 2 \
  --level_start 0 --level_end 7 \
  --log_dir comparison/training/logs_test

# ── SEMANTIC SPLIT: add --use_semantic_split --task_id {1,2,3} and point
#    CKPT to the matching semantic_split_<task_id> run.
