#!/bin/bash
#SBATCH --job-name=iab_rn50_test
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=iab_rn50_test_%j.out
#SBATCH --error=iab_rn50_test_%j.err

# ============================================================================
# Evaluate a trained ResNet-50 attributor checkpoint on ImageAttributionBench.
# Tests degraded levels [level_start, level_end). Use 0..7 for all 7 levels.
# ============================================================================

set -euo pipefail

# ---- EDIT THESE ------------------------------------------------------------
PROJECT_ROOT=/leonardo_work/EUHPC_D26_009B/hyp_fine_tuning/hyperbolic_CLIP
DATA=/leonardo_work/EUHPC_D26_009B/hyp_fine_tuning/iab_dataset
VENV=/leonardo_work/EUHPC_D26_009B/hyp_fine_tuning/bin/activate
# Path to the checkpoint produced by training (ckpt_best.pth or ckpt_epoch_N.pth):
CKPT=comparison/training/logs/default_split/resnet50/<RUN_FOLDER>/ckpt_best.pth
# ---------------------------------------------------------------------------

source "$VENV"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

CONFIG=comparison/training/config/model/resnet50.yaml

# ---- STANDARD SPLIT, all 7 degraded levels --------------------------------
srun python -m comparison.training.test \
  --config "$CONFIG" \
  --resume_checkpoint "$CKPT" \
  --root_dir "$DATA" \
  --batch_size 32 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --level_start 0 --level_end 7 \
  --log_dir comparison/training/logs_test

# ---- SEMANTIC SPLIT: add --use_semantic_split --task_id {1,2,3} and point
#      CKPT to the matching semantic_split_<task_id> run.
