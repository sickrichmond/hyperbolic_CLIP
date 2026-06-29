#!/bin/bash
# ============================================================================
# CINECA Leonardo — Evaluate a trained DCT-CNN attributor checkpoint.
# Tests degraded levels [level_start, level_end). Use 0..7 for all 7 levels.
#
# Submit:  sbatch comparison/training/scripts/cineca_dct_test.sh
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_dct_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
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
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP
DATA=$WORK/hyp_fine_tuning/iab_dataset
cd $REPO
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# ---- EDIT: checkpoint produced by training (ckpt_best.pth or ckpt_epoch_N.pth):
CKPT=comparison/training/logs/default_split/dct/<RUN_FOLDER>/ckpt_best.pth

CONFIG=comparison/training/config/model/dct.yaml

# ── STANDARD SPLIT, all 7 degraded levels ───────────────────────────────────
python -m comparison.training.test \
  --config "$CONFIG" \
  --resume_checkpoint "$CKPT" \
  --root_dir "$DATA" \
  --batch_size 32 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --level_start 0 --level_end 7 \
  --log_dir comparison/training/logs_test

# ── SEMANTIC SPLIT: add --use_semantic_split --task_id {1,2,3} and point
#    CKPT to the matching semantic_split_<task_id> run.
