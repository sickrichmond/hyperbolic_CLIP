#!/bin/bash
# ============================================================================
# CINECA Leonardo — 22-CLASS eval of the trained LiDA attributor.
# Uses the repo's standard baseline-style evaluation entrypoint:
#   python -m comparison.training.test --config model/lida.yaml --resume_checkpoint ...
#
# CKPT and LOGDIR are overridable. If not set, it auto-picks the most recent
# LiDA training run and its ckpt_best.pth.
# Submit:  sbatch slurm/slurm_eval_22cls.sh
# ============================================================================
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_22cls_lida
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aspis2000@virgilio.it

set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate
export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_giuseppe
DATA=$FAST/datasets/iab_dataset
LIDA_LOGDIR="$REPO/comparison/training/logs/default_split/lida"
LIDA_RUN_DIR=$(find "$LIDA_LOGDIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
CKPT=${CKPT:-$(find "$LIDA_RUN_DIR" -path '*/ckpt_best.pth' -print -quit 2>/dev/null || true)}
LOGDIR=${LOGDIR:-${LIDA_RUN_DIR:-$LIDA_LOGDIR}}

mkdir -p "$LOGDIR"
cd "$REPO"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found. Tried: $CKPT"; exit 1
fi

echo "Using checkpoint: $CKPT"
echo "Using log dir: $LOGDIR"

CUDA_VISIBLE_DEVICES=0 python -m comparison.training.test \
    --config comparison/training/config/model/lida.yaml \
    --resume_checkpoint "$CKPT" \
    --root_dir "$DATA" \
    --batch_size 32 \
    --num_workers 8 \
    --level_start 0 \
    --level_end 7 \
    --log_dir "$LOGDIR"

echo "Done. Results under $LOGDIR/default_split/lida/test_*/"
