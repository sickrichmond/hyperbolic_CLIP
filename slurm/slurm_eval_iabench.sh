#!/bin/bash
# CINECA Leonardo — IABench test evaluation through the comparison harness.
# Levels 0..6 match ImageAttributionBench: clean, DS0.5/0.25, JPEG65/30, Blur3/5.
# Defaults to the augmented checkpoint; CKPT and LOGDIR can select another run.
# SPLIT_MANIFEST must match the selected checkpoint's digest.
# Submit: sbatch slurm/slurm_eval_iabench.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=eval_iabench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -e
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO="$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo"
DATA=/leonardo_scratch/large/userexternal/imaljkov/datasets/IABench/data
CKPT="${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_aug_vitl14.pt}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_vitl14.splits.json}"
LOGDIR="${LOGDIR:-$WORK/outputs/hypclip_iabench_${SLURM_JOB_ID}}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "$REPO"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi
if [ ! -f "$SPLIT_MANIFEST" ]; then
    echo "ERROR: split manifest not found at $SPLIT_MANIFEST"
    exit 1
fi
mkdir -p "$LOGDIR"

python -m comparison.training.test_hypclip \
    --dataset iabench \
    --checkpoint "$CKPT" \
    --split_manifest "$SPLIT_MANIFEST" \
    --root_dir "$DATA" \
    --batch_size 64 \
    --num_workers "$NUM_WORKERS" \
    --level_start 0 \
    --level_end 7 \
    --log_dir "$LOGDIR"

echo "Done. Results in $LOGDIR/test_results_degraded_*.txt"
