#!/usr/bin/env bash
# Usage: sbatch slurm_eval_iab_styled_increfa.sh CHECKPOINT [DATASET [OUTPUT_DIR]]
# Set EVAL_DATASET=omnifake for clean OmniFake evaluation.
#SBATCH --job-name=eval_iab_styled_increfa
#SBATCH --account=EUHPC_D33_218
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
INCREFA_REPO="${INCREFA_REPO:-$WORK/IncreFA}"
CKPT="${1:-${CKPT:?Pass the static IAB IncreFA checkpoint as the first argument}}"
EVAL_ARGS=()
case "${EVAL_DATASET:-styled}" in
    styled)
        DATA="${2:-${DATA:-$FAST/datasets/iab_styled_eval}}"
        TEST_DIR="$DATA"
        RESULT_PREFIX=increfa_iab_styled_jpeg99
        CONDITION="JPEG99 only"
        ;;
    omnifake)
        DATA="${2:-${DATA:-$SCRATCH/datasets/omnifake_iab_test}}"
        TEST_DIR="$DATA/val"
        RESULT_PREFIX=increfa_omnifake
        CONDITION="OmniFake test split"
        EVAL_ARGS+=(--omnifake)
        ;;
    *) echo "EVAL_DATASET must be styled or omnifake" >&2; exit 2 ;;
esac
LOGDIR="${3:-${LOGDIR:-$WORK/outputs/${RESULT_PREFIX}_${SLURM_JOB_ID}}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

test -r "$CKPT"
test -r "$INCREFA_REPO/train.py"
test -d "$TEST_DIR"
cd "$INCREFA_REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
echo "Checkpoint: $CKPT | $CONDITION | Results: $LOGDIR"
srun "$WORK/hyp_fine_tuning/bin/python" -u -m comparison.training.test_increfa_styled \
    --increfa_dir "$INCREFA_REPO" --checkpoint "$CKPT" \
    --root_dir "$DATA" --log_dir "$LOGDIR" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" "${EVAL_ARGS[@]}"
