#!/bin/bash
# IAB: 18 independent (lambda_anchor, z, tau) runs, then clean/JPEG30/BLUR5 tests.
# Submit: sbatch slurm/slurm_sweep_iab_omnidfa.sh
# Results: $WORK/hyp_fine_tuning/checkpoints/sweep/iab_omnidfa_ARRAY_JOB_ID
# Collect: python -m comparison.training.scripts.collect_iab_anchor_sweep --dir RESULTS

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iab_omnidfa_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=3-00:00:00
#SBATCH --array=0-17%4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-7])$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be between 0 and 17" >&2
    exit 2
fi

REPO=${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}
SWEEP_DIR="$WORK/hyp_fine_tuning/checkpoints/sweep/iab_omnidfa_${SLURM_ARRAY_JOB_ID}"
printf -v TASK_NAME 'task_%02d' "$SLURM_ARRAY_TASK_ID"
TASK_DIR="$SWEEP_DIR/$TASK_NAME"
mkdir -p "$SWEEP_DIR"
# Refuse retries in place: old metrics must never be paired with a new checkpoint.
mkdir "$TASK_DIR"
trap 'code=$?; if (( code != 0 )); then echo "failed (exit $code)" > "$TASK_DIR/status.txt"; fi' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
echo running > "$TASK_DIR/status.txt"

WEIGHTS=(0.2 0.5)
MARGINS=(2 3 4)
TARGETS=(2 4 6)
LAMBDA_ANCHOR=${WEIGHTS[SLURM_ARRAY_TASK_ID / 9]}
GEODESIC_MARGIN=${MARGINS[SLURM_ARRAY_TASK_ID / 3 % 3]}
TARGET_NORM=${TARGETS[SLURM_ARRAY_TASK_ID % 3]}
AUGMENT=1
AUG_POLICY=omnidfa
ANCHOR_INIT=random
OPTIMIZER=adamw
NUM_EPOCHS=30
OUT="$TASK_DIR"
DIAG_PLOT_DIR="$TASK_DIR/diagnostics"
# An inherited CKPT would let different array tasks overwrite the same model.
unset CKPT
echo "$TASK_NAME: lambda_anchor=$LAMBDA_ANCHOR z=$GEODESIC_MARGIN tau=$TARGET_NORM"
source "$REPO/slurm/slurm_train_22cls_final_loss.sh" > "$TASK_DIR/train.log" 2>&1

for level in 0 4 6; do
    python -m comparison.training.test_hypclip \
        --checkpoint "$CKPT" \
        --root_dir "$DATA" \
        --batch_size 64 \
        --num_workers 8 \
        --level_start "$level" \
        --level_end "$((level + 1))" \
        --pre_resize 0 \
        --log_dir "$TASK_DIR/evaluation" > "$TASK_DIR/eval_${level}.log" 2>&1
done
echo complete > "$TASK_DIR/status.txt"
echo "Done: $TASK_DIR"
