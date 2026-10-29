#!/bin/bash
# Evaluate every style under DATA against all saved IAB checkpoint anchors.
# Submit: CKPT=/path/to/checkpoint.pt sbatch slurm/slurm_eval_iab_styled.sh
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_iab_styled
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
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

REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$FAST/datasets/iab_styled_eval}"
CKPT="${CKPT:?Set CKPT to the IAB checkpoint to evaluate}"
LOGDIR="${LOGDIR:-$WORK/outputs/hypclip_iab_styled_${SLURM_JOB_ID}}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "$REPO"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi
if [ ! -d "$DATA" ]; then
    echo "ERROR: styled dataset not found at $DATA"
    exit 1
fi
echo "Dataset: $DATA | Checkpoint: $CKPT | Results: $LOGDIR"

python -m comparison.training.test_hypclip_styled \
    --checkpoint "$CKPT" \
    --root_dir "$DATA" \
    --log_dir "$LOGDIR" \
    --batch_size 64 \
    --num_workers "$NUM_WORKERS"

echo "Done. Results in $LOGDIR/summary.csv and $LOGDIR/STYLE/metrics.json"
