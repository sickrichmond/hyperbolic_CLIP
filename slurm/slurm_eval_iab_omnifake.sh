#!/bin/bash
# CKPT=/path/to/iab.pt sbatch slurm/slurm_eval_iab_omnifake.sh
# Prepare DATA with python -m scripts.download_omnifake_test --output "$DATA".
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_iab_omnifake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
CKPT="${CKPT:?Set CKPT to the IAB checkpoint to evaluate}"
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$SCRATCH/datasets/omnifake_iab_test}"
LOGDIR="${LOGDIR:-$WORK/outputs/hypclip_omnifake_${SLURM_JOB_ID}}"

cd "$REPO"
if [ ! -f "$CKPT" ] || [ ! -d "$DATA/val" ]; then
    echo "ERROR: missing checkpoint $CKPT or test directory $DATA/val" >&2
    exit 1
fi

module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1

python -m comparison.training.test_hypclip_omnifake \
    --checkpoint "$CKPT" \
    --root_dir "$DATA" \
    --log_dir "$LOGDIR" \
    --num_workers "${NUM_WORKERS:-8}"
