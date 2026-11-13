#!/bin/bash
# Submit with either styled or omnifake as the sole argument.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_lida_retrieval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
case "${1:-}" in
    styled) query_root=/leonardo_scratch/fast/EUHPC_D35_189/datasets/iab_styled_eval ;;
    omnifake) query_root=/leonardo_scratch/large/userexternal/rtrebian/datasets/omnifake_iab_test ;;
    *) echo "Usage: sbatch $0 styled|omnifake" >&2; exit 2 ;;
esac

repo=${REPO:-/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/hyperbolic_CLIP_riccardo}
igor_env=/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/hyperbolic_CLIP_igor/LIDA/slurm/env.sh
source "$igor_env"
set -u
export PYTHONPATH="$CODE:$repo:${PYTHONPATH:-}"
cd "$repo"

python -m comparison.training.evaluate_lida_retrieval \
    --checkpoint "$RUNS/lida_full_iab/T0.pt" \
    --gallery-root "$DATA" --query-root "$query_root" --dataset "$1" \
    --output-dir "$WORK/outputs/lida_retrieval_${1}_${SLURM_JOB_ID}" \
    --num-workers "$SLURM_CPUS_PER_TASK"
