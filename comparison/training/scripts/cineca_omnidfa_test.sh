#!/bin/bash
# IAB evaluation; IABench uses test_iabench.sh CHECKPOINT.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iab_omnidfa_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export HF_HUB_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$FAST/datasets/iab_dataset}"
CKPT="${1:-${CKPT:-}}"
[[ -n "$CKPT" && -f "$CKPT" ]] || { echo 'Pass an OmniDFA checkpoint.' >&2; exit 1; }
cd "$REPO"
PYDEPS="${IAB_PYDEPS:-$HOME/iab_pydeps}"
export PYTHONPATH="$REPO:$PYDEPS:${PYTHONPATH:-}"
python -m comparison.training.test \
  --config comparison/training/config/model/omnidfa.yaml \
  --resume_checkpoint "$CKPT" --root_dir "$DATA" \
  --batch_size "${BATCH_SIZE:-32}" --num_workers "${NUM_WORKERS:-8}" \
  --level_start 0 --level_end 7 \
  --log_dir "${LOGDIR:-$WORK/outputs/omnidfa_iab_${SLURM_JOB_ID}}"
