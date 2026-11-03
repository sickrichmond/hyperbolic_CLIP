#!/bin/bash
# Closed-set IAB OmniDFA, excluding dalle3; resume keeps the original run.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iab_omnidfa_train_22
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
export TORCH_HOME="$WORK/hyp_fine_tuning/torch_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$FAST/datasets/iab_dataset}"
LOGDIR="${LOGDIR:-$REPO/comparison/training/logs}"
cd "$REPO"
PYDEPS="${IAB_PYDEPS:-$HOME/iab_pydeps}"
export PYTHONPATH="$REPO:$PYDEPS:${PYTHONPATH:-}"
extra_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Missing resume checkpoint: $RESUME_CHECKPOINT" >&2; exit 1; }
  extra_args+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi
python -m comparison.training.train \
  --config comparison/training/config/model/omnidfa.yaml \
  --root_dir "$DATA" --n_epoch 20 -n 2000 --batch_size "${BATCH_SIZE:-144}" \
  --num_workers "${NUM_WORKERS:-8}" --log_dir "$LOGDIR" "${extra_args[@]}"
