#!/bin/bash
# Ours: sbatch slurm/slurm_train_hierarchy_holdout.sh ours
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=hierarchy_holdout_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com
set -euo pipefail
METHOD="${1:-ours}"
case "$METHOD" in
  ours) ;;
  *) echo "Only ours is supported in the hierarchy holdout experiment." >&2; exit 2 ;;
esac
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TORCH_HOME="$WORK/hyp_fine_tuning/torch_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
EXPERIMENT="${EXPERIMENT:-$WORK/hyp_fine_tuning/experiments/hybrid_hierarchy_20seen_v1/experiment.json}"
LOGDIR="${LOGDIR:-$(dirname "$EXPERIMENT")/runs/${METHOD}_${SLURM_JOB_ID}}"
export PYTHONPATH="$REPO:${IAB_PYDEPS:-$HOME/iab_pydeps}:${PYTHONPATH:-}"
ROOT_FLAGS=()
[[ -z "${IABENCH_DATA:-}" ]] || ROOT_FLAGS+=(--iabench_root "$IABENCH_DATA")
[[ -z "${IAB_DATA:-}" ]] || ROOT_FLAGS+=(--iab_root "$IAB_DATA")
cd "$REPO"
[[ -f "$EXPERIMENT" ]] || { echo "Prepare the experiment first: $EXPERIMENT" >&2; exit 1; }
echo "Method: $METHOD | Experiment: $EXPERIMENT | GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output: $LOGDIR"
python -m comparison.training.hierarchy_experiment train \
  --experiment "$EXPERIMENT" --method "$METHOD" --output_dir "$LOGDIR" \
  --num_workers "${NUM_WORKERS:-8}" "${ROOT_FLAGS[@]}"
