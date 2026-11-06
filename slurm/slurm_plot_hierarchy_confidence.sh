#!/bin/bash
# All seen and held-out generators, clean test rows; no retraining.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=hierarchy_confidence
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com
set -euo pipefail
[[ $# -eq 0 ]] || { echo "This diagnostic evaluates all generators; supply the checkpoint through CKPT." >&2; exit 2; }
: "${CKPT:?Set CKPT to the selected hierarchy training checkpoint.}"
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
EXPERIMENT="${EXPERIMENT:-$WORK/hyp_fine_tuning/experiments/hybrid_hierarchy_20seen_v1/experiment.json}"
LOGDIR="${LOGDIR:-$WORK/outputs/hierarchy_confidence_${SLURM_JOB_ID}}"
[[ -f "$CKPT" && -f "$EXPERIMENT" ]] || { echo "Missing checkpoint or experiment: $CKPT | $EXPERIMENT" >&2; exit 1; }
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TORCH_HOME="$WORK/hyp_fine_tuning/torch_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO:${IAB_PYDEPS:-$HOME/iab_pydeps}:${PYTHONPATH:-}"
ROOT_FLAGS=()
[[ -z "${IABENCH_DATA:-}" ]] || ROOT_FLAGS+=(--iabench_root "$IABENCH_DATA")
[[ -z "${IAB_DATA:-}" ]] || ROOT_FLAGS+=(--iab_root "$IAB_DATA")
cd "$REPO"
echo "All seen/held-out clean test rows | Checkpoint: $CKPT | Output: $LOGDIR"
python -m comparison.training.hierarchy_confidence \
  --checkpoint "$CKPT" --experiment "$EXPERIMENT" --output_dir "$LOGDIR" \
  --batch_size "${BATCH_SIZE:-8}" --num_workers "${NUM_WORKERS:-8}" "${ROOT_FLAGS[@]}"
