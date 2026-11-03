#!/bin/bash
# sbatch slurm/slurm_plot_hierarchy_pair.sh [SEEN_GENERATOR [UNSEEN_GENERATOR]]
# Defaults: stabilityai@stable-diffusion-2 vs stabilityai@stable-diffusion-2-1.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=hierarchy_pair
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
[[ $# -le 2 ]] || { echo "Expected at most two arguments: SEEN_GENERATOR UNSEEN_GENERATOR" >&2; exit 2; }
SEEN="${1:-stabilityai@stable-diffusion-2}"
UNSEEN="${2:-stabilityai@stable-diffusion-2-1}"
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
EXPERIMENT="${EXPERIMENT:-$WORK/hyp_fine_tuning/experiments/hybrid_hierarchy_20seen_v1/experiment.json}"
CKPT="${CKPT:-$(dirname "$EXPERIMENT")/runs/ours_62755694/ours_anchor0p5_z2_ep30.pt}"
LOGDIR="${LOGDIR:-$WORK/outputs/hierarchy_pair_${SLURM_JOB_ID}}"
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
echo "Seen: $SEEN | Unseen: $UNSEEN | Checkpoint: $CKPT"
echo "Clean frozen test rows | GPU: ${CUDA_VISIBLE_DEVICES:-unset} | Output: $LOGDIR"
python -m comparison.training.hierarchy_pair "$SEEN" "$UNSEEN" \
  --checkpoint "$CKPT" --experiment "$EXPERIMENT" --output_dir "$LOGDIR" \
  --batch_size "${BATCH_SIZE:-8}" --num_workers "${NUM_WORKERS:-8}" "${ROOT_FLAGS[@]}"
