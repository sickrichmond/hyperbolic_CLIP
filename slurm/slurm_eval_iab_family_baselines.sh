#!/bin/bash
# Usage: sbatch slurm/slurm_eval_iab_family_baselines.sh [BASELINE_MAP]
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=eval_iab_family_baselines
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
if (( $# > 1 )); then
    echo "Usage: sbatch $0 [BASELINE_MAP]" >&2
    exit 2
fi
TASK="${SLURM_ARRAY_TASK_ID:-}"
[[ "$TASK" =~ ^([0-9]|10)$ ]] || { echo "Expected an array index from 0 to 10" >&2; exit 2; }
METHODS=(resnet50 dct hifi_net defl dna repmix patch ucf lida defake pose)
METHOD="${METHODS[$TASK]}"

REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$SCRATCH/datasets/IABench_images}"
MANIFEST="${SPLIT_MANIFEST:-$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_vitl14.splits.json}"
LOGDIR="${LOGDIR:-$WORK/outputs/iab_family_baselines_${SLURM_ARRAY_JOB_ID}}"
MAP="${1:-$REPO/comparison/training/config/iab_styled_final_checkpoints.json}"
[[ -f "$MAP" ]] || { echo "Missing baseline checkpoint map: $MAP" >&2; exit 2; }
[[ -d "$DATA" && -f "$MANIFEST" ]] || {
    echo "Expected IABench data at $DATA and a manifest at $MANIFEST" >&2
    exit 2
}
DATA="$(realpath "$DATA")"
MANIFEST="$(realpath "$MANIFEST")"
LOGDIR="$(realpath -m "$LOGDIR")"
MAP="$(realpath "$MAP")"

module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TORCH_HOME="$WORK/hyp_fine_tuning/torch_cache"
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3
export PYTHONPATH="$REPO:${IAB_PYDEPS:-$HOME/iab_pydeps}:${PYTHONPATH:-}"
cd "$REPO"
echo "Model: $METHOD | IABench: $DATA | Manifest: $MANIFEST | Results: $LOGDIR"
python -u -m comparison.training.iab_family evaluate --method "$METHOD" --checkpoints "$MAP" \
    --root-dir "$DATA" --split-manifest "$MANIFEST" --output-dir "$LOGDIR" \
    --batch-size "${BATCH_SIZE:-8}" --num-workers "${NUM_WORKERS:-8}"
