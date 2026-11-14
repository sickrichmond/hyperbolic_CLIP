#!/bin/bash
# Usage: sbatch slurm/slurm_eval_iab_family_baselines.sh [--output-dir DIR] [BASELINE_MAP]
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=eval_iab_family_baselines
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
OUTPUT_DIR=
if [[ "${1:-}" == --output-dir ]]; then
    [[ -n "${2:-}" ]] || { echo "--output-dir requires a directory" >&2; exit 2; }
    OUTPUT_DIR="$2"
    shift 2
fi
if (( $# > 1 )); then
    echo "Usage: sbatch $0 [--output-dir DIR] [BASELINE_MAP]" >&2
    exit 2
fi
TASK="${SLURM_ARRAY_TASK_ID:-}"
[[ "$TASK" =~ ^([0-9]|10|11)$ ]] || { echo "Expected an array index from 0 to 11" >&2; exit 2; }
METHODS=(resnet50 dct hifi_net defl dna repmix patch ucf lida defake pose increfa)
METHOD="${METHODS[$TASK]}"

REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$SCRATCH/datasets/IABench_images}"
MANIFEST="${SPLIT_MANIFEST:-$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_vitl14.splits.json}"
LOGDIR="${OUTPUT_DIR:-${LOGDIR:-$WORK/outputs/iab_family_baselines_${SLURM_ARRAY_JOB_ID}}}"
[[ -d "$DATA" && -f "$MANIFEST" ]] || {
    echo "Expected IABench data at $DATA and a manifest at $MANIFEST" >&2
    exit 2
}
family_repo="$(realpath -- "$REPO")"
EVAL_ARGS=(--method "$METHOD" --root-dir "$(realpath -- "$DATA")"
           --split-manifest "$(realpath -- "$MANIFEST")" --output-dir "$(realpath -m -- "$LOGDIR")"
           --batch-size "${BATCH_SIZE:-8}" --num-workers "${NUM_WORKERS:-8}")

if [[ "$METHOD" == lida ]]; then
    # The query DATA override must not become Igor's IAB gallery root.
    IGOR_ENV="${LIDA_ENV:-/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/hyperbolic_CLIP_igor/LIDA/slurm/env.sh}"
    unset DATA
    set +u
    source "$IGOR_ENV"
    set -u
    export PYTHONPATH="$CODE:$family_repo:${PYTHONPATH:-}"
    EVAL_ARGS+=(--checkpoint "$RUNS/lida_full_iab/T0.pt" --gallery-root "$DATA")
else
    module load python/3.11.7
    module load cuda/12.6
    source "$WORK/hyp_fine_tuning/bin/activate"
    export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
    export TORCH_HOME="$WORK/hyp_fine_tuning/torch_cache"
    export IAB_EXCLUDE_GENERATORS=dalle3
    export PYTHONPATH="$family_repo:${IAB_PYDEPS:-$HOME/iab_pydeps}:${PYTHONPATH:-}"
    if [[ "$METHOD" == increfa ]]; then
        NATIVE_REPO="${INCREFA_REPO:-$WORK/IncreFA}"
        CKPT="${INCREFA_CKPT:-$NATIVE_REPO/outputs/iab22_static_vitb16_replica2/latest.pt}"
        EVAL_ARGS+=(--checkpoint "$(realpath -- "$CKPT")" --increfa-dir "$(realpath -- "$NATIVE_REPO")")
    else
        MAP="${1:-$family_repo/comparison/training/config/iab_styled_final_checkpoints.json}"
        [[ -f "$MAP" ]] || { echo "Missing baseline checkpoint map: $MAP" >&2; exit 2; }
        EVAL_ARGS+=(--checkpoints "$(realpath -- "$MAP")")
    fi
fi
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$family_repo"
printf 'Family evaluation arguments: %q\n' "${EVAL_ARGS[@]}"
python -u -m comparison.training.iab_family evaluate "${EVAL_ARGS[@]}"
