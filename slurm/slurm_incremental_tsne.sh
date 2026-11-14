#!/usr/bin/env bash
# Resource requests and dependencies are set by submit_incremental_tsne.sh.
set -euo pipefail

stage=${1:?Expected preflight, pilot, extract, fit, or render}
repo=${REPO:-"${SLURM_SUBMIT_DIR:-$PWD}"}
work_root=${TSNE_WORK_ROOT:-/leonardo_work/EUHPC_D35_189/hyp_fine_tuning}
: "${OUT:?Set OUT to the run directory}"
module load python/3.11.7
if [[ $stage == pilot || $stage == extract ]]; then
    module load cuda/12.6
fi
export HF_HOME="${TSNE_HF_HOME:-$work_root/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="${TORCH_HOME:-$work_root/torch_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/incremental_tsne_mpl_${SLURM_JOB_ID:-$$}"
cd "$repo"

python=("${TSNE_PYTHON:-$work_root/bin/python}" -u -m explanation.incremental_tsne)
options=(--output-dir "$OUT" --batch-size "${BATCH_SIZE:-64}" --num-workers "${NUM_WORKERS:-8}")
[[ -z ${IGOR_ROOT:-} ]] || options+=(--igor-root "$IGOR_ROOT")
[[ -z ${FAST_ROOT:-} ]] || options+=(--fast-root "$FAST_ROOT")
[[ -z ${DATA:-} ]] || options+=(--dataset-root "$DATA")
[[ -z ${MANIFEST:-} ]] || options+=(--manifest "$MANIFEST")
[[ -z ${CAPTIONS:-} ]] || options+=(--captions-dir "$CAPTIONS")

case "$stage" in
    pilot)
        read -ra smoke_tasks <<< "${PILOT_TASKS:-0 1 16 17 32 33}"
        for task in "${smoke_tasks[@]}"; do
            echo "Smoke testing array task $task"
            for action in extract fit render; do
                "${python[@]}" "$action" "${options[@]}" --output-dir "$OUT/smoke" \
                    --task-id "$task" --max-per-class 17
            done
        done
        benchmark=${BENCHMARK_TASK:-15}
        echo "Benchmarking full array task $benchmark"
        "${python[@]}" extract "${options[@]}" --task-id "$benchmark"
        "${python[@]}" fit "${options[@]}" --task-id "$benchmark"
        ;;
    extract)
        "${python[@]}" "$stage" "${options[@]}" --task-id "${SLURM_ARRAY_TASK_ID:?}"
        ;;
    fit)
        first=$(( ${SLURM_ARRAY_TASK_ID:?} * 8 ))
        for (( task=first; task<first+8; task++ )); do
            "${python[@]}" fit "${options[@]}" --task-id "$task"
        done
        ;;
    preflight|render)
        read -ra methods <<< "${TSNE_METHODS:-}"
        [[ ${#methods[@]} == 0 ]] || options+=(--methods "${methods[@]}")
        "${python[@]}" "$stage" "${options[@]}"
        ;;
    *)
        echo "Unknown stage: $stage" >&2
        exit 2
        ;;
esac
