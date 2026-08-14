#!/bin/bash
set -euo pipefail

MODEL=${1:?usage: run_array_task.sh MODEL}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MANIFEST=$SCRIPT_DIR/manifest.tsv
DATASET_LIST=$SCRIPT_DIR/datasets.txt

TASK_ID=${SLURM_ARRAY_TASK_ID:-${AUG_TEST_TASK_ID:-}}
if [[ ! $TASK_ID =~ ^[0-3]$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (or AUG_TEST_TASK_ID) must be 0, 1, 2, or 3" >&2
    exit 2
fi

DATASET_NAME=$(sed -n "$((TASK_ID + 1))p" "$DATASET_LIST")
if [[ -z $DATASET_NAME ]]; then
    echo "ERROR: no dataset mapped to array index $TASK_ID" >&2
    exit 2
fi

ROW=$(awk -F '\t' -v model="$MODEL" '$1 == model { print; exit }' "$MANIFEST")
if [[ -z $ROW ]]; then
    echo "ERROR: model '$MODEL' is absent from $MANIFEST" >&2
    exit 2
fi
IFS=$'\t' read -r _ CONFIG_REL DEFAULT_CHECKPOINT DEFAULT_BATCH DEFAULT_WORKERS <<< "$ROW"

DATA_ROOT=${AUG_TEST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D35_189/datasets}
OUTPUT_ROOT=${AUG_TEST_OUTPUT_ROOT:-/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/aug_test_results}
CHECKPOINT=${AUG_TEST_CHECKPOINT:-$DEFAULT_CHECKPOINT}
BATCH_SIZE=${AUG_TEST_BATCH_SIZE:-$DEFAULT_BATCH}
NUM_WORKERS=${AUG_TEST_NUM_WORKERS:-$DEFAULT_WORKERS}
DATASET_ROOT=$DATA_ROOT/$DATASET_NAME
OUTPUT_DIR=$OUTPUT_ROOT/$MODEL/$DATASET_NAME
CONFIG=$REPO_ROOT/$CONFIG_REL

for required in "$CONFIG" "$CHECKPOINT" "$DATASET_ROOT"; do
    if [[ ! -e $required ]]; then
        echo "ERROR: required path does not exist: $required" >&2
        exit 2
    fi
done

module load python/3.11.7
module load cuda/12.6
source /leonardo_work/EUHPC_D35_189/hyp_fine_tuning/bin/activate

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export IAB_EXCLUDE_GENERATORS=dalle3
export HF_HOME=${HF_HOME:-/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/hf_cache}
export TORCH_HOME=${TORCH_HOME:-/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/torch_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ $MODEL == dna ]]; then
    DNA_PYDEPS=${IAB_PYDEPS:-$HOME/iab_pydeps}
    if [[ -d $DNA_PYDEPS ]]; then
        export PYTHONPATH="$DNA_PYDEPS:$PYTHONPATH"
    fi
fi

ARGS=(
    --model "$MODEL"
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --dataset-root "$DATASET_ROOT"
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
)

if [[ $MODEL == ucf ]]; then
    UCF_PRETRAINED=${AUG_TEST_UCF_PRETRAINED:-/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/hyperbolic_CLIP_riccardo/pretrained/xception-b5690688.pth}
    ARGS+=(--ucf-pretrained "$UCF_PRETRAINED")
fi
if [[ -n ${AUG_TEST_MAX_IMAGES:-} ]]; then
    ARGS+=(--max-images "$AUG_TEST_MAX_IMAGES")
fi
if [[ ${AUG_TEST_VALIDATE_ONLY:-0} == 1 ]]; then
    ARGS+=(--validate-only)
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
echo "model=$MODEL dataset=$DATASET_NAME checkpoint=$CHECKPOINT"
echo "batch_size=$BATCH_SIZE workers=$NUM_WORKERS output=$OUTPUT_DIR"
python -m comparison.training.eval_aug_splits "${ARGS[@]}"
