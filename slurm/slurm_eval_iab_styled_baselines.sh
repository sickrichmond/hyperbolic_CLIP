#!/bin/bash
# One task per IAB baseline, every style, native preprocessing/prediction rules.
# Submit: sbatch slurm/slurm_eval_iab_styled_baselines.sh /path/to/checkpoints.json
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_iab_styled_baselines
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TORCH_HOME="$WORK/hyp_fine_tuning/torch_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3

REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$FAST/datasets/iab_styled_eval}"
CHECKPOINTS="${1:-${CHECKPOINTS:-}}"
LOGDIR="${LOGDIR:-$WORK/outputs/baselines_iab_styled_${SLURM_ARRAY_JOB_ID}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
METHODS=(resnet50 dct hifi_net defl dna repmix patch ucf lida defake pose)
TASK="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
[[ "$TASK" =~ ^(0|[1-9]|10)$ ]] || { echo "Invalid array index: $TASK" >&2; exit 1; }
METHOD="${METHODS[$TASK]}"
[[ -n "$CHECKPOINTS" && -f "$CHECKPOINTS" ]] || {
    echo "Pass a checkpoint map: sbatch $0 /absolute/path/to/checkpoints.json" >&2; exit 1;
}
[[ -d "$DATA" ]] || { echo "Styled dataset not found: $DATA" >&2; exit 1; }
# Resolve before cd so relative map paths refer to the submission directory.
CHECKPOINTS="$(realpath "$CHECKPOINTS")"
cd "$REPO"
PYDEPS="${IAB_PYDEPS:-$HOME/iab_pydeps}"
export PYTHONPATH="$REPO:$PYDEPS:${PYTHONPATH:-}"
echo "Baseline: $METHOD | Dataset: $DATA | Checkpoints: $CHECKPOINTS"
echo "Results: $LOGDIR/$METHOD | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
python -m comparison.training.test_baselines_styled \
    --method "$METHOD" --checkpoints "$CHECKPOINTS" \
    --root_dir "$DATA" --log_dir "$LOGDIR/$METHOD" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
