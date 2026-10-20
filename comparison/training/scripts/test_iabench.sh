#!/bin/bash
# Evaluate one IABench baseline checkpoint on the exact manifest test rows, levels 0..6.
# Submit: sbatch comparison/training/scripts/test_iabench.sh /path/to/ckpt_best.pth
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iabench_baseline_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
DATA="${DATA:-$SCRATCH/datasets/IABench_images}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_vitl14.splits.json}"
CKPT="${1:-${CKPT:-}}"
LOGDIR="${LOGDIR:-$WORK/outputs/baselines_iabench_${SLURM_JOB_ID}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
cd "$REPO"
PYDEPS="${IAB_PYDEPS:-$HOME/iab_pydeps}"
export PYTHONPATH="$REPO:$PYDEPS:${PYTHONPATH:-}"
[[ -n "$CKPT" && -f "$CKPT" ]] || {
  echo "Pass a checkpoint: sbatch $0 /path/to/ckpt_best.pth (or set CKPT)." >&2
  exit 1
}
[[ -f "$SPLIT_MANIFEST" ]] || { echo "Missing manifest: $SPLIT_MANIFEST" >&2; exit 1; }
echo "Checkpoint: $CKPT | Dataset: $DATA | Manifest: $SPLIT_MANIFEST"
echo "Results: $LOGDIR | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
python -m comparison.training.test \
  --dataset iabench --resume_checkpoint "$CKPT" \
  --split_manifest "$SPLIT_MANIFEST" --root_dir "$DATA" \
  --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
  --level_start 0 --level_end 7 --log_dir "$LOGDIR"
echo "Done. Results in $LOGDIR/test_results_degraded_*.txt"
