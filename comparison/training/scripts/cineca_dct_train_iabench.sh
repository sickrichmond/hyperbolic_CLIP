#!/bin/bash
# IABench shared-manifest training; clean validation selects checkpoints.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iabench_dct
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
LOGDIR="${LOGDIR:-$REPO/comparison/training/logs}"
NUM_WORKERS="${NUM_WORKERS:-8}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
extra_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Missing resume checkpoint: $RESUME_CHECKPOINT" >&2; exit 1; }
  extra_args+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "Dataset: $DATA | Manifest: $SPLIT_MANIFEST | Logs: $LOGDIR"
python -m comparison.training.train \
  --config comparison/training/config/model/dct.yaml \
  --dataset iabench --split_manifest "$SPLIT_MANIFEST" \
  --root_dir "$DATA" \
  --n_epoch 10 --batch_size 32 \
  --num_workers "$NUM_WORKERS" \
  --log_dir "$LOGDIR" "${extra_args[@]}"
