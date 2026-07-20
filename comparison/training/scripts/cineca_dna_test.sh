#!/bin/bash
# Evaluate a trained dna checkpoint @ 22 CLASSES on all 7 degradation levels.
# Batch size 8, as in the IAB reference script training/scripts_test/dna.bash.
#
# Submit: sbatch comparison/training/scripts/cineca_dna_test.sh [ckpt]
# Without an argument it picks the most recent run's ckpt_best.pth.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_dna_test
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
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate
export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3          # <-- 22-class toggle
REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
cd $REPO
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?} | exclude=${IAB_EXCLUDE_GENERATORS}"
CKPT="${1:-}"
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t comparison/training/logs/default_split/dna/*/ckpt_best.pth 2>/dev/null | head -1 || true)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: no checkpoint found. Pass one: sbatch $0 <path/to/ckpt_best.pth>" >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

python -m comparison.training.test \
  --config comparison/training/config/model/dna_default.yaml \
  --resume_checkpoint "$CKPT" \
  --root_dir "$DATA" \
  --batch_size 8 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --level_start 0 --level_end 7 \
  --log_dir comparison/training/logs_test
echo "Done. Results under comparison/training/logs_test/default_split/dna/test_<ts>/"
