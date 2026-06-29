#!/bin/bash
#SBATCH --job-name=iab_rn50_train
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=iab_rn50_train_%j.out
#SBATCH --error=iab_rn50_train_%j.err

# ============================================================================
# Train ResNet-50 attributor (ImageAttributionBench) on CINECA Leonardo.
# train.py runs the degraded-test loop (levels 0..6) automatically at the end,
# so this single job already produces a first evaluation. Use
# cineca_resnet50_test.sbatch to re-evaluate a saved checkpoint.
# ============================================================================

set -euo pipefail

# ---- EDIT THESE TWO IF NEEDED ---------------------------------------------
# Directory that CONTAINS the `comparison/` folder (the import root).
PROJECT_ROOT=/leonardo_work/EUHPC_D26_009B/hyp_fine_tuning/hyperbolic_CLIP
# Dataset root (the folder with the 23 model-class subdirs + real).
DATA=/leonardo_work/EUHPC_D26_009B/hyp_fine_tuning/iab_dataset
# Python venv that has the deps (see comparison/requirements_resnet50.txt).
VENV=/leonardo_work/EUHPC_D26_009B/hyp_fine_tuning/bin/activate
# ---------------------------------------------------------------------------

# Activate environment ------------------------------------------------------
# module load cuda/12.1                      # usually NOT needed: pip torch+cu121 bundles its CUDA libs
source "$VENV"

# Imports are absolute (comparison.*) so the import root must be on sys.path.
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

CONFIG=comparison/training/config/model/resnet50.yaml
LOGDIR=comparison/training/logs

# ---- STANDARD SPLIT -------------------------------------------------------
srun python -m comparison.training.train \
  --config "$CONFIG" \
  --root_dir "$DATA" \
  --n_epoch 10 \
  -n 2000 \
  --batch_size 32 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --log_dir "$LOGDIR"

# ---- SEMANTIC SPLIT (the paper's hard setting): uncomment to run instead ---
# for TASK in 1 2 3; do
#   srun python -m comparison.training.train \
#     --config "$CONFIG" \
#     --root_dir "$DATA" \
#     --use_semantic_split --task_id "$TASK" \
#     --n_epoch 10 -n 2000 --batch_size 32 \
#     --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
#     --log_dir "$LOGDIR"
# done

echo "Done. Checkpoints + test_results_degraded_*.txt under $LOGDIR/<split>/resnet50/<run>/"
