#!/bin/bash
# ============================================================================
# CINECA Leonardo — Train the ResNet-50 attributor (ImageAttributionBench).
# train.py runs the degraded-test loop (levels 0..6) automatically at the end,
# so this single job already produces a first evaluation. Use
# cineca_resnet50_test.sh to re-evaluate a saved checkpoint.
#
# Submit:  sbatch comparison/training/scripts/cineca_resnet50_train.sh
#
# Pretrained weights (compute nodes have NO internet): the backbone is
# torchvision resnet50(pretrained=True). Pre-fetch ONCE on a login node into the
# shared TORCH_HOME below before submitting:
#   module load python/3.11.7 && source $WORK/hyp_fine_tuning/bin/activate
#   TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache \
#     python -c "import torchvision.models as m; m.resnet50(pretrained=True)"
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_rn50_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail

# ── Environment ─────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache   # torchvision pretrained cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP
DATA=$WORK/hyp_fine_tuning/iab_dataset                # 23 model-class subdirs + real
cd $REPO
export PYTHONPATH="$REPO:${PYTHONPATH:-}"             # imports are absolute (comparison.*)

echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

CONFIG=comparison/training/config/model/resnet50.yaml
LOGDIR=comparison/training/logs

# ── STANDARD SPLIT ──────────────────────────────────────────────────────────
python -m comparison.training.train \
  --config "$CONFIG" \
  --root_dir "$DATA" \
  --n_epoch 10 \
  -n 2000 \
  --batch_size 32 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --log_dir "$LOGDIR"

# ── SEMANTIC SPLIT (the paper's hard setting): uncomment to run instead ──────
# for TASK in 1 2 3; do
#   python -m comparison.training.train \
#     --config "$CONFIG" --root_dir "$DATA" \
#     --use_semantic_split --task_id "$TASK" \
#     --n_epoch 10 -n 2000 --batch_size 32 \
#     --num_workers "${SLURM_CPUS_PER_TASK:-8}" --log_dir "$LOGDIR"
# done

echo "Done. Checkpoints + test_results_degraded_*.txt under $LOGDIR/<split>/resnet50/<run>/"
