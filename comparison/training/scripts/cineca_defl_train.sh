#!/bin/bash
# ============================================================================
# CINECA Leonardo — Train the DEFL attributor (ImageAttributionBench).
# DEFL = handcrafted directional filters + ResNet-50 fused with a FROZEN CLIP
# RN50x16 visual encoder; loss = dual-margin contrastive + CE (Li et al. 2024).
# train.py runs the degraded-test loop (levels 0..6) automatically at the end.
# Use cineca_defl_test.sh to re-evaluate a saved checkpoint.
#
# Submit:  sbatch comparison/training/scripts/cineca_defl_train.sh
#
# Pretrained weights (compute nodes have NO internet) — pre-fetch BOTH ONCE on
# a login node before submitting:
#   module load python/3.11.7 && source $WORK/hyp_fine_tuning/bin/activate
#   # (a) torchvision ResNet-50 → shared TORCH_HOME:
#   TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache \
#     python -c "import torchvision.models as m; m.resnet50(pretrained=True)"
#   # (b) openai-CLIP RN50x16 (~3.5 GB → $HOME/.cache/clip):
#   python -c "import clip; clip.load('RN50x16', device='cpu')"
# Needs openai-clip in the venv: pip install git+https://github.com/openai/CLIP.git
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
# boost_qos_lprod = Leonardo long QOS, walltime up to 4 days (needed: >24h).
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iab_defl_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
# ~8h/epoch x 10 epochs = ~79h (under the 4-day cap)
#SBATCH --time=3-18:00:00
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
DATA=$WORK/hyp_fine_tuning/iab_dataset
cd $REPO
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

CONFIG=comparison/training/config/model/defl.yaml
LOGDIR=comparison/training/logs

# num_workers is kept low: the DEFL dataset initialises a CUDA CLIP model, which
# does not survive DataLoader worker forking at a high worker count.

# ── STANDARD SPLIT ──────────────────────────────────────────────────────────
python -m comparison.training.train \
  --config "$CONFIG" \
  --root_dir "$DATA" \
  --n_epoch 10 \
  -n 2000 \
  --batch_size 8 \
  --num_workers 2 \
  --log_dir "$LOGDIR"

# ── SEMANTIC SPLIT (the paper's hard setting): uncomment to run instead ──────
# for TASK in 1 2 3; do
#   python -m comparison.training.train \
#     --config "$CONFIG" --root_dir "$DATA" \
#     --use_semantic_split --task_id "$TASK" \
#     --n_epoch 10 -n 2000 --batch_size 8 --num_workers 2 --log_dir "$LOGDIR"
# done

echo "Done. Checkpoints + test_results_degraded_*.txt under $LOGDIR/<split>/defl/<run>/"
