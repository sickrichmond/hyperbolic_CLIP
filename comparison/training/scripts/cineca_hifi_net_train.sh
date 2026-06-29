#!/bin/bash
# ============================================================================
# CINECA Leonardo — Train the HiFi-Net attributor (ImageAttributionBench).
# HiFi-Net = HRNet feature extractor + NLCDetection hierarchical head
# (Guo et al. 2023); loss sums CE over the 4 hierarchy levels. Heavier than
# ResNet-50/DCT, so batch_size=8 (mirrors the upstream script).
# train.py runs the degraded-test loop (levels 0..6) automatically at the end.
# Use cineca_hifi_net_test.sh to re-evaluate a saved checkpoint.
#
# Submit:  sbatch comparison/training/scripts/cineca_hifi_net_train.sh
#
# Optional HRNet pretrained init: seg_hrnet loads models/hrnet_w18_small_v2.pth
# (relative to $REPO) ONLY if the file exists, else it trains the backbone from
# scratch. Drop that .pth under $REPO/models/ to enable the pretrained init.
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_hifi_train
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

CONFIG=comparison/training/config/model/hifi_net.yaml
LOGDIR=comparison/training/logs

# ── STANDARD SPLIT ──────────────────────────────────────────────────────────
python -m comparison.training.train \
  --config "$CONFIG" \
  --root_dir "$DATA" \
  --n_epoch 10 \
  -n 2000 \
  --batch_size 8 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --log_dir "$LOGDIR"

# ── SEMANTIC SPLIT (the paper's hard setting): uncomment to run instead ──────
# for TASK in 1 2 3; do
#   python -m comparison.training.train \
#     --config "$CONFIG" --root_dir "$DATA" \
#     --use_semantic_split --task_id "$TASK" \
#     --n_epoch 10 -n 2000 --batch_size 8 \
#     --num_workers "${SLURM_CPUS_PER_TASK:-8}" --log_dir "$LOGDIR"
# done

echo "Done. Checkpoints + test_results_degraded_*.txt under $LOGDIR/<split>/hifi_net/<run>/"
