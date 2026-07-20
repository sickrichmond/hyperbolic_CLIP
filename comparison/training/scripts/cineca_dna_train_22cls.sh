#!/bin/bash
# DNA-Det (Yang et al., AAAI 2022) — STAGE 2: attribution @ 22 CLASSES (dalle3
# excluded), initialised from the stage-1 checkpoint. Mirrors the IAB reference
# script training/scripts/dna.bash (n_epoch=10, batch=32, -n 2000).
#
# Submit: sbatch comparison/training/scripts/cineca_dna_train_22cls.sh [ckpt]
# Without an argument it picks the most recent stage-1 ckpt_best.pth.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iab_dna_train_22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=3-18:00:00
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
# Stage-1 checkpoint: explicit arg, else the newest pretrain run.
CKPT="${1:-}"
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t comparison/training/logs/default_split/dna/*pretrain*/ckpt_best.pth 2>/dev/null | head -1 || true)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: no stage-1 checkpoint. Run cineca_dna_pretrain_22cls.sh first," >&2
  echo "       or pass one: sbatch $0 <path/to/ckpt_best.pth>" >&2
  exit 1
fi
echo "Stage-1 checkpoint: $CKPT"

python -m comparison.training.train \
  --config comparison/training/config/model/dna_default.yaml \
  --pretrained_path "$CKPT" \
  --root_dir "$DATA" \
  --n_epoch 10 -n 2000 --batch_size 32 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --log_dir comparison/training/logs
echo "Done (22cls). Results under comparison/training/logs/default_split/dna/<run>/"
