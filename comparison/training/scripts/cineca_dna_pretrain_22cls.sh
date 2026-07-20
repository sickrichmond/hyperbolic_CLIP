#!/bin/bash
# DNA-Det (Yang et al., AAAI 2022) — STAGE 1: self-supervised pretraining over the
# 170 image-transformation classes. Produces the checkpoint that stage 2 loads.
# Mirrors the IAB reference script training/scripts/dna_pretrain.bash (n_epoch=10,
# batch=32, -n 2000). Must finish BEFORE cineca_dna_train_22cls.sh.
# Submit: sbatch comparison/training/scripts/cineca_dna_pretrain_22cls.sh
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iab_dna_pretrain_22
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
PYDEPS="${IAB_PYDEPS:-$HOME/iab_pydeps}"   # DNA-only: albumentations installed here (venv is read-only)
export PYTHONPATH="$REPO:$PYDEPS:${PYTHONPATH:-}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?} | exclude=${IAB_EXCLUDE_GENERATORS}"
python -m comparison.training.train \
  --config comparison/training/config/model/dna_pretrain.yaml \
  --root_dir "$DATA" \
  --n_epoch 10 -n 2000 --batch_size 32 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --do_test \
  --log_dir comparison/training/logs
# NB: --do_test is action='store_false' -> passing it DISABLES the trailing
# degraded test loop. Stage 1 classifies the 170 transform classes, so the
# 0..6 degraded eval would be meaningless (and slow) here. Stage 2 keeps it.
echo "Stage 1 done. Pass the checkpoint to stage 2:"
echo "  sbatch comparison/training/scripts/cineca_dna_train_22cls.sh <path/to/ckpt_best.pth>"
