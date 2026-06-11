#!/bin/bash
# ============================================================================
# CINECA Leonardo — Train the LINEAR PROBE on cached frozen-CLIP features
# (Fase B). No CLIP here: it loads $WORK/clip_features and trains a single
# nn.Linear with class-balanced cross-entropy, then prints overall / balanced /
# per-class accuracy + confusion — same metrics as the fine-tuned evals.
#
# Requires the cache from slurm_extract_features.sh to exist first.
#
# It's tiny (operates on 768-d vectors), so 1 GPU + 30 min is plenty; you can
# re-run with different --lr / --epochs / --no_class_weight cheaply.
#
# Submit:  sbatch slurm/slurm_train_probe.sh
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=train_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export TOKENIZERS_PARALLELISM=false

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

mkdir -p $WORK/checkpoints

python train_linear_probe.py \
    --features_dir $WORK/clip_features \
    --epochs       100 \
    --lr           1e-3 \
    --weight_decay 1e-4 \
    --batch_size   4096 \
    --output       $WORK/checkpoints/linear_probe.pt

echo "Done: $WORK/checkpoints/linear_probe.pt"
