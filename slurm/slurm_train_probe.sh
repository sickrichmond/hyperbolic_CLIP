#!/bin/bash
# ============================================================================
# CINECA Leonardo — Phase B: train the LINEAR PROBE on cached features.
#
# No CLIP here: it loads one of the caches written by slurm_extract_features.sh
# and trains a single nn.Linear with class-balanced cross-entropy, then prints
# overall / balanced / per-class accuracy + confusion — the same metrics as the
# fine-tuned evals, so the numbers line up directly against 0.993.
#
# Tiny (768-d vectors), so 1 GPU + 30 min is plenty; re-run with different
# --lr / --epochs / --no_class_weight cheaply.
#
# Submit:  sbatch --export=ALL,SOURCE=frozen     slurm/slurm_train_probe.sh
#          sbatch --export=ALL,SOURCE=lora       slurm/slurm_train_probe.sh
#          sbatch --export=ALL,SOURCE=projection slurm/slurm_train_probe.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
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
export IAB_EXCLUDE_GENERATORS=dalle3

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
SOURCE=${SOURCE:-frozen}
FEAT=$WORK/hyp_fine_tuning/clip_features_${SOURCE}
OUT=$WORK/hyp_fine_tuning/checkpoints

mkdir -p $OUT
cd $REPO

python train_linear_probe.py \
    --features_dir $FEAT \
    --epochs       30 \
    --lr           1e-3 \
    --weight_decay 1e-4 \
    --batch_size   4096 \
    --output       $OUT/linear_probe_${SOURCE}.pt

echo "Done: $OUT/linear_probe_${SOURCE}.pt"
