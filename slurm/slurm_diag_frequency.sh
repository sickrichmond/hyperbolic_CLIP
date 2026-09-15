#!/bin/bash
# CINECA Leonardo — blur/JPEG diagnostic using harness test images.
#
# Runs comparison.training.diag_frequency with a blur-sigma sweep and JPEG
# quality sweep, plus family-based error routing. The evaluator uses exterior
# angles with default text anchors, not stored free anchors.
# Worker degradation overrides require the fork start method.
# CKPT overrides the checkpoint path.
#
# Submit: sbatch slurm/slurm_diag_frequency.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=diag_freq_22cls
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate
export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3        # <-- 22-class (anchors + test set)

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_base_vitl14.pt}
LOGDIR=$WORK/outputs/hypclip_diag_22cls

mkdir -p $LOGDIR
cd $REPO

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"; exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m comparison.training.diag_frequency \
    --checkpoint $CKPT \
    --root_dir   $DATA \
    --batch_size 64 \
    --num_workers 8 \
    --log_dir    $LOGDIR

echo "Done. Results in $LOGDIR/diag_frequency.{txt,json}"
