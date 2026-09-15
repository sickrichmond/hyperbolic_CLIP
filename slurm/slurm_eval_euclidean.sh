#!/bin/bash
# CINECA Leonardo — comparison-harness evaluation of the spherical classifier.
#
# Runs degradation levels 0–6 and writes the harness result format.
# DIM selects the default checkpoint and log directory; CKPT and LOGDIR
# override them directly. IAB_EXCLUDE_GENERATORS excludes dalle3.
#
# Submit: sbatch --export=ALL,DIM=128 slurm/slurm_eval_euclidean.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_22cls_eucl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
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
DIM=${DIM:-128}
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_euclidean_d${DIM}_vitl14.pt}
LOGDIR=${LOGDIR:-$WORK/outputs/euclidean_22cls_d${DIM}}

mkdir -p $LOGDIR
cd $REPO

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"; exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m comparison.training.test_euclidean \
    --checkpoint $CKPT \
    --root_dir   $DATA \
    --batch_size 64 \
    --num_workers 8 \
    --level_start 0 \
    --level_end   7 \
    --log_dir    $LOGDIR

echo "Done. Results in $LOGDIR/test_results_degraded_*.txt"
