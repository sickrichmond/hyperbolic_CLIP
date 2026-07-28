#!/bin/bash
# ============================================================================
# CINECA Leonardo — 22-CLASS eval of the hyperbolic-CLIP attributor (ours).
# Same protocol as the baselines (get_dataloader + calculate_metrics_for_test),
# at 22 classes (IAB_EXCLUDE_GENERATORS=dalle3). Levels 0..6.
#
# CKPT and LOGDIR are overridable, e.g. for the image-centroid runs:
#   sbatch --export=ALL,CKPT=$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_centroid_vitl14.pt,\
# LOGDIR=$WORK/outputs/hypclip_centroid slurm/slurm_eval_22cls.sh
# Submit:  sbatch slurm/slurm_eval_22cls.sh
# ============================================================================
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_22cls
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
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_base_vitl14.pt}
LOGDIR=${LOGDIR:-$WORK/outputs/hypclip_fair_22cls}

mkdir -p $LOGDIR
cd $REPO

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"; exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m comparison.training.test_hypclip \
    --checkpoint $CKPT \
    --root_dir   $DATA \
    --batch_size 64 \
    --num_workers 8 \
    --level_start 0 \
    --level_end   7 \
    --log_dir    $LOGDIR

echo "Done. Results in $LOGDIR/test_results_degraded_*.txt"
