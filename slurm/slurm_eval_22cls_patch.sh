#!/bin/bash
# ============================================================================
# CINECA Leonardo — 22-CLASS eval of the MULTI-VIEW attributors (patch / patchfreq).
# Same protocol as every baseline (get_dataloader + calculate_metrics_for_test),
# levels 0..6, output files in the baselines' format.
#
# Defaults to the patch model. For the pixel+spectrum one:
#   sbatch --export=ALL,MODULE=patch_freq_attribution.eval,\
# CKPT=$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_patchfreq_vitl14.pt,\
# LOGDIR=$WORK/outputs/hypclip_patchfreq slurm/slurm_eval_22cls_patch.sh
#
# Submit:  sbatch slurm/slurm_eval_22cls_patch.sh
# ============================================================================
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_22cls_patch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
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
MODULE=${MODULE:-patch_attribution.eval}
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_patch_vitl14.pt}
LOGDIR=${LOGDIR:-$WORK/outputs/hypclip_patch_22cls}

mkdir -p $LOGDIR
cd $REPO

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"; exit 1
fi

# 10-11 views per image → smaller batch than the single-view eval, and 12h
# walltime instead of 4h for the seven degradation levels.
CUDA_VISIBLE_DEVICES=0 python -m $MODULE \
    --checkpoint $CKPT \
    --root_dir   $DATA \
    --batch_size 16 \
    --num_workers 8 \
    --level_start 0 \
    --level_end   7 \
    --log_dir    $LOGDIR

echo "Done. Results in $LOGDIR/test_results_degraded_*.txt"
