#!/bin/bash
# ============================================================================
# CINECA Leonardo — 22-CLASS eval of the MULTI-VIEW attributors (patch / patchfreq).
# Same protocol as every baseline (get_dataloader + calculate_metrics_for_test),
# levels 0..6, output files in the baselines' format.
#
# ANCHOR_INIT / PATCH_SOURCE mirror slurm_train_22cls_patch.sh and derive BOTH the
# checkpoint name and the output dir, so the four runs of the ablation matrix never
# overwrite each other's result files:
#   sbatch slurm/slurm_eval_22cls_patch.sh                                    # P1
#   sbatch --export=ALL,ANCHOR_INIT=text slurm/slurm_eval_22cls_patch.sh      # P2
#   sbatch --export=ALL,PATCH_SOURCE=native slurm/slurm_eval_22cls_patch.sh   # P3
#   sbatch --export=ALL,ANCHOR_INIT=text,PATCH_SOURCE=native \
#          slurm/slurm_eval_22cls_patch.sh                                    # P4
#
# For the pixel+spectrum model (its checkpoint keeps a flat name):
#   sbatch --export=ALL,MODULE=patch_freq_attribution.eval,\
# CKPT=$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_patchfreq_vitl14.pt,\
# LOGDIR=$WORK/outputs/hypclip_patchfreq_22cls slurm/slurm_eval_22cls_patch.sh
#
# The VIEW SOURCE is read from the checkpoint, never from a flag: evaluating a
# native-grid model on 224-tensor views would silently produce a full set of wrong
# metrics. CKPT/LOGDIR can still be overridden directly.
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
ANCHOR_INIT=${ANCHOR_INIT:-image_centroid}   # image_centroid | text
PATCH_SOURCE=${PATCH_SOURCE:-tensor}         # tensor | native
SUFFIX=${ANCHOR_INIT}_${PATCH_SOURCE}
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_patch_${SUFFIX}_vitl14.pt}
LOGDIR=${LOGDIR:-$WORK/outputs/hypclip_patch_${SUFFIX}_22cls}

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
