#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, MULTI-VIEW (whole image + 3x3 patch grid)
#
# Every sample contributes 10 views and ALL of them must land inside the cone of
# the sample's generator. Same class anchors for every view: a patch of a FLUX
# image is still a FLUX image.
#
# RECIPE = IDENTICAL to the single-view reference run, so the ONLY variable is the
# number of views: full train split, 5 epochs, effective batch 256, lr 3e-4. That
# gives 1366 optimizer steps/epoch x 5 = 6830, exactly the reference run's count.
#
# BUDGET (measured, not guessed): the finished jobs ran at 210-238 forward/s on
# 2 GPUs. 349888 draws x 5 epochs x 10 views = 17.5M forwards ~= 21h → one job.
# The first attempt failed because it used --samples_per_epoch AND grad_accum 16,
# which together left only 748 optimizer steps; the model never even escaped its
# initialisation (xi ~ pi = every embedding still at the origin). Do not re-add
# either without recomputing the optimizer-step count the trainer now prints.
#
# The four runs of the ablation matrix, all from this one script:
#   sbatch slurm/slurm_train_22cls_patch.sh                                   # P1
#   sbatch --export=ALL,ANCHOR_INIT=text slurm/slurm_train_22cls_patch.sh     # P2
#   sbatch --export=ALL,PATCH_SOURCE=native slurm/slurm_train_22cls_patch.sh  # P3
#   sbatch --export=ALL,ANCHOR_INIT=text,PATCH_SOURCE=native \
#          slurm/slurm_train_22cls_patch.sh                                   # P4
# ============================================================================

#SBATCH --account=EUHPC_D35_189          # verify with `saldo -b`
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_patch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
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
export IAB_EXCLUDE_GENERATORS=dalle3      # <-- 22-class toggle (whole pipeline)

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/hyp_fine_tuning/checkpoints
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json
ANCHOR_CACHE=$WORK/hyp_fine_tuning/anchor_centroids_22cls.pt

ANCHOR_INIT=${ANCHOR_INIT:-image_centroid}   # image_centroid | text
PATCH_SOURCE=${PATCH_SOURCE:-tensor}         # tensor | native

# Each anchor mode keeps the norm regulariser of its own single-view reference, so
# every patch run sits ONE variable away from something already measured:
#   text   -> sweep rank-1 (task 3): lambda_norm 0.5 / target 4.0
#   centroid -> the attribution_22cls_centroid run: lambda_norm 0
if [ "$ANCHOR_INIT" = "text" ]; then
    LAMBDA_NORM=${LAMBDA_NORM:-0.5}; TARGET_NORM=${TARGET_NORM:-4.0}
else
    LAMBDA_NORM=${LAMBDA_NORM:-0.0}; TARGET_NORM=${TARGET_NORM:-0.0}
fi

CKPT=$OUT/attribution_22cls_patch_${ANCHOR_INIT}_${PATCH_SOURCE}_vitl14.pt

mkdir -p $OUT
cd $REPO

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 22-class manifest not found at $MANIFEST"
    echo "Regenerate with IAB_EXCLUDE_GENERATORS=dalle3 (see slurm_train_22cls_base_2gpu.sh)."
    exit 1
fi

WARM=""
if [ -n "$INIT_FROM" ]; then WARM="--init_from $INIT_FROM"; fi

echo "=== patch run: anchors=$ANCHOR_INIT source=$PATCH_SOURCE "
echo "    lambda_norm=$LAMBDA_NORM target_norm=$TARGET_NORM -> $CKPT"

CUDA_VISIBLE_DEVICES=0,1 python -m patch_attribution.train \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --split_manifest  $MANIFEST \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --anchor_init     $ANCHOR_INIT \
    --patch_source    $PATCH_SOURCE \
    --patch_size      112 \
    --anchor_init_norm  2.0 \
    --anchor_init_cache $ANCHOR_CACHE \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  128 \
    --curv            1.0 \
    --min_radius      0.5 \
    --margin          0.3 \
    --lambda_neg      1.0 \
    --lambda_norm     $LAMBDA_NORM \
    --target_norm     $TARGET_NORM \
    --batch_size      16 \
    --grad_accum      16 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    $WARM \
    --output          $CKPT

echo "Done: $CKPT"
