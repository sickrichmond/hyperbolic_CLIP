#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, MULTI-VIEW (whole image + 3x3 patch grid)
#
# Every sample contributes 10 views (the frame plus nine 112px windows, cut from
# the 224px tensor and resized back) and ALL of them must land inside the cone of
# the sample's generator. Same class anchors for every view: a patch of a FLUX
# image is still a FLUX image.
#
# COST: 10 forward passes per sample. The single-view run did 5 epochs x 350k
# images in ~40h on 2 GPUs, so a 48h job here fits roughly 190k sample draws —
# hence --samples_per_epoch. The micro-batch drops to 16 so each GPU sees 80
# images per forward (the proven single-view run peaked at 128/GPU), and
# --grad_accum 16 keeps the EFFECTIVE batch at the 256 the lr was swept at.
#
# Anchors: image centroids, reusing the cache written by the centroid run — same
# space, same head, so the ~20 min pre-pass is not paid again.
#
# 2 GPUs + boost_qos_lprod: whole-node jobs starve on this cluster.
#
# Submit:  sbatch slurm/slurm_train_22cls_patch.sh
# Continue a finished job:
#   sbatch --export=ALL,INIT_FROM=$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_patch_vitl14.pt \
#          slurm/slurm_train_22cls_patch.sh
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

mkdir -p $OUT
cd $REPO

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 22-class manifest not found at $MANIFEST"
    echo "Regenerate with IAB_EXCLUDE_GENERATORS=dalle3 (see slurm_train_22cls_base_2gpu.sh)."
    exit 1
fi

WARM=""
if [ -n "$INIT_FROM" ]; then WARM="--init_from $INIT_FROM"; fi

CUDA_VISIBLE_DEVICES=0,1 python -m patch_attribution.train \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --split_manifest  $MANIFEST \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
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
    --lambda_norm     0.0 \
    --target_norm     0.0 \
    --batch_size      16 \
    --grad_accum      16 \
    --samples_per_epoch 48000 \
    --num_epochs      4 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    $WARM \
    --output          $OUT/attribution_22cls_patch_vitl14.pt

echo "Done: $OUT/attribution_22cls_patch_vitl14.pt"
