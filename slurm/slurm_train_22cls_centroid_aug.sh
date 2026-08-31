#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, IMAGE-CENTROID ANCHORS + AUGMENTATION
#
# Identical to slurm_train_22cls_centroid.sh plus --train_augment: random JPEG /
# blur / downsample on the TRAIN split (data/degradations.py:random_degradation),
# same corruption families as the test pipeline but with continuously sampled
# parameters. Val stays clean.
#
# ⚠️ FAIRNESS: this model HAS seen the test-time corruption families in training,
# which no baseline has (fairness audit 2026-07-21 — the same thing DNA-Det is
# asterisked for). Report it as `ours-aug*`, as a separate axis (ours-clean vs
# ours-aug), NOT head-to-head with the baselines on the degraded rows.
#
# Anchors are NOT the encoded text templates any more: a single forward pass over
# the train split gives the per-class mean CLIP embedding, the projection head
# maps it to tangent space, and from there each anchor is a free parameter. Where
# SD3 and SD3.5 images actually land decides how far apart their anchors start,
# instead of what the text encoder makes of the two strings.
#
# Hyperparameters = the sweep winners (slurm/sweep_configs_22cls.txt): lr 3e-4 is
# the dominant axis (+5pt over 5e-5) and the anchor-norm regulariser hurts
# (λ_norm 0 was 2nd overall) — the two have never been combined before. λ_neg
# stays at 1.0: 2.0 collapses training to ~random.
#
# 2 GPUs + boost_qos_lprod: whole-node jobs starve on this cluster (see the
# 4-GPU→2-GPU note in slurm_train_22cls_base_2gpu.sh). Results are unaffected —
# DataParallel splits the same total batch of 256.
#
# Reuses the centroid cache written by the clean run (centroids are always
# computed on clean images, so both runs start from the same anchors). Launch
# this one after the clean run has written ANCHOR_CACHE; starting both at once
# only wastes the ~20 min pre-pass twice, it is not harmful (atomic write).
#
# Submit:  sbatch slurm/slurm_train_22cls_centroid_aug.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189          # verify with `saldo -b`
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_centr_aug
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

CUDA_VISIBLE_DEVICES=0,1 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --anchor_init       image_centroid \
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
    --no_captions \
    --train_augment \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $OUT/attribution_22cls_centroid_aug_vitl14.pt

echo "Done: $OUT/attribution_22cls_centroid_aug_vitl14.pt"
