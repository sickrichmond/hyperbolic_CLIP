#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, PIXEL (image + 3x3 patches) + SPECTRUM
#
# Two hyperbolic spaces over one shared CLIP+LoRA backbone: the 10 pixel views in
# one, the centred log-magnitude FFT in another with its own anchors. Class score
# = sum of the two branches' logits with a learned per-branch temperature (the CE
# term sees detached angles, so it trains only those two scalars).
#
# RECIPE = identical to the single-view reference run (full train split, 5 epochs,
# effective batch 256, lr 3e-4 → 6830 optimizer steps), so the only variable is the
# views. 11 forwards/sample x 349888 x 5 = 19.2M forwards ~= 23h at the measured
# 210-238 forward/s on 2 GPUs → one job. See the budget note in
# slurm_train_22cls_patch.sh for why the first attempt collapsed.
#
# The pixel anchors reuse the centroid cache; the SPECTRAL anchors need their own
# pre-pass (~20 min, cached in ANCHOR_CACHE_SPEC) the first time.
#
# ⚠️ RUN tests/inspect_centroids.py ON $ANCHOR_CACHE_SPEC FIRST. In the failed run
# the spectral branch sat at 1/22 = chance for four epochs while its loss fell: if
# the spectral centroids are near-parallel (off-diagonal cosine ~0.99+), the
# FFT-into-CLIP embedding carries no class signal and this job is 23h wasted.
#
# Submit:  sbatch slurm/slurm_train_22cls_patchfreq.sh
# Fusion ablation (plain sum, temperatures frozen at 1):
#   sbatch --export=ALL,LAMBDA_FUSE=0 slurm/slurm_train_22cls_patchfreq.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189          # verify with `saldo -b`
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_patchfreq
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
ANCHOR_CACHE_SPEC=$WORK/hyp_fine_tuning/anchor_centroids_22cls_spectral.pt

mkdir -p $OUT
cd $REPO

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 22-class manifest not found at $MANIFEST"; exit 1
fi

WARM=""
if [ -n "$INIT_FROM" ]; then WARM="--init_from $INIT_FROM"; fi

CUDA_VISIBLE_DEVICES=0,1 python -m patch_freq_attribution.train \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --split_manifest  $MANIFEST \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --patch_size      112 \
    --anchor_init_norm       2.0 \
    --anchor_init_cache      $ANCHOR_CACHE \
    --anchor_init_cache_spec $ANCHOR_CACHE_SPEC \
    --lambda_fuse     ${LAMBDA_FUSE:-1.0} \
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
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    $WARM \
    --output          $OUT/attribution_22cls_patchfreq_vitl14.pt

echo "Done: $OUT/attribution_22cls_patchfreq_vitl14.pt"
