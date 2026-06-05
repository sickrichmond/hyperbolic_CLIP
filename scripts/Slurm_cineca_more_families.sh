#!/bin/bash
# ============================================================================
# CINECA Leonardo — Attribution-CLIP fine-tuning (Stage 1), K=8 generators
# CLIP ViT-L/14, LoRA on both encoders, hyperbolic entailment-cone loss.
#
# Classes (8): real + one cone per generator, spanning several architectures:
#   real    — reference images
#   FLUX    — DiT (FLUX.1 schnell)
#   SD3_5   — LDM/DiT (Stable Diffusion 3.5-medium)
#   SDXL    — LDM (Stable Diffusion XL base 1.0)
#   4o      — GPT-4o native image generation (multimodal)
#   grok3   — Grok-3 image (multimodal; watermark auto-cropped in dataset)
#   infinity— Infinity-2B (autoregressive)
#   dalle3  — DALL-E 3 (commercial t2i)
#
# Submit:  sbatch scripts/Slurm_cineca_more_families.sh
#
# Required data (download first if missing):
#   python scripts/download_iab.py --dataset_path $WORK/iab_dataset \
#       --model_classes 4o grok3 infinity dalle3 SD3_5 SDXL \
#       --semantic_classes COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
#       --delete_zip
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=attr_more_families
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32              # 8 workers × 4 GPUs
#SBATCH --gpus-per-node=4               # 4× A100 80GB
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hf_cache          # avoid filling home quota
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP
DATA=$WORK/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/checkpoints

mkdir -p $OUT
cd $REPO

# ── Training ──────────────────────────────────────────────────────────────────
# 8-way attribution with hyperbolic entailment cones. Each image is pulled into
# the cone of its class anchor and out of the other 7 cones. 80/20 split per
# (generator, semantic). target_norm raised to 5.0 (vs 4.0 for K=4) so the cones
# stay narrow enough to separate 8 classes: ψ ≈ arcsin(2·0.5/5) ≈ 11.5°.

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real FLUX SD3_5 SDXL 4o grok3 infinity dalle3 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  128 \
    --curv            1.0 \
    --min_radius      0.5 \
    --margin          0.3 \
    --lambda_neg      1.0 \
    --lambda_cap_in_class 1.0 \
    --lambda_img_in_cap   0.5 \
    --lambda_norm     0.5 \
    --target_norm     5.0 \
    --batch_size      256 \
    --num_epochs      8 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --val_frac        0.2 \
    --num_workers     8 \
    --output          $OUT/attribution_more_families.pt

echo "Done: $OUT/attribution_more_families.pt"
