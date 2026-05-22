#!/bin/bash
# ============================================================================
# CINECA Leonardo — Attribution-CLIP fine-tuning (Stage 1)
# CLIP ViT-L/14, LoRA on both encoders, InfoNCE loss
#
# Submit:  sbatch scripts/slurm_cineca.sh
#
# Data setup (run once from your local machine):
#   rsync -avz --progress /mnt/data3/rtrebiani/iab_dataset/ \
#       <username>@login.leonardo.cineca.it:$WORK/iab_dataset/
#   rsync -avz --progress /mnt/data3/rtrebiani/iab_captions/ \
#       <username>@login.leonardo.cineca.it:$WORK/iab_captions/
#   rsync -avz --progress /mnt/data3/rtrebiani/hyperbolic_CLIP/ \
#       <username>@login.leonardo.cineca.it:$WORK/hyperbolic_CLIP/
# ============================================================================

#SBATCH --account=<YOUR_ACCOUNT>         # e.g. IscrB_myproject — fill this in
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=attr_clip_flux
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32              # 8 workers × 4 GPUs
#SBATCH --gpus-per-node=4              # 4× A100 80GB
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
# Load whichever conda/CUDA modules are available on this allocation
for _m in anaconda3/2023.03-2 anaconda3/2023.09 anaconda3/2024.02 \
           miniconda3/24.1.2-0 miniconda3/23.5.2-0 miniconda3/4.12.0; do
    module load "$_m" 2>/dev/null && break
done
for _m in cuda/12.1 cuda/12.3 cuda/12.4 cuda/11.8; do
    module load "$_m" 2>/dev/null && break
done
source activate deepfake-hyp       # 'conda activate' doesn't work in SLURM — use 'source activate'

export HF_HOME=$WORK/hf_cache      # avoid filling home quota
export TOKENIZERS_PARALLELISM=false

REPO=$WORK/hyperbolic_CLIP
DATA=$WORK/iab_dataset
CAPS=$WORK/iab_captions
OUT=$WORK/checkpoints

mkdir -p $OUT
cd $REPO

# ── Training ──────────────────────────────────────────────────────────────────
# ViT-L/14: 768D embeddings, stronger backbone than ViT-B/32.
# batch_size=1024 → 1022 negatives per InfoNCE step (4× more than with 256).
# lora_r=16 gives slightly more capacity for the larger backbone.

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path  $DATA \
    --captions_dir  $CAPS \
    --generators    FLUX real \
    --semantics     COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name     openai/clip-vit-large-patch14 \
    --lora_r        16 \
    --lora_alpha    32 \
    --batch_size    1024 \
    --num_epochs    3 \
    --lr            5e-5 \
    --weight_decay  0.01 \
    --num_workers   8 \
    --output        $OUT/attribution_FLUX_vitl14.pt

echo "Done: $OUT/attribution_FLUX_vitl14.pt"
