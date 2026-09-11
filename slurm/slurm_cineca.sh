#!/bin/bash
# CINECA Leonardo — four-class cone training with caption terms.
#
# Classes: real, FLUX, SD3, gemini. Uses the dataset's internal 80/20 split,
# not a comparison-harness manifest. Outputs attribution_k4_vitl14.pt.
#
# Submit: sbatch slurm/slurm_cineca.sh

#SBATCH --account=EUHPC_D26_009B         # e.g. IscrB_myproject — fill this in
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
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache      # avoid filling home quota
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1      # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP
DATA=$WORK/hyp_fine_tuning/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/hyp_fine_tuning/checkpoints

mkdir -p $OUT
cd $REPO

# ── Training ──────────────────────────────────────────────────────────────────
# Train class-anchor containment and caption terms on the internal 80/20 split.

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real FLUX SD3 gemini \
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
    --target_norm     4.0 \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --val_frac        0.2 \
    --num_workers     8 \
    --output          $OUT/attribution_k4_vitl14.pt

echo "Done: $OUT/attribution_FLUX_vitl14.pt"
