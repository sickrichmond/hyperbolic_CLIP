#!/bin/bash
# ============================================================================
# CINECA Leonardo — EUCLIDEAN BASELINE for attribution (Stage 1), ALL generators
# CLIP ViT-L/14, LoRA on both encoders, cosine-similarity + cross-entropy.
#
# This is the geometry ablation of slurm/slurm_cineca_all.sh: SAME backbone,
# SAME LoRA (r=16, alpha=32), SAME projection-head capacity (→128), SAME 22
# classes, semantics, batch size, epochs, LR and split. The ONLY difference is
# that embeddings live on the Euclidean unit sphere and images are matched to
# the text anchors by cosine similarity (trainable zero-shot CLIP) instead of by
# hyperbolic entailment cones. Compare its eval against the hyperbolic
# checkpoint to read off how much the hyperbolic geometry contributes.
#
# 22 classes = real + 21 generators (full IAB set MINUS dalle3, as in the
# hyperbolic run, so the two are directly comparable).
#
# Submit:  sbatch slurm/slurm_euclidean_baseline.sh
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=attr_eucl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=20:00:00
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
# Image-vs-anchor cross-entropy on cosine similarities. No cone / norm / caption
# hyperparameters: the sphere has no entailment hierarchy, so those terms have no
# analogue and are intentionally absent (see losses/euclidean_attribution_loss.py).

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution_euclidean.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o gemini grok3 FLUX \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL \
                      PIXART PLAYGROUND_2_5 KANDINSKY CogView3_PLUS \
                      hidream hunyuan ideogram infinity janus-pro kling \
                      mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --lora_r          16 \
    --lora_alpha      32 \
    --embed_dim       128 \
    --batch_size      256 \
    --num_epochs      8 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --val_frac        0.2 \
    --num_workers     8 \
    --output          $OUT/attribution_all_euclidean.pt

echo "Done: $OUT/attribution_all_euclidean.pt"
