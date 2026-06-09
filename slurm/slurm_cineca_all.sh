#!/bin/bash
# ============================================================================
# CINECA Leonardo — Attribution-CLIP fine-tuning (Stage 1), ALL generators
# CLIP ViT-L/14, LoRA on both encoders, hyperbolic entailment-cone loss.
#
# 22 classes = real + 21 generators (the full IAB set MINUS dalle3, which is
# left out on purpose: it is near-identical to 4o — both confuse heavily, see
# the more_families eval where 4o↔dalle3 alone dragged balanced acc to 91.6%).
#
# Submit:  sbatch slurm/slurm_cineca_all.sh
#
# Notes on the data (verified present, ~20k images/generator):
#   - SDXL is missing the FFHQ semantic (18k instead of 20k) — harmless, it
#     just contributes fewer SDXL face samples.
#   - Expect some intrinsic confusion between near-twin generators that share a
#     lineage: SD1_5↔SD2_1, SD3↔SD3_5, mid-5.2↔mid-6.0. This is real (visible
#     in the confusion matrix), NOT a bug, and won't be fixed by hyperparams.
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=attr_all
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
# 22-way attribution with hyperbolic entailment cones. Each image is pulled into
# the cone of its class anchor and out of the other 21 cones. 80/20 split per
# (generator, semantic).
#
# target_norm kept at 5.0 (NOT raised): in the K=10 run images sit just inside
# their own cone (ξ̄≈0.18 < ψ≈0.20). Raising target_norm shrinks ψ and would push
# images out of their cones, collapsing inside%. 128-D leaves plenty of angular
# room for 22 well-separated cone axes, so 5.0 is fine.
#
# The best checkpoint (by balanced val accuracy) is saved every time val
# improves, so even if the job hits the walltime you keep the best-so-far model.

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
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
    --num_epochs      8 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --val_frac        0.2 \
    --num_workers     8 \
    --output          $OUT/attribution_all_no_dalle.pt

echo "Done: $OUT/attribution_all_no_dalle.pt"
