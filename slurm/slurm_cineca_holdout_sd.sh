#!/bin/bash
# ============================================================================
# CINECA Leonardo — Family-cone attribution + leave-one-out novelty test.
# CLIP ViT-L/14, LoRA on both encoders, hyperbolic entailment cones with the
# generator + FAMILY hierarchy (losses/attribution_loss.py family terms).
#
# Experiment: hold out one Stable Diffusion member (SD3_5) from TRAINING, then
# evaluate INCLUDING SD3_5 with the hierarchical back-off. A never-seen SD model
# should abstain at the generator level and route to the "Stable Diffusion"
# family cone ("looks like a SD") — measured by family_route in the eval.
#
# Change HOLDOUT below to leave out a different SD member (SD1_5/SD2_1/SD3/SDXL).
#
# Submit:  sbatch slurm/slurm_cineca_holdout_sd.sh        # default dim=128
#          sbatch slurm/slurm_cineca_holdout_sd.sh 16     # other embedding dim
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=attr_holdout_sd
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

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache          # avoid filling home quota
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP
DATA=$WORK/hyp_fine_tuning/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/hyp_fine_tuning/checkpoints
DIM=${1:-128}
HOLDOUT=SD3_5               # the SD member left out of training
CKPT=$OUT/attribution_holdout_${HOLDOUT}_d${DIM}.pt

# Full generator set (same as slurm_cineca_all.sh). The held-out generator stays
# in the list: --holdout_generators removes it from TRAINING but the eval below
# loads it back in.
GENERATORS="real 4o gemini grok3 FLUX \
            SD1_5 SD2_1 SD3 SD3_5 SDXL \
            PIXART PLAYGROUND_2_5 KANDINSKY CogView3_PLUS \
            hidream hunyuan ideogram infinity janus-pro kling \
            mid-5.2 mid-6.0"
SEMANTICS="COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k"

mkdir -p $OUT
cd $REPO

# ── Training (SD3_5 held out) ─────────────────────────────────────────────────
# Family terms active: family cones (e.g. Stable Diffusion) sit above the
# generator cones. target_norm_family < target_norm so family cones stay broader.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      $GENERATORS \
    --holdout_generators $HOLDOUT \
    --semantics       $SEMANTICS \
    --clip_name       openai/clip-vit-large-patch14 \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  $DIM \
    --curv            1.0 \
    --min_radius      0.5 \
    --margin          0.3 \
    --lambda_neg      1.0 \
    --lambda_cap_in_class 1.0 \
    --lambda_img_in_cap   0.5 \
    --lambda_norm     0.5 \
    --target_norm     4.0 \
    --lambda_img_in_family 1.0 \
    --lambda_gen_in_family 1.0 \
    --target_norm_family   2.5 \
    --batch_size      256 \
    --num_epochs      8 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --val_frac        0.2 \
    --num_workers     8 \
    --output          $CKPT

echo "Done training: $CKPT"

# ── Eval with novelty detection (SD3_5 included) ──────────────────────────────
python -m tests.eval_attribution_hierarchical \
    --checkpoint   $CKPT \
    --dataset_path $DATA \
    --captions_dir $CAPS \
    --generators   $GENERATORS \
    --semantics    $SEMANTICS \
    --split        val \
    --tau          0.05

echo "Done eval: $CKPT"
