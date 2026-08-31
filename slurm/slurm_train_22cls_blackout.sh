#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, SWEEP WINNER + RANDOM PIXEL BLACKOUT.
#
# Byte-identical to slurm_train_22cls_sweepwinner.sh except for --blackout_max, so
# the A/B against that run's 0.993 clean / 0.186 JPEG65 has ONE variable.
#
# Why, instead of mixup: mixup imports features from a semantically unrelated
# image, and its mixed target needs TWO positive hinges whose optimum is degenerate
# (along the arc between the two anchors the derivative is 2λ−1, so every λ>0.5
# lands on the border of the dominant class's cone — λ picks the winner, not the
# proportion). Blacking out a fraction of the pixels keeps ONE label: one cone-loss
# call, no degenerate optimum, and λ is a true intensity. What it regularises is the
# model's reliance on any particular subset of pixels.
#
# NOT an asterisk run: none of the three test corruption families (JPEG, blur,
# downsample) is involved, so this stays head-to-head comparable with the baselines
# — unlike --train_augment. It IS, however, an input-space corruption (mixup was
# not), which is worth declaring.
#
# ~1.75M forwards at the measured 210 forward/s on 2 GPUs → ~2.5h.
#
# BLACKOUT_MAX is the upper end of λ~U(0,λmax), drawn per sample: 0.5 means a mean
# occlusion of 25% and a batch spanning 0-50%.
#
# Submit:  sbatch slurm/slurm_train_22cls_blackout.sh
#          for M in 0.25 0.5 0.75; do \
#            sbatch --export=ALL,BLACKOUT_MAX=$M slurm/slurm_train_22cls_blackout.sh; done
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_blackout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
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
BLACKOUT_MAX=${BLACKOUT_MAX:-0.5}
CKPT=$OUT/attribution_22cls_blackout${BLACKOUT_MAX}_vitl14.pt

mkdir -p $OUT
cd $REPO

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 22-class manifest not found at $MANIFEST"; exit 1
fi

CUDA_VISIBLE_DEVICES=0,1 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --anchor_init     text \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  128 \
    --curv            1.0 \
    --min_radius      0.5 \
    --margin          0.3 \
    --lambda_neg      1.0 \
    --lambda_norm     0.5 \
    --target_norm     4.0 \
    --no_captions \
    --blackout_max    $BLACKOUT_MAX \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
