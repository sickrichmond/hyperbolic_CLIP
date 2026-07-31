#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, SWEEP WINNER + TANGENT-SPACE MIXUP.
#
# Byte-identical to slurm_train_22cls_sweepwinner.sh except for --mixup_alpha, so
# the A/B against that run's 0.993 clean / 0.186 JPEG65 has ONE variable.
#
# Why: on JPEG65 the methods split cleanly by feature type — dct .199, hifi_net
# .188, ours .186, dna .174, patch .154 collapse (high-frequency / local texture)
# while ucf .480, defl .356, resnet50 .352 and repmix .672 hold up (global
# features). repmix is the interesting one: its ImageNet-C perturbation is
# COMMENTED OUT in dataset_repmix.py:47, so it gets there without ever seeing a
# degradation, and what is left is mixup. This run ports that mechanism.
#
# NOT augmentation: the model sees no corruption, so unlike --train_augment the
# result stays head-to-head comparable with the baselines (no asterisk).
#
# ~1.75M forwards at the measured 210 forward/s on 2 GPUs → ~2.5h.
#
# Submit:  sbatch slurm/slurm_train_22cls_mixup.sh
#          sbatch --export=ALL,MIXUP_ALPHA=0.4 slurm/slurm_train_22cls_mixup.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_mixup
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
MIXUP_ALPHA=${MIXUP_ALPHA:-0.2}
CKPT=$OUT/attribution_22cls_mixup${MIXUP_ALPHA}_vitl14.pt

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
    --mixup_alpha     $MIXUP_ALPHA \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
