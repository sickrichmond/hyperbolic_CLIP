#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, SWEEP WINNER, run to convergence.
#
# Config = rank 1 of slurm/sweep_configs_22cls.txt (task 3, 98.86% val_balanced):
# TEXT anchors, lr 3e-4, lora 16/32, min_radius 0.5, margin 0.3, lambda_neg 1.0,
# lambda_norm 0.5 / target_norm 4.0. In the sweep it only ran 3 EPOCHS, for ranking
# purposes; this is the pending "retrain the winner for >=5 epochs" action from
# sweep_results_hypclip.md. Base loss, all images (parity with the baselines).
#
# NOTE on lambda_norm: the image-centroid run combined lr 3e-4 (rank 1) with
# lambda_norm 0 (rank 2) — a combination the sweep never tested. So the current
# "centroids 93.3% vs text 98.86%" gap has two variables in it; this run pins down
# the text-anchor side properly.
#
# ~1.75M forwards at the measured 210 forward/s on 2 GPUs → ~2.5h.
#
# Submit:  sbatch slurm/slurm_train_22cls_sweepwinner.sh
#          sbatch --export=ALL,AUGMENT=1 slurm/slurm_train_22cls_sweepwinner.sh   # -> aug*
#          sbatch --export=ALL,AUGMENT=1,AUG_POLICY=omnidfa \
#                 slurm/slurm_train_22cls_sweepwinner.sh                          # -> omniaug†
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_sweepwin
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

# AUGMENT=1 -> train-time augmentation on the TRAIN split. Two policies:
#   AUG_POLICY=corruption (default) — random JPEG/blur/downsample, i.e. the test-time
#     corruption FAMILIES → the run goes in the tables with an asterisk, like dna*.
#   AUG_POLICY=omnidfa — Table 8 of arXiv 2509.25682. Milder: only DS0.5 of the seven
#     test levels falls inside its ranges → weaker asterisk (†), a literature recipe
#     rather than an augmentation shaped on the test set.
AUGMENT=${AUGMENT:-0}
AUG_POLICY=${AUG_POLICY:-corruption}
if [ "$AUGMENT" = 1 ]; then
    AUG_FLAG="--train_augment --aug_policy $AUG_POLICY"
    if [ "$AUG_POLICY" = omnidfa ]; then
        CKPT=$OUT/attribution_22cls_sweepwin_omniaug_vitl14.pt
    else
        CKPT=$OUT/attribution_22cls_sweepwin_aug_vitl14.pt
    fi
else
    AUG_FLAG=
    CKPT=$OUT/attribution_22cls_sweepwin_vitl14.pt
fi

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
    $AUG_FLAG \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
