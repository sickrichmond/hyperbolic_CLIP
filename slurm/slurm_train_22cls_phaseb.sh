#!/bin/bash
# ============================================================================
# CINECA Leonardo — Phase B: give the geometry a job it can actually do.
#
# The sweepwin recipe, one variable at a time. What Phase A measured, and what each
# run answers:
#
#   The projection head zeroes the class geometry the LoRA builds (centroid ARI
#   0.253 -> -0.007 against the generator taxonomy). The same head trained with a
#   plain CE keeps 0.119, so the ARCHITECTURE is innocent and the SATURATING HINGE is
#   what gives the head permission to collapse. But CE alone recovers less than half.
#
#   RUN=ce      hinge OFF, CE ON, nothing else.        Does the geometry matter, or did
#                                                      only the loss shape ever matter?
#                                                      Compare against the euclidean
#                                                      ablation: one variable apart.
#   RUN=flat    + bilateral norm + separation.         Can the cones become a genuine
#                                                      partition? Target: 2psi/margin
#                                                      below 1 (12.8 today). psi is
#                                                      still uniform here, so argmin xi
#                                                      still equals argmax cos —
#                                                      by design, this isolates spread
#                                                      from depth.
#   RUN=hifi    + nested family cones, HiFi-Net tree.  Does the asserted taxonomy help?
#   RUN=emerg   + nested family cones, data tree.      Does the tree the centroids show
#                                                      help more? (data/tree_emergent.json)
#
# Only the hierarchy runs make psi vary across anchors, and equal psi is exactly why
# argmin xi has never differed from argmax cos.
#
# Read out of the epoch line: min∠ must climb above 2·psi (overlap -> 0%), and with a
# hierarchy, in_fam should reach 100% while fam_acc stays high.
#
# Submit:  sbatch --export=ALL,RUN=ce    slurm/slurm_train_22cls_phaseb.sh
#          sbatch --export=ALL,RUN=flat  slurm/slurm_train_22cls_phaseb.sh
#          sbatch --export=ALL,RUN=hifi  slurm/slurm_train_22cls_phaseb.sh
#          sbatch --export=ALL,RUN=emerg slurm/slurm_train_22cls_phaseb.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_phaseb
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

RUN=${RUN:-ce}

# Depth is what makes psi vary. With min_radius 0.5: ‖x‖=1.5 -> psi 41.8 deg,
# ‖x‖=5.0 -> psi 11.5 deg. Families must stay SHALLOWER than models, and models
# shallower than the images (which sit at ~7), or containment points the wrong way.
case "$RUN" in
  ce)
    EXTRA="--lambda_hinge 0 --lambda_ce 1.0 --lambda_norm 0.5 --target_norm 4.0"
    ;;
  flat)
    EXTRA="--lambda_hinge 0 --lambda_ce 1.0 \
           --lambda_norm 0.5 --target_norm 4.0 --norm_mode bilateral \
           --lambda_sep 1.0 --theta_max 150.0"
    ;;
  hifi|emerg)
    [ "$RUN" = hifi ] && TREE="hifi" || TREE="emergent"
    EXTRA="--lambda_hinge 0 --lambda_ce 1.0 \
           --lambda_norm 0.5 --target_norm 5.0 --norm_mode bilateral \
           --target_norm_family 1.5 \
           --lambda_sep 1.0 --theta_max 150.0 \
           --hierarchy $TREE --lambda_family 1.0"
    ;;
  *) echo "RUN must be ce, flat, hifi or emerg (got '$RUN')"; exit 2 ;;
esac
CKPT=$OUT/attribution_22cls_phaseb_${RUN}_vitl14.pt

mkdir -p $OUT
cd $REPO

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
    $EXTRA \
    --no_captions \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
