#!/bin/bash
# CINECA Leonardo — axis-cone classifier recipes (22 classes).
#
# q is squared directional chord distance divided by the cone-wall chord:
# q=0 on the axis, approximately 1 on the wall, and >1 outside.
# Free apertures are sigmoid-bounded to PSI_RANGE; simplex modes fix them.
# Coverage and separation are weighted objectives, not hard guarantees.
#
# RUN=axis: SGD, trainable random axes/apertures.
# RUN=axis_adam: AdamW, trainable random axes/apertures.
# RUN=axis_constrained: fixed image radius, separate image/anchor learning rates.
# RUN=axis_anchors: frozen image encoder/head, trainable axes/apertures.
# RUN=axis_simplex: frozen simplex axes and 45-degree half-apertures, with CE.
# RUN=axis_simplex_cover: 20 epochs; coverage weight 5, center 0.1, CE 0.5.
#
# All modes use vision-layer 12–23 q/v LoRA and no caption terms.
# Simplex modes attempt calibration on the selected checkpoint's clean train
# embeddings; infeasible coverage/separation leaves the fixed aperture unchanged.
# Use high-dimensional coverage and pairwise-angle statistics to assess fit.
#
# Submit: sbatch --export=ALL,RUN=axis_simplex_cover slurm/slurm_train_22cls_axis.sh
# CHECKPOINT_DIR overrides checkpoint storage; RUN defaults to axis.

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_axis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# Propagate training/save failures to SLURM; never print Done after a failed run.
set -e

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
OUT=${CHECKPOINT_DIR:-$WORK/hyp_fine_tuning/checkpoints}
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json

RUN=${RUN:-axis}
PSI_RANGE="5.0 60.0"
NEG_SAMPLES=8
EPOCHS=10

# peft re.fullmatch's this against the base model's module names, so one string
# picks both the encoder and the layer range: vision blocks 12-23, q and v.
LORA_T='vision_model\.encoder\.layers\.(1[2-9]|2[0-3])\.self_attn\.(q|v)_proj'

case "$RUN" in
  axis)      EXTRA="--optimizer sgd   --lr 1e-2 --lr_min 1e-3" ;;
  axis_adam) EXTRA="--optimizer adamw --lr 3e-4 --lr_min 3e-5" ;;
  axis_constrained)
    NEG_SAMPLES=0
    EXTRA="--optimizer sgd --lr 1e-3 --anchor_lr 1e-2 --lr_min 1e-4 --fixed_image_radius 4.0 --radial_margin 0.5 --inside_margin 2.0 --lambda_sep 10.0 --separation_margin 2.0 --log_every 10 --snapshot_every 100 --plot_all_train"
    ;;
  axis_anchors)
    NEG_SAMPLES=0
    EXTRA="--optimizer sgd --lr 1e-2 --lr_schedule constant --anchors_only --fixed_image_radius 4.0 --radial_margin 0.5 --inside_margin 2.0 --lambda_sep 10.0 --separation_margin 2.0 --log_every 10 --snapshot_every 100 --plot_all_train"
    ;;
  axis_simplex)
    NEG_SAMPLES=0
    EXTRA="--optimizer sgd --lr 1e-3 --lr_min 1e-4 --anchor_init simplex --freeze_anchors --fixed_psi 45.0 --fixed_image_radius 4.0 --radial_margin 0.5 --inside_margin 2.0 --separation_margin 2.0 --lambda_aperture 0 --lambda_sep 0 --lambda_ce 1.0 --calibrate_psi --log_every 10 --snapshot_every 100 --plot_all_train"
    ;;
  axis_simplex_cover)
    NEG_SAMPLES=0
    EPOCHS=20
    EXTRA="--optimizer sgd --lr 1e-3 --lr_min 1e-4 --anchor_init simplex --freeze_anchors --fixed_psi 45.0 --fixed_image_radius 4.0 --radial_margin 0.5 --inside_margin 2.0 --separation_margin 2.0 --lambda_aperture 0 --lambda_sep 0 --lambda_neg 0 --lambda_cover 5.0 --lambda_center 0.1 --lambda_ce 0.5 --calibrate_psi --log_every 10 --snapshot_every 100 --plot_all_train"
    ;;
  *) echo "RUN must be axis, axis_adam, axis_constrained, axis_anchors, axis_simplex, or axis_simplex_cover (got '$RUN')"; exit 2 ;;
esac
CKPT=$OUT/attribution_22cls_${RUN}_vitl14.pt

mkdir -p "$OUT"
cd "$REPO"

# No norm penalty: fixed-radius runs constrain image depth exactly in the projection.
CUDA_VISIBLE_DEVICES=0,1 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --loss            axis \
    --lora_target     "$LORA_T" \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  128 \
    --curv            1.0 \
    --min_radius      0.5 \
    --anchor_init       random \
    --anchor_init_norm  2.0 \
    --psi_range       $PSI_RANGE \
    --nu              0.05 \
    --lambda_aperture 1.0 \
    --lambda_neg      1.0 \
    --neg_samples     $NEG_SAMPLES \
    --momentum        0.9 \
    --weight_decay    0.01 \
    --diag_plot_dir   $WORK/hyp_fine_tuning/viz/$RUN \
    $EXTRA \
    --no_captions \
    --batch_size      256 \
    --num_epochs      $EPOCHS \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          "$CKPT"

echo "Done: $CKPT"
