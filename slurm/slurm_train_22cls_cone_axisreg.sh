#!/bin/bash
# CINECA Leonardo — exterior-angle cone loss with axis-ray regularization.
#
# Free random tangent anchors have norms projected into [1, 2.5]; apertures
# are depth-coupled. Image-head initialization targets tangent norm 3, without
# fixing image norms during training. SGD uses a constant learning rate.
# Caption terms and anchor-norm penalties are disabled.
#
# RUN=coneaxis uses lambda_axis=0.1; coneaxis_off uses zero.
# The regularizer measures distance to the outward geodesic ray beyond the
# correct anchor. Step CSVs and snapshots report training geometry.
#
# Submit: sbatch --export=ALL,RUN=coneaxis slurm/slurm_train_22cls_cone_axisreg.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_coneaxis
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

RUN=${RUN:-coneaxis}

# peft re.fullmatch's this against the base model's module names, so one string
# picks both the encoder and the layer range: vision blocks 12-23, q and v.
LORA_T='vision_model\.encoder\.layers\.(1[2-9]|2[0-3])\.self_attn\.(q|v)_proj'

case "$RUN" in
  coneaxis)     LAMBDA_AXIS=0.1 ;;
  coneaxis_off) LAMBDA_AXIS=0.0 ;;
  *) echo "RUN must be coneaxis or coneaxis_off (got '$RUN')"; exit 2 ;;
esac
CKPT=$OUT/attribution_22cls_${RUN}_vitl14.pt
VIZ=$WORK/hyp_fine_tuning/viz/$RUN

mkdir -p $OUT $VIZ
cd $REPO

# Anchor tangent norms start at 2 and are projected into [1, 2.5].
# --init_depth targets image tangent norm 3 on the initialization batch;
# it does not constrain image depth during subsequent training.
CUDA_VISIBLE_DEVICES=0,1 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --loss            cone \
    --lora_target     "$LORA_T" \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  128 \
    --curv            1.0 \
    --min_radius      0.5 \
    --anchor_init       random \
    --anchor_init_norm  2.0 \
    --anchor_norm_range 1.0 2.5 \
    --init_depth      3.0 \
    --lambda_hinge    1.0 \
    --lambda_neg      1.0 \
    --margin          0.1 \
    --lambda_axis     $LAMBDA_AXIS \
    --lambda_norm     0.0 \
    --neg_samples     0 \
    --optimizer       sgd \
    --lr_schedule     constant \
    --lr              1e-2 \
    --anchor_lr       1e-2 \
    --momentum        0.9 \
    --weight_decay    0.01 \
    --log_every       10 \
    --snapshot_every  100 \
    --diag_plot_dir   $VIZ \
    --no_captions \
    --batch_size      256 \
    --num_epochs      10 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
echo "Trace: $VIZ/stats.csv   Frames: $VIZ/step_*.png"
