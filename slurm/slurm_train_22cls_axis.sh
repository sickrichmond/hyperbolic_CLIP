#!/bin/bash
# ============================================================================
# CINECA Leonardo — the axis loss: an objective that never saturates.
#
# Phase B failed all four stop criteria and the diagnosis was structural. Three
# independent measurements say the same thing:
#
#   - L_pos = max(0, xi - psi) goes to ZERO GRADIENT the moment the image is inside
#     its cone. That is the measured reason the projection head is free to collapse
#     the class geometry the LoRA builds (centroid ARI 0.253 -> -0.007; the same head
#     under a plain CE keeps 0.119).
#   - The text anchors all sit at the same depth => the same psi => argmin xi IS
#     argmax cos algebraically (agreement 0.9985-0.9998 on every run measured).
#   - LoRA runs on all 36 blocks, including the low ones that carry CLIP's word
#     knowledge, and the text encoder's adapters are dead weight once the anchors
#     stop being text.
#
# Four things change together, so the runs below are NOT one variable apart from
# sweepwin — axis_adam is what isolates the optimizer.
#
#   loss     L_pos = xi^2, an MSE from the cone AXIS. Gradient 2*xi everywhere,
#            zero only ON the axis, and psi is absent so the loss cannot be lowered
#            by widening the cone.
#   depth    --init_depth 3.0 calibrates the projection head so the images START at
#            tangent norm 3.0 (‖x‖ = sinh 3 = 10), DEEPER than the anchors at 2.0
#            (‖x‖ = 3.63). A cone contains what is farther from the origin than its
#            apex: with the head's untouched output (‖t_img‖ ~ 0.01) every image sits
#            at xi = pi, where oxy_angle's acos clamp makes the gradient EXACTLY zero
#            and the run reports a constant loss at every learning rate. In 'text' mode
#            anchors come out of the same head and the two scales track each other for
#            free; a free anchor rescaled to --anchor_init_norm does not.
#   anchors  random directions, free parameters, tangent norm hard-clamped to
#            [1.0, 3.0] after every step => psi pinned between 58.3 and 5.7 deg.
#            No text encoder, no centroid pre-pass: where they end up is entirely
#            what the loss put there.
#   LoRA     upper HALF of the vision encoder only (layers 12-23), 24 adapters
#            instead of 72, nothing on the text side.
#   optim    SGD + nesterov momentum instead of AdamW. NOTE the lr does not carry
#            over: 3e-4 is an Adam number and barely moves under SGD.
#
#   RUN=axis       the loss on its own, SGD.
#   RUN=axis_ce    + the CE ranking term, SGD.
#   RUN=axis_adam  the loss on its own, AdamW at its own lr. The control: without
#                  it, a bad axis run cannot be told apart from an untuned SGD lr.
#
# Read out of the epoch line: xi_sat must stay near 0% (it is the fraction of xi
# pinned at the acos clamp — anything above a few percent and the run is not learning,
# whatever the loss says), L_img_cls must KEEP FALLING after inside_img hits 100%
# (with the hinge it flattened there), ‖t_anc‖ must stay inside [1.00, 3.00], and
# min∠ must climb above 2*psi.
#
# Submit:  sbatch --export=ALL,RUN=axis      slurm/slurm_train_22cls_axis.sh
#          sbatch --export=ALL,RUN=axis_ce   slurm/slurm_train_22cls_axis.sh
#          sbatch --export=ALL,RUN=axis_adam slurm/slurm_train_22cls_axis.sh
# ============================================================================

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

RUN=${RUN:-axis}

# peft re.fullmatch's this against the base model's module names, so one string
# picks both the encoder and the layer range: vision blocks 12-23, q and v.
LORA_T='vision_model\.encoder\.layers\.(1[2-9]|2[0-3])\.self_attn\.(q|v)_proj'

case "$RUN" in
  axis)      EXTRA="--lambda_ce 0   --optimizer sgd   --lr 1e-2" ;;
  axis_ce)   EXTRA="--lambda_ce 1.0 --optimizer sgd   --lr 1e-2" ;;
  axis_adam) EXTRA="--lambda_ce 0   --optimizer adamw --lr 3e-4" ;;
  *) echo "RUN must be axis, axis_ce or axis_adam (got '$RUN')"; exit 2 ;;
esac
CKPT=$OUT/attribution_22cls_${RUN}_vitl14.pt

mkdir -p $OUT
cd $REPO

# --lambda_norm 0: the hard clamp on the tangent norm replaces L_norm. They constrain
# the same quantity (‖x_anc‖ = sinh‖t‖), imposing both means constraining it twice.
# --lambda_sep 1.0 stays: in Phase B it is the one thing that genuinely worked
# (min anchor margin 2.2 -> 24.5 deg, 2psi/margin 12.8 -> 1.2).
CUDA_VISIBLE_DEVICES=0,1 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --lora_target     "$LORA_T" \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  128 \
    --init_depth      3.0 \
    --curv            1.0 \
    --min_radius      0.5 \
    --anchor_init       random \
    --anchor_init_norm  2.0 \
    --anchor_norm_range 1.0 3.0 \
    --pos_mode        axis \
    --lambda_hinge    1.0 \
    --margin          0.3 \
    --lambda_neg      1.0 \
    --neg_samples     8 \
    --lambda_norm     0 \
    --lambda_sep      1.0 \
    --theta_max       150.0 \
    --momentum        0.9 \
    --weight_decay    0.01 \
    --diag_plot_dir   $WORK/hyp_fine_tuning/viz/$RUN \
    $EXTRA \
    --no_captions \
    --batch_size      256 \
    --num_epochs      5 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
