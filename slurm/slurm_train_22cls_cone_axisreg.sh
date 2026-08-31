#!/bin/bash
# ============================================================================
# CINECA Leonardo — entailment hinge + AXIS-RAY regulariser, free anchors.
#
# The question this run exists to answer: where do the anchors go when they are
# genuinely free — direction AND depth — and can we watch them get there?
#
#     L = mean_i max(0, xi(a_yi, x_i) - psi_yi)                    pos hinge
#       + lambda_axis * mean_i d_ray(x_i, axis_yi)                 NEW
#       + lambda_neg  * mean_{k != yi} max(0, psi_k + m - xi(a_k, x_i))   neg hinge
#
# d_ray is the geodesic distance to the cone's AXIS RAY: the geodesic from the origin
# through the apex, restricted to the side the cone opens toward (geometry/lorentz.py,
# invariants in tests/test_axis_ray_dist.py). It exists because the hinge's gradient is
# exactly zero the moment a point is inside its cone, and d_ray's vanishes only ON the
# axis. Two things distinguish it from the previous attempts:
#
#   it READS THE RADIUS. The --loss axis score normalises both sides, so it measures an
#   angle at the ORIGIN and is invariant to depth. That is how the last run reported
#   inside=100% for five epochs while ||x_img||=0.009 sat against ||x_anc||=1.861 —
#   images ~200x SHALLOWER than their anchors, hence inside no cone at all, since an
#   entailment cone holds the points FARTHER from the origin than its apex. d_ray cannot
#   make that mistake, and --init_depth below plus the trainer's hard depth check make
#   the configuration impossible to start in.
#
#   its apex branch moves the anchor RADIALLY. Measured to the full geodesic the
#   distance is bilateral (theta=160 deg scores as theta=20 deg) and the antipode
#   becomes an attractor; measured to the ray from the ORIGIN it goes flat past 90 deg
#   and the anchor drops out entirely. From the APEX it is strictly monotone over all of
#   [0, pi] with a live gradient everywhere, and past the perpendicular foot it becomes
#   d_H(a, x), which pulls on the anchor's DEPTH. Nothing else in this loss does.
#
# Which matters because psi is COUPLED here, psi_k = asin(2K/||a_k||), and L_norm is
# GONE. Every previous run used L_norm as a one-sided floor whose only effect was to put
# all 22 anchors at the same norm, hence the same psi — and with equal psi, argmin xi IS
# argmax cos algebraically, which is the 0.9985-0.9998 cone-cosine agreement that has
# pinned every checkpoint so far. Here depth is set by tension instead: the positive
# hinge wants a wide cone (shallow anchor), the negative hinge wants a narrow one (deep
# anchor), and each class settles where its own data balances them.
# --anchor_norm_range replaces the penalty with a PROJECTION, so anchors move freely
# inside a wide band and are only clamped at the extremes (without any control at all,
# the pure-CE run blew up to ||x_anc||=658 with all 22 anchors on one direction).
#
# Constant LR, on purpose. Under the cosine schedule the 5-epoch runs gave epoch 1
# 38.7% of the total lr budget and epoch 5 1.3%, so a quantity that stopped moving was
# indistinguishable from one whose updates had gone to zero, and every final number
# recorded where the clock stopped. With a constant lr a plateau means equilibrium.
#
# Read out of the epoch line, in order of importance:
#   psi_anc range   must SPREAD. Equal psi across classes is exactly the condition under
#                   which the cone rule IS a cosine, however high the accuracy climbs.
#                   With L_norm gone and depth free, a spread is now possible for the
#                   first time; if it still collapses, that is a real finding.
#   shallow         fraction of images shallower than their own anchor. Must stay ~0.
#                   Anything else and those images are in oxy_angle's acos clamp, where
#                   the hinge gradient is exactly zero — the failure this run replaces.
#   xi_sat          the same failure seen from the other side, as a fraction of the
#                   whole xi matrix.
#   min-angle       the open problem: random init in 128-d starts at ~78.7 deg (mean
#                   ~89.9 deg, near-orthogonal). The last axis run was at 41.2 deg by the
#                   end of epoch 1 and 9.8 deg by epoch 5. Those are the numbers to beat.
#
# And out of stats.csv, which is the point of the run: min-angle, mean-angle, the psi
# spread and mean_anc_norm against STEP. The last collapse happened entirely inside
# epoch 1, so the epoch line could not see it; --log_every 10 can.
#
# Submit:  sbatch --export=ALL,RUN=coneaxis      slurm/slurm_train_22cls_cone_axisreg.sh
#          sbatch --export=ALL,RUN=coneaxis_off  slurm/slurm_train_22cls_cone_axisreg.sh
#
#   coneaxis      the run.
#   coneaxis_off  lambda_axis=0, everything else identical. The control: it isolates the
#                 regulariser from the other three changes (constant lr, free depth,
#                 correct depth ordering), which otherwise all land at once.
# ==========================================================================

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

# Depth budget, and why these three numbers go together:
#   --anchor_init_norm 2.0    ||t_anc|| = 2.0  -> ||x_anc|| = sinh 2   = 3.63, psi = 16.0 deg
#   --anchor_norm_range 1.0 2.5   ||x_anc|| may roam [1.18, 6.05] -> psi in [9.5, 58.4] deg
#   --init_depth 3.0          ||t_img|| = 3.0  -> ||x_img|| = sinh 3   = 10.02
# so the images start deeper than the DEEPEST reachable anchor (10.02 > 6.05) and stay
# there for the whole run. The trainer now refuses to start if that ordering is wrong,
# rather than training five epochs inside the acos clamp.
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
