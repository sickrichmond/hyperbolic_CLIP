#!/bin/bash
# ============================================================================
# CINECA Leonardo — the axis-distance cone loss (losses/axis_cone_loss.py).
#
# One idea: an image belongs ON THE AXIS of its class's cone and OUTSIDE every other
# class's cone.
#
#     q_ik = ‖x̂ᵢ − ûₖ‖² / ‖wallₖ‖²  =  (1 − cos θ) / (1 − cos ψₖ)
#     L    = mean_i [ q_i,yi  +  λ_neg · mean_k≠yi max(0, 1 − q_ik)² ]
#
# q = 0 on the axis, 1 exactly on the cone WALL, > 1 outside. So the negative term needs
# no margin: the margin IS the wall. The aperture is not tuned either — the positive term
# widens the cone (contain your own class) and the negative narrows it (stop swallowing
# the others), and it settles where they balance. Everything is algebraic: no arccos, no
# acosh, no asin, hence no clamp for a gradient to die on, which is how both previous
# formulations failed.
#
# The anchors are free parameters in tangent space, random directions, and the aperture
# psi is a SEPARATE free parameter per class, bounded to --psi_range by a sigmoid.
# Decoupling them is not cosmetic: deriving psi from the anchor's depth (sin psi = 2K/‖a‖)
# makes "widen my cone" and "move toward the origin" the same action, and the optimiser
# takes that scalar shortcut over the 128-dimensional rotation every time. Measured over
# five epochs with the coupled version: psi 53.3° -> 65.0° monotone, the anchor norm pinned
# at its floor, and the psi SPREAD across classes collapsing 8.7° -> 0.6° — which is exactly
# when argmin q becomes argmax cos. Decoupled, the radial gradient on the anchor is exactly
# zero and the trainer keeps ‖a‖ = 2K/sin psi so the stored anchor is still the point it
# represents. Where psi settles is set by the log-W term, whose equilibrium puts each cone's
# wall on the RMS angular radius of its own class.
#
# LoRA is on the upper half of the vision encoder only (layers 12-23, 24 adapters instead
# of 72) and nothing on the text side: with free anchors the text encoder is out of the
# objective entirely. Optimiser is SGD; note --lr does not carry over from the AdamW runs.
#
#   RUN=axis       SGD, lr 1e-2.
#   RUN=axis_adam  the same loss under AdamW at its own lr. The control: without it a bad
#                  axis run cannot be told apart from an untuned SGD lr.
#
# Read out of the epoch line, in order of importance:
#   ψ∈[min,max]  must SPREAD. If the apertures stay equal then argmin q IS argmax cos,
#                algebraically, however high the accuracy climbs — that is the 0.9985
#                cone-cosine agreement that has pinned every run so far.
#   q_pos        must fall, and `inside` rise.
#   min∠         must not collapse (the previous loss went 78.7° → 9.2° in two epochs).
#
# Submit:  sbatch --export=ALL,RUN=axis      slurm/slurm_train_22cls_axis.sh
#          sbatch --export=ALL,RUN=axis_adam slurm/slurm_train_22cls_axis.sh
# ==========================================================================

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
  axis)      EXTRA="--optimizer sgd   --lr 1e-2" ;;
  axis_adam) EXTRA="--optimizer adamw --lr 3e-4" ;;
  *) echo "RUN must be axis or axis_adam (got '$RUN')"; exit 2 ;;
esac
CKPT=$OUT/attribution_22cls_${RUN}_vitl14.pt

mkdir -p $OUT
cd $REPO

# No --lambda_sep and no --lambda_norm: this loss has neither term. The separation comes
# out of the negative term instead of a floor that fought the aperture, and the radius is
# constrained exactly by the projection rather than approximately by a penalty.
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
    --psi_range       5.0 60.0 \
    --lambda_aperture 1.0 \
    --lambda_neg      1.0 \
    --neg_samples     8 \
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
