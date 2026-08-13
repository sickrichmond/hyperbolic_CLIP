#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, STRUCTURAL TEXT ANCHORS. Three runs, each
# one variable from the previous, all on the ours-sweepwin recipe (lr 3e-4,
# lora 16/32, min_radius 0.5, margin 0.3, lambda_neg 1.0, lambda_norm 0.5 /
# target_norm 4.0, 5 epochs, base loss, full manifest).
#
# Why: the default anchors are two templates, so 21 of the 22 sentences differ by
# one token and CLIP embeds them nearly collinearly — the cones start on top of
# each other. data/anchor_prompts_structural.json varies verb, voice and syntax
# per class instead. Measure a set first:  python -m tests.probe_anchor_prompts
#
#   RUN=A  prompts only. Anchors are still re-encoded every step and move through
#          the text-encoder LoRA. ONE variable from ours-sweepwin.
#   RUN=B  --anchor_init text_free: free tangent parameters initialised at the
#          encoded prompts, drifting by t = t0 + softplus(s)·δ with δ init 0 and
#          s a single LEARNED scalar. One variable from A.
#   RUN=C  --lambda_ce: CE on softmax(-ξ/τ) with τ learned, on top of the cone
#          hinge. Attacks the AUC-holds/accuracy-collapses gap, not the anchor
#          geometry. One variable from A. Inference is unchanged.
#
# ~2.5h each on 2 GPUs.
#
# Submit:  sbatch --export=ALL,RUN=A slurm/slurm_train_22cls_anchors.sh
#          sbatch --export=ALL,RUN=B slurm/slurm_train_22cls_anchors.sh
#          sbatch --export=ALL,RUN=C slurm/slurm_train_22cls_anchors.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_anchors
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
PROMPTS=$REPO/data/anchor_prompts_structural.json

RUN=${RUN:-A}
case "$RUN" in
  A) EXTRA="--anchor_init text"
     CKPT=$OUT/attribution_22cls_promptsA_vitl14.pt ;;
  B) EXTRA="--anchor_init text_free --anchor_init_norm 2.0 --anchor_drift_init 0.1"
     CKPT=$OUT/attribution_22cls_promptsB_free_vitl14.pt ;;
  C) EXTRA="--anchor_init text --lambda_ce 1.0 --ce_tau_init 1.0"
     CKPT=$OUT/attribution_22cls_promptsC_ce_vitl14.pt ;;
  *) echo "RUN must be A, B or C (got '$RUN')"; exit 1 ;;
esac

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
    --anchor_prompts  $PROMPTS \
    $EXTRA \
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
    --batch_size      256 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
