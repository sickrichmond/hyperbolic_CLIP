#!/bin/bash
# ============================================================================
# CINECA Leonardo — 22 classes, the OPTUNA WINNER (study hypclip_22cls, trial #38).
#
# 62 trials with a value (42 complete, 20 pruned), best 0.9950 clean val_balanced,
# against 0.9886 for the hand-tuned sweepwin recipe. Four trials tied at 0.9950
# (#38, #49, #53, #57); #38 is the one Optuna reports as best.
#
# The trials threw their checkpoints away (they wrote to $TMPDIR), so the winner has
# to be retrained once to exist as an artifact. Values below are the trial's arguments
# verbatim, at the precision optuna_search.py formats them with (%.4g, lr %.6g), so
# this reproduces trial #38 exactly rather than approximately.
#
# What the search moved, relative to sweepwin — it LOOSENED every hyperbolic-specific
# constraint: min_radius 0.5 -> 0.9921 and target_norm 4.0 -> 3.892 more than double the
# cone half-aperture (ψ ≈ 0.52 rad against 0.246), and lambda_norm halves from 0.5 to
# 0.25. lora_r drops 16 -> 8 while lr doubles. Worth measuring after this run:
#   python -m tests.probe_open_set --anchors_only $CKPT     # is ψ still uniform?
#   python -m tests.probe_cone_vs_cosine $CKPT              # still 0.9998 agreement?
#
# Submit:  sbatch slurm/slurm_train_22cls_optunawin.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_optunawin
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
CKPT=$OUT/attribution_22cls_optunawin_vitl14.pt

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
    --lora_r          8 \
    --lora_alpha      16 \
    --hyperbolic_dim  128 \
    --curv            1.131 \
    --min_radius      0.9921 \
    --margin          0.2577 \
    --lambda_neg      1.498 \
    --lambda_norm     0.25 \
    --target_norm     3.892 \
    --no_captions \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              0.000613738 \
    --weight_decay    0.003111 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
