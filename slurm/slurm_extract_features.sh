#!/bin/bash
# ============================================================================
# CINECA Leonardo — Phase A of the linear probe: cache image features once.
#
# Three sources, selected by SOURCE=, all written to their own directory and all
# consumed by the same slurm_train_probe.sh. Together they answer "how much of the
# 0.993 is CLIP, how much is the LoRA, how much is the geometry":
#
#   SOURCE=frozen      off-the-shelf CLIP ViT-L/14, no LoRA        (the baseline)
#   SOURCE=lora        the trained LoRA CLIP embedding             (CKPT required)
#   SOURCE=projection  the tangent vectors out of the projection   (CKPT required)
#
# --split_manifest puts all three on the SAME images as the results tables. Without
# it the split is the legacy caption-based one (94,673 val images) and the numbers
# are not comparable with anything.
#
# A euclidean CKPT works too (extract_clip_features branches on ckpt['geometry']) —
# that is the control for whether the projection head or the saturating hinge is what
# collapses the class geometry. Give it its own OUT_DIR or it overwrites the
# hyperbolic cache of the same SOURCE.
#
# Submit:  sbatch --export=ALL,SOURCE=frozen slurm/slurm_extract_features.sh
#          sbatch --export=ALL,SOURCE=lora,CKPT=$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_sweepwin_vitl14.pt \
#                 slurm/slurm_extract_features.sh
#          sbatch --export=ALL,SOURCE=projection,CKPT=$CK/attribution_22cls_euclidean_d128_vitl14.pt,\
# OUT_DIR=$WORK/hyp_fine_tuning/clip_features_projection_eucl slurm/slurm_extract_features.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=extract_clip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
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
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json
FEAT=$WORK/hyp_fine_tuning/clip_features

SOURCE=${SOURCE:-frozen}
case "$SOURCE" in
  frozen)     EXTRA="" ;;
  lora)       EXTRA="--checkpoint $CKPT --features clip" ;;
  projection) EXTRA="--checkpoint $CKPT --features projection" ;;
  *) echo "SOURCE must be frozen, lora or projection (got '$SOURCE')"; exit 1 ;;
esac

cd $REPO

python -m scripts.extract_clip_features \
    --dataset_path   $DATA \
    --captions_dir   $CAPS \
    --clip_name      openai/clip-vit-large-patch14 \
    --generators     real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                     SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                     ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics      COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split_manifest $MANIFEST \
    $EXTRA \
    --batch_size     256 \
    --num_workers    8 \
    --out_dir        ${OUT_DIR:-${FEAT}_${SOURCE}}

echo "Done. Cache in ${OUT_DIR:-${FEAT}_${SOURCE}}/"
