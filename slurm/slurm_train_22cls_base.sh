#!/bin/bash
# ============================================================================
# CINECA Leonardo — ours @ 22 CLASSES, PURE BASE LOSS (no captions)
# CLIP ViT-L/14, LoRA, hyperbolic entailment-cone loss — image-in-class-cone only.
#
# Changes vs the 23-class run (slurm_train_23cls_fair.sh):
#   - IAB_EXCLUDE_GENERATORS=dalle3  → whole pipeline runs at 22 classes.
#   - --no_captions  → disables BOTH caption terms → trains on ALL images (no
#     caption requirement) = same sample set as the baselines.
#   - manifest split_manifest_22cls.json (regenerate it first, see below), and
#     ours validates on the HARNESS VAL split (no double val carve).
#   - --generators WITHOUT dalle3.
#
# PREREQUISITE — regenerate the 22-class manifest once (login node, cheap):
#   IAB_EXCLUDE_GENERATORS=dalle3 python -m comparison.training.scripts.dump_split_manifest \
#       --root_dir $FAST/datasets/iab_dataset \
#       --out $WORK/hyp_fine_tuning/split_manifest_22cls.json
#
# Submit:  sbatch slurm/slurm_train_22cls_base.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189          # verify with `saldo -b`
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=attr_22cls_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
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

mkdir -p $OUT
cd $REPO

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 22-class manifest not found at $MANIFEST"
    echo "Regenerate with IAB_EXCLUDE_GENERATORS=dalle3 (see header)."
    exit 1
fi

# 22 generatori (dalle3 escluso) + real, pure base loss (--no_captions), tutte le immagini,
# selezione sulla val dell'harness. Iperparametri = quelli tuned (poi lo sweep li cerca).
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
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
    --lr              5e-5 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $OUT/attribution_22cls_base_vitl14.pt

echo "Done: $OUT/attribution_22cls_base_vitl14.pt"
