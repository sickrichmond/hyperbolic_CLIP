#!/bin/bash
# ============================================================================
# CINECA Leonardo — EUCLIDEAN BASELINE, 22 classes. The geometry ablation.
#
# One variable from `ours-sweepwin`: SAME backbone (ViT-L/14), SAME LoRA (16/32),
# SAME embedding width (128), SAME 22 classes, semantics, manifest, batch size,
# epochs, LR and weight decay. The only difference is that embeddings live on the
# Euclidean unit sphere and images are matched to the text anchors by cosine
# similarity (a trainable zero-shot CLIP) instead of by hyperbolic entailment
# cones. The gap against attribution_22cls_sweepwin_vitl14.pt IS the contribution
# of the geometry — the ablation the whole premise of the paper rests on.
#
# The cone / norm / caption hyperparameters have no analogue on the sphere and are
# intentionally absent (see losses/euclidean_attribution_loss.py). What replaces
# them is a single learned logit_scale, CLIP-style.
#
# --split_manifest is NOT optional here: without it this would train on the
# caption-based split while the hyperbolic run trains on the harness manifest, and
# the comparison would carry two variables instead of one.
#
# ~2.5h on 2 GPUs. Eval afterwards with slurm/slurm_eval_euclidean.sh.
#
# Submit:  sbatch slurm/slurm_euclidean_baseline.sh
#          sbatch --export=ALL,DIM=8 slurm/slurm_euclidean_baseline.sh   # low-d sweep
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_22cls_eucl
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

# Embedding width. 128 matches the hyperbolic run's --hyperbolic_dim; smaller values
# are the low-dimension sweep (hyperbolic geometry is supposed to win most at small d).
DIM=${DIM:-128}

mkdir -p $OUT
cd $REPO

CUDA_VISIBLE_DEVICES=0,1 python train_attribution_euclidean.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --lora_r          16 \
    --lora_alpha      32 \
    --embed_dim       $DIM \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $OUT/attribution_22cls_euclidean_d${DIM}_vitl14.pt

echo "Done: $OUT/attribution_22cls_euclidean_d${DIM}_vitl14.pt"
