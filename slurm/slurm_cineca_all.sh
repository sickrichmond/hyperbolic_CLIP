#!/bin/bash
# CINECA Leonardo — 22-class cone training with caption terms.
#
# Uses all configured classes except dalle3 and the internal 80/20 split.
# The positional dimension argument defaults to 4 and enters the checkpoint
# filename. This recipe does not use a comparison-harness manifest.
#
# Submit: sbatch slurm/slurm_cineca_all.sh 4

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=attr_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache          # avoid filling home quota
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP
DATA=$WORK/hyp_fine_tuning/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/hyp_fine_tuning/checkpoints
DIM=${1:-4}                 # embedding dimension; pass on the CLI, e.g.
                            #   sbatch slurm/slurm_cineca_all.sh 8
                            # The dimension is included in the checkpoint name.

mkdir -p $OUT
cd $REPO

# ── Training ──────────────────────────────────────────────────────────────────
# 22-way attribution with hyperbolic entailment cones. Each image is pulled into
# the cone of its class anchor and out of the other 21 cones. 80/20 split per
# (generator, semantic).
#
#
# The best checkpoint (by balanced val accuracy) is saved every time val
# improves, so even if the job hits the walltime you keep the best-so-far model.

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o gemini grok3 FLUX \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL \
                      PIXART PLAYGROUND_2_5 KANDINSKY CogView3_PLUS \
                      hidream hunyuan ideogram infinity janus-pro kling \
                      mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  $DIM \
    --curv            1.0 \
    --min_radius      0.5 \
    --margin          0.3 \
    --lambda_neg      1.0 \
    --lambda_cap_in_class 1.0 \
    --lambda_img_in_cap   0.5 \
    --lambda_norm     0.5 \
    --target_norm     4.0 \
    --batch_size      256 \
    --num_epochs      8 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --val_frac        0.2 \
    --num_workers     8 \
    --output          $OUT/attribution_all_no_dalle_d${DIM}.pt

echo "Done: $OUT/attribution_all_no_dalle_d${DIM}.pt"
