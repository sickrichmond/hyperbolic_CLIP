#!/bin/bash
# CINECA Leonardo — IABench, random anchors, ViT-L/14, clean training.
# All generator labels are discovered; the trainer creates/reuses an 80/10/10
# manifest beside the checkpoint. Requires datasets and a cached CLIP backbone.
# Submit: sbatch slurm/slurm_train_iabench.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_iabench_random
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -e
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO="$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo"
DATA=/leonardo_scratch/large/userexternal/imaljkov/datasets/IABench/data
OUT="$WORK/hyp_fine_tuning/checkpoints"
CKPT="$OUT/attribution_iabench_random_vitl14.pt"

mkdir -p "$OUT"
cd "$REPO"

echo "Slurm GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"

python train_iabench.py \
    --dataset_path "$DATA" \
    --clip_name openai/clip-vit-large-patch14 \
    --anchor_init random \
    --lora_target 'vision_model\.encoder\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)' \
    --lora_r 16 \
    --lora_alpha 32 \
    --hyperbolic_dim 128 \
    --curv 1.0 \
    --min_radius 0.5 \
    --margin 0.3 \
    --lambda_neg 1.0 \
    --lambda_norm 0.5 \
    --target_norm 4.0 \
    --lambda_cosine 0.5 \
    --init_depth 3.0 \
    --batch_size 256 \
    --num_epochs 20 \
    --lr 3e-4 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --lr_schedule cosine \
    --num_workers 8 \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --seed 42 \
    --diag_plot_dir "$WORK/hyp_fine_tuning/viz/iabench_${SLURM_JOB_ID}" \
    --log_every 10 \
    --output "$CKPT"

echo "Done: $CKPT"
