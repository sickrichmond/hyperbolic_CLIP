#!/bin/bash
# CINECA Leonardo — IABench, random anchors, ViT-L/14.
# All generator labels are discovered; the trainer creates/reuses an 80/10/10
# manifest shared by clean and augmented training. Validation stays clean.
# AUGMENT=1 enables corruption (default) or omnidfa via AUG_POLICY.
# Submit: sbatch slurm/slurm_train_iabench.sh
#         sbatch --export=ALL,AUGMENT=1 slurm/slurm_train_iabench.sh
#         sbatch --export=ALL,AUGMENT=1,AUG_POLICY=omnidfa slurm/slurm_train_iabench.sh
# Timing: PROFILE_STEPS=100 sbatch slurm/slurm_train_iabench.sh

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
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$OUT/attribution_iabench_random_vitl14.splits.json}"
PROFILE_STEPS="${PROFILE_STEPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AUGMENT="${AUGMENT:-0}"
AUG_POLICY="${AUG_POLICY:-corruption}"

case "$AUGMENT" in
    0|1) ;;
    *) echo "ERROR: AUGMENT must be 0 or 1 (got: $AUGMENT)" >&2; exit 2 ;;
esac
case "$AUG_POLICY" in
    corruption|omnidfa) ;;
    *) echo "ERROR: AUG_POLICY must be corruption or omnidfa (got: $AUG_POLICY)" >&2; exit 2 ;;
esac

AUG_FLAGS=()
AUG_SUFFIX=
if [ "$AUGMENT" = 1 ]; then
    AUG_FLAGS=(--train_augment --aug_policy "$AUG_POLICY")
    if [ "$AUG_POLICY" = omnidfa ]; then
        AUG_SUFFIX=_omniaug
    else
        AUG_SUFFIX=_aug
    fi
fi
CKPT="$OUT/attribution_iabench_random${AUG_SUFFIX}_vitl14.pt"

mkdir -p "$OUT"
cd "$REPO"

echo "Slurm GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Training with AUGMENT=$AUGMENT, AUG_POLICY=$AUG_POLICY → $CKPT"

python train_iabench.py \
    --dataset_path "$DATA" \
    --clip_name openai/clip-vit-large-patch14 \
    --anchor_init random \
    "${AUG_FLAGS[@]}" \
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
    --num_workers "$NUM_WORKERS" \
    --profile_steps "$PROFILE_STEPS" \
    --split_manifest "$SPLIT_MANIFEST" \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --seed 42 \
    --diag_plot_dir "$WORK/hyp_fine_tuning/viz/iabench${AUG_SUFFIX}_${SLURM_JOB_ID}" \
    --log_every 10 \
    --output "$CKPT"

echo "Done: $CKPT"
