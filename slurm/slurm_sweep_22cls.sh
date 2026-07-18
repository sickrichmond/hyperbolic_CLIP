#!/bin/bash
# ============================================================================
# CINECA Leonardo — HypCLIP hyperparameter SWEEP @ 22 classes, base loss.
# SLURM job array: one training per line of slurm/sweep_configs_22cls.txt.
# Short (3 epochs) for ranking; retrain the winner longer afterwards.
#
# Each task selects on the HARNESS VAL split and saves val_balanced in its .pt.
# Rank them with: python comparison/training/scripts/collect_sweep.py \
#     --dir $WORK/hyp_fine_tuning/checkpoints/sweep --configs slurm/sweep_configs_22cls.txt
#
# PREREQUISITE: 22-class manifest (split_manifest_22cls.json). See slurm_train_22cls_base.sh.
# Submit:  sbatch slurm/slurm_sweep_22cls.sh
# The array size (0-11) must match the number of lines in sweep_configs_22cls.txt.
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=hyp_sweep_22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4               # 12 configs, up to 4 concurrent
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate
export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json
OUT=$WORK/hyp_fine_tuning/checkpoints/sweep
CONFIGS=$REPO/slurm/sweep_configs_22cls.txt
mkdir -p $OUT
cd $REPO

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIGS")
echo "=== sweep task $SLURM_ARRAY_TASK_ID ==="
echo "config: $LINE"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --no_captions \
    --batch_size      256 \
    --num_epochs      3 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    $LINE \
    --output          $OUT/sweep_${SLURM_ARRAY_TASK_ID}.pt

echo "Done task $SLURM_ARRAY_TASK_ID -> $OUT/sweep_${SLURM_ARRAY_TASK_ID}.pt"
