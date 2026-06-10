#!/bin/bash
# ============================================================================
# CINECA Leonardo — Evaluate the EUCLIDEAN BASELINE checkpoint (image-only).
# Mirror of slurm/slurm_eval.sh; reports the same metrics so it lines up
# directly against the hyperbolic eval. Keep --generators / --semantics /
# --val_frac identical to the hyperbolic run being compared against.
#
# Submit:  sbatch slurm/slurm_eval_euclidean.sh
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_eucl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

# Dimension from the CLI (default 4): sbatch slurm/slurm_eval_euclidean.sh 8
# Override the full path with CKPT=... to eval e.g. the legacy d=128 file
# (attribution_all_euclidean.pt, no _d suffix).
DIM=${1:-4}
CKPT=${CKPT:-$WORK/checkpoints/attribution_all_euclidean_d${DIM}.pt}

python -m tests.eval_attribution_euclidean \
    --checkpoint   $CKPT \
    --dataset_path $WORK/iab_dataset \
    --captions_dir $WORK/hyp_fine_tuning/iab_captions \
    --generators   real 4o gemini grok3 FLUX \
                   SD1_5 SD2_1 SD3 SD3_5 SDXL \
                   PIXART PLAYGROUND_2_5 KANDINSKY CogView3_PLUS \
                   hidream hunyuan ideogram infinity janus-pro kling \
                   mid-5.2 mid-6.0 \
    --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split        val \
    --val_frac     0.2 \
    --batch_size   256 \
    --num_workers  4
