#!/bin/bash
# ============================================================================
# CINECA Leonardo — Extract FROZEN CLIP image features (Fase A of the probe).
# Runs off-the-shelf CLIP ViT-L/14 (no LoRA) once over train+val and caches
# (features 768-d, label) to $WORK/clip_features/clip_features_{train,val}.pt.
# Extract once → then train_linear_probe.py many times on the cache (cheap).
#
# Same 22 classes / semantics / val_frac / seed as slurm_cineca_all.sh, so the
# val set matches the fine-tuned models' eval exactly (apples-to-apples).
#
# Smoke test first (fast): add  --max_per_class 50  to the python call.
#
# Submit:  sbatch slurm/slurm_extract_features.sh
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=extract_clip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
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
export HF_DATASETS_OFFLINE=1

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

python -m scripts.extract_clip_features \
    --dataset_path $WORK/iab_dataset \
    --captions_dir $WORK/hyp_fine_tuning/iab_captions \
    --clip_name    openai/clip-vit-large-patch14 \
    --generators   real 4o gemini grok3 FLUX \
                   SD1_5 SD2_1 SD3 SD3_5 SDXL \
                   PIXART PLAYGROUND_2_5 KANDINSKY CogView3_PLUS \
                   hidream hunyuan ideogram infinity janus-pro kling \
                   mid-5.2 mid-6.0 \
    --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --val_frac     0.2 \
    --seed         42 \
    --batch_size   256 \
    --num_workers  8 \
    --out_dir      $WORK/clip_features

echo "Done. Cache in $WORK/clip_features/"
