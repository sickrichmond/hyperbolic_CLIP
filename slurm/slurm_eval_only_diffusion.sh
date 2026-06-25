#!/bin/bash
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_only_diffusion
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

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

# Overridable via env var (defaults = original IAB dataset + diffusion checkpoint).
# To eval the OLD diffusion checkpoint on the NEW regenerated fakes:
#   DATA=$WORK/hyp_fine_tuning/iab_recap_dataset_v2 sbatch slurm/slurm_eval_only_diffusion.sh
# (the new root holds only the fakes; symlink the reals in first — see README/notes:
#   ln -s ../iab_dataset/real $WORK/hyp_fine_tuning/iab_recap_dataset_v2/real)
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_diffusion.pt}
DATA=${DATA:-$WORK/hyp_fine_tuning/iab_dataset}
CAPS=${CAPS:-$WORK/hyp_fine_tuning/iab_captions}
SPLIT=${SPLIT:-val}
VAL_FRAC=${VAL_FRAC:-0.2}

python -m tests.eval_attribution \
    --checkpoint   "$CKPT" \
    --dataset_path "$DATA" \
    --captions_dir "$CAPS" \
    --generators   real SD3 SD3_5 SDXL FLUX \
    --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split        "$SPLIT" \
    --val_frac     "$VAL_FRAC" \
    --batch_size   256 \
    --num_workers  4
