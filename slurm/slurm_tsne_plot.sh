#!/bin/bash
# CINECA Leonardo. Submit: sbatch slurm/slurm_tsne_plot.sh /path/to/checkpoint.pt
#
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=tsne_plot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "Usage: sbatch slurm/slurm_tsne_plot.sh /path/to/checkpoint.pt" >&2
    exit 2
fi
CKPT=$(realpath "$1")
CKPT_STEM=${CKPT##*/}
CKPT_STEM=${CKPT_STEM%.*}

module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

export HF_HOME="$WORK/hyp_fine_tuning/hf_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=${OUT:-$WORK/hyp_fine_tuning/outputs/tsne/$CKPT_STEM}

cd "$REPO"

python -m explanation.tsne_plot \
    --checkpoint "$CKPT" \
    --dataset_path "$DATA" \
    --captions_dir "$CAPS" \
    --batch_size 128 \
    --num_workers 8 \
    --output_dir "$OUT"

echo "Saved $OUT/tsne_plot.png"
