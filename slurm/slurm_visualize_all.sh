#!/bin/bash
# ============================================================================
# Visualise hyperbolic embeddings (Poincaré disk via HoroPCA + 3-D UMAP).
#
# One-time setup (run on login node from $WORK/hyp_fine_tuning/hyperbolic_CLIP):
#   source $WORK/hyp_fine_tuning/bin/activate
#   pip install umap-learn matplotlib networkx
#   git clone https://github.com/HazyResearch/HoroPCA $WORK/hyp_fine_tuning/horopca
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=viz_all
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
export HOROPCA_DIR=${HOROPCA_DIR:-$WORK/hyp_fine_tuning/horopca}

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

# Embedding dimension; pass on the CLI, e.g.
#   sbatch slurm/slurm_visualize_all.sh 8
# (default 4). Must match the d used at training time so the right checkpoint is
# picked up — see slurm/slurm_cineca_all.sh, which writes attribution_all_no_dalle_d${DIM}.pt.
DIM=${1:-4}

# Override via env vars when needed, e.g. CKPT=... OUT=... sbatch slurm/slurm_visualize_all.sh
CKPT=${CKPT:-$WORK/checkpoints/attribution_all_no_dalle_d${DIM}.pt}
OUT=${OUT:-$WORK/viz/all_no_dalle_d${DIM}}
GENERATORS=${GENERATORS:-"real 4o gemini grok3 FLUX SD1_5 SD2_1 SD3 SD3_5 SDXL PIXART PLAYGROUND_2_5 KANDINSKY CogView3_PLUS hidream hunyuan ideogram infinity janus-pro kling mid-5.2 mid-6.0"}

python -m tests.visualize_horopca \
    --checkpoint    $CKPT \
    --dataset_path  $WORK/iab_dataset \
    --captions_dir  $WORK/hyp_fine_tuning/iab_captions \
    --generators    $GENERATORS \
    --semantics     COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split         val \
    --val_frac      0.2 \
    --max_per_class 500 \
    --batch_size    128 \
    --num_workers   4 \
    --output_dir    $OUT

echo "Plots saved to $OUT"
