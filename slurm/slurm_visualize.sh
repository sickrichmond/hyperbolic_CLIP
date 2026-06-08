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
#SBATCH --job-name=viz_attribution
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
export HOROPCA_DIR=${HOROPCA_DIR:-$WORK/hyp_fine_tuning/horopca}

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

# Override via env vars when needed, e.g. CKPT=... OUT=... sbatch scripts/slurm_visualize.sh
CKPT=${CKPT:-$WORK/checkpoints/attribution_k4_vitl14.pt}
OUT=${OUT:-$WORK/viz/k4_hier}
GENERATORS=${GENERATORS:-"real FLUX SD3 gemini"}

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
