#!/bin/bash
# ============================================================================
# Extract embeddings + HoroPCA + UMAP visualisation.
#
# One-time setup (run on login node first):
#   pip install umap-learn
#   git clone https://github.com/HazyResearch/HoroPCA \
#       $WORK/hyp_fine_tuning/hyperbolic_CLIP/external/HoroPCA
#   pip install -r $WORK/hyp_fine_tuning/hyperbolic_CLIP/external/HoroPCA/requirements.txt
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=visualize_horopca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
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

CKPT=$WORK/checkpoints/attribution_FLUX_vitl14_hier.pt
EMB=$WORK/hyp_fine_tuning/embeddings/val_hier.npz
FIG_DIR=$WORK/hyp_fine_tuning/figures

mkdir -p $WORK/hyp_fine_tuning/embeddings $FIG_DIR

# ── 1. Extract embeddings (val split, image-only) ────────────────────────────
python -m tests.extract_embeddings \
    --checkpoint   $CKPT \
    --dataset_path $WORK/iab_dataset \
    --captions_dir $WORK/hyp_fine_tuning/iab_captions \
    --generators   real FLUX \
    --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split        val \
    --val_frac     0.2 \
    --batch_size   256 \
    --num_workers  4 \
    --output       $EMB

# ── 2. HoroPCA + UMAP, coloured by class (real vs FLUX) ──────────────────────
python -m tests.visualize_horopca \
    --embeddings $EMB \
    --output     $FIG_DIR/val_hier_horopca_by_class.png \
    --n_pca      8 \
    --color_by   class

# ── 3. Same, coloured by semantic class ──────────────────────────────────────
python -m tests.visualize_horopca \
    --embeddings $EMB \
    --output     $FIG_DIR/val_hier_horopca_by_semantic.png \
    --n_pca      8 \
    --color_by   semantic
