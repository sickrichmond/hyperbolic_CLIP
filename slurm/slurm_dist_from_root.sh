#!/bin/bash
# ============================================================================
# CINECA Leonardo — distance-from-root distributions (HySAC Fig.4 analogue).
# Writes <OUT>_per_class.png and <OUT>_real_vs_fake.png.
#
# Submit (DIM picks the checkpoint, mirrors slurm_cineca_all.sh):
#   sbatch slurm/slurm_dist_from_root.sh 16
#
# Override, e.g. all semantics instead of just COCO:
#   SEMANTICS="COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k" \
#       sbatch slurm/slurm_dist_from_root.sh 16
# ============================================================================

#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=dist_from_root
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

DIM=${1:-16}
CKPT=${CKPT:-$WORK/checkpoints/attribution_all_no_dalle_d${DIM}.pt}
DATA=${DATA:-$WORK/iab_dataset}
SEMANTICS=${SEMANTICS:-COCO}
MAX_PER_CLASS=${MAX_PER_CLASS:-300}
OUT=${OUT:-$WORK/outputs/dist_from_root/d${DIM}}

python -m tests.plot_distance_from_root \
    --checkpoint    $CKPT \
    --dataset_path  $DATA \
    --semantics     $SEMANTICS \
    --max_per_class $MAX_PER_CLASS \
    --output        $OUT

echo "Distance-from-root figures → ${OUT}_per_class.png / ${OUT}_real_vs_fake.png"
