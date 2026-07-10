#!/bin/bash
# ============================================================================
# CINECA Leonardo — Image-only attribution eval under the SAME 7-level
# degradation pipeline used for the comparison baselines (HiFi-Net, DCT-CNN, …).
# Loops degraded levels 0..6 (0=clean, 1/2=DS, 3/4=JPEG, 5/6=Blur) and writes
# one report per level. Same checkpoint/generators/semantics as slurm_eval_all.sh.
#
# Submit:  sbatch slurm/slurm_eval_all_degraded.sh [DIM]
#   e.g.   sbatch slurm/slurm_eval_all_degraded.sh 4
#   override checkpoint:  CKPT=/path/to/model.pt sbatch slurm/slurm_eval_all_degraded.sh
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_all_degraded
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
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

# Embedding dimension from CLI (default 4). Override CKPT for the legacy d=128 file.
DIM=${1:-4}
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_all_no_dalle_d${DIM}.pt}

OUTDIR=$WORK/hyp_fine_tuning/eval_degraded/$(basename "${CKPT%.pt}")
mkdir -p "$OUTDIR"
echo "Checkpoint: $CKPT"
echo "Writing per-level reports to: $OUTDIR"

for D in 0 1 2 3 4 5 6; do
    echo "===== degraded level $D ====="
    python -m tests.eval_attribution \
        --checkpoint   "$CKPT" \
        --dataset_path $WORK/hyp_fine_tuning/iab_dataset \
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
        --num_workers  "${SLURM_CPUS_PER_TASK:-8}" \
        --degraded     "$D" \
        > "$OUTDIR/eval_degraded_${D}.txt" 2>&1
    echo "  -> $OUTDIR/eval_degraded_${D}.txt"
done

echo "Done. Reports under $OUTDIR"
