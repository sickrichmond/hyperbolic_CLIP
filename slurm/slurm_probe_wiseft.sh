#!/bin/bash
# ============================================================================
# CINECA Leonardo — WiSE-FT PROBE (no training).
# Walks the weight-space path between the frozen CLIP (alpha=0) and the LoRA
# fine-tuned model (alpha=1) by scaling every LoraLayer's `scaling`, and measures
# clean + JPEG accuracy at each alpha. Tests whether the JPEG fragility was
# introduced by the adapter (shortcut on pristine-PNG statistics) rather than
# being intrinsic to CLIP features.
#
# Read the RETENTION table (acc_degraded / acc_clean), not absolute accuracy:
# the projection head was trained at alpha=1, so absolutes necessarily fall as
# alpha decreases. Rising retention = hypothesis supported.
#
# NOTE: this probes the CURRENT (pre-bugfix) 22-class checkpoint, which was
# trained without the `real` class. That is fine for this question — it is about
# feature fragility, not about the label space — but re-run it on the corrected
# checkpoint once the 22-class retrainings finish.
#
# Submit:  sbatch slurm/slurm_probe_wiseft.sh
# Override the checkpoint with:  CKPT=/path/to.pt sbatch slurm/slurm_probe_wiseft.sh
# ============================================================================
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=wiseft_22cls
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
export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_base_vitl14.pt}
LOGDIR=$WORK/outputs/hypclip_wiseft_22cls

mkdir -p $LOGDIR
cd $REPO

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"; exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m comparison.training.probe_wiseft \
    --checkpoint  $CKPT \
    --root_dir    $DATA \
    --alphas      1.0 0.9 0.75 0.5 0.25 0.0 \
    --scope       both \
    --max_samples 8000 \
    --batch_size  64 \
    --num_workers 8 \
    --log_dir     $LOGDIR

echo "Done. Results in $LOGDIR/probe_wiseft.{txt,json}"
