#!/bin/bash
# ============================================================================
# CINECA Leonardo — FREQUENCY / DEGRADATION DIAGNOSTIC (no training).
# Runs comparison.training.diag_frequency on the current 22-class base-loss
# checkpoint: a Gaussian-blur sweep (sigma 0.5..5) + a JPEG-quality ramp
# (q 90..30) over the SAME clean test images, plus error-routing analysis
# (-> real / -> same family / -> cross family) via the HiFi family hierarchy.
#
# Built-in sanity check: blur3.0/blur5.0 and jpeg65/jpeg30 must reproduce the
# benchmark eval numbers (~0.531 / 0.203 / 0.125 / 0.096). If they instead equal
# the clean accuracy, the degradation monkeypatch/globals didn't propagate to the
# DataLoader workers (needs the Linux 'fork' start method — the default here).
#
# CKPT defaults to the 22-class base-loss run; override with CKPT=... sbatch ...
# Submit:  sbatch slurm/slurm_diag_frequency.sh
# ============================================================================
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=diag_freq_22cls
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
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
export IAB_EXCLUDE_GENERATORS=dalle3        # <-- 22-class (anchors + test set)

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CKPT=${CKPT:-$WORK/hyp_fine_tuning/checkpoints/attribution_22cls_base_vitl14.pt}
LOGDIR=$WORK/outputs/hypclip_diag_22cls

mkdir -p $LOGDIR
cd $REPO

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"; exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m comparison.training.diag_frequency \
    --checkpoint $CKPT \
    --root_dir   $DATA \
    --batch_size 64 \
    --num_workers 8 \
    --log_dir    $LOGDIR

echo "Done. Results in $LOGDIR/diag_frequency.{txt,json}"
