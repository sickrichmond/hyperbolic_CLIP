#!/bin/bash
# ============================================================================
# CINECA Leonardo — OPTUNA hyperparameter search, 22 classes, text anchors.
#
# Four array tasks = four workers against ONE shared journal file, so they
# coordinate: each pulls the next trial from the same study. Same 2-GPU/lprod
# shape as every other HypCLIP job (whole-node starves on this cluster).
#
# Each trial is a 5-epoch training (~2h18m measured). 4 workers x 24h / 2.3h
# ~= 40 full trials, more once the MedianPruner starts killing the collapses.
# Trials write their throwaway checkpoint to $TMPDIR — only the hyperparameters
# are kept, the winner is retrained afterwards (as sweepwin was).
#
# Prereq: optuna in the venv (or in $HOME/iab_pydeps + PYTHONPATH, see below).
#
# Submit:  sbatch --array=0-3 slurm/slurm_optuna_22cls.sh
# Ranking: python -m scripts.optuna_search --storage $STORAGE --report
# ============================================================================

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=optuna_22cls
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3      # <-- 22-class toggle (whole pipeline)

# Only needed if the venv's site-packages is read-only again:
#   pip install --no-cache-dir --target $HOME/iab_pydeps optuna
export PYTHONPATH="${IAB_PYDEPS:-$HOME/iab_pydeps}:$PYTHONPATH"

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json
STORAGE=$WORK/hyp_fine_tuning/optuna/hypclip_22cls.log

cd $REPO

# Stop launching NEW trials 2.5h before the walltime, so the last one finishes
# instead of being killed mid-epoch and recorded as a failure.
TIMEOUT=$(( 24*3600 - 9000 ))

CUDA_VISIBLE_DEVICES=0,1 python -m scripts.optuna_search \
    --storage        $STORAGE \
    --study_name     hypclip_22cls \
    --dataset_path   $DATA \
    --captions_dir   $CAPS \
    --split_manifest $MANIFEST \
    --clip_name      openai/clip-vit-large-patch14 \
    --num_epochs     5 \
    --batch_size     256 \
    --num_workers    8 \
    --timeout        $TIMEOUT

echo "Done. Ranking: python -m scripts.optuna_search --storage $STORAGE --report"
