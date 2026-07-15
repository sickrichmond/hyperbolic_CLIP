#!/bin/bash
# ============================================================================
# CINECA Leonardo — 23-CLASS FAIR evaluation of the hyperbolic-CLIP attributor
#
# Evaluates the trained checkpoint on the ImageAttributionBench test split under
# a protocol BYTE-IDENTICAL to the baselines (same images / split / cap /
# degradations / metrics), by reusing get_dataloader + calculate_metrics_for_test
# through comparison/training/test_hypclip.py. Runs degradation levels 0..6.
#
# Output: one test_results_degraded_{L}.txt per level (acc / macro-AUC / macro-AP /
# precision-recall-F1 / per-semantic acc / confusion matrix) → directly comparable
# to the baselines' own test_results_degraded_{L}.txt.
#
# Submit (after training produced the checkpoint):
#   sbatch slurm/slurm_eval_23cls_fair.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189          # matches $WORK=/leonardo_work/EUHPC_D35_189 — verify
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=eval_23cls
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1                # eval is single-GPU
#SBATCH --time=04:00:00                  # generous for 7 degradation levels
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CKPT=$WORK/hyp_fine_tuning/checkpoints/attribution_23cls_vitl14.pt
LOGDIR=$WORK/outputs/hypclip_fair

mkdir -p $LOGDIR
cd $REPO

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT (train first)."
    exit 1
fi

# ── Fair eval on the baselines' exact test split, degradation levels 0..6 ─────
CUDA_VISIBLE_DEVICES=0 python -m comparison.training.test_hypclip \
    --checkpoint $CKPT \
    --root_dir   $DATA \
    --batch_size 64 \
    --num_workers 8 \
    --level_start 0 \
    --level_end   7 \
    --log_dir    $LOGDIR

# Semantic-split variant (train on one semantic, test on the rest); run separately:
#   CUDA_VISIBLE_DEVICES=0 python -m comparison.training.test_hypclip \
#       --checkpoint $CKPT --root_dir $DATA --use_semantic_split --task_id 1 \
#       --level_start 0 --level_end 1 --log_dir $WORK/outputs/hypclip_fair_sem1

echo "Done. Results in $LOGDIR/test_results_degraded_*.txt"
