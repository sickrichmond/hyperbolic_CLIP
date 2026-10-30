#!/bin/bash
# All baseline checkpoints; input images from PLAYGROUND_2_5, hunyuan and SDXL only.
# Predictions still compete over all 22 IAB classes.
# Submit: sbatch slurm/slurm_eval_iab_styled_baselines_3models.sh checkpoints.json
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_iab_styled_3models
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail
export LOGDIR="${LOGDIR:-$WORK/outputs/baselines_iab_styled_3models_${SLURM_ARRAY_JOB_ID}}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/slurm_eval_iab_styled_baselines.sh" \
    "${1:-${CHECKPOINTS:-}}" --generators PLAYGROUND_2_5 hunyuan SDXL
