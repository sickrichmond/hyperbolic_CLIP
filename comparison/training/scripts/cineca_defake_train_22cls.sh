#!/bin/bash
# Image-only De-Fake attributor @ 22 classes (dalle3 excluded).
# Submit: sbatch comparison/training/scripts/cineca_defake_train_22cls.sh
# Requires OpenAI CLIP installed and its ViT-B/32 checkpoint cached/downloadable.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_defake_train_22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aspis2000@virgilio.com
set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate
export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TORCH_HOME=$WORK/hyp_fine_tuning/torch_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3
REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_giuseppe
DATA=$FAST/datasets/iab_dataset
cd $REPO
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?} | exclude=${IAB_EXCLUDE_GENERATORS}"
python -m comparison.training.train \
  --config comparison/training/config/model/defake.yaml \
  --root_dir "$DATA" \
  --n_epoch 30 -n 2000 --batch_size 32 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --log_dir comparison/training/logs
echo "Done (22cls). Results under comparison/training/logs/default_split/defake/<run>/"