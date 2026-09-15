#!/bin/bash
# HiFi-Net attributor @ 22 CLASSES (dalle3 excluded). Derived from
# cineca_hifi_net_train.sh. The hierarchical head auto-adapts to 22 fine classes
# (BranchCLS level-4 + parent_idx_4 derived from hifi_label_mapping()). ~40h.
# Submit: sbatch comparison/training/scripts/cineca_hifi_net_train_22cls.sh
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=iab_hifi_train_22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=44:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com
set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate
export HF_HOME=$WORK/hyp_fine_tuning/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export IAB_EXCLUDE_GENERATORS=dalle3          # <-- 22-class toggle (also drives the HiFi tree)
REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
cd $REPO
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?} | exclude=${IAB_EXCLUDE_GENERATORS}"
python -m comparison.training.train \
  --config comparison/training/config/model/hifi_net.yaml \
  --root_dir "$DATA" \
  --n_epoch 10 -n 2000 --batch_size 8 \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  --log_dir comparison/training/logs
echo "Done (22cls). Results under comparison/training/logs/default_split/hifi_net/<run>/"
