#!/bin/bash
# Submit once for resnet50 and once for hifi_net.
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=hierarchy_holdout_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com
set -euo pipefail
METHOD="${1:?Pass resnet50 or hifi_net}"
case "$METHOD" in
  resnet50|hifi_net) ;;
  *) echo "Expected resnet50 or hifi_net." >&2; exit 2 ;;
esac
REPO="${REPO:-$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo}"
exec bash "$REPO/slurm/slurm_train_hierarchy_holdout.sh" "$METHOD"
