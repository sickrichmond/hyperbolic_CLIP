#!/bin/bash
# CINECA Leonardo — train a readout on cached features.
#
# SOURCE selects clip_features_SOURCE. HEAD=linear fits a linear classifier;
# HEAD=cone learns exterior-angle anchors and temperature on projection features.
# The readout uses weighted CE and reports cached validation metrics, not
# harness test results. No image encoder is run during training.
#
# Submit: sbatch --export=ALL,SOURCE=projection,HEAD=linear slurm/slurm_train_probe.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=train_probe
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

export TOKENIZERS_PARALLELISM=false
export IAB_EXCLUDE_GENERATORS=dalle3

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
SOURCE=${SOURCE:-frozen}
HEAD=${HEAD:-linear}
FEAT=$WORK/hyp_fine_tuning/clip_features_${SOURCE}
OUT=$WORK/hyp_fine_tuning/checkpoints

mkdir -p $OUT
cd $REPO

python train_linear_probe.py \
    --features_dir $FEAT \
    --head         $HEAD \
    --epochs       30 \
    --lr           1e-3 \
    --weight_decay 1e-4 \
    --batch_size   4096 \
    --output       $OUT/${HEAD}_probe_${SOURCE}.pt

echo "Done: $OUT/${HEAD}_probe_${SOURCE}.pt"
