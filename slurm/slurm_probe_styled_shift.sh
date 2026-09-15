#!/bin/bash
# CINECA Leonardo — compare embeddings of paired generated styles.
#
# STYLE selects the second root; the first is the unstyled recap root.
# The probe pairs matching relative paths, assuming corresponding source content.
# N sets the sample count and CKPTS the checkpoint list. It reports angular
# shift/separation ratios and cosine-prediction flips, not calibrated thresholds.
#
# Submit: sbatch --export=ALL,STYLE=cartoon,N=4000 slurm/slurm_probe_styled_shift.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=styled_shift
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
CK=$WORK/hyp_fine_tuning/checkpoints
STYLE=${STYLE:-cartoon}
N=${N:-4000}
CKPTS=${CKPTS:-"$CK/attribution_22cls_sweepwin_vitl14.pt $CK/attribution_22cls_euclidean_d128_vitl14.pt"}

cd $REPO

python -m tests.probe_degradation_shift \
    --styled $FAST/datasets/iab_recap_dataset_v2 $FAST/datasets/iab_recap_${STYLE}_v2 \
    --n $N \
    $CKPTS

echo "Done: styled shift, $STYLE"
