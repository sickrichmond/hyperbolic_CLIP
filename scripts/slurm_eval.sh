#!/bin/bash
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval_attribution
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

export HF_HOME=$WORK/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1

cd $WORK/hyp_fine_tuning/hyperbolic_CLIP

python -m tests.eval_attribution \
    --checkpoint   $WORK/checkpoints/attribution_FLUX_vitl14_strict.pt \
    --dataset_path $WORK/iab_dataset \
    --captions_dir $WORK/hyp_fine_tuning/iab_captions \
    --generators   real FLUX \
    --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split        val \
    --val_frac     0.2 \
    --batch_size   256 \
    --num_workers  4
