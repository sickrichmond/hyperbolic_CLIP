#!/bin/bash
# CINECA Leonardo — 22-class cone training without caption terms, two GPUs.
#
# Uses the same training arguments and checkpoint path as
# slurm_train_22cls_base.sh, with different resource requests. Do not run both
# against that output path concurrently. The total batch size remains 256;
# GPU count can affect numerical results, memory use and elapsed time.
#
# Create the 22-class manifest before submission:
#   IAB_EXCLUDE_GENERATORS=dalle3 python -m comparison.training.scripts.dump_split_manifest \
#       --root_dir $FAST/datasets/iab_dataset \
#       --out $WORK/hyp_fine_tuning/split_manifest_22cls.json
#
# Submit: sbatch slurm/slurm_train_22cls_base_2gpu.sh

#SBATCH --account=EUHPC_D35_189          # verify with `saldo -b`
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod            # long QOS (up to 4 days) for the ~2x wall time
#SBATCH --job-name=attr_22cls_base_2g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
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
export IAB_EXCLUDE_GENERATORS=dalle3      # <-- 22-class toggle (whole pipeline)

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/hyp_fine_tuning/checkpoints
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json

mkdir -p $OUT
cd $REPO

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 22-class manifest not found at $MANIFEST"
    echo "Regenerate with IAB_EXCLUDE_GENERATORS=dalle3 (see header)."
    exit 1
fi

# Train 21 generator classes plus real without captions; select on harness val.
# DataParallel uses the two visible devices with a total batch size of 256.
CUDA_VISIBLE_DEVICES=0,1 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                      ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --lora_r          16 \
    --lora_alpha      32 \
    --hyperbolic_dim  128 \
    --curv            1.0 \
    --min_radius      0.5 \
    --margin          0.3 \
    --lambda_neg      1.0 \
    --lambda_norm     0.5 \
    --target_norm     4.0 \
    --no_captions \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $OUT/attribution_22cls_base_vitl14.pt

echo "Done: $OUT/attribution_22cls_base_vitl14.pt"
