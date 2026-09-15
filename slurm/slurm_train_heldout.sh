#!/bin/bash
# CINECA Leonardo — omit one generator from 22-class training.
#
# HELDOUT defaults to infinity. Exclude both dalle3 and HELDOUT from the active
# class map, and train the remaining 21 classes using the 22-class manifest.
# The held-out generator's images are not enumerated for training.
#
# The optional tests.probe_open_set diagnostic uses the cone decision rule.
# For a matching probe, set IAB_EXCLUDE_GENERATORS=dalle3,HELDOUT and pass
# --unknown HELDOUT. Dataset similarity must be assessed separately.
#
# Submit: sbatch --export=ALL,HELDOUT=infinity slurm/slurm_train_heldout.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=attr_heldout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
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

HELDOUT=${HELDOUT:-infinity}

# Keep dataset enumeration, anchors and evaluation on the same 21-class map.
export IAB_EXCLUDE_GENERATORS=dalle3,$HELDOUT

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/hyp_fine_tuning/checkpoints
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json
CKPT=$OUT/attribution_21cls_no_${HELDOUT}_vitl14.pt

ALL="real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 SD1_5 SD2_1 SD3 SD3_5
     SDXL gemini grok3 hidream hunyuan ideogram infinity janus-pro kling mid-5.2 mid-6.0"
GENS=$(echo $ALL | tr ' ' '\n' | grep -vxF "$HELDOUT")   # -F: 'mid-5.2' is not a regex

# Reject unknown held-out labels before launching training.
[ "$(echo "$GENS" | wc -w)" = 21 ] || { echo "ERROR: '$HELDOUT' is not one of the 22"; exit 1; }
echo "Holding out '$HELDOUT' — training on 21 classes"

mkdir -p $OUT
cd $REPO

CUDA_VISIBLE_DEVICES=0,1 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      $GENS \
    --semantics       COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --clip_name       openai/clip-vit-large-patch14 \
    --anchor_init     text \
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
    --lr              3e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $CKPT

echo "Done: $CKPT"
