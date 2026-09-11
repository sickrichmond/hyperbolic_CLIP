#!/bin/bash
# CINECA Leonardo — cache features on the comparison train/val manifest.
#
# SOURCE=frozen: frozen CLIP features.
# SOURCE=lora: adapted CLIP features from CKPT.
# SOURCE=projection: tangent or Euclidean pre-normalization head output from CKPT.
# The last two require CKPT. OUT_DIR overrides the source-specific cache path;
# use distinct directories for different checkpoints or datasets.
#
# Submit: sbatch --export=ALL,SOURCE=frozen slurm/slurm_extract_features.sh

#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=extract_clip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
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
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_22cls.json
FEAT=$WORK/hyp_fine_tuning/clip_features

SOURCE=${SOURCE:-frozen}
case "$SOURCE" in
  frozen)     EXTRA="" ;;
  lora)       EXTRA="--checkpoint $CKPT --features clip" ;;
  projection) EXTRA="--checkpoint $CKPT --features projection" ;;
  *) echo "SOURCE must be frozen, lora or projection (got '$SOURCE')"; exit 1 ;;
esac

cd $REPO

python -m scripts.extract_clip_features \
    --dataset_path   $DATA \
    --captions_dir   $CAPS \
    --clip_name      openai/clip-vit-large-patch14 \
    --generators     real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                     SD1_5 SD2_1 SD3 SD3_5 SDXL gemini grok3 hidream hunyuan \
                     ideogram infinity janus-pro kling mid-5.2 mid-6.0 \
    --semantics      COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \
    --split_manifest $MANIFEST \
    $EXTRA \
    --batch_size     256 \
    --num_workers    8 \
    --out_dir        ${OUT_DIR:-${FEAT}_${SOURCE}}

echo "Done. Cache in ${OUT_DIR:-${FEAT}_${SOURCE}}/"
