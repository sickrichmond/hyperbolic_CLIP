#!/bin/bash
# ============================================================================
# CINECA Leonardo — Attribution-CLIP fine-tuning, 23-CLASS FAIR run
# CLIP ViT-L/14, LoRA on both encoders, hyperbolic entailment-cone loss.
#
# Trains ours on ALL 23 IAB classes (22 generators + real), STRICTLY on the
# baselines' train split (via --split_manifest → include-only), so the head-to-head
# with resnet50/dct/hifi_net/defl is on identical data and (via test_hypclip.py)
# identical test images. See NEXT_STEPS.md / test_hypclip.py.
#
# PREREQUISITE — dump the split manifest once (cheap, login node is fine):
#   python -m comparison.training.scripts.dump_split_manifest \
#       --root_dir $FAST/datasets/iab_dataset \
#       --out $WORK/hyp_fine_tuning/split_manifest_default.json
#
# Submit:  sbatch slurm/slurm_train_23cls_fair.sh
# ============================================================================

#SBATCH --account=EUHPC_D35_189          # matches $WORK=/leonardo_work/EUHPC_D35_189 — verify
#SBATCH --partition=boost_usr_prod       # A100 partition on Leonardo
#SBATCH --job-name=attr_clip_23cls
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32               # 8 workers × 4 GPUs
#SBATCH --gpus-per-node=4                # 4× A100 80GB
#SBATCH --time=24:00:00                  # high walltime for safety (partition max)
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source $WORK/hyp_fine_tuning/bin/activate

export HF_HOME=$WORK/hyp_fine_tuning/hf_cache      # avoid filling home quota
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1      # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

REPO=$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo
DATA=$FAST/datasets/iab_dataset
CAPS=$WORK/hyp_fine_tuning/iab_captions
OUT=$WORK/hyp_fine_tuning/checkpoints
MANIFEST=$WORK/hyp_fine_tuning/split_manifest_default.json

mkdir -p $OUT
cd $REPO

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: split manifest not found at $MANIFEST"
    echo "Run dump_split_manifest.py first (see header)."
    exit 1
fi

# ── Training (all 23 IAB classes, strict train-set parity) ────────────────────
# --split_manifest: ours is trained ONLY on the manifest's 'train' images (∩ captioned)
# and never on the baselines' val/test → leakage-free + identical eval images.
# Watch the startup log for "Split manifest: train ours on N ..." with N > 0, and
# per-split "filtered out X (outside split manifest)".

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_attribution.py \
    --dataset_path    $DATA \
    --captions_dir    $CAPS \
    --generators      real 4o CogView3_PLUS FLUX KANDINSKY PIXART PLAYGROUND_2_5 \
                      SD1_5 SD2_1 SD3 SD3_5 SDXL dalle3 gemini grok3 hidream hunyuan \
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
    --lambda_cap_in_class 1.0 \
    --lambda_img_in_cap   0.5 \
    --lambda_norm     0.5 \
    --target_norm     4.0 \
    --batch_size      256 \
    --num_epochs      5 \
    --lr              5e-5 \
    --weight_decay    0.01 \
    --val_frac        0.2 \
    --num_workers     8 \
    --split_manifest  $MANIFEST \
    --output          $OUT/attribution_23cls_vitl14.pt

echo "Done: $OUT/attribution_23cls_vitl14.pt"
