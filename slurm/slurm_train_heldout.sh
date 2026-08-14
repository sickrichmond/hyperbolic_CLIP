#!/bin/bash
# ============================================================================
# CINECA Leonardo — OPEN-SET setup: the sweepwin recipe with ONE GENERATOR HELD OUT
# of training entirely, so it can serve as a genuine unknown at evaluation time.
#
# Why this run has to exist: IAB ships 23 generators, but `dalle3` is a duplicate of
# `4o` (measured: a 22-class model sends 99.8% of dalle3 to 4o), so the benchmark
# offers no unknown at all. The only honest open-set test is to make one.
#
# HELDOUT=infinity is the default and the right default: no twin, a genuinely
# different family (autoregressive, not diffusion), and it sits in the 1024px
# resolution group — so the resampling channel cannot hand over the separation and
# inflate the AUROC. janus-pro would be more convenient and much worse: it is the
# only class at 384px, and the score would largely measure resampling.
#
# Everything else is byte-identical to slurm_train_22cls_sweepwinner.sh. The 22-class
# manifest is reused as-is: the held-out generator is simply never enumerated, so the
# remaining 21 classes train on exactly the images they trained on before.
#
# Then, on a GPU node:
#   IAB_EXCLUDE_GENERATORS=dalle3,infinity \
#   python -m tests.probe_open_set --unknown infinity <checkpoint>
#
# Submit:  sbatch slurm/slurm_train_heldout.sh
#          sbatch --export=ALL,HELDOUT=kling slurm/slurm_train_heldout.sh
# ============================================================================

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

# dalle3 was never in the label space; the held-out class leaves it now, so the whole
# pipeline (eval, anchors, probe_open_set) sees the same 21 classes the model knows.
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

# A typo in HELDOUT would silently train on all 22 and the run would be worthless —
# the entire point is that one class is absent.
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
