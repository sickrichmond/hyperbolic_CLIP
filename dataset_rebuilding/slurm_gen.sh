#!/bin/bash
# ============================================================================
# CINECA Leonardo — regenerate fakes for ONE diffusion generator from the new
# captions. Launch one job per generator (each gets its own GPU) to run them in
# parallel:
#
#   sbatch dataset_rebuilding/slurm_gen.sh SD3
#   sbatch dataset_rebuilding/slurm_gen.sh SD3_5
#   sbatch dataset_rebuilding/slurm_gen.sh SDXL
#   sbatch dataset_rebuilding/slurm_gen.sh FLUX
#
# Prereqs (run once on the LOGIN node):
#   bash dataset_rebuilding/setup_diffusion_cineca.sh     # diffusers + weights
#   (SD3 / SD3.5 are GATED — accept their licenses on HF first; see that script)
#   and the cleaned captions in $WORK/iab_captions_detailed_clean.
#
# FULL set with IAB naming ({prefix}_p{N}_i{K}.png), mirroring the original IAB
# fake set 1:1 (~2000/class × 10 ≈ 20k imgs/generator). This is heavy for the
# 1024² models — it WILL likely exceed 24h; just re-`sbatch` (resumable, existing
# PNGs are skipped). FLUX at 512² is far faster.
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=recap_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1            # one A100 64GB holds SD3/SD3.5/SDXL/FLUX in bf16
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail

GEN="${1:?usage: sbatch slurm_gen.sh <FLUX|SD3|SD3_5|SDXL> [semantic ...]}"
shift
SEMS=("$@")   # optional: subset of semantic classes → one class per job for
              # class-sharded parallelism, e.g.  sbatch slurm_gen.sh SD3 COCO

module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

export HF_HOME="$WORK/hf_cache"
export TRANSFORMERS_OFFLINE=1        # compute nodes have no internet
export HF_HUB_OFFLINE=1

REPO="$WORK/hyp_fine_tuning/hyperbolic_CLIP"
CAPS="$WORK/iab_captions_detailed_clean"     # our dense captions (keyed by stem)
SRC_CAPS="$WORK/hyp_fine_tuning/iab_captions" # original IAB CSVs (row N → stem)
OUT="$WORK/iab_recap_dataset_v2"             # ROUND 2: new dir, keeps round-1 intact

cd "$REPO"
echo "Generator: $GEN  →  $OUT/$GEN/"
nvidia-smi || true

# --naming iab → {prefix}_p{N}_i{K}.png, mirroring the IAB fake set. Resolution /
# steps / guidance / dtype default to the IAB-matched per-generator values
# (FLUX → 512², the rest → 1024²). No --max_per_class = full set.
SEM_ARG=()
[ ${#SEMS[@]} -gt 0 ] && SEM_ARG=(--semantics "${SEMS[@]}")

python dataset_rebuilding/generate_fakes.py \
    --generator            "$GEN" \
    --naming               iab \
    --captions_dir         "$CAPS" \
    --fake_src_captions_dir "$SRC_CAPS" \
    --out_root             "$OUT" \
    "${SEM_ARG[@]}"

echo "Fakes under: $OUT/$GEN/"
