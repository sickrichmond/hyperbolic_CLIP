#!/bin/bash
# ============================================================================
# One-time setup on the CINECA Leonardo LOGIN node for SD3 / SD3.5 / SDXL fake
# generation. Installs deps and pre-downloads the weights into HF_HOME (compute
# nodes are offline). Uses snapshot_download (files only) to avoid OOM on the
# memory-limited login node.
#
# Usage:  bash dataset_rebuilding/setup_diffusion_cineca.sh
#         bash dataset_rebuilding/setup_diffusion_cineca.sh SD3 SDXL   # subset
#
# GATING — SD3 and SD3.5 are gated. ONCE, with the account behind your HF token:
#   1) accept the licenses:
#        https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
#        https://huggingface.co/stabilityai/stable-diffusion-3.5-large
#   2) export HF_TOKEN=hf_xxx   (or: huggingface-cli login --token $HF_TOKEN)
#   SDXL is ungated.
# ============================================================================
set -euo pipefail

# Generator label → HF repo id (keep in sync with generate_fakes.py GENERATORS).
declare -A REPO=(
    [SD3]="stabilityai/stable-diffusion-3-medium-diffusers"
    [SD3_5]="stabilityai/stable-diffusion-3.5-large"
    [SDXL]="stabilityai/stable-diffusion-xl-base-1.0"
    [FLUX]="black-forest-labs/FLUX.1-schnell"
)

GENS=("$@")
if [ ${#GENS[@]} -eq 0 ]; then
    GENS=(SD3 SD3_5 SDXL)        # FLUX handled by setup_flux_cineca.sh
fi

module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"
export HF_HOME="$WORK/hf_cache"

echo "=== Diffusion generators setup ==="
echo "HF_HOME: $HF_HOME"
echo "Models:  ${GENS[*]}"

echo "Installing diffusers + deps…"
pip install --upgrade diffusers sentencepiece protobuf --quiet
python -c "import diffusers; print('diffusers', diffusers.__version__)"

for gen in "${GENS[@]}"; do
    model="${REPO[$gen]:-}"
    if [ -z "$model" ]; then
        echo "!! unknown generator '$gen' — skipping"; continue
    fi
    echo ""
    echo "── Downloading $gen → $model ──"
    python - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download
model = sys.argv[1]
path = snapshot_download(repo_id=model)
print(f"Cached {model} → {path}")
PY
done

echo ""
echo "=== Done ==="
echo "Launch (one job per generator, in parallel):"
for gen in "${GENS[@]}"; do
    echo "    sbatch dataset_rebuilding/slurm_gen.sh $gen"
done
