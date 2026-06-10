#!/bin/bash
# ============================================================================
# One-time setup on the CINECA Leonardo LOGIN node for FLUX fake generation.
# Installs diffusers into the existing venv and pre-downloads the FLUX weights
# into HF_HOME (compute nodes are offline).
#
# Usage:  bash dataset_rebuilding/setup_flux_cineca.sh
#         bash dataset_rebuilding/setup_flux_cineca.sh black-forest-labs/FLUX.1-dev
#
# Default is FLUX.1-schnell (what IAB used): Apache-2.0, UNGATED — no token needed.
# To use FLUX.1-dev instead, it is GATED: first accept the license at
#   https://huggingface.co/black-forest-labs/FLUX.1-dev
# then `huggingface-cli login` (or export HF_TOKEN=hf_xxx) before running this.
# ============================================================================
set -euo pipefail

MODEL="${1:-black-forest-labs/FLUX.1-schnell}"

module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

export HF_HOME="$WORK/hf_cache"
echo "=== FLUX setup ==="
echo "MODEL:   $MODEL"
echo "HF_HOME: $HF_HOME"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "Installing diffusers + deps…"
pip install --upgrade diffusers sentencepiece protobuf --quiet
python -c "import diffusers; print('diffusers', diffusers.__version__)"

# ── Pre-download weights into HF_HOME ─────────────────────────────────────────
# Use snapshot_download (fetch files only) — NOT FluxPipeline.from_pretrained,
# which also loads ~34 GB of weights into RAM and gets OOM-killed on the
# memory-limited login node. The GPU compute node does the actual load.
echo "Pre-downloading $MODEL into $HF_HOME (one-time, ~34 GB)…"
python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
model = sys.argv[1]
path = snapshot_download(repo_id=model)
print(f"Cached {model} → {path}")
PY

echo ""
echo "=== Done ==="
echo "Submit the generation job with:"
echo "    sbatch dataset_rebuilding/slurm_flux.sh"
