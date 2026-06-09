#!/bin/bash
# ============================================================================
# CINECA Leonardo — re-caption IAB real images with Qwen3.5-9B (Ollama).
#
# Prereqs (run once on the LOGIN node, which has internet):
#   bash dataset_rebuilding/setup_ollama_cineca.sh        # installs Ollama + pulls model
#
# Submit:
#   sbatch dataset_rebuilding/slurm_caption.sh
#
# Resumable: relaunching skips already-captioned images, so if the 6h wall-time
# is hit just `sbatch` again.
# ============================================================================
#SBATCH --account=EUHPC_D26_009B
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=iab_recaption
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=richitrebbia@gmail.com

set -euo pipefail

# ── Python env (for requests / Pillow / tqdm) ─────────────────────────────────
module load python/3.11.7
module load cuda/12.6
source "$WORK/hyp_fine_tuning/bin/activate"

# ── Ollama env ────────────────────────────────────────────────────────────────
# Pinned to v0.24.0: the v0.30.x line crashes on Leonardo's driver (535/CUDA 12.2)
# with "CUDA error: device kernel image is invalid". See README / setup script.
OLLAMA_DIR="$WORK/ollama-0.24.0"
export PATH="$OLLAMA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="$WORK/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NUM_PARALLEL=8          # let Ollama batch concurrent requests
export OLLAMA_KEEP_ALIVE=-1           # keep the model resident for the whole job
export OLLAMA_MAX_LOADED_MODELS=1

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="$WORK/hyp_fine_tuning/hyperbolic_CLIP"
DATA="$WORK/iab_dataset"
ORIG_CAPS="$WORK/hyp_fine_tuning/iab_captions"
OUT="$WORK/iab_captions_detailed"
MODEL="qwen3.5:9b"
WORKERS=8

cd "$REPO"

# ── Start the Ollama server on the allocated GPU ──────────────────────────────
echo "Starting Ollama server (logs → ollama_serve_${SLURM_JOB_ID}.log)…"
ollama serve > "ollama_serve_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'echo "Stopping Ollama…"; kill $OLLAMA_PID 2>/dev/null || true' EXIT

# Give it a moment, then warm the model into VRAM via the API (caption_real_images
# also waits for readiness; this just surfaces load errors early). Guarded so a
# warm-up hiccup never fails the job.
sleep 10
curl -sf "http://$OLLAMA_HOST/api/generate" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"think\":false,\"keep_alive\":-1}" \
    >/dev/null 2>&1 || true
nvidia-smi || true

# ── Caption ───────────────────────────────────────────────────────────────────
python dataset_rebuilding/caption_real_images.py \
    --dataset_path      "$DATA" \
    --output_dir        "$OUT" \
    --orig_captions_dir "$ORIG_CAPS" \
    --model             "$MODEL" \
    --ollama_host       "$OLLAMA_HOST" \
    --num_workers       "$WORKERS" \
    --max_image_side    1024

echo "Captioning finished. Output CSVs in: $OUT"
