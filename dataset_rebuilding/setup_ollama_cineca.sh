#!/bin/bash
# ============================================================================
# One-time setup on the CINECA Leonardo LOGIN node (which HAS internet).
# Installs a local Ollama into $WORK and pre-pulls the captioning model so the
# offline compute nodes can serve it.
#
# Usage:  bash dataset_rebuilding/setup_ollama_cineca.sh
#         bash dataset_rebuilding/setup_ollama_cineca.sh qwen3.5:9b   # custom model
# ============================================================================
set -euo pipefail

MODEL="${1:-qwen3.5:9b}"

# Persist large data under $WORK (home quota is small). $WORK is shared across
# login and compute nodes, so models pulled here are visible to the SLURM job.
OLLAMA_DIR="$WORK/ollama"                 # the extracted Ollama distribution
export OLLAMA_MODELS="$WORK/ollama_models"  # where pulled model blobs live
OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_HOST

mkdir -p "$OLLAMA_DIR" "$OLLAMA_MODELS"

echo "=== Ollama setup on CINECA login node ==="
echo "WORK:          $WORK"
echo "OLLAMA_DIR:    $OLLAMA_DIR"
echo "OLLAMA_MODELS: $OLLAMA_MODELS"
echo "MODEL:         $MODEL"

# ── Install Ollama binary (no root needed; it's a self-contained tarball) ─────
if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
    echo "Downloading Ollama (linux-amd64)…"
    curl -L https://ollama.com/download/ollama-linux-amd64.tgz \
        -o "$OLLAMA_DIR/ollama.tgz"
    tar -C "$OLLAMA_DIR" -xzf "$OLLAMA_DIR/ollama.tgz"
    rm -f "$OLLAMA_DIR/ollama.tgz"
    echo "Installed Ollama → $OLLAMA_DIR/bin/ollama"
else
    echo "Ollama already installed at $OLLAMA_DIR/bin/ollama"
fi

export PATH="$OLLAMA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_DIR/lib:${LD_LIBRARY_PATH:-}"
echo "Ollama version: $(ollama --version 2>/dev/null || echo '??')"

# ── Persist env vars for future shells ────────────────────────────────────────
add_line() { grep -qxF "$1" ~/.bashrc || echo "$1" >> ~/.bashrc; }
add_line "export PATH=\$WORK/ollama/bin:\$PATH"
add_line "export LD_LIBRARY_PATH=\$WORK/ollama/lib:\${LD_LIBRARY_PATH:-}"
add_line "export OLLAMA_MODELS=\$WORK/ollama_models"

# ── Start a temporary server and pull the model ───────────────────────────────
echo "Starting a temporary Ollama server to pull the model…"
ollama serve > "$OLLAMA_DIR/serve_setup.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

# Wait until the API responds.
for _ in $(seq 1 60); do
    if curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then break; fi
    sleep 2
done

echo "Pulling $MODEL (this downloads several GB; one-time)…"
ollama pull "$MODEL"

echo ""
echo "Models now available locally:"
ollama list

kill $SERVE_PID 2>/dev/null || true
trap - EXIT

echo ""
echo "=== Done ==="
echo "Model '$MODEL' cached in $OLLAMA_MODELS."
echo "Submit the captioning job with:"
echo "    sbatch dataset_rebuilding/slurm_caption.sh"
