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

# ── Install Ollama binary (no root needed; it's a self-contained bundle) ──────
# Recent Ollama releases ship a zstd-compressed tarball (ollama-linux-amd64.tar.zst),
# NOT the old .tgz. We fetch it from GitHub releases. Override the version with
# OLLAMA_VERSION=v0.30.6 if you want to pin it.
OLLAMA_VERSION="${OLLAMA_VERSION:-latest}"
if [ "$OLLAMA_VERSION" = "latest" ]; then
    URL="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst"
else
    URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
fi

if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
    ARCHIVE="$OLLAMA_DIR/ollama-linux-amd64.tar.zst"
    echo "Downloading Ollama ($OLLAMA_VERSION) from $URL …"
    # -f: fail on HTTP errors (so a 404 body is NOT saved as the archive).
    curl -fL --retry 3 --retry-delay 3 "$URL" -o "$ARCHIVE"

    # Decompress zstd robustly: RHEL8's tar (1.30) has no --zstd, so prefer the
    # standalone zstd binary; fall back to a zstd-aware tar if present.
    echo "Extracting …"
    if command -v zstd >/dev/null 2>&1; then
        zstd -dc "$ARCHIVE" | tar -x -C "$OLLAMA_DIR"
    elif command -v unzstd >/dev/null 2>&1; then
        unzstd -c "$ARCHIVE" | tar -x -C "$OLLAMA_DIR"
    elif tar --help 2>/dev/null | grep -q -- '--zstd'; then
        tar --zstd -xf "$ARCHIVE" -C "$OLLAMA_DIR"
    else
        echo "ERROR: need 'zstd' to extract $ARCHIVE but none found." >&2
        echo "Try:  module load zstd   (or: module spider zstd)   then re-run." >&2
        exit 1
    fi
    rm -f "$ARCHIVE"

    if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
        echo "ERROR: extraction did not produce $OLLAMA_DIR/bin/ollama." >&2
        exit 1
    fi
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
