#!/bin/bash
# =============================================================================
# setup_env.sh — One-time environment setup on Endeavour using uv + Ollama
# =============================================================================
#
# Run this ONCE on a login node before submitting any jobs:
#
#     bash slurm/setup_env.sh
#
# This script:
#   1. Installs uv  → ~/.local/bin/uv
#   2. Creates Python 3.10 venv at a2i2_final/.venv
#   3. Installs PyTorch (CUDA 12.1) + all project dependencies
#   4. Installs Ollama binary  → a2i2_final/.ollama/ollama
#   5. Downloads the llama3 model to
#        /project2/biyik_1165/mousumid/a2i2_new/ollama_models/
#      (project space, not home quota)
#
# After this, all SLURM job scripts start Ollama automatically on the
# compute node using the pre-downloaded model.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."          # always run from a2i2_final/

# ── Configurable paths ────────────────────────────────────────────────────────
# Where Ollama stores model weights. Use project space, NOT home dir.
OLLAMA_MODELS_DIR="/project2/biyik_1165/mousumid/a2i2_new/ollama_models"
OLLAMA_BIN="$(pwd)/.ollama/ollama"   # self-contained binary inside the repo

echo "=============================================="
echo "  A2I2 — Environment Setup via uv + Ollama"
echo "  $(date)"
echo "=============================================="

# ── 1. Install uv ─────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo ""
    echo ">>> Installing uv …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo ">>> uv already installed: $(uv --version)"
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo "    uv: $(uv --version)"

# ── 2. Create virtual environment ─────────────────────────────────────────────
echo ""
echo ">>> Creating .venv with Python 3.10 …"
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.10
else
    echo "    .venv already exists — skipping creation."
fi
source .venv/bin/activate
echo "    Python: $(python --version)"

# ── 3. Install PyTorch (CUDA 12.1) ────────────────────────────────────────────
echo ""
echo ">>> Installing PyTorch (CUDA 12.1) …"
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── 4. Install project dependencies ───────────────────────────────────────────
echo ""
echo ">>> Installing project dependencies …"
uv pip install -r requirements.txt

echo ""
echo ">>> Replacing faiss-cpu with faiss-gpu …"
uv pip uninstall faiss-cpu -y 2>/dev/null || true
uv pip install faiss-gpu

# ── 5. Verify PyTorch GPU access ──────────────────────────────────────────────
echo ""
echo ">>> Verifying PyTorch (note: no GPU on login node — OK to see CUDA=False) …"
python - <<'EOF'
import torch
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device         : {torch.cuda.get_device_name(0)}")
EOF

# ── 6. Install Ollama binary ───────────────────────────────────────────────────
echo ""
echo ">>> Installing Ollama binary …"
mkdir -p .ollama
if [ ! -f "$OLLAMA_BIN" ] || [ ! -s "$OLLAMA_BIN" ]; then
    # Use GitHub releases — the ollama.com/download URL is unreliable on HPC nodes
    OLLAMA_RELEASE_URL="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64"
    echo "    Downloading from: $OLLAMA_RELEASE_URL"
    curl -L --fail --show-error --progress-bar \
        "$OLLAMA_RELEASE_URL" -o "$OLLAMA_BIN"
    chmod +x "$OLLAMA_BIN"

    # Sanity check: binary must be at least 30 MB
    BINARY_SIZE=$(stat -c%s "$OLLAMA_BIN" 2>/dev/null || stat -f%z "$OLLAMA_BIN")
    if [ "$BINARY_SIZE" -lt 30000000 ]; then
        echo "[ERROR] Downloaded file is only ${BINARY_SIZE} bytes — not a valid binary."
        echo "        Contents: $(head -c 200 "$OLLAMA_BIN")"
        rm -f "$OLLAMA_BIN"
        exit 1
    fi
    echo "    Ollama installed (${BINARY_SIZE} bytes) → $OLLAMA_BIN"
else
    echo "    Ollama already present: $("$OLLAMA_BIN" --version 2>/dev/null || echo 'unknown version')"
fi

# ── 7. Pull llama3 model ───────────────────────────────────────────────────────
# We start Ollama briefly on the login node (CPU-only, just for the download).
# The model is stored under OLLAMA_MODELS_DIR in project space.
echo ""
echo ">>> Pulling llama3 model (~4.7 GB) to $OLLAMA_MODELS_DIR …"
echo "    This only runs once; subsequent jobs reuse the cached weights."
mkdir -p "$OLLAMA_MODELS_DIR"

export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="127.0.0.1:11435"   # non-default port to avoid conflicts

# Start temporary server for the pull
"$OLLAMA_BIN" serve &
OLLAMA_SETUP_PID=$!
trap "kill $OLLAMA_SETUP_PID 2>/dev/null || true" EXIT

# Wait for server readiness (up to 60 s)
echo -n "    Waiting for Ollama server"
for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo " ready (${i}s)"
        break
    fi
    echo -n "."
    sleep 1
done

# Pull the model (skipped if already cached)
"$OLLAMA_BIN" pull llama3

# Stop the temporary server
kill "$OLLAMA_SETUP_PID" 2>/dev/null || true
trap - EXIT
wait "$OLLAMA_SETUP_PID" 2>/dev/null || true

echo "    llama3 model ready."

# ── 8. Write shared config for SLURM jobs ─────────────────────────────────────
# Persist paths so lib_ollama.sh can pick them up without hardcoding
cat > slurm/ollama_env.sh <<EOF
# Auto-generated by setup_env.sh — do not edit manually
export OLLAMA_BIN="$OLLAMA_BIN"
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
EOF
echo ""
echo ">>> Wrote slurm/ollama_env.sh (shared config for job scripts)"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Activate the env:  source .venv/bin/activate"
echo "  Submit all jobs:   bash slurm/submit_all.sh"
echo "=============================================="
