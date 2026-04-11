#!/bin/bash
# =============================================================================
# lib_ollama.sh — Reusable Ollama lifecycle helper for SLURM jobs
# =============================================================================
#
# SOURCE this file (do not execute it) at the top of any job script:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/lib_ollama.sh"
#     ollama_start          # starts server, waits until ready
#     # ... run your Python experiment ...
#     ollama_stop           # clean shutdown (also called automatically on exit)
#
# WHAT IT DOES:
#   - Loads paths from slurm/ollama_env.sh (written by setup_env.sh)
#   - Picks a free port so parallel jobs on the same node don't collide
#   - Starts `ollama serve` as a background process using the compute node GPU
#   - Exports OLLAMA_URL so llm_client.py finds the server
#   - Registers a trap to kill the server when the job exits (clean or error)
# =============================================================================

# ── Load paths written by setup_env.sh ────────────────────────────────────────
_LIB_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
_OLLAMA_ENV="$_LIB_DIR/ollama_env.sh"

if [ ! -f "$_OLLAMA_ENV" ]; then
    echo "[ERROR] slurm/ollama_env.sh not found."
    echo "        Run  bash slurm/setup_env.sh  first."
    exit 1
fi
# shellcheck source=/dev/null
source "$_OLLAMA_ENV"

# ── Pick a free port ──────────────────────────────────────────────────────────
# Default 11434; if taken (e.g. two jobs on same node) pick a random free port.
_pick_free_port() {
    local port=11434
    while ss -ltn 2>/dev/null | grep -q ":${port} "; do
        port=$(( RANDOM % 10000 + 20000 ))
    done
    echo "$port"
}

# ── Public: start Ollama ──────────────────────────────────────────────────────
ollama_start() {
    local port
    port=$(_pick_free_port)
    export OLLAMA_HOST="127.0.0.1:${port}"
    export OLLAMA_URL="http://${OLLAMA_HOST}/api/generate"

    echo "[Ollama] Starting server on ${OLLAMA_HOST} (GPU: ${CUDA_VISIBLE_DEVICES:-all}) …"
    OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
        "$OLLAMA_BIN" serve >"$_LIB_DIR/logs/ollama_${SLURM_JOB_ID:-local}.log" 2>&1 &
    _OLLAMA_PID=$!

    # Register cleanup on any exit (success, error, or signal)
    trap ollama_stop EXIT INT TERM

    # Wait up to 90 s for the server to be ready
    echo -n "[Ollama] Waiting for server"
    local ready=0
    for i in $(seq 1 90); do
        if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
            echo " ready (${i}s)"
            ready=1
            break
        fi
        echo -n "."
        sleep 1
    done

    if [ "$ready" -eq 0 ]; then
        echo ""
        echo "[ERROR] Ollama did not start within 90 s."
        echo "        Check: $_LIB_DIR/logs/ollama_${SLURM_JOB_ID:-local}.log"
        exit 1
    fi

    echo "[Ollama] PID=$_OLLAMA_PID  |  OLLAMA_URL=$OLLAMA_URL"

    # Export so llm_client.py picks up the correct URL
    export LLM_PROVIDER=ollama
    export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
}

# ── Public: stop Ollama ───────────────────────────────────────────────────────
ollama_stop() {
    if [ -n "${_OLLAMA_PID:-}" ]; then
        echo "[Ollama] Stopping server (PID=$_OLLAMA_PID) …"
        kill "$_OLLAMA_PID" 2>/dev/null || true
        wait "$_OLLAMA_PID" 2>/dev/null || true
        _OLLAMA_PID=""
    fi
    trap - EXIT INT TERM
}
