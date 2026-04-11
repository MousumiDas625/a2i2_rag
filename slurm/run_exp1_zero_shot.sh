#!/bin/bash
#SBATCH --job-name=a2i2_exp1_zeroshot
#SBATCH --account=<YOUR_ACCOUNT>          # e.g. ttrojan_123
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                      # request 1 GPU
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/exp1_%j.out
#SBATCH --error=slurm/logs/exp1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# =============================================================================
# Experiment 1 — Zero-Shot Simulation  (5 runs × all residents)
# =============================================================================
# PREREQUISITES:
#   Pipeline stages 1-2 (preprocessing + IQL build) must be completed first.
#   The trained IQL model is NOT needed for this experiment, but the virtual
#   environment must be activated and the LLM backend must be reachable.
#
# LLM BACKEND OPTIONS (pick one):
#   A) OpenAI API  → set LLM_PROVIDER=openai and OPENAI_API_KEY below.
#   B) Ollama      → start Ollama on the node before this job, set OLLAMA_URL.
# =============================================================================

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"
mkdir -p slurm/logs

source .venv/bin/activate
echo "[INFO] Python: $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ── LLM backend — OpenAI API ──────────────────────────────────────────────────
# OPENAI_API_KEY must be set in ~/.bashrc on the cluster.
# Uncomment and fill in below only if not set globally:
# export OPENAI_API_KEY="sk-proj-..."
export LLM_PROVIDER=openai
export OPENAI_MODEL="gpt-4o-mini"

echo "=============================================="
echo "  EXP 1 — ZERO-SHOT  |  job=$SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Start: $(date)"
echo "=============================================="

python experiments/exp1_zero_shot.py \
    --runs 5

echo "=============================================="
echo "  EXP 1 DONE  |  $(date)"
echo "=============================================="
