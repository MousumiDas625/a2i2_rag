#!/bin/bash
#SBATCH --job-name=a2i2_exp2_rag
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/exp2_%j.out
#SBATCH --error=slurm/logs/exp2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# =============================================================================
# Experiment 2 — RAG over Successful Operators  (5 runs × all residents)
# =============================================================================
# PREREQUISITES:
#   iql/I05_extract_successful_utterances.py must have been run to build the
#   FAISS index at data/indexes/faiss/.
# =============================================================================

set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"
mkdir -p slurm/logs

source .venv/bin/activate
echo "[INFO] Python: $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# export OPENAI_API_KEY="sk-proj-..."
export LLM_PROVIDER=openai
export OPENAI_MODEL="gpt-4o-mini"

echo "=============================================="
echo "  EXP 2 — RAG-SUCCESSFUL  |  job=$SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Start: $(date)"
echo "=============================================="

python experiments/exp2_rag_successful.py \
    --runs 5

echo "=============================================="
echo "  EXP 2 DONE  |  $(date)"
echo "=============================================="
