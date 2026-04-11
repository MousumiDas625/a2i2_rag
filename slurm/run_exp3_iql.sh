#!/bin/bash
#SBATCH --job-name=a2i2_exp3_iql
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/exp3_%j.out
#SBATCH --error=slurm/logs/exp3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# =============================================================================
# Experiment 3 — IQL Policy Selection + Per-Policy RAG  (5 runs × all residents)
# =============================================================================
# PREREQUISITES:
#   Full pipeline stages 1-2 must be completed:
#     - iql/I03_train_iql.py --mode state   (trained Q-network)
#     - iql/I04_build_rag_indexes.py         (per-policy FAISS indexes)
#   The trained model is loaded from data/iql/selector/
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
echo "  EXP 3 — IQL+RAG  |  job=$SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Start: $(date)"
echo "=============================================="

python experiments/exp3_iql_policy.py \
    --runs 5

echo "=============================================="
echo "  EXP 3 DONE  |  $(date)"
echo "=============================================="
