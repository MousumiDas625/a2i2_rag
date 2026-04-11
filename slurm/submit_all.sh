#!/bin/bash
# =============================================================================
# submit_all.sh — Submit all three experiment jobs to SLURM
# =============================================================================
#
# Run this from the a2i2_final/ directory on the Endeavour login node:
#     bash slurm/submit_all.sh
#
# All three jobs are submitted independently and run in parallel.
# After all three finish, run:
#     python experiments/make_success_matrices.py
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p slurm/logs

echo "Submitting Experiment 1 — Zero-Shot …"
JOB1=$(sbatch --parsable slurm/run_exp1_zero_shot.sh)
echo "  → Job ID: $JOB1"

echo "Submitting Experiment 2 — RAG-Successful …"
JOB2=$(sbatch --parsable slurm/run_exp2_rag.sh)
echo "  → Job ID: $JOB2"

echo "Submitting Experiment 3 — IQL+RAG …"
JOB3=$(sbatch --parsable slurm/run_exp3_iql.sh)
echo "  → Job ID: $JOB3"

# Submit matrix-generation job that waits for all three to finish
echo ""
echo "Submitting success-matrix generation (depends on $JOB1,$JOB2,$JOB3) …"
MATRIX_JOB=$(sbatch --parsable \
    --job-name=a2i2_matrices \
    --dependency=afterok:${JOB1}:${JOB2}:${JOB3} \
    --partition=main \
    --cpus-per-task=4 \
    --mem=8G \
    --time=00:30:00 \
    --output=slurm/logs/matrices_%j.out \
    --error=slurm/logs/matrices_%j.err \
    --wrap="cd $(pwd) && source .venv/bin/activate && export LLM_PROVIDER=openai && export OPENAI_MODEL=gpt-4o-mini && python experiments/make_success_matrices.py")
echo "  → Job ID: $MATRIX_JOB"

echo ""
echo "All jobs submitted:"
echo "  Exp1 (Zero-Shot)  : $JOB1"
echo "  Exp2 (RAG)        : $JOB2"
echo "  Exp3 (IQL+RAG)    : $JOB3"
echo "  Matrices          : $MATRIX_JOB (waits for all three)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       slurm/logs/"
