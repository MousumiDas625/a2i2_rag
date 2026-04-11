#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end pipeline for a2i2_final
# =============================================================================
#
# Run from the a2i2_final/ directory:
#     bash run_pipeline.sh
#
# Prerequisites:
#     - Python 3.10+ with packages from requirements.txt installed.
#     - Ollama running locally with llama3 pulled:
#           ollama pull llama3
#           ollama serve
#
# The script runs every stage sequentially. Comment out stages you've
# already completed to save time.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo "  A2I2 FINAL — Full Pipeline"
echo "=============================================="

# ── STAGE 1: Preprocessing ──────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: Preprocessing"
python preprocessing/P01_xlsx_to_jsonl.py
python preprocessing/P02_clean_and_merge.py
python preprocessing/P03_extract_residents.py
python preprocessing/P04_add_rewards.py

# ── STAGE 2: IQL Pipeline ───────────────────────────────────────────────────
echo ""
echo ">>> STAGE 2: IQL Pipeline"
python iql/I01_build_iql_dataset.py
python iql/I02_build_operator_policies.py
python iql/I03_train_iql.py
python iql/I04_build_rag_indexes.py
python iql/I05_extract_successful_utterances.py

# ── STAGE 3: Experiments ────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 3: Running Experiments"

echo "  Experiment 1: Zero-shot simulation …"
python experiments/exp1_zero_shot.py --runs 3

echo "  Experiment 2: RAG over successful operators …"
python experiments/exp2_rag_successful.py --runs 3

echo "  Experiment 3: IQL policy selection + RAG …"
python experiments/exp3_iql_policy.py --runs 3

# ── STAGE 4: Evaluation ────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 4: Evaluation"
python experiments/evaluate.py

echo ""
echo "=============================================="
echo "  Pipeline complete."
echo "=============================================="
