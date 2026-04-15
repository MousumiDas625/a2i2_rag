#!/usr/bin/env python3
"""
exp2_rag_successful.py — Experiment 2: RAG Over Successful Operators
=====================================================================

PURPOSE:
    At each operator turn, retrieves the top-K most similar utterances
    from the FAISS index of ALL successful operator utterances (built by
    I05).  These are injected as few-shot examples into the prompt.
    No IQL policy selection is used.

HOW IT WORKS:
    For each resident persona × N runs:
        1. Start conversation with a seed operator line.
        2. At each operator turn:
           a. Embed the latest resident utterance.
           b. Retrieve top-K similar operator utterances from the
              global successful-ops FAISS index.
           c. Build a prompt with those as few-shot examples.
           d. LLM generates the operator reply.
        3. Resident replies via persona-grounded LLM prompting.
        4. Decision judge determines success/failure.

PREREQUISITES:
    - Run I05_extract_successful_utterances.py first to build the corpus.

USAGE:
    python experiments/exp2_rag_successful.py
    python experiments/exp2_rag_successful.py --residents ross,bob --runs 3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import RUNS_DIR
from simulation.conversation_loop import run_conversation
from simulation.llm_client import token_tracker


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: RAG over successful operators"
    )
    parser.add_argument("--residents", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--seed", default="Hello, this is the fire department. "
                        "We need you to evacuate immediately.")
    args = parser.parse_args()

    residents = (
        [r.strip().lower() for r in args.residents.split(",")]
        if args.residents
        else sorted(PERSONA.keys())
    )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_dir = RUNS_DIR / f"exp2_rag_successful_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_results: list = []

    print("=" * 60)
    print(f"  EXPERIMENT 2 — RAG OVER SUCCESSFUL OPERATORS")
    print(f"  Residents: {', '.join(residents)}")
    print(f"  Runs/resident: {args.runs}")
    print("=" * 60)

    for resident in residents:
        for run_idx in range(1, args.runs + 1):
            rid = f"exp2_{resident}_run{run_idx}"
            print(f"\n--- {rid} ---")
            result = run_conversation(
                resident_name=resident,
                strategy="rag_successful",
                seed_text=args.seed,
                max_turns=args.max_turns,
                run_id=rid,
            )
            all_results.append({
                "experiment": "rag_successful",
                "resident": resident,
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": result["turns"],
                "path": result["path"],
            })

    summary = {
        "experiment": "exp2_rag_successful",
        "timestamp": ts,
        "results": all_results,
        "per_resident": {},
    }
    for resident in residents:
        runs = [r for r in all_results if r["resident"] == resident]
        succ = sum(r["success"] for r in runs)
        summary["per_resident"][resident] = {
            "runs": len(runs),
            "successes": succ,
            "success_rate": round(succ / len(runs), 4) if runs else 0,
        }
    total_succ = sum(r["success"] for r in all_results)
    summary["overall_success_rate"] = round(
        total_succ / len(all_results), 4
    ) if all_results else 0

    summary_file = exp_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}")
    print("  RAG-SUCCESSFUL RESULTS")
    print(f"{'=' * 60}")
    for name, data in summary["per_resident"].items():
        print(f"  {name:<14} {data['successes']}/{data['runs']} = {data['success_rate']:.1%}")
    print(f"  {'OVERALL':<14} {summary['overall_success_rate']:.1%}")
    print(f"\n  Summary → {summary_file}")
    token_tracker.print_summary()


if __name__ == "__main__":
    main()
