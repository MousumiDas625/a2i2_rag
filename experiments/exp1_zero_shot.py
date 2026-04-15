#!/usr/bin/env python3
"""
exp1_zero_shot.py — Experiment 1: Zero-Shot Simulation
========================================================

PURPOSE:
    Baseline experiment with NO training data.  The LLM receives only a
    system prompt describing the wildfire evacuation scenario and is asked
    to generate operator responses.  The resident side is also generated
    by the LLM using persona prompts.

HOW IT WORKS:
    For each resident persona × N runs:
        1. Start conversation with a seed operator line.
        2. Operator replies via zero-shot prompting (no RAG, no IQL).
        3. Resident replies via persona-grounded LLM prompting.
        4. Decision judge determines success/failure.

    Results are saved to data/runs/ and a summary JSON is written.

USAGE:
    python experiments/exp1_zero_shot.py
    python experiments/exp1_zero_shot.py --residents ross,bob --runs 3
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
    parser = argparse.ArgumentParser(description="Experiment 1: Zero-shot baseline")
    parser.add_argument("--residents", default=None,
                        help="Comma-separated residents (default: all)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Conversations per resident (default: 5)")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--seed", default="Hello, this is the fire department. "
                        "We need you to evacuate immediately.")
    args = parser.parse_args()

    residents = (
        [r.strip().lower() for r in args.residents.split(",")]
        if args.residents
        else sorted(PERSONA.keys())
    )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_dir = RUNS_DIR / f"exp1_zero_shot_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_results: list = []

    print("=" * 60)
    print(f"  EXPERIMENT 1 — ZERO-SHOT SIMULATION")
    print(f"  Residents: {', '.join(residents)}")
    print(f"  Runs/resident: {args.runs}")
    print("=" * 60)

    for resident in residents:
        for run_idx in range(1, args.runs + 1):
            rid = f"exp1_{resident}_run{run_idx}"
            print(f"\n--- {rid} ---")
            result = run_conversation(
                resident_name=resident,
                strategy="zero_shot",
                seed_text=args.seed,
                max_turns=args.max_turns,
                run_id=rid,
            )
            all_results.append({
                "experiment": "zero_shot",
                "resident": resident,
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": result["turns"],
                "path": result["path"],
            })

    # Summary
    summary = {
        "experiment": "exp1_zero_shot",
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
    print("  ZERO-SHOT RESULTS")
    print(f"{'=' * 60}")
    for name, data in summary["per_resident"].items():
        print(f"  {name:<14} {data['successes']}/{data['runs']} = {data['success_rate']:.1%}")
    print(f"  {'OVERALL':<14} {summary['overall_success_rate']:.1%}")
    print(f"\n  Summary → {summary_file}")
    token_tracker.print_summary()


if __name__ == "__main__":
    main()
