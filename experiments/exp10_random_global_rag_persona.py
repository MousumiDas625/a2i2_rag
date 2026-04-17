#!/usr/bin/env python3
"""
exp10_random_global_rag_persona.py — Experiment 10: Random Policy + Global RAG + Persona
==========================================================================================

PURPOSE:
    Baseline. At each operator turn:
        1. A policy is chosen UNIFORMLY AT RANDOM from the training personas.
        2. The matching persona profile (from config/personas.py) is injected.
        3. RAG examples are retrieved from the GLOBAL successful-operator
           corpus (across all policies).
        4. NO IQL.

    Comparing Exp10 vs Exp4 (IQL+Global-RAG) isolates the value of IQL
    policy selection when global RAG and persona info are both present.

USAGE:
    python experiments/exp10_random_global_rag_persona.py
    python experiments/exp10_random_global_rag_persona.py --residents ross,bob --runs 3
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
        description="Experiment 10: Random policy + global RAG + persona"
    )
    parser.add_argument("--residents", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", default="Hello, this is the fire department. "
                        "We need you to evacuate immediately.")
    args = parser.parse_args()

    residents = (
        [r.strip().lower() for r in args.residents.split(",")]
        if args.residents
        else sorted(PERSONA.keys())
    )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder = f"exp10_random_global_rag_persona_{ts}" + (f"_{args.tag}" if args.tag else "")
    exp_dir = RUNS_DIR / folder
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_results: list = []

    print("=" * 60)
    print("  EXPERIMENT 10 — RANDOM POLICY + GLOBAL RAG + PERSONA")
    print(f"  Residents: {', '.join(residents)}")
    print(f"  Runs/resident: {args.runs}")
    print("=" * 60)

    for resident in residents:
        for run_idx in range(1, args.runs + 1):
            rid = f"exp10_{resident}_run{run_idx}"
            print(f"\n--- {rid} ---")
            result = run_conversation(
                resident_name=resident,
                strategy="random_global_rag_persona",
                seed_text=args.seed,
                max_turns=args.max_turns,
                run_id=rid,
                output_dir=exp_dir,
            )
            all_results.append({
                "experiment": "random_global_rag_persona",
                "resident": resident,
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": result["turns"],
                "path": result["path"],
            })

    summary = {
        "experiment": "exp10_random_global_rag_persona",
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
    print("  RANDOM POLICY + GLOBAL RAG + PERSONA RESULTS")
    print(f"{'=' * 60}")
    for name, data in summary["per_resident"].items():
        print(f"  {name:<14} {data['successes']}/{data['runs']} = {data['success_rate']:.1%}")
    print(f"  {'OVERALL':<14} {summary['overall_success_rate']:.1%}")
    print(f"\n  Summary → {summary_file}")
    token_tracker.print_summary()


if __name__ == "__main__":
    main()
