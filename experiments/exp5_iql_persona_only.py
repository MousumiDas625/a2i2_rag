#!/usr/bin/env python3
"""
exp5_iql_persona_only.py — Experiment 5: IQL Policy Selection + Persona Only
=============================================================================

PURPOSE:
    At each operator turn:
        1. The trained IQL Q-network selects the best operator policy.
        2. The matching persona profile (from config/personas.py) is injected
           into the operator prompt.
        3. NO RAG examples are used.

    This isolates the contribution of persona information alone (without RAG).

PREREQUISITES:
    - Full pipeline: P01–P04, I01–I03 must have been run.
    - Trained IQL model.

USAGE:
    python experiments/exp5_iql_persona_only.py
    python experiments/exp5_iql_persona_only.py --residents ross,bob --runs 3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import RUNS_DIR
from retrieval.policy_selector import IQLPolicySelector
from simulation.conversation_loop import run_conversation
from simulation.llm_client import token_tracker


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: IQL policy selection + persona only (no RAG)"
    )
    parser.add_argument("--residents", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=16)
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
    folder = f"exp5_iql_persona_only_{ts}" + (f"_{args.tag}" if args.tag else "")
    exp_dir = RUNS_DIR / folder
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading IQL policy selector …")
    selector = IQLPolicySelector()

    all_results: list = []

    print("=" * 60)
    print(f"  EXPERIMENT 5 — IQL + PERSONA ONLY (NO RAG)")
    print(f"  Residents: {', '.join(residents)}")
    print(f"  Runs/resident: {args.runs}")
    print("=" * 60)

    for resident in residents:
        for run_idx in range(1, args.runs + 1):
            rid = f"exp5_{resident}_run{run_idx}"
            print(f"\n--- {rid} ---")
            result = run_conversation(
                resident_name=resident,
                strategy="iql_persona_only",
                seed_text=args.seed,
                max_turns=args.max_turns,
                selector=selector,
                run_id=rid,
            )
            all_results.append({
                "experiment": "iql_persona_only",
                "resident": resident,
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": result["turns"],
                "path": result["path"],
            })

    summary = {
        "experiment": "exp5_iql_persona_only",
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
    print("  IQL + PERSONA ONLY RESULTS")
    print(f"{'=' * 60}")
    for name, data in summary["per_resident"].items():
        print(f"  {name:<14} {data['successes']}/{data['runs']} = {data['success_rate']:.1%}")
    print(f"  {'OVERALL':<14} {summary['overall_success_rate']:.1%}")
    print(f"\n  Summary → {summary_file}")
    token_tracker.print_summary()


if __name__ == "__main__":
    main()
