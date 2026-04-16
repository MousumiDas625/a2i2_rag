#!/usr/bin/env python3
"""
exp3_iql_policy.py — Experiment 3: IQL Policy Selection + Per-Policy RAG
=========================================================================

PURPOSE:
    The main algorithm.  At each operator turn:
        1. Embed the last N resident utterances → state vector.
        2. The trained IQL Q-network selects the best operator policy.
        3. Query that policy's FAISS index for top-K few-shot examples.
        4. Generate the operator reply with policy-conditioned prompting.

    This should outperform both zero-shot and RAG-all baselines because
    it adapts the communication style to the resident's behaviour.

PREREQUISITES:
    - Full pipeline: P01–P04, I01–I05 must have been run.
    - Trained IQL model: iql/I03_train_iql.py

USAGE:
    python experiments/exp3_iql_policy.py
    python experiments/exp3_iql_policy.py --residents ross,bob --runs 3
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


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: IQL policy selection + per-policy RAG"
    )
    parser.add_argument("--residents", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--tag", default="",
                        help="Optional tag appended to the run folder name")
    parser.add_argument("--seed", default="Hello, this is the fire department. "
                        "We need you to evacuate immediately.")
    args = parser.parse_args()

    residents = (
        [r.strip().lower() for r in args.residents.split(",")]
        if args.residents
        else sorted(PERSONA.keys())
    )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder = f"exp3_iql_policy_{ts}" + (f"_{args.tag}" if args.tag else "")
    exp_dir = RUNS_DIR / folder
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load the IQL selector once (expensive to init)
    print("[INFO] Loading IQL policy selector …")
    selector = IQLPolicySelector()

    all_results: list = []

    print("=" * 60)
    print(f"  EXPERIMENT 3 — IQL POLICY SELECTION + RAG")
    print(f"  Residents: {', '.join(residents)}")
    print(f"  Runs/resident: {args.runs}")
    print("=" * 60)

    for resident in residents:
        for run_idx in range(1, args.runs + 1):
            rid = f"exp3_{resident}_run{run_idx}"
            print(f"\n--- {rid} ---")
            result = run_conversation(
                resident_name=resident,
                strategy="iql_rag",
                seed_text=args.seed,
                max_turns=args.max_turns,
                selector=selector,
                run_id=rid,
            )
            all_results.append({
                "experiment": "iql_policy",
                "resident": resident,
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": result["turns"],
                "path": result["path"],
            })

    summary = {
        "experiment": "exp3_iql_policy",
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
    print("  IQL POLICY SELECTION RESULTS")
    print(f"{'=' * 60}")
    for name, data in summary["per_resident"].items():
        print(f"  {name:<14} {data['successes']}/{data['runs']} = {data['success_rate']:.1%}")
    print(f"  {'OVERALL':<14} {summary['overall_success_rate']:.1%}")
    print(f"\n  Summary → {summary_file}")


if __name__ == "__main__":
    main()
