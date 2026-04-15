#!/usr/bin/env python3
"""
run_all_final.py — Run All 7 Experiments (3 runs per persona)
==============================================================

Creates:
    data/runs/exp_final/
    ├── exp1_zero_shot/           # conversations + summary.json
    ├── exp2_rag_successful/
    ├── exp3_iql_rag/
    ├── exp4_iql_global_rag/
    ├── exp5_iql_persona_only/
    ├── exp6_random_persona/
    ├── exp7_random_no_persona/
    └── final_summary.json        # combined results across all experiments

USAGE:
    python experiments/run_all_final.py
    python experiments/run_all_final.py --residents ross,bob
    python experiments/run_all_final.py --runs 3
    python experiments/run_all_final.py --experiments 1,3,5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import RUNS_DIR
from simulation.conversation_loop import run_conversation
from simulation.llm_client import token_tracker

SEED = ("Hello, this is the fire department. "
        "We need you to evacuate immediately.")

EXPERIMENTS = {
    1: {"name": "exp1_zero_shot",         "strategy": "zero_shot",         "needs_iql": False},
    2: {"name": "exp2_rag_successful",    "strategy": "rag_successful",    "needs_iql": False},
    3: {"name": "exp3_iql_rag",           "strategy": "iql_rag",           "needs_iql": True},
    4: {"name": "exp4_iql_global_rag",    "strategy": "iql_global_rag",    "needs_iql": True},
    5: {"name": "exp5_iql_persona_only",  "strategy": "iql_persona_only",  "needs_iql": True},
    6: {"name": "exp6_random_persona",    "strategy": "random_persona",    "needs_iql": False},
    7: {"name": "exp7_random_no_persona", "strategy": "random_no_persona", "needs_iql": False},
}


def _build_summary(exp_name, results, residents):
    summary = {
        "experiment": exp_name,
        "results": results,
        "per_resident": {},
    }
    for res in residents:
        runs = [r for r in results if r["resident"] == res]
        succ = sum(r["success"] for r in runs)
        summary["per_resident"][res] = {
            "runs": len(runs),
            "successes": succ,
            "success_rate": round(succ / len(runs), 4) if runs else 0,
        }
    total_succ = sum(r["success"] for r in results)
    summary["overall_success_rate"] = (
        round(total_succ / len(results), 4) if results else 0
    )
    return summary


def run_experiment(exp_num, exp_dir, residents, runs, max_turns, selector, seed):
    cfg = EXPERIMENTS[exp_num]
    exp_name = cfg["name"]
    strategy = cfg["strategy"]

    sub_dir = exp_dir / exp_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"\n{'=' * 60}")
    print(f"  {exp_name.upper().replace('_', ' ')}")
    print(f"  Strategy: {strategy}")
    print(f"  Residents: {', '.join(residents)}")
    print(f"  Runs/resident: {runs}")
    print(f"{'=' * 60}")

    for resident in residents:
        for run_idx in range(1, runs + 1):
            rid = f"{exp_name}_{resident}_run{run_idx}"
            print(f"\n--- {rid} ---")

            result = run_conversation(
                resident_name=resident,
                strategy=strategy,
                seed_text=seed,
                max_turns=max_turns,
                selector=selector if cfg["needs_iql"] else None,
                run_id=rid,
                output_dir=sub_dir,
            )

            all_results.append({
                "experiment": exp_name,
                "resident": resident,
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": result["turns"],
                "path": result["path"],
            })

    summary = _build_summary(exp_name, all_results, residents)
    summary_file = sub_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n  --- {exp_name} ---")
    for name, data in summary["per_resident"].items():
        print(f"  {name:<14} {data['successes']}/{data['runs']} = {data['success_rate']:.1%}")
    print(f"  {'OVERALL':<14} {summary['overall_success_rate']:.1%}")
    print(f"  Summary → {summary_file}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments (3 runs/persona) into exp_final/"
    )
    parser.add_argument("--residents", default=None,
                        help="Comma-separated residents (default: all 10)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per resident per experiment (default: 3)")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--experiments", default=None,
                        help="Comma-separated exp numbers to run, e.g. '1,3,5' "
                             "(default: all 1-7)")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: 1 run with first resident only "
                             "(overrides --runs and --residents)")
    args = parser.parse_args()

    if args.test:
        args.runs = 1

    residents = (
        [r.strip().lower() for r in args.residents.split(",")]
        if args.residents
        else sorted(PERSONA.keys())
    )
    if args.test and not args.residents:
        residents = residents[:1]

    exp_nums = (
        [int(x.strip()) for x in args.experiments.split(",")]
        if args.experiments
        else sorted(EXPERIMENTS.keys())
    )

    exp_dir = RUNS_DIR / "exp_final"
    exp_dir.mkdir(parents=True, exist_ok=True)

    needs_iql = any(EXPERIMENTS[n]["needs_iql"] for n in exp_nums)
    selector = None
    if needs_iql:
        print("[INFO] Loading IQL policy selector (shared across IQL experiments) …")
        from retrieval.policy_selector import IQLPolicySelector
        selector = IQLPolicySelector()

    ts_start = datetime.now()
    all_summaries = {}

    for exp_num in exp_nums:
        t0 = time.time()
        summary = run_experiment(
            exp_num, exp_dir, residents, args.runs, args.max_turns,
            selector, args.seed,
        )
        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        all_summaries[EXPERIMENTS[exp_num]["name"]] = summary

    final = {
        "timestamp": ts_start.strftime("%Y%m%dT%H%M%S"),
        "residents": residents,
        "runs_per_resident": args.runs,
        "experiments": all_summaries,
        "token_usage": {
            "prompt_tokens": token_tracker.prompt_tokens,
            "completion_tokens": token_tracker.completion_tokens,
            "total_tokens": token_tracker.total_tokens,
            "num_llm_calls": token_tracker.num_calls,
            "estimated_cost_usd": round(token_tracker.total_cost_usd(), 6),
        },
    }
    final_file = exp_dir / "final_summary.json"
    final_file.write_text(json.dumps(final, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("  FINAL RESULTS — ALL EXPERIMENTS")
    print("=" * 60)
    for name, s in all_summaries.items():
        rate = s["overall_success_rate"]
        elapsed = s.get("elapsed_seconds", "?")
        print(f"  {name:<30} {rate:.1%}   ({elapsed}s)")
    print(f"\n  Full summary → {final_file}")
    print("=" * 60)

    token_tracker.print_summary()


if __name__ == "__main__":
    main()
