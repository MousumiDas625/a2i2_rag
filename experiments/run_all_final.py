#!/usr/bin/env python3
"""
run_all_final.py — Run All 7 Experiments with Detailed Metrics
===============================================================

Creates a timestamped output folder:
    data/runs/exp_run_<TIMESTAMP>/
    ├── exp1_zero_shot/           # conversation JSONL files + summary.json
    ├── exp2_rag_successful/
    ├── exp3_iql_rag/
    ├── exp4_iql_global_rag/
    ├── exp5_iql_persona_only/
    ├── exp6_random_persona/
    ├── exp7_random_no_persona/
    ├── final_summary.json        # combined results across all experiments
    └── detailed_metrics.csv      # per-conversation metrics with persona types

USAGE:
    python experiments/run_all_final.py
    python experiments/run_all_final.py --residents ross,bob
    python experiments/run_all_final.py --runs 5
    python experiments/run_all_final.py --experiments 1,3,5
    python experiments/run_all_final.py --test
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import RUNS_DIR, RESIDENTS_LIST
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

TRAINING_PERSONAS = set(RESIDENTS_LIST)


def _persona_type(name: str) -> str:
    return "training" if name in TRAINING_PERSONAS else "new"


def _build_summary(exp_name, results, residents):
    summary = {
        "experiment": exp_name,
        "results": results,
        "per_resident": {},
    }
    for res in residents:
        runs = [r for r in results if r["resident"] == res]
        succ = sum(r["success"] for r in runs)
        avg_turns = round(sum(r["turns"] for r in runs) / len(runs), 1) if runs else 0
        avg_time = round(sum(r["elapsed_seconds"] for r in runs) / len(runs), 1) if runs else 0
        summary["per_resident"][res] = {
            "persona_type": _persona_type(res),
            "runs": len(runs),
            "successes": succ,
            "success_rate": round(succ / len(runs), 4) if runs else 0,
            "avg_turns": avg_turns,
            "avg_time_seconds": avg_time,
        }
    total_succ = sum(r["success"] for r in results)
    summary["overall_success_rate"] = (
        round(total_succ / len(results), 4) if results else 0
    )

    training_runs = [r for r in results if _persona_type(r["resident"]) == "training"]
    new_runs = [r for r in results if _persona_type(r["resident"]) == "new"]
    summary["training_persona_success_rate"] = (
        round(sum(r["success"] for r in training_runs) / len(training_runs), 4)
        if training_runs else 0
    )
    summary["new_persona_success_rate"] = (
        round(sum(r["success"] for r in new_runs) / len(new_runs), 4)
        if new_runs else 0
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
                "strategy": strategy,
                "resident": resident,
                "persona_type": _persona_type(resident),
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": result["turns"],
                "elapsed_seconds": result.get("elapsed_seconds", 0),
                "path": result["path"],
            })

    summary = _build_summary(exp_name, all_results, residents)
    summary_file = sub_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n  --- {exp_name} ---")
    print(f"  {'Resident':<14} {'Type':<10} {'Rate':<12} {'Avg Turns':<12} {'Avg Time'}")
    print(f"  {'-'*14} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for name, data in summary["per_resident"].items():
        ptype = data["persona_type"]
        rate = f"{data['successes']}/{data['runs']} = {data['success_rate']:.0%}"
        print(f"  {name:<14} {ptype:<10} {rate:<12} {data['avg_turns']:<12} {data['avg_time_seconds']:.1f}s")
    print(f"\n  Overall: {summary['overall_success_rate']:.0%}  |  "
          f"Training: {summary['training_persona_success_rate']:.0%}  |  "
          f"New: {summary['new_persona_success_rate']:.0%}")
    print(f"  Summary → {summary_file}\n")

    return summary, all_results


def _write_csv(all_rows, csv_path):
    fieldnames = [
        "experiment", "strategy", "resident", "persona_type",
        "run", "status", "success", "turns", "elapsed_seconds", "path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments (5 runs/persona) with detailed metrics"
    )
    parser.add_argument("--residents", default=None,
                        help="Comma-separated residents (default: all 10)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per resident per experiment (default: 5)")
    parser.add_argument("--max-turns", type=int, default=15)
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

    ts_label = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_dir = RUNS_DIR / f"exp_run_{ts_label}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output folder: {exp_dir}\n")

    needs_iql = any(EXPERIMENTS[n]["needs_iql"] for n in exp_nums)
    selector = None
    if needs_iql:
        print("[INFO] Loading IQL policy selector (shared across IQL experiments) …")
        from retrieval.policy_selector import IQLPolicySelector
        selector = IQLPolicySelector()

    ts_start = datetime.now()
    all_summaries = {}
    all_csv_rows = []

    for exp_num in exp_nums:
        t0 = time.time()
        summary, rows = run_experiment(
            exp_num, exp_dir, residents, args.runs, args.max_turns,
            selector, args.seed,
        )
        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        all_summaries[EXPERIMENTS[exp_num]["name"]] = summary
        all_csv_rows.extend(rows)

    # ── Write detailed CSV ────────────────────────────────────────────────────
    csv_path = exp_dir / "detailed_metrics.csv"
    _write_csv(all_csv_rows, csv_path)

    # ── Write final JSON summary ──────────────────────────────────────────────
    final = {
        "timestamp": ts_start.strftime("%Y%m%dT%H%M%S"),
        "output_folder": str(exp_dir),
        "residents": residents,
        "training_personas": sorted(TRAINING_PERSONAS),
        "new_personas": sorted(set(residents) - TRAINING_PERSONAS),
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

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RESULTS — ALL EXPERIMENTS")
    print("=" * 80)
    header = f"  {'Experiment':<30} {'Overall':<10} {'Training':<10} {'New':<10} {'Time'}"
    print(header)
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, s in all_summaries.items():
        overall = f"{s['overall_success_rate']:.0%}"
        train = f"{s['training_persona_success_rate']:.0%}"
        new = f"{s['new_persona_success_rate']:.0%}"
        elapsed = f"{s.get('elapsed_seconds', '?')}s"
        print(f"  {name:<30} {overall:<10} {train:<10} {new:<10} {elapsed}")

    print(f"\n  Training personas : {', '.join(sorted(TRAINING_PERSONAS))}")
    print(f"  New personas      : {', '.join(sorted(set(residents) - TRAINING_PERSONAS))}")
    print(f"\n  Conversations     → {exp_dir}/")
    print(f"  Detailed CSV      → {csv_path}")
    print(f"  JSON summary      → {final_file}")
    print("=" * 80)

    token_tracker.print_summary()


if __name__ == "__main__":
    main()
