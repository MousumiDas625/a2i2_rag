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
    python experiments/run_all_final.py --experiments all --output-name "hopefully final"
    # Quick smoke test: first two residents (alphabetically), one run each, all exps:
    python experiments/run_all_final.py --experiments all --sample-residents 2 --runs 1
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import RUNS_DIR
from simulation.conversation_loop import run_conversation

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


def _parse_experiment_list(raw: Optional[str]) -> List[int]:
    if raw is None or str(raw).strip().lower() == "all":
        return sorted(EXPERIMENTS.keys())
    nums = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() == "all":
            return sorted(EXPERIMENTS.keys())
        nums.append(int(x))
    return nums


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments into data/runs/<output-name>/ (default: exp_final)"
    )
    parser.add_argument("--residents", default=None,
                        help="Comma-separated residents (default: all personas; "
                             "overrides --sample-residents)")
    parser.add_argument(
        "--sample-residents",
        type=int,
        default=None,
        metavar="N",
        help="When --residents is omitted, use only the first N personas "
             "(alphabetical by key), e.g. 2 for a quick test",
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per resident per experiment (default: 3)")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated exp numbers 1-7, or the word 'all' (default: all)",
    )
    parser.add_argument(
        "--output-name",
        default="exp_final",
        help='Folder name under data/runs/ (default: exp_final). Example: "hopefully final"',
    )
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()

    all_keys = sorted(PERSONA.keys())
    if args.residents:
        residents = [r.strip().lower() for r in args.residents.split(",") if r.strip()]
    elif args.sample_residents is not None:
        n = max(1, args.sample_residents)
        residents = all_keys[: min(n, len(all_keys))]
    else:
        residents = all_keys

    unknown = [r for r in residents if r not in PERSONA]
    if unknown:
        parser.error(f"Unknown resident(s): {unknown}. Known: {all_keys}")

    try:
        exp_nums = _parse_experiment_list(args.experiments)
    except ValueError as e:
        parser.error(f"Invalid --experiments: {args.experiments!r} ({e})")

    bad_exp = [n for n in exp_nums if n not in EXPERIMENTS]
    if bad_exp:
        parser.error(f"Invalid experiment number(s): {bad_exp}. Valid: {sorted(EXPERIMENTS)}")

    exp_dir = RUNS_DIR / args.output_name.strip()
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


if __name__ == "__main__":
    main()
