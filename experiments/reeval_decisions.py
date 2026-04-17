#!/usr/bin/env python3
"""
reeval_decisions.py — Re-evaluate existing conversations with updated decision judge.

Reads all dialogue JSONL files from specified run folders, re-runs
is_successful_session on each, and reports updated success rates.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.decision import is_successful_session

BASE = Path(__file__).resolve().parent.parent / "data" / "runs"

RUN_DIRS = [
    "exp_run_20260416T184447",  # exp1
    "exp_run_20260416T184459",  # exp2
    "exp_run_20260416T184503",  # exp3
    "exp_run_20260416T184508",  # exp4
    "exp_run_20260416T184512",  # exp5
    "exp_run_20260416T184613",  # exp6
]

TRAINING = {"bob", "michelle", "ross", "niki", "lindsay"}


def load_conversation(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reeval_run_dir(run_dir: Path):
    exp_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    results = defaultdict(list)

    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        for jsonl_file in sorted(exp_dir.glob("dialogue_*.jsonl")):
            history = load_conversation(jsonl_file)
            if not history:
                continue

            # Extract resident name from filename
            parts = jsonl_file.stem.split("_")
            resident = parts[1] if len(parts) > 1 else "unknown"

            decision, _ = is_successful_session(
                history, max_turns=len(history) + 1, allow_early_stop=True
            )
            success = 1 if decision is True else 0

            results[exp_name].append({
                "resident": resident,
                "success": success,
                "file": jsonl_file.name,
            })

    return results


def print_summary(all_results):
    print("\n" + "=" * 75)
    print("  UPDATED SUCCESS RATES (re-evaluated with new decision judge)")
    print("=" * 75)
    header = f"  {'Experiment':<30} {'Overall':>8}  {'Training':>10}  {'New':>8}"
    print(header)
    print(f"  {'-'*30} {'-'*8}  {'-'*10}  {'-'*8}")

    for exp_name, rows in sorted(all_results.items()):
        total = len(rows)
        if total == 0:
            continue
        succ = sum(r["success"] for r in rows)
        train_rows = [r for r in rows if r["resident"] in TRAINING]
        new_rows   = [r for r in rows if r["resident"] not in TRAINING]
        train_rate = sum(r["success"] for r in train_rows) / len(train_rows) if train_rows else 0
        new_rate   = sum(r["success"] for r in new_rows)   / len(new_rows)   if new_rows   else 0
        overall    = succ / total
        print(f"  {exp_name:<30} {overall:>7.0%}   {train_rate:>9.0%}   {new_rate:>7.0%}")

    print("=" * 75)


def main():
    all_results = defaultdict(list)

    for run_dir_name in RUN_DIRS:
        run_dir = BASE / run_dir_name
        if not run_dir.exists():
            print(f"[WARN] Not found: {run_dir}")
            continue
        print(f"[Processing] {run_dir_name} ...", flush=True)
        exp_results = reeval_run_dir(run_dir)
        for exp_name, rows in exp_results.items():
            all_results[exp_name].extend(rows)

    print_summary(all_results)


if __name__ == "__main__":
    main()
