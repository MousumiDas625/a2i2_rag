#!/usr/bin/env python3
"""
evaluate.py — Unified Evaluation & Comparison
===============================================

PURPOSE:
    Reads the summary JSON files produced by each experiment and
    generates a side-by-side comparison table.

HOW IT WORKS:
    1. Scans data/runs/ for the most recent summary.json from each of
       the three experiments (exp1_zero_shot, exp2_rag_successful,
       exp3_iql_policy).
    2. Extracts per-resident success rates and overall rate.
    3. Prints a comparison table and writes a combined CSV.

USAGE:
    python experiments/evaluate.py
    python experiments/evaluate.py --exp1 path/to/summary.json --exp2 ... --exp3 ...
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RUNS_DIR


def _find_latest_summary(prefix: str) -> Path:
    candidates = sorted(RUNS_DIR.glob(f"{prefix}_*/summary.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No summary.json found for '{prefix}' in {RUNS_DIR}. "
            "Run the experiment first."
        )
    return candidates[0]


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--exp1", default=None, help="Path to exp1 summary.json")
    parser.add_argument("--exp2", default=None, help="Path to exp2 summary.json")
    parser.add_argument("--exp3", default=None, help="Path to exp3 summary.json")
    args = parser.parse_args()

    experiments = {}
    labels = {
        "exp1": "Zero-Shot",
        "exp2": "RAG-Successful",
        "exp3": "IQL+RAG",
    }

    for key, path_arg, prefix in [
        ("exp1", args.exp1, "exp1_zero_shot"),
        ("exp2", args.exp2, "exp2_rag_successful"),
        ("exp3", args.exp3, "exp3_iql_policy"),
    ]:
        try:
            p = Path(path_arg) if path_arg else _find_latest_summary(prefix)
            experiments[key] = _load_summary(p)
            print(f"[OK] Loaded {labels[key]} → {p}")
        except FileNotFoundError as e:
            print(f"[SKIP] {labels[key]}: {e}")

    if not experiments:
        print("[WARN] No experiment results found. Nothing to compare.")
        return

    # Collect all resident names across experiments
    all_residents: set = set()
    for data in experiments.values():
        all_residents.update(data.get("per_resident", {}).keys())
    all_residents_sorted = sorted(all_residents)

    # Print comparison table
    header = f"{'Resident':<14}"
    for key in ["exp1", "exp2", "exp3"]:
        if key in experiments:
            header += f"  {labels[key]:>16}"
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT COMPARISON — SUCCESS RATES")
    print(f"{'=' * 60}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    rows = []
    for resident in all_residents_sorted:
        row = {"resident": resident}
        line = f"  {resident:<14}"
        for key in ["exp1", "exp2", "exp3"]:
            if key in experiments:
                pr = experiments[key].get("per_resident", {})
                rate = pr.get(resident, {}).get("success_rate", None)
                if rate is not None:
                    line += f"  {rate:>15.1%}"
                    row[labels[key]] = rate
                else:
                    line += f"  {'N/A':>15}"
                    row[labels[key]] = "N/A"
        print(line)
        rows.append(row)

    # Overall
    line = f"  {'OVERALL':<14}"
    overall_row = {"resident": "OVERALL"}
    for key in ["exp1", "exp2", "exp3"]:
        if key in experiments:
            rate = experiments[key].get("overall_success_rate", None)
            if rate is not None:
                line += f"  {rate:>15.1%}"
                overall_row[labels[key]] = rate
            else:
                line += f"  {'N/A':>15}"
                overall_row[labels[key]] = "N/A"
    print(f"  {'-' * len(header)}")
    print(line)
    print(f"{'=' * 60}")
    rows.append(overall_row)

    # Write CSV
    csv_path = RUNS_DIR / "comparison.csv"
    fieldnames = ["resident"] + [labels[k] for k in ["exp1", "exp2", "exp3"] if k in experiments]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Comparison CSV → {csv_path}")


if __name__ == "__main__":
    main()
