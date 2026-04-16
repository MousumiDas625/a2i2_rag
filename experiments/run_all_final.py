#!/usr/bin/env python3
"""
run_all_final.py — Run All 9 Experiments (3 runs per persona by default)
=======================================================================

Creates:
    data/runs/exp_final/
    ├── exp1_zero_shot/
    ├── exp2_rag_successful/
    ├── exp3_iql_rag/
    ├── exp4_iql_global_rag/
    ├── exp5_iql_persona_only/
    ├── exp6_random_persona/
    ├── exp7_random_local_rag/
    ├── exp8_random_local_rag_persona/
    ├── exp9_random_persona_global_rag/
    └── final_summary.json

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
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    7: {"name": "exp7_random_local_rag", "strategy": "random_local_rag", "needs_iql": False},
    8: {"name": "exp8_random_local_rag_persona", "strategy": "random_local_rag_persona", "needs_iql": False},
    9: {"name": "exp9_random_persona_global_rag", "strategy": "random_persona_global_rag", "needs_iql": False},
}


def _turn_stats(turns: List[int]) -> Dict[str, float]:
    if not turns:
        return {"mean_turns": 0.0, "min_turns": 0, "max_turns": 0, "stdev_turns": 0.0}
    return {
        "mean_turns": round(statistics.mean(turns), 3),
        "min_turns": min(turns),
        "max_turns": max(turns),
        "stdev_turns": round(statistics.stdev(turns), 3) if len(turns) > 1 else 0.0,
    }


def _build_summary(exp_name, results, residents):
    summary: Dict[str, Any] = {
        "experiment": exp_name,
        "results": results,
        "per_resident": {},
    }
    all_turns: List[int] = []
    for res in residents:
        runs = [r for r in results if r["resident"] == res]
        succ = sum(r["success"] for r in runs)
        turns = [int(r["turns"]) for r in runs]
        all_turns.extend(turns)
        pr: Dict[str, Any] = {
            "runs": len(runs),
            "successes": succ,
            "success_rate": round(succ / len(runs), 4) if runs else 0,
        }
        pr.update(_turn_stats(turns))
        summary["per_resident"][res] = pr
    total_succ = sum(r["success"] for r in results)
    summary["overall_success_rate"] = (
        round(total_succ / len(results), 4) if results else 0
    )
    summary.update(_turn_stats(all_turns))
    summary["total_conversations"] = len(results)
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

            turns = int(result["turns"])
            all_results.append({
                "experiment": exp_name,
                "resident": resident,
                "run": run_idx,
                "status": result["status"],
                "success": result["success"],
                "turns": turns,
                "path": result["path"],
            })
            print(f"  → turns={turns} success={result['success']} status={result['status']}")

    summary = _build_summary(exp_name, all_results, residents)
    summary_file = sub_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n  --- {exp_name} ---")
    for name, data in summary["per_resident"].items():
        mt = data.get("mean_turns", 0)
        print(
            f"  {name:<14} {data['successes']}/{data['runs']} = {data['success_rate']:.1%}  "
            f"| mean turns {mt:.1f}"
        )
    print(
        f"  {'OVERALL':<14} {summary['overall_success_rate']:.1%}  "
        f"| mean turns {summary.get('mean_turns', 0):.1f}"
    )
    print(f"  Summary → {summary_file}\n")

    return summary


def write_master_run_log(
    exp_dir: Path, all_summaries: Dict[str, Dict], exp_nums: List[int],
) -> Path:
    """Single CSV of every conversation (turns + success) across experiments."""
    path = exp_dir / "all_conversations.csv"
    fieldnames = [
        "experiment", "resident", "run", "turns", "success", "status", "path",
    ]
    names_in_order = [
        EXPERIMENTS[n]["name"]
        for n in sorted(exp_nums)
        if EXPERIMENTS[n]["name"] in all_summaries
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for exp_name in names_in_order:
            summ = all_summaries[exp_name]
            for row in summ.get("results", []):
                w.writerow(
                    {
                        "experiment": row.get("experiment", exp_name),
                        "resident": row["resident"],
                        "run": row["run"],
                        "turns": row["turns"],
                        "success": row["success"],
                        "status": row["status"],
                        "path": row["path"],
                    }
                )
    return path


def generate_charts(
    exp_dir: Path,
    all_summaries: Dict[str, Dict],
    residents: List[str],
    exp_nums: List[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordered: List[Tuple[str, str, Dict]] = []
    for n in sorted(exp_nums):
        name = EXPERIMENTS[n]["name"]
        if name in all_summaries:
            ordered.append((f"Exp {n}", name, all_summaries[name]))
    if not ordered:
        print("[Charts] No summaries to plot.")
        return

    labels = [x[0] for x in ordered]
    rates = [x[2]["overall_success_rate"] * 100 for x in ordered]
    means = [x[2].get("mean_turns", 0) for x in ordered]

    x = np.arange(len(labels))

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x, rates, color="steelblue", alpha=0.88)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Success rate by experiment (all residents × all runs)")
    for i, v in enumerate(rates):
        ax1.text(i, min(v + 2, 102), f"{v:.0f}%", ha="center", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(exp_dir / "chart_success_rate_by_experiment.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 4.5))
    ax2.bar(x, means, color="darkseagreen", alpha=0.9)
    ax2.set_ylabel("Mean dialogue turns")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Mean turns by experiment")
    fig2.tight_layout()
    fig2.savefig(exp_dir / "chart_mean_turns_by_experiment.png", dpi=150)
    plt.close(fig2)

    res_sorted = sorted(residents)
    mat = np.zeros((len(res_sorted), len(ordered)))
    for j, (_, _, summ) in enumerate(ordered):
        for i, res in enumerate(res_sorted):
            pr = summ["per_resident"].get(res, {})
            mat[i, j] = pr.get("success_rate", 0) * 100

    fig3, ax3 = plt.subplots(figsize=(max(10.5, len(ordered) * 1.05), 6.2))
    im = ax3.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax3.set_yticks(np.arange(len(res_sorted)))
    ax3.set_yticklabels(res_sorted)
    ax3.set_xticks(np.arange(len(ordered)))
    ax3.set_xticklabels(labels, rotation=22, ha="right")
    ax3.set_title("Success rate (%) — resident × experiment")
    cbar = fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("% success")
    for i in range(len(res_sorted)):
        for j in range(len(ordered)):
            ax3.text(
                j, i, f"{mat[i, j]:.0f}",
                ha="center", va="center", color="black", fontsize=8,
            )
    fig3.tight_layout()
    fig3.savefig(exp_dir / "chart_success_rate_heatmap_resident_experiment.png", dpi=150)
    plt.close(fig3)

    exp_count = len(ordered)
    res_rates: List[float] = []
    for res in res_sorted:
        total = 0.0
        for _, _, summ in ordered:
            pr = summ["per_resident"].get(res, {})
            total += pr.get("success_rate", 0)
        res_rates.append((total / exp_count) * 100 if exp_count else 0.0)

    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    x4 = np.arange(len(res_sorted))
    ax4.bar(x4, res_rates, color="coral", alpha=0.88)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(res_sorted, rotation=28, ha="right")
    ax4.set_ylabel("Mean success rate across experiments (%)")
    ax4.set_title("Residents — average success rate over all experiments")
    ax4.set_ylim(0, 105)
    fig4.tight_layout()
    fig4.savefig(exp_dir / "chart_success_rate_by_resident.png", dpi=150)
    plt.close(fig4)

    print(f"\n[Charts] Saved under {exp_dir}:")
    print("  chart_success_rate_by_experiment.png")
    print("  chart_mean_turns_by_experiment.png")
    print("  chart_success_rate_heatmap_resident_experiment.png")
    print("  chart_success_rate_by_resident.png")


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
        help="Comma-separated exp numbers 1-9, or the word 'all' (default: all)",
    )
    parser.add_argument(
        "--output-name",
        default="exp_final",
        help='Folder name under data/runs/ (default: exp_final). Example: "hopefully final"',
    )
    parser.add_argument("--seed", default=SEED)
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Do not write PNG charts (still writes CSV + JSON).",
    )
    args = parser.parse_args()

    # When stdout is piped (e.g. `| tee log.txt`), Python uses block buffering and
    # nothing appears for a long time. Prefer line buffering so logs show up live.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass

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

    print(
        f"\n[BOOT] {datetime.now().isoformat()}  output_dir={exp_dir}\n"
        f"        experiments={exp_nums}  residents={len(residents)}  "
        f"runs_each={args.runs}\n",
        flush=True,
    )

    needs_iql = any(EXPERIMENTS[n]["needs_iql"] for n in exp_nums)
    selector = None
    if needs_iql:
        print(
            "[INFO] Loading IQL policy selector (can take 1–3+ min on first load) …",
            flush=True,
        )
        from retrieval.policy_selector import IQLPolicySelector
        selector = IQLPolicySelector()
        print("[INFO] IQL policy selector ready.", flush=True)

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

    csv_path = write_master_run_log(exp_dir, all_summaries, exp_nums)
    print(f"\n  Per-conversation log (all turns) → {csv_path}")

    if not args.skip_charts:
        try:
            generate_charts(exp_dir, all_summaries, residents, exp_nums)
        except Exception as e:
            print(f"\n[WARN] Chart generation failed: {e}")

    print("\n" + "=" * 60)
    print("  FINAL RESULTS — ALL EXPERIMENTS")
    print("=" * 60)
    for name, s in all_summaries.items():
        rate = s["overall_success_rate"]
        elapsed = s.get("elapsed_seconds", "?")
        mt = s.get("mean_turns", "?")
        print(f"  {name:<30} {rate:.1%}   mean_turns={mt}   ({elapsed}s)")
    print(f"\n  Full summary → {final_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
