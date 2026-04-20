#!/usr/bin/env python3
"""
run_all_detailed.py — Run All Experiments with Full Per-Conversation Detail
============================================================================

Runs every experiment 5 times per persona (configurable) and prints:
  1. Per-conversation table per experiment (resident, run, status, turns)
  2. Per-resident summary table per experiment (success rate, avg/min/max turns)
  3. Final cross-experiment summary table at the end

OUTPUT FILES (under data/runs/<tag>/):
  detailed_metrics.csv   — one row per conversation
  final_summary.json     — machine-readable full summary

USAGE:
    python experiments/run_all_detailed.py --tag full_run1
    python experiments/run_all_detailed.py --experiments 1,2,3,3b --runs 3 --tag compare
    python experiments/run_all_detailed.py --test
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

POLICIES = RESIDENTS_LIST

EXPERIMENTS = {
    1:    {"name": "exp1_zero_shot",                  "strategy": "zero_shot",                "needs_iql": False, "is_cross_policy": False},
    2:    {"name": "exp2_rag_successful",              "strategy": "rag_successful",           "needs_iql": False, "is_cross_policy": False},
    3:    {"name": "exp3_iql_rag",                     "strategy": "iql_rag",                  "needs_iql": True,  "is_cross_policy": False},
    "3b": {"name": "exp3b_iql_rag_no_persona",         "strategy": "iql_rag_no_persona",       "needs_iql": True,  "is_cross_policy": False},
    4:    {"name": "exp4_iql_global_rag",              "strategy": "iql_global_rag",           "needs_iql": True,  "is_cross_policy": False},
    5:    {"name": "exp5_iql_persona_only",            "strategy": "iql_persona_only",         "needs_iql": True,  "is_cross_policy": False},
    6:    {"name": "exp6_random_persona",              "strategy": "random_persona",           "needs_iql": False, "is_cross_policy": False},
    7:    {"name": "exp7_cross_policy",                "strategy": "fixed_policy",             "needs_iql": False, "is_cross_policy": True},
    8:    {"name": "exp8_random_rag",                  "strategy": "random_rag",               "needs_iql": False, "is_cross_policy": False},
    9:    {"name": "exp9_random_rag_persona",          "strategy": "random_rag_persona",       "needs_iql": False, "is_cross_policy": False},
    10:   {"name": "exp10_random_global_rag_persona",  "strategy": "random_global_rag_persona","needs_iql": False, "is_cross_policy": False},
}

TRAINING_PERSONAS = set(RESIDENTS_LIST)


def _parse_exp_key(s):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return s


def _persona_type(name):
    return "train" if name in TRAINING_PERSONAS else "new"


def _div(a, b):
    return round(a / b, 4) if b else 0


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_conversation_table(exp_name, rows):
    """Print one row per conversation for a single experiment."""
    print(f"\n  {'─'*70}")
    print(f"  PER-CONVERSATION DETAIL — {exp_name.upper()}")
    print(f"  {'─'*70}")
    header = f"  {'Resident':<14} {'Type':<6} {'Run':>4}  {'Status':<10} {'Turns':>6}  {'Time(s)':>8}"
    print(header)
    print(f"  {'-'*14} {'-'*6} {'-'*4}  {'-'*10} {'-'*6}  {'-'*8}")
    prev_res = None
    for r in rows:
        sep = "  " if r["resident"] == prev_res else "\n  "
        prev_res = r["resident"]
        status_icon = "✓" if r["success"] else "✗"
        print(
            f"{sep}{r['resident']:<14} {_persona_type(r['resident']):<6} "
            f"{r['run']:>4}  {status_icon} {r['status']:<8} "
            f"{r['turns']:>6}  {r.get('elapsed_seconds', 0):>8.1f}"
        )
    print()


def _print_resident_summary_table(exp_name, rows, residents):
    """Print per-resident aggregated stats for one experiment."""
    print(f"  {'─'*70}")
    print(f"  PER-RESIDENT SUMMARY — {exp_name.upper()}")
    print(f"  {'─'*70}")
    header = (
        f"  {'Resident':<14} {'Type':<6} {'Succ/Runs':<12} {'Rate':>6}  "
        f"{'AvgTurns':>9} {'MinTurns':>9} {'MaxTurns':>9} {'AvgTime':>8}"
    )
    print(header)
    print(f"  {'-'*14} {'-'*6} {'-'*12} {'-'*6}  {'-'*9} {'-'*9} {'-'*9} {'-'*8}")

    total_succ = total_runs = 0
    for res in residents:
        res_rows = [r for r in rows if r["resident"] == res]
        if not res_rows:
            continue
        succ  = sum(r["success"] for r in res_rows)
        n     = len(res_rows)
        turns = [r["turns"] for r in res_rows]
        times = [r.get("elapsed_seconds", 0) for r in res_rows]
        rate  = f"{_div(succ, n):.0%}"
        total_succ += succ
        total_runs += n
        print(
            f"  {res:<14} {_persona_type(res):<6} {succ}/{n:<10}  {rate:>6}  "
            f"{sum(turns)/n:>9.1f} {min(turns):>9} {max(turns):>9} "
            f"{sum(times)/n:>8.1f}s"
        )

    overall_rate = f"{_div(total_succ, total_runs):.0%}"
    train_rows = [r for r in rows if _persona_type(r["resident"]) == "train"]
    new_rows   = [r for r in rows if _persona_type(r["resident"]) == "new"]
    train_rate = f"{_div(sum(r['success'] for r in train_rows), len(train_rows)):.0%}" if train_rows else "—"
    new_rate   = f"{_div(sum(r['success'] for r in new_rows),   len(new_rows)):.0%}"   if new_rows   else "—"

    print(f"  {'─'*70}")
    print(f"  {'OVERALL':<14} {'':<6} {total_succ}/{total_runs:<8}  {overall_rate:>6}  "
          f"  (train={train_rate}  new={new_rate})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_num, exp_dir, residents, runs, max_turns, selector, seed):
    cfg      = EXPERIMENTS[exp_num]
    exp_name = cfg["name"]
    strategy = cfg["strategy"]

    sub_dir = exp_dir / exp_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print(f"\n{'═'*72}")
    print(f"  {exp_name.upper().replace('_', ' ')}")
    print(f"  Strategy: {strategy}  |  Residents: {len(residents)}  |  Runs/resident: {runs}")
    print(f"{'═'*72}")

    for resident in residents:
        for run_idx in range(1, runs + 1):
            rid = f"{exp_name}_{resident}_run{run_idx}"
            print(f"\n  [{rid}]")
            result = run_conversation(
                resident_name=resident,
                strategy=strategy,
                seed_text=seed,
                max_turns=max_turns,
                selector=selector if cfg["needs_iql"] else None,
                run_id=rid,
                output_dir=sub_dir,
            )
            all_rows.append({
                "experiment":       exp_name,
                "strategy":         strategy,
                "resident":         resident,
                "persona_type":     _persona_type(resident),
                "policy":           "",
                "run":              run_idx,
                "status":           result["status"],
                "success":          result["success"],
                "turns":            result["turns"],
                "elapsed_seconds":  result.get("elapsed_seconds", 0),
                "path":             result["path"],
            })

    _print_conversation_table(exp_name, all_rows)
    _print_resident_summary_table(exp_name, all_rows, residents)

    summary_file = sub_dir / "summary.json"
    summary_file.write_text(json.dumps(
        {"experiment": exp_name, "results": all_rows}, indent=2, ensure_ascii=False
    ))
    return all_rows


def run_cross_policy_experiment(exp_dir, residents, runs, max_turns, seed):
    exp_name = "exp7_cross_policy"
    sub_dir  = exp_dir / exp_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    matrix   = {res: {pol: [] for pol in POLICIES} for res in residents}
    total    = len(residents) * len(POLICIES) * runs
    done     = 0

    print(f"\n{'═'*72}")
    print(f"  EXP7 CROSS-POLICY MATRIX")
    print(f"  Residents: {len(residents)}  |  Policies: {len(POLICIES)}  |  Runs/pair: {runs}  |  Total: {total}")
    print(f"{'═'*72}")

    for resident in residents:
        for policy in POLICIES:
            for run_idx in range(1, runs + 1):
                done += 1
                rid = f"{exp_name}_{resident}_pol{policy}_run{run_idx}"
                print(f"\n  [{done}/{total}] {rid}")
                result = run_conversation(
                    resident_name=resident,
                    strategy="fixed_policy",
                    fixed_policy_name=policy,
                    seed_text=seed,
                    max_turns=max_turns,
                    run_id=rid,
                    output_dir=sub_dir / f"{resident}_x_{policy}",
                )
                row = {
                    "experiment":      exp_name,
                    "strategy":        "fixed_policy",
                    "resident":        resident,
                    "persona_type":    _persona_type(resident),
                    "policy":          policy,
                    "run":             run_idx,
                    "status":          result["status"],
                    "success":         result["success"],
                    "turns":           result["turns"],
                    "elapsed_seconds": result.get("elapsed_seconds", 0),
                    "path":            result["path"],
                }
                all_rows.append(row)
                matrix[resident][policy].append(result["success"])

    # Per-conversation table
    _print_conversation_table(exp_name, all_rows)

    # Cross-policy matrix table
    col_w = 11
    print(f"  {'─'*72}")
    print(f"  CROSS-POLICY MATRIX (success rate per resident × policy)")
    print(f"  {'─'*72}")
    hdr = f"  {'Resident':<14} {'Type':<6}" + "".join(f"  {p:>{col_w}}" for p in POLICIES)
    print(hdr)
    print(f"  {'-'*14} {'-'*6}" + f"  {'-'*col_w}" * len(POLICIES))
    for res in residents:
        ptype = _persona_type(res)
        row_str = f"  {res:<14} {ptype:<6}"
        for pol in POLICIES:
            vals = matrix[res][pol]
            succ = sum(vals)
            n    = len(vals)
            cell = f"{succ}/{n}={_div(succ,n):.0%}"
            row_str += f"  {cell:>{col_w}}"
        print(row_str)

    train_rows = [r for r in all_rows if _persona_type(r["resident"]) == "train"]
    new_rows   = [r for r in all_rows if _persona_type(r["resident"]) == "new"]
    total_succ = sum(r["success"] for r in all_rows)
    print(f"\n  Overall={_div(total_succ,len(all_rows)):.0%}  "
          f"Train={_div(sum(r['success'] for r in train_rows),len(train_rows)):.0%}  "
          f"New={_div(sum(r['success'] for r in new_rows),len(new_rows)):.0%}\n")

    summary_file = sub_dir / "summary.json"
    summary_file.write_text(json.dumps(
        {"experiment": exp_name, "results": all_rows, "matrix": {
            res: {pol: {"successes": sum(matrix[res][pol]), "runs": len(matrix[res][pol]),
                        "rate": _div(sum(matrix[res][pol]), len(matrix[res][pol]))}
                  for pol in POLICIES} for res in residents}
        }, indent=2, ensure_ascii=False
    ))
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Final summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_final_summary(all_rows_by_exp, residents, exp_order):
    """Print the big cross-experiment × resident success-rate table."""

    # ── Per-experiment × per-resident grid ───────────────────────────────────
    print(f"\n{'═'*100}")
    print("  FINAL SUMMARY — SUCCESS RATE PER EXPERIMENT × RESIDENT")
    print(f"{'═'*100}")

    exp_names  = [EXPERIMENTS[k]["name"] for k in exp_order if k in EXPERIMENTS]
    short_names = [n.replace("exp", "").replace("_", " ") for n in exp_names]

    col = 10
    header  = f"  {'Resident':<14} {'Type':<6}"
    for sn in short_names:
        header += f"  {sn[:col]:>{col}}"
    print(header)
    print(f"  {'-'*14} {'-'*6}" + f"  {'-'*col}" * len(short_names))

    for res in residents:
        ptype = _persona_type(res)
        row_str = f"  {res:<14} {ptype:<6}"
        for k in exp_order:
            if k not in EXPERIMENTS:
                continue
            rows = [r for r in all_rows_by_exp.get(k, []) if r["resident"] == res]
            if not rows:
                row_str += f"  {'—':>{col}}"
            else:
                succ = sum(r["success"] for r in rows)
                n    = len(rows)
                row_str += f"  {f'{succ}/{n}={_div(succ,n):.0%}':>{col}}"
        print(row_str)

    # Totals row
    print(f"  {'-'*14} {'-'*6}" + f"  {'-'*col}" * len(short_names))
    for label, filt in [("OVERALL", None), ("TRAIN", "train"), ("NEW", "new")]:
        row_str = f"  {label:<14} {'':<6}"
        for k in exp_order:
            if k not in EXPERIMENTS:
                continue
            rows = all_rows_by_exp.get(k, [])
            if filt:
                rows = [r for r in rows if _persona_type(r["resident"]) == filt]
            if not rows:
                row_str += f"  {'—':>{col}}"
            else:
                succ = sum(r["success"] for r in rows)
                row_str += f"  {f'{_div(succ,len(rows)):.0%}':>{col}}"
        print(row_str)

    # ── Per-experiment avg turns table ───────────────────────────────────────
    print(f"\n{'═'*100}")
    print("  FINAL SUMMARY — AVG TURNS PER EXPERIMENT × RESIDENT")
    print(f"{'═'*100}")
    print(header)
    print(f"  {'-'*14} {'-'*6}" + f"  {'-'*col}" * len(short_names))

    for res in residents:
        ptype = _persona_type(res)
        row_str = f"  {res:<14} {ptype:<6}"
        for k in exp_order:
            if k not in EXPERIMENTS:
                continue
            rows = [r for r in all_rows_by_exp.get(k, []) if r["resident"] == res]
            if not rows:
                row_str += f"  {'—':>{col}}"
            else:
                avg = sum(r["turns"] for r in rows) / len(rows)
                row_str += f"  {avg:>{col}.1f}"
        print(row_str)

    print(f"  {'-'*14} {'-'*6}" + f"  {'-'*col}" * len(short_names))
    row_str = f"  {'AVG TURNS':<14} {'':<6}"
    for k in exp_order:
        if k not in EXPERIMENTS:
            continue
        rows = all_rows_by_exp.get(k, [])
        if not rows:
            row_str += f"  {'—':>{col}}"
        else:
            avg = sum(r["turns"] for r in rows) / len(rows)
            row_str += f"  {avg:>{col}.1f}"
    print(row_str)

    # ── Compact single summary line per experiment ────────────────────────────
    print(f"\n{'═'*100}")
    print("  EXPERIMENT LEADERBOARD")
    print(f"{'═'*100}")
    print(f"  {'Experiment':<35} {'Overall':>8} {'Train':>8} {'New':>8} {'AvgTurns':>10} {'MinTurns':>10} {'MaxTurns':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    leaderboard = []
    for k in exp_order:
        if k not in EXPERIMENTS:
            continue
        rows       = all_rows_by_exp.get(k, [])
        name       = EXPERIMENTS[k]["name"]
        train_rows = [r for r in rows if _persona_type(r["resident"]) == "train"]
        new_rows   = [r for r in rows if _persona_type(r["resident"]) == "new"]
        overall    = _div(sum(r["success"] for r in rows), len(rows)) if rows else 0
        train_rate = _div(sum(r["success"] for r in train_rows), len(train_rows)) if train_rows else 0
        new_rate   = _div(sum(r["success"] for r in new_rows), len(new_rows)) if new_rows else 0
        turns      = [r["turns"] for r in rows] if rows else [0]
        leaderboard.append((name, overall, train_rate, new_rate, turns))

    leaderboard.sort(key=lambda x: x[1], reverse=True)
    for name, overall, train_rate, new_rate, turns in leaderboard:
        avg_t = sum(turns) / len(turns)
        print(
            f"  {name:<35} {overall:>8.0%} {train_rate:>8.0%} {new_rate:>8.0%} "
            f"{avg_t:>10.1f} {min(turns):>10} {max(turns):>10}"
        )
    print(f"{'═'*100}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(all_rows, csv_path):
    fieldnames = [
        "experiment", "strategy", "resident", "policy", "persona_type",
        "run", "status", "success", "turns", "elapsed_seconds", "path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments with full per-conversation detail"
    )
    parser.add_argument("--residents", default=None,
                        help="Comma-separated residents (default: all 10)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per resident per experiment (default: 5)")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--experiments", default=None,
                        help="e.g. '1,2,3,3b' (default: all)")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 1 run, first resident only")
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

    exp_order = (
        [_parse_exp_key(x) for x in args.experiments.split(",")]
        if args.experiments
        else sorted(EXPERIMENTS.keys(), key=lambda k: (int(str(k).rstrip("abcdefgh")), str(k)))
    )

    ts_label    = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder_name = args.tag if args.tag else f"exp_run_{ts_label}"
    exp_dir     = RUNS_DIR / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output folder : {exp_dir}")
    print(f"  Residents     : {', '.join(residents)}")
    print(f"  Experiments   : {exp_order}")
    print(f"  Runs/resident : {args.runs}")
    print(f"  Max turns     : {args.max_turns}\n")

    # Load IQL selector once if any experiment needs it
    needs_iql = any(EXPERIMENTS[k]["needs_iql"] for k in exp_order if k in EXPERIMENTS)
    selector  = None
    if needs_iql:
        print("[INFO] Loading IQL policy selector …")
        from retrieval.policy_selector import IQLPolicySelector
        selector = IQLPolicySelector()

    ts_start          = datetime.now()
    all_rows_by_exp   = {}
    all_rows_flat     = []

    for k in exp_order:
        if k not in EXPERIMENTS:
            print(f"[WARN] Unknown experiment key '{k}' — skipping.")
            continue
        t0  = time.time()
        cfg = EXPERIMENTS[k]

        if cfg["is_cross_policy"]:
            rows = run_cross_policy_experiment(
                exp_dir, residents, args.runs, args.max_turns, args.seed
            )
        else:
            rows = run_experiment(
                k, exp_dir, residents, args.runs, args.max_turns,
                selector, args.seed
            )

        elapsed = round(time.time() - t0, 1)
        print(f"  [{cfg['name']} done in {elapsed}s]\n")

        all_rows_by_exp[k] = rows
        all_rows_flat.extend(rows)

    # ── Write outputs ─────────────────────────────────────────────────────────
    csv_path = exp_dir / "detailed_metrics.csv"
    _write_csv(all_rows_flat, csv_path)

    final_json = {
        "timestamp":        ts_start.strftime("%Y%m%dT%H%M%S"),
        "output_folder":    str(exp_dir),
        "residents":        residents,
        "training_personas": sorted(TRAINING_PERSONAS),
        "new_personas":     sorted(set(residents) - TRAINING_PERSONAS),
        "runs_per_resident": args.runs,
        "experiments_run":  [str(k) for k in exp_order],
        "token_usage": {
            "prompt_tokens":      token_tracker.prompt_tokens,
            "completion_tokens":  token_tracker.completion_tokens,
            "total_tokens":       token_tracker.total_tokens,
            "num_llm_calls":      token_tracker.num_calls,
            "estimated_cost_usd": round(token_tracker.total_cost_usd(), 6),
        },
    }
    final_file = exp_dir / "final_summary.json"
    final_file.write_text(json.dumps(final_json, indent=2, ensure_ascii=False))

    # ── Print final tables ────────────────────────────────────────────────────
    _print_final_summary(all_rows_by_exp, residents, exp_order)

    print(f"  Detailed CSV   → {csv_path}")
    print(f"  JSON summary   → {final_file}")
    print(f"  Dialogues      → {exp_dir}/\n")
    token_tracker.print_summary()


if __name__ == "__main__":
    main()
