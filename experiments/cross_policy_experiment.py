#!/usr/bin/env python3
"""
cross_policy_experiment.py — Cross-Policy Matrix Experiment
=============================================================

PURPOSE:
    Tests every resident against every training policy for a full
    conversation, 3 runs each.  Produces a resident × policy success-rate
    matrix showing how each resident responds when a specific policy is
    applied to them for the entire conversation.

    Residents  : all 10 (ana, ben, bob, lindsay, mary, mia, michelle, niki, ross, tom)
    Policies   : 5 training policies (bob, lindsay, michelle, niki, ross)
    Runs       : 3 per (resident, policy) pair
    Total      : 10 × 5 × 3 = 150 conversations

    The matrix lets you see, for example:
        - Does applying the "ross" policy help convince Michelle?
        - Does the "bob" policy work on new persona Tom?

USAGE:
    python experiments/cross_policy_experiment.py
    python experiments/cross_policy_experiment.py --runs 5
    python experiments/cross_policy_experiment.py --residents ross,michelle
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import RUNS_DIR, RESIDENTS_LIST
from simulation.conversation_loop import run_conversation
from simulation.llm_client import token_tracker

SEED = ("Hello, this is the fire department. "
        "We need you to evacuate immediately.")

POLICIES = RESIDENTS_LIST  # ["bob", "lindsay", "michelle", "niki", "ross"]
TRAINING_PERSONAS = set(RESIDENTS_LIST)


def _persona_type(name: str) -> str:
    return "train" if name in TRAINING_PERSONAS else "new"


def main():
    parser = argparse.ArgumentParser(
        description="Cross-policy matrix: every resident × every training policy"
    )
    parser.add_argument("--residents", default=None,
                        help="Comma-separated residents (default: all 10)")
    parser.add_argument("--policies", default=None,
                        help="Comma-separated policies to test (default: all 5 training)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per (resident, policy) pair (default: 3)")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()

    residents = (
        [r.strip().lower() for r in args.residents.split(",")]
        if args.residents else sorted(PERSONA.keys())
    )
    policies = (
        [p.strip().lower() for p in args.policies.split(",")]
        if args.policies else POLICIES
    )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_dir = RUNS_DIR / f"cross_policy_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # results[resident][policy] = list of success (0/1)
    results = defaultdict(lambda: defaultdict(list))

    total = len(residents) * len(policies) * args.runs
    done = 0

    print("=" * 70)
    print("  CROSS-POLICY MATRIX EXPERIMENT")
    print(f"  Residents : {', '.join(residents)}")
    print(f"  Policies  : {', '.join(policies)}")
    print(f"  Runs/pair : {args.runs}  |  Total conversations: {total}")
    print("=" * 70)

    for resident in residents:
        for policy in policies:
            for run_idx in range(1, args.runs + 1):
                done += 1
                rid = f"cross_{resident}_policy{policy}_run{run_idx}"
                print(f"\n[{done}/{total}] resident={resident} | policy={policy} | run={run_idx}")

                result = run_conversation(
                    resident_name=resident,
                    strategy="fixed_policy",
                    fixed_policy_name=policy,
                    seed_text=args.seed,
                    max_turns=args.max_turns,
                    run_id=rid,
                    output_dir=exp_dir / f"{resident}_x_{policy}",
                )
                results[resident][policy].append(result["success"])

    # ── Build summary ──────────────────────────────────────────────────────────
    summary = {"timestamp": ts, "runs_per_pair": args.runs, "matrix": {}}
    for resident in residents:
        summary["matrix"][resident] = {}
        for policy in policies:
            runs_list = results[resident][policy]
            s = sum(runs_list)
            summary["matrix"][resident][policy] = {
                "successes": s,
                "runs": len(runs_list),
                "rate": round(s / len(runs_list), 4) if runs_list else 0,
            }

    summary_file = exp_dir / "cross_policy_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ── Print matrix ───────────────────────────────────────────────────────────
    col_w = 12
    print("\n" + "=" * 70)
    print("  CROSS-POLICY SUCCESS RATE MATRIX  (successes / runs)")
    print("=" * 70)

    # Header
    header = f"  {'Resident':<12} {'Type':<6}"
    for p in policies:
        header += f"  {p:>{col_w}}"
    print(header)
    print(f"  {'-'*12} {'-'*6}" + (f"  {'-'*col_w}" * len(policies)))

    for resident in residents:
        ptype = _persona_type(resident)
        row = f"  {resident:<12} {ptype:<6}"
        for policy in policies:
            d = summary["matrix"][resident][policy]
            cell = f"{d['successes']}/{d['runs']} ({d['rate']:.0%})"
            row += f"  {cell:>{col_w}}"
        print(row)

    # Per-policy overall
    print(f"\n  {'OVERALL':<12} {'':6}", end="")
    for policy in policies:
        all_runs = [r for res in residents for r in results[res][policy]]
        rate = sum(all_runs) / len(all_runs) if all_runs else 0
        cell = f"{rate:.0%}"
        print(f"  {cell:>{col_w}}", end="")
    print()

    print(f"\n  Summary → {summary_file}")
    print("=" * 70)
    token_tracker.print_summary()


if __name__ == "__main__":
    main()
