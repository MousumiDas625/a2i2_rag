#!/usr/bin/env python3
"""
batch_runner.py — Batch Simulation Driver
==========================================

PURPOSE:
    Runs multiple conversations across residents and seed utterances for
    any experiment strategy.  Produces per-conversation JSON files and a
    master index (JSON + CSV).

HOW IT WORKS:
    Iterates over (seed_text × resident × repetition) and calls
    conversation_loop.run_conversation() for each combination.

USAGE:
    python experiments/batch_runner.py \\
        --strategy iql_rag \\
        --residents ross,michelle,bob \\
        --num-seeds 3 \\
        --reps 2 \\
        --run-id my_batch_01
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import RUNS_DIR
from simulation.conversation_loop import run_conversation

SEEDS = [
    "Hello, this is the fire department. We need you to evacuate immediately.",
    "There's a wildfire nearby. Please leave your home now.",
    "For your safety, we urge you to exit the house and move to the designated safe zone.",
    "This is an emergency alert, you must leave the area right away.",
    "The fire is moving quickly, please evacuate immediately.",
    "It's not safe to remain here, please head to the safe shelter now.",
    "For your safety, gather essentials and evacuate right away.",
    "Authorities have issued an evacuation order, you must leave now.",
    "The situation is critical, please prioritize safety over property.",
    "Emergency services require you to leave the area for your own safety.",
]


def main():
    parser = argparse.ArgumentParser(description="Batch simulation driver")
    parser.add_argument("--strategy", choices=["zero_shot", "rag_successful", "iql_rag"],
                        default="iql_rag")
    parser.add_argument("--residents", required=True,
                        help="Comma-separated resident names")
    parser.add_argument("--num-seeds", type=int, default=2,
                        help="How many seed utterances to use (from SEEDS list)")
    parser.add_argument("--reps", type=int, default=1,
                        help="Repetitions per (seed, resident) pair")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--mode", choices=["state", "embed"], default="state",
                        help="IQL mode (only for iql_rag strategy)")
    args = parser.parse_args()

    residents = [r.strip().lower() for r in args.residents.split(",")]
    seeds_to_use = SEEDS[:args.num_seeds]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = RUNS_DIR / "seed_runs" / f"{ts}_{args.run_id}"
    conv_dir = batch_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    selector = None
    if args.strategy == "iql_rag":
        from retrieval.policy_selector import IQLPolicySelector
        selector = IQLPolicySelector(mode=args.mode)

    master: dict = {"run_id": args.run_id, "timestamp": ts, "results": []}

    print("=" * 70)
    print(f"  BATCH RUN: {args.run_id}  |  strategy={args.strategy}")
    print(f"  Residents: {residents}  |  seeds={len(seeds_to_use)}  |  reps={args.reps}")
    print("=" * 70)

    run_idx = 0
    for sid, seed_text in enumerate(seeds_to_use):
        for resident in residents:
            for rep in range(1, args.reps + 1):
                run_idx += 1
                rid = f"s{sid}_{resident}_r{rep}"
                print(f"\n[RUN {run_idx}] {rid}")

                result = run_conversation(
                    resident_name=resident,
                    strategy=args.strategy,
                    seed_text=seed_text,
                    max_turns=args.max_turns,
                    selector=selector,
                    run_id=rid,
                )

                conv_meta = {
                    "seed_id": sid,
                    "seed_text": seed_text,
                    "resident": resident,
                    "rep": rep,
                    "status": result.get("status", "unknown"),
                    "success": result.get("success", 0),
                    "num_turns": result.get("turns", 0),
                    "history": result.get("history", []),
                    "file": result.get("path", ""),
                }
                outfile = conv_dir / f"seed{sid}__{resident}__rep{rep}.json"
                outfile.write_text(
                    json.dumps(conv_meta, indent=2, ensure_ascii=False)
                )

                master["results"].append({
                    "seed_id": sid,
                    "resident": resident,
                    "rep": rep,
                    "success": conv_meta["success"],
                    "num_turns": conv_meta["num_turns"],
                    "file": str(outfile),
                })

    # Save master index
    index_file = batch_dir / "master_index.json"
    index_file.write_text(json.dumps(master, indent=2, ensure_ascii=False))

    # CSV summary
    csv_file = batch_dir / "master_index.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed_id", "resident", "rep", "success", "num_turns", "file"
        ])
        writer.writeheader()
        writer.writerows(master["results"])

    print(f"\n[OK] Master index → {index_file}")
    print(f"[OK] CSV summary  → {csv_file}")


if __name__ == "__main__":
    main()
