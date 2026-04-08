#!/usr/bin/env python3
"""
interactive.py — Interactive Console Mode
===========================================

PURPOSE:
    Provides a terminal-based interactive loop where a human can
    participate in the evacuation conversation on either or both sides.

MODES:
    --role operator   → Human plays operator, AI plays resident.
    --role resident   → AI plays operator, human plays resident.
    --role both       → Human plays both sides (full manual).
    --role none       → AI plays both sides (automated, same as experiments).

HOW IT WORKS:
    Wraps conversation_loop.run_conversation() with the appropriate
    interactive flags.

USAGE:
    python simulation/interactive.py --resident ross --role operator
    python simulation/interactive.py --resident bob --role both
    python simulation/interactive.py --role none --strategy zero_shot
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.conversation_loop import run_conversation


def main():
    parser = argparse.ArgumentParser(
        description="Interactive evacuation conversation"
    )
    parser.add_argument(
        "--resident", default="ross",
        help="Resident persona name (default: ross)",
    )
    parser.add_argument(
        "--role", choices=["operator", "resident", "both", "none"],
        default="operator",
        help="Which role the human plays (default: operator)",
    )
    parser.add_argument(
        "--strategy", choices=["zero_shot", "rag_successful", "iql_rag"],
        default="iql_rag",
        help="Operator strategy (default: iql_rag)",
    )
    parser.add_argument(
        "--seed", default=None,
        help="Optional first operator line",
    )
    parser.add_argument(
        "--max-turns", type=int, default=16,
        help="Maximum turns (default: 16)",
    )
    args = parser.parse_args()

    interactive_op = args.role in ("operator", "both")
    interactive_res = args.role in ("resident", "both")

    print("=" * 60)
    print(f"  Interactive Mode")
    print(f"  Resident   : {args.resident}")
    print(f"  Human role : {args.role}")
    print(f"  Strategy   : {args.strategy}")
    print("=" * 60)

    result = run_conversation(
        resident_name=args.resident,
        strategy=args.strategy,
        seed_text=args.seed,
        max_turns=args.max_turns,
        interactive_operator=interactive_op,
        interactive_resident=interactive_res,
    )

    print(f"\n{'=' * 60}")
    print(f"  Result: {result['status']} ({result['turns']} turns)")
    print(f"  Saved:  {result['path']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
