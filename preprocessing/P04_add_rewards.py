#!/usr/bin/env python3
"""
P04_add_rewards.py — Assign Reward Labels
==========================================

PURPOSE:
    Adds a "reward" field to every record in each cleaned JSONL dialogue.
    This is the signal consumed by the IQL training pipeline.

HOW IT WORKS:
    For each dialogue file:
        • Default reward = 0 for every utterance.
        • The *last resident utterance* gets reward = 1.
          Rationale: in the original dataset the final resident turn in
          a successful conversation corresponds to the resident agreeing
          to evacuate, which is the positive outcome we want to reinforce.
    Files are updated in place.

INPUTS:  data/cleaned/*.jsonl  (updated in place)
OUTPUTS: same files, with "reward" key added to every record.

USAGE:
    python preprocessing/P04_add_rewards.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CLEANED_DIR


def main():
    jsonl_files = sorted(
        f for f in CLEANED_DIR.glob("*.jsonl") if not f.name.startswith("._")
    )
    if not jsonl_files:
        print(f"[WARN] No cleaned files found in {CLEANED_DIR}")
        return

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as fin:
            records = [json.loads(line) for line in fin if line.strip()]

        dialogues: dict = defaultdict(list)
        for rec in records:
            dialogues[rec["dialogue_id"]].append(rec)

        updated: list = []
        for _did, turns in dialogues.items():
            resident_idxs = [
                i for i, t in enumerate(turns) if t.get("role") == "resident"
            ]
            last_res_idx = resident_idxs[-1] if resident_idxs else None

            for i, rec in enumerate(turns):
                rec["reward"] = 1 if (last_res_idx is not None and i == last_res_idx) else 0
                updated.append(rec)

        with open(fpath, "w", encoding="utf-8") as fout:
            for rec in updated:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        reward_count = sum(1 for r in updated if r["reward"] == 1)
        print(
            f"[OK] {fpath.name}: {len(updated)} records, "
            f"{reward_count} reward=1 turns"
        )

    print("\n[DONE] Reward field added to all cleaned dialogues.")


if __name__ == "__main__":
    main()
