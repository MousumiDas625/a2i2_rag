#!/usr/bin/env python3
"""
P02_clean_and_merge.py — Clean and Structure Dialogues
=======================================================

PURPOSE:
    Takes raw JSONL dialogue files from P01, cleans them, and writes
    structured cleaned versions ready for downstream IQL processing.

HOW IT WORKS:
    1. Drop all Julie turns (speaker/addressee ID 3).
    2. Assign a simple role label: 1→operator, 2→resident.
    3. Normalize whitespace in utterance text.
    4. Sort turns by their original index.
    5. Merge consecutive same-role turns into one (concatenate text).
    6. Re-index merged turns sequentially (t = 0, 1, 2, …).
    7. Trim any trailing operator turns after the last resident utterance
       (the conversation logically ends at the resident's final word).

INPUTS:  data/jsonl/<dialogue_id>.jsonl
OUTPUTS: data/cleaned/<dialogue_id>.jsonl
         data/reports/clean_summary.json

USAGE:
    python preprocessing/P02_clean_and_merge.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import JSONL_DIR, CLEANED_DIR, REPORTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_role(speakers: list) -> str:
    if 1 in speakers:
        return "operator"
    if 2 in speakers:
        return "resident"
    if 3 in speakers:
        return "julie"
    return "unknown"


def drop_julie_turns(turns: list) -> list:
    return [
        t for t in turns
        if 3 not in t.get("speakers", []) and 3 not in t.get("addressees", [])
    ]


def merge_consecutive_turns(turns: list) -> list:
    if not turns:
        return []
    merged = [turns[0].copy()]
    for t in turns[1:]:
        if t.get("role") == merged[-1].get("role"):
            merged[-1]["text"] += " " + t.get("text", "").strip()
        else:
            merged.append(t.copy())
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    files = sorted(
        f for f in JSONL_DIR.glob("*.jsonl") if not f.name.startswith("._")
    )
    summary = []
    total_kept, total_removed = 0, 0

    for f in files:
        if f.name == "combined.jsonl":
            continue

        with f.open("r", encoding="utf-8") as fin:
            turns = [json.loads(line) for line in fin]

        cleaned = drop_julie_turns(turns)
        removed = len(turns) - len(cleaned)

        for t in cleaned:
            t["role"] = normalize_role(t.get("speakers", []))
            t["text"] = re.sub(r"\s+", " ", t.get("text", "").strip())

        cleaned = sorted(cleaned, key=lambda x: x.get("t", 0))
        merged = merge_consecutive_turns(cleaned)

        for i, turn in enumerate(merged):
            turn["t"] = i

        # Trim trailing operator turns after last resident utterance
        last_resident_idx = -1
        for i, turn in reversed(list(enumerate(merged))):
            if turn.get("role") == "resident":
                last_resident_idx = i
                break

        if last_resident_idx != -1:
            merged = merged[: last_resident_idx + 1]

        if not merged:
            continue

        resident_name = merged[0].get("resident", "unknown")
        dialogue_id = merged[0].get("dialogue_id", f.stem)

        summary.append({
            "dialogue_id": dialogue_id,
            "resident": resident_name,
            "total_turns": len(merged),
            "removed_julie_turns": removed,
        })
        total_removed += removed
        total_kept += len(cleaned)

        out_path = CLEANED_DIR / f.name
        with out_path.open("w", encoding="utf-8") as fout:
            for t in merged:
                fout.write(json.dumps(t, ensure_ascii=False) + "\n")

    summary_path = REPORTS_DIR / "clean_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "total_files": len(files) - 1,
                "total_turns_kept_before_merge": total_kept,
                "total_turns_removed": total_removed,
                "dialogue_summaries": summary,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[OK] Cleaning and merging complete.")
    print(f"  Cleaned files  → {CLEANED_DIR}")
    print(f"  Summary report → {summary_path}")


if __name__ == "__main__":
    main()
