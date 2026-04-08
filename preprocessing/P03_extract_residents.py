#!/usr/bin/env python3
"""
P03_extract_residents.py — Discover Unique Residents
=====================================================

PURPOSE:
    Scans all cleaned dialogue JSONL files to find every unique resident
    name and saves the sorted list as a JSON metadata file.  This metadata
    is consumed by the IQL dataset builder and the policy builder.

HOW IT WORKS:
    1. Iterate over every line in every cleaned JSONL file.
    2. Collect the "resident" field, lowercased and stripped.
    3. Deduplicate into a sorted list.
    4. Write to data/meta/residents.json.

INPUTS:  data/cleaned/*.jsonl
OUTPUTS: data/meta/residents.json

USAGE:
    python preprocessing/P03_extract_residents.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CLEANED_DIR, RESIDENTS_META_FILE


def main():
    clean_files = sorted(
        f for f in CLEANED_DIR.glob("*.jsonl") if not f.name.startswith("._")
    )
    if not clean_files:
        raise FileNotFoundError(
            f"No cleaned files found in {CLEANED_DIR}. Run P02 first."
        )

    unique_residents: set = set()
    print(f"[INFO] Scanning {len(clean_files)} cleaned files …")

    for fpath in clean_files:
        with open(fpath, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    rec = json.loads(line)
                    resident = rec.get("resident")
                    if resident:
                        unique_residents.add(resident.lower().strip())
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping malformed line in {fpath.name}")

    sorted_residents = sorted(unique_residents)
    meta = {"residents": sorted_residents}

    RESIDENTS_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESIDENTS_META_FILE, "w", encoding="utf-8") as fout:
        json.dump(meta, fout, indent=2)

    print(f"\n[DONE] Found {len(sorted_residents)} unique residents.")
    print(f"  Meta file → {RESIDENTS_META_FILE}")
    for res in sorted_residents:
        print(f"    - {res}")


if __name__ == "__main__":
    main()
