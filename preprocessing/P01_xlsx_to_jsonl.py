#!/usr/bin/env python3
"""
P01_xlsx_to_jsonl.py — XLSX → JSONL Conversion
================================================

PURPOSE:
    Reads every .xlsx dialogue transcript from data/raw_xlsx/ and converts
    each one into a structured JSONL file.  A combined.jsonl with all
    dialogues is also produced.

HOW IT WORKS:
    1. Parse each XLSX file, auto-detecting speaker / addressee / text columns.
    2. Extract operator-id, session-id, and resident name from the filename
       (expected pattern: "<op>_<session>_<resident>_*.xlsx").
    3. Emit one JSON record per utterance row:
           { dialogue_id, resident, t, speakers, addressees, text }
    4. Flag dialogues that contain Julie (speaker/addressee ID 3) for reporting.

INPUTS:  data/raw_xlsx/*.xlsx
OUTPUTS: data/jsonl/<dialogue_id>.jsonl   (one per file)
         data/jsonl/combined.jsonl        (all dialogues concatenated)
         data/reports/has_julie_count.json (report)

USAGE:
    python preprocessing/P01_xlsx_to_jsonl.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_XLSX_DIR, JSONL_DIR, REPORTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_ids_from_filename(fname: str) -> dict:
    stem = Path(fname).stem
    m = re.match(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)", stem)
    if m:
        op, sess, res = m.groups()
        return {"dialogue_id": f"{op}_{sess}_{res}", "resident": res.lower()}
    return {"dialogue_id": stem, "resident": "unknown"}


def clean_ids(val) -> list:
    if pd.isna(val):
        return []
    if isinstance(val, (int, float)):
        return [int(val)]
    if isinstance(val, str):
        return [int(x) for x in re.findall(r"\d+", val)]
    if isinstance(val, (list, tuple, set)):
        out = []
        for v in val:
            out.extend(clean_ids(v))
        return sorted(set(out))
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    xlsx_files = sorted(
        f for f in RAW_XLSX_DIR.glob("*.xlsx") if not f.name.startswith("._")
    )
    if not xlsx_files:
        print(f"[WARN] No XLSX files found in {RAW_XLSX_DIR}")
        return

    has_julie_dialogues: set = set()
    combined_path = JSONL_DIR / "combined.jsonl"

    with combined_path.open("w", encoding="utf-8") as fout_all:
        for f in xlsx_files:
            ids = extract_ids_from_filename(f.name)
            dialogue_id = ids["dialogue_id"]
            resident_name = ids["resident"]

            try:
                xls = pd.ExcelFile(f)
                dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
                df = pd.concat(dfs, ignore_index=True)
            except Exception as e:
                print(f"[ERROR] Failed reading {f.name}: {e}")
                continue

            sp_col = next((c for c in df.columns if "speak" in c.lower()), None)
            to_col = next(
                (c for c in df.columns if "addressee" in c.lower() or "to" in c.lower()),
                None,
            )
            txt_col = next(
                (c for c in df.columns if "text" in c.lower() or "utter" in c.lower() or "message" in c.lower()),
                None,
            )

            if not (sp_col and txt_col):
                print(f"[WARN] Skipping {f.name}: missing speaker/text columns.")
                continue

            out_file = JSONL_DIR / f"{dialogue_id}.jsonl"
            found_julie = False

            with out_file.open("w", encoding="utf-8") as fout:
                for i, row in df.iterrows():
                    sp_ids = clean_ids(row.get(sp_col))
                    to_ids = clean_ids(row.get(to_col))
                    text = str(row.get(txt_col, "")).strip()

                    if 3 in sp_ids or 3 in to_ids:
                        found_julie = True

                    record = {
                        "dialogue_id": dialogue_id,
                        "resident": resident_name,
                        "t": i,
                        "speakers": sp_ids,
                        "addressees": to_ids,
                        "text": text,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout_all.write(json.dumps(record, ensure_ascii=False) + "\n")

            if found_julie:
                has_julie_dialogues.add(dialogue_id)

    report = {
        "total_dialogues": len(xlsx_files),
        "dialogues_with_julie": sorted(list(has_julie_dialogues)),
        "count_with_julie": len(has_julie_dialogues),
    }
    report_path = REPORTS_DIR / "has_julie_count.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] XLSX → JSONL conversion done.")
    print(f"  JSONL written to : {JSONL_DIR}")
    print(f"  Combined file    : {combined_path}")
    print(f"  Report           : {report_path}")


if __name__ == "__main__":
    main()
