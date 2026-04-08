#!/usr/bin/env python3
"""
I05_extract_successful_utterances.py — Build Successful-Operator Corpus
========================================================================

PURPOSE:
    Creates a corpus + FAISS index of operator utterances from *successful*
    evacuation dialogues only.  This powers Experiment 2 (RAG over all
    successful operators' responses).

HOW IT WORKS:
    1. Read iql_dataset.jsonl — a dialogue is "successful" if it contains
       at least one record with reward == 1 (the resident agreed).
    2. For each successful dialogue_id, load the cleaned JSONL file and
       extract every operator turn.
    3. Embed all operator utterances and build a single FAISS IndexFlatIP.

INPUTS:  data/iql/iql_dataset.jsonl
         data/cleaned/{dialogue_id}.jsonl
OUTPUTS: data/successful_ops/utterances.jsonl
         data/successful_ops/index.faiss
         data/successful_ops/meta.json

USAGE:
    python iql/I05_extract_successful_utterances.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    IQL_DATASET_FILE, CLEANED_DIR, SUCCESSFUL_OPS_DIR,
    EMBED_MODEL_NAME, DEVICE,
)


def main():
    # ── 1. Find successful dialogue IDs ───────────────────────────────────────
    if not IQL_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"IQL dataset not found: {IQL_DATASET_FILE}. Run I01 first."
        )

    all_ids: set = set()
    rewarded_ids: set = set()

    with IQL_DATASET_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            did = rec.get("dialogue_id")
            if not did:
                continue
            all_ids.add(did)
            if rec.get("reward") == 1:
                rewarded_ids.add(did)

    print(f"[INFO] Total dialogues       : {len(all_ids)}")
    print(f"[INFO] Successful (reward=1) : {len(rewarded_ids)}")

    # ── 2. Extract operator turns from successful dialogues ──────────────────
    all_records: list = []
    found, missing = 0, 0

    for did in sorted(rewarded_ids):
        fpath = CLEANED_DIR / f"{did}.jsonl"
        if not fpath.exists():
            missing += 1
            continue
        found += 1
        with fpath.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line.strip())
                if rec.get("role") == "operator":
                    text = (rec.get("text") or "").strip()
                    if text:
                        all_records.append({
                            "dialogue_id": rec.get("dialogue_id", did),
                            "resident": rec.get("resident", "unknown"),
                            "text": text,
                        })

    print(f"[INFO] Cleaned files matched : {found}")
    print(f"[INFO] Operator records      : {len(all_records)}")

    if not all_records:
        print("[WARN] No records — nothing to write.")
        return

    # ── 3. Write utterances ──────────────────────────────────────────────────
    utters_file = SUCCESSFUL_OPS_DIR / "utterances.jsonl"
    with utters_file.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 4. Build FAISS index ─────────────────────────────────────────────────
    print(f"[INFO] Loading embedding model on {DEVICE} …")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))

    texts = [r["text"] for r in all_records]
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=64,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_file = SUCCESSFUL_OPS_DIR / "index.faiss"
    faiss.write_index(index, str(index_file))

    meta = {
        "model_name": EMBED_MODEL_NAME,
        "num_utterances": len(texts),
        "embedding_dim": dim,
        "index_file": str(index_file),
        "utterances_file": str(utters_file),
    }
    meta_file = SUCCESSFUL_OPS_DIR / "meta.json"
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Successful-operator corpus built → {SUCCESSFUL_OPS_DIR}")
    print(f"  Utterances : {utters_file}  ({len(all_records)} records)")
    print(f"  FAISS      : {index_file}  ({index.ntotal} vectors, dim={dim})")


if __name__ == "__main__":
    main()
