#!/usr/bin/env python3
"""
I04_build_rag_indexes.py — Build Per-Policy FAISS Indexes for RAG
==================================================================

PURPOSE:
    Creates a FAISS index for each operator policy so that at runtime the
    system can retrieve the most relevant (resident→operator) example
    pairs for in-context few-shot prompting.

HOW IT WORKS:
    For each policy (= resident name):
        1. Load the (resident_text, operator_text) pairs from I02.
        2. Embed all resident_texts with sentence-transformers (GPU).
        3. Store embeddings in a FAISS IndexFlatIP (cosine similarity
           on L2-normalised vectors).
        4. Save the index and metadata.

INPUTS:  data/indexes/policies/policies_meta.json
         data/indexes/policies/{name}_pairs.json
OUTPUTS: data/indexes/faiss/{name}.faiss   (one per policy)
         data/indexes/faiss/meta_faiss.json

USAGE:
    python iql/I04_build_rag_indexes.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    POLICIES_META_FILE, FAISS_DIR, FAISS_META_FILE, DEVICE,
)


def main():
    if not POLICIES_META_FILE.exists():
        raise FileNotFoundError(
            f"Missing {POLICIES_META_FILE}. Run I02 first."
        )

    meta_policies = json.loads(POLICIES_META_FILE.read_text())
    policies = list(meta_policies["policies"].keys())
    model_name = meta_policies["model_name"]

    print(f"\n[INFO] Building FAISS indexes for {len(policies)} policies.")
    print(f"[INFO] Loading embedding model ({model_name}) on {DEVICE} …")
    model = SentenceTransformer(model_name, device=str(DEVICE))

    faiss_meta: dict = {"model_name": model_name, "policies": {}}

    for policy_name in policies:
        print(f"\n[INFO] Processing policy: {policy_name}")
        pair_file = Path(meta_policies["policies"][policy_name]["pairs_file"])
        if not pair_file.exists():
            print(f"[WARN] Missing pair file for {policy_name}, skipping.")
            continue

        pairs = json.loads(pair_file.read_text())
        if not pairs:
            print(f"[WARN] No pairs for {policy_name}, skipping.")
            continue

        resident_texts = [p["resident_text"] for p in pairs]
        res_embeds = model.encode(
            resident_texts, show_progress_bar=True, batch_size=32,
            convert_to_numpy=True, normalize_embeddings=True,
        )

        dim = res_embeds.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(res_embeds.astype("float32"))

        index_path = FAISS_DIR / f"{policy_name}.faiss"
        faiss.write_index(index, str(index_path))

        faiss_meta["policies"][policy_name] = {
            "resident_texts_file": str(pair_file),
            "index_file": str(index_path),
            "num_pairs": len(resident_texts),
            "dim": dim,
        }

    FAISS_META_FILE.write_text(
        json.dumps(faiss_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[OK] FAISS indexes built → {FAISS_DIR}")
    print(f"  Metadata → {FAISS_META_FILE}")


if __name__ == "__main__":
    main()
