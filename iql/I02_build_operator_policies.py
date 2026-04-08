#!/usr/bin/env python3
"""
I02_build_operator_policies.py — Build Operator Policy Prototypes
==================================================================

PURPOSE:
    Creates one "operator policy" per resident.  A policy is represented
    as a mean embedding vector (prototype) computed from all operator
    utterances directed at that resident across the training dialogues.

HOW IT WORKS:
    1. For each resident in meta/residents.json, collect every
       (resident_text → operator_text) pair from cleaned dialogues.
    2. Embed all operator texts with sentence-transformers (on GPU).
    3. Compute the centroid (mean) embedding = policy prototype.
    4. Stack all per-resident prototypes into a single matrix
       operator_prototypes.npy  (shape: num_residents × embed_dim).

INPUTS:  data/cleaned/*.jsonl, data/meta/residents.json
OUTPUTS: data/indexes/policies/{name}_pairs.json
         data/indexes/policies/{name}_op_embeds.npy
         data/indexes/policies/{name}_prototype.npy
         data/indexes/policies/operator_prototypes.npy
         data/indexes/policies/policies_meta.json

USAGE:
    python iql/I02_build_operator_policies.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CLEANED_DIR, RESIDENTS_META_FILE, POLICY_DIR,
    POLICIES_META_FILE, EMBED_MODEL_NAME, DEVICE,
)


def main():
    if not RESIDENTS_META_FILE.exists():
        raise FileNotFoundError(
            f"Missing {RESIDENTS_META_FILE}. Run P03_extract_residents.py first."
        )

    meta = json.loads(RESIDENTS_META_FILE.read_text())
    residents = meta["residents"]

    print("\n[INFO] Building operator policy prototypes for:")
    for i, r in enumerate(residents):
        print(f"   {i}: {r}")

    print(f"\n[INFO] Loading embedding model on {DEVICE} …")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))

    all_dialogues = sorted(
        f for f in CLEANED_DIR.glob("*.jsonl") if not f.name.startswith("._")
    )
    if not all_dialogues:
        raise FileNotFoundError(f"No cleaned dialogues in {CLEANED_DIR}")

    resident_policies: dict = {r: [] for r in residents}

    print("\n[INFO] Extracting (resident → operator) pairs …")
    for fpath in tqdm(all_dialogues):
        lines = [json.loads(line) for line in fpath.open("r", encoding="utf-8")]
        if not lines:
            continue
        res_name = lines[0].get("resident", "").lower().strip()
        if res_name not in residents:
            continue

        for i in range(len(lines) - 1):
            if lines[i]["role"] == "resident" and lines[i + 1]["role"] == "operator":
                r_text = lines[i]["text"].strip()
                o_text = lines[i + 1]["text"].strip()
                if r_text and o_text:
                    resident_policies[res_name].append({
                        "resident_text": r_text,
                        "operator_text": o_text,
                    })

    meta_summary: dict = {"model_name": EMBED_MODEL_NAME, "policies": {}}

    print("\n[INFO] Computing embeddings and prototypes …")
    for res_name, pairs in resident_policies.items():
        if not pairs:
            print(f"[WARN] No pairs for '{res_name}', skipping.")
            continue

        out_json = POLICY_DIR / f"{res_name}_pairs.json"
        json.dump(pairs, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        operator_texts = [p["operator_text"] for p in pairs]
        op_embeds = model.encode(
            operator_texts, show_progress_bar=True, batch_size=32,
            convert_to_numpy=True,
        )

        npy_path = POLICY_DIR / f"{res_name}_op_embeds.npy"
        np.save(npy_path, op_embeds)

        centroid = op_embeds.mean(axis=0)
        centroid_path = POLICY_DIR / f"{res_name}_prototype.npy"
        np.save(centroid_path, centroid)

        meta_summary["policies"][res_name] = {
            "num_pairs": len(pairs),
            "pairs_file": str(out_json),
            "embeds_file": str(npy_path),
            "prototype_file": str(centroid_path),
        }

    # Combine all prototypes into a single matrix
    print("\n[INFO] Combining prototypes …")
    res_order = sorted(meta_summary["policies"].keys())
    prototype_vecs = []
    for res_name in res_order:
        proto_path = Path(meta_summary["policies"][res_name]["prototype_file"])
        if proto_path.exists():
            prototype_vecs.append(np.load(proto_path))
        else:
            print(f"[WARN] Missing prototype for {res_name}, inserting zeros.")
            prototype_vecs.append(np.zeros(model.get_sentence_embedding_dimension()))

    combined = np.stack(prototype_vecs, axis=0)
    combined_path = POLICY_DIR / "operator_prototypes.npy"
    np.save(combined_path, combined)

    json.dump(meta_summary, open(POLICIES_META_FILE, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"\n[OK] Operator policies built → {POLICY_DIR}")
    print(f"  Prototypes matrix : {combined_path}  (shape {combined.shape})")
    print(f"  Metadata          : {POLICIES_META_FILE}")


if __name__ == "__main__":
    main()
