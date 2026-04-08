#!/usr/bin/env python3
"""
I01_build_iql_dataset.py — Build the IQL Training Dataset
==========================================================

PURPOSE:
    Converts cleaned dialogues into a dataset suitable for training an
    Implicit Q-Learning (IQL) policy selector.

HOW IT WORKS:
    For each operator turn in every cleaned dialogue:
        • state_vec  = sentence embedding of the last N resident utterances.
        • action_id  = integer policy ID (from label_map — one per resident).
        • reward     = the reward of the next resident turn (0 or 1).
    A terminal record (reward=2) is appended at the end of each dialogue
    so the Q-network can learn end-of-episode dynamics.

    Embeddings are computed on GPU when available via sentence-transformers.

INPUTS:  data/cleaned/*.jsonl, data/meta/residents.json
OUTPUTS: data/iql/iql_dataset.jsonl
         data/iql/label_map.json
         data/iql/config.json

USAGE:
    python iql/I01_build_iql_dataset.py
"""

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CLEANED_DIR, RESIDENTS_META_FILE, IQL_DIR,
    IQL_DATASET_FILE, IQL_CONFIG_FILE, LABEL_MAP_FILE,
    EMBED_MODEL_NAME, N_LAST_RESIDENT_TRAIN, DEVICE,
)


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # ── 1. Load residents / label map ─────────────────────────────────────────
    if not RESIDENTS_META_FILE.exists():
        raise FileNotFoundError(
            f"Missing {RESIDENTS_META_FILE}. Run P03_extract_residents.py first."
        )
    meta = json.loads(RESIDENTS_META_FILE.read_text())
    residents = meta["residents"]

    label_map = {res: i for i, res in enumerate(residents)}
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[INFO] {len(residents)} residents → label map created.")
    for k, v in label_map.items():
        print(f"  {v}: {k}")

    # ── 2. Load embedding model (GPU-accelerated) ────────────────────────────
    print(f"\n[INFO] Loading embedding model '{EMBED_MODEL_NAME}' on {DEVICE} …")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))

    # ── 3. Iterate cleaned dialogues ──────────────────────────────────────────
    clean_files = sorted(
        f for f in CLEANED_DIR.glob("*.jsonl") if not f.name.startswith("._")
    )
    if not clean_files:
        raise FileNotFoundError(f"No cleaned JSONL files in {CLEANED_DIR}")

    print(f"[INFO] Found {len(clean_files)} cleaned dialogue files.")
    dataset: list = []

    for fpath in tqdm(clean_files, desc="[BUILDING IQL DATASET]", ncols=100):
        records = load_jsonl(fpath)
        if not records:
            continue

        dialogues: dict = defaultdict(list)
        for r in records:
            dialogues[r["dialogue_id"]].append(r)

        for dialogue_id, turns in dialogues.items():
            resident_name = turns[0].get("resident", "").lower().strip()
            if resident_name not in label_map:
                continue
            action_id = label_map[resident_name]
            resident_history: deque = deque(maxlen=N_LAST_RESIDENT_TRAIN)

            for i, turn in enumerate(turns):
                text = turn.get("text", "").strip()
                if not text:
                    continue

                if turn.get("role") == "resident":
                    resident_history.append(text)

                if turn.get("role") == "operator" and resident_history:
                    state_text = " ".join(resident_history)
                    state_vec = model.encode(state_text, convert_to_numpy=True)

                    reward = 0
                    for next_turn in turns[i + 1:]:
                        if next_turn.get("role") == "resident":
                            reward = next_turn.get("reward", 0)
                            break

                    dataset.append({
                        "dialogue_id": dialogue_id,
                        "resident": resident_name,
                        "state_text": state_text,
                        "action_id": action_id,
                        "reward": reward,
                        "state_vec": state_vec.tolist(),
                    })

            # Terminal record
            if resident_history:
                final_text = " ".join(resident_history)
                final_vec = model.encode(final_text, convert_to_numpy=True)
                dataset.append({
                    "dialogue_id": dialogue_id,
                    "resident": resident_name,
                    "state_text": final_text,
                    "action_id": action_id,
                    "reward": 2,
                    "state_vec": final_vec.tolist(),
                })

    # ── 4. Save dataset ──────────────────────────────────────────────────────
    with open(IQL_DATASET_FILE, "w", encoding="utf-8") as fout:
        for rec in dataset:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[DONE] IQL dataset saved → {IQL_DATASET_FILE}")
    print(f"  Total samples: {len(dataset)}")

    # ── 5. Save config ───────────────────────────────────────────────────────
    config = {
        "embedding_model": EMBED_MODEL_NAME,
        "n_last_resident": N_LAST_RESIDENT_TRAIN,
        "num_actions": len(residents),
        "residents": residents,
    }
    with open(IQL_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Config saved → {IQL_CONFIG_FILE}")


if __name__ == "__main__":
    main()
