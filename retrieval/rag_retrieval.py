"""
rag_retrieval.py — RAG Retrieval Module
========================================

PURPOSE:
    Provides retrieval functions used at runtime to fetch in-context
    few-shot examples for the operator LLM prompt.

TWO RETRIEVAL MODES:
    1. retrieve_topk_pairs(policy_name, query, k)
       → Per-policy retrieval (Experiment 3 — IQL + RAG).
         Queries the FAISS index for a specific operator policy and
         returns the top-K most similar (resident→operator) pairs.

    2. retrieve_from_successful(query, k)
       → Global retrieval over ALL successful operator utterances
         (Experiment 2 — RAG over successful ops).

HOW IT WORKS:
    • Embeds the query (latest resident utterance) with sentence-transformers.
    • Searches the appropriate FAISS index (cosine similarity via IndexFlatIP).
    • Returns matching pairs / utterances with scores.

USAGE (as a library):
    from retrieval.rag_retrieval import retrieve_topk_pairs, retrieve_from_successful
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    FAISS_META_FILE, SUCCESSFUL_OPS_DIR, EMBED_MODEL_NAME, DEVICE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────
_embed_model: SentenceTransformer = None  # type: ignore[assignment]
_faiss_meta: dict = {}
_succ_index: faiss.Index = None  # type: ignore[assignment]
_succ_records: list = []


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))
    return _embed_model


def _get_faiss_meta() -> dict:
    global _faiss_meta
    if not _faiss_meta:
        if not FAISS_META_FILE.exists():
            raise FileNotFoundError(
                f"Missing {FAISS_META_FILE}. Run I04 first."
            )
        _faiss_meta = json.loads(FAISS_META_FILE.read_text())
    return _faiss_meta


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Per-policy retrieval (for Experiment 3 — IQL + RAG)
# ═══════════════════════════════════════════════════════════════════════════════
def retrieve_topk_pairs(
    policy_name: str, query_text: str, k: int = 2
) -> List[Dict]:
    """
    Retrieve top-K (resident_text, operator_text) pairs from a specific
    policy's FAISS index.
    """
    meta = _get_faiss_meta()
    if policy_name not in meta["policies"]:
        raise ValueError(f"Policy '{policy_name}' not found in FAISS metadata.")

    pmeta = meta["policies"][policy_name]
    index_file = Path(pmeta["index_file"])
    pairs_file = Path(pmeta["resident_texts_file"])

    if not index_file.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_file}")
    if not pairs_file.exists():
        raise FileNotFoundError(f"Missing pairs file: {pairs_file}")

    index = faiss.read_index(str(index_file))
    pairs = json.loads(pairs_file.read_text())

    model = _get_embed_model()
    q_embed = model.encode(
        [query_text], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    D, I = index.search(q_embed, k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(pairs):
            results.append(pairs[idx])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Global retrieval over successful operator utterances (for Experiment 2)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_successful_index():
    global _succ_index, _succ_records

    utters_file = SUCCESSFUL_OPS_DIR / "utterances.jsonl"
    index_file = SUCCESSFUL_OPS_DIR / "index.faiss"

    if not utters_file.exists() or not index_file.exists():
        raise FileNotFoundError(
            f"Successful-ops corpus not found in {SUCCESSFUL_OPS_DIR}. "
            "Run I05 first."
        )

    _succ_records = [
        json.loads(line)
        for line in utters_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _succ_index = faiss.read_index(str(index_file))


def retrieve_from_successful(
    query_text: str, k: int = 3
) -> List[Dict]:
    """
    Retrieve top-K operator utterances from the global successful-ops
    FAISS index.  Each result dict has: text, dialogue_id, resident, _score.
    """
    global _succ_index, _succ_records
    if _succ_index is None:
        _load_successful_index()

    model = _get_embed_model()
    q_embed = model.encode(
        [query_text], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    scores, indices = _succ_index.search(q_embed, k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(_succ_records):
            hit = dict(_succ_records[idx])
            hit["_score"] = float(score)
            results.append(hit)
    return results
