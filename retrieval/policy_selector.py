"""
policy_selector.py — IQL Policy Selector (Runtime)
====================================================

Loads the trained Dueling-Embed IQL Q-network and selects the best
operator policy at runtime.

Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
where  A(s,a) = project(encode(s)) · frozen_embed(a)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    SELECTOR_DIR, LABEL_MAP_FILE, PROTOTYPES_FILE,
    EMBED_MODEL_NAME, N_LAST_RESIDENT_INFER, DEVICE,
)
from iql.networks import QNetworkEmbed

NORM_STATS_FILE = SELECTOR_DIR / "norm_stats.npz"


def _embed_state(model: SentenceTransformer, texts: list) -> np.ndarray:
    if not texts:
        return np.zeros(
            (model.get_sentence_embedding_dimension(),), dtype=np.float32
        )
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.mean(embs, axis=0).astype(np.float32)


class IQLPolicySelector:
    def __init__(self, n_last: int = N_LAST_RESIDENT_INFER):
        self.device = DEVICE
        self.n_last = n_last

        label_map = json.load(open(LABEL_MAP_FILE))
        self.policy_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
        self.num_actions  = len(self.policy_names)

        if not PROTOTYPES_FILE.exists():
            raise FileNotFoundError(f"Missing {PROTOTYPES_FILE}. Run I02 first.")
        action_embeds = torch.tensor(
            np.load(PROTOTYPES_FILE), dtype=torch.float32, device=self.device
        )

        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=str(self.device))

        model_path = SELECTOR_DIR / "iql_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}. Run I03 first.")

        state_dim  = self.embed_model.get_sentence_embedding_dimension()
        self.qnet  = QNetworkEmbed(state_dim, action_embeds).to(self.device)
        self.qnet.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.qnet.eval()

        if NORM_STATS_FILE.exists():
            norms = np.load(NORM_STATS_FILE)
            self.norm_mean = norms["mean"]
            self.norm_std  = norms["std"]
        else:
            self.norm_mean = None
            self.norm_std  = None
            print("[WARN] norm_stats.npz not found — states will NOT be normalized")

        print(f"[IQLPolicySelector] Loaded Dueling-Embed Q-net on {self.device}")

    def select_policy(self, history: list) -> tuple:
        res_texts = [h["text"] for h in history if h["role"] == "resident"]
        last_n    = res_texts[-self.n_last:] if res_texts else []

        state_vec = _embed_state(self.embed_model, last_n)
        if self.norm_mean is not None:
            state_vec = (state_vec - self.norm_mean) / self.norm_std
        state_t = torch.tensor(state_vec, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.qnet(state_t).cpu().numpy().flatten()

        best_idx = int(np.argmax(q_values))
        return self.policy_names[best_idx], dict(zip(self.policy_names, q_values.tolist()))
