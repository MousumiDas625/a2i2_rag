"""
policy_selector.py — IQL Policy Selector (Runtime)
====================================================

PURPOSE:
    Loads the trained IQL Q-network and uses it at runtime to select the
    best operator policy given the current conversation history.

HOW IT WORKS:
    1. Extract the last N resident utterances from the conversation.
    2. Embed them with sentence-transformers → mean state vector (384-dim).
    3. Feed the state into the embedding-based Q-network:
           Q(s, a) = f( [state_vec ‖ policy_embedding] ) → scalar
       for every policy a.
    4. Return the argmax policy name and the full Q-value dictionary.

USAGE (as a library):
    from retrieval.policy_selector import IQLPolicySelector

    selector = IQLPolicySelector()
    best_policy, q_values = selector.select_policy(history)
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
        print(f"[IQLPolicySelector] Loaded on {self.device}")

    def select_policy(self, history: list) -> tuple:
        """
        Select the best operator policy for the current conversation state.

        Parameters
        ----------
        history : list[dict]
            Conversation so far.  Each dict has "role" and "text".

        Returns
        -------
        (best_policy_name, {policy_name: q_value, ...})
        """
        res_texts = [h["text"] for h in history if h["role"] == "resident"]
        last_n    = res_texts[-self.n_last:] if res_texts else []

        state_vec = _embed_state(self.embed_model, last_n)
        state_t   = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = np.array([
                self.qnet(state_t, torch.tensor([a], dtype=torch.long, device=self.device)).item()
                for a in range(self.num_actions)
            ])

        best_idx = int(np.argmax(q_values))
        return self.policy_names[best_idx], dict(zip(self.policy_names, q_values.tolist()))
