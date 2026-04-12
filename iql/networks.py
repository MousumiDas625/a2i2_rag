"""
networks.py — Shared Neural Network Definitions for IQL
========================================================

ARCHITECTURES:
    QNetworkEmbed  — Dueling embedding-based Q-network.
                     Uses frozen operator prototype embeddings + dot-product
                     advantage to produce *distinct* Q-values per action.
                     Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
                     where A(s,a) = project(encode(s)) · embed(a)

    QNetworkState  — State-only Q-network: Q(s) → [Q(s,a1), ...]
                     (retained for comparison / fallback)

    ValueNetwork   — V(s) → scalar baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import IQL_DROPOUT, IQL_HIDDEN_DIM_Q, IQL_HIDDEN_DIM_V


class QNetworkEmbed(nn.Module):
    """
    Dueling embedding-based Q-network with dot-product advantages.

    Instead of concatenating [state ‖ action_emb] (which lets the network
    ignore the action part), this architecture:

        1. Encodes the state through a shared trunk → hidden features h.
        2. Value head:     V(s) = linear(h) → scalar.
        3. Advantage head: Projects h into the action-embedding space,
                           then takes the dot product with each (frozen)
                           action embedding → one advantage per action.
        4. Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

    The action embeddings are registered as a non-learnable buffer, so
    they never collapse.  The dot product structurally guarantees that
    geometrically different embeddings produce different Q-values.
    """

    def __init__(self, state_dim: int, action_embeds: torch.Tensor,
                 hidden_dim: int = IQL_HIDDEN_DIM_Q,
                 dropout: float = IQL_DROPOUT):
        super().__init__()
        num_actions, action_dim = action_embeds.shape
        self.register_buffer("action_embeds", action_embeds)

        mid_dim = hidden_dim // 2  # 128 when hidden_dim=256
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.LayerNorm(mid_dim),
            nn.Dropout(dropout),
        )
        self.value_head = nn.Linear(mid_dim, 1)
        self.adv_proj   = nn.Linear(mid_dim, action_dim)

    def forward(self, state: torch.Tensor,
                action_id: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        state     : [B, state_dim]
        action_id : [B] int64 — if given, returns Q for that action only.
                    If None, returns all Q-values [B, num_actions].
        """
        h   = self.encoder(state)
        v   = self.value_head(h)                          # [B, 1]
        adv = self.adv_proj(h) @ self.action_embeds.T     # [B, num_actions]
        adv = adv - adv.mean(dim=1, keepdim=True)         # center
        q   = v + adv                                     # [B, num_actions]

        if action_id is not None:
            return q.gather(1, action_id.view(-1, 1)).squeeze(1)
        return q


class QNetworkState(nn.Module):
    """
    State-only Q-network: Q(s) = [Q(s,a1), ..., Q(s,aN)].
    Retained for comparison / fallback.
    """

    def __init__(self, state_dim: int, num_actions: int,
                 hidden_dim: int = IQL_HIDDEN_DIM_Q,
                 dropout: float = IQL_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """V(s) → scalar estimate of state value."""

    def __init__(self, state_dim: int,
                 hidden_dim: int = IQL_HIDDEN_DIM_V,
                 dropout: float = IQL_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
