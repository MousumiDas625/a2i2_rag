"""
networks.py — Shared Neural Network Definitions for IQL
========================================================

PURPOSE:
    Defines the Q-network and Value-network architectures used for
    Implicit Q-Learning (IQL).  These classes are imported by the
    training script (I03) and the runtime policy selector.

ARCHITECTURES:
    QNetworkState  — State-only: Q(s) → [Q(s,a1), …, Q(s,aN)]
    QNetworkEmbed  — Embedding-based: Q(s,a) = f([state ‖ emb(a)]) → scalar
    ValueNetwork   — V(s) → scalar estimate of state value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import IQL_DROPOUT, IQL_HIDDEN_DIM_Q, IQL_HIDDEN_DIM_V


class QNetworkState(nn.Module):
    """
    State-only Q-network.
    Input:  state vector (batch, state_dim)
    Output: Q-values for all actions (batch, num_actions)
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


class QNetworkEmbed(nn.Module):
    """
    Embedding-based Q-network.
    Input:  state vector + action_id (used to look up a learnable embedding)
    Output: scalar Q(s, a)
    """

    def __init__(self, state_dim: int, action_embeds: torch.Tensor,
                 hidden_dim: int = IQL_HIDDEN_DIM_Q,
                 dropout: float = IQL_DROPOUT):
        super().__init__()
        self.action_embeds = nn.Parameter(action_embeds, requires_grad=True)
        input_dim = state_dim + action_embeds.shape[1]
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor, action_id: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_embeds[action_id]
        x = torch.cat([state, a_emb], dim=-1)
        x = F.relu(self.ln1(self.f1(x)))
        x = self.drop(x)
        x = F.relu(self.ln2(self.f2(x)))
        x = self.drop(x)
        return self.head(x).squeeze(-1)


class ValueNetwork(nn.Module):
    """
    State value network V(s).
    Input:  state vector (batch, state_dim)
    Output: scalar value estimate (batch,)
    """

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
