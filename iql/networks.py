"""
networks.py — Shared Neural Network Definitions for IQL
========================================================

PURPOSE:
    Defines the Q-network and Value-network architectures used for
    Implicit Q-Learning (IQL).

ARCHITECTURES:
    QNetworkEmbed  — Embedding-based Q-network: Q(s,a) = f([state ‖ emb(a)]) → scalar
                     Each operator policy is represented by a learnable embedding
                     vector (initialised from its prototype).  The network scores
                     how well a policy matches the current conversation state.
    ValueNetwork   — V(s) → scalar baseline estimate of state value
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
    Embedding-based Q-network  Q(s, a) = f( [state ‖ emb(a)] ) → scalar.

    The conversation state `s` is the mean sentence embedding of the last N
    resident utterances.  Each action `a` (operator policy) is looked up in a
    learnable embedding table initialised from per-policy prototype vectors.
    The two representations are concatenated and passed through an MLP that
    outputs a single Q-value for the (state, policy) pair.

    An orthogonality regularisation term is applied during training to keep
    the policy embeddings geometrically distinct from each other.

    Parameters
    ----------
    state_dim      : Dimensionality of the sentence-embedding state vector.
    action_embeds  : Tensor of shape (num_policies, embed_dim) — initial
                     prototype embeddings loaded from operator_prototypes.npy.
    """

    def __init__(self, state_dim: int, action_embeds: torch.Tensor,
                 hidden_dim: int = IQL_HIDDEN_DIM_Q,
                 dropout: float = IQL_DROPOUT):
        super().__init__()
        self.action_embeds = nn.Parameter(action_embeds, requires_grad=True)
        input_dim = state_dim + action_embeds.shape[1]
        self.f1   = nn.Linear(input_dim, hidden_dim)
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.f2   = nn.Linear(hidden_dim, hidden_dim)
        self.ln2  = nn.LayerNorm(hidden_dim)
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
    State value network  V(s) → scalar.

    Estimates the expected future return from state `s`, used as a Bellman
    baseline during IQL training.

    Parameters
    ----------
    state_dim : Dimensionality of the state vector.
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
