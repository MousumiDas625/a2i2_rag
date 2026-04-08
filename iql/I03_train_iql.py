#!/usr/bin/env python3
"""
I03_train_iql.py — Train the IQL Policy Selector
==================================================

PURPOSE:
    Trains an Implicit Q-Learning (IQL) model that learns to select the
    best operator policy given the current conversation state.

HOW IT WORKS:
    Two modes (selected via --mode):
        state  → Q(s) = [Q(s,a1), …, Q(s,aN)]      (state-only head)
        embed  → Q(s,a) = f([state ‖ emb(a)]) → R   (embedding-based)

    Training:
        • Loads iql_dataset.jsonl (state_vec, action_id, reward).
        • Constructs next_state by shifting within the same dialogue.
        • Bellman target: r + γ·V(s').
        • Joint loss: L_Q (MSE of Q vs target) + λ·L_V (MSE of V vs Q).
        • Optional orthogonality regularization on operator embeddings.
        • Early stopping on validation loss.

    All tensors live on GPU (cuda/mps) throughout training.

INPUTS:  data/iql/iql_dataset.jsonl
         data/iql/label_map.json
         data/indexes/policies/operator_prototypes.npy
OUTPUTS: data/iql/selector/iql_model_{mode}.pt
         data/iql/selector/value_model_{mode}.pt
         data/iql/selector/training_curves_{mode}.png
         data/iql/selector/per_policy_qvalues_{mode}.png

USAGE:
    python iql/I03_train_iql.py --mode state
    python iql/I03_train_iql.py --mode embed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    IQL_DATASET_FILE, LABEL_MAP_FILE, PROTOTYPES_FILE,
    SELECTOR_DIR, DEVICE,
    IQL_EPOCHS, IQL_BATCH_SIZE, IQL_LR_Q, IQL_LR_V,
    IQL_VAL_SPLIT, IQL_GAMMA, IQL_LAMBDA_V,
    IQL_EARLY_STOP_PATIENCE,
)
from iql.networks import QNetworkState, QNetworkEmbed, ValueNetwork

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train IQL policy selector")
parser.add_argument("--mode", choices=["embed", "state"], default="state")
args = parser.parse_args()

MODEL_Q_OUT = SELECTOR_DIR / f"iql_model_{args.mode}.pt"
MODEL_V_OUT = SELECTOR_DIR / f"value_model_{args.mode}.pt"
PLOT_OUT    = SELECTOR_DIR / f"training_curves_{args.mode}.png"

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading dataset from {IQL_DATASET_FILE}")
states_list, actions_list, rewards_list, dids = [], [], [], []
with open(IQL_DATASET_FILE, "r") as f:
    for line in f:
        rec = json.loads(line)
        states_list.append(rec["state_vec"])
        actions_list.append(rec["action_id"])
        rewards_list.append(rec["reward"])
        dids.append(rec.get("dialogue_id"))

states  = np.array(states_list, dtype=np.float32)
actions = np.array(actions_list, dtype=np.int64)
rewards = np.array(rewards_list, dtype=np.float32)

# Next-state vectors (shifted by one within the same dialogue)
next_states = np.roll(states, shift=-1, axis=0)
next_states[-1] = states[-1]
for i in range(len(states) - 1):
    if dids[i] != dids[i + 1]:
        next_states[i] = states[i]

# Normalize
mean = states.mean(axis=0)
std  = states.std(axis=0) + 1e-6
states      = (states - mean) / std
next_states = (next_states - mean) / std

label_map   = json.load(open(LABEL_MAP_FILE))
num_actions = len(label_map)
state_dim   = states.shape[1]
print(f"[INFO] {len(states)} samples | state_dim={state_dim}, num_actions={num_actions} | device={DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Operator embeddings
# ─────────────────────────────────────────────────────────────────────────────
if not PROTOTYPES_FILE.exists():
    raise FileNotFoundError(f"Missing {PROTOTYPES_FILE}. Run I02 first.")

action_embeds = torch.tensor(np.load(PROTOTYPES_FILE), dtype=torch.float32, device=DEVICE)
action_dim = action_embeds.shape[1]

# ─────────────────────────────────────────────────────────────────────────────
# Build models
# ─────────────────────────────────────────────────────────────────────────────
if args.mode == "embed":
    print("[MODE] Embedding-based Q-network")
    qnet = QNetworkEmbed(state_dim, action_embeds).to(DEVICE)
    with torch.no_grad():
        qnet.action_embeds[:] *= 2.0
else:
    print("[MODE] State-only Q-network")
    qnet = QNetworkState(state_dim, num_actions).to(DEVICE)

vnet  = ValueNetwork(state_dim).to(DEVICE)
opt_q = optim.Adam(qnet.parameters(), lr=IQL_LR_Q)
opt_v = optim.Adam(vnet.parameters(), lr=IQL_LR_V)
mse   = nn.MSELoss()


def orthogonality_loss(embeds: torch.Tensor) -> torch.Tensor:
    norm = F.normalize(embeds, dim=1)
    gram = norm @ norm.T
    I = torch.eye(len(norm), device=norm.device)
    return ((gram - I) ** 2).mean()


def batch_iter(X, X_next, a, r, bs):
    n = len(X)
    idx = np.random.permutation(n)
    for i in range(0, n, bs):
        j = idx[i : i + bs]
        yield (
            torch.tensor(X[j], dtype=torch.float32, device=DEVICE),
            torch.tensor(X_next[j], dtype=torch.float32, device=DEVICE),
            torch.tensor(a[j], dtype=torch.long, device=DEVICE),
            torch.tensor(r[j], dtype=torch.float32, device=DEVICE),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split
# ─────────────────────────────────────────────────────────────────────────────
X_tr, X_val, Xn_tr, Xn_val, a_tr, a_val, r_tr, r_val = train_test_split(
    states, next_states, actions, rewards, test_size=IQL_VAL_SPLIT, random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
metrics: dict = {k: [] for k in [
    "epoch", "train_LQ", "train_LV", "val_LQ", "val_LV",
    "meanQ", "stdQ", "val_total", "per_policy_Q",
]}
best_val_loss = float("inf")
patience = IQL_EARLY_STOP_PATIENCE

for epoch in range(1, IQL_EPOCHS + 1):
    qnet.train(); vnet.train()
    tot_LQ, tot_LV, cnt = 0.0, 0.0, 0
    q_means, q_stds = [], []

    for xb, xnb, ab, rb in batch_iter(X_tr, Xn_tr, a_tr, r_tr, IQL_BATCH_SIZE):
        if args.mode == "embed":
            q_sel = qnet(xb, ab)
        else:
            q_sel = qnet(xb).gather(1, ab.unsqueeze(1)).squeeze(1)

        v_next   = vnet(xnb)
        v_vals   = vnet(xb)
        target_q = rb + IQL_GAMMA * v_next.detach()

        L_Q = mse(q_sel, target_q)
        L_V = mse(v_vals, q_sel.detach())
        L_ortho = 1e-3 * orthogonality_loss(qnet.action_embeds) if args.mode == "embed" else 0.0
        loss = L_Q + IQL_LAMBDA_V * L_V + L_ortho

        opt_q.zero_grad(); opt_v.zero_grad()
        loss.backward()
        opt_q.step(); opt_v.step()

        with torch.no_grad():
            if args.mode == "embed":
                q_batch = torch.cat(
                    [qnet(xb, torch.full_like(ab, aid))[:, None] for aid in range(num_actions)], dim=1
                )
            else:
                q_batch = qnet(xb)
            q_means.append(q_batch.mean().item())
            q_stds.append(q_batch.std().item())

        tot_LQ += L_Q.item() * len(xb)
        tot_LV += L_V.item() * len(xb)
        cnt += len(xb)

    train_LQ = tot_LQ / cnt
    train_LV = tot_LV / cnt

    # ── Validation ────────────────────────────────────────────────────────────
    qnet.eval(); vnet.eval()
    with torch.no_grad():
        Xv   = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        Xnv  = torch.tensor(Xn_val, dtype=torch.float32, device=DEVICE)
        av   = torch.tensor(a_val, dtype=torch.long, device=DEVICE)
        rv   = torch.tensor(r_val, dtype=torch.float32, device=DEVICE)

        if args.mode == "embed":
            q_sel_v = qnet(Xv, av)
        else:
            q_sel_v = qnet(Xv).gather(1, av.unsqueeze(1)).squeeze(1)

        vv     = vnet(Xv)
        vv_nxt = vnet(Xnv)
        LQ_val = mse(q_sel_v, rv + IQL_GAMMA * vv_nxt).item()
        LV_val = mse(vv, q_sel_v).item()
        val_loss = LQ_val + IQL_LAMBDA_V * LV_val

        policy_means = []
        for aid in range(num_actions):
            at = torch.full((len(Xv),), aid, dtype=torch.long, device=DEVICE)
            if args.mode == "embed":
                q_vals = qnet(Xv, at)
            else:
                q_vals = qnet(Xv)[:, aid]
            policy_means.append(q_vals.mean().item())
        metrics["per_policy_Q"].append(policy_means)

    metrics["epoch"].append(epoch)
    metrics["train_LQ"].append(train_LQ)
    metrics["train_LV"].append(train_LV)
    metrics["val_LQ"].append(LQ_val)
    metrics["val_LV"].append(LV_val)
    metrics["meanQ"].append(np.mean(q_means))
    metrics["stdQ"].append(np.mean(q_stds))
    metrics["val_total"].append(val_loss)

    print(
        f"Epoch {epoch:03d} | Train_Q {train_LQ:.5f} | Train_V {train_LV:.5f} | "
        f"Val_Q {LQ_val:.5f} | Val_V {LV_val:.5f} | "
        f"meanQ {np.mean(q_means):.3f} | stdQ {np.mean(q_stds):.3f}"
    )

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        patience = IQL_EARLY_STOP_PATIENCE
        torch.save(qnet.state_dict(), MODEL_Q_OUT)
        torch.save(vnet.state_dict(), MODEL_V_OUT)
        print(f"  [BEST] epoch {epoch:03d} | val_total {val_loss:.6f}")
    else:
        patience -= 1
        if patience == 0:
            print(f"[EARLY STOP] No improvement for {IQL_EARLY_STOP_PATIENCE} epochs.")
            break

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
epochs = metrics["epoch"]
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(epochs, metrics["train_LQ"], label="Train Q-loss")
axs[0, 0].plot(epochs, metrics["val_LQ"], label="Val Q-loss", ls="--")
axs[0, 0].set_title("Q Loss (Bellman)"); axs[0, 0].legend()

axs[0, 1].plot(epochs, metrics["train_LV"], label="Train V-loss")
axs[0, 1].plot(epochs, metrics["val_LV"], label="Val V-loss", ls="--")
axs[0, 1].set_title("V Loss"); axs[0, 1].legend()

axs[1, 0].plot(epochs, metrics["meanQ"], color="tab:blue")
axs[1, 0].fill_between(
    epochs,
    np.array(metrics["meanQ"]) - np.array(metrics["stdQ"]),
    np.array(metrics["meanQ"]) + np.array(metrics["stdQ"]),
    alpha=0.2, color="tab:blue",
)
axs[1, 0].set_title("Avg Q-value (±std)")

axs[1, 1].plot(epochs, metrics["val_total"], color="tab:red")
axs[1, 1].set_title("Validation Total Loss")

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300); plt.close()

# Per-policy plot
policy_arr = np.array(metrics["per_policy_Q"])
policy_names = list(label_map.keys())
plt.figure(figsize=(10, 6))
for i, name in enumerate(policy_names):
    plt.plot(epochs, policy_arr[:, i], label=name)
plt.title(f"Per-Policy Avg Q ({args.mode} mode)")
plt.xlabel("Epoch"); plt.ylabel("Avg Q(s,a)")
plt.legend(); plt.grid(True, ls="--", alpha=0.5); plt.tight_layout()
plt.savefig(SELECTOR_DIR / f"per_policy_qvalues_{args.mode}.png", dpi=300)
plt.close()

print(f"\n[OK] Q-network  → {MODEL_Q_OUT}")
print(f"[OK] V-network  → {MODEL_V_OUT}")
print(f"[OK] Plots      → {PLOT_OUT}")


if __name__ == "__main__":
    pass
