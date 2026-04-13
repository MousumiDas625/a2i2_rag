#!/usr/bin/env python3
"""
I03_train_iql.py — Train the IQL Policy Selector (Dueling Embed Mode)
======================================================================

Architecture:
    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
    where A(s,a) = project(encode(s)) · trainable_embed(a)

Key training features:
    - EMA target V-network for stable Bellman targets
    - Class-balanced sampling (oversamples minority policies)
    - Gradient clipping
    - Gaussian state augmentation
    - Cosine LR schedule with AdamW

INPUTS:  data/iql/iql_dataset.jsonl, data/iql/label_map.json,
         data/indexes/policies/operator_prototypes.npy
OUTPUTS: data/iql/selector/iql_model.pt, data/iql/selector/value_model.pt,
         data/iql/selector/norm_stats.npz,
         iql/plots/training_curves.png, iql/plots/per_policy_qvalues.png
"""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    IQL_DATASET_FILE, LABEL_MAP_FILE, PROTOTYPES_FILE,
    SELECTOR_DIR, IQL_PLOTS_DIR, DEVICE,
    IQL_EPOCHS, IQL_BATCH_SIZE, IQL_LR_Q, IQL_LR_V,
    IQL_WEIGHT_DECAY, IQL_VAL_SPLIT, IQL_GAMMA, IQL_LAMBDA_V,
    IQL_EARLY_STOP_PATIENCE,
    IQL_MARGIN, IQL_MARGIN_WEIGHT, IQL_NOISE_STD,
    IQL_TARGET_TAU, IQL_GRAD_CLIP, IQL_ORTHO_WEIGHT,
)
from iql.networks import QNetworkEmbed, ValueNetwork

MODEL_Q_OUT = SELECTOR_DIR / "iql_model.pt"
MODEL_V_OUT = SELECTOR_DIR / "value_model.pt"
NORM_OUT    = SELECTOR_DIR / "norm_stats.npz"
PLOT_OUT    = IQL_PLOTS_DIR / "training_curves.png"

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading dataset from {IQL_DATASET_FILE}")
states_list, actions_list, rewards_list, dids = [], [], [], []
with open(IQL_DATASET_FILE) as f:
    for line in f:
        rec = json.loads(line)
        states_list.append(rec["state_vec"])
        actions_list.append(rec["action_id"])
        rewards_list.append(rec["reward"])
        dids.append(rec.get("dialogue_id"))

states  = np.array(states_list, dtype=np.float32)
actions = np.array(actions_list, dtype=np.int64)
rewards = np.array(rewards_list, dtype=np.float32)

# Next-state vectors (shift by 1 within same dialogue)
next_states = np.roll(states, shift=-1, axis=0)
next_states[-1] = states[-1]
for i in range(len(states) - 1):
    if dids[i] != dids[i + 1]:
        next_states[i] = states[i]

# Normalise states and save stats for inference
mean = states.mean(axis=0)
std  = states.std(axis=0) + 1e-6
states      = (states - mean) / std
next_states = (next_states - mean) / std

np.savez(NORM_OUT, mean=mean, std=std)
print(f"[INFO] Saved normalization stats → {NORM_OUT}")

label_map   = json.load(open(LABEL_MAP_FILE))
num_actions = len(label_map)
state_dim   = states.shape[1]

# Print class distribution
act_counts = Counter(actions)
inv_map = {v: k for k, v in label_map.items()}
print(f"[INFO] {len(states)} samples | state_dim={state_dim} | "
      f"num_actions={num_actions} | device={DEVICE}")
print("[INFO] Action distribution:")
for aid in sorted(act_counts):
    print(f"  {inv_map[aid]:12s}: {act_counts[aid]:4d} ({100*act_counts[aid]/len(states):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Action embeddings (frozen prototypes)
# ─────────────────────────────────────────────────────────────────────────────
if not PROTOTYPES_FILE.exists():
    raise FileNotFoundError(f"Missing {PROTOTYPES_FILE}. Run I02 first.")

action_embeds = torch.tensor(np.load(PROTOTYPES_FILE), dtype=torch.float32,
                             device=DEVICE)
print(f"[INFO] Loaded {num_actions} frozen operator embeddings "
      f"(dim={action_embeds.shape[1]})")

# ─────────────────────────────────────────────────────────────────────────────
# Build models + EMA target network
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Building Dueling-Embed Q-network + ValueNetwork + Target V …")
qnet  = QNetworkEmbed(state_dim, action_embeds).to(DEVICE)
vnet  = ValueNetwork(state_dim).to(DEVICE)

# Target V-network: slow-moving copy used for stable Bellman targets
vnet_target = copy.deepcopy(vnet)
for p in vnet_target.parameters():
    p.requires_grad_(False)

opt_q = optim.AdamW(qnet.parameters(), lr=IQL_LR_Q, weight_decay=IQL_WEIGHT_DECAY)
opt_v = optim.AdamW(vnet.parameters(), lr=IQL_LR_V, weight_decay=IQL_WEIGHT_DECAY)
sched_q = optim.lr_scheduler.CosineAnnealingLR(opt_q, T_max=IQL_EPOCHS, eta_min=1e-5)
sched_v = optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=IQL_EPOCHS, eta_min=1e-5)
mse   = nn.MSELoss()


import torch.nn.functional as F


def orthogonality_loss(embeds: torch.Tensor) -> torch.Tensor:
    """Penalise cosine similarity between embedding pairs to keep them diverse."""
    norm = F.normalize(embeds, dim=1)
    gram = norm @ norm.T
    eye  = torch.eye(len(norm), device=norm.device)
    return ((gram - eye) ** 2).mean()


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft-update target parameters: θ_t ← τ·θ_s + (1−τ)·θ_t"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def balanced_batch_iter(X, X_next, a, r, bs):
    """Yield batches with class-balanced sampling (oversamples minority actions)."""
    n = len(X)
    counts = np.bincount(a, minlength=num_actions).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts[a]
    weights /= weights.sum()

    idx = np.random.choice(n, size=n, replace=True, p=weights)
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
    states, next_states, actions, rewards,
    test_size=IQL_VAL_SPLIT, random_state=42,
)

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
metrics: dict = {k: [] for k in [
    "epoch", "train_LQ", "train_LV", "val_LQ", "val_LV",
    "meanQ", "stdQ", "val_total", "per_policy_Q", "spread",
]}
best_val_q = float("inf")
patience = IQL_EARLY_STOP_PATIENCE

for epoch in range(1, IQL_EPOCHS + 1):
    qnet.train(); vnet.train()
    tot_LQ, tot_LV, tot_Lm, tot_Lo, cnt = 0.0, 0.0, 0.0, 0.0, 0
    q_means, q_stds = [], []

    for xb, xnb, ab, rb in balanced_batch_iter(X_tr, Xn_tr, a_tr, r_tr, IQL_BATCH_SIZE):
        if IQL_NOISE_STD > 0:
            xb  = xb  + IQL_NOISE_STD * torch.randn_like(xb)
            xnb = xnb + IQL_NOISE_STD * torch.randn_like(xnb)

        q_all    = qnet(xb)
        q_sel    = q_all.gather(1, ab.unsqueeze(1)).squeeze(1)

        # Use TARGET V-network for stable Bellman targets
        with torch.no_grad():
            v_next_target = vnet_target(xnb)
        v_vals   = vnet(xb)
        target_q = rb + IQL_GAMMA * v_next_target

        L_Q = mse(q_sel, target_q)
        L_V = mse(v_vals, q_sel.detach())

        q_spread = q_all.max(dim=1).values - q_all.min(dim=1).values
        L_margin = IQL_MARGIN_WEIGHT * torch.clamp(
            IQL_MARGIN - q_spread, min=0.0
        ).mean()

        L_ortho = IQL_ORTHO_WEIGHT * orthogonality_loss(qnet.action_embeds)

        loss = L_Q + IQL_LAMBDA_V * L_V + L_margin + L_ortho

        opt_q.zero_grad(); opt_v.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(qnet.parameters(), IQL_GRAD_CLIP)
        nn.utils.clip_grad_norm_(vnet.parameters(), IQL_GRAD_CLIP)

        opt_q.step(); opt_v.step()

        # EMA update of target V-network
        ema_update(vnet_target, vnet, IQL_TARGET_TAU)

        with torch.no_grad():
            q_means.append(q_all.mean().item())
            q_stds.append(q_all.std().item())

        tot_LQ += L_Q.item() * len(xb)
        tot_LV += L_V.item() * len(xb)
        tot_Lm += L_margin.item() * len(xb)
        tot_Lo += L_ortho.item() * len(xb)
        cnt    += len(xb)

    train_LQ = tot_LQ / cnt
    train_LV = tot_LV / cnt
    train_Lm = tot_Lm / cnt
    train_Lo = tot_Lo / cnt

    # ── Validation ────────────────────────────────────────────────────────────
    qnet.eval(); vnet.eval()
    with torch.no_grad():
        Xv   = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        Xnv  = torch.tensor(Xn_val, dtype=torch.float32, device=DEVICE)
        av   = torch.tensor(a_val, dtype=torch.long, device=DEVICE)
        rv   = torch.tensor(r_val, dtype=torch.float32, device=DEVICE)

        q_all_v = qnet(Xv)
        q_sel_v = q_all_v.gather(1, av.unsqueeze(1)).squeeze(1)
        vv      = vnet(Xv)
        vv_nxt  = vnet_target(Xnv)
        LQ_val  = mse(q_sel_v, rv + IQL_GAMMA * vv_nxt).item()
        LV_val  = mse(vv, q_sel_v).item()
        val_loss = LQ_val + IQL_LAMBDA_V * LV_val

        policy_means = [q_all_v[:, a].mean().item() for a in range(num_actions)]
        metrics["per_policy_Q"].append(policy_means)

    sched_q.step(); sched_v.step()

    spread = max(policy_means) - min(policy_means)
    metrics["epoch"].append(epoch)
    metrics["train_LQ"].append(train_LQ)
    metrics["train_LV"].append(train_LV)
    metrics["val_LQ"].append(LQ_val)
    metrics["val_LV"].append(LV_val)
    metrics["meanQ"].append(np.mean(q_means))
    metrics["stdQ"].append(np.mean(q_stds))
    metrics["val_total"].append(val_loss)
    metrics["spread"].append(spread)

    print(
        f"Epoch {epoch:03d} | Train_Q {train_LQ:.5f} | Train_V {train_LV:.5f} | "
        f"L_margin {train_Lm:.5f} | L_ortho {train_Lo:.5f} | "
        f"Val_Q {LQ_val:.5f} | Val_V {LV_val:.5f} | "
        f"meanQ {np.mean(q_means):.3f} | stdQ {np.mean(q_stds):.3f} | "
        f"spread {spread:.3f}"
    )

    if LQ_val < best_val_q - 1e-6:
        best_val_q = LQ_val
        patience = IQL_EARLY_STOP_PATIENCE
        torch.save(qnet.state_dict(), MODEL_Q_OUT)
        torch.save(vnet.state_dict(), MODEL_V_OUT)
        print(f"  [BEST] epoch {epoch:03d} | val_Q {LQ_val:.6f} | val_total {val_loss:.6f}")
    else:
        patience -= 1
        if patience == 0:
            print(f"[EARLY STOP] No improvement in val_Q for {IQL_EARLY_STOP_PATIENCE} epochs.")
            break

# ─────────────────────────────────────────────────────────────────────────────
# Debug final Q-values
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DEBUG] Average Q-values per policy (full dataset):")
states_tensor = torch.tensor(states, dtype=torch.float32, device=DEVICE)
qnet.eval()
with torch.no_grad():
    q_full = qnet(states_tensor)
    for i, name in enumerate(label_map.keys()):
        col = q_full[:, i].cpu().numpy()
        print(f"  {name:10s}: mean={col.mean():.4f}  std={col.std():.4f}")
    spread_final = (q_full.max(dim=1).values - q_full.min(dim=1).values).mean().item()
    print(f"\n  Avg spread (max−min per sample): {spread_final:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
epochs = metrics["epoch"]
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

axs[0, 0].plot(epochs, metrics["train_LQ"], label="Train Q-loss")
axs[0, 0].plot(epochs, metrics["val_LQ"], label="Val Q-loss", ls="--")
axs[0, 0].set_title("Q Loss (Bellman)"); axs[0, 0].legend()

axs[0, 1].plot(epochs, metrics["train_LV"], label="Train V-loss")
axs[0, 1].plot(epochs, metrics["val_LV"], label="Val V-loss", ls="--")
axs[0, 1].set_title("V Loss"); axs[0, 1].legend()

axs[0, 2].plot(epochs, metrics["spread"], color="tab:green")
axs[0, 2].axhline(IQL_MARGIN, color="gray", ls=":", label=f"margin={IQL_MARGIN}")
axs[0, 2].set_title("Per-Policy Q Spread (max−min)"); axs[0, 2].legend()

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

policy_arr   = np.array(metrics["per_policy_Q"])
policy_names = list(label_map.keys())
for i, name in enumerate(policy_names):
    axs[1, 2].plot(epochs, policy_arr[:, i], label=name)
axs[1, 2].set_title("Per-Policy Avg Q-value")
axs[1, 2].legend(); axs[1, 2].grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300); plt.close()

plt.figure(figsize=(10, 6))
for i, name in enumerate(policy_names):
    plt.plot(epochs, policy_arr[:, i], label=name)
plt.title("Per-Policy Avg Q-value During Training")
plt.xlabel("Epoch"); plt.ylabel("Avg Q(s,a)")
plt.legend(); plt.grid(True, ls="--", alpha=0.5); plt.tight_layout()
plt.savefig(IQL_PLOTS_DIR / "per_policy_qvalues.png", dpi=300)
plt.close()

print(f"\n[OK] Q-network  → {MODEL_Q_OUT}")
print(f"[OK] V-network  → {MODEL_V_OUT}")
print(f"[OK] Norm stats → {NORM_OUT}")
print(f"[OK] Plots      → {IQL_PLOTS_DIR}")
