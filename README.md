# A2I2 Final — IQL-based Operator Policy Selection for Wildfire Evacuation Dialogues

This project trains an **Implicit Q-Learning (IQL)** policy selector that learns to pick the best persuasion strategy for an emergency operator trying to convince different residents to evacuate during a wildfire. It includes three experiment baselines, a conversation simulation engine, and a FastAPI server.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup (uv)](#environment-setup-uv)
3. [OpenAI API Key](#openai-api-key)
4. [Project Structure](#project-structure)
5. [Pipeline Overview](#pipeline-overview)
6. [How to Run](#how-to-run)
7. [Running on Endeavour (SLURM)](#running-on-endeavour-slurm)
8. [Success Rate Matrices](#success-rate-matrices)
9. [Where to Find Outputs](#where-to-find-outputs)
10. [IQL Dueling-Embed Architecture](#iql-dueling-embed-architecture)
11. [Switching LLM Providers](#switching-llm-providers)
12. [Interactive Mode](#interactive-mode)
13. [API Server](#api-server)

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10+ |
| **uv** | Fast Python package manager — installed by `slurm/setup_env.sh` |
| **OpenAI API key** | For LLM calls during experiments (see below) |
| **GPU** | CUDA (cluster) or Apple MPS (local) — auto-detected; falls back to CPU |
| **curl** | For installing uv (pre-installed on all Linux systems) |

> **Ollama alternative**: If you prefer running a local LLM instead of OpenAI, install Ollama, pull `llama3`, run `ollama serve`, and set `LLM_PROVIDER=ollama` (see [Switching LLM Providers](#switching-llm-providers)).

---

## Environment Setup (uv)

[uv](https://github.com/astral-sh/uv) is a fast, self-contained Python package manager — no conda or system Python required.

### Local machine (macOS / Linux)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"     # add to ~/.zshrc or ~/.bashrc permanently

# 2. Create the virtual environment
cd /path/to/a2i2_final
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. Install dependencies
uv pip install torch                     # CPU torch is fine for local dev
uv pip install -r requirements.txt

# 4. Activate whenever you work on the project
source .venv/bin/activate
```

### Endeavour HPC cluster

Run the one-time setup script **on a login node** before submitting any SLURM jobs:

```bash
cd /project2/biyik_1165/mousumid/a2i2_new/A2I2_Chatbot/a2i2_final
bash slurm/setup_env.sh
```

This script automatically:
- Downloads and installs `uv` to `~/.local/bin/`
- Creates `.venv/` with Python 3.10
- Installs PyTorch with CUDA 12.1 support
- Installs all dependencies and replaces `faiss-cpu` with `faiss-gpu`
- Verifies CUDA is visible to PyTorch

You only need to run this **once**. All SLURM job scripts then activate `.venv` automatically.

---

## OpenAI API Key (optional — only if not using Ollama)

If you prefer OpenAI over Ollama, follow these steps:

1. Go to [platform.openai.com](https://platform.openai.com) → sign in → **API keys** → **Create new secret key**. Copy the key (`sk-proj-...`).
2. Add billing at **Billing** → add a payment method. Estimated cost for all experiments: **$5–10** using `gpt-4o-mini`.
3. Set it on the cluster:

```bash
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.bashrc
source ~/.bashrc
```

Then in each `slurm/run_exp*.sh`, comment out the Ollama block and uncomment the OpenAI block (see [Running on Endeavour](#running-on-endeavour-slurm)).

---

## Project Structure

```
a2i2_final/
├── config/
│   ├── settings.py          # All paths, hyperparameters, device, LLM config
│   └── personas.py          # 10 resident persona definitions
│
├── preprocessing/            # STAGE 1: Data preparation
│   ├── P01_xlsx_to_jsonl.py
│   ├── P02_clean_and_merge.py
│   ├── P03_extract_residents.py
│   └── P04_add_rewards.py
│
├── iql/                      # STAGE 2: IQL training pipeline
│   ├── networks.py
│   ├── I01_build_iql_dataset.py
│   ├── I02_build_operator_policies.py
│   ├── I03_train_iql.py
│   ├── I04_build_rag_indexes.py
│   └── I05_extract_successful_utterances.py
│
├── retrieval/
│   ├── policy_selector.py    # Load trained IQL model, select best policy
│   └── rag_retrieval.py      # Retrieve few-shot examples from FAISS
│
├── simulation/               # Conversation engine
│   ├── llm_client.py         # Unified LLM client (Ollama / OpenAI)
│   ├── operator_generator.py # Generate operator utterances (3 strategies)
│   ├── resident_simulator.py # Generate resident utterances via persona LLM
│   ├── decision.py           # Judge conversation success/failure
│   ├── conversation_loop.py  # Core turn-by-turn simulation loop
│   └── interactive.py        # Interactive console mode
│
├── experiments/              # STAGE 3 & 4: Run and evaluate
│   ├── exp1_zero_shot.py
│   ├── exp2_rag_successful.py
│   ├── exp3_iql_policy.py
│   ├── batch_runner.py
│   ├── evaluate.py           # Side-by-side comparison table + CSV
│   └── make_success_matrices.py  # Per-experiment success matrices + heatmaps
│
├── slurm/                    # Endeavour HPC job scripts
│   ├── setup_env.sh          # One-time environment setup (run on login node)
│   ├── submit_all.sh         # Submit all 3 jobs + matrix job in one command
│   ├── run_exp1_zero_shot.sh
│   ├── run_exp2_rag.sh
│   └── run_exp3_iql.sh
│
├── api/
│   └── server.py
│
├── data/                     # All generated data (created by the pipeline)
│   ├── raw_xlsx/
│   ├── jsonl/
│   ├── cleaned/
│   ├── meta/
│   ├── reports/
│   ├── iql/
│   │   └── selector/         # Trained model weights + plots
│   ├── indexes/
│   │   ├── policies/
│   │   └── faiss/
│   ├── successful_ops/
│   └── runs/                 # Experiment results + success matrices
│
├── .venv/                    # Virtual environment (created by setup_env.sh)
├── run_pipeline.sh
└── requirements.txt
```

### Resident Personas

| Persona | Has Training Data | Description |
|---|---|---|
| bob | **Yes** | Stubborn, prioritizes work over safety |
| niki | **Yes** | Cooperative, willing to follow instructions |
| lindsay | **Yes** | Caregiver responsible for children |
| michelle | **Yes** | Stubborn, determined to protect property |
| ross | **Yes** | Van driver helping evacuate elderly people |
| mary | No | Elderly person living alone with a pet |
| ben | No | Young professional working from home |
| ana | No | Caregiver at a senior center |
| tom | No | Community-connected, wants to help others first |
| mia | No | Teen student absorbed in a school robotics project |

The 5 "trained" personas have human-collected dialogues used for IQL training. The 5 "extended" personas are LLM-simulated and test generalization.

---

## Pipeline Overview

### Stage 1: Preprocessing (P01–P04)

| Script | Output |
|---|---|
| `P01_xlsx_to_jsonl.py` | `data/jsonl/` |
| `P02_clean_and_merge.py` | `data/cleaned/` |
| `P03_extract_residents.py` | `data/meta/residents.json` |
| `P04_add_rewards.py` | Updates `data/cleaned/` in-place |

### Stage 2: IQL Pipeline (I01–I05)

| Script | What it does | Output |
|---|---|---|
| `I01_build_iql_dataset.py` | Encodes states via sentence embeddings, builds (state, action, reward) tuples | `data/iql/iql_dataset.jsonl` |
| `I02_build_operator_policies.py` | Computes per-policy prototype embeddings (centroid of each operator's utterances) | `data/indexes/policies/` |
| `I03_train_iql.py` | Trains Dueling-Embed Q-network and V-network (see [IQL Dueling-Embed Architecture](#iql-dueling-embed-architecture)) | `data/iql/selector/*.pt` |
| `I04_build_rag_indexes.py` | Builds per-policy FAISS indexes for few-shot RAG retrieval | `data/indexes/faiss/` |
| `I05_extract_successful_utterances.py` | Builds a global corpus of operator utterances from successful dialogues | `data/successful_ops/` |

### Stage 3: Experiments

| Experiment | Strategy | Description |
|---|---|---|
| **Exp 1** | `zero_shot` | No training data — pure instruction following |
| **Exp 2** | `rag_successful` | Retrieves from global successful-operator corpus |
| **Exp 3** | `iql_rag` | IQL selects best policy → RAG from that policy's index |

### Stage 4: Evaluation

`evaluate.py` prints a comparison table and writes `data/runs/comparison.csv`.
`make_success_matrices.py` generates per-experiment run-level matrices (see below).

---

## How to Run

### Option A: Full pipeline (one command)

```bash
source .venv/bin/activate
cd /path/to/a2i2_final

export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

bash run_pipeline.sh
```

### Option B: Step by step

```bash
source .venv/bin/activate
cd /path/to/a2i2_final

export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# Stage 1
python preprocessing/P01_xlsx_to_jsonl.py
python preprocessing/P02_clean_and_merge.py
python preprocessing/P03_extract_residents.py
python preprocessing/P04_add_rewards.py

# Stage 2
python iql/I01_build_iql_dataset.py
python iql/I02_build_operator_policies.py
python iql/I03_train_iql.py
python iql/I04_build_rag_indexes.py
python iql/I05_extract_successful_utterances.py

# Stage 3 (5 runs each)
python experiments/exp1_zero_shot.py --runs 5
python experiments/exp2_rag_successful.py --runs 5
python experiments/exp3_iql_policy.py --runs 5

# Stage 4
python experiments/evaluate.py
python experiments/make_success_matrices.py
```

### Experiment flags

| Flag | Default | Description |
|---|---|---|
| `--residents` | all 10 | Comma-separated names, e.g. `bob,ross` |
| `--runs` | 5 | Conversations per resident |
| `--max-turns` | 16 | Hard turn limit |
| `--seed` | "Hello, this is the fire department…" | Opening operator line |

---

## Running on Endeavour (SLURM)

### Step 1 — One-time setup (login node)

Run this **once** before submitting any jobs. It installs `uv`, creates the venv, installs all Python dependencies, downloads the Ollama binary, and pulls the llama3 model (~4.7 GB) into project space:

```bash
cd /project2/biyik_1165/mousumid/a2i2_new/A2I2_Chatbot/a2i2_final
bash slurm/setup_env.sh
```

What happens under the hood:
- `uv` → `~/.local/bin/uv`
- Python 3.10 venv → `a2i2_final/.venv/`
- Ollama binary → `a2i2_final/.ollama/ollama`
- llama3 model weights → `/project2/biyik_1165/mousumid/a2i2_new/ollama_models/`
  *(project space, not home directory quota)*
- Shared config → `slurm/ollama_env.sh` (read by job scripts automatically)

### Step 2 — Fill in your account in each SLURM script

Open each `slurm/run_exp*.sh` and replace `<YOUR_ACCOUNT>` and `<YOUR_EMAIL>`:

```bash
#SBATCH --account=ttrojan_123       # find yours with: myquota
#SBATCH --mail-user=you@usc.edu
```

### Step 3 — Submit everything

```bash
bash slurm/submit_all.sh
```

This submits all three experiments **in parallel**, then automatically queues `make_success_matrices.py` with `--dependency=afterok` so it only runs once all three finish.

### How Ollama runs inside each job

Each SLURM script sources `slurm/lib_ollama.sh` and calls `ollama_start`, which:

1. Picks a free port (avoids collisions if two jobs land on the same node)
2. Starts `ollama serve` as a background process — **uses the compute node's GPU** automatically via `CUDA_VISIBLE_DEVICES`
3. Waits up to 90 seconds for the server to be ready
4. Exports `OLLAMA_URL` so `llm_client.py` finds it
5. Registers a `trap` so `ollama stop` is called on job exit (clean or error)

Ollama server logs are written to `slurm/logs/ollama_<jobid>.log`.

### Monitor

```bash
squeue -u $USER                              # see running/queued jobs
tail -f slurm/logs/exp1_<jobid>.out         # stream experiment output
tail -f slurm/logs/ollama_<jobid>.log       # stream Ollama server log
```

### GPU usage

Each job requests `--gres=gpu:1`. The GPU is used by:

| Component | Usage |
|---|---|
| **Ollama / llama3** | LLM inference for operator + resident utterances (primary GPU load) |
| **Sentence-transformer** | Embedding resident utterances for RAG + IQL state encoding |
| **IQL Q-network** (Exp 3) | Policy selection inference |
| **FAISS-GPU** | Nearest-neighbour retrieval from policy indexes |

### Switching to OpenAI instead of Ollama

If you prefer the OpenAI API (no GPU needed for LLM, lower latency), comment out the Ollama block in each job script and uncomment the OpenAI block:

```bash
# In slurm/run_exp1_zero_shot.sh (and exp2, exp3):

# Comment out:
# source "$(dirname "${BASH_SOURCE[0]}")/lib_ollama.sh"
# ollama_start

# Uncomment:
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."     # or set in ~/.bashrc
export OPENAI_MODEL="gpt-4o-mini"
```

---

## Success Rate Matrices

After all experiments complete, generate per-experiment matrices:

```bash
python experiments/make_success_matrices.py
```

Output in `data/runs/matrices_<timestamp>/`:

| File | Contents |
|---|---|
| `zero_shot_matrix.csv` | Rows = residents, Columns = Run1…Run5, cells = 0/1 |
| `rag_successful_matrix.csv` | Same format for Exp 2 |
| `iql_policy_matrix.csv` | Same format for Exp 3 |
| `zero_shot_matrix.png` | Colour heatmap (green=success, red=fail) |
| `rag_successful_matrix.png` | Same for Exp 2 |
| `iql_policy_matrix.png` | Same for Exp 3 |

Console output example:

```
══════════════════════════════════════════════════════════════
  SUCCESS MATRIX — ZERO-SHOT
══════════════════════════════════════════════════════════════
  Resident        Run 1   Run 2   Run 3   Run 4   Run 5   Rate
  ──────────────────────────────────────────────────────────
  ana                 ✓       ✗       ✓       ✗       ✓    60%
  bob                 ✗       ✗       ✗       ✓       ✗    20%
  ...
```

You can also pass explicit summary paths:

```bash
python experiments/make_success_matrices.py \
    --exp1 data/runs/exp1_zero_shot_20260410T.../summary.json \
    --exp2 data/runs/exp2_rag_successful_20260410T.../summary.json \
    --exp3 data/runs/exp3_iql_policy_20260410T.../summary.json
```

---

## Where to Find Outputs

### Training artifacts

| Artifact | Path |
|---|---|
| Trained Q-network | `data/iql/selector/iql_model.pt` |
| Trained V-network | `data/iql/selector/value_model.pt` |
| Normalisation stats | `data/iql/selector/norm_stats.npz` |
| Training loss curves | `iql/plots/training_curves.png` |
| Per-policy Q-values | `iql/plots/per_policy_qvalues.png` |
| IQL dataset | `data/iql/iql_dataset.jsonl` |

### Experiment results

```
data/runs/
├── exp1_zero_shot_<timestamp>/summary.json
├── exp2_rag_successful_<timestamp>/summary.json
├── exp3_iql_policy_<timestamp>/summary.json
├── comparison.csv
└── matrices_<timestamp>/
    ├── zero_shot_matrix.csv / .png
    ├── rag_successful_matrix.csv / .png
    └── iql_policy_matrix.csv / .png
```

---

## Switching LLM Providers

### OpenAI (recommended for cluster)

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"    # or gpt-4o
```

### Ollama (local machine)

```bash
ollama pull llama3
ollama serve                          # keep running in a separate terminal

export LLM_PROVIDER=ollama
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3"
```

### Via `config/settings.py`

Edit `LLM_PROVIDER`, `OPENAI_MODEL`, etc. directly in `config/settings.py`.
Environment variables always override the file values.

---

## Interactive Mode

```bash
source .venv/bin/activate
python simulation/interactive.py
```

Starts a console session where you play as the operator and the LLM simulates the resident (or vice versa).

---

## API Server

```bash
source .venv/bin/activate
python api/server.py
# or: uvicorn api.server:app --host 0.0.0.0 --port 8001
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/personas` | GET | List available resident personas |
| `/chat` | POST | Single-turn resident response |
| `/simulate` | POST | Full multi-turn simulation |

---

## IQL Dueling-Embed Architecture

`I03_train_iql.py` trains the Q-network using a **Dueling embedding-based** architecture designed to produce *distinct* Q-values per policy. Here is exactly what happens:

### State representation
The conversation state `s` is the **mean sentence embedding** (via `all-MiniLM-L6-v2`, 384-dim) of the last N resident utterances. This captures what the resident has been saying recently — their resistance level, concerns, and emotional tone — without any manual feature engineering. States are z-normalised using training-set statistics saved to `norm_stats.npz`.

### Policy (action) representations
Each of the 5 operator policies (bob, niki, ross, michelle, lindsay) is represented by a **frozen 384-dim embedding vector** — the centroid of all operator utterances in that policy's corpus (the "prototype"). These embeddings are registered as non-learnable buffers, which **prevents embedding collapse** (a failure mode seen with learnable embeddings where all policies converge to the same vector).

### Why Dueling + Dot-Product (not concatenation)

The original embed architecture concatenated `[state ‖ action_emb]` and passed it through a shared MLP. This consistently produced collapsed Q-values (spread ~0.002) because the network learned to ignore the action embedding portion of the input — the state alone was sufficient to minimise the Bellman loss, so the 384 action-embedding dimensions became dead weight.

The **Dueling architecture** fixes this structurally:

1. **Shared encoder** processes only the state, producing hidden features `h`.
2. **Value head** computes `V(s) = linear(h)` — a scalar baseline.
3. **Advantage head** projects `h` into action-embedding space via a learned linear layer, then computes `A(s,a) = project(h) · embed(a)` via **dot product** with each frozen action embedding.
4. Advantages are centred: `A(s,a) ← A(s,a) − mean_a[A(s,a)]`.
5. Final: `Q(s,a) = V(s) + A(s,a)`.

Because the dot product is computed against geometrically distinct frozen embeddings, different actions are **structurally guaranteed** to produce different advantage values, resulting in well-separated Q-values.

### Q-network architecture

```
State (384-dim)
  └─→ Encoder:
        Linear(384 → 256) + ReLU + LayerNorm + Dropout(0.1)
        Linear(256 → 256) + ReLU + LayerNorm + Dropout(0.1)
        → hidden features h (256-dim)
  └─→ Value head:    Linear(256 → 1)              → V(s)    scalar
  └─→ Advantage head: Linear(256 → 384)            → projection (384-dim)
                       projection · action_embeds.T → A(s,a)  (5 values)
                       A ← A − mean(A)             → centred advantages

  Q(s,a) = V(s) + A(s,a)   →  5 Q-values per state
```

A separate **ValueNetwork** `V(s)` (256 hidden, same depth) is trained in parallel for the Bellman target.

### Training objective

For each `(state s, policy a, reward r, next_state s')` tuple:

```
Bellman target:  y        = r + γ · V(s')
Q-loss:          L_Q      = MSE( Q(s,a),  y )
V-loss:          L_V      = MSE( V(s),    Q(s,a).detach() )
Margin loss:     L_margin = w · mean( max(0, margin − spread) )
                   where spread = max_a Q(s,a) − min_a Q(s,a)
Total loss:      L        = L_Q  +  λ_V · L_V  +  L_margin
```

The **margin loss** penalises the network when the per-sample Q-value spread falls below a threshold (default 0.5), encouraging the model to maintain policy differentiation throughout training.

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `IQL_EPOCHS` | 500 | Max training epochs |
| `IQL_BATCH_SIZE` | 32 | Mini-batch size |
| `IQL_LR_Q` | 1e-3 | Q-network learning rate |
| `IQL_LR_V` | 3e-4 | Value network learning rate |
| `IQL_GAMMA` | 0.50 | Bellman discount factor |
| `IQL_LAMBDA_V` | 0.3 | V-loss weight |
| `IQL_MARGIN` | 0.5 | Minimum desired Q-value spread |
| `IQL_MARGIN_WEIGHT` | 0.1 | Margin loss coefficient |
| `IQL_EARLY_STOP_PATIENCE` | 40 | Epochs without improvement before stopping |
| `IQL_DROPOUT` | 0.1 | Dropout rate |
| `IQL_HIDDEN_DIM_Q` | 256 | Q-network hidden layer width |
| `IQL_HIDDEN_DIM_V` | 256 | Value network hidden layer width |

### Inference (at runtime)
1. Embed the last N resident utterances → `state_vec` (384-dim)
2. Normalise with saved training mean/std
3. Single forward pass: `Q(s) → [Q(s,a1), ..., Q(s,a5)]` — all 5 Q-values at once
4. Return `argmax` policy → operator uses that policy's FAISS index for RAG retrieval

### Q-value separation (verified)

| Metric | Old (concat) | New (Dueling) |
|---|---|---|
| Avg per-sample spread | 0.002 | **1.275** |
| Policies selected per test set | 1 (always same) | **3–5 (context-dependent)** |

---

## Training Summary (Current Run)

- **Dataset**: 104 dialogues, 383 IQL training samples, 5 residents (bob, lindsay, michelle, niki, ross)
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim)
- **Architecture**: Dueling-Embed Q-network with frozen action embeddings + dot-product advantages
- **Training**: Early-stopped at epoch 46 (best val loss: 0.513 at epoch 6), avg Q-spread: 1.275
- **Per-policy Q-values**: bob=1.706, lindsay=1.928, michelle=1.787, niki=1.734, ross=1.664
- **Successful dialogues**: 98 out of 104 (94.2%)
- **Successful operator utterances corpus**: 312 utterances with FAISS index
